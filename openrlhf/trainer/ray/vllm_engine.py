import asyncio
import os
from copy import deepcopy
from typing import Any, List, Optional

import ray
import vllm
from packaging import version
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.inputs import TokensPrompt
from vllm.utils import random_uuid

from openrlhf.utils.rollout import RolloutAndRewardWorker, RolloutWithAgentWorker, RolloutWorker

from .utils import get_bundle_indices, ray_noset_visible_devices


class BaseLLMRayActor:
    """Shared setup for Ray actors backed by vLLM."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=kwargs.pop("num_gpus"),
        )
        self._configure_vllm_env(version, vllm, kwargs.pop("full_determinism", False))

        self.kwargs = kwargs

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        if backend == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif ray_noset_visible_devices():
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

    def _configure_vllm_env(self, version, vllm, full_determinism: bool):
        assert version.parse(vllm.__version__) > version.parse(
            "0.8.5"
        ), "Streaming VLLM version must be greater than 0.8.5"

        if version.parse(vllm.__version__) >= version.parse("0.9.0"):
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

        os.environ["VLLM_USE_V1"] = "1"


@ray.remote
class LLMRayActorAsync(BaseLLMRayActor):
    """Unified async actor that delegates work to provided executors."""

    async def __init__(
        self,
        *args,
        bundle_indices: list = None,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        remote_rm_batch_size: Optional[int] = None,
        max_tasks: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        if agent_func_path:
            self.rollout_worker = RolloutWithAgentWorker(agent_func_path=agent_func_path)
        elif remote_rm_url:
            self.rollout_worker = RolloutAndRewardWorker(remote_rm_url, remote_rm_batch_size)
        else:
            self.rollout_worker = RolloutWorker()

        self.rollout_semaphore = asyncio.Semaphore(max_tasks) if max_tasks else None
        self._pending_rollouts = 0
        self._pending_rollouts_lock = asyncio.Lock()

        engine_args = vllm.AsyncEngineArgs(*args, **self.kwargs)
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        await self.llm.is_sleeping()

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight",
            args=(name, dtype, shape, empty_cache),
        )

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return await self.llm.collective_rpc(
            "update_weight_cuda_ipc",
            args=(name, dtype, shape, ipc_handles, empty_cache),
        )

    async def reset_prefix_cache(self):
        await self.llm.reset_prefix_cache()

    async def sleep(self, level=1):
        await self.llm.sleep(level=level)

    async def wake_up(self):
        await self.llm.wake_up()

    async def generate(self, prompt_token_ids, sampling_params):
        """Token-level generation for rollout executors."""
        generator = self.llm.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            deepcopy(sampling_params),
            request_id=random_uuid(),
        )

        final_output = None
        async for request_output in generator:
            final_output = request_output

        return final_output

    def get_num_unfinished_rollouts(self) -> int:
        # Count pending rollout calls (not raw vLLM request count).
        return int(self._pending_rollouts)

    async def generate_sample(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        num_samples: int = 1,
    ):
        """Rollout locally without a separate worker."""
        async with self._pending_rollouts_lock:
            self._pending_rollouts += num_samples

        async def _run_one(sampling_params):
            return await self.rollout_worker.run(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=self,
            )

        async def _run_with_semaphore(sampling_params):
            if self.rollout_semaphore is None:
                return await _run_one(sampling_params)
            async with self.rollout_semaphore:
                return await _run_one(sampling_params)

        tasks = []
        for _ in range(num_samples):
            tasks.append(_run_with_semaphore(deepcopy(sampling_params)))

        try:
            return await asyncio.gather(*tasks)
        finally:
            async with self._pending_rollouts_lock:
                self._pending_rollouts -= num_samples


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    logprobs_mode=None,
    rollout_tasks_per_gpu: Optional[int] = None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
    remote_rm_batch_size: Optional[int] = None,
):
    """Spin up a set of vLLM Ray actors with consistent placement."""
    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # allow two engines to share one GPU in hybrid mode
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_extension_cls": "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "full_determinism": full_determinism,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": 0.2 if use_hybrid_engine else 1,
            "enable_sleep_mode": vllm_enable_sleep,
        }

        actor_kwargs.update(
            {
                "agent_func_path": agent_func_path,
                "remote_rm_url": remote_rm_url,
                "remote_rm_batch_size": remote_rm_batch_size,
                "max_tasks": (rollout_tasks_per_gpu * tensor_parallel_size if rollout_tasks_per_gpu else None),
            }
        )

        if logprobs_mode:
            actor_kwargs["logprobs_mode"] = logprobs_mode
            actor_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

        vllm_engines.append(
            LLMRayActorAsync.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Call the same method on a list of engines and gather results."""
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
