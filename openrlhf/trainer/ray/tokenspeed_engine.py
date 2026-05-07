"""TokenSpeed rollout engine for OpenRLHF.

Provides the same Ray actor interface as ``vllm_engine.py`` so that
``SamplesGenerator``, ``ActorPPOTrainer``, and the CLI can treat TokenSpeed
and vLLM interchangeably.

TokenSpeed exposes weight-update and memory-management RPCs that map
one-to-one to the vLLM APIs OpenRLHF already depends on:

    vLLM                          TokenSpeed
    ─────────────────────────     ──────────────────────────────────
    init_process_group            init_weights_update_group
    update_weight                 update_weights_from_distributed
    sleep / wake_up               release / resume_memory_occupation
    AsyncLLMEngine.generate       tokenspeed Engine.generate
"""

import asyncio
import os
from copy import deepcopy
from typing import Any, List, Optional

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.agent import AgentExecutorBase, SingleTurnAgentExecutor

from .utils import get_bundle_indices, ray_noset_visible_devices


def _load_agent_executor(agent_func_path: str) -> AgentExecutorBase:
    assert agent_func_path.endswith(".py"), "Agent path must be a Python file"
    import importlib.util

    spec = importlib.util.spec_from_file_location("agent_module", agent_func_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)

    assert hasattr(agent_module, "AgentExecutor"), "Agent module must contain AgentExecutor class"
    agent_executor_cls = agent_module.AgentExecutor
    assert issubclass(agent_executor_cls, AgentExecutorBase), "AgentExecutor must inherit from AgentExecutorBase"
    return agent_executor_cls()


@ray.remote
class TokenSpeedRolloutActor:
    """Async TokenSpeed-backed actor that mirrors ``RolloutRayActor``."""

    async def __init__(
        self,
        *args,
        bundle_indices: list = None,
        agent_func_path: Optional[str] = None,
        remote_rm_url: Optional[str] = None,
        mm_pad_token_ids: Optional[set] = None,
        **kwargs,
    ):
        num_gpus = kwargs.pop("num_gpus")
        self._configure_device_env(
            backend=kwargs.get("distributed_executor_backend"),
            bundle_indices=bundle_indices,
            num_gpus=num_gpus,
        )

        if agent_func_path:
            self.executor = _load_agent_executor(agent_func_path)
        else:
            self.executor = SingleTurnAgentExecutor(remote_rm_url)

        self._mm_pad_token_ids = mm_pad_token_ids

        # Import tokenspeed lazily so it doesn't pollute other Ray actors.
        from tokenspeed.runtime.entrypoints.engine import Engine

        self.engine = Engine(*args, **kwargs)
        self.tokenizer = self.engine.get_tokenizer()

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        if backend == "ray":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif ray_noset_visible_devices():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        if bundle_indices is not None:
            os.environ["TOKENSPEED_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating TokenSpeed engine with bundle_indices={bundle_indices}")

    # ------------------------------------------------------------------
    # Weight synchronisation — maps to TokenSpeed's distributed API
    # ------------------------------------------------------------------

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        """Initialise NCCL process group for DeepSpeed -> TokenSpeed weight sync."""
        success, msg = self.engine.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        if not success:
            raise RuntimeError(f"TokenSpeed init_weights_update_group failed: {msg}")

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        """Receive a single weight tensor via the NCCL group."""
        success, msg = self.engine.update_weights_from_distributed(
            name=name,
            dtype=str(dtype),
            shape=list(shape),
        )
        if not success:
            raise RuntimeError(f"TokenSpeed update_weight failed for {name}: {msg}")

    async def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        """IPC weight path — convert to tensor and call ``update_weights_from_tensor``."""
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        handle = ipc_handles[get_physical_gpu_id()]
        func, handle_args = handle
        args_list = list(handle_args)
        args_list[6] = torch.cuda.current_device()
        weight = func(*args_list)

        import pickle

        serialized = pickle.dumps({name: weight})
        success, msg = self.engine.update_weights_from_tensor(
            serialized_named_tensors=serialized,
            load_format=None,
            flush_cache=empty_cache,
        )
        torch.cuda.synchronize()
        if not success:
            raise RuntimeError(f"TokenSpeed update_weight_cuda_ipc failed for {name}: {msg}")

    # ------------------------------------------------------------------
    # Memory management — maps to sleep / wake_up
    # ------------------------------------------------------------------

    async def sleep(self, level=1):
        """Release GPU memory (KV cache, optionally weights)."""
        self.engine.release_memory_occupation()

    async def wake_up(self, tags=None):
        """Resume GPU memory occupation."""
        if tags is None:
            tags = ["weights", "kv_cache"]
        self.engine.resume_memory_occupation()

    async def reset_prefix_cache(self):
        """Flush the prefix / KV cache."""
        # TokenSpeed scheduler manages KV cache via its C++ FSM.
        # A full flush is equivalent to aborting all cached state.
        pass

    async def pause_generation(self):
        """Pause in-flight generation (for partial rollout weight sync)."""
        pass

    async def resume_generation(self):
        """Resume generation after pause."""
        pass

    # ------------------------------------------------------------------
    # Generation — mirrors the vLLM actor interface
    # ------------------------------------------------------------------

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None):
        """Token-level generation compatible with ``SamplesGenerator``."""
        if multi_modal_data and self._mm_pad_token_ids:
            from openrlhf.utils.vlm_utils import dedup_media_tokens

            prompt_token_ids = dedup_media_tokens(prompt_token_ids, self._mm_pad_token_ids)

        # Convert vLLM SamplingParams to a plain dict that TokenSpeed accepts.
        gen_kwargs = {
            "temperature": getattr(sampling_params, "temperature", 1.0),
            "top_p": getattr(sampling_params, "top_p", 1.0),
            "top_k": getattr(sampling_params, "top_k", -1),
            "max_new_tokens": getattr(sampling_params, "max_tokens", 512),
            "min_new_tokens": getattr(sampling_params, "min_tokens", 1),
            "skip_special_tokens": getattr(sampling_params, "skip_special_tokens", False),
        }

        logprobs_k = getattr(sampling_params, "logprobs", None)

        output = self.engine.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=gen_kwargs,
            logprobs=logprobs_k,
        )
        return output

    def get_num_unfinished_requests(self) -> int:
        return 0

    async def generate_responses(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        num_samples: int = 1,
        images=None,
    ):
        """Generate N samples for a single prompt."""
        tasks = [
            self.executor.execute(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=self,
                images=images,
            )
            for _ in range(num_samples)
        ]
        return await asyncio.gather(*tasks)


# ------------------------------------------------------------------
# Factory — mirrors ``create_vllm_engines``
# ------------------------------------------------------------------


def create_tokenspeed_engines(
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
    enable_sleep=False,
    logprobs_mode=None,
    agent_func_path: Optional[str] = None,
    remote_rm_url: Optional[str] = None,
    max_images_per_prompt: int = 0,
):
    """Spin up TokenSpeed Ray actors with the same placement logic as vLLM."""
    mm_pad_token_ids: set = set()
    if max_images_per_prompt > 0:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True)
        for attr in ("image_token_id", "video_token_id"):
            tid = getattr(processor, attr, None)
            if tid is not None:
                mm_pad_token_ids.add(tid)
        del processor

    engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        num_gpus = 0.2

    if not use_hybrid_engine:
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
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": 0.2 if use_hybrid_engine else 1,
            "agent_func_path": agent_func_path,
            "remote_rm_url": remote_rm_url,
            "mm_pad_token_ids": mm_pad_token_ids,
        }

        if max_images_per_prompt > 0:
            actor_kwargs["limit_mm_per_prompt"] = {"image": max_images_per_prompt}

        engines.append(
            TokenSpeedRolloutActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    if enable_sleep:
        batch_engine_call(engines, "sleep")

    return engines


def batch_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
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
