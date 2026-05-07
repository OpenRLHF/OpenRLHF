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
from dataclasses import dataclass
from typing import Any, List, Optional

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.agent import AgentExecutorBase, SingleTurnAgentExecutor

from .utils import ray_noset_visible_devices


@dataclass
class _TokenSpeedLogprob:
    logprob: float


@dataclass
class _TokenSpeedCompletionOutput:
    token_ids: list[int]
    text: str
    finish_reason: Optional[str]
    logprobs: Optional[list[dict[int, _TokenSpeedLogprob]]] = None


@dataclass
class _TokenSpeedRequestOutput:
    outputs: list[_TokenSpeedCompletionOutput]


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


def _torch_dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _tokenspeed_sampling_params(sampling_params) -> dict[str, Any]:
    max_tokens = getattr(sampling_params, "max_tokens", None)
    if max_tokens is None:
        raise ValueError("TokenSpeed rollout requires sampling_params.max_tokens to be set")

    params = {"max_new_tokens": max_tokens}
    attr_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "min_p": "min_p",
        "min_tokens": "min_new_tokens",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repetition_penalty": "repetition_penalty",
        "skip_special_tokens": "skip_special_tokens",
        "spaces_between_special_tokens": "spaces_between_special_tokens",
        "stop": "stop",
        "stop_token_ids": "stop_token_ids",
        "ignore_eos": "ignore_eos",
        "seed": "seed",
    }
    for source, target in attr_map.items():
        value = getattr(sampling_params, source, None)
        if value is not None:
            params[target] = value

    return params


def _tokenspeed_image_data(multi_modal_data):
    if multi_modal_data is None:
        return None

    unsupported = set(multi_modal_data) - {"image"}
    if unsupported:
        raise NotImplementedError(f"TokenSpeed rollout does not support multimodal keys: {sorted(unsupported)}")

    return multi_modal_data.get("image")


def _finish_reason_from_tokenspeed(reason) -> Optional[str]:
    if reason is None:
        return None
    if isinstance(reason, dict):
        return reason.get("type")
    return str(reason)


def _tokenspeed_logprobs(meta_info: dict[str, Any], token_ids: list[int]):
    raw_logprobs = meta_info.get("output_token_logprobs")
    if raw_logprobs is None:
        raise RuntimeError("TokenSpeed did not return output token logprobs")
    if len(raw_logprobs) != len(token_ids):
        raise RuntimeError(
            f"TokenSpeed returned {len(raw_logprobs)} logprobs for {len(token_ids)} generated tokens"
        )

    converted = []
    for token_id, raw in zip(token_ids, raw_logprobs):
        if isinstance(raw, dict):
            logged_token_id = raw.get("token_id", token_id)
            logprob = raw.get("logprob")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            logprob, logged_token_id = raw[0], raw[1]
        else:
            raise TypeError(f"Unsupported TokenSpeed logprob entry: {raw!r}")

        if int(logged_token_id) != int(token_id):
            raise RuntimeError(
                f"TokenSpeed logprob token id {logged_token_id} does not match output token id {token_id}"
            )
        converted.append({int(token_id): _TokenSpeedLogprob(float(logprob))})

    return converted


def _adapt_tokenspeed_output(output: dict[str, Any] | list[dict[str, Any]], require_logprobs: bool):
    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(f"Expected one TokenSpeed generation output, got {len(output)}")
        output = output[0]
    if not isinstance(output, dict):
        raise TypeError(f"Expected TokenSpeed generation output to be a dict, got {type(output)!r}")
    if "text" not in output:
        raise ValueError("TokenSpeed generation output is missing text")
    if "output_ids" not in output:
        raise ValueError("TokenSpeed generation output is missing output_ids")

    token_ids = list(output["output_ids"])
    meta_info = output.get("meta_info") or {}
    logprobs = _tokenspeed_logprobs(meta_info, token_ids) if require_logprobs else None

    return _TokenSpeedRequestOutput(
        outputs=[
            _TokenSpeedCompletionOutput(
                token_ids=token_ids,
                text=output["text"],
                finish_reason=_finish_reason_from_tokenspeed(meta_info.get("finish_reason")),
                logprobs=logprobs,
            )
        ]
    )


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
        distributed_executor_backend = kwargs.pop("distributed_executor_backend", None)
        self._configure_device_env(
            backend=distributed_executor_backend,
            bundle_indices=bundle_indices,
            num_gpus=num_gpus,
        )

        if agent_func_path:
            self.executor = _load_agent_executor(agent_func_path)
        else:
            self.executor = SingleTurnAgentExecutor(remote_rm_url)

        self._mm_pad_token_ids = mm_pad_token_ids
        self._num_unfinished_requests = 0
        self._generation_resumed = asyncio.Event()
        self._generation_resumed.set()

        # Import tokenspeed lazily so it doesn't pollute other Ray actors.
        from tokenspeed.runtime.entrypoints.engine import Engine

        self.engine = Engine(*args, **kwargs)

    def _configure_device_env(self, backend, bundle_indices, num_gpus):
        if ray_noset_visible_devices():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        if bundle_indices is not None:
            os.environ["TOKENSPEED_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating TokenSpeed engine with bundle_indices={bundle_indices}")

    def _run_tokenizer_manager(self, coroutine):
        return self.engine.llm.run(coroutine)

    # ------------------------------------------------------------------
    # Weight synchronisation — maps to TokenSpeed's distributed API
    # ------------------------------------------------------------------

    async def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray
    ):
        """Initialise NCCL process group for DeepSpeed -> TokenSpeed weight sync."""
        if use_ray:
            raise NotImplementedError("TokenSpeed rollout weight sync does not support Ray collective mode")

        from tokenspeed.runtime.engine.io_struct import InitWeightsUpdateGroupReqInput

        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        success, msg = self._run_tokenizer_manager(self.engine.tokenizer_manager.init_weights_update_group(obj))
        if not success:
            raise RuntimeError(f"TokenSpeed init_weights_update_group failed: {msg}")

    async def update_weight(self, name, dtype, shape, empty_cache=False):
        """Receive a single weight tensor via the NCCL group."""
        from tokenspeed.runtime.engine.io_struct import UpdateWeightsFromDistributedReqInput

        obj = UpdateWeightsFromDistributedReqInput(
            name=name,
            dtype=_torch_dtype_name(dtype),
            shape=list(shape),
        )
        success, msg = self._run_tokenizer_manager(self.engine.tokenizer_manager.update_weights_from_distributed(obj))
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

        success, msg = self.engine.update_weights_from_tensor(
            named_tensors=[(name, weight)],
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
        from tokenspeed.runtime.engine.io_struct import ReleaseMemoryOccupationReqInput

        obj = ReleaseMemoryOccupationReqInput()
        self._run_tokenizer_manager(self.engine.tokenizer_manager.release_memory_occupation(obj))

    async def wake_up(self, tags=None):
        """Resume GPU memory occupation."""
        from tokenspeed.runtime.engine.io_struct import ResumeMemoryOccupationReqInput

        obj = ResumeMemoryOccupationReqInput()
        self._run_tokenizer_manager(self.engine.tokenizer_manager.resume_memory_occupation(obj))

    async def reset_prefix_cache(self):
        """Flush the prefix / KV cache."""
        result = self.engine.flush_cache()
        if not getattr(result, "success", False):
            raise RuntimeError("TokenSpeed flush_cache failed")

    async def pause_generation(self):
        """Pause in-flight generation (for partial rollout weight sync)."""
        self._generation_resumed.clear()

    async def resume_generation(self):
        """Resume generation after pause."""
        self._generation_resumed.set()

    # ------------------------------------------------------------------
    # Generation — mirrors the vLLM actor interface
    # ------------------------------------------------------------------

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None):
        """Token-level generation compatible with ``SamplesGenerator``."""
        await self._generation_resumed.wait()
        if multi_modal_data and self._mm_pad_token_ids:
            from openrlhf.utils.vlm_utils import dedup_media_tokens

            prompt_token_ids = dedup_media_tokens(prompt_token_ids, self._mm_pad_token_ids)

        require_logprobs = getattr(sampling_params, "logprobs", None) is not None
        self._num_unfinished_requests += 1
        try:
            output = await asyncio.to_thread(
                self.engine.generate,
                input_ids=list(prompt_token_ids),
                image_data=_tokenspeed_image_data(multi_modal_data),
                sampling_params=_tokenspeed_sampling_params(sampling_params),
                return_logprob=require_logprobs,
                logprob_start_len=-1 if require_logprobs else None,
                top_logprobs_num=0,
            )
        finally:
            self._num_unfinished_requests -= 1

        return _adapt_tokenspeed_output(output, require_logprobs=require_logprobs)

    def get_num_unfinished_requests(self) -> int:
        return self._num_unfinished_requests

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
    use_hybrid_engine = shared_pg is not None
    if use_hybrid_engine and tensor_parallel_size > 1:
        raise ValueError("TokenSpeed rollout does not support colocated tensor_parallel_size > 1")

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
    actor_num_gpus = 0.2 if use_hybrid_engine else tensor_parallel_size
    actor_num_cpus = actor_num_gpus

    if not use_hybrid_engine:
        bundles = [{"GPU": tensor_parallel_size, "CPU": max(1, tensor_parallel_size)} for _ in range(num_engines)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "seed": seed + i,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "world_size": tensor_parallel_size,
            "nprocs_per_node": tensor_parallel_size,
            "attn_tp_size": tensor_parallel_size,
            "enable_output_logprobs": bool(logprobs_mode),
            "num_gpus": actor_num_gpus,
            "agent_func_path": agent_func_path,
            "remote_rm_url": remote_rm_url,
            "mm_pad_token_ids": mm_pad_token_ids,
        }

        engines.append(
            TokenSpeedRolloutActor.options(
                num_cpus=actor_num_cpus,
                num_gpus=actor_num_gpus,
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
