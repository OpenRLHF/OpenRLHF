from typing import Any, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..utils import get_bundle_indices
from .async_actor import LLMRayActorAsync


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
    llm_actor_cls=LLMRayActorAsync,
    logprobs_mode=None,
    agent_func_path=None,
    remote_rm_url=None,
    remote_rm_batch_size=None,
):
    import vllm
    from packaging import version

    assert version.parse(vllm.__version__) > version.parse("0.8.2"), "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
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

        additional_kwargs = {}
        if logprobs_mode:
            additional_kwargs["logprobs_mode"] = logprobs_mode
            additional_kwargs["max_logprobs"] = 1
            assert version.parse(vllm.__version__) > version.parse(
                "0.10.0"
            ), "vLLM > 0.10.0 is required for logprobs_mode"

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
            "agent_func_path": agent_func_path,
            **additional_kwargs,
        }

        # Only non-agent actors support remote_rm_url-based rewards.
        if remote_rm_url:
            actor_kwargs["remote_rm_url"] = remote_rm_url
            if remote_rm_batch_size is not None:
                actor_kwargs["remote_rm_batch_size"] = remote_rm_batch_size

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
