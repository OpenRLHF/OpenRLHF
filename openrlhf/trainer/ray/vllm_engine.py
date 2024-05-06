import os
from typing import Dict, List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        assert vllm.__version__ >= "0.4.1", "OpenRLHF only supports vLLM >= 0.4.1"

        # See https://github.com/vllm-project/vllm/blob/0650e5935b0f6af35fb2acf71769982c47b804d7/vllm/executor/gpu_executor.py
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # WorkerWrapperBase was used by GPU Executor and Ray GPU Executor in vLLM
        # We hack this base class to implement weight updates
        # https://github.com/vllm-project/vllm/blob/0650e5935b0f6af35fb2acf71769982c47b804d7/vllm/worker/worker_base.py#L92
        class WorkerWrapperBase2(vllm.worker.worker_base.WorkerWrapperBase):
            def __init__(self, *args, **kwargs) -> None:
                kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                kwargs["worker_class_name"] = "WorkerWrap"
                super().__init__(*args, **kwargs)

        vllm.worker.worker_base.WorkerWrapperBase = WorkerWrapperBase2

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)


def create_vllm_engines(num_engines: int, tensor_parallel_size: int, pretrain: str, seed: int):
    vllm_engines = []
    for _ in range(num_engines):
        # When tensor_parallel_size=1, vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1)
        scheduling_strategy = None

        if tensor_parallel_size > 1:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed,
            )
        )

    return vllm_engines


if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
