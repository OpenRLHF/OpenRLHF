import os
from typing import List

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        if vllm.__version__ < "0.2.7" or kwargs["tensor_parallel_size"] == 1:
            from vllm.worker import worker
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            worker.Worker = WorkerWrap
        else:
            # NOTE: In 0.2.7, vLLM made a major change to its architecture which move one worker into the driver process.
            # Driver process will manually set CUDA_VISIBLE_DEVICES before worker init. To avoid importing torch before
            # set CUDA_VISIBLE_DEVICES, we must defer monkey patch.
            # For more detail, see: https://github.com/vllm-project/vllm/pull/2221
            def _set_cuda_visible_devices(device_ids: List[int]):
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

                from vllm.worker import worker
                from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

                worker.Worker = WorkerWrap

            vllm.engine.llm_engine.set_cuda_visible_devices = _set_cuda_visible_devices

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name):
        return self.llm.llm_engine._run_workers(
            "init_process_group", master_address, master_port, rank_offset, world_size, group_name
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.llm_engine._run_workers("update_weight", name, dtype, shape, empty_cache)


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
