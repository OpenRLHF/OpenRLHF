import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.trainer.ray.utils import ray_noset_visible_devices

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert self.__version__ >= "0.4.2", "OpenRLHF only supports vLLM >= 0.4.2"

        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            vllm.worker.worker.Worker = WorkerWrap
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs["worker_cls"] = "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "WorkerWrap"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        self.llm = vllm.LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
):
    vllm_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    for i in range(num_engines):
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (vLLM mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # vLLM init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None

        if tensor_parallel_size > 1 or noset_visible_devices:
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
                noset_visible_devices=noset_visible_devices,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
            )
        )

    return vllm_engines


if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
