import time

import ray
import sglang
import torch
import vllm
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)
from abc import ABC, abstractmethod


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


@ray.remote
class LLMRayActor(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def init_process_group(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_weight(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop_remote_worker_execution_loop(self, *args, **kwargs):
        pass


@ray.remote
class VllmLLMRayActor(LLMRayActor):
    def __init__(self, *args, **kwargs):
        print("LLMRayActor __init__")
        torch.cuda.synchronize()
        print("torch.cuda.synchronize()")
        start = time.time()
        print("start", start)
        print(f"kwargs: {kwargs}")
        print(f"using backend: {self.backend}")
        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices
        print("noset_visible_devices", noset_visible_devices)
        self.__version__ = vllm.__version__
        print("self.__version__", self.__version__)
        assert self.__version__ >= "0.4.2", "OpenRLHF only supports vLLM >= 0.4.2"

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            print("self.use_gpu_executor")
            from openrlhf.trainer.ray.vllm_worker_wrap import WorkerWrap

            print("WorkerWrap", WorkerWrap)
            vllm.worker.worker.Worker = WorkerWrap
        else:
            print("else")
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            #! worker_use_ray is a vllm only parameter
            kwargs["worker_use_ray"] = True
            print("kwargs['worker_use_ray']", kwargs["worker_use_ray"])
            if vllm.__version__ > "0.6.4.post1":
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs["worker_cls"] = "openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap"
            else:
                print("else")
                RayWorkerWrapperPath = vllm.executor.ray_utils
                print("RayWorkerWrapperPath", RayWorkerWrapperPath)

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs["worker_module_name"] = "openrlhf.trainer.ray.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "WorkerWrap"
                        super().__init__(*args, **kwargs)

                    print("RayWorkerWrapper __init__")

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper
        kwargs.pop("backend")
        print("kwargs.pop('backend')")
        self.llm = vllm.LLM(*args, **kwargs)
        print("self.llm", self.llm)
        torch.cuda.synchronize()
        end = time.time()
        print("torch.cuda.synchronize()")
        print(f"Create inference engines takes: {end - start}s for {self.backend}")

    def generate(self, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        print("Start generate")
        print("self.llm.generate")
        outputs = self.llm.generate(
            sampling_params=kwargs["sampling_params"], prompt_token_ids=kwargs["prompt_token_ids"]
        )
        torch.cuda.synchronize()
        end = time.time()
        print(f"Generate samples takes: {end - start}s for {self.backend}")
        print("return outputs")
        return outputs

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        print("self.use_gpu_executor", self.use_gpu_executor)
        if self.use_gpu_executor:
            print("self.llm.llm_engine.model_executor.driver_worker.init_process_group")
            return self.llm.llm_engine.model_executor.driver_worker.init_process_group(
                master_address, master_port, rank_offset, world_size, group_name, backend
            )
        else:
            print("self.llm.llm_engine.model_executor._run_workers")
            return self.llm.llm_engine.model_executor._run_workers(
                "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
            )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        print("self.stop_remote_worker_execution_loop")
        self.stop_remote_worker_execution_loop()

        if self.use_gpu_executor:
            print("self.llm.llm_engine.model_executor.driver_worker.update_weight")
            return self.llm.llm_engine.model_executor.driver_worker.update_weight(name, dtype, shape, empty_cache)
        else:
            print("self.llm.llm_engine.model_executor._run_workers")
            return self.llm.llm_engine.model_executor._run_workers("update_weight", name, dtype, shape, empty_cache)

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if self.__version__ > "0.4.2":
            print("self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop")
            self.llm.llm_engine.model_executor.stop_remote_worker_execution_loop()


@ray.remote
class SGLangLLMRayActor(LLMRayActor):
    def __init__(self, *args, **kwargs):
        #! TODO chenyang check engine params
        print("LLMRayActor __init__")
        self.backend = kwargs["backend"]
        torch.cuda.synchronize()
        print("torch.cuda.synchronize()")
        start = time.time()
        print("start", start)
        print(f"kwargs: {kwargs}")
        print(f"using backend: {self.backend}")
        noset_visible_devices = kwargs.pop("noset_visible_devices", False)
        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1 and not noset_visible_devices
        print("noset_visible_devices", noset_visible_devices)
        print("import sglang")
        #! TODO chenyang check engine params
        sglang_sever_params = {
            "model_path": args[0],  # pretrain path
            "trust_remote_code": kwargs.get("trust_remote_code", True),
            "dtype": kwargs.get("dtype", "auto"),
            "tp_size": kwargs.get("tensor_parallel_size", 1),
            "device": "cuda",
            "disable_radix_cache": not kwargs.get("enable_prefix_caching", False),
            "random_seed": kwargs.get("seed", 42),
            "disable_cuda_graph": not kwargs.get("enforce_eager", False),
            "disable_cuda_graph_padding": not kwargs.get("enable_prefix_caching", False),
            "context_length": kwargs.get("max_model_len", None),
            "log_level": "info",
            "return_token_ids": True,
        }
        print("sglang_params", sglang_sever_params)
        self.llm = sglang.Engine(**sglang_sever_params)
        print("self.llm", self.llm)

    torch.cuda.synchronize()
    end = time.time()
    print("torch.cuda.synchronize()")
    print(f"Create inference engines takes: {end - start}s for {self.backend}")

    def generate(self, *args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        print("Start generate")
        # Note that sglang sampling params are different from vllm
        print("sglang sampling params")
        sampling_params = kwargs["sampling_params"]
        all_prompts = kwargs["all_prompts"]

        # min_tokens, include_stop_str_in_output is not used in sglang

        sampling_params = dict(
            max_new_tokens=sampling_params.max_tokens,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            temperature=sampling_params.temperature,
            repetition_penalty=sampling_params.repetition_penalty,
            skip_special_tokens=sampling_params.skip_special_tokens,
        )
        print("outputs = self.llm.generate(all_prompts, sampling_params)")
        outputs = self.llm.generate(all_prompts, sampling_params)
        torch.cuda.synchronize()
        end = time.time()
        print(f"Generate samples takes: {end - start}s for {self.backend}")
        print("return outputs")
        return outputs

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        print("self.llm.init_weights_update_group")
        return self.llm.init_weights_update_group(
            master_address, master_port, rank_offset, world_size, group_name, backend="nccl"
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        print("self.llm.update_weights_from_distributed")
        return self.llm.update_weights_from_distributed(name, dtype, shape)

    def stop_remote_worker_execution_loop(self):
        # SGLang does not needs this function
        pass


def create_llm_ray_actor(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    backend: str = "vllm",
):
    print(f"backend: {backend}")
    inference_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    print("noset_visible_devices", noset_visible_devices)
    for i in range(num_engines):
        print("i", i)
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (vLLM/SGLang mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # vLLM/SGLang init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None
        print("tensor_parallel_size", tensor_parallel_size)
        print("noset_visible_devices", noset_visible_devices)
        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())
            print("pg.ready()")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )
            print("scheduling_strategy", scheduling_strategy)
        if backend == "vllm":
            actor_cls = VllmLLMRayActor
        elif backend == "sglang":
            actor_cls = SGLangLLMRayActor

        inference_engines.append(
            actor_cls.options(
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
                backend=backend,
            )
        )
        print("inference_engines", inference_engines)
    return inference_engines


if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
