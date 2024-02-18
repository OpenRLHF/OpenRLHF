import inspect
import importlib
from functools import partial

import ray
import torch
from vllm import LLM
from vllm.model_executor.weight_utils import hf_model_weights_iterator
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


def _hf_model_weights_iterator_wrap(model_name_or_path, *args, **kwargs):
    if isinstance(model_name_or_path, dict):
        for name, param in model_name_or_path.items():
            yield name, param
    else:
        yield from hf_model_weights_iterator(model_name_or_path, *args, **kwargs)


class _WorkerWrap(Worker):
    def __init__(self, *args, **kwargs):
        # Monkey patch hf_model_weights_iterator to allow update single weight
        import vllm

        if vllm.__version__ < "0.2.5":
            import vllm.model_executor.models

            modules = inspect.getmembers(vllm.model_executor.models, inspect.ismodule)
            for _, m in modules:
                m.hf_model_weights_iterator = _hf_model_weights_iterator_wrap
        else:
            from vllm.model_executor.models import ModelRegistry, _MODELS

            load_model_cls = ModelRegistry.load_model_cls

            def patched_load_model_cls(model_arch: str):
                module_name, _ = _MODELS[model_arch]
                module = importlib.import_module(f"vllm.model_executor.models.{module_name}")
                module.hf_model_weights_iterator = _hf_model_weights_iterator_wrap
                logger.info(f"Monkey patch hf_model_weights_iterator for module {module_name}")

                return load_model_cls(model_arch)

            ModelRegistry.load_model_cls = patched_load_model_cls

        super().__init__(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank, world_size, group_name):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        self._model_update_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        if torch.distributed.get_rank() == 0:
            logger.debug(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        self.model_runner.model.load_weights(model_name_or_path={name: weight})

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()


@ray.remote
class LLMRayActor:
    def __init__(self, *args, **kwargs):
        from vllm.worker import worker

        worker.Worker = _WorkerWrap

        self.llm = LLM(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank, world_size, group_name):
        all_outputs = []
        use_ray = len(self.llm.llm_engine.workers) > 1

        for i, worker in enumerate(self.llm.llm_engine.workers):
            if use_ray:
                executor = partial(worker.execute_method.remote, "init_process_group")
            else:
                executor = getattr(worker, "init_process_group")

            output = executor(master_address, master_port, rank + i, world_size, group_name)
            all_outputs.append(output)

        if use_ray:
            ray.get(all_outputs)

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.llm.llm_engine._run_workers("update_weight", name, dtype, shape, empty_cache)


if __name__ == "__main__":
    llm = LLMRayActor.remote("meta-llama/Llama-2-7b-chat-hf", tensor_parallel_size=4)
    output = ray.get(llm.generate.remote("San Franciso is a"))
    print(f"output: {output}")
