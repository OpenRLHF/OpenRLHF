import importlib
import inspect

import torch
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


class WorkerWrap(Worker):
    def __init__(self, *args, **kwargs):
        # Monkey patch hf_model_weights_iterator to allow update single weight
        import vllm

        if vllm.__version__ < "0.2.5":
            import vllm.model_executor.models

            modules = inspect.getmembers(vllm.model_executor.models, inspect.ismodule)
            for _, m in modules:
                m.hf_model_weights_iterator = _hf_model_weights_iterator_wrap
        else:
            # NOTE: In 0.2.5, vLLM introduce lazy model loader
            # https://github.com/vllm-project/vllm/pull/2044
            from vllm.model_executor.models import _MODELS, ModelRegistry

            load_model_cls = ModelRegistry.load_model_cls

            def patched_load_model_cls(model_arch: str):
                module_name, _ = _MODELS[model_arch]
                module = importlib.import_module(f"vllm.model_executor.models.{module_name}")
                module.hf_model_weights_iterator = _hf_model_weights_iterator_wrap
                logger.info(f"Monkey patch hf_model_weights_iterator for module {module_name}")

                return load_model_cls(model_arch)

            ModelRegistry.load_model_cls = patched_load_model_cls

        super().__init__(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f"init_process_group: master_address={master_address}, master_port={master_port}, "
            f"rank={rank}, world_size={world_size}, group_name={group_name}"
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
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
