import importlib
import inspect

import torch
from peft.utils.other import transpose
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging import init_logger

logger = init_logger(__name__)


class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False, lora_param_info=None):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")
            if lora_param_info is not None:
                print(f"lora_param_info: {lora_param_info}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)

        if lora_param_info is not None:
            lora_dtype = lora_param_info["dtype"]
            lora_shape_A = lora_param_info["shape_A"]
            lora_shape_B = lora_param_info["shape_B"]
            lora_scaling = lora_param_info["scaling"]
            fan_in_fan_out = lora_param_info["fan_in_fan_out"]
            lora_weight_A = torch.empty(lora_shape_A, dtype=lora_dtype, device="cuda")
            lora_weight_B = torch.empty(lora_shape_B, dtype=lora_dtype, device="cuda")
            torch.distributed.broadcast(lora_weight_A, 0, group=self._model_update_group)
            torch.distributed.broadcast(lora_weight_B, 0, group=self._model_update_group)
            delta_weight = get_delta_weight(lora_weight_A, lora_weight_B, lora_scaling, fan_in_fan_out)
            weight.data = weight.data + delta_weight

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()


def get_delta_weight(weight_A, weight_B, scaling, fan_in_fan_out=False):
    """
    Copied from peft/tuners/lora/layer.py
    Compute the delta weight for the given adapter.

    Args:
        adapter (str):
            The name of the adapter for which the delta weight should be computed.
    """
    device = weight_B.device
    dtype = weight_B.dtype

    # In case users wants to merge the adapter weights that are in
    # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
    # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
    cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

    if cast_to_fp32:
        weight_A = weight_A.float()
        weight_B = weight_B.float()

    output_tensor = transpose(weight_B @ weight_A, fan_in_fan_out) * scaling

    if cast_to_fp32:
        output_tensor = output_tensor.to(dtype=dtype)

        # cast back the weights
        weight_A.data = weight_A.to(dtype)
        weight_B.data = weight_B.to(dtype)

    return output_tensor
