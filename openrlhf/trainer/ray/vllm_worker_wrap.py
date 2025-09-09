import gc
from typing import Callable, Optional, TypedDict

import torch
import zmq


def rebuild_ipc(handle: tuple[Callable, tuple], device_id: Optional[int] = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int


class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        import torch

        from openrlhf.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handles[self.report_device_id()])
        buffer: Optional[torch.Tensor] = None
        while True:
            payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = socket.recv_pyobj()
            if payload is None:
                # means the update is done
                process_weights_after_loading(self.model_runner.model, self.model_config, self.device)
                torch.cuda.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, tuple):
                # an ipc handle that vLLM can use `func, args = handle`
                # and `func(*args)` to rebuild GPU tensor.
                buffer = rebuild_ipc(payload, self.device.index)
                assert buffer.dtype == torch.uint8
                socket.send(b"")
                continue
            assert isinstance(payload, list)
            assert buffer is not None
            weights = []
            for item in payload:
                shape = item["shape"]
                if isinstance(shape, (list, tuple)):
                    shape = torch.Size(shape)
                assert isinstance(shape, torch.Size)
                dtype, offset = item["dtype"], item["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                weights.append((item["name"], tensor))
            self.model_runner.model.load_weights(weights=weights)
            del weights
            torch.cuda.synchronize()
            socket.send(b"")

        socket.close()
        del buffer
        gc.collect()
        torch.cuda.empty_cache()
