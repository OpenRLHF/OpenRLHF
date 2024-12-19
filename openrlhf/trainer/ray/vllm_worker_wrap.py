import torch
from vllm.worker.worker import Worker

from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging_utils import init_logger

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

    def update_weight(self, param_chunk_list):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        for param_chunk in param_chunk_list:
            handles = []
            weights = []
            for name, dtype, shape in param_chunk:
                if torch.distributed.get_rank() == 0:
                    print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

                weight = torch.empty(shape, dtype=dtype, device="cuda")
                handle = torch.distributed.broadcast(weight, 0, group=self._model_update_group, async_op=True)
                handles.append(handle)
                weights.append((name, weight))

            for handle in handles:
                handle.wait()

            self.model_runner.model.load_weights(weights=weights)
            del weights

        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()
