import os

import torch
import torch.distributed as dist
import torch.nn as nn
from chatgpt.models import Actor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from .naive import NaiveStrategy


class DDPStrategy(NaiveStrategy):
    """
        Strategy for distributed training using torch.distributed.
    """

    def setup_distributed(self) -> None:
        try:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            host = os.environ['MASTER_ADDR']
            port = int(os.environ['MASTER_PORT'])
        except KeyError as e:
            raise RuntimeError(
                f"DDP: Could not find {e} in the torch environment"
            )
        dist.init_process_group('nccl', init_method=f'tcp://[{host}]:{port}', world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        self.world_size = dist.get_world_size()

    def setup_model(self, model: nn.Module) -> nn.Module:
        device = torch.cuda.current_device()
        return DDP(model, device_ids=[device])

    def setup_dataloader(self, replay_buffer, batch_size: int, pin_memory: bool = False, shuffle=True, collate_fn=None) -> DataLoader:
        # DDP only mode, replay buffers on each rank are different.
        sampler = DistributedSampler(replay_buffer,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=shuffle,
                                     seed=self.seed,
                                     drop_last=True)
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=pin_memory)


    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return model.model
        elif isinstance(model, DDP):
            return self._unwrap_model(model.module)
        return model

    def save_model(self, model: nn.Module, path: str, only_rank0: bool = False) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = unwrapped_model.state_dict()
        if dist.get_rank() != 0:
            return
        torch.save(state_dict, path)

    def all_reduce(self, data, op='mean'):
        assert op in ('mean', 'max', 'sum')
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"
            
            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == 'mean':
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == 'max' else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0
    

