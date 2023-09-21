import logging
import os
import socket
from typing import Optional, Type

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openllama2.models import Actor, Critic, RewardModel
from openllama2.utils import DeepspeedStrategy, get_tokenizer


class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        address = address.lstrip("[")
        address = address.rstrip("]")
        return address

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port


class BasePPORole(DistributedTorchRayActor):
    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()

        # configure flash attention
        if strategy.args.flash_attn:
            from openllama2.models.llama2_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

            replace_llama_attn_with_flash_attn()

    def _from_pretrained(self, model_type, pretrain, model_path, **kwargs):
        # load huggingface model/config
        from_config = bool(model_path)
        model = model_type(pretrain, from_config, **kwargs)

        # configure tokenizer
        tokenizer = get_tokenizer(pretrain, model.model, "left", self.strategy)

        # load PyTorch model
        if model_path:
            self.strategy.load_model(model, model_path)

        return model, tokenizer

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote(num_gpus=1)
class ReferenceModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, model_path):
        self._setup_distributed(strategy)
        model, _ = self._from_pretrained(Actor, pretrain, model_path)
        strategy.print(model)

        self.model = self.strategy.prepare(model, is_rlhf=True)

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(sequences.to(device), num_actions, attention_mask.to(device), return_output)
        return log_probs.to("cpu")


@ray.remote(num_gpus=1)
class RewardModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, model_path):
        self._setup_distributed(strategy)
        model, _ = self._from_pretrained(
            RewardModel, pretrain, model_path, normalize_reward=strategy.args.normalize_reward
        )
        strategy.print(model)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        self.model = self.strategy.prepare(model, is_rlhf=True)

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(sequences.to(device), attention_mask.to(device))
        return reward.to("cpu")


class PPORayActorGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs
    """

    def __init__(self, num_nodes, num_gpus_per_node, ray_actor_type: Type[BasePPORole]) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node
        self.ray_actor_type = ray_actor_type
        self._initiate_actors()

    def _initiate_actors(self):
        world_size = self._num_nodes * self._num_gpus_per_node
        # Use placement group to lock resources for models of same type
        pg = None
        if self._num_gpus_per_node > 1:
            bundles = [
                {"GPU": self._num_gpus_per_node, "CPU": self._num_gpus_per_node} for _ in range(self._num_nodes)
            ]
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
        if pg:
            master_actor = self.ray_actor_type.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=0
                )
            ).remote(world_size, 0, 0, None, None)
        else:
            master_actor = self.ray_actor_type.options(num_gpus=1).remote(world_size, 0, 0, None, None)
        self._actor_handlers = [master_actor]

        # Create worker actors
        if world_size > 1:
            master_addr, master_port = ray.get(master_actor.get_master_addr_port.remote())
            for rank in range(1, world_size):
                local_rank = rank % self._num_gpus_per_node
                if pg:
                    worker_actor = self.ray_actor_type.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=rank // self._num_gpus_per_node,
                        )
                    ).remote(world_size, rank, local_rank, master_addr, master_port)
                else:
                    worker_actor = self.ray_actor_type.options(num_gpus=1).remote(
                        world_size, rank, local_rank, master_addr, master_port
                    )
                self._actor_handlers.append(worker_actor)

    def async_init_model_from_pretrained(
        self,
        *args,
        **kwargs,
    ):
        return [actor.init_model_from_pretrained.remote(*args, **kwargs) for actor in self._actor_handlers]

    def async_fit_actor_model(
        self,
        critic_model_group: "PPORayActorGroup",
        initial_model: "PPORayActorGroup",
        reward_model: "PPORayActorGroup",
    ):
        critic_actors = critic_model_group._actor_handlers
        reward_actors = reward_model._actor_handlers
        initial_actors = initial_model._actor_handlers
        return [
            actor.fit.remote(
                critic_actors[i % len(critic_actors)],
                reward_actors[i % len(reward_actors)],
                initial_actors[i % len(initial_actors)],
                critic_train_remote=(i < len(critic_actors)),
            )
            for i, actor in enumerate(self._actor_handlers)
        ]

    def async_save_actor_model(self):
        return [actor.save_model.remote() for actor in self._actor_handlers]
