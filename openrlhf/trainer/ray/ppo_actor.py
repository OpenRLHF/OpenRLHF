import math
import os
import socket
from abc import ABC
from dataclasses import fields
from typing import Dict, List, Optional, Union

import deepspeed
import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.trainer import get_scheduler

from openrlhf.models import Actor, PolicyLoss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.vlm_utils import merge_mm_train_inputs

from ..ppo_utils import NaiveReplayBuffer

logger = init_logger(__name__)

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


class ActorPPOTrainer(ABC):

    def __init__(
        self,
        strategy,
        actor: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        ema_beta: float = 0.992,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        tokenizer=None,
        dataloader_pin_memory: bool = True,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
        """
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.generate_kwargs = kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.micro_train_batch_size = micro_train_batch_size
        self.ema_beta = ema_beta

        self.actor = actor
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.vllm_engines = vllm_engines
        self.max_epochs = self.args.max_epochs

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.eps_clip_low_high[0],
            clip_eps_high=self.args.eps_clip_low_high[1],
            dual_clip=self.args.dual_clip,
            policy_loss_type=self.args.policy_loss_type,
            enable_vllm_is_correction=self.args.enable_vllm_is_correction,
            vllm_is_truncated_threshold=(
                self.args.vllm_is_truncated_threshold if self.args.enable_vllm_is_correction else None
            ),
            vllm_is_correction_type=self.args.vllm_is_correction_type,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            getattr(self.args, "packing_samples", False),
            self.args.use_dynamic_batch,
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = backend == "nccl" and self.args.colocate_all_models and not self.args.async_train

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            self._init_vllm_sync_group(backend)

        torch_dist_barrier_and_cuda_sync()

    def _init_vllm_sync_group(self, backend: str):
        """Create a torch process group between DeepSpeed rank 0 and all vLLM engine ranks.

        Layout example (3 engines, TP=4):
            [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
            |ds rank 0 |  engine-0  |  engine-1  |   engine-2   |

        ZeRO-1/2: broadcast params from rank 0 to all engines.
        ZeRO-3:   allgather to rank 0 first, then broadcast.
        """
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        vllm_num_engines = self.strategy.args.vllm_num_engines
        vllm_tensor_parallel_size = self.strategy.args.vllm_tensor_parallel_size
        world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

        use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
        group_name = "openrlhf"
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * vllm_tensor_parallel_size + 1,
                world_size,
                group_name,
                backend=backend,
                use_ray=use_ray,
            )
            for i, engine in enumerate(self.vllm_engines)
        ]
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=0, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address, master_port, 0, world_size, torch.cuda.current_device()
            )

        ray.get(refs)

    def ppo_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.use_dynamic_batch:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = (
            self.strategy.ring_attn_group is None
            and self.args.ds_tensor_parallel_size <= 1
            and not self.args.use_dynamic_batch
        )
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=should_shuffle,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for step, experience in enumerate(pbar):

                experience.to_device(device)
                status = self.training_step(experience, kl_ctl, step)

                metrics = status["metrics"]
                weights = status["weights"]
                n_tokens = status["num_action_tokens"]
                n_samples = status["num_samples"]

                reduced_status = {"_num_action_tokens": n_tokens, "_num_samples": n_samples}
                last_metrics = {}
                for k, value in metrics.items():
                    weight = weights[k]
                    if weight is None:
                        last_metrics[k] = value
                        continue
                    scale = n_tokens if weight == "token" else n_samples
                    reduced_status[k] = (
                        value.float().mean().item() * scale if isinstance(value, torch.Tensor) else value * scale
                    )

                reduced_status = self.strategy.all_reduce(reduced_status)

                n_tokens = reduced_status.pop("_num_action_tokens")
                n_samples = reduced_status.pop("_num_samples")
                merged_status = {}
                for k, value in reduced_status.items():
                    denom = n_tokens if weights[k] == "token" else n_samples
                    merged_status[k] = value / denom

                merged_status.update(last_metrics)
                merged_status["_num_samples"] = n_samples
                merged_status["_num_action_tokens"] = n_tokens
                merged_status["_weights"] = weights
                actor_lr = merged_status.get("actor_lr", 0)

                short_status = {
                    "act_loss": merged_status["policy_loss"],
                    "reward": merged_status.get("reward", 0),
                    "return": merged_status.get("return", 0),
                    "gen_len": merged_status.get("response_length", 0),
                    "tot_len": merged_status.get("total_length", 0),
                    "kl": merged_status.get("kl", 0),
                    "act_lr": actor_lr,
                    "grad_norm": merged_status.get("actor_grad_norm", 0),
                }
                if "entropy_loss" in merged_status:
                    short_status["ent_loss"] = merged_status["entropy_loss"]

                status_list.append(merged_status)
                pbar.set_postfix(short_status)

        if status_list:
            total_tokens = sum(s["_num_action_tokens"] for s in status_list)
            total_samples = sum(s["_num_samples"] for s in status_list)
            status_mean = {}
            for k in set().union(*(s.keys() for s in status_list)):
                if k in ("_num_samples", "_num_action_tokens", "_weights"):
                    continue
                if k in ("actor_grad_norm", "actor_lr"):
                    vals = [s[k] for s in status_list if k in s]
                    status_mean[k] = vals[-1] if vals else 0.0
                elif status_list[0].get("_weights", {}).get(k) == "token":
                    status_mean[k] = sum(s.get(k, 0) * s["_num_action_tokens"] for s in status_list) / total_tokens
                else:
                    status_mean[k] = sum(s.get(k, 0) * s["_num_samples"] for s in status_list) / total_samples
        return status_mean

    def training_step(self, experience: Experience, kl_ctl: float, step: int) -> Dict[str, float]:
        self.actor.train()

        sequences = experience.sequences
        action_mask = experience.action_mask
        attention_mask = experience.attention_mask
        packed_seq_lens = None
        old_action_log_probs = experience.action_log_probs
        advantages = experience.advantages
        base_action_log_probs = experience.base_action_log_probs

        # VLM: merge pre-processed multimodal inputs for training forward
        mm_inputs = {}
        if (
            hasattr(experience, "mm_train_inputs")
            and experience.mm_train_inputs
            and getattr(self.actor, "is_vlm", False)
        ):
            mm_inputs = merge_mm_train_inputs(experience.mm_train_inputs, sequences.device)

        # actor loss
        action_log_probs, output = self.actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.entropy_loss_coef is not None,
            **mm_inputs,
        )

        # loss function
        actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()

        if self.args.use_kl_loss:
            if self.args.init_kl_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs)
                logprobs_diff = torch.zeros_like(action_log_probs)
            kl_loss = masked_mean(kl, experience.action_mask)
            logprobs_diff = masked_mean(logprobs_diff, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
            experience.info["logprobs_diff"] = logprobs_diff.detach()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * kl_ctl
        # mixtral
        if self.aux_loss:
            loss += output.aux_loss * self.args.aux_loss_coef
        # entropy loss
        if self.args.entropy_loss_coef is not None:
            entropy_loss = masked_mean(output.entropy[:, -experience.action_mask.shape[1] :], experience.action_mask)
            if self.args.entropy_loss_coef != 0:
                loss -= entropy_loss * self.args.entropy_loss_coef

        if self.args.use_dynamic_batch:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        self.strategy.backward(loss, self.actor, self.actor_optim)
        if self.args.use_dynamic_batch:
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        else:
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        # Per-token losses (0-D tensors, shape carries weighting info for ppo_train)
        metrics = {"policy_loss": actor_loss.detach()}
        weights = {"policy_loss": "token"}
        if self.args.entropy_loss_coef is not None:
            metrics["entropy_loss"] = entropy_loss.detach()
            weights["entropy_loss"] = "token"

        # Non-reducible meta
        metrics["actor_lr"] = self.actor_scheduler.get_last_lr()[0]
        weights["actor_lr"] = None
        is_optimizer_step = not self.args.use_dynamic_batch or self.replay_buffer.dynamic_optimizer_step[step]
        if is_optimizer_step:
            metrics["actor_grad_norm"] = self.strategy.get_grad_norm(self.actor)
            weights["actor_grad_norm"] = None

        # Merge all loggable tensors.
        # `info` keeps algorithm metrics; episode tensor fields are added explicitly below.
        for k, v in experience.info.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v
                weights[k] = "token" if v.dim() == 0 else "sample"
            elif isinstance(v, list):
                metrics[k] = torch.tensor(v, dtype=torch.float)
                weights[k] = "sample"

        for f in fields(Experience):
            if f.name in {"rewards", "scores"} or not Experience.is_episode_tensor_field(f.name):
                continue
            value = getattr(experience, f.name)
            if isinstance(value, torch.Tensor) and f.name not in metrics:
                metrics[f.name] = value
                weights[f.name] = "sample"

        return {
            "metrics": metrics,
            "weights": weights,
            "num_samples": float(experience.action_mask.shape[0]),
            "num_action_tokens": float(experience.action_mask.sum().item()),
        }

    def broadcast_to_vllm(self):
        use_prefix_cache = getattr(self.strategy.args, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        model = self.actor.model.module
        count = 0

        def _broadcast_param(param, count, num_params):
            use_ray = getattr(self.strategy.args, "vllm_sync_with_ray", False)
            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(param.data, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
                ray.get(refs)

        def _handle_cuda_ipc(param, count, num_params):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        def _gather_params_ctx(param):
            """Context manager that gathers sharded/TP-split parameters for weight sync."""
            if self.strategy.args.ds_tensor_parallel_size > 1:
                return deepspeed.module_inject.layers.GatherReplacedLayerParams([param], model, enabled=True)
            return deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3)

        sync_fn = _handle_cuda_ipc if self.use_cuda_ipc else _broadcast_param

        # VLM: only sync trainable (language model) params — vision encoder is frozen.
        params_to_sync = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad or not getattr(self.actor, "is_vlm", False)
        ]
        num_params = len(params_to_sync)

        for name, param in params_to_sync:
            count += 1  # empty_cache at last param
            with _gather_params_ctx(param):
                sync_fn(param, count, num_params)

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        # Skip for vLLM >= 0.16 where NCCL_CUMEM_ENABLE=0 causes ncclCommInitRank to fail
        # with "unhandled cuda error" under NCCL 2.27+.
        if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
            freeze_visual_encoder=getattr(strategy.args, "freeze_visual_encoder", False),
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.attn_implementation,
                param_dtype=strategy.args.param_dtype,  # default: bf16
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states = states

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=self.vllm_engines,
        )

    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return status

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        mm_train_inputs_list=None,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()

        # VLM: merge pre-processed multimodal inputs from all samples in batch
        mm_inputs = {}
        if mm_train_inputs_list and getattr(self.actor, "is_vlm", False):
            mm_inputs = merge_mm_train_inputs(mm_train_inputs_list, device)

        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                **mm_inputs,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        self.trainer.broadcast_to_vllm()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        reload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)

    def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
        args = self.strategy.args
        client_states = client_states or {}
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
                metric_value=metric_value,
                metric_key=metric_key,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_cuda_sync()
