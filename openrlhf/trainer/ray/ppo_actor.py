import gc
import os
import socket
from abc import ABC
from dataclasses import fields
from typing import Dict, List, Optional, Union

import ray
import torch
import torch.distributed
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, PolicyLoss, agg_loss
from openrlhf.models.utils import compute_approx_kl, masked_mean
from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.utils import get_tokenizer
from openrlhf.utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync
from openrlhf.utils.fsdp import FsdpStrategy
from openrlhf.utils.fsdp.refit import gather_full_param
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.vlm_utils import merge_mm_train_inputs

from ..ppo_utils import NaiveReplayBuffer

logger = init_logger(__name__)

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


def _maybe_adapt_tensor_to_hf(model: torch.nn.Module, name: str, tensor: torch.Tensor):
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        return [(name, tensor)]

    convert_one = getattr(adapter, "convert_single_tensor_to_hf", None)
    if convert_one is None:
        raise _vllm_refit_unsupported_error(model)
    return convert_one(
        name,
        tensor,
        exclude_key_regex=r".*_extra_state.*",
        quantization=False,
    )


def _vllm_refit_unsupported_error(model: torch.nn.Module) -> RuntimeError:
    return RuntimeError(
        f"{type(model).__name__} uses an AutoModel state_dict_adapter without "
        "`convert_single_tensor_to_hf`; vLLM refit cannot safely map this custom "
        "weight layout to HuggingFace/vLLM names. Use --fsdp.force_hf_model for RL runs."
    )


def _validate_vllm_refit_supported(model: torch.nn.Module) -> None:
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is not None and getattr(adapter, "convert_single_tensor_to_hf", None) is None:
        raise _vllm_refit_unsupported_error(model)


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
        self.max_epochs = self.args.train.max_epochs

        self.actor_loss_fn = PolicyLoss(
            clip_eps_low=self.args.actor.eps_clip_low_high[0],
            clip_eps_high=self.args.actor.eps_clip_low_high[1],
            dual_clip=self.args.actor.dual_clip,
            policy_loss_type=self.args.actor.policy_loss_type,
            enable_vllm_is_correction=self.args.algo.advantage.is_correction_enable,
            vllm_is_truncated_threshold=(
                self.args.algo.advantage.is_correction_threshold
                if self.args.algo.advantage.is_correction_enable
                else None
            ),
            vllm_is_correction_type=self.args.algo.advantage.is_correction_type,
        )

        # Mixtral 8x7b
        self.aux_loss = self.args.actor.aux_loss_coef > 1e-8

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            dynamic_batch=self.args.train.dynamic_batch_enable,
        )

        # Init torch group for weights sync
        backend = getattr(self.strategy.args.vllm, "sync_backend", "nccl")
        self.use_cuda_ipc = backend == "nccl" and self.args.train.colocate_all and not self.args.train.async_enable

        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            self._init_vllm_sync_group(backend)

        torch_dist_barrier_and_cuda_sync()

    def _init_vllm_sync_group(self, backend: str):
        """Create a torch process group between trainer rank 0 and all vLLM engine ranks.

        Layout example (3 engines, TP=4):
            [    0,      1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12]
            |train rank|  engine-0  |  engine-1  |   engine-2   |

        FSDP2/TP params are materialized before broadcasting to all engines.
        """
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        vllm_num_engines = self.strategy.args.vllm.num_engines
        vllm_tensor_parallel_size = self.strategy.args.vllm.tensor_parallel_size
        world_size = vllm_num_engines * vllm_tensor_parallel_size + 1

        use_ray = getattr(self.strategy.args.vllm, "sync_with_ray", False)
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

    def _record_status(self, status, status_list, pbar):
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

    def ppo_train(self, kl_ctl: float):
        # replay buffer may be empty at first, we should rebuild at each training
        if self.args.train.dynamic_batch_enable:
            self.replay_buffer.setup_dynamic_batch(self.strategy)

        should_shuffle = self.args.fsdp.tp_size <= 1 and not self.args.train.dynamic_batch_enable
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
            accum_window = []
            accum_steps = self.strategy.accumulated_gradient
            for step, experience in enumerate(pbar):
                experience.to_device(device)
                if self.args.train.dynamic_batch_enable:
                    status = self.training_step(experience, kl_ctl, step)
                    self._record_status(status, status_list, pbar)
                    continue

                accum_window.append((step, experience))
                if len(accum_window) < accum_steps:
                    continue

                local_num_tokens = torch.zeros((), dtype=torch.float32, device=torch.device("cuda", device))
                local_batch_size = torch.zeros((), dtype=torch.float32, device=torch.device("cuda", device))
                for _, window_experience in accum_window:
                    local_num_tokens += window_experience.action_mask.sum()
                    local_batch_size += (window_experience.action_mask.sum(dim=-1) > 0).sum()
                global_num_tokens = self.strategy.global_token_count(local_num_tokens)
                global_batch_size = self.strategy.global_token_count(local_batch_size)

                for window_step, window_experience in accum_window:
                    status = self.training_step(
                        window_experience,
                        kl_ctl,
                        window_step,
                        global_num_tokens=global_num_tokens,
                        global_batch_size=global_batch_size,
                        loss_dp_size=self.strategy.dp_size,
                        loss_report_scale=1,
                        scale_loss_by_accumulation=False,
                    )
                    self._record_status(status, status_list, pbar)
                accum_window = []

            # Trailing partial windows are dropped on purpose: feeding them
            # through training_step would advance time_steps but never fire
            # optimizer_step (since len < accum_steps), leaking the local
            # gradients into the next epoch's first window. The dataloader
            # already uses drop_last=True, so this only matters when the per-
            # rank length is not a multiple of accum_steps (rare; misconfig).
            if accum_window:
                self.strategy.print(
                    f"[PPO] dropping {len(accum_window)} trailing microbatches "
                    f"(< accum_steps={accum_steps}); check that train.batch_size, "
                    f"micro_batch_size, dp_size, and rollout n_samples align."
                )

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

    def training_step(
        self,
        experience: Experience,
        kl_ctl: float,
        step: int,
        global_num_tokens=None,
        global_batch_size=None,
        loss_dp_size=None,
        loss_report_scale: int = 1,
        scale_loss_by_accumulation: bool = True,
    ) -> Dict[str, float]:
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
            packed_seq_lens=packed_seq_lens,
            return_entropy=self.args.actor.entropy_coef is not None,
            **mm_inputs,
        )

        if global_num_tokens is None:
            global_num_tokens = self.strategy.global_token_count(experience.action_mask)
        if global_batch_size is None:
            global_batch_size = self.strategy.global_token_count((experience.action_mask.sum(dim=-1) > 0).sum())
        if loss_dp_size is None:
            loss_dp_size = self.strategy.dp_size

        # loss function
        actor_loss, clip_ratio, ppo_kl, vllm_kl = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
            rollout_log_probs=experience.rollout_log_probs,
            dp_size=loss_dp_size,
            batch_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
        )
        experience.info["ppo_clip_ratio"] = clip_ratio.detach()
        experience.info["ppo_kl"] = ppo_kl.detach()
        if vllm_kl is not None:
            experience.info["vllm_kl"] = vllm_kl.detach()

        if self.args.algo.kl.use_loss:
            if self.args.algo.kl.init_coef > 0:
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.algo.kl.estimator,
                )
                logprobs_diff = action_log_probs.float() - base_action_log_probs.float()
            else:
                kl = torch.zeros_like(action_log_probs)
                logprobs_diff = torch.zeros_like(action_log_probs)
            kl_loss = agg_loss(
                kl,
                experience.action_mask,
                "token-mean",
                dp_size=loss_dp_size,
                batch_num_tokens=global_num_tokens,
            )
            logprobs_diff = masked_mean(logprobs_diff, experience.action_mask)
            experience.info["kl"] = kl_loss.detach()
            experience.info["logprobs_diff"] = logprobs_diff.detach()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * kl_ctl
        # mixtral
        if self.aux_loss:
            aux_scale = 1 if scale_loss_by_accumulation else self.strategy.accumulated_gradient
            loss += getattr(output, "aux_loss", 0) * self.args.actor.aux_loss_coef / aux_scale
        # entropy loss
        if self.args.actor.entropy_coef is not None:
            entropy_loss = agg_loss(
                output.entropy[:, -experience.action_mask.shape[1] :],
                experience.action_mask,
                "token-mean",
                dp_size=loss_dp_size,
                batch_num_tokens=global_num_tokens,
            )
            if self.args.actor.entropy_coef != 0:
                loss -= entropy_loss * self.args.actor.entropy_coef

        if self.args.train.dynamic_batch_enable:
            loss = loss * self.replay_buffer.dynamic_loss_scale[step]

        if self.args.train.dynamic_batch_enable:
            self.strategy.backward(loss, self.actor, self.actor_optim, name="actor", accumulate=False)
            if self.replay_buffer.dynamic_optimizer_step[step]:
                self.strategy.optimizer_step(
                    self.actor_optim, self.actor, self.actor_scheduler, name="actor", accumulate=False
                )
        else:
            self.strategy.backward(
                loss,
                self.actor,
                self.actor_optim,
                name="actor",
                scale_loss_by_accumulation=scale_loss_by_accumulation,
            )
            self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            if self.args.train.dynamic_batch_enable:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")
            else:
                if (step + 1) % self.strategy.accumulated_gradient == 0:
                    self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cuda")

        metrics = {"policy_loss": actor_loss.detach() / loss_report_scale}
        weights = {"policy_loss": "token"}
        if self.args.actor.entropy_coef is not None:
            metrics["entropy_loss"] = entropy_loss.detach()
            weights["entropy_loss"] = "token"

        metrics["actor_lr"] = self.actor_scheduler.get_last_lr()[0]
        weights["actor_lr"] = None
        is_optimizer_step = (
            self.replay_buffer.dynamic_optimizer_step[step]
            if self.args.train.dynamic_batch_enable
            else (step + 1) % self.strategy.accumulated_gradient == 0
        )
        if is_optimizer_step:
            metrics["actor_grad_norm"] = self.strategy.get_grad_norm(self.actor)
            weights["actor_grad_norm"] = None

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
        if getattr(self.strategy.args.fsdp.lora, "rank", 0) > 0:
            raise NotImplementedError("FSDP2/AutoModel vLLM weight refit does not support LoRA adapters yet.")

        use_prefix_cache = getattr(self.strategy.args.vllm, "enable_prefix_caching", False)
        cache_reset_refs = []
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            for engine in self.vllm_engines:
                cache_reset_refs.append(engine.reset_prefix_cache.remote())

        torch.cuda.empty_cache()
        # FSDP2/AutoModel: `actor.model` is the FSDP2-wrapped HF model directly
        # (no DS-engine `.module` indirection); params are DTensors when sharded.
        model = self.actor.model

        def _broadcast_param(name, weight, dtype, shape):
            use_ray = getattr(self.strategy.args.vllm, "sync_with_ray", False)
            if torch.distributed.get_rank() == 0:
                refs = [
                    engine.update_weight.remote(name, dtype=dtype, shape=shape, empty_cache=False)
                    for engine in self.vllm_engines
                ]
                if use_ray:
                    import ray.util.collective as collective

                    collective.broadcast(weight, 0, group_name=self._model_update_group)
                else:
                    self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())
                ray.get(refs)

        def _handle_cuda_ipc(name, weight, dtype, shape):
            from torch.multiprocessing.reductions import reduce_tensor

            ipc_handle = reduce_tensor(weight.clone())
            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=False,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        sync_fn = _handle_cuda_ipc if self.use_cuda_ipc else _broadcast_param

        from openrlhf.utils.utils import convert_to_torch_dtype

        sync_dtype = convert_to_torch_dtype(self.strategy.args.fsdp.param_dtype)
        param_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}

        for name, tensor in model.state_dict().items():
            # Keep the previous VLM behavior: frozen visual params are already
            # present in vLLM from the base checkpoint, so only language params
            # that can change during training need to be refit.
            if name not in param_requires_grad:
                continue
            if getattr(self.actor, "is_vlm", False) and not param_requires_grad[name]:
                continue

            weight, _ = gather_full_param(tensor)
            for hf_name, hf_weight in _maybe_adapt_tensor_to_hf(model, name, weight):
                if not torch.is_tensor(hf_weight):
                    continue
                if not hf_weight.is_floating_point():
                    continue
                hf_weight = hf_weight.to(
                    device=torch.device("cuda", torch.cuda.current_device()),
                    dtype=sync_dtype,
                    non_blocking=True,
                ).contiguous()
                sync_fn(hf_name, hf_weight, sync_dtype, hf_weight.shape)
                del hf_weight
            del weight

        if cache_reset_refs:
            ray.get(cache_reset_refs)
        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: FsdpStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
        self.save_hf_ckpt = args.ckpt.save_hf
        self.disable_ds_ckpt = args.ckpt.disable_ds
        self.vllm_engines = vllm_engines
        self.max_steps = max_steps

        # Skip for vLLM >= 0.16 where NCCL_CUMEM_ENABLE=0 causes ncclCommInitRank to fail
        # with "unhandled cuda error" under NCCL 2.27+.
        if getattr(args.vllm, "sync_backend", "nccl") == "nccl":
            import vllm
            from packaging import version as pkg_version

            if pkg_version.parse(vllm.__version__) < pkg_version.parse("0.16"):
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            attn_implementation=strategy.args.fsdp.attn_implementation,
            param_dtype=strategy.args.fsdp.param_dtype,
            lora_rank=strategy.args.fsdp.lora.rank,
            lora_alpha=strategy.args.fsdp.lora.alpha,
            target_modules=strategy.args.fsdp.lora.target_modules,
            lora_dropout=strategy.args.fsdp.lora.dropout,
            device_mesh=strategy.device_mesh,
            moe_mesh=strategy.moe_mesh,
            distributed_config=strategy.distributed_config,
            moe_config=strategy.moe_config,
            activation_checkpointing=args.actor.gradient_checkpointing_enable,
            packing_samples=strategy.args.fsdp.packing_samples,
            force_hf_model=strategy.args.fsdp.force_hf_model,
            temperature=strategy.args.rollout.temperature,
            use_liger_kernel=strategy.args.fsdp.use_liger_kernel,
            freeze_visual_encoder=getattr(strategy.args.actor, "freeze_visual_encoder", False),
        )
        if vllm_engines is not None:
            _validate_vllm_refit_supported(actor.model)
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
        )

        if args.train.enable_ema:
            # EMA must mirror the actor's parameter graph; under LoRA the actor
            # has frozen base weights + trainable lora_A/lora_B, while a
            # LoRA-less EMA has only the (constant) base — so EMA would
            # silently freeze and miss every adapter update. Forbid the
            # combination explicitly until we wire LoRA-aware EMA.
            if strategy.args.fsdp.lora.rank > 0:
                raise NotImplementedError(
                    "EMA (--train.enable_ema) with LoRA (--fsdp.lora.rank>0) is not supported: "
                    "the EMA model would only track frozen base weights and miss adapter updates."
                )
            ema_model = Actor(
                pretrain,
                attn_implementation=strategy.args.fsdp.attn_implementation,
                param_dtype=strategy.args.fsdp.param_dtype,
                device_mesh=strategy.device_mesh,
                moe_mesh=strategy.moe_mesh,
                distributed_config=strategy.distributed_config,
                moe_config=strategy.moe_config,
                activation_checkpointing=False,
                packing_samples=strategy.args.fsdp.packing_samples,
                force_hf_model=strategy.args.fsdp.force_hf_model,
                use_liger_kernel=strategy.args.fsdp.use_liger_kernel,
            )
        else:
            ema_model = None

        actor_cfg = dict(
            optim=args.actor.optim,
            muon=vars(args.actor.muon),
            adam=vars(args.actor.adam),
            lr_scheduler=args.actor.lr_scheduler,
            lr_warmup_ratio=args.actor.lr_warmup_ratio,
            min_lr_ratio=args.actor.min_lr_ratio,
            max_norm=args.actor.max_norm,
            scheduler_steps=max_steps,
        )
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare((actor, actor_cfg))

        if ema_model:
            self.ema_model = strategy.prepare(ema_model)
        else:
            self.ema_model = None

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt.path, "_actor")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(
                self.actor.model, ckpt_path, optimizer=self.actor_optim, scheduler=self.actor_scheduler
            )
            self.checkpoint_states = states

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.train.micro_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.actor.eps_clip,
            ema_beta=args.train.ema_beta,
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
            self.ema_model if args.train.enable_ema else self.actor,
            self.tokenizer,
            args.ckpt.output_dir,
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
                **mm_inputs,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    def broadcast_to_vllm(self):
        # Refit only needs *model* params on GPU (gather full_tensor + IPC).
        # The optimizer was already moved off-GPU at the end of fit() via
        # offload_before_refit, so this is just model→cuda → broadcast →
        # model→cpu so vLLM can wake_up its KV+weights.
        if self._sleep_enabled:
            self.strategy.move_model_to_device(self.actor, "cuda")
            if self.ema_model is not None:
                self.strategy.move_model_to_device(self.ema_model, "cuda")
        self.trainer.broadcast_to_vllm()
        if self._sleep_enabled:
            self.offload_after_refit()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    def reload_states(self):
        """Sleep entry for fit(): full reload of model + optimizer to GPU.
        Mirrors NeMo-RL's prepare_for_training. No-op when sleep is off."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.actor, "cuda")
        self.strategy.move_optimizer_to_device(self.actor_optim, "cuda")
        if self.ema_model is not None:
            self.strategy.move_model_to_device(self.ema_model, "cuda")
        self.actor.train()

    def offload_states(self):
        """Sleep exit symmetric to reload_states. Used by code paths that
        don't need the staged refit dance — i.e. critic-style end-of-fit."""
        if not self._sleep_enabled:
            return
        self.offload_before_refit()
        self.offload_after_refit()

    def offload_before_refit(self):
        """End-of-fit hook: optimizer→cpu but keep params on GPU so the
        immediately following broadcast_to_vllm can gather full_tensor without
        an extra cpu→cuda round-trip. NeMo-RL pattern."""
        if not self._sleep_enabled:
            return
        self.strategy.move_optimizer_to_device(self.actor_optim, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_after_refit(self):
        """End-of-broadcast hook: model→cpu. Optimizer is assumed already
        cpu from offload_before_refit."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.actor, "cpu")
        if self.ema_model is not None:
            self.strategy.move_model_to_device(self.ema_model, "cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_for_lp_inference(self):
        """Used when actor needs to compute logprobs but not train: bring
        model back to GPU, but keep optimizer on CPU. Pair with
        offload_after_refit to symmetrically tear down."""
        if not self._sleep_enabled:
            return
        self.strategy.move_model_to_device(self.actor, "cuda")
        self.actor.eval()

    def save_checkpoint(self, tag, client_states=None, metric_value=None, metric_key=None):
        args = self.strategy.args
        client_states = client_states or {}
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt.path, "_actor"),
                tag,
                args.ckpt.max_num,
                args.ckpt.max_mem,
                client_states,
                metric_value=metric_value,
                metric_key=metric_key,
                optimizer=self.actor_optim,
                scheduler=self.actor_scheduler,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt.path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.train.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        torch_dist_barrier_and_cuda_sync()
