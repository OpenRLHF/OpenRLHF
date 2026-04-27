import math
import os
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.optimization import get_scheduler

from openrlhf.utils.distributed_sampler import DistributedSampler


def _get_actor_cls():
    """Lazy import to avoid circular dep: openrlhf.models.actor imports from this package."""
    from openrlhf.models import Actor

    return Actor


class FsdpStrategy:
    """FSDP2 + TP/CP/SP/EP backend, wrapping NVIDIA-NeMo/Automodel.

    Mirrors DeepspeedStrategy's public surface so trainers are agnostic to the
    backend. The model is built and parallelized via Automodel's official
    entry point ``NeMoAutoModelForCausalLM.from_pretrained`` inside ``Actor``;
    this strategy only handles distributed setup, optimizer/scheduler
    construction, the train-step (loss backward, grad clip, optimizer step),
    collectives, and checkpointing. Grad norm / clip are imported directly
    from ``nemo_automodel.components.distributed.grad_utils``.
    """

    CKPT_METRIC_FILENAME = "metric.json"

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ) -> None:
        self.args = args
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        fsdp = args.fsdp
        self.tp_size = getattr(fsdp, "tp_size", 1)
        self.cp_size = getattr(fsdp, "cp_size", 1)
        self.ep_size = getattr(fsdp, "ep_size", 1)
        self.pp_size = getattr(fsdp, "pp_size", 1)
        self.param_dtype = getattr(fsdp, "param_dtype", "bf16")
        # Activation checkpointing — train_sft/rm/dpo expose it as
        # --model.gradient_checkpointing_enable; train_ppo_ray uses
        # --actor.gradient_checkpointing_enable (PPO has separate actor/critic).
        # Read from whichever namespace is present.
        _model_ns = getattr(args, "model", None)
        _actor_ns = getattr(args, "actor", None)
        self.activation_checkpointing = (
            getattr(_model_ns, "gradient_checkpointing_enable", None)
            if _model_ns is not None and hasattr(_model_ns, "gradient_checkpointing_enable")
            else getattr(_actor_ns, "gradient_checkpointing_enable", False)
        )
        self.cpu_offload = getattr(fsdp, "cpu_offload", False)
        sp = getattr(fsdp, "sequence_parallel", None)
        # SP defaults to ON whenever TP>1 (user-stated default).
        self.sequence_parallel = sp if sp is not None else (self.tp_size > 1)
        self.optim = getattr(args, "optim", "adam")
        self.use_dynamic_batch = getattr(args.train, "dynamic_batch_enable", False)
        # LoRA flag — flows into Automodel Checkpointer's `is_peft`. Without it
        # the checkpointer saves the merged base only and the trained adapter
        # is lost. Read the rank from the FSDP namespace if present.
        _lora_ns = getattr(fsdp, "lora", None)
        self.is_peft = bool(getattr(_lora_ns, "rank", 0) > 0) if _lora_ns is not None else False

        self.world_size: int = 1
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_size = 1
        self.dp_cp_size = 1
        self.accumulated_gradient: int = 1
        self._last_grad_norm: float = 0.0
        self.time_steps = defaultdict(int)
        self._max_norm_by_optimizer = {}

    # ProcessGroup / DeviceMesh aren't picklable. `datasets.map(self.process_data,
    # num_proc>1)` indirectly pickles the strategy via the dataset's bound method;
    # drop the distributed handles so the workers spawn cleanly. Workers don't need
    # them — they only run pure-CPU data preprocessing.
    _UNPICKLABLE_ATTRS = ("device_mesh", "moe_mesh", "distributed_config")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._UNPICKLABLE_ATTRS:
            self.__dict__.setdefault(k, None)

    def _get_automodel_mesh(self, name: str, required: bool = False):
        if self.device_mesh is None:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        try:
            return get_flat_mesh(self.device_mesh, name)
        except KeyError:
            if required:
                raise
            return None

    def _get_automodel_group(self, name: str):
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
        return mesh.get_group() if mesh is not None else None

    def _get_dp_group(self, include_cp: bool = False):
        name = "dp_cp" if include_cp and self.cp_size > 1 else "dp"
        return self._get_automodel_group(name)

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        group = self._get_dp_group(include_cp=include_cp)
        if group is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        return dist.get_world_size(group=group)

    def _get_automodel_rank(self, name: str) -> int:
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
        return mesh.get_local_rank() if mesh is not None else 0

    def _get_dp_rank(self, include_cp: bool = False) -> int:
        if include_cp and self.cp_size > 1:
            return self._get_automodel_rank("dp_cp")
        return self._get_automodel_rank("dp")

    # ---------------------------------------------------------------- bring-up

    def setup_distributed(self, timeout: timedelta = timedelta(minutes=60)) -> None:
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
        else:
            transformers.set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timeout)

        self.world_size = dist.get_world_size()
        if self.pp_size > 1:
            raise NotImplementedError("OpenRLHF trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")
        if self.cp_size > 1 and getattr(self.args.fsdp, "packing_samples", False):
            raise ValueError(
                "Context parallel does not support --fsdp.packing_samples yet; disable packing for CP tests."
            )

        from nemo_automodel.components.distributed.config import FSDP2Config
        from nemo_automodel.components.distributed.mesh_utils import create_device_mesh
        from nemo_automodel.components.moe.config import MoEParallelizerConfig

        # Monkey-patch Automodel's TPLinear forward: their guard misses some
        # non-contiguous DTensor inputs and falls through to F.linear, which
        # then dies on aten.view (RuntimeError: view size is not compatible).
        # Always contiguify before F.linear; cheap when already contiguous.
        if self.tp_size > 1:
            from nemo_automodel.components.distributed import parallel_styles

            _TPLinear = parallel_styles.TPLinear
            if not getattr(_TPLinear, "_orlhf_patched", False):
                import torch.nn.functional as F
                from torch.distributed.tensor import DTensor

                def _safe_forward(self, x):
                    # Always avoid F.linear for DTensor inputs: F.linear's
                    # internal view fails on sharded DTensor regardless of
                    # is_contiguous(). Use mm/bmm directly which dispatch
                    # through DTensor's matmul strategy.
                    if isinstance(x, DTensor):
                        if x.dim() == 3:
                            b = x.shape[0]
                            out = torch.bmm(x, self.weight.t().unsqueeze(0).expand(b, -1, -1))
                        else:
                            out = torch.mm(x, self.weight.t())
                        return out + self.bias if self.bias is not None else out
                    return F.linear(x, self.weight, self.bias)

                _TPLinear.forward = _safe_forward
                _TPLinear._orlhf_patched = True

        torch_dtype = _torch_dtype(self.param_dtype)
        # FSDP2 mixed precision: params/forward in bf16, gradient reduce-scatter
        # in fp32. Reducing in bf16 accumulates rounding error across DP ranks
        # and inflates grad_norm (observed 60–130 vs. ~1 expected). cast_forward
        # _inputs handles the lm_head dtype mismatch by casting fp32 inputs to
        # bf16 at FSDP unit boundaries. Mirrors NeMo-Automodel's recipe.
        mp_policy = (
            None
            if self.param_dtype == "fp32"
            else MixedPrecisionPolicy(
                param_dtype=torch_dtype,
                reduce_dtype=torch.float32,
                # output_dtype omitted (defaults to param_dtype=bf16). Setting
                # it to fp32 makes the inner Qwen2Model FSDP root return fp32
                # hidden_states; lm_head is a sibling sub-module, not its own
                # FSDP unit, so it gets fp32 input × bf16 weight → matmul
                # mismatch ('expected mat1 and mat2 to have the same dtype').
                # Stable loss is achieved by casting logits to fp32 in
                # actor.forward right after the model call.
                cast_forward_inputs=True,
            )
        )
        # Public attribute — `Actor` reads it as `strategy.distributed_config`
        # and forwards to `NeMoAutoModelForCausalLM.from_pretrained`.
        self.distributed_config = FSDP2Config(
            sequence_parallel=self.sequence_parallel,
            activation_checkpointing=self.activation_checkpointing,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else None,
            # Force grad sync inside backward(): we read p.grad immediately after
            # backward() to compute the global grad_norm. With deferred sync,
            # each rank sees its un-reduced local grad and the global norm
            # comes out √dp_size× too high (observed: SFT step1 23 vs PR#1176
            # reference 13 with dp=3, ratio √3 ≈ 1.73).
            defer_fsdp_grad_sync=False,
        )
        # MoE-specific parallelization config — required by Automodel when
        # ep_size > 1 (raises 'NoneType has no to_dict' otherwise).
        self.moe_config = MoEParallelizerConfig(mp_policy=mp_policy) if self.ep_size > 1 else None

        self.device_mesh, self.moe_mesh = create_device_mesh(
            self.distributed_config,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            cp_size=self.cp_size,
            ep_size=self.ep_size,
            world_size=self.world_size,
        )

        # Process groups are resolved from Automodel's official mesh on demand.
        # OpenRLHF only caches scalar sizes derived from that mesh.
        # Automodel's FSDP2 mesh has *native* dims ("pp","dp_replicate","dp_shard","cp","tp")
        # plus *flattened* dims ("dp","dp_shard_cp","dp_cp") stored on
        # ``device_mesh._flatten_mapping``. ``mesh.mesh_dim_names`` lists only the
        # native dims, so all access goes through Automodel's ``get_flat_mesh``.

        # Automodel uses flattened "dp" for data loading/token denominators and
        # flattened "dp_cp" when CP ranks participate in FSDP/loss scaling.
        # Only sizes are cached; ProcessGroups are resolved from the Automodel
        # mesh on demand so there is a single source of truth.
        dp_size = self._get_dp_group_size(include_cp=False)
        self.dp_cp_size = self._get_dp_group_size(include_cp=True)
        # Effective grad-accum = train_batch_size / (micro_bs × DP).
        self.accumulated_gradient = max(self.train_batch_size // (self.micro_train_batch_size * dp_size), 1)
        self.dp_size = dp_size

    # ---------------------------------------------------------------- prepare

    def prepare(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, tuple):
                assert len(arg) == 2, f"prepare() tuple must be (model, cfg); got len={len(arg)}"
                model, cfg = arg
                if model is None:
                    ret.append((None, None, None))
                else:
                    ret.append(self._init_train_model(model, cfg))
            else:
                ret.append(self._init_eval_model(arg))
        return ret[0] if len(ret) == 1 else ret

    def _init_train_model(self, model, cfg: dict):
        # Model is already parallelized — Actor builds via
        # NeMoAutoModelForCausalLM.from_pretrained (Automodel's official entry),
        # which handles FSDP2 wrap + TP plan + CP hooks internally given the
        # device_mesh + distributed_config we expose on this strategy.
        train_model = self._unwrap_model(model)
        params = [p for p in train_model.parameters() if p.requires_grad]

        kind = cfg.get("optim", self.optim)
        if kind == "muon":
            raise NotImplementedError("Muon under FSDP2 not yet wired; use --optim adam")

        adam = cfg["adam"]
        optimizer = torch.optim.AdamW(
            params,
            lr=adam["lr"],
            betas=tuple(adam["betas"]),
            eps=adam["eps"],
            weight_decay=adam["weight_decay"],
            foreach=False,
            fused=False,
        )
        self._max_norm_by_optimizer[id(optimizer)] = cfg.get("max_norm", self.max_norm)

        scheduler_steps = cfg["scheduler_steps"]
        scheduler = get_scheduler(
            cfg.get("lr_scheduler", "cosine_with_min_lr"),
            optimizer,
            num_warmup_steps=math.ceil(scheduler_steps * cfg.get("lr_warmup_ratio", 0.03)),
            num_training_steps=scheduler_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.get("min_lr_ratio", 0.1)},
        )
        return model, optimizer, scheduler

    def _init_eval_model(self, model):
        # Eval models are also built+parallelized via Automodel's official entry
        # at construction time (Actor or get_llm_for_sequence_regression).
        return model

    # ---------------------------------------------------------------- step loop

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        # Static gradient accumulation mirrors the old trainer contract by
        # default: each microbatch loss is divided by the accumulation steps.
        # Callers that already use verl-style global batch denominators can set
        # ``scale_loss_by_accumulation=False`` and keep the no-sync behavior.
        if accumulate and self.accumulated_gradient > 1:
            if kwargs.get("scale_loss_by_accumulation", True):
                loss = loss / self.accumulated_gradient
            # Skip the DP all-reduce on intermediate accum steps (saves bandwidth);
            # only sync on the final micro-batch where optimizer_step will fire.
            # The key MUST match optimizer_step's so the counter advances together.
            unwrapped = self._unwrap_model(model)
            if hasattr(unwrapped, "set_requires_gradient_sync"):
                key = f"step_{name}"
                sync = (self.time_steps[key] + 1) % self.accumulated_gradient == 0
                unwrapped.set_requires_gradient_sync(sync)
        if self.moe_mesh is not None:
            try:
                from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                    float(getattr(self, "dp_cp_size", self.dp_size)),
                    device=loss.device,
                )
            except Exception:
                pass
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        # Gradient accumulation: skip the optimizer step until the last micro
        # batch in the accumulation window has run. Mirrors PR#1176 / NeMo-RL.
        key = f"step_{name}"
        if accumulate:
            self.time_steps[key] += 1
            if self.time_steps[key] % self.accumulated_gradient != 0:
                return

        model = self._unwrap_model(model)
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            self._last_grad_norm = 0.0
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            return
        max_norm = self._max_norm_by_optimizer.get(id(optimizer), self.max_norm)
        if max_norm and max_norm > 0:
            from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

            self._last_grad_norm = float(
                scale_grads_and_clip_grad_norm(
                    max_norm,
                    [model],
                    pp_enabled=False,
                    device_mesh=self.device_mesh,
                    moe_mesh=self.moe_mesh,
                    ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
                    foreach=False,
                    num_label_tokens=None,
                    dp_group_size=getattr(self, "dp_cp_size", self.dp_size),
                )
            )
        else:
            self._last_grad_norm = 0.0
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def get_grad_norm(self, model: nn.Module) -> float:
        return self._last_grad_norm

    def global_token_count(self, mask: torch.Tensor) -> torch.Tensor:
        """All-reduce ``mask.sum()`` across the data-parallel data mesh.

        For CP, call this before ``make_cp_batch_and_ctx`` while each CP rank
        still sees the full local sequence. This mirrors Automodel's recipe:
        token denominators are reduced over DP only; the backward loss scale
        still uses ``dp_size * cp_size`` to counter FSDP's dp_shard_cp averaging.
        """
        local = mask if mask.ndim == 0 else mask.sum()
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else local.device
        local = local.to(dtype=torch.float32, device=device)
        dp_group = self._get_dp_group(include_cp=False)
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_group)
        return local

    # ---------------------------------------------------------------- data

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = True,
        sampler=None,
        consumed_samples: int = 0,
        num_workers: int = 0,
    ):
        dp_group = self._get_dp_group(include_cp=False)
        if sampler is None and dist.is_initialized() and dp_group is not None:
            num_replicas = dist.get_world_size(group=dp_group)
            rank = dist.get_rank(group=dp_group)
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    # ---------------------------------------------------------------- comm

    def all_reduce(self, data, op: str = "mean"):
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=torch.cuda.current_device(), dtype=torch.float32)
        else:
            data = data.detach().clone().to(torch.cuda.current_device())
        reduce_op = {"mean": dist.ReduceOp.SUM, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
        dist.all_reduce(data, op=reduce_op)
        if op == "mean":
            data = data / dist.get_world_size()
        return data.item() if data.ndim == 0 else data

    def all_gather(self, data):
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=torch.cuda.current_device())
        gathered = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, data)
        return torch.cat(gathered, dim=0)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    # ---------------------------------------------------------------- compat shims

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, _get_actor_cls()):
            return self._unwrap_model(model.model)
        if hasattr(model, "get_base_model_for_fsdp"):
            return model.get_base_model_for_fsdp()
        if hasattr(model, "module"):
            return model.module
        return model

    def get_ds_train_config(self, *args, **kwargs):
        # Returned dict is consumed by Actor for HfDeepSpeedConfig under the DS
        # backend; under fsdp it's a no-op. Trainers pass it through unchanged.
        return None

    def get_ds_eval_config(self, *args, **kwargs):
        return None

    # ---------------------------------------------------------------- I/O (MVP)

    @staticmethod
    def _checkpoint_source(model: nn.Module) -> tuple[str | None, str | None]:
        config = getattr(model, "config", None)
        model_repo_id = (
            getattr(model, "name_or_path", None)
            or getattr(config, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
        )
        model_cache_dir = os.environ.get("HF_HUB_CACHE")
        if not model_cache_dir and os.environ.get("HF_HOME"):
            model_cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
        return model_cache_dir, model_repo_id

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        # Use Automodel's Checkpointer — its custom-model save_pretrained mixin
        # requires it (raises "No checkpointer provided" otherwise). Outputs
        # consolidated HF safetensors that vLLM can hot-load.
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        model = self._unwrap_model(model)
        model_cache_dir, model_repo_id = self._checkpoint_source(model)

        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir=model_cache_dir,
            model_repo_id=model_repo_id,
            save_consolidated=True,
            is_peft=self.is_peft,
            original_model_root_dir=model_cache_dir,
            v4_compatible=True,  # produce HF-style config.json that vLLM consumes
        )
        ckpt = Checkpointer(
            config=config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_automodel_rank("tp"),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )
        ckpt.save_model(model=model, weights_path=output_dir, tokenizer=tokenizer)
        if dist.is_initialized():
            dist.barrier()

    def _build_checkpointer(self, output_dir: str, save_consolidated: bool, model: nn.Module | None = None):
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        model_cache_dir, model_repo_id = self._checkpoint_source(model) if model is not None else (None, None)
        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir=model_cache_dir,
            model_repo_id=model_repo_id,
            save_consolidated=save_consolidated,
            is_peft=self.is_peft,
            original_model_root_dir=model_cache_dir,
            v4_compatible=True,
        )
        return Checkpointer(
            config=config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_automodel_rank("tp"),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )

    def save_ckpt(
        self,
        model: nn.Module,
        ckpt_path: str,
        tag: str,
        max_num: int = 3,
        max_mem: int = 0,
        client_states=None,
        **kwargs,
    ) -> None:
        """DCP-format checkpoint for resumable training (model + optimizer +
        scheduler + RL stats). HF-safetensors export goes through ``save_model``.
        """
        import json
        import shutil

        model = self._unwrap_model(model)

        save_dir = os.path.join(ckpt_path, tag)
        os.makedirs(save_dir, exist_ok=True)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        ckpt = self._build_checkpointer(save_dir, save_consolidated=False, model=model)
        ckpt.save_model(model=model, weights_path=save_dir, tokenizer=None)
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        if optimizer is not None:
            ckpt.save_optimizer(optimizer=optimizer, model=model, weights_path=save_dir, scheduler=scheduler)

        if is_rank0:
            extra = {"client_state": dict(client_states or {})}
            unwrapped = self._unwrap_model(model)
            cfg = getattr(unwrapped, "config", None)
            if cfg is not None and getattr(cfg, "normalize_reward", False):
                extra["runtime_state"] = {
                    "normalize_reward": True,
                    "mean": float(getattr(cfg, "mean", 0.0)),
                    "std": float(getattr(cfg, "std", 1.0)),
                }
            with open(os.path.join(save_dir, "extra_state.json"), "w") as f:
                json.dump(extra, f)
            with open(os.path.join(ckpt_path, "latest"), "w") as f:
                f.write(tag)
            if max_num and max_num > 0:
                tags = sorted(
                    (d for d in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, d))),
                    key=lambda d: os.path.getmtime(os.path.join(ckpt_path, d)),
                )
                for old in tags[:-max_num]:
                    shutil.rmtree(os.path.join(ckpt_path, old), ignore_errors=True)
        if dist.is_initialized():
            dist.barrier()

    def load_ckpt(self, model: nn.Module, ckpt_path: str, optimizer=None, scheduler=None, **kwargs):
        """Load the most recent DCP checkpoint under ``ckpt_path``. Returns
        ``(load_path, states)`` where ``states`` carries ``client_state`` keys
        (e.g. ``consumed_samples``); ``(None, {})`` if no checkpoint is found.
        """
        import json

        if not os.path.isdir(ckpt_path):
            return None, {}
        latest = os.path.join(ckpt_path, "latest")
        if os.path.isfile(latest):
            with open(latest) as f:
                tag = f.read().strip()
            load_dir = os.path.join(ckpt_path, tag)
        else:
            return None, {}
        if not os.path.isdir(load_dir):
            return None, {}

        wrapper = model
        model = self._unwrap_model(model)

        # Tied-weight load: Automodel saves in HF safetensors shards with
        # .hf_metadata sidecar (not DCP .metadata), and tie_word_embeddings=True
        # models (e.g. Qwen2.5-0.5B) deduplicate the tied pair. Use HF storage
        # reader + allow_partial_load to handle both. Mirrors PR#1176.
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import HuggingFaceStorageReader
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            set_model_state_dict,
        )

        model_state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=False))
        dcp.load(
            model_state,
            storage_reader=HuggingFaceStorageReader(path=os.path.join(load_dir, "model")),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        set_model_state_dict(model, model_state, options=StateDictOptions(strict=False))
        optim_dir = os.path.join(load_dir, "optim")
        if optimizer is not None and os.path.isdir(optim_dir):
            from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

            optimizer_state = OptimizerState(model, optimizer, scheduler, is_peft=self.is_peft)
            optim_state_dict = optimizer_state.state_dict()
            dcp.load(
                optim_state_dict,
                checkpoint_id=optim_dir,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            optimizer_state.load_state_dict(optim_state_dict)

        states = {}
        extra_path = os.path.join(load_dir, "extra_state.json")
        if os.path.isfile(extra_path):
            with open(extra_path) as f:
                extra = json.load(f)
            states = extra.get("client_state", {}) or {}
            runtime = extra.get("runtime_state") or {}
            unwrapped = self._unwrap_model(model)
            cfg = getattr(unwrapped, "config", None)
            if runtime and cfg is not None:
                for k in ("normalize_reward", "mean", "std"):
                    if k in runtime:
                        setattr(cfg, k, runtime[k])
                if hasattr(wrapper, "mean") and "mean" in runtime:
                    wrapper.mean[0] = runtime["mean"]
                if hasattr(wrapper, "std") and "std" in runtime:
                    wrapper.std[0] = runtime["std"]
                if hasattr(wrapper, "normalize_reward") and "normalize_reward" in runtime:
                    wrapper.normalize_reward = runtime["normalize_reward"]
        if dist.is_initialized():
            dist.barrier()
        return load_dir, states

    def load_model(self, model: nn.Module, model_path: str, **kwargs) -> None:
        """Load HF-safetensors weights into an already-parallelized model.
        Initial weight load is normally handled by ``NeMoAutoModelForCausalLM.
        from_pretrained``; use this only for explicit later reloads.
        """
        model = self._unwrap_model(model)
        ckpt = self._build_checkpointer(model_path, save_consolidated=False)
        ckpt.load_model(model=model, model_path=model_path, use_checkpoint_id=False)
        if dist.is_initialized():
            dist.barrier()

    @torch.no_grad()
    def moving_average(self, model: nn.Module, ema_model: nn.Module, beta: float, device: str = "cuda") -> None:
        """In-place EMA update: ``ema = beta * ema + (1 - beta) * model``.
        Iterates over (param, ema_param) pairs by name; both sides are FSDP2-
        sharded DTensors so the lerp is local to each rank's shard.
        """
        src = self._unwrap_model(model)
        tgt = self._unwrap_model(ema_model)
        ema_params = dict(tgt.named_parameters())
        for name, param in src.named_parameters():
            ema_param = ema_params.get(name)
            if ema_param is None:
                continue
            ema_param.data.mul_(beta).add_(param.data, alpha=1.0 - beta)


def _torch_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]
