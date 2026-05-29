import os

import ray

from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states

from .launcher import BaseModelActor


@ray.remote(num_gpus=1)
class PolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps=None, vllm_engines=None):
        args = strategy.args
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
            attn_implementation=strategy.args.ds.attn_implementation,
            experts_implementation=strategy.args.ds.experts_implementation,
            param_dtype=strategy.args.ds.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.ds.load_in_4bit,
            lora_rank=strategy.args.ds.lora.rank,
            lora_alpha=strategy.args.ds.lora.alpha,
            target_modules=strategy.args.ds.lora.target_modules,
            lora_dropout=strategy.args.ds.lora.dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            use_liger_kernel=strategy.args.ds.use_liger_kernel,
        )
        strategy.print(actor)

        if args.actor.gradient_checkpointing_enable:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )

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

        # load checkpoint
        ckpt_path = os.path.join(args.ckpt.path, "_actor")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.actor.model, ckpt_path)

        # initial offload
        if strategy.args.ds.enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def offload_states(self):
        offload_deepspeed_states(self.actor.model)
