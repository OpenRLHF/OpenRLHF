import os

import ray

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states

from .launcher import BaseModelActor


@ray.remote(num_gpus=1)
class CriticModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        args = strategy.args

        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.reward.normalize_enable,
            attn_implementation=strategy.args.ds.attn_implementation,
            experts_implementation=strategy.args.ds.experts_implementation,
            param_dtype=strategy.args.ds.param_dtype,  # default: bf16
            load_in_4bit=strategy.args.ds.load_in_4bit,
            lora_rank=strategy.args.ds.lora.rank,
            lora_alpha=strategy.args.ds.lora.alpha,
            target_modules=strategy.args.ds.lora.target_modules,
            lora_dropout=strategy.args.ds.lora.dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.ds.value_head_prefix,
            init_value_head=strategy.args.actor.model_name_or_path == strategy.args.critic.model_name_or_path,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.reward.normalize_enable))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        if args.actor.gradient_checkpointing_enable:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.actor.gradient_checkpointing_reentrant}
            )

        # Critic reads its own args.critic.* sub-namespace.  Typical setup: actor may
        # use Muon but --critic.optim stays adam because value heads are essentially 1D.
        critic_cfg = dict(
            optim=args.critic.optim,
            muon=vars(args.critic.muon),
            adam=vars(args.critic.adam),
            lr_scheduler=args.critic.lr_scheduler,
            lr_warmup_ratio=args.critic.lr_warmup_ratio,
            min_lr_ratio=args.critic.min_lr_ratio,
            max_norm=args.critic.max_norm,
            scheduler_steps=max_steps,
        )
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare((critic, critic_cfg))

        # load checkpoint
        ckpt_path = os.path.join(args.ckpt.path, "_critic")
        if args.ckpt.load_enable and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        if strategy.args.ds.enable_sleep:
            self.offload_states()

    def offload_states(self):
        offload_deepspeed_states(self.critic)
