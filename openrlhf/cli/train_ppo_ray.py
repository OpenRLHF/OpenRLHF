import argparse
import os
from datetime import datetime

import ray
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import create_vllm_engines
from openrlhf.trainer.ray.launcher import (
    RayActorGroup,
    ReferenceModelActor,
    RewardModelActor,
)
from openrlhf.trainer.ray.ppo_actor import PolicyModelActor
from openrlhf.trainer.ray.ppo_critic import CriticModelActor
from openrlhf.utils import get_strategy


def train(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        # Use os.environ.get() to respect user-set values (e.g. NCCL_DEBUG=INFO via
        # ray job submit --runtime-env-json), falling back to sensible defaults.
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
                    "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                    "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "1"),
                }
            }
        )

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # init vllm / actor /critic /ref /reward model
    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.train.colocate_actor_ref or args.train.colocate_all:
        if args.algo.kl.init_coef > 0:
            assert (
                args.actor.num_nodes == args.ref.num_nodes
                and args.actor.num_gpus_per_node == args.ref.num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor.num_nodes * args.actor.num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
        max_len = args.data.max_len
        if args.train.colocate_all and not args.train.async_enable:
            assert (
                args.actor.num_nodes * args.actor.num_gpus_per_node
                == args.vllm.num_engines * args.vllm.tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor.num_nodes * args.actor.num_gpus_per_node} "
                f"and {args.vllm.num_engines * args.vllm.tensor_parallel_size}"
            )

        vllm_engines = create_vllm_engines(
            args.vllm.num_engines,
            args.vllm.tensor_parallel_size,
            args.actor.model_name_or_path,
            args.train.seed,
            args.train.full_determinism_enable,
            args.vllm.enable_prefix_caching,
            args.vllm.enforce_eager,
            max_len,
            pg if args.train.colocate_all and not args.train.async_enable else None,
            args.vllm.gpu_memory_utilization,
            args.vllm.enable_sleep,
            "processed_logprobs" if args.algo.advantage.is_correction_enable else None,
            agent_func_path=args.train.agent_func_path,
            remote_rm_url=args.reward.remote_url,
            max_images_per_prompt=getattr(args.data, "max_images_per_prompt", 0),
        )

    actor_model = RayActorGroup(
        args.actor.num_nodes,
        args.actor.num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
    )

    if args.algo.kl.init_coef > 0:
        ref_model = RayActorGroup(
            args.ref.num_nodes,
            args.ref.num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
        )
    else:
        ref_model = None

    if not args.train.colocate_all:
        pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic.model_name_or_path and args.train.colocate_critic_reward:
        assert (
            args.critic.num_nodes == args.reward.num_nodes
            and args.critic.num_gpus_per_node == args.reward.num_gpus_per_node
        ), "num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic.num_nodes * args.critic.num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic.model_name_or_path:
        critic_model = RayActorGroup(
            args.critic.num_nodes,
            args.critic.num_gpus_per_node,
            CriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
        )
    else:
        critic_model = None

    # multiple reward models
    if not args.reward.remote_url:
        reward_model = RayActorGroup(
            args.reward.num_nodes,
            args.reward.num_gpus_per_node,
            RewardModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=args.ds.ring_attn_size * args.ds.tensor_parallel_size,
        )
    else:
        reward_model = None

    # Select trainer by mode
    if args.train.async_enable:
        from openrlhf.trainer.ppo_trainer_async import PPOTrainerAsync as PPOTrainer
    else:
        from openrlhf.trainer.ppo_trainer import PPOTrainer

    # init PPO trainer (Single controller)
    ppo_trainer = PPOTrainer.remote(
        args.actor.model_name_or_path,
        strategy,
        actor_model,
        critic_model,
        reward_model,
        ref_model,
        vllm_engines,
        # generate kwargs
        do_sample=True,
        max_len=max_len,
        max_new_tokens=args.rollout.max_new_tokens,
        temperature=args.rollout.temperature,
        top_p=args.rollout.top_p,
    )

    # training update steps
    max_steps = ray.get(ppo_trainer.get_max_steps.remote())

    # init actor/reference/reward model
    refs = []
    refs.extend(
        actor_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path, max_steps, vllm_engines)
    )
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path))
    if reward_model is not None and args.reward.model_name_or_path:
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward.model_name_or_path))
    ray.get(refs)

    if critic_model is not None and args.critic.model_name_or_path:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        refs = critic_model.async_init_model_from_pretrained(strategy, args.critic.model_name_or_path, max_steps)
        ray.get(refs)

    # train actor and critic model
    ray.get(ppo_trainer.fit.remote())

    # save model
    ray.get(actor_model.async_save_model())

    if args.critic.model_name_or_path and args.critic.save_value_network and critic_model is not None:
        ray.get(critic_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Ray and vLLM
    parser.add_argument("--ref.num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref.num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward.num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward.num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--train.colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor.num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor.num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic.num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic.num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--train.colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--train.colocate_all",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # vLLM for text generation
    parser.add_argument(
        "--vllm.num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm.tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm.sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm.sync_with_ray", action="store_true", default=False)
    parser.add_argument("--vllm.enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--vllm.enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm.enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm.gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
    parser.add_argument("--algo.advantage.is_correction_enable", action="store_true", default=False)
    parser.add_argument(
        "--algo.advantage.is_correction_threshold",
        type=float,
        nargs=2,
        default=[0.5, 5.0],
        help="Low and high thresholds for vllm importance sampling truncation",
    )
    parser.add_argument(
        "--algo.advantage.is_correction_type",
        type=str,
        default="tis",
        choices=["tis", "icepop", "seq-mask-tis"],
        help="vLLM IS correction type: tis (token-level clamp), icepop (token-level filter), seq-mask-tis (sequence-level geom mean)",
    )

    # Async training using ray
    parser.add_argument("--train.async_enable", action="store_true", default=False, help="Enable async training")
    parser.add_argument("--train.async_queue_size", type=int, default=1, help="Queue size for async sampler<->trainer")
    parser.add_argument(
        "--train.partial_rollout_enable",
        action="store_true",
        default=False,
        help="Enable partial rollout in async mode. Uses vLLM pause/resume for weight sync "
        "instead of locking, allowing generation to overlap with training. "
        "In-flight samples may contain tokens from both old and new weights.",
    )

    # Checkpoints
    parser.add_argument("--eval.steps", type=int, default=-1)
    parser.add_argument("--ckpt.save_steps", type=int, default=-1)
    parser.add_argument("--logger.logging_steps", type=int, default=1)
    parser.add_argument("--ckpt.path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--ckpt.save_hf", action="store_true", default=False)
    parser.add_argument("--ckpt.disable_ds", action="store_true", default=False)
    parser.add_argument("--ckpt.max_num", type=int, default=3)
    parser.add_argument("--ckpt.max_mem", type=float, default=float("inf"))
    parser.add_argument("--ckpt.load_enable", action="store_true", default=False)
    parser.add_argument(
        "--ckpt.best_metric_key",
        type=str,
        default="",
        help="Eval metric key for best checkpoint saving (e.g., eval_default_pass1). "
        "Empty string auto-detects first pass1 metric. Set to 'none' to disable best checkpoint saving.",
    )
    parser.add_argument(
        "--ds.use_universal_ckpt", action="store_true", help="Use deepspeed universal checkpoint", default=False
    )

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--ds.zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--actor.gradient_checkpointing_enable", action="store_true", default=False)
    parser.add_argument("--ds.deepcompile", action="store_true", default=False)
    parser.add_argument(
        "--ds.param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type",
    )
    ## Make EMA as an optional feature
    parser.add_argument("--train.enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--train.ema_beta", type=float, default=0.992, help="EMA beta coefficient")
    parser.add_argument("--ds.zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--ds.adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument(
        "--ds.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument(
        "--ds.experts_implementation",
        type=str,
        default=None,
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
        help="MoE expert computation strategy passed to transformers from_pretrained (default: auto — transformers picks grouped_mm when supported, else eager)",
    )
    parser.add_argument("--ds.use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--ds.grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--ds.overlap_comm", action="store_true", default=False)
    parser.add_argument("--actor.gradient_checkpointing_reentrant", action="store_true", default=False)
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--data.dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers for IO (for Ray training, ensure sufficient CPU resources per actor)",
    )
    parser.add_argument(
        "--ds.enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using --colocate_all_models",
    )
    parser.add_argument("--ds.tensor_parallel_size", type=int, default=1, help="DeepSpeed tensor parallel size")

    # packing samples using Flash Attention2
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)

    # dynamic batch size
    parser.add_argument("--train.dynamic_batch_enable", action="store_true", default=False)
    parser.add_argument("--rollout.max_tokens_per_gpu", type=int, default=None)
    parser.add_argument("--train.max_tokens_per_gpu", type=int, default=16192)

    # LoRA
    parser.add_argument("--ds.load_in_4bit", action="store_true", default=False)
    parser.add_argument("--ds.lora.rank", type=int, default=0)
    parser.add_argument("--ds.lora.alpha", type=int, default=16)
    parser.add_argument("--ds.lora.target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--ds.lora.dropout", type=float, default=0)

    # PPO
    parser.add_argument("--ckpt.output_dir", type=str, default="./ckpt")
    parser.add_argument("--train.num_episodes", type=int, default=1)
    parser.add_argument("--rollout.batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument(
        "--rollout.vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--rollout.micro_batch_size", type=int, default=1)
    parser.add_argument("--train.max_epochs", type=int, default=1)
    parser.add_argument("--data.max_len", type=int, default=2048, help="Max total sequence length (prompt + response)")
    parser.add_argument(
        "--rollout.max_new_tokens",
        type=int,
        default=None,
        help="Max tokens to generate per sample. If None, dynamically computed as max_len - prompt_len per sample.",
    )
    parser.add_argument("--data.max_samples", type=int, default=int(1e8), help="Max number of samples")
    parser.add_argument("--actor.eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--actor.eps_clip_low_high", type=float, nargs=2, default=None, help="PPO-clip low and high")
    parser.add_argument("--actor.dual_clip", type=float, default=None, help="Dual-clip PPO")
    parser.add_argument("--critic.value_clip", type=float, default=0.5, help="PPO value clip range")
    parser.add_argument("--algo.advantage.lambd", type=float, default=1, help="PPO GAE lambd")
    parser.add_argument("--algo.advantage.gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--train.micro_batch_size", type=int, default=1, help="batch size per GPU")
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument(
        "--reward.normalize_enable", action="store_true", default=False, help="Enable Reward Normalization"
    )
    parser.add_argument("--rollout.top_p", type=float, default=1.0)
    parser.add_argument("--rollout.temperature", type=float, default=1.0)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument(
        "--train.full_determinism_enable",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument(
        "--critic.freezing_steps",
        type=int,
        default=-1,
        help="Freeze the actor for the first N steps to let the critic warm up",
    )
    parser.add_argument(
        "--rollout.n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--critic.save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--algo.kl.target", type=float, default=None)
    parser.add_argument("--algo.kl.horizon", type=int, default=10000)
    parser.add_argument("--algo.kl.init_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--actor.policy_loss_type", type=str, default="ppo", choices=["ppo", "gspo"])
    parser.add_argument(
        "--algo.kl.estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument("--actor.aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--actor.entropy_coef",
        type=float,
        default=None,
        help="Entropy loss coef, set to 0 means only enable entropy logs",
    )
    parser.add_argument("--reward.clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Optimizer + scheduler + grad clip, per entity (actor / critic).  Two sections:
    #   --{prefix}.muon.*  Muon-specific hypers (only used when --{prefix}.optim=muon)
    #   --{prefix}.adam.*  AdamW hypers — drives pure AdamW when --{prefix}.optim=adam,
    #                      and Muon's aux-Adam subgroup when --{prefix}.optim=muon.
    # Note: DS v0.18.x Muon hard-codes ns_steps=5 / nesterov=True inside
    # muon_update() and ignores config overrides. We still expose the CLI slots
    # so training scripts can be ready for future DS versions; the runtime emits
    # a warning when a non-default value is set on an ignoring DS.
    for prefix in ("actor", "critic"):
        parser.add_argument(f"--{prefix}.optim", type=str, default="adam", choices=["adam", "muon"])
        # Muon-specific
        parser.add_argument(
            f"--{prefix}.muon.lr",
            type=float,
            default=0.02,
            help=f"LR for {prefix}'s Muon 2D-weight group",
        )
        parser.add_argument(f"--{prefix}.muon.momentum", type=float, default=0.95)
        # Placeholder slots: DS v0.18.x ignores these (see note above); retained
        # so upgrade to a future DS version is a zero-diff change.
        parser.add_argument(
            f"--{prefix}.muon.ns_steps", type=int, default=5, help="Newton-Schulz steps (placeholder, DS-ignored)"
        )
        parser.add_argument(f"--{prefix}.muon.nesterov", action="store_true", default=True)
        parser.add_argument(f"--{prefix}.muon.no_nesterov", dest=f"{prefix}.muon.nesterov", action="store_false")
        # AdamW (shared: pure-AdamW when --{prefix}.optim=adam, Muon's aux-Adam subgroup when =muon)
        parser.add_argument(f"--{prefix}.adam.lr", type=float, default=1e-6 if prefix == "actor" else 9e-6)
        parser.add_argument(f"--{prefix}.adam.betas", type=float, nargs=2, default=(0.9, 0.95))
        parser.add_argument(f"--{prefix}.adam.eps", type=float, default=1e-8)
        parser.add_argument(f"--{prefix}.adam.weight_decay", type=float, default=0.0)
        # Scheduler
        parser.add_argument(f"--{prefix}.lr_scheduler", type=str, default="cosine_with_min_lr")
        parser.add_argument(f"--{prefix}.lr_warmup_ratio", type=float, default=0.03)
        parser.add_argument(f"--{prefix}.min_lr_ratio", type=float, default=0.1)
        # Gradient clip
        parser.add_argument(f"--{prefix}.max_norm", type=float, default=1.0)

    # Reinforce/GRPO, etc
    parser.add_argument(
        "--algo.advantage.estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "reinforce_baseline", "group_norm", "dr_grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo",
    )
    parser.add_argument(
        "--algo.kl.use_loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
    )
    parser.add_argument(
        "--algo.advantage.no_std_norm",
        action="store_true",
        default=False,
        help="disable dividing by std for advantages while keeping mean normalization",
    )
    parser.add_argument(
        "--reward.overlong_buffer_len", type=float, default=None, help="reward with optional overlong penalty"
    )
    parser.add_argument("--reward.overlong_penalty_factor", type=float, default=1, help="overlong penalty factor")
    parser.add_argument(
        "--reward.stop_properly_penalty_coef",
        type=float,
        default=None,
        help="Penalty for truncated samples (finish_reason='length'). "
        "If >= 0: multiplicative scaling [0,1]. If < 0: fixed reward override (e.g., -0.5).",
    )

    # Context Parallel
    parser.add_argument("--ds.ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ds.ring_attn_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    #  Models
    parser.add_argument("--actor.model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward.model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward.remote_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic.model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument("--ds.value_head_prefix", type=str, default="score")
    parser.add_argument("--ref.offload", action="store_true", default=False, help="Offload reference model to CPU")
    parser.add_argument("--reward.offload", action="store_true", default=False, help="Offload reward model to CPU")
    parser.add_argument("--train.agent_func_path", type=str, default=None, help="Agent script path")

    # Custom dataset
    parser.add_argument("--data.prompt_dataset", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--data.prompt_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--data.prompt_split", type=str, default="train")
    parser.add_argument("--eval.dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument("--eval.temperature", type=float, default=0.6, help="Temperature for evaluation")
    parser.add_argument(
        "--eval.n_samples_per_prompt", type=int, default=4, help="Number of samples per prompt for evaluation"
    )

    parser.add_argument("--data.input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--data.label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--data.input_template", type=str, default=None)
    parser.add_argument(
        "--data.apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # Dynamic filtering
    parser.add_argument(
        "--algo.dynamic_filtering_enable", action="store_true", default=False, help="Enable dynamic filtering"
    )
    parser.add_argument(
        "--algo.dynamic_filtering_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )

    # VLM (Vision-Language Model) parameters
    parser.add_argument("--data.image_key", type=str, default="images", help="Dataset key for image paths/URLs")
    parser.add_argument(
        "--data.max_images_per_prompt", type=int, default=0, help="Max images per prompt for vLLM (0 = text-only)"
    )
    parser.add_argument(
        "--actor.freeze_visual_encoder",
        action="store_true",
        default=False,
        help="Freeze vision encoder weights (only train language model). Reduces memory and weight sync overhead.",
    )

    # TensorBoard parameters
    parser.add_argument("--logger.tensorboard_dir", type=str, default=None, help="TensorBoard logging path")

    # performance tuning

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()
    from openrlhf.utils.config import hierarchize

    args = hierarchize(args)

    # Validate arguments
    if args.actor.eps_clip_low_high is None:
        args.actor.eps_clip_low_high = (args.actor.eps_clip, args.actor.eps_clip)

    if args.train.agent_func_path:
        args.reward.remote_url = "agent"

    if args.algo.advantage.estimator not in ["gae"]:
        args.critic.model_name_or_path = None
    elif args.critic.model_name_or_path is None:
        if not args.reward.remote_url:
            args.critic.model_name_or_path = args.reward.model_name_or_path.split(",")[0]
        else:
            args.critic.model_name_or_path = args.actor.model_name_or_path

    if args.algo.advantage.estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert (
            args.rollout.n_samples_per_prompt > 1
        ), f"{args.algo.advantage.estimator} requires n_samples_per_prompt > 1"

    # VLM constraints: critic and packing_samples are not supported
    if args.data.max_images_per_prompt > 0:
        assert args.critic.model_name_or_path is None, (
            "VLM training does not support critic model. "
            "Use --advantage_estimator other than 'gae' (e.g., reinforce_baseline, rloo, group_norm)."
        )
        assert not args.ds.packing_samples, (
            "VLM training does not support --packing_samples. "
            "Packing collapses the batch dimension, breaking alignment between image tokens and pixel_values. "
            "VLM models also require model-computed position_ids (e.g., M-RoPE) which is incompatible with packing."
        )

    if args.reward.remote_url:
        args.reward.remote_url = args.reward.remote_url.split(",")

    if args.data.input_template and "{}" not in args.data.input_template:
        print("[Warning] '{}' not in args.data.input_template, set to None")
        args.data.input_template = None

    if args.data.input_template and "\\n" in args.data.input_template:
        print(
            "[Warning] input_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.ds.ring_attn_size > 1:
        if not args.ds.packing_samples:
            print("[Warning] --ring_attn_size > 1 requires --packing_samples.")
            args.ds.packing_samples = True

    if args.train.dynamic_batch_enable:
        if not args.ds.packing_samples:
            print("[Warning] Please --packing_samples to accelerate when --use_dynamic_batch is enabled.")
            args.ds.packing_samples = True
        if args.rollout.max_tokens_per_gpu is None:
            print("[Warning] Set --rollout_max_tokens_per_gpu to --train_max_tokens_per_gpu.")
            args.rollout.max_tokens_per_gpu = args.train.max_tokens_per_gpu

    if args.ds.packing_samples:
        if "flash_attention" not in args.ds.attn_implementation:
            print(
                "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
            )
            args.ds.attn_implementation = "flash_attention_2"
        assert args.vllm.num_engines > 0, "Only support `--packing_samples` with vLLM."

    if args.vllm.enable_sleep and not args.train.colocate_all:
        print("Set args.vllm.enable_sleep to False when args.train.colocate_all is disabled.")
        args.vllm.enable_sleep = False

    if args.train.colocate_all and args.train.async_enable:
        print("[Warning] Using --colocate_all_models in async RLHF only colocates DeepSpeed models.")

    if args.train.async_enable:
        assert not args.vllm.enable_sleep, "Async RLHF is not supported with --vllm_enable_sleep."

    if args.train.partial_rollout_enable:
        assert args.train.async_enable, "--partial_rollout requires --async_train."

    if args.eval.dataset:
        assert args.reward.remote_url, "`--eval_dataset` is only supported with `--remote_rm_url`."

    if args.algo.kl.use_loss:
        if args.algo.kl.estimator not in ["k2", "k3"]:
            print(f"Recommend setting {args.algo.kl.estimator} to 'k2' or 'k3' when using KL as a loss")
    else:
        if args.algo.kl.estimator not in ["k1"]:
            print(f"Recommend setting {args.algo.kl.estimator} to 'k1' when not using KL as a loss.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.rollout.vllm_generate_batch_size:
        args.rollout.vllm_generate_batch_size = args.rollout.batch_size

    if args.rollout.vllm_generate_batch_size > args.rollout.batch_size:
        assert args.train.async_enable, (
            "--vllm_generate_batch_size > --rollout_batch_size requires --async_train "
            "(over-sampling needs async queue to buffer extra batches)."
        )

    if args.algo.dynamic_filtering_enable:
        assert (
            args.algo.dynamic_filtering_range[0] < args.algo.dynamic_filtering_range[1]
        ), "reward_clip_range[0] must be less than reward_clip_range[1]"
        assert (
            args.reward.remote_url or args.train.agent_func_path
        ), "remote_rm_url or agent_func_path must be specified when using dynamic filtering"
        assert (
            args.rollout.n_samples_per_prompt > 1
        ), "n_samples_per_prompt must be greater than 1 when using dynamic filtering"

    assert (
        args.rollout.n_samples_per_prompt * args.rollout.batch_size // args.rollout.micro_batch_size
        >= args.actor.num_nodes
        * args.actor.num_gpus_per_node
        // args.ds.ring_attn_size
        // args.ds.tensor_parallel_size
    ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
