<div align="center">
    <img alt="OpenRLHF logo" src="./docs/logo.png" style="height: 140px;" />
</div>
<div align="center">
<p align="center">
      <a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/OpenRLHF/OpenRLHF" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/OpenRLHF/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/OpenRLHF/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?color=ccf" />
      </a>
      <a href="https://deepwiki.com/OpenRLHF/OpenRLHF"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
      <br>
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</div>

<hr>

<span>[ English | <a href="README_zh.md">中文</a> | <a href="README_ja.md">日本語</a> ]</span>

OpenRLHF is **the first** high-performance, production-ready open-source RLHF framework that combines **Ray + vLLM distributed architecture** with a **unified agent-based design paradigm** for scalable and extensible reinforcement learning from human feedback.

📚 **Learn More**: [Documentation](https://openrlhf.readthedocs.io/) | [Slides](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [Technical Report](https://www.researchgate.net/publication/393414548_OpenRLHF_An_Easy-to-use_Scalable_and_High-performance_RLHF_Framework) | [Video](https://www.bilibili.com/video/BV1dv2jBxEQG/)

## 📖 Table of Contents

- [🗞️ News](#news)
- [🏗️ Architecture Foundation](#architecture-foundation-ray--vllm-distribution) - Ray + vLLM + DeepSpeed distributed infrastructure
- [🎯 Design Paradigm](#design-paradigm-agent-based-execution) - Unified agent-based execution pipeline
- [🚀 RL Algorithms](#state-of-the-art-rl-algorithms) - PPO, REINFORCE++, GRPO, RLOO
- [📋 Features Overview](#comprehensive-features) - Complete RLHF pipeline capabilities
- [🎬 Quick Start](#quick-start) - Installation and typical workflow
- [🎓 Training Guide](#supervised-fine-tuning) - SFT, Reward Model, RL Training
- [🎯 Single-Turn Agent](#single-turn-agent-reinforced-fine-tuning-with-custom-rewards) - Custom reward functions
- [🤖 Multi-Turn Agent](#multi-turn-agent-complex-environment-interactions) - Complex environments
- [🔧 Advanced Topics](#advanced-topics) - LoRA, performance tuning

---

<a id="news"></a>
## News

<details>
<summary>Show News</summary>

- [2026/4] OpenRLHF 0.10 adds **Multi-Turn VLM RL** — multi-step interactions with images in both prompts and environment feedback (e.g. screenshots). Example: [vlm_multiturn_agent.py](./examples/python/vlm_multiturn_agent.py)
- [2026/4] OpenRLHF 0.10 adds **VLM (Vision-Language Model) RLHF support** — train VLMs like Qwen3.5 with image inputs end-to-end. Training script: [train_vlm_math_hybrid_engine.sh](./examples/scripts/train_vlm_math_hybrid_engine.sh)
- [2026/2] [ProRL V2](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/) uses REINFORCE++-baseline to train a state-of-the-art 1.5B reasoning model with prolonged RL training. Training script: [train_prorlv2_math_hybrid_engine.sh](./examples/scripts/train_prorlv2_math_hybrid_engine.sh)
- [2026/10] [ScaleRL](https://arxiv.org/abs/2510.13786) validates the effectiveness of REINFORCE++-baseline in large-scale training scenarios. Releases [REINFORCE++ slides](https://docs.google.com/presentation/d/1stieP_3PM1z4Hq1YWR3GywFkxcHEAlstXMaS23KlGN4)
- [2026/6] [Magistral](https://mistral.ai/static/research/magistral.pdf) uses the method quite similar to REINFORCE++-baseline to train the reasoning models.
- [2026/5] [MARTI](https://github.com/TsinghuaC3I/MARTI) has been released as a fork of OpenRLHF. It is designed to train LLM-based multi-agent systems using RL, by integrating centralized multi-agent interactions with distributed policy training.
- [2026/5] OpenRLHF 0.8.0 supports async RLHF training via `--train.async_enable` and async agent RLHF via `--train.agent_func_path`. See [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh) for a runnable example.
- [2026/4] Post the blog [Accelerating RLHF with vLLM, Best Practice from OpenRLHF](https://blog.vllm.ai/2026/04/23/openrlhf-vllm.html)
- [2026/4] Clean OpenRLHF: Refactored the source code based on Single Controller and Unified Packing Samples
- [2026/3] The CMU [Advanced Natural Language Processing Spring 2026](https://cmu-l3.github.io/anlp-spring2026/) course uses OpenRLHF as the RLHF framework teaching case.
- [2026/2] [Logic-RL](https://arxiv.org/abs/2502.14768) and [PRIME](https://arxiv.org/abs/2502.01456) demonstrate that REINFORCE++ is more stable in training compared to GRPO and faster than PPO.
- [2026/2] [LMM-R1](https://github.com/TideDra/lmm-r1) is a fork of OpenRLHF, aimed at providing high-performance RL infrastructure for reproduction of DeepSeek-R1 on multimodal tasks.
- [2026/2] MIT & Microsoft proposed the [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) using OpenRLHF
- [2026/1] HKUST reproduced the [DeepSeek-R1-Zero and DeepSeek-R1 training on small models using OpenRLHF](https://github.com/hkust-nlp/simpleRL-reason)
- [2024/12] We "proposed" 😊 the [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://www.researchgate.net/publication/387487679_REINFORCE_An_Efficient_RLHF_Algorithm_with_Robustnessto_Both_Prompt_and_Reward_Models).
- [2024/12] We analyzed the PPO, REINFORCE++, GRPO and RLOO in the [Notion Blogpost](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05).
- [2023/8] OpenRLHF was open-sourced.

</details>

---

<a id="architecture-foundation-ray--vllm-distribution"></a>
## 🏗️ Architecture Foundation: Ray + vLLM Distribution

OpenRLHF is **the first RLHF framework** built on Ray + vLLM distributed architecture, orchestrating multiple components across GPUs efficiently:

<div align="center">
  <img alt="OpenRLHF Architecture (Ray + vLLM)" src="./docs/openrlhf_architecture.svg" style="max-width: 100%; height: auto;" />
</div>

### Core Infrastructure Components

**Ray - Distributed Scheduler and Controller**  
OpenRLHF leverages [Ray](https://github.com/ray-project/ray) for efficient distributed scheduling. It separates the Actor, Reward, Reference, and Critic models across different GPUs, enabling scalable training for models up to **70B+ parameters**.

**Hybrid Engine Scheduling**: All models and vLLM engines can share GPU resources—minimizing idle time and maximizing GPU utilization. This allows running full RLHF pipelines on limited hardware.

**vLLM - High-Performance Inference Engine**  
RLHF training spends **80% of the time on sample generation**. Powered by [vLLM](https://github.com/vllm-project/vllm) with Auto Tensor Parallelism (AutoTP) and Pipeline Parallelism (PP), OpenRLHF delivers high-throughput, memory-efficient generation.

**DeepSpeed - Memory-Efficient Training**  
Built on [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) ZeRO-3, [deepcompile](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/deepcompile/README.md), [AutoTP](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md), and RingAttention. Enables large model training without heavyweight frameworks while working directly with HuggingFace models.

**Transformers - Model Interface**  
Native integration with HuggingFace Transformers for seamless model loading, state management, and fine-tuning of pretrained models.

**NCCL / CUDA IPC - High-Speed Communication**  
Efficient inter-GPU communication for distributed training and inference.

---

<a id="design-paradigm-agent-based-execution"></a>
## 🎯 Design Paradigm: Agent-Based Execution

**On top of the Ray distributed architecture**, OpenRLHF is **the first RLHF framework** to implement a **unified agent-based paradigm**. Every training run—whether standard PPO or complex multi-turn reasoning—follows a consistent agent execution pipeline.

### Why Agent-Based?

OpenRLHF **unifies generation and training through token-in-token-out agent execution**, ensuring perfect consistency, easy single/multi-turn extension, and zero text-level mismatches.

### Agent Architecture

```
                 ┌─────────────────────────────┐
                 │    AgentExecutorBase        │
                 │  (Token-in-Token-out Core)  │
                 └─────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
         SingleTurnExecutor        MultiTurnExecutor
                 │                         │
      ┌──────────┴──────────┐   ┌─────────┴──────────┐
      ↓                     ↓   ↓                    ↓
  Standard RLHF      Custom Reward   Multi-Step    External Env
  (One-shot gen)     Function      Reasoning     (OpenAI Agent Server)
      ↓                     ↓           ↓                ↓
      └─────────────────────┴───────────┴────────────────┘
                              │
                    Consistent Token Trajectories
                              │
                    ┌─────────┴─────────┐
                    │  RL Algorithms    │
                    │  (Decoupled)      │
                    │                   │
                    │  PPO, REINFORCE++ │
                    │  GRPO, RLOO, etc. │
                    └───────────────────┘
```

### Core Design Principles

<details>
<summary>Show core design principles</summary>

| Principle | Description | Benefit |
|-----------|-------------|---------|
| **Token-in-Token-out** | All sampling produces token-level trajectories | Zero text-level mismatch |
| **Unified Interface** | Same `AgentExecutorBase` API for all modes | Switch modes with one flag |
| **Algorithm-Agnostic** | RL algorithms (PPO, REINFORCE++, etc.) are decoupled from agent executors | Any algorithm works with any mode |
| **Extensible** | Plug in custom rewards/environments easily | Rapid experimentation |
| **Production-Ready** | Sync/Async/Hybrid Engine support | From research to deployment |

</details>

### Two Execution Modes (Orthogonal to RL Algorithms)

The agent execution mode is **independent** of the RL algorithm you choose. You can use **any algorithm** (PPO, REINFORCE++, GRPO, etc.) with **any execution mode**:

| Mode | Use Cases | Interface | Complexity |
|------|-----------|-----------|------------|
| **Single-Turn** | Standard RLHF, custom reward functions | Optional `reward_func()` | ⭐ Default (99% use cases) |
| **Multi-Turn** | Multi-step reasoning, interactive environments | `reset()` + `step()` | ⭐⭐ Advanced |

---

<a id="state-of-the-art-rl-algorithms"></a>
## 🚀 State-of-the-Art RL Algorithms

OpenRLHF implements **PPO, REINFORCE++, REINFORCE++-baseline, GRPO, RLOO** with advanced optimization tricks inspired by practical guides and community best practices. 

**Key Design**: RL algorithms are **decoupled from agent execution modes**. All algorithms work seamlessly with both single-turn and multi-turn agent executors, running through the unified token-in-token-out pipeline for consistent behavior.

<details>
<summary>Show algorithm comparison table</summary>

| Algorithm | `--algo.advantage.estimator` | Key Feature | Best Use Case |
|-----------|------------------------|-------------|---------------|
| **PPO** | (default) | Full critic network | Stable training, proven results |
| **REINFORCE++** | `reinforce` | PPO tricks without critic | Efficient training, less memory |
| **REINFORCE++-baseline** | `reinforce_baseline` | Mean reward baseline | Reasoning tasks (RLVR), robust to reward scales |
| **RLOO** | `rloo` | Per-token KL + PPO-clip | Multi-sample training |
| **GRPO** | `group_norm` | Group normalization | Batch-based training |
| **Dr. GRPO** | `dr_grpo` | Simplified GRPO | Removes local `/std` norm |

</details>

References: [Zhihu article](https://zhuanlan.zhihu.com/p/622134699) | [Notion best practices](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361)

---

<a id="comprehensive-features"></a>
## 📋 Comprehensive Features

OpenRLHF provides a complete RLHF pipeline with agent-based flexibility:

### 🎯 Agent-Based RL Training (Core Innovation)

<details>
<summary>Show agent-based RL training details</summary>

**Single-Turn Mode** (Default - 99% of use cases)
- One-shot generation per prompt
- Works with all RL algorithms: [PPO](./examples/scripts/train_ppo_ray_hybrid_engine.sh), [REINFORCE++/baseline/GRPO/RLOO](./examples/scripts/train_reinforce_baseline_hybrid_engine.sh)
- [Custom reward functions](./examples/scripts/train_ppo_with_reward_fn.sh) (`--reward.remote_url`)
- [Hybrid Engine](./examples/scripts/train_ppo_ray_hybrid_engine.sh) for maximum GPU utilization

**Multi-Turn Mode** (Advanced - Interactive tasks)
- Multi-step interactions with environment feedback
- Works with all RL algorithms
- [Custom agent functions](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh) (`--train.agent_func_path`)
- OpenAI-compatible server: see `examples/python/agent_func_openai_server_executor.py` for an agent executor that wraps vLLM as a local OpenAI Agent Server
- Async pipeline (`--train.async_enable`) for higher throughput: [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh)

</details>

### 🎓 Supervised Training & Preference Learning

<details>
<summary>Show supervised training & preference learning table</summary>

| Method | Script | Description |
|--------|--------|-------------|
| **SFT** | [train_sft.sh](./examples/scripts/train_sft.sh) | Supervised fine-tuning with packing |
| **DPO/IPO/cDPO** | [train_dpo_llama.sh](./examples/scripts/train_dpo_llama.sh) | Direct preference optimization |
| **Reward Model** | [train_rm.sh](./examples/scripts/train_rm.sh) | Train reward models |

</details>

### ⚡ Advanced Capabilities

<details>
<summary>Show advanced capabilities</summary>

**Efficiency Optimizations**
- Sample packing (`--ds.packing_samples`) for all training modes
- vLLM acceleration (`--vllm.num_engines`) for fast generation
- DAPO [dynamic filtering](./examples/scripts/train_dapo_ray_hybrid_engine.sh) (`--algo.dynamic_filtering_enable`)
  - 🎲 Dynamic Sampling: for each prompt, generate multiple responses and **filter** them by your reward / agent **0–1 `scores`** signal
    - Enable: `--algo.dynamic_filtering_enable`
    - Score range: `--algo.dynamic_filtering_range 0.0 1.0`
    - Requires: `--rollout.n_samples_per_prompt > 1` and either `--reward.remote_url` or `--train.agent_func_path`
    - Example: `./examples/scripts/train_dapo_ray_hybrid_engine.sh`

**Scalability**
- DeepSpeed AutoTP for tensor parallelism (see `--ds.tensor_parallel_size` in training scripts)
- [RingAttention](./examples/test_scripts/train_dpo_ring_llama.sh) for long context (`--ds.ring_attn_size`)
- Multi-node training with [SLURM](./examples/scripts/train_ppo_ray_slurm.sh)

**Model Support**
- [VLM (Vision-Language Models)](./examples/scripts/train_vlm_math_hybrid_engine.sh) — single-turn and [multi-turn with image feedback](./examples/python/vlm_multiturn_agent.py) (`--data.image_key`, `--data.max_images_per_prompt`)
- [LoRA/QLoRA](./examples/scripts/train_sft_mixtral_lora.sh) (`--ds.lora.rank`, `--ds.load_in_4bit`)
- [Mixture of Experts (MoE)](./examples/test_scripts/train_sft_moe.sh) (`--actor.aux_loss_coef`)
- FlashAttention (`--ds.attn_implementation`)
- HuggingFace chat templates (`--data.apply_chat_template`)

**Optimizers**
- AdamW (default): `--{actor,critic}.optim adam --{actor,critic}.adam.lr 2e-6`
- [Muon](https://kellerjordan.github.io/posts/muon/) (via DeepSpeed ≥ 0.18.2, 2D weights only; embeddings / head / 1-D params use aux-AdamW): `--{actor,critic}.optim muon --{actor,critic}.muon.lr 1e-4 --{actor,critic}.muon.momentum 0.95`. Newton-Schulz produces scale-invariant updates, so disable global grad clipping with `--{actor,critic}.max_norm 0` (the Adam default `1.0` would clip away the Muon update).

**Reward Shaping**
- DAPO-style overlong penalty for length control (`--reward.overlong_buffer_len`, `--reward.overlong_penalty_factor`) — soft-penalize responses that exceed `max_new_tokens - overlong_buffer_len`
- ProRL-style truncation penalty (`--reward.stop_properly_penalty_coef`) — for samples with `finish_reason='length'`: `coef ∈ [0, 1]` multiplicatively scales the reward; `coef < 0` sets the reward to that fixed value (e.g. `-0.5`)

**Production Features**
- Wandb (`--logger.wandb.key`) and TensorBoard (`--logger.tensorboard_dir`) logging
- Checkpoint recovery (`--ckpt.load_enable`, `--ckpt.save_steps`)
- Best-checkpoint saving on eval metrics (`--ckpt.best_metric_key`)
- Evaluation datasets (`--eval.dataset`, `--eval.temperature`, `--eval.n_samples_per_prompt`) — supported in async training
- Multi-process data loading (`--data.dataloader_num_workers`, available for PPO/SFT/RM/DPO)
- PPO observability: actor/critic grad-norm and per-phase timing (`timing/make_experience`, `timing/ppo_train`, `timing/broadcast`, `timing/generation`, `timing/step_total`)

</details>

---

<a id="quick-start"></a>
## 🎬 Quick Start

### Installation

**Recommended**: Use Docker for hassle-free setup

```bash
# 1. Launch Docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.11-py3 bash

# 2. Clean conflicting packages
sudo pip uninstall xgboost transformer_engine flash_attn pynvml -y

# 3. Install OpenRLHF (choose one)
pip install openrlhf                    # Basic
pip install openrlhf[vllm]              # + vLLM 0.19.1 (recommended)
pip install openrlhf[vllm_latest]       # + Latest vLLM
pip install openrlhf[vllm,ring,liger]   # + All optimizations
```

**Alternative: Install from source**

```bash
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!TIP]
> We recommend **vLLM 0.19.1+** for best performance. See [Dockerfiles](./dockerfile/) and [Nvidia-Docker Install Script](./examples/scripts/nvidia_docker_install.sh).

### Prepare Datasets

OpenRLHF provides flexible data processing methods:

**Key Parameters**:
- `--data.input_key`: Specify JSON key name for input data
- `--data.apply_chat_template`: Use HuggingFace tokenizer's [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
- `--data.input_template`: Custom template string (alternative to chat template)
- `--data.prompt_probs` / `--data.dataset_probs`: Mix multiple datasets (e.g., `0.1,0.4,0.5`)
- `--eval.dataset`: Specify evaluation dataset path

**Chat Template Example**:

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)
# Output: "<s>[INST] Hello, how are you? [/INST]I'm doing great...</s> [INST] I'd like to show off... [/INST]"
```

> [!NOTE]
> JSON key options vary by dataset type. See [Reward Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10), [SFT Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9), and [Prompt Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6)

<a id="supervised-fine-tuning"></a>
### Supervised Fine-tuning

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using `--actor.model_name_or_path  {name or path}`, `--reward.model_name_or_path  {name or path}` and `--critic.model_name_or_path  {name or path}`. We have provided some pre-trained checkpoints and datasets on [HuggingFace OpenRLHF](https://huggingface.co/OpenRLHF).

Then you can use the startup scripts we provide in the [examples/scripts](./examples/scripts/) directory, or start the training using the following commands.

<details>
<summary>SFT command</summary>

```bash
deepspeed --module openrlhf.cli.train_sft \
   --data.max_len 4096 \
   --data.dataset Open-Orca/OpenOrca \
   --data.input_key question \
   --data.output_key response \
   --data.input_template $'User: {}\nAssistant: ' \
   --train.batch_size 256 \
   --train.micro_batch_size 2 \
   --data.max_samples 500000 \
   --actor.model_name_or_path meta-llama/Meta-Llama-3-8B \
   --ckpt.output_dir ./checkpoint/llama3-8b-sft \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --ds.zero_stage 2 \
   --train.max_epochs 1 \
   --ds.packing_samples \
   --ds.param_dtype bf16 \
   --adam.lr 5e-6 \
   --actor.gradient_checkpointing_enable \
   --logger.wandb.key {wandb_token}

# Additional options:
# --data.apply_chat_template                # Use HF tokenizer chat template
# --ds.ring_attn_size 2                      # Enable RingAttention (install ring_flash_attn first)
# --data.multiturn                          # Multi-turn fine-tuning loss
# --actor.pretrain_mode_enable                      # Continued pre-training mode
```

</details>


### Reward Model Training

<details>
<summary>Reward model training command</summary>

```bash
deepspeed --module openrlhf.cli.train_rm \
   --ckpt.output_dir ./checkpoint/llama3-8b-rm \
   --ckpt.save_steps -1 \
   --logger.logging_steps 1 \
   --eval.steps -1 \
   --train.batch_size 256 \
   --train.micro_batch_size 1 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --ds.param_dtype bf16 \
   --train.max_epochs 1 \
   --data.max_len 8192 \
   --ds.zero_stage 3 \
   --adam.lr 9e-6 \
   --data.dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --data.apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --ds.packing_samples \
   --actor.gradient_checkpointing_enable \
   --logger.wandb.key {wandb_token}

```

</details>

It is recommended to set the `--value_prefix_head` option of the Reward Model to `score`, so that we can load the model using `AutoModelForSequenceClassification`:

```python
reward_model = AutoModelForSequenceClassification.from_pretrained(
              reward_model_path,
              num_labels=1,
              torch_dtype=torch.bfloat16,
              attn_implementation="flash_attention_2",
              use_cache=False,
          )
inputs = xxxx (Left Padding Input Tokens)
reward = reward_model.model(*inputs).last_hidden_state
reward = reward_model.score(reward)[:, -1]
```

### RL Training: PPO/REINFORCE++ with Ray and vLLM

All RL training in OpenRLHF runs through the **agent execution pipeline**. The following example shows single-turn agent execution (default mode) with Hybrid Engine for optimal performance:

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 8 \
   --reward.num_nodes 1 \
   --reward.num_gpus_per_node 8 \
   --critic.num_nodes 1 \
   --critic.num_gpus_per_node 8 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 8 \
   --vllm.num_engines 4 \
   --vllm.tensor_parallel_size 2 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.5 \
   --actor.model_name_or_path OpenRLHF/Llama-3-8b-sft-mixture \
   --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
   --ckpt.output_dir /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt.path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --ckpt.save_hf \
   --train.batch_size 128 \
   --rollout.batch_size 1024 \
   --train.dynamic_batch_enable \
   --rollout.n_samples_per_prompt 1 \
   --train.max_epochs 1 \
   --prompt_max_len 1024 \
   --data.max_samples 100000 \
   --generate_max_len 1024 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 5e-7 \
   --critic.adam.lr 9e-6 \
   --algo.kl.init_coef 0.01 \
   --data.prompt_dataset OpenRLHF/prompt-collection-v0.1 \
   --data.input_key context_messages \
   --data.apply_chat_template \
   --reward.normalize_enable \
   --actor.gradient_checkpointing_enable \
   --ds.packing_samples \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep \
   --logger.wandb.key {wandb_token}

# Algorithm Variants (all use single-turn agent execution):
# --algo.advantage.estimator reinforce        # REINFORCE++
# --algo.advantage.estimator rloo             # RLOO
# --algo.advantage.estimator reinforce_baseline  # REINFORCE++-baseline (best for RLVR)
# --algo.advantage.estimator group_norm       # GRPO
# --algo.advantage.estimator dr_grpo          # Dr. GRPO

# Advanced Options:
# --algo.kl.init_coef 0                                    # No reference model
# --reward.remote_url http://host:5000/get_reward         # HTTP reward model
# --rollout.n_samples_per_prompt 4                            # Multiple samples per prompt
# --rollout.vllm_generate_batch_size 2048                     # Oversample at generation (> rollout_batch_size); requires --train.async_enable
# --algo.advantage.is_correction_enable                         # vLLM importance sampling correction for off-policy rollouts
# --algo.advantage.is_correction_type tis                       # Correction type: tis (token clamp) | icepop (token filter) | seq-mask-tis (seq-level geom mean)
# --algo.advantage.is_correction_threshold 0.5 5.0               # IS truncation interval: [low, high]
# --ckpt.best_metric_key eval_default_pass1                # Save best checkpoint by eval metric (empty = auto-detect first pass1, 'none' = disable)
# --actor.policy_loss_type gspo                             # Use GSPO policy loss variant (vs default 'ppo')
```

> [!TIP]
> **For reasoning tasks (RLVR)**: Use `--algo.advantage.estimator reinforce_baseline` for REINFORCE++-baseline—it's robust to different reward scales.

> [!NOTE]
> **Ray Environment Setup**: Let Ray auto-deploy with `--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`

> [!NOTE]
> **Troubleshooting GPU index errors**: Set `export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` if you encounter DeepSpeed GPU device setup issues.

📚 **More Examples**: See [examples/scripts](./examples/scripts/) and [Documentation](https://openrlhf.readthedocs.io/en/latest/usage.html)

---

<a id="single-turn-agent-reinforced-fine-tuning-with-custom-rewards"></a>
## 🎯 Single-Turn Agent: Reinforced Fine-tuning with Custom Rewards

The **single-turn agent execution** (default mode) supports custom reward functions—perfect for reinforced fine-tuning without a trained reward model. Instead of using a pre-trained reward model, you provide a Python function that computes rewards on-the-fly.

**Ideal for**:
- Rule-based rewards (length, format, code execution, math verification)
- External API rewards (judge models, compilers, test suites)
- Hybrid rewards (combining multiple signals)

### Example: Custom Reward Function

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    """
    Compute custom rewards for generated responses.
    
    Args:
        queries: List[str] - Full text (prompt + response)
        prompts: List[str] - Original prompts only
        labels: List[str] - Ground truth labels (from --label_key)
    
    Returns:
        dict with:
            - rewards: Tensor for advantage calculation
            - scores: Tensor for dynamic filtering (0-1 range)
            - extra_logs: Dict for wandb logging
    """
    batch_size = len(queries)
    
    # Example: Random rewards (replace with your logic)
    # Real examples: code execution, math verification, format checking
    reward = torch.randint(0, 2, (batch_size,)).float()

    return {
        "rewards": reward,           # Used in RL advantage calculation
        "scores": reward,            # Used for dynamic filtering (--dynamic_filtering)
        "extra_logs": {              # Logged to wandb
            "custom_metric": reward.mean().item(),
        },
    }
```

### Usage

```bash
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path meta-llama/Meta-Llama-3-8B \
  --train.dynamic_batch_enable \
  --reward.remote_url /path/to/reward_func.py \
  --data.label_key answer \
  --data.prompt_dataset your_prompt_dataset \
  ... # other training args
```

**Key Parameter**: `--data.label_key answer` passes the "answer" field from your dataset to `reward_func` as `labels`.

> [!TIP]
> **Use Cases**: Code generation (execute tests), Math (verify solutions), Formatting (check structure), Multi-objective (combine multiple signals)

📖 **Full Example**: [examples/scripts/train_ppo_with_reward_fn.sh](./examples/scripts/train_ppo_with_reward_fn.sh)

---

<a id="multi-turn-agent-complex-environment-interactions"></a>
## 🤖 Multi-Turn Agent: Complex Environment Interactions

For tasks requiring **multi-step interactions** (reasoning chains, coding with feedback, game playing), OpenRLHF provides the **Multi-Turn Agent Execution** mode.

### Building Custom Multi-Turn Agents

Implement `AgentInstanceBase` with `reset/step` methods:

```python
# agent_func.py
import random
from typing import Any, Dict

import torch
from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor


# A simple n-step random environment
class AgentInstance(AgentInstanceBase):
    async def __init__(self, *args, **kwargs):
        self.step_idx = 0
        self.max_steps = random.randint(1, 3)  # 1-3 steps

    async def reset(self, states: dict, **kwargs):
        return {"observation": states["observation"]}  # Return original text observation

    async def step(self, states: dict, **kwargs) -> Dict[str, Any]:
        print(f"step_idx: {self.step_idx}, max_steps: {self.max_steps}")

        observation_text = states["observation_text"]
        action_text = states["action_text"]
        label = states["label"]

        # Check if episode is done
        done = self.step_idx >= self.max_steps
        reward = torch.randint(0, 2, (1,)).float() if done else torch.tensor(0)

        # Generate environment feedback based on whether episode is done
        environment_feedback = (
            "\n\nHuman: [CORRECT]\n</s>"
            if done
            else "\n\nHuman: [INCORRECT]\nPlease analyze the issues and try again.\n</s>\n\nAssistant: "
        )

        self.step_idx += 1

        return {
            "rewards": reward,  # Rewards for advantage calculation
            "scores": reward,  # Scores for dynamic filtering (0-1 reward)
            "environment_feedback": environment_feedback,  # Environment feedback text
            "done": done,  # Boolean indicating if the episode is complete
            "sampling_params": states.get("sampling_params", None),  # Parameters for vLLM sampling in next step
            "extra_logs": {"dummy_scores": reward},  # Additional logging information
        }


class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(AgentInstance)
```

Then launch with:

```bash
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  ...
  --train.dynamic_batch_enable \
  --train.agent_func_path /path/to/agent_func.py \
  --train.async_enable  # Optional: enable async pipeline
```

### Configuration Options

**Async Pipeline** (for higher throughput):
- Enable: `--train.async_enable`
- Buffer size: `--train.async_queue_size 1` (larger = more off-policy, default 1)
- Partial rollout: `--train.partial_rollout_enable` — uses vLLM pause/resume for weight sync instead of locking, allowing generation to overlap with training. In-flight samples may contain tokens from both old and new weights.

**Training Modes**:
- **Synchronous**: Default, better stability
- **Asynchronous**: Higher throughput, may affect convergence
- **Hybrid Engine**: Best GPU utilization with `--train.colocate_all` (remove `--train.async_enable`)

> [!NOTE]
> For fully custom token-level execution, inherit `AgentExecutorBase` and implement `execute()`. This design enforces the **token-in-token-out principle** to keep sampling and training consistent.

> [!WARNING] 
> Asynchronous training may affect training stability. Use it only when throughput is critical and convergence is validated.

📚 **Examples**:
- Single-turn: [train_ppo_ray_hybrid_engine.sh](./examples/scripts/train_ppo_ray_hybrid_engine.sh)
- Custom reward: [train_ppo_with_reward_fn.sh](./examples/scripts/train_ppo_with_reward_fn.sh)
- Multi-turn: [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh)
- Multi-turn VLM (image feedback): [vlm_multiturn_agent.py](./examples/python/vlm_multiturn_agent.py)

### OpenAI-Compatible Agent Server

For multi-turn agents that need an OpenAI-compatible chat API (e.g., integrating external tool-use frameworks), [`agent_func_openai_server_executor.py`](./examples/python/agent_func_openai_server_executor.py) wraps vLLM as a local `/v1/chat/completions` server while collecting token-level traces for RL training.

- Exposes standard OpenAI endpoints (`/v1/chat/completions`, `/v1/models`, `/tokenize`)
- Automatically collects token IDs and logprobs per session for RL training
- Delta-tokenization reuses prefix tokens across multi-turn calls
- Override `run_agent()` to plug in your own multi-turn workflow

```bash
python3 -m openrlhf.cli.train_ppo_ray \
  --train.agent_func_path examples/python/agent_func_openai_server_executor.py \
  ... # other training args
```

---

<a id="advanced-topics"></a>
## 🔧 Advanced Topics

### LoRA: Merging Adapters

When using LoRA/QLoRA, OpenRLHF saves only the adapter weights. To deploy or continue training, merge the adapter with the base model:

```bash
python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama3-8b-rm \
    --output_path ./checkpoint/llama-3-8b-rm-combined \
    --is_rm \
    --ds.param_dtype bf16
```

### Performance Tuning Guide

Optimize OpenRLHF for your hardware and workload with these recommendations:

#### 🎯 Execution Modes: Throughput vs. Stability

Pick the execution mode based on your priority — OpenRLHF gives you a clear tradeoff knob:

| Mode | Flags | Characteristics | When to Use |
|------|-------|-----------------|-------------|
| **Hybrid Engine (colocated)** | `--train.colocate_all`<br>`--vllm.enable_sleep`<br>`--ds.enable_sleep` | **Most stable** — strictly on-policy, every rollout uses the latest weights. Serial generate→train cycle. | Research, sensitive RL algorithms, reproducibility, recipe validation |
| **Async Training** | `--train.async_enable`<br>`--train.async_queue_size N` | **Highest throughput** — generation and training run in parallel. Tune off-policyness via `--train.async_queue_size` (larger = more off-policy). | Production throughput when convergence is already validated |
| **Async + Partial Rollout** | `--train.async_enable`<br>`--train.partial_rollout_enable` | **Maximum overlap** — vLLM pause/resume instead of locking, in-flight samples may mix old/new weights. Most aggressive off-policy. | Pushing async throughput further; pair with `--algo.advantage.is_correction_enable` |

#### ⚡ Other Speed Optimizations

| Optimization | Flag | When to Use |
|--------------|------|-------------|
| **Sample Packing** | `--ds.packing_samples` | Always (especially training) |
| **Dynamic Batch** | `--train.dynamic_batch_enable` | Variable sequence lengths |
| **DeepCompile** | `--ds.deepcompile` | PyTorch 2.0+ |
| **Overlap Comm** | `--ds.overlap_comm` | Sufficient GPU memory |
| **Prefix Caching** | vLLM config | `n_samples_per_prompt` > 1 |
| **Oversampling** | `--rollout.vllm_generate_batch_size > --rollout.batch_size` | Async mode, to amortize generation cost / feed dynamic filtering |

#### 💾 Memory Management

**When you have enough memory**:
- ✅ Disable `--ds.adam_offload`
- ✅ Enable `--ds.overlap_comm`
- ✅ Use `--train.colocate_critic_reward` and `--train.colocate_actor_ref`

**When hitting OOM**:
- ❌ Disable all `--colocate_*` options
- ✅ Reduce batch sizes
- ✅ Enable gradient checkpointing

#### 🎮 Batch Size Tuning

1. **Generation Phase**: Maximize `--rollout.micro_batch_size`, minimize vLLM TP size
2. **Training Phase**: Maximize `--train.micro_batch_size`, enable `--ds.packing_samples`
3. **vLLM**: Always use `--vllm.sync_backend nccl`

> [!TIP]
> **Quick Start Template**: For 8x A100 (80GB), try Hybrid Engine + `--vllm.gpu_memory_utilization 0.5` + `--train.colocate_all`

📖 **More Details**: [Performance Tuning Documentation](https://openrlhf.readthedocs.io/en/latest/performance.html)


## Companies and Organizations using OpenRLHF

- Google
- ByteDance
- Tencent
- Alibaba
- Baidu
- China Telecom
- Vivo
- Allen AI
- NexusFlow
- Jülich Supercomputing Centre (JSC)
- Berkeley Starling Team
- M-A-P
- ...

## Join Us

**How to Join?**

1. Email us at janhu9527@gmail.com or join [GitHub Organization](https://github.com/OpenRLHF). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/OpenRLHF).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRLHF/OpenRLHF&type=Date)](https://star-history.com/#OpenRLHF/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRLHF/OpenRLHF" />
</a>

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA ↗](https://llama.meta.com/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalChat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design. 
Our project would like to thank [Netmind.AI](https://www.netmind.ai/) for the GPU support of developing ring attention.

(2024/7) Our GitHub organization has changed from OpenLLMAI to OpenRLHF.

## Citation
OpenRLHF

```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```
REINFORCE++-baseline
```
@article{hu2026reinforce++,
  title={Reinforce++: A simple and efficient approach for aligning large language models},
  author={Hu, Jian},
  journal={arXiv preprint arXiv:2501.03262},
  year={2026}
}
```

______________________________________________________________________

*OpenRLHF © 2026 OpenRLHF. All Rights Reserved.*
