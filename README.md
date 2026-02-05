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

- [2026/2] [ProRL V2](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/) uses REINFORCE++-baseline to train a state-of-the-art 1.5B reasoning model with prolonged RL training. Training script: [train_prorlv2_math_hybrid_engine.sh](./examples/scripts/train_prorlv2_math_hybrid_engine.sh)
- [2025/10] [ScaleRL](https://arxiv.org/abs/2510.13786) validates the effectiveness of REINFORCE++-baseline in large-scale training scenarios. Releases [REINFORCE++ slides](https://docs.google.com/presentation/d/1stieP_3PM1z4Hq1YWR3GywFkxcHEAlstXMaS23KlGN4)
- [2025/6] [Magistral](https://mistral.ai/static/research/magistral.pdf) uses the method quite similar to REINFORCE++-baseline to train the reasoning models.
- [2025/5] [MARTI](https://github.com/TsinghuaC3I/MARTI) has been released as a fork of OpenRLHF. It is designed to train LLM-based multi-agent systems using RL, by integrating centralized multi-agent interactions with distributed policy training.
- [2025/5] OpenRLHF 0.8.0 supports async RLHF training via `--async_train` and async agent RLHF via `--agent_func_path`. See [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh) for a runnable example.
- [2025/4] Post the blog [Accelerating RLHF with vLLM, Best Practice from OpenRLHF](https://blog.vllm.ai/2025/04/23/openrlhf-vllm.html)
- [2025/4] Clean OpenRLHF: Refactored the source code based on Single Controller and Unified Packing Samples
- [2025/3] The CMU [Advanced Natural Language Processing Spring 2025](https://cmu-l3.github.io/anlp-spring2025/) course uses OpenRLHF as the RLHF framework teaching case.
- [2025/2] [Logic-RL](https://arxiv.org/abs/2502.14768) and [PRIME](https://arxiv.org/abs/2502.01456) demonstrate that REINFORCE++ is more stable in training compared to GRPO and faster than PPO.
- [2025/2] [LMM-R1](https://github.com/TideDra/lmm-r1) is a fork of OpenRLHF, aimed at providing high-performance RL infrastructure for reproduction of DeepSeek-R1 on multimodal tasks.
- [2025/2] MIT & Microsoft proposed the [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) using OpenRLHF
- [2025/1] HKUST reproduced the [DeepSeek-R1-Zero and DeepSeek-R1 training on small models using OpenRLHF](https://github.com/hkust-nlp/simpleRL-reason)
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
  (One-shot gen)     Function      Reasoning     (NeMo Gym)
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

| Algorithm | `--advantage_estimator` | Key Feature | Best Use Case |
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
- [Custom reward functions](./examples/scripts/train_ppo_with_reward_fn.sh) (`--remote_rm_url`)
- [Hybrid Engine](./examples/scripts/train_ppo_ray_hybrid_engine.sh) for maximum GPU utilization

**Multi-Turn Mode** (Advanced - Interactive tasks)
- Multi-step interactions with environment feedback
- Works with all RL algorithms
- [Custom agent functions](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh) (`--agent_func_path`)
- NeMo Gym integration: see `examples/python/agent_func_nemogym_executor.py` for an agent executor that integrates NeMo Gym rollouts
- Async pipeline (`--async_train`) for higher throughput: [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh)

</details>

### 🎓 Supervised Training & Preference Learning

<details>
<summary>Show supervised training & preference learning table</summary>

| Method | Script | Description |
|--------|--------|-------------|
| **SFT** | [train_sft.sh](./examples/scripts/train_sft.sh) | Supervised fine-tuning with packing |
| **DPO/IPO/cDPO** | [train_dpo_llama.sh](./examples/scripts/train_dpo_llama.sh) | Direct preference optimization |
| **KTO** | [train_kto_llama.sh](./examples/scripts/train_kto_llama.sh) | Kahneman-Tversky optimization |
| **Iterative DPO** | [train_iterative_dpo.sh](./examples/scripts/train_iterative_dpo.sh) | Online preference learning |
| **Reward Model** | [train_rm.sh](./examples/scripts/train_rm.sh) | Train reward models |
| **Process RM** | [train_prm_mistral.sh](./examples/scripts/train_prm_mistral.sh) | Step-by-step reward models |
| **Rejection Sampling** | [train_rejection_sampling_llama.sh](./examples/scripts/train_rejection_sampling_llama.sh) | Best-of-N sampling |
| **Conditional SFT** | [train_conditional.sh](./examples/scripts/train_conditional.sh) | Quality-conditioned training |
| **Distillation** | [train_knowledge_distillation.sh](./examples/scripts/train_knowledge_distillation.sh) | Knowledge transfer |

</details>

### ⚡ Advanced Capabilities

<details>
<summary>Show advanced capabilities</summary>

**Efficiency Optimizations**
- Sample packing (`--packing_samples`) for all training modes
- vLLM acceleration (`--vllm_num_engines`) for fast generation
- DAPO [dynamic filtering](./examples/scripts/train_dapo_ray_hybrid_engine.sh) (`--dynamic_filtering`)
  - 🎲 Dynamic Sampling: for each prompt, generate multiple responses and **filter** them by your reward / agent **0–1 `scores`** signal
    - Enable: `--dynamic_filtering`
    - Score range: `--dynamic_filtering_reward_range 0.0 1.0`
    - Requires: `--n_samples_per_prompt > 1` and either `--remote_rm_url` or `--agent_func_path`
    - Example: `./examples/scripts/train_dapo_ray_hybrid_engine.sh`

**Scalability**
- DeepSpeed AutoTP for tensor parallelism (see `--ds_tensor_parallel_size` in training scripts)
- [RingAttention](./examples/test_scripts/train_dpo_ring_llama.sh) for long context (`--ring_attn_size`)
- Multi-node training with [SLURM](./examples/scripts/train_ppo_ray_slurm.sh)

**Model Support**
- [LoRA/QLoRA](./examples/scripts/train_sft_mixtral_lora.sh) (`--lora_rank`, `--load_in_4bit`)
- [Mixture of Experts (MoE)](./examples/test_scripts/train_sft_moe.sh) (`--aux_loss_coef`)
- FlashAttention (`--attn_implementation`)
- HuggingFace chat templates (`--apply_chat_template`)

**Production Features**
- Wandb (`--use_wandb`) and TensorBoard (`--use_tensorboard`) logging
- Checkpoint recovery (`--load_checkpoint`, `--save_steps`)
- Evaluation datasets (`--eval_dataset`)

</details>

---

<a id="quick-start"></a>
## 🎬 Quick Start

### Installation

**Recommended**: Use Docker for hassle-free setup

```bash
# 1. Launch Docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
  -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.11-py3bash

# 2. Clean conflicting packages
sudo pip uninstall xgboost transformer_engine flash_attn pynvml -y

# 3. Install OpenRLHF (choose one)
pip install openrlhf                    # Basic
pip install openrlhf[vllm]              # + vLLM 0.15.0 (recommended)
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
> We recommend **vLLM 0.15.0+** for best performance. See [Dockerfiles](./dockerfile/) and [Nvidia-Docker Install Script](./examples/scripts/nvidia_docker_install.sh).

### Prepare Datasets

OpenRLHF provides flexible data processing methods:

**Key Parameters**:
- `--input_key`: Specify JSON key name for input data
- `--apply_chat_template`: Use HuggingFace tokenizer's [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating)
- `--input_template`: Custom template string (alternative to chat template)
- `--prompt_data_probs` / `--dataset_probs`: Mix multiple datasets (e.g., `0.1,0.4,0.5`)
- `--eval_dataset`: Specify evaluation dataset path

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

OpenRLHF's model checkpoint is fully compatible with HuggingFace models. You can specify the model name or path using `--pretrain  {name or path}`, `--reward_pretrain  {name or path}` and `--critic_pretrain  {name or path}`. We have provided some pre-trained checkpoints and datasets on [HuggingFace OpenRLHF](https://huggingface.co/OpenRLHF).

Then you can use the startup scripts we provide in the [examples/scripts](./examples/scripts/) directory, or start the training using the following commands.

<details>
<summary>SFT command</summary>

```bash
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --input_template $'User: {}\nAssistant: ' \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --packing_samples \
   --param_dtype bf16 \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# Additional options:
# --apply_chat_template                # Use HF tokenizer chat template
# --ring_attn_size 2                   # Enable RingAttention (install ring_flash_attn first)
# --multiturn                          # Multi-turn fine-tuning loss
# --pretrain_mode                      # Continued pre-training mode
```

</details>


### Reward Model Training

<details>
<summary>Reward model training command</summary>

```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --param_dtype bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --packing_samples \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

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
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
   --save_hf_ckpt \
   --train_batch_size 128 \
   --rollout_batch_size 1024 \
   --use_dynamic_batch \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --use_wandb {wandb_token}

# Algorithm Variants (all use single-turn agent execution):
# --advantage_estimator reinforce        # REINFORCE++
# --advantage_estimator rloo             # RLOO
# --advantage_estimator reinforce_baseline  # REINFORCE++-baseline (best for RLVR)
# --advantage_estimator group_norm       # GRPO
# --advantage_estimator dr_grpo          # Dr. GRPO

# Advanced Options:
# --init_kl_coef 0                      # No reference model
# --remote_rm_url http://host:5000/get_reward  # HTTP reward model
# --n_samples_per_prompt 4              # Multiple samples per prompt
# --enable_vllm_is_correction           # TIS (vLLM importance sampling correction) for off-policy rollouts (PPO only)
# --vllm_is_truncated_threshold 0.5 5.0 # TIS truncation interval: [low, high]
# --use_icepop                          # ICEPOP: set coefficients outside [low, high] to 0 (instead of clamp)
```

> [!TIP]
> **For reasoning tasks (RLVR)**: Use `--advantage_estimator reinforce_baseline` for REINFORCE++-baseline—it's robust to different reward scales.

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
  --pretrain meta-llama/Meta-Llama-3-8B \
  --use_dynamic_batch \
  --remote_rm_url /path/to/reward_func.py \
  --label_key answer \
  --prompt_data your_prompt_dataset \
  ... # other training args
```

**Key Parameter**: `--label_key answer` passes the "answer" field from your dataset to `reward_func` as `labels`.

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
  --use_dynamic_batch \
  --agent_func_path /path/to/agent_func.py \
  --async_train  # Optional: enable async pipeline
```

### Configuration Options

**Async Pipeline** (for higher throughput):
- Enable: `--async_train`
- Buffer size: `--async_queue_size 1` (larger = more off-policy, default 1)

**Training Modes**:
- **Synchronous**: Default, better stability
- **Asynchronous**: Higher throughput, may affect convergence
- **Hybrid Engine**: Best GPU utilization with `--colocate_all_models` (remove `--async_train`)

> [!NOTE]
> For fully custom token-level execution, inherit `AgentExecutorBase` and implement `execute()`. This design enforces the **token-in-token-out principle** to keep sampling and training consistent.

> [!WARNING] 
> Asynchronous training may affect training stability. Use it only when throughput is critical and convergence is validated.

📚 **Examples**:
- Single-turn: [train_ppo_ray_hybrid_engine.sh](./examples/scripts/train_ppo_ray_hybrid_engine.sh)
- Custom reward: [train_ppo_with_reward_fn.sh](./examples/scripts/train_ppo_with_reward_fn.sh)
- Multi-turn: [train_reinforce_baseline_ray_agent_async.sh](./examples/scripts/train_reinforce_baseline_ray_agent_async.sh)
- NeMo Gym: `examples/python/agent_func_nemogym_executor.py`

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
    --param_dtype bf16
```

### Performance Tuning Guide

Optimize OpenRLHF for your hardware and workload with these recommendations:

#### 🎯 Resource Allocation (Distributed Mode)

**Recommended ratio**: `vLLM : Actor : Critic = 1:1:1`

```bash
# Example: 70B model on 48 A100 GPUs
# - 16 GPUs → vLLM Engine
# - 16 GPUs → Actor
# - 16 GPUs → Critic
```

#### ⚡ Speed Optimizations

| Optimization | Flag | When to Use |
|--------------|------|-------------|
| **Hybrid Engine** | `--colocate_all_models`<br>`--vllm_enable_sleep`<br>`--deepspeed_enable_sleep` | Sufficient GPU memory |
| **Async Training** | `--async_train` | Convergence validated, need throughput |
| **Sample Packing** | `--packing_samples` | Always (especially training) |
| **DeepCompile** | `--deepcompile` | PyTorch 2.0+ |
| **Overlap Comm** | `--overlap_comm` | Sufficient GPU memory |
| **Dynamic Batch** | `--use_dynamic_batch` | Variable sequence lengths |
| **Prefix Caching** | vLLM config | `n_samples_per_prompt` > 1 |

#### 💾 Memory Management

**When you have enough memory**:
- ✅ Disable `--adam_offload`
- ✅ Enable `--overlap_comm`
- ✅ Use `--colocate_critic_reward` and `--colocate_actor_ref`

**When hitting OOM**:
- ❌ Disable all `--colocate_*` options
- ✅ Reduce batch sizes
- ✅ Enable gradient checkpointing

#### 🎮 Batch Size Tuning

1. **Generation Phase**: Maximize `--micro_rollout_batch_size`, minimize vLLM TP size
2. **Training Phase**: Maximize `--micro_train_batch_size`, enable `--packing_samples`
3. **vLLM**: Always use `--vllm_sync_backend nccl`

> [!TIP]
> **Quick Start Template**: For 8x A100 (80GB), try Hybrid Engine + `--vllm_gpu_memory_utilization 0.5` + `--colocate_all_models`

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
@article{hu2025reinforce++,
  title={Reinforce++: A simple and efficient approach for aligning large language models},
  author={Hu, Jian},
  journal={arXiv preprint arXiv:2501.03262},
  year={2025}
}
```

______________________________________________________________________

*OpenRLHF © 2025 OpenRLHF. All Rights Reserved.*
