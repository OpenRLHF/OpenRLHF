<div align="center">
<p align="center">
<img alt="" src="./docs/logo.png" style="display: inline-block; height: 140px" />
</p>
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
      <a href="https://github.com/OpenRLHF/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>开源 / 全面 / 轻量级 / 易用</em>
    </p>
</p>
</div>

<hr>

<span>[ <a href="README.md">English</a> | 中文 | <a href="README_ja.md">日本語</a> ]</span>

OpenRLHF 是一个基于 Ray、DeepSpeed 和 HF Transformers 构建的高性能 RLHF 框架：

- **简单易用**: OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，无缝兼容 Huggingface 模型和数据集。
- **高性能**: RLHF 训练中 80% 的时间用于样本生成阶段。得益于使用 Ray, Packing Samples 以及 vLLM 生成加速的能力，OpenRLHF 的性能是极致优化的 DeepSpeedChat with Hybrid Engine 的3~4倍以上。
- **分布式 RLHF**:  OpenRLHF 使用 Ray 将 Actor、Reward、Reference 和 Critic 模型分布到不同的 GPU 上，同时将 Adam 优化器放在 CPU 上。这使得使用多个 A100 80G GPU 和 vLLM 可以全面微调超过 70B+ 的模型 以及在多个 24GB RTX 4090 GPU 上微调 7B 模型。
- **Hybrid Engine**: OpenRLHF 还支持还支持 Hybrid engine 所有训练引擎和推理引擎共用GPU来避免资源闲置。
- **PPO 实现技巧**: 我们集成了 PPO 的实现技巧以提高训练稳定性，详情参考 [知乎](https://zhuanlan.zhihu.com/p/622134699) 和 [Notion blog](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361).

更多细节请参考 [PPT](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [技术报告](https://arxiv.org/abs/2405.11143) | [使用文档](https://openrlhf.readthedocs.io/)


## 新闻  
- [2025/2] MIT & Microsoft 提出了 [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) 基于 OpenRLHF
- [2025/1] 港科大复现了 [DeepSeek-R1-Zero and DeepSeek-R1 training on small models 使用 OpenRLHF](https://github.com/hkust-nlp/simpleRL-reason)
- [2024/12] 我们"提出"了 😊 [REINFORCE++ 对齐算法](https://www.researchgate.net/publication/387487679_REINFORCE_A_SIMPLE_AND_EFFICIENT_APPROACH_FOR_ALIGNING_LARGE_LANGUAGE_MODELS).
- [2024/12] 在 [Notion Blog](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05) 中，我们对 PPO、REINFORCE++、GRPO 和 RLOO 进行了分析。  

## 特性  

- 基于 Ray 的分布式 [PPO](./examples/scripts/train_ppo_llama_ray.sh) 和 [REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray.sh) 实现。  
- 支持对 [超过 700 亿参数的模型](./examples/scripts/train_ppo_llama_ray_70b.sh) 进行完整的 RLHF 微调。  
- 支持基于 Ray 和 Hybrid Engine 的 [PPO](./examples/scripts/train_ppo_llama_ray_hybrid_engine.sh) 和 [REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray_hybrid_engine.sh) (`--colocate_all_models`, `--vllm_enable_sleep` and `--vllm_gpu_memory_utilization 0.5`)
- [Ray-based Reinforced Finetuning](./examples/scripts/train_ppo_llama_with_reward_fn.sh)
- 集成 vLLM，加速 RLHF 任务中的样本生成（`--vllm_num_engines`）。  
- 支持多个奖励模型（`--reward_pretrain model1,model2...`）和远程奖励模型（`--remote_rm_url`）。  
- 实现 [DPO（直接偏好优化）/IPO/cDPO](./examples/scripts/train_dpo_llama.sh) 和 [Kahneman-Tversky Optimization（KTO）](./examples/scripts/train_kto_llama.sh)。  
- 支持 [迭代 DPO](./examples/scripts/train_iterative_dpo_llama.sh)（[GitHub: Online-RLHF](https://github.com/RLHFlow/Online-RLHF)）。  
- 支持 [拒绝采样](./examples/scripts/train_rejection_sampling_llama.sh)。  
- 实现 [条件 SFT](./examples/scripts/train_conditional_llama.sh)（[arXiv:2308.12050](https://arxiv.org/abs/2308.12050)）。  
- 支持 [知识蒸馏](./examples/scripts/train_knowledge_distillation.sh)（[Microsoft: minillm](https://github.com/microsoft/LMOps/tree/main/minillm)）。  
- 集成 [过程奖励模型（PRM）](./examples/scripts/train_prm_mistral.sh)。  
- 支持 SFT、DPO、RM、PRM 和 PPO 的训练样本打包（`--packing_samples`）。  
- 实现 [RingAttention](./examples/scripts/train_dpo_ring_llama.sh)（`--ring_attn_size`，`--ring_head_stride`）。  
- 支持 [专家混合模型（MoE）](./examples/test_scripts/train_sft_mixtral_lora.sh)（`--aux_loss_coef`）。  
- 集成 FlashAttention2（`--flash_attn`）。  
- 支持 QLoRA（`--load_in_4bit`）和 [LoRA](./examples/scripts/train_sft_mixtral_lora.sh)（`--lora_rank`，`--target_modules`）。  
- 兼容 HuggingFace 的 `tokenizer.apply_chat_template` 数据集格式（`--apply_chat_template` 和 `--input_key`）。  
- 支持使用 Wandb（`--use_wandb`）和 TensorBoard（`--use_tensorboard`）进行日志记录。  
- 支持从检查点恢复训练（`--load_checkpoint` 和 `--save_steps`）。  
- 提供了多节点训练脚本, 比如 [DPO](./examples/scripts/train_llama_slurm.sh) 和 [RLHF](./examples/scripts/train_ppo_llama_ray_slurm.sh)


### PPO 支持矩阵

| 特性 | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
| 使用 16 个 A100 完成 70B+ 全微调      | ✅ | ❌ | ❌ | ❌ ||
| 使用 4 个 RTX4090 完成 7B 全微调 | ✅      |    ❌ | ❌ | ❌ | 
| 使用 8 个 A100 完成 34B DPO 全微调 | ✅      |    ❌ | ❌ | ❌ |   
| 支持推理引擎加速 | ✅      |    ✅ | ❌ | ❌ |  
| PPO 实现技巧 | ✅      |    ❌ | ❌ | ✅ | 
| 支持 QLoRA | ✅      |    ❌ | ❌ | ✅ | 
| 支持 Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ | 
| 支持未合并的 Actor-Critic | ✅     |   ✅ | ✅ | ❌ | 
| 支持多个奖励模型 | ✅      |    ❌ | ❌ | ❌ |   
| 支持 Huggingface 模型 | ✅      |    ✅ | ✅ | ✅ | 
| 易于使用 | ✅      |   ❌ (HybridEngine bugs) | ✅ | ✅ | 

## 快速开始

### 安装

要使用 OpenRLHF，首先启动 Docker 容器（**推荐**）然后执行 `pip install` 安装 `openrlhf`：

```bash
# 启动 docker container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
sudo pip uninstall xgboost transformer_engine flash_attn pynvml -y

# pip install
pip install openrlhf

# 如果你需要使用 vLLM 加速 (安装 vLLM 0.7.2)
pip install openrlhf[vllm]
# 最新的 vLLM 也是支持的
pip install openrlhf[vllm_latest]

# pip install GitHub 上的最新版
pip install git+https://github.com/OpenRLHF/OpenRLHF.git

# 或者 git clone
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!NOTE]
>我们推荐使用 vLLM 0.7.2 及以上版本。
>我们也提供了 [Dockerfiles for vLLM](./dockerfile/) 和 [Nvidia-Docker 一键安装脚本](./examples/scripts/nvidia_docker_install.sh)。

### 准备数据集
OpenRLHF 在其数据集类中提供了多种数据处理方法。
例如在 [Prompt Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6) 中：

```python
def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt
```

- 我们可以使用 `--input_key` 指定 `JSON key name` 为输入数据集 `--prompt_data {name or path}` (PPO) 或 `--dataset {name or path}`，并使用 `--apply_chat_template` 利用 [Huggingface Tokenizer](https://huggingface.co/docs/transformers/main/en/chat_templating) 中的 `chat_template`。
- 如果不想使用 `--apply_chat_template`，可以改用 `--input_template`，或预先离线处理数据集。
- OpenRLHF 还支持使用 `--prompt_data_probs 0.1,0.4,0.5` (PPO) 或 `--dataset_probs 0.1,0.4,0.5` 混合多个数据集。

Chat Templating 的工作原理如下:

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

如何指定训练和测试数据分区 ?

你可以使用 `data_type@data_dir` 的方式指定, 比如下面的数据集可以设置为 `--dataset json@./data`

```
data
├── test.jsonl
└── train.jsonl
```

> [!NOTE]
>默认情况下我们使用 `train` 和 `test` 作为 split 区分 Huggingface 的训练/测试数据。
>`JSON key` 选项取决于具体的数据集。请参阅 [Reward Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10) 和 [SFT Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/mai


### Supervised Fine-tuning

OpenRLHF 的模型检查点完全兼容 HuggingFace 模型。您可以使用 `--pretrain  {name or path}`、`--reward_pretrain  {name or path}` 和 `--critic_pretrain  {name or path}` 指定模型名称或路径。我们在 [HuggingFace OpenRLHF](https://huggingface.co/OpenRLHF) 上提供了一些预训练的检查点和数据集。

然后您可以使用我们在 [examples/scripts](./examples/scripts/) 目录中提供的启动脚本，或者使用以下命令启动训练：

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
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples \
   --load_checkpoint \
   --use_wandb {wandb_token}

# 支持 HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# 支持 RingAttention
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# 也可用于 continued pre-training
# --pretrain_mode
```

> [!NOTE]
> OpenRLHF SFT/DPO/RewardModel/PPO 训练支持 `--packing_samples` [基于 `--flash_attn`](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)

### Reward Model Training
```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb {wandb_token}

```

推荐设置 Reward Model 的 `--value_prefix_head` 选项为 `score`, 这样使得我们可以用 `AutoModelForSequenceClassification` 加载模型:

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

### 不使用 Ray 的 PPO

```bash
deepspeed --module openrlhf.cli.train_ppo \
  --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
  --save_path ./checkpoint/llama-3-8b-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --load_checkpoint \
  --use_wandb {wandb_token}

# 支持远程 reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward
```

### 使用 Ray 和 vLLM 的 PPO/REINFORCE++

为了提高 RLHF 训练速度或支持 70B 模型，我们可以使用 Ray 和 vLLM 加速的 PPO

```bash
# 在容器中启动 Ray 的主节点
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# 如果要在更多节点上启动 Ray，请使用
ray start --address {MASTER-NODE-ADDRESS}:6379 --num-gpus 8

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
  --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 32 \
  --rollout_batch_size 1024 \
  --max_samples 100000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --load_checkpoint \
  --use_wandb {wandb_token}

# 支持 REINFORCE++  | RLOO  | REINFORCE++-baseline
# --advantage_estimator reinforce | rloo | reinforce_baseline

# 支持远程 reward model (HTTP)
# --remote_rm_url http://localhost:5000/get_reward

# 支持 N 倍采样
# --n_samples_per_prompt 4
```

> [!NOTE]
> 不设置 `--vllm_num_engines` 则是不使用 vLLM engine。
> 您也可以通过 ``setup_commands`` 让 Ray 自动初始化环境, 比如 `--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`

> [!NOTE]
> OpenRLHF 中的 RLOO 和 REINFORCE++-baseline 是基于 REINFORCE++ 的修改版本:
> - REINFORCE++ 在 REINFORCE 的基础上集成了 PPO 的关键优化技术，所以不依赖于 Value Network
> - REINFORCE++-baseline 使用了相同 prompt 多次采样的 reward 均值作为 baseline
> - OpenRLHF 中的 RLOO 在原版的基础上采用了 **per token KL reward** 和 **PPO-clip loss** 


> [!NOTE]
> 如果您由于某种原因，在 deepspeed 设置显卡设备时遇到与索引超出范围相关的错误，您可以尝试设置环境变量 [`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`](openrlhf/trainer/ray/utils.py)。
> ```bash
> # 对于 NVIDIA 显卡:
> export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
> ```

所有支持算法的启动脚本和文档在 [example/scripts](./examples/scripts/) 和 [Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)


### 使用 LoRA
如果您使用了 `LoRA (Low-Rank Adaptation)`，默认保存下来的文件**并非**完整模型权重，而是 `LoRA Adapter`，若想按完整权重的方式进行后续任务，您需要将 `Adapter` 与训练前的模型权重进行合并

```bash
python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama3-8b-rm \
    --output_path ./checkpoint/llama-3-8b-rm-combined \
    --is_rm \
    --bf16
```

## 性能
我们通过启用Adam卸载、奖励模型(RM)和参考模型(Ref)卸载等技术,尽可能优化了DSChat的性能,从而在推理阶段增加小批量大小并避免内存不足问题。我们甚至修复了DSChat中的一些bug,以启用LLaMA2的混合引擎(HE)。使用优化后的DSChat和OpenRLHF训练1024个提示需要1个PPO轮次的平均时间(秒)如下:

| **Size** | **NVIDIA A800 GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |


> [!NOTE]
> 数据已经过时; 请参考后面的调优指南重新测试

## 调优指南
为了获得最佳的性能，我们建议您分配节点为 `vLLM:Actor:Critic = 1:1:1`。例如，对于 70B 模型以及 48 张 A100，建议分配 16 张以上 A100 给 vLLM Engine，16 张给 Actor 模型，以及最后 16 张给 Critic 模型，同时开启 `--colocate_critic_reward`, `--colocate_actor_ref` 或者 `--ref_reward_offload (可选)` 选项合并部分节点。最后您应该尽可能增大 `--rollout_micro_batch_size` ，以及减小 vLLM 的 TP 切分数量。训练阶段的 `micro_train_batch_size` 也是越大越好，请同时使用 `--packing_samples` 。当 GPU 数量足够时请关闭 `--adam_offload` 以及启用 `--overlap_comm`. 对于多节点 RLHF, 请使用 `--vllm_sync_backend nccl` with vLLM 0.7.2+. 启用 `enable_prefix_caching` 对于 vLLM 当 ``n_samples_per_prompts`` > 1. 当模型规模和上下文长度较小时，使用 Hybrid Engine `--colocate_all_models` 和 `--vllm_enable_sleep`，而不是分布式 RLHF。对于尺寸大的基础模型, 如果出现 OOM, 请不要使用任何`--colocate_xxxx`选项。

## 使用 OpenRLHF 的公司和组织

- Google
- ByteDance
- Tencent
- Alibaba
- Baidu
- China Telecom
- Allen AI
- Vivo
- NexusFlow
- Jülich Supercomputing Centre (JSC)
- Berkeley Starling Team
- M-A-P
- ...


## 加入我们

**如何加入？**

1. 通过联系邮箱 janhu9527@gmail.com 或者加入 [GitHub Organization](https://github.com/OpenRLHF)。请包含以下信息：
   - 您的姓名
   - 您的 GitHub 用户名
   - 您感兴趣的领域
   - 您在 NLP 和/或 AI 相关的技能和经验
2. 您也可以通过官方 GitHub [OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF) 项目页面加入我们。只需创建一个关于您想要贡献的兴趣的 issue，我们会与您联系。

**您能做什么？**

1. 加入团队，参与 OpenRLHF 项目的开发。
2. 通过提交 pull 请求来为项目做出贡献。
3. 帮助改进文档，修复 bugs 或创建新功能。
4. 分享项目并帮助我们发展社区。

## 赞助我们

您的赞助可以帮助我们维护和改进 OpenRLHF。如果您觉得这个项目有用，请考虑赞助我们。您可以在 [Open Collective ↗](https://opencollective.com/OpenRLHF) 上赞助我们。

## 星图

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRLHF/OpenRLHF&type=Date)](https://star-history.com/#OpenRLHF/OpenRLHF&Date)

## 贡献者

非常感谢所有的贡献者！如果您想贡献，请随时创建 pull 请求或创建 issue。

<a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRLHF/OpenRLHF" />
</a>

## 引用与致谢

我们想要对以下项目和组织在 AI 和 NLP 领域的贡献表示感谢：

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA ↗](https://llama.meta.com/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

我们的项目还想要感谢 [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) 和 [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)。在项目的早期阶段，我们参考了他们的代码设计。

(2024/7) 我们的 GitHub 组织从 OpenLLMAI 迁移到了 OpenRLHF.

## 引用
```
@article{hu2024openrlhf,
  title={OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
  author={Jian Hu and Xibin Wu and Zilin Zhu and Xianyu and Weixun Wang and Dehao Zhang and Yu Cao},
  journal={arXiv preprint arXiv:2405.11143},
  year={2024}
}
```


______________________________________________________________________

*OpenRLHF © 2025 OpenRLHF. 版权所有。*
