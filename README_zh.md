<div align="center">
<p align="center">
<img alt="" src="./docs/logo.png" style="display: inline-block; height: 160px" />
</p>
</div>

<div align="center">
<p align="center">
      <a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenRLHF" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenRLHF?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenRLHF/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenRLHF?color=0088ff" />
      <a href="https://github.com/openllmai/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>开源 / 全面 / 轻量级 / 易用</em>
    </p>
</p>
</div>

<hr>

<span>[ <a href="README.md">English</a> | 中文 ]</span>

OpenRLHF 是一个基于 Ray、DeepSpeed 和 HF Transformers 构建的高性能 RLHF 框架：

- **简单易用**: OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，兼容 Huggingface 模型和数据集。
- **高性能**: RLHF 训练中 80% 的时间用于样本生成阶段。得益于使用 Ray 和 Adam Offload（固定内存）以及 vLLM 生成加速的能力，OpenRLHF 的性能是极致优化的 DeepSpeedChat with Hybrid Engine 的两倍以上。
- **分布式 RLHF**:  OpenRLHF 使用 Ray 将 Actor、Reward、Reference 和 Critic 模型分布到不同的 GPU 上，同时将 Adam 优化器放在 CPU 上。这使得使用多个 A100 80G GPU 和 vLLM 可以全面微调超过 70B+ 的模型 (见 [architecture](./docs/ray_architecture.png)) 以及在多个 24GB RTX 4090 GPU 上微调 7B 模型。
- **PPO 实现技巧**: 我们集成了 PPO 的实现技巧以提高训练稳定性，参考 [Notion AI blog](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f).


## 特性

- 基于 Ray 的分布式 [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- 支持使用 [超过 70 亿参数的模型进行全面 RLHF 微调](./examples/scripts/train_ppo_llama_ray_70b.sh).
- 支持 vLLM 生成加速在 RLHF 中 (--vllm_num_engines).
- 支持多个奖励模型 (--reward_pretrain model1,model2...).
- 支持 [DPO (直接偏好优化)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- 支持 [Kahneman-Tversky 优化 (KTO)](./examples/scripts/train_kto_llama.sh).
- 支持 [拒绝采样](./examples/scripts/train_rejection_sampling_llama.sh).
- 支持 [Iterative DPO](./examples/scripts/train_iterative_dpo_llama.sh) (https://github.com/RLHFlow/Online-RLHF).
- 支持 [条件 SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- 支持 [Mixtral 8*7b](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- 支持 Wandb 日志 (--wandb).
- 支持 FlashAttention2 (--flash_attn).
- 支持 QLoRA (--load_in_4bit), LoRA (--lora_rank, --target_modules).
- 支持 HuggingFace `tokenizer.apply_chat_template` 用于数据集处理 ([--apply_chat_template and --input_key](./examples/scripts/train_ppo_llama3_ray_colocate.sh)).
- 多节点 [训练脚本](./examples/scripts/train_llama_slurm.sh) 适用于 Slurm.

**待办事项** 
- 允许保存和加载训练检查点。

**PPO 支持矩阵**

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


## 性能
我们通过启用Adam卸载、奖励模型(RM)和参考模型(Ref)卸载等技术,尽可能优化了DSChat的性能,从而在推理阶段增加小批量大小并避免内存不足问题。我们甚至修复了DSChat中的一些bug,以启用LLaMA2的混合引擎(HE)。使用优化后的DSChat和OpenRLHF训练1024个提示需要1个PPO轮次的平均时间(秒)如下:

| **Size** | **NVIDIA A800 GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

## 调优指南
为了获得最佳的性能，我们建议您分配更多的节点给 vLLM Engine。例如，对于 70B 模型以及 32 张 A100，建议分配 16 张以上 A100 给 vLLM Engine，8 张给 Actor 模型，以及最后 8 张给 Critic 模型，同时开启 `--colocate_critic_reward`, `--colocate_actor_ref` 和 `--ref_reward_offload` 选项合并部分节点，可参考脚本 [Llama3 Ray PPO](./examples/scripts/train_ppo_llama3_ray_colocate.sh)。最后您应该尽可能在避免 OOM 的前提下增大 micro-batch-size (以及减小 vLLM 的 TP 切分数量) 尤其对于 PPO 样本推理生成阶段。


## 运行示例
> [!IMPORTANT]
> 您可以通过 **nvidia-docker(推荐)** 或 conda 环境构建 openrlhf。

```shell
# 克隆仓库: 
git clone https://github.com/openllmai/OpenRLHF.git
```

**安装 nvidia-docker 和 OpenRLHF**
  
```bash
cd examples/scripts

# 安装 nvidia-docker（可选）
./nvidia_docker_install.sh

# 构建 vLLM nvidia 容器（推荐）
./docker_run.sh build

# 运行 nvidia 容器
./docker_run.sh

# 进入 nvidia 容器
cd /openrlhf/examples/scripts

# 构建 OpenRLHF（例如，pip 安装）
./build_openrlhf.sh

# 登录 huggingface
huggingface-cli login

# 登录 wandb（可选，同时在脚本中设置 --wandb True）
wandb.login()

```
**单节点训练**

```shell
# 继续预训练
./train_continue_pretrain_llama.sh

# 监督式微调
./train_sft_llama.sh

# 奖励模型调整
./train_rm_llama.sh

# PPO 训练
./train_ppo_llama.sh

# DPO 训练
./train_dpo_llama.sh

# KTO 
./train_kto_llama.sh

# 使用 vLLM 的拒绝采样训练
./train_rejection_sampling_llama.sh

# Iterative DPO with vLLM
./train_iterative_dpo_llama.sh

# 条件 SFT
./train_conditional_llama.sh
```

**使用Ray进行PPO训练**
> [!TIP]
> 适用于V100/A100/H100的13B以上模型或RTX4090上的7B模型

```bash
# 在容器中启动 ray 的主节点
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# 如果你想在更多节点上启动 ray，请使用
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

# Ray PPO 训练，默认配置需要 8 个 GPU
./train_ppo_llama_ray.sh

# 对于 70B 模型
# 启动使用 vLLM 的 Ray PPO，默认配置需要 16 个 A100
./train_ppo_llama_ray_70b.sh
```

> [!NOTE]
> 我们建议使用 vLLM 0.4.2，因为 0.4.3 及以上版本目前只能通过 GLOO (--vllm_sync_backend gloo) 进行权重同步 (DeepSpeed => vLLM)。

**在 Slurm 上进行多节点训练**

```bash
cd examples/scripts

# 修改 `train_llama_slurm.sh` 中的 Slurm 账户/节点等信息

# 对于 SFT、RM、PPO、DPO、KTO 训练：
# 在 `train_llama_slurm.sh` 中修改变量 `training_script` 
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"
readonly training_script="train_dpo_llama.sh"
readonly training_script="train_kto_llama.sh"

# 在 `train_llama_slurm.sh` 中设置 `GPUS_PER_NODE`
readonly GPUS_PER_NODE=8

# 运行多节点训练脚本
# `train_llama_slurm.sh` 将从 `training_script` 加载训练参数
sbatch ./train_llama_slurm.sh

# 对于使用 Slurm 的 Ray PPO 训练
sbatch ./train_ppo_llama_ray_slurm.sh
```

**推理和评估**

完成训练后，您可以使用 `inference` 脚本评估您的模型：



```bash
# 批量生成
# 支持 vLLM 加速（--eval_task generate_vllm）
python examples/batch_inference.py {args}

# 交互式聊天
python examples/interactive_chat.py --bf16 --pretrain { pretrain_model_path }
```

**从 conda 环境构建 openrlhf**

如果您真的不想使用 nvidia-docker，我们还提供了从 conda 环境构建 openrlhf 的教程。（我们更推荐使用 nvidia-docker，以避免环境引起的错误。）
```shell
# 我们需要 conda
conda create -n openrlhf python=3.10
# 因此，我们需要手动安装一些包：安装 torch 时，您可能需要匹配相应的 cuda 版本。
pip install packaging ninja
pip3 install torch
# 检查 ninja
ninja --version
echo $? # output: 0
# 安装 flash-attn：可能需要一些时间。
pip install flash-attn==2.5.8
./build_openrlhf.sh
# 享受它！
```

## 加入我们

**如何加入？**

1. 通过官方邮箱 xianyuai@openllmai.top 或个人联系邮箱 janhu9527@gmail.com/jjgxw@outlook.com 向我们发送邮件。请包含以下信息：
   - 您的姓名
   - 您的 GitHub 用户名
   - 您感兴趣的领域
   - 您在 NLP 和/或 AI 相关的技能和经验
2. 您也可以通过官方 GitHub [OpenRLHF ↗](https://github.com/openllmai/OpenRLHF) 项目页面加入我们。只需创建一个关于您想要贡献的兴趣的 issue，我们会与您联系。

**您能做什么？**

1. 加入团队，参与 OpenRLHF 项目的开发。
2. 通过提交 pull 请求来为项目做出贡献。
3. 帮助改进文档，修复 bugs 或创建新功能。
4. 分享项目并帮助我们发展社区。

## 赞助我们

您的赞助可以帮助我们维护和改进 OpenRLHF。如果您觉得这个项目有用，请考虑赞助我们。您可以在 [Open Collective ↗](https://opencollective.com/openllmai) 上赞助我们。

## 星图

[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenRLHF&type=Date)](https://star-history.com/#openllmai/OpenRLHF&Date)

## 贡献者

非常感谢所有的贡献者！如果您想贡献，请随时创建 pull 请求或创建 issue。

<a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenRLHF" />
</a>

## 引用与致谢

我们想要对以下项目和组织在 AI 和 NLP 领域的贡献表示感谢：

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

我们的项目还想要感谢 [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) 和 [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)。在项目的早期阶段，我们参考了他们的代码设计。

## 引用
```
@misc{hu23openrlhf,
author = {Jian Hu and Xibin Wu and Xianyu and Chen Su and Leon Qiu and Daoning Jiang and Qing Wang and Weixun Wang},
title = {OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework},
year={2023},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/OpenLLMAI/OpenRLHF}}
}
```


______________________________________________________________________

*OpenRLHF © 2024 OpenLLMAI. 版权所有。*
