<div align="center">
    <img alt="OpenRLHF logo" src="./docs/logo.png" style="height: 160px;" />
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
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

<span>[ English | <a href="README_zh.md">中文</a> ]</span>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, and compatible with Huggingface models and datasets.
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Adam Offload (Pinned Memory) and vLLM generation acceleration, the performance of OpenRLHF 2x+ that of Optimized DeepSpeedChat with Hybrid Engine.
- **Distributed RLHF**:  OpenRLHF distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 70B+ models with multiple A100 80G GPUs and vLLM (see [architecture](./docs/ray_architecture.png)) and 7B models across multiple 24GB RTX 4090 GPUs.
- **PPO Implementation Tricks**: We integrated the implementation tricks for PPO to improve the training stability, referencing the [Notion AI blog](https://difficult-link-dd7.notion.site/eb7b2d1891f44b3a84e7396d19d39e6f?v=01bcb084210149488d730064cbabc99f).


## Features

- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- Support full RLHF fine-tuning of models with [over 70 billion parameters](./examples/scripts/train_ppo_llama_ray_70b.sh).
- Support vLLM generation acceleration in RLHF (--vllm_num_engines).
- Support multiple reward models (--reward_pretrain model1,model2...).
- Support [DPO (direct-preference-optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- Support [Kahneman-Tversky optimization (KTO)](./examples/scripts/train_kto_llama.sh).
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).
- Support [Iterative DPO](./examples/scripts/train_iterative_dpo_llama.sh) (https://github.com/RLHFlow/Online-RLHF).
- Support [Conditional SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [Mixtral 8*7b](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- Support Wandb log (--wandb).
- Support FlashAttention2 (--flash_attn).
- Support QLoRA (--load_in_4bit), LoRA (--lora_rank, --target_modules).
- Support HuggingFace `tokenizer.apply_chat_template` in datasets ([--apply_chat_template and --input_key](./examples/scripts/train_ppo_llama3_ray_colocate.sh)).
- Multi-nodes [training scripts](./examples/scripts/train_llama_slurm.sh) for Slurm.

**TODO** 
- Allows saving and loading training checkpoints.

**PPO Support Matrix** 

| Feature | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:|
| 70B+ Full Tuning with 16 A100-80GB      | ✅ | ❌ | ❌ | ❌ |
| 7B Full Tuning with 4 RTX4090 | ✅      |    ❌ | ❌ | ❌ |
| 34B DPO Full Tuning with 8 A100-80GB | ✅      |    ❌ | ❌ | ❌ |  
| Inference Engine in PPO | ✅      |    ✅ | ❌ | ❌ |  
| PPO Implementation Tricks | ✅      |    ❌ | ❌ | ✅ |
| Support QLoRA | ✅      |    ❌ | ❌ | ✅ | 
| Support Mixtral 8*7b | ✅      |    ❌ | ❌ | ❌ |  
| Support Unmerged Actor-Critic | ✅     |   ✅ | ✅ | ❌ | 
| Support Multiple Reward Models | ✅      |    ❌ | ❌ | ❌ |   
| Support Huggingface Models | ✅      |    ✅ | ✅ | ✅ | 
| Easy-to-use | ✅      |   ❌ (HybridEngine bugs) | ✅ | ✅ | 


## Performance

We optimized DSChat's performance to the greatest extent possible by employing techniques such as enabling Adam offload, along with reward model (RM) and reference model (Ref) offload to increase the micro-batch size during the inference stage and avoid out-of-memory issues. We even fixed some bugs in DSChat to enable the Hybrid Engine (HE) for LLaMA2. The average time (seconds) it took to train 1024 prompts with 1 PPO epoch using the Optimized DSChat and OpenRLHF:

| **Size** | **NVIDIA A800-80GB GPUs** | **Optimized DSChat (with  Hybrid Engine)** | **OpenRLHF** | **Speedup** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

## Performance Tuning Guide

To achieve optimal performance, we recommend allocating more nodes to the vLLM Engine. For example, for a 70B model with 32 A100 GPUs, it is advised to allocate more than 16 A100 GPUs to the vLLM Engine, 8 GPUs to the Actor model, and the remaining 8 GPUs to the Critic model. Additionally, enable the `--colocate_critic_reward`, `--colocate_actor_ref`, and `--ref_reward_offload` options to merge nodes. You can refer to the script [Llama3 Ray PPO](./examples/scripts/train_ppo_llama3_ray_colocate.sh) for more details. Finally, you should increase the micro-batch-size (and minimize the TP size of vLLM engine) as much as possible while avoiding OOM (Out Of Memory) issues, especially during the generation phase of PPO.


## Running Example
> [!IMPORTANT]
> You can build openrlhf from **nvidia-docker(recommended)** or from conda envs.

```shell
# Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git
```

**install nvidia-docker and OpenRLHF**
  
```bash
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# build nvidia container with vLLM (Recommended)
./docker_run.sh build

# run nvidia container
./docker_run.sh

# cd in nvidia container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
huggingface-cli login

# wandb login (Optional, also set --wandb True in script)
wandb.login()

```

**Single-node training**

```shell
# Continue Pre-training
./train_continue_pretrain_llama.sh

# Supervised Finetuning
./train_sft_llama.sh

# Reward Model Tuning
./train_rm_llama.sh

# PPO Training
./train_ppo_llama.sh

# DPO
./train_dpo_llama.sh

# KTO
./train_kto_llama.sh

# Rejection Sampling with vLLM
./train_rejection_sampling_llama.sh

# Iterative DPO with vLLM
./train_iterative_dpo_llama.sh

# Conditional SFT
./train_conditional_llama.sh
```

**PPO training with Ray**
> [!TIP]
> for >= 13B models on V100/A100/H100.. or 7B models on RTX4090

```bash
# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

# Ray PPO training, requires 8 GPUs in default config
./train_ppo_llama_ray.sh

# for 70B models
# Launch Ray PPO with vLLM, requires 16 A100s in default config
./train_ppo_llama_ray_70b.sh
```
> [!NOTE]
> We recommend using vLLM 0.4.2, as versions 0.4.3+ currently only support weight synchronization (DeepSpeed => vLLM) via GLOO (--vllm_sync_backend gloo).

**Multi-nodes training on Slurm**

```bash
cd examples/scripts

# Moidfy the Slurm Account/Nodes ... in `train_llama_slurm.sh`

# For SFT, RM, PPO, DPO, KTO training:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"
readonly training_script="train_dpo_llama.sh"
readonly training_script="train_kto_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh

# for Ray PPO training with Slurm
sbatch ./train_ppo_llama_ray_slurm.sh
```

**Inference and Evaluation**

After completing the training, you can evaluate your model by using the `inference` script:

```bash
# batch generate
# support vLLM acceleration (--eval_task generate_vllm)
python examples/batch_inference.py {args}

# interactive_chat
python examples/interactive_chat.py --bf16 --pretrain { pretrain_model_path }
```

**build openrlhf from conda envs**

If you really don't want to use nvidia-docker, we also provide tutorials for building openrlhf from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n openrlhf python=3.10
# so, we need install some package manually: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip3 install torch
# check ninjia
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
pip install flash-attn==2.5.8
./build_openrlhf.sh
# enjoy it!
```

## Join Us

**How to Join?**

1. Email us at xianyuai@openllmai.top(official email) or janhu9527@gmail.com/jjgxw@outlook.com(PIC). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenRLHF ↗](https://github.com/openllmai/OpenRLHF) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenRLHF project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenRLHF. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenRLHF&type=Date)](https://star-history.com/#openllmai/OpenRLHF&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenRLHF" />
</a>

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

Our project would also like to thank [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat) and [DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat). In the early stages of the project, we referred to their code design.


## Citation
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

*OpenRLHF © 2024 OpenLLMAI. All Rights Reserved.*
