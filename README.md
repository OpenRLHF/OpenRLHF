# OpenLLaMA2

</br>

<h1 align="center">OpenLLaMA2</h1>
<div align="center">
<p align="center">
    <h3>A Ray-based High-performance RLHF framework!</h3>
      <a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
        <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/openllmai/OpenLLaMA2" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/openllmai/OpenLLaMA2?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/openllmai/OpenLLaMA2?color=0088ff" />
      </a>
      <a href="https://github.com/openllmai/OpenLLaMA2/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/openllmai/OpenLLaMA2?color=0088ff" />
      <a href="https://github.com/openllmai/OpenLLaMA2/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/openllmai/OpenLLaMA2?color=ccf" />
      </a>
      <br/>
      <em>Open-source ChatGPT / Comprehensive / Lightweight / Easy-to-use</em>
      <br/>
    </p>

</p>
</div>

> **The code is open-source, feel free to use it, contributions are welcome! Note: The license of the model depends on the provider of the model.**

- [💥Latest News](#latest-news)
- [💫OpenLLaMA2](#openllama2-project)
- [💫Features](#features)
- [💫Performance](#performance)
- [📄Running Example](#running-example)
- [⛏️Pull Request](#pull-request)
- [💐References & Acknowledgements](#references-&-acknowledgements)
- [🌟Sponsor Us](#sponsor-us)
- [🌈Starchart](#starchart)
- [🏆Contributors](#contributors)

## Latest News

- 2023/10/12: Support [Decision Transformer Alignment](./examples/scripts/train_decision_transformer_llama.sh).
- 2023/10/2: 34B codellama model with Ray-based RLHF + 1 DGX A100 test passed! Configs:
  - Adam Offload = True
  - micro_batch_size = 1
  - Enable FlashAttention2 to support 4096 seq length
- 2023/9/20: Support [Ray-based RLHF](./examples/scripts/train_ppo_llama_ray.sh)
- 2023/9/13: Upload [7B/13B SFT/RM/DPO/PPO checkpoint](https://huggingface.co/chuyi777/openllama2_checkpoint)
- 2023/9/9: Support [DPO (direct-preference-optimization)](./examples/scripts/train_dpo_llama.sh)
- 2023/9/6: Support FlashAttention2 (--flash_attn)
- 2023/8/26: Support wandb logs (--wandb)
- 2023/8/20: Upload PPO vs SFT <a href="./docs/ppo_examples.md">examples</a>
- 2023/8/18: Support **LLaMA2 7B PPO fintune on Single A100**
- 2023/8/13: LLaMA2 7B + SFT+ RM + PPO + DeepSpeed test passed
- 2023/07/30: OpenLLaMA2 project officially launched

## OpenLLaMA2 Project

OpenLLaMA2 aims to develop a **High-performance RLHF training framework** based on Ray and DeepSpeed/HuggingFace.

OpenLlaMA2 is the **Simplest** high-performance RLHF library that supports 34B models RLHF training with Single DGXA100.


### Features

- [✔️] A fast LLaMA2 SFT/PPO Training Framework based on DeepSpeed.
- [✔️] Multi-nodes training scripts for Slurm.
- [✔️] Wandb log.
- [✔️] Support conda env.
- [✔️] FlashAttention2.
- [✔️] Support [DPO (direct-preference-optimization)](./examples/scripts/train_dpo_llama.sh).
- [✔️] Distributed RLHF based on Ray (for 34B models).
- [✔️] [Decision Transformer Alignment](./examples/scripts/train_decision_transformer_llama.sh) (https://arxiv.org/abs/2308.12050).
- [WIP] Multiple Reward models.
- [WIP] Rejection Sampling.
- [WIP] Support QLora.
- [WIP] Develop the [RLHF datasets ↗](https://github.com/OpenLLMAI/OpenLLMData) for Multiple reward models.


Support Matrix


|        | PPO-max & Best Hyperparameters  | Ray (Distributed RL) | 34B Full Tuning with 1 DGXA100   | 7B Full Tuning with 1 A100  | Decision Transformer Alignment|
|  ----  | ----  |  ----  | ----  | ----  | ----  |  
| OpenLLaMA2  | ✔ | ✔  | ✔ | ✔ | ✔ |
| DeepSpeedChat  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |
| ColossalAIChat  | ✖️ | ✖️  | ✖️ |✖️ | ✖️ |
| TRL  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |

### Performance

Ray/DeepSpeed Config: 

4 A100 80G for Actor / 2 for Critic / 1 for Reward / 1 for InitPolicy + ZeRO2 + Adam Offload + Seq length: 2048 

- 7B llama2: 0.105 samples/gpu/secs (micro_batch_size = 16/8 [rollout/train]; generation_length = 100~300)
- 13B llama2: 0.04 samples/gpu/secs (micro_batch_size = 8/4 [rollout/train]; generation_length = 200~400)
- 34B codellama: 0.007 samples/gpu/secs (micro_batch_size = 2/1 [rollout/train]; generation_length = 300~800)

samples/gpu/secs = Number of PPO Samples / Number of A100 GPUS / Seconds

## Running Example

You can build openllama2 from **nvidia-docker(recomended)** or from conda envs.

```shell
Clone the repository: 
git clone https://github.com/openllmai/OpenLLaMA2.git

# Download the pre-trained SFT/RM checkpoints (Optional)
git lfs install
git clone https://huggingface.co/chuyi777/openllama2_checkpoint
```

* Single-node training with nvidia-docker

```shell
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh

# cd in container
cd /openllama2/examples/scripts

# build OpenLLaMA2 (i.e, pip install)
./build_openllama2.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh

# train DPO model
./train_dpo_llama.sh

# train Decision Transformer model
./train_decision_transformer_llama.sh
```

* PPO training with Ray (for 13B/34B models)

```shell
cd examples/scripts

# launch nvidia container
./docker_run.sh

# cd in container
cd /openllama2/examples/scripts

# build OpenLLaMA2 (i.e, pip install)
./build_openllama2.sh

# huggingface login 
export PATH=$HOME/.local/bin/:$PATH
~/.local/bin/huggingface-cli login

# launch ray in container
nohup ray start --head --node-ip-address 0.0.0.0 --num-cpus 128 --num-gpus 8 --block &> ray.log &

# if you want to launch ray on more nodes, use
# ray start --address {MASTER-NODE-ADDRESS}:6379 --num-cpus 128 --num-gpus 8 --block

# train ray PPO model, requires 8 gpus in default config
./train_ppo_llama_ray.sh
```

* Multi-nodes training on Slurm

```shell
cd examples/scripts

# huggingface login on Slurm 
pip install transformers
huggingface-cli login

# Moidfy the Slurm Account/Nodes ... in `train_llama_slurm.sh`

# For SFT, RM, and PPO training stage:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh
```

* build openllama2 from conda envs 

If you really don't want to use nvidia-docker, we also provide tutorials for building openllama2 from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n llama2 python=3.10
# so, we need install some package manualy: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip install torch --index-url https://download.pytorch.org/whl/cu118
# check ninjia
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
# For network error: you can download sepecified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.1.1 --no-build-isolation
./build_openllama2.sh
# enjoy it!
```

* Inference and Evaluation

After completing the training, you can evaluate your model by using the `inference` script:

```shell
./interactive_chat_llama.sh { model_path }
```

## Pull Request
If you want to contribute code please format the code using the following command,

```
pip install pre-commit
pre-commit install
git add .
git commit -m "xxx"
```

## References & Acknowledgements

We would like to express our gratitude to the following projects and organizations for their contributions to the field of AI and NLP:

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA2 ↗](https://ai.meta.com/llama/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)


### Join Us

**How to Join?**

1. Email us at xianyuai@openllmai.top(official email) or janhu9527@gmail.com/jjgxw@outlook.com(PIC). Please include the following details:
   - Your name
   - Your GitHub username
   - Your areas of interest
   - Your skills and experience related to NLP and/or AI
1. You can also join us through the official GitHub [OpenLLaMA2 ↗](https://github.com/openllmai/OpenLLaMA2) project page. Just create an issue about your interest to contribute and we will get back to you.

**What can you do?**

1. Join the team and participate in the development of the OpenLLaMA2 project.
1. Contribute to the project by submitting pull requests.
1. Help improve documentation, fix bugs, or create new features.
1. Share the project and help us grow the community.

## Sponsor Us

Your sponsorship can help us maintain and improve OpenLLaMA2. If you find this project useful, please consider sponsoring us. You can sponsor us on [Open Collective ↗](https://opencollective.com/openllmai).

## Starchart


[![Star History Chart](https://api.star-history.com/svg?repos=openllmai/OpenLLaMA2&type=Date)](https://star-history.com/#openllmai/OpenLLaMA2&Date)

## Contributors

A big thank you to all our contributors! If you want to contribute, feel free to make a pull request or create an issue.

<a href="https://github.com/openllmai/OpenLLaMA2/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openllmai/OpenLLaMA2" />
</a>

## Citation
```
@misc{openllmai23,
   author = {OpenLLMAI},
   title = {OpenLLaMA2},
   year={2023},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenLLaMA2}}
}
```

______________________________________________________________________

*OpenLLaMA2 © 2023 OpenLLMAI. All Rights Reserved.*
