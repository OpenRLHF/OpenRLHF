# OpenRLHF

</br>

<h1 align="center">OpenRLHF</h1>
<div align="center">
<p align="center">
    <h3>A Ray-based High-performance RLHF framework!</h3>
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
      <br/>
      <em>Open-source ChatGPT / Comprehensive / Lightweight / Easy-to-use</em>
      <br/>
    </p>

</p>
</div>

> **The code is open-source, feel free to use it, contributions are welcome! Note: The license of the model depends on the provider of the model.**

- [💫OpenRLHF](#openrlhf-project)
- [💫Features](#features)
- [💫Performance](#performance)
- [📄Running Example](#running-example)
- [⛏️Pull Request](#pull-request)
- [💐References & Acknowledgements](#references-&-acknowledgements)
- [🌟Sponsor Us](#sponsor-us)
- [🌈Starchart](#starchart)
- [🏆Contributors](#contributors)

## OpenRLHF Project

OpenRLHF aims to develop a **High-performance RLHF training framework** based on Ray and DeepSpeed. OpenRLHF is the **Simplest** high-performance RLHF library that supports 34B models RLHF training with Single DGXA100 ([script](./examples/scripts/train_ppo_llama_ray_34b.sh)).

The key idea of OpenRLHF is to distribute the Actor Model, Reward Model, Reference Model, and the Critic Model onto separate GPUs using Ray, while placing the Adam Optimizer on the CPU. This enables full-scale fine-tuning of 7B models across multiple 24GB RTX 4090 GPUs (or 34B models with multiple A100 80G), with high training efficiency thanks to the ability to use a large generate batch size with Adam Offload and Ray. **Our PPO performance with the 13B llama2 models is 4 times that of DeepSpeedChat.**


### Features

- A fast LLaMA2 SFT/PPO Training Framework based on DeepSpeed.
- Multi-nodes [training scripts](./examples/scripts/train_llama_slurm.sh) for Slurm.
- Support [DPO (direct-preference-optimization)](./examples/scripts/train_dpo_llama.sh).
- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh) for 34B+ models and 7B models on RTX4090. 
- Support [Decision Transformer (DT) Alignment](./examples/scripts/train_decision_transformer_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [top chinese models](https://github.com/OpenLLMAI/OpenRLHF/issues/116).
- Support Wandb log (--wandb).
- Support conda env/nvidia docker.
- Support FlashAttention2 (--flash_attn).
- Pre-trained 7B/13B llama2 [checkpoints](https://huggingface.co/OpenLLMAI/openrlhf_checkpoint)
- Support [GPT4 evaluation](./evaluation/gpt4/README.md) \& PPO vs SFT <a href="./docs/ppo_examples.md">examples</a>
- Support Multiple Reward models.
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling.sh).

**TODO** 
- Support samples checkpoint.
- Support QLora.


Support Matrix


|        | PPO-max & Best Hyperparameters  | Ray  | 34B Full Tuning with 4 A100   | 7B Full Tuning with 1 A100 (80G)  | 7B Full Tuning with 4 RTX4090 |
|  ----  | ----  |  ----  | ----  | ----  | ----  |  
| OpenRLHF  | ✔ | ✔  | ✔ | ✔ | ✔ |
| DeepSpeedChat  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |
| ColossalAIChat  | ✖️ | ✖️  | ✖️ |✖️ | ✖️ |
| TRL  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ |

### Performance

|        | 7B llama2 RLHF | 13B llama2 RLHF (50k samples) | 
|  ----  | ----  |  ----  |
| OpenRLHF  | - | 22 hours with 8 A100  | 
| DeepSpeedChat  | - | 48 hours with 16 A100  |

**Ray/DeepSpeed Config:** 

- 4 A100 80G for Actor, 2 A100 80G for Critic, 1 A100 80G for RM, and 1 A100 80G for InitPolicy
- ZeRO2 with Adam Offload
- Max Sequence Length: 2048 

**Throughput:**

- 7B llama2: 0.105 samples/gpu/secs
  - micro_batch_size = 16/8 (rollout/train), generation_length = 100~300
- 13B llama2: 0.04 samples/gpu/secs
  - micro_batch_size = 8/4 (rollout/train), generation_length = 200~400
- 34B codellama: 0.007 samples/gpu/secs
  - micro_batch_size = 2/1 (rollout/train), generation_length = 300~800

samples/gpu/secs = Number of PPO Samples / Number of A100 GPUS / Seconds

## Running Example

You can build openrlhf from **nvidia-docker(recommended)** or from conda envs.

```shell
Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git

# Download the pre-trained SFT/RM checkpoints (Optional)
git lfs install
git clone  https://huggingface.co/OpenLLMAI/openrlhf_checkpoint
```

* Single-node training with nvidia-docker

```shell
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

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

# train Rejection Sampling model
./train_rejection_sampling_llama.sh

# train Decision Transformer model
./train_decision_transformer_llama.sh
```

* PPO training with Ray
> for 13B/34B models on A100/H100.. or 7B models on RTX4090

```shell
cd examples/scripts

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# launch ray in container
nohup ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --block &> ray.log &

# if you want to launch ray on more nodes, use
# ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8 --block

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

# For SFT, RM, and PPO and DPO training:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"
readonly training_script="train_dpo_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh
```

* build openrlhf from conda envs 

If you really don't want to use nvidia-docker, we also provide tutorials for building openrlhf from a conda environment. (We prefer nvidia-docker to avoid errors caused by the environment.)
```shell
# we need conda
conda create -n openrlhf python=3.10
# so, we need install some package manually: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip install torch --index-url https://download.pytorch.org/whl/cu118
# check ninjia
ninja --version
echo $? # output: 0
# install flash-attn: may take some time.
# For network error: you can download specified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.1.1 --no-build-isolation
./build_openrlhf.sh
# enjoy it!
```

* Inference and Evaluation

After completing the training, you can evaluate your model by using the `inference` script:

```shell
# interactive_chat
./interactive_chat_llama.sh { model_path }

# batch generate
python examples/batch_inference.py {args}
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

## Citation
```
@misc{openllmai23,
   author = {OpenLLMAI},
   title = {OpenRLHF},
   year={2023},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenRLHF}}
}
```

______________________________________________________________________

*OpenRLHF © 2023 OpenLLMAI. All Rights Reserved.*
