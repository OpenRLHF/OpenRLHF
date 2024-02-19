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
      <em>Open-source / Comprehensive / Lightweight / Easy-to-use</em>
    </p>
</p>
</div>

<hr>

OpenRLHF is a high-performance RLHF framework built on Ray, DeepSpeed and HF Transformers:

- **Simple and easy to use**: OpenRLHF is one of the simplest high-performance RLHF libraries currently available, enabling 34B model RLHF training with just a single DGXA100 node (see the training [script](./examples/scripts/train_ppo_llama_ray_34b.sh)).
- **Distributed RLHF**: The key idea behind OpenRLHF is to distribute the Actor, Reward, Reference, and Critic models onto separate GPUs using Ray, while placing the Adam optimizer on the CPU. This enables full-scale fine-tuning of 7B models across multiple 24GB RTX 4090 GPUs (or 70B+ models with multiple A100 80G GPUs and vLLM, see [architecture](./docs/ray_architecture.png)).
- **High performance**: RLHF training spends 80% of the time on the sample generation stage. Thanks to the ability to use a large inference batch size with Ray and Adam Offload (Pinned Memory), the performance of OpenRLHF with the 13B LLaMA2 model is 4x that of DeepSpeedChat. We also support vLLM generation acceleration to further improve the generation performance.
- **PPO Implementation Tricks**: We integrated the implementation tricks for PPO to improve the training stability, referencing https://arxiv.org/abs/2005.12729 and https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.


## Features

- Distributed [PPO based on Ray](./examples/scripts/train_ppo_llama_ray.sh). 
- Support full RLHF fine-tuning of models with [over 70 billion parameters](./examples/scripts/train_ppo_llama_ray_70b.sh).
- Support vLLM generation acceleration in RLHF (--vllm_num_engines).
- Support multiple reward models (--reward_pretrain model1,model2...).
- Support [DPO (direct-preference-optimization)/IPO/cDPO](./examples/scripts/train_dpo_llama.sh).
- Support [Kahneman-Tversky optimization (KTO)](./examples/scripts/train_kto_llama.sh).
- Support [Rejection Sampling](./examples/scripts/train_rejection_sampling_llama.sh).
- Support [Conditional SFT](./examples/scripts/train_conditional_llama.sh) (https://arxiv.org/abs/2308.12050).
- Support [Mixtral 8*7b](./examples/test_scripts/train_sft_mixtral_lora.sh) (--aux_loss_coef)
- Support Wandb log (--wandb).
- Support FlashAttention2 (--flash_attn).
- Support QLoRA (--load_in_4bit), LoRA (--lora_rank, --target_modules).
- Multi-nodes [training scripts](./examples/scripts/train_llama_slurm.sh) for Slurm.
- Pre-trained LLaMA2 [checkpoints](https://huggingface.co/OpenLLMAI)


**TODO** 
- Allows saving and loading training checkpoints.
- Support Hybrid vLLM inference engine.

RLHF Support Matrix


|        | PPO Implementation Tricks  | 34B Full Tuning with 4 A100   | 70B+ Full Tuning with 16 A100  | 7B Full Tuning with 4 RTX4090 | QLoRA | Mixtral 8*7b |
|  ----  | ----  |  ----  | ----  | ----  | ----  |  ----  | 
| OpenRLHF  | ✔ | ✔  | ✔ | ✔ | ✔ | ✔ |
| DeepSpeedChat  | ✖️ | ✖️  | ✖️ | ✖️ | ✖️ | ✖️ |
| ColossalAIChat  | ✖️ | ✖️  | ✖️ |✖️ | ✖️ | ✖️ |
| TRL  | ✖️ | ✖️  | ✖️ | ✖️ | ✔ | ✖️ |

## Performance

|        | 7B llama2 RLHF | 13B llama2 RLHF (50k samples) | 
|  ----  | ----  |  ----  |
| OpenRLHF  | - | 17 hours with 8 A100  | 
| DeepSpeedChat  | - | 48 hours with 16 A100  |

**Configs for Ray and DeepSpeed:** 

- 4 A100 80G for Actor, 2 A100 80G for Critic, 1 A100 80G for RM, and 1 A100 80G for InitPolicy
- ZeRO2 with Adam Offload
- Max Sequence Length: 2048 

**Throughput:**

- 7B llama2: 0.136 samples/gpu/secs
  - micro_batch_size = 16/8 (rollout/train), generation_length = 100~300
- 13B llama2: 0.05 samples/gpu/secs
  - micro_batch_size = 8/4 (rollout/train), generation_length = 200~400
- 34B codellama: 0.009 samples/gpu/secs
  - micro_batch_size = 2/1 (rollout/train), generation_length = 300~800

samples/gpu/secs = Number of PPO Samples / Number of A100 GPUS / Seconds

## Running Example

You can build openrlhf from **nvidia-docker(recommended)** or from conda envs.

```shell
Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git
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

# continue pretrain
./train_continue_pretrain_llama.sh

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh

# train DPO model
./train_dpo_llama.sh

# train KTO model
./train_kto_llama.sh

# train Rejection Sampling model
./train_rejection_sampling_llama.sh

# train Conditional SFT model
./train_conditional_llama.sh
```

* PPO training with Ray
> for > 13B models on V100/A100/H100.. or 7B models on RTX4090

```shell
cd examples/scripts

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh
# due to the compatibility of nVIDIA PyTorch image
pip uninstall xgboost transformer_engine -y

# huggingface login 
~/.local/bin/huggingface-cli login

# launch the master node of ray in container
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8
# if you want to launch ray on more nodes, use
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8


# train ray PPO model, requires 8 gpus in default config
./train_ppo_llama_ray.sh

# for 70B models and vLLM-based RLHF (important!)
pip install vllm==0.3.1
# due to the compatibility of vLLM
pip uninstall flash_attn -y

./train_ppo_llama_ray_70b.sh
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

# for Ray PPO training with Slurm
sbatch ./train_ppo_llama_ray_slurm.sh
```

* Inference and Evaluation

After completing the training, you can evaluate your model by using the `inference` script:

```shell
# interactive_chat
./interactive_chat_llama.sh { pretrain_model_path }

# batch generate
# support vLLM acceleration (--eval_task generate_vllm)
python examples/batch_inference.py {args}
```

* build openrlhf from conda envs 

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
# For network error: you can download specified version from https://github.com/Dao-AILab/flash-attention/releases.
pip install flash-attn==2.4.2
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
   title = {OpenRLHF: A Ray-based High-performance RLHF framework},
   year={2023},
   publisher = {GitHub},
   journal = {GitHub repository},
   howpublished = {\url{https://github.com/OpenLLMAI/OpenRLHF}}
}
```

______________________________________________________________________

*OpenRLHF © 2023 OpenLLMAI. All Rights Reserved.*
