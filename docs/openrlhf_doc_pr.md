**OpenRLHF：轻量高效的工业级LLM训练和对齐框架**，支持70B模型RLHF全参数全流程训练！

## 文档更新日志

​	本文最新版本维护在GitHub（微信和知乎更新都不甚方便），更好的阅读体验可转[公众号](https://mp.weixin.qq.com/s/9UmYgs3b-WZMPEkzo-kTSw)，更新记录：

- 24.2.29，完善框架对比
- 24.2.25，本文初次公开发表

## 项目简介

### 背景

​	自ChatGPT横空出世以后，大家开始关注到以InstructGPT为代表的RLHF对齐技术，并以此为基础尝试复现ChatGPT的训练流程，逐步出现了 ColossalChat、DeepSpeed-Chat等代表性的RLHF复现工作。但彼时大家对对齐技术的理解，基本都是围绕着InstructGPT展开的，由于OpenAI最近不太Open，实际上是缺乏第三方的充分验证的。幸运的是，LLaMA2很快就横空出世了，不仅充分验证了RLHF技术的有效性，还有着足够的创新之处（比如拒绝采样和多RM等），立马引爆了整个LLM开源社区。

​	鉴于InstructGPT和LLaMA2的火爆，我们OpenLLMAI开源社区调研了当前主流的对齐训练框架，发现大部分框架还缺乏对LLaMA2全流程全参数训练的支持、缺乏足够的可扩展性或者不够轻量易用。因此我们决心做一个真正工业级的LLM对齐训练框架，复现以InstructGPT和LLaMA2为代表的大模型训练流程，支持主流的RLHF/DPO等对齐技术，帮助大家快速实现自己的对齐想法。

​	所以，欢迎来到OpenRLHF，快速开启你的对齐工作吧！

​	https://github.com/OpenLLMAI/OpenRLHF

### 设计思路

1.设计目标：**轻量高效**的**工业级**LLM训练和对齐框架

​	由于目前业界缺乏真正工业级别的LLM对齐框架，大部分厂商可能会选择自己实现（感谢OpenAI开了个好头），短期来看这无可厚非，但长远来看终归难免重复造轮子的问题。

​	因此，我们的目标是做出一个轻量高效的工业级LLM训练和对齐框架。为了实现这一目标，我们一方面在第一个版本中做了比较审慎的开发和测试，力求第一个版本的可用性；另一方面在此正式开源，以吸引更多志同道合者来参与共建。对于框架，我们坚信开源才有生命力！

2.设计思想：简单易用、高性能、可扩展、探索性

- **简单易用**：易用性是我们设计OpenRLHF框架的第一个指导思想，因为高性能是一个合格框架的应有之义，所以我们并不会过多的强调这个事情，在保证高性能的前提下，提升易用性是我们的第一目标。
- **可扩展**：以7B为基础，向下兼容1-2B小模型的训练，向上逐步支持不断增长的模型规模，比如34B/70B/170B的训练。
- **探索性**：在保证基础的框架功能之外，我们会保持对齐技术的前沿性，跟踪最新进展并快速实现，同时也会提供我们团队开发的最新的对齐算法。后续我们还会开发LLMPipeline模块，以提供主流对齐算法或者主流模型训练技术的快速实践和公平比较。

3.实现思路

- 易用性：在基础大模型框架方面，我们调研了DeepSpeed/[Megatron-LM](https://github.com/NVIDIA/Megatron-LM#pr5e0g)等LLM训练框架，在第一个版本中选择了更简洁易用的DeepSpeed；在模型库方面，我们毫不犹豫的选择了拥抱抱抱脸；在分布式扩展方面，我们选择了ray，别问，问就是祥ray！（主要用于资源调度）
- **可扩展和高性能**：使用ray进行合理的**GPU资源调度**，将Actor、Reward、Reference 和 Critic 模型分配到单独的 GPU 上，**将训练和推理隔离**以充分利用推理社区的优秀工具，同时配合offload、PEFT等**显存节省**技术，实现大模型的规模扩展和高效训练。
- 探索性：第一个版本我们完整的复现了InstructGPT和LLaMA2的训练流程，并支持了DPO等更新的对齐技术，未来也将继续保持探索性，并开发pipeline模块，以支持InstructGPTPipeline和LLaMA2Pipeline等主流模型的pipeline，帮助社区进行更科学的比较和研究。

### 主要亮点

### 主要特性

- **首个开源的全面复现LLaMA2**和InstructGPT的RLHF对齐框架；
	- 支持SFT/RM/PPO全流程训练；
	- 支持**拒绝采样、多个RM；**

- 简单易用：OpenRLHF 是目前最简单的高性能 RLHF 库之一，只需单个8卡 DGXA100 节点即可实现 **34B** 模型 RLHF 训练，可通过[脚本](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_ppo_llama_ray_34b.sh)**一键启动训练**；
- 训推分离，分布式可扩展的RLHF；
	- **训推分离**：训练和推理分离，以复用社区良好的推理工具（我们最终使用了vLLM）降低推理延时；
	- **分布式可扩展**：通过ray/deepspeed/vLLM的加持，在合理的资源调度下，我们实现了高效可扩展的训练，以下为两个示例：
		- 使用多卡**24GB** RTX 4090 GPU 进行**7B** 模型的全流程训练
		- 使用多卡 **A100 80G** GPU 和 vLLM 进行**70B+ 模型**的全流程训练

- 高性能：得益于ray/deepspeed和其他的显存节省技术、推理加速框架，我们在13B LLaMA2 模型上的训练性能是 DeepSpeedChat 的 4 倍以上；
	- 推理加速：vLLM
	- 显存节省技巧：
		- zero系列
		- FlashAttention2
		- LoRA、QLoRA
		- offload
		- gradient checkpointing

- 前沿性：紧跟前沿进展，目前支持主流的对齐技术、主流的大模型；

	- 最先进的对齐技术：
		- 标准的RLHF：SFT/RM/**PPO**；
		- [**Rejection Sampling**](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_rejection_sampling_llama.sh#dhcwju)；
		- [**DPO** (direct-preference-optimization)/IPO/cDPO](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_dpo_llama.sh#rvyrih)；
		- [Kahneman-Tversky optimization (**KTO**)](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_kto_llama.sh#n1700j)；
		- [Conditional SFT](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_conditional_llama.sh#h2s086) ([https://arxiv.org/abs/2308.12050](https://arxiv.org/abs/2308.12050#jd6s1k))；

	- 最前沿的模型：

		- LLaMA

		- baichuan

		- qwen

		- **[Mixtral 8*7b](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/test_scripts/train_sft_mixtral_lora.sh#bv9ttu)**

- 强化学习技巧：We integrated the implementation tricks for PPO to improve the training stability, referencing [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729#0) and [ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#0).

	

![img](https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/v2-04e64a645c8dba68cc2a4dfeaa56c5c4_1440w.png)



![img](https://pic1.zhimg.com/80/v2-55d46172a942474cc6ad3de6a4c38cf2_1440w.png?source=d16d100b)



### 性能展示

### 支持矩阵：

​	下面的支持矩阵展示了OpenRLHF与主流LLM对齐训练框架的比较（调研可能存在延迟，错漏之处请联系我们修正）：

|                | PPO Tricks | 34B 全参/4 A100 | 70B+全参/16 A100 | 7B 全/4 RTX4090 | QLoRA | Mixtral MOE 8*7b |
| -------------- | ---------- | --------------- | ---------------- | --------------- | ----- | ---------------- |
| OpenRLHF       | ✔          | ✔               | ✔                | ✔               | ✔     | ✔                |
| DeepSpeedChat  | ✖️          | ✖️               | ✖️                | ✖️               | ✖️     | ✖️                |
| ColossalAIChat | ✖️          | ✖️               | ✖️                | ✖️               | ✖️     | ✖️                |
| TRL            | ✔          | ✖️               | ✖️                | ✖️               | ✔     | ✖️                |
| LLaMA-Factory  | ✖️          | ✖️               | ✖️                | ✖️               | ✔     | ✔(QLoRA)         |

​	OpenRLHF的主要优势在于**良好的可扩展性**和**高效的性能**，可以支持70B模型的全流程全参数高效训练，也可以应对未来更大规模的扩展。而LLaMA-Factory/trl/trlx 等框架都存在类似的问题， **不支持 70B 全参数RLHF训练**，有的框架主打Lora 微调 13b 级别的模型，一般采样**合并 actor critic 的方案**（节省显存，这是小规模上进行RLHF的权宜之计，但并不符合标准RLHF的实现，而且可扩展性很差，总有放不下的时候）。当然了，OpenRLHF也存在一些劣势，比如文档和benchmark不够完善，**易用性还有待提高**。具体而言，就OpenRLHF与各流行RLHF框架的对比我们做如下说明（错漏之处，欢迎大家指正），更详细和全面的对比后续可以在我们正式的技术报告中找到。

- LLaMA-Factory：优势是高效微调和易用性（这一点非常值得我们学习，甚至有web-ui），使用merged actor-critic，无法支持70B 全参数PPO训练，也不便于扩展模型规模；
- Colossal-Chat：使用single-step RL，而我们的框架使用的是step-wise RL。详见[OpenRLHF vs Colossal-Chat](https://github.com/OpenLLMAI/OpenRLHF/issues/149)；
- trl/trlx：优势是与Hugging Face的生态兼容的非常好，但可能存在封装过深不易修改的问题，同样的，目前暂不支持70B 全参数PPO训练；而且使用的是merged actor-critic以节省显存，但这与标准实现不符；
- NeMo-Aligner：基于Megatron的生成目前效率不高，影响了整体训练效率，与Hugging Face的生态兼容性不太好，模型可能需要做专门的修改；

![image-20240229012616418](https://xianyunlp.oss-cn-hangzhou.aliyuncs.com/uPic/image-20240229012616418.png)



### 性能数据：

​	根据现有测试，我们的OpenRLHF框架在13B模型上的训练效率大约是DeepSpeedChat的4倍左右（人力所限，测试可能存在延迟，大家可以向我们报告其他框架的性能数据以做修正）。

|               | 7B llama2 RLHF | 13B llama2 RLHF (50k samples) |
| ------------- | -------------- | ----------------------------- |
| OpenRLHF      | -              | 17 hours with 8 A100          |
| DeepSpeedChat | -              | 48 hours with 16 A100         |

​	

​	训练吞吐测试：

- 默认配置：
	- 4 A100 80G for Actor, 2 A100 80G for Critic, 1 A100 80G for RM, and 1 A100 80G for InitPolicy
	- ZeRO2 with Adam Offload
	- Max Sequence Length: 2048

- 性能吞吐（默认配置下samples/s，后续会换成tokens/s）：
	- 7B llama2: 0.105 samples/gpu/secsmicro_batch_size = 16/8 (rollout/train), generation_length = 100~300
	- 13B llama2: 0.04 samples/gpu/secsmicro_batch_size = 8/4 (rollout/train), generation_length = 200~400
	- 34B codellama: 0.007 samples/gpu/secsmicro_batch_size = 2/1 (rollout/train), generation_length = 300~800

​	主流模型性能数据（人力原因，暂时来不及重新测试，这里报告的是当时支持该模型时的测试数据，当前版本PPO应该会快很多，后续会在正式的技术报告中补充更多的模型并更新性能数据）：

| model        | SFT  | RM   | PPO  | Notes |
| ------------ | ---- | ---- | ---- | ----- |
| Baichuan2-7B | 1h   | 4h   | 71h  |       |
| Qwen-7B      | -    | -    | -    |       |

## 使用方法

### 官方文档：

​	包括本文在内的官方正式文档都将在Github上维护，提升文档质量以改善易用性也是我们后续工作的重点方向之一（人力原因，文档目前比较粗糙，欢迎大家参与贡献）：

- [项目主页](https://github.com/OpenLLMAI/OpenRLHF)
- [官方文档](https://github.com/OpenLLMAI/OpenRLHF/blob/main/docs/openrlhf_doc.md)

### 安装

​	我们支持**nvidia-docker（推荐，以避免潜在的环境问题）**或者conda环境安装（后续可以提供配置好的conda环境或者镜像）**：**

​	首先，clone仓库:

```
Clone the repository: 
git clone https://github.com/openllmai/OpenRLHF.git
```

​	然后，安装**nv-docker**或者conda环境：

```
#安装nv-docker
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh
```

```
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
pip install flash-attn==2.5.8
./build_openrlhf.sh
# enjoy it!

conda activate openrlhf
```



### 训练

### 训练脚本：

​	配置好环境之后，进入/openrlhf/examples/scripts目录，根据自己的需求修改训练脚本，一键即可启动训练，支持单机和多机训练，**支持7B-70B+的模型全量全流程训练**。以下为部分重要参数，用户可以根据情况进行修改，以支持自己的模型训练：

- -pretrain：预训练模型地址, 抱抱脸格式
- -dataset ：数据集地址, 抱抱脸格式
- -dataset_probs：多个数据集混合的采样概率比如: 0.5,0.4,0.1
- -save_path：模型保存地址, 抱抱脸格式
- -max_epochs：训练 epoch 次数
- -micro_train_batch_size：单 GPU batch_size
- -train_batch_size：全局 batch_size
- -learning_rate：学习速率

​	单机训练脚本：

```
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



​	多机训练脚本，**16卡A100 70B模型全参数RLHF训练**：

```
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
pip install vllm==0.3.2
# due to the compatibility of vLLM
pip uninstall flash_attn -y

./train_ppo_llama_ray_70b.sh
```



### 推理

​	推理和评估我们推荐复用业界开源的工具或者代码，可以参考以下脚本：

```
# interactive_chat
./interactive_chat_llama.sh { pretrain_model_path }

# batch generate
# support vLLM acceleration (--eval_task generate_vllm)
python examples/batch_inference.py {args}
```



## 未来工作

​	OpenRLHF未来的开发工作将以易用性和实用性（文档、教程、实践经验等）、前沿性（新算法、模型pipeline等）和稳定性为主，具体而言，有以下潜在的工作，希望大家可以一起参与：

- **文档：中英文版本**
	- 教程：提供良好的**教程**
	- **环境：提供配置好的镜像或conda环境**；

- **性能测试、benchmark；**
	- 基础功能的测试
	- **与其他框架的对比**
	- 支持模型的测试
	- 对齐算法的测试

- 进一步的性能优化；
- 稳定性提升：定期代码review；
- 新功能、新算法；
- 新模型支持：Google的新模型等；
- evaluation模块提供更全面的评估能力；

## 组织介绍

### 组织简介

​	OpenLLMAI: Open AI for everyone.

​	我们或许离OpenAI很远，但我们离Open很近。所以。我们对组织成员的要求也只有两个，那就是希望大家足够Open也足够自信。我们的态度：”有一分热发一分光“。愿与大家同走AI之毅行路，士不可以不弘毅，任重而道远！

​	大家因热爱相聚，主要想做的事情就两个：1.交流LLM的技术（技术分享、知识传播）；2.开发LLM的工具（训练框架、模型、数据工程等），欢迎感兴趣的同学加入我们！组织的详细介绍见[知乎旧文OpenLLMAI组织介绍](https://zhuanlan.zhihu.com/p/647882819)。

### 项目介绍

​	目前OpenLLMAI主要在做的项目有两个：

- LLM框架：https://github.com/OpenLLMAI/OpenRLHF
- LLMWiki：https://github.com/OpenLLMAI/OpenLLMWiki大模型炼丹知识库

​	其余项目看后续人力和兴趣而定，近期我们可能会启动一个KD或者SE的框架，暂时可能对训练一个通用的小模型兴趣不大，主要原因是时间、资金和精力都很有限，用爱发电有点难以为继，所以很多时候更多会考虑兴趣导向。但是，兴趣毕竟不能当饭吃，所以我们近期花了比较大的精力来准备这份宣传材料（以前羡鱼同学太过佛系/强迫症/忙，毛病很多hh）。诚然，OpenLLMAI还很懵懂，**OpenRLHF也还不够完善**，但是我们尽了最大的诚意，也希望可以得到社区更广泛的认可和支持，一群人可以走的更远！

![img](https://pic1.zhimg.com/80/v2-03906b88a7828550402392da4a395c26_1440w.png?source=d16d100b)



### 加入&赞助我们

### 开发者们

​	一路走来，OpenRLHF项目吸引了20+的贡献者，贡献了130+的提交，收获了800+ stars。在此感谢所有的贡献者，尤其是[hijkzzz](https://github.com/hijkzzz)、[wuxibin](https://github.com/wuxibin89)和[Xianyu](https://github.com/catqaq)等同学为项目的发展做出了突出贡献，其中[hijkzzz](https://github.com/hijkzzz)和[Xianyu](https://github.com/catqaq)同学是本项目的发起人，[hijkzzz](https://github.com/hijkzzz)作为项目的管理员提交了本项目的第一版代码，长期投入了大量精力进行维护，为项目的发展做出了不可替代的贡献；[wuxibin](https://github.com/wuxibin89)作为项目的核心开发者，主要负责基于Ray对框架进行大规模扩展，并且长期投入了大量精力进行日常维护；[Xianyu](https://github.com/catqaq)作为项目的管理员负责了NLP部分的开发工作和一些项目规划工作；此外还有pikaqqqqqq、li-plus、wwxFromTju、jovany-wang、xffxff、dabney777、suc16、Dylancer1998等同学也对项目的发展做出了重要贡献（这里无法一一列出，后续所有的开发者都会在正式的技术报告/论文中说明；还有很多的同学和老师虽然没有直接参与贡献，但是提出了很多宝贵的意见，真的非常感谢大家）。也欢迎越来越多志同道合的朋友们加入我们，愿OpenLLMAI与大家一起成长！

![img](https://pic1.zhimg.com/80/v2-6b902cdbd97e073ab5bebdd44de08243_1440w.png?source=d16d100b)



![img](https://picx.zhimg.com/80/v2-c1b8bfd3699964a1bbbdf0a255be4e0b_1440w.png?source=d16d100b)

​	有意参与贡献的同学可直接在git上参与开发、联系相关负责人或者官方邮箱。

- RL：[hijkzzz](https://github.com/hijkzzz)
- Ray：[wuxibin](https://github.com/wuxibin89)
- NLP：[Xianyu](https://github.com/catqaq)
- 官方邮箱：[xianyuai@openllmai.top](mailto:xianyuai@openllmai.top#e54z4m)



### 赞助我们

​	目前OpenLLMAI是一个纯开源组织，无论是OpenRLHF/OpenLLMWiki等项目，还是OpenLLM Talk和技术交流群，都是完全开源开放的。但长远来看，没有资金支持注定难以为继，用爱发电走到今天并不容易，感谢大家一路的支持。最后，求赞助呀，欢迎大家有钱的捧个钱场（算力！！！），有人的捧个人场（参与开发或者其他贡献）！赞助或合作请联系[xianyuai@openllmai.top](mailto:xianyuai@openllmai.top#al782z)。

## 参考资料

https://github.com/OpenLLMAI/OpenRLHF

https://github.com/NVIDIA/Megatron-LM

https://chat.openai.com/

[InstructGPT](https://arxiv.org/abs/2203.02155)

[LLaMA2](https://ai.meta.com/llama/#jt1oe1)

https://github.com/facebookresearch/llama

[Hugging Face Transformers](https://github.com/huggingface/transformers#0)

[DeepSpeed](https://github.com/microsoft/DeepSpeed#dwb51f)

https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat

[Ray](https://github.com/ray-project/ray#2wguhk) 

https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat 

https://github.com/CarperAI/trlx

https://github.com/NVIDIA/NeMo-Aligner

https://github.com/hiyouga/LLaMA-Factory

https://github.com/OpenLLMAI/OpenLLMWiki

【OpenLLMAI】相信开源的力量：我们有自己的组织了！任重道远，行则将至！ - OpenLLMAI的文章 - 知乎https://zhuanlan.zhihu.com/p/647882819

如何正确复现 Instruct GPT / RLHF? - 蜗牛在花园跑酷的文章 - 知乎 https://zhuanlan.zhihu.com/p/622134699

开启训练之旅: 基于Ray和vLLM构建70B+模型的开源RLHF全量训练框架 - 蜗牛在花园跑酷的文章 - 知乎https://zhuanlan.zhihu.com/p/678828949

【OpenLLM 006】LoRA:大模型的低秩适配-最近大火的lora到底是什么东西？为啥stable diffusion和开源ChatGPT复现都在用？ - OpenLLMAI的文章 - 知乎

https://zhuanlan.zhihu.com/p/620327907

https://arxiv.org/abs/2005.12729

https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

https://github.com/baichuan-inc/Baichuan2

https://github.com/QwenLM/Qwen

https://mistral.ai/news/mixtral-of-experts/

https://github.com/OpenLLMAI/OpenRLHF/issues/221