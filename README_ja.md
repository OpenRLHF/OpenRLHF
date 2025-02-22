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
      <a href="https://github.com/OpenRLHF/OpenRLHF/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?color=ccf" />
      </a>
      <br>
      <em>オープンソース / 包括的 / 軽量 / 使いやすい</em>
    </p>
</p>
</div>

<hr>

<span>[ <a href="README.md">English</a> | <a href="README_zh.md">中文</a> | 日本語 ]</span>

OpenRLHFは、Ray、DeepSpeed、およびHF Transformersを基盤とした高性能なRLHFフレームワークです：

- **シンプルで使いやすい**: OpenRLHFは現在利用可能な最もシンプルな高性能RLHFライブラリの一つであり、Huggingfaceのモデルとデータセットとシームレスに互換性があります。
- **高性能**: RLHFトレーニングの80％の時間はサンプル生成段階に費やされます。RayとPacking SamplesおよびvLLM生成加速の能力を活用することで、OpenRLHFのパフォーマンスはOptimized DeepSpeedChat with Hybrid Engineの3〜4倍以上です。
- **分散RLHF**: OpenRLHFは、Actor、Reward、Reference、およびCriticモデルをRayを使用して別々のGPUに分散し、AdamオプティマイザをCPUに配置します。これにより、複数のA100 80G GPUとvLLMを使用して70B+モデルのフルスケールの微調整が可能になり、複数の24GB RTX 4090 GPUで7Bモデルを微調整できます。
- **PPO実装の最適化**: トレーニングの安定性を向上させるために、PPOの実装トリックを統合しました。詳細は[Zhihu](https://zhuanlan.zhihu.com/p/622134699)および[Notionブログ](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361)を参照してください。

詳細は[スライド](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [技術報告](https://arxiv.org/abs/2405.11143) | [ドキュメント](https://openrlhf.readthedocs.io/)をご覧ください。

## ニュース
- [2025/2] [Logic-RL](https://arxiv.org/abs/2502.14768) と [PRIME](https://arxiv.org/abs/2502.01456) は、REINFORCE++ が訓練の安定性において GRPO より優れ、PPO より高速であることを示した。
- [2025/2] StepFunc は [OpenRLHF のシングルコントローラーバージョン](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)を実装しました。
- [2025/2] [LMM-R1](https://github.com/TideDra/lmm-r1) は OpenRLHF のフォークで、マルチモーダルタスクでの DeepSeek-R1 の再現のための高性能 RL インフラストラクチャを提供することを目的としています。
- [2025/2] MIT & Microsoft は OpenRLHF を使用して [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) を提案しました。
- [2025/1] HKUSTは [OpenRLHF を使用して小規模モデルでの DeepSeek-R1-Zero と DeepSeek-R1 のトレーニング](https://github.com/hkust-nlp/simpleRL-reason)を再現しました。
- [2024/12] 私たちは😊 [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://www.researchgate.net/publication/387487679_REINFORCE_A_SIMPLE_AND_EFFICIENT_APPROACH_FOR_ALIGNING_LARGE_LANGUAGE_MODELS)を「提案」しました。
- [2024/12] [Notionブログ](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05)でPPO、REINFORCE++、GRPO、およびRLOOを分析しました。

## 特徴

- Rayに基づく分散[ PPO](./examples/scripts/train_ppo_llama_ray.sh)および[REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray.sh)の実装。
- [Ray-based Reinforced Finetuning](./examples/scripts/train_ppo_llama_with_reward_fn.sh)
- RayとHybrid Engineに基づく[PPO](./examples/scripts/train_ppo_llama_ray_hybrid_engine.sh)および[REINFORCE++/RLOO](./examples/scripts/train_reinforce_llama_ray_hybrid_engine.sh)のサポート (`--colocate_all_models`, `--vllm_enable_sleep` and `--vllm_gpu_memory_utilization 0.5`)
- [70億以上のパラメータを持つモデル](./examples/scripts/train_ppo_llama_ray_70b.sh)の完全なRLHF微調整のサポート。
- RLHFタスクでの生成を加速するためのvLLMの統合（`--vllm_num_engines`）。
- 複数の報酬モデル（`--reward_pretrain model1,model2...`）およびリモート報酬モデル（`--remote_rm_url`）のサポート。
- [DPO（直接選好最適化）/IPO/cDPO](./examples/scripts/train_dpo_llama.sh)および[Kahneman-Tversky Optimization（KTO）](./examples/scripts/train_kto_llama.sh)の実装。
- [反復DPO](./examples/scripts/train_iterative_dpo_llama.sh)（[GitHub: Online-RLHF](https://github.com/RLHFlow/Online-RLHF)）のサポート。
- [拒否サンプリング](./examples/scripts/train_rejection_sampling_llama.sh)のサポート。
- [条件付きSFT](./examples/scripts/train_conditional_llama.sh)（[arXiv:2308.12050](https://arxiv.org/abs/2308.12050)）の実装。
- [知識蒸留](./examples/scripts/train_knowledge_distillation.sh)（[Microsoft: minillm](https://github.com/microsoft/LMOps/tree/main/minillm)）のサポート。
- [プロセス報酬モデル（PRM）](./examples/scripts/train_prm_mistral.sh)の統合。
- SFT、DPO、RM、PRM、およびPPOのトレーニングサンプルのパッキング（`--packing_samples`）。
- [RingAttention](./examples/scripts/train_dpo_ring_llama.sh)の実装（`--ring_attn_size`、`--ring_head_stride`）。
- [専門家の混合モデル（MoE）](./examples/test_scripts/train_sft_mixtral_lora.sh)のサポート（`--aux_loss_coef`）。
- FlashAttention2の統合（`--flash_attn`）。
- QLoRA（`--load_in_4bit`）および[LoRA](./examples/scripts/train_sft_mixtral_lora.sh)（`--lora_rank`、`--target_modules`）のサポート。
- HuggingFaceの`tokenizer.apply_chat_template`との互換性（`--apply_chat_template`および`--input_key`）。
- Wandb（`--use_wandb`）, SwanLab（`--use_swanlab`）およびTensorBoard（`--use_tensorboard`）によるログ記録のサポート。
- チェックポイントの回復機能（`--load_checkpoint`および`--save_steps`）。
- [DPO](./examples/scripts/train_llama_slurm.sh)および[Ray PPO](./examples/scripts/train_ppo_llama_ray_slurm.sh)などのマルチノードトレーニングスクリプトを提供。

### PPOサポートマトリックス

| 特徴 | OpenRLHF | DSChat | CAIChat | TRL |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:|
| 16 A100-80GBで70B+のフルチューニング      | ✅ | ❌ | ❌ | ❌ |
| 4 RTX4090で7Bのフルチューニング | ✅      |    ❌ | ❌ | ❌ |
| 8 A100-80GBで34B DPOのフルチューニング | ✅      |    ❌ | ❌ | ❌ |  
| PPOでの推論エンジンのサポート | ✅      |    ✅ | ❌ | ❌ |  
| PPO実装のトリック | ✅      |    ❌ | ❌ | ✅ |
| QLoRAのサポート | ✅      |    ❌ | ❌ | ✅ | 
| Mixtral 8*7bのサポート | ✅      |    ❌ | ❌ | ❌ |  
| 未結合のActor-Criticのサポート | ✅     |   ✅ | ✅ | ❌ | 
| 複数の報酬モデルのサポート | ✅      |    ❌ | ❌ | ❌ |   
| Huggingfaceモデルのサポート | ✅      |    ✅ | ✅ | ✅ | 
| 使いやすさ | ✅      |   ❌ (HybridEngineのバグ) | ✅ | ✅ | 

## クイックスタート

### インストール

OpenRLHFを使用するには、まずDockerコンテナを起動し（**推奨**）、Dockerコンテナ内で`pip install`を実行してopenrlhfをインストールします：

```bash
# Dockerコンテナを起動
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:24.07-py3 bash
sudo pip uninstall xgboost transformer_engine flash_attn -y

# pip install
pip install openrlhf

# vLLM加速を使用する場合（vLLM 0.7.2をインストール）
pip install openrlhf[vllm]
# 最新のvLLMもサポートされています
pip install openrlhf[vllm_latest]

# 最新バージョンをpip install
pip install git+https://github.com/OpenRLHF/OpenRLHF.git

# またはgit clone
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

> [!NOTE]
>vLLM 0.6.4以降の使用をお勧めします。他のバージョン（vLLM >= 0.4.2）は、Glooを介して重みの同期が必要な場合があります（`--vllm_sync_backend gloo`）。
>また、[vLLM用のDockerfile](./dockerfile/)および[Nvidia-Dockerのワンクリックインストールスクリプト](./examples/scripts/nvidia_docker_install.sh)も提供しています。

### データセットの準備
OpenRLHFは、データセットクラス内で複数のデータ処理方法を提供しています。
例えば、[Prompt Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/prompts_dataset.py#L6)では：

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

- `--input_key`を使用して、入力データセットの`JSON key name`を指定し、`--prompt_data {name or path}`（PPO）または`--dataset {name or path}`を使用し、`--apply_chat_template`を使用して[Huggingface Tokenizer](https://huggingface.co/docs/transformers/main/en/chat_templating)の`chat_template`を利用できます。
- `--apply_chat_template`を使用したくない場合は、代わりに`--input_template`を使用するか、事前にデータセットをオフラインで前処理することができます。
- OpenRLHFは、`--prompt_data_probs 0.1,0.4,0.5`（PPO）または`--dataset_probs 0.1,0.4,0.5`を使用して複数のデータセットを混合することもサポートしています。

Chat Templatingの動作方法：

```python
dataset = [{"input_key": [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]}]

tokenizer.apply_chat_template(dataset[0]["input_key"], tokenize=False)

"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

トレーニングおよびテストデータセットの指定方法：

`data_type@data_dir`形式を使用して指定できます。例えば、データセットは`--dataset json@./data`として設定できます。

```
data
├── test.jsonl
└── train.jsonl
```

> [!NOTE]
> デフォルトでは、`train`および`test`を使用してHuggingfaceのトレーニングおよびテストデータセットを区別します。
> `JSON key`オプションは特定のデータセットに依存します。詳細は[Reward Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10)および[SFT Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9)を参照してください。

### 教師あり微調整

OpenRLHFのモデルチェックポイントはHuggingFaceモデルと完全に互換性があります。`--pretrain  {name or path}`、`--reward_pretrain  {name or path}`、および`--critic_pretrain  {name or path}`を使用してモデル名またはパスを指定できます。いくつかの事前トレーニング済みチェックポイントとデータセットを[HuggingFace OpenRLHF](https://huggingface.co/OpenRLHF)で提供しています。

次に、[examples/scripts](./examples/scripts/)ディレクトリに提供されている起動スクリプトを使用するか、以下のコマンドを使用してトレーニングを開始できます。

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
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_wandb {wandb_token}

# HF tokenizer.apply_chat_templateのサポート
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# RingAttentionのサポート
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# 継続的な事前トレーニングにも使用できます
# --pretrain_mode
```

> [!NOTE]
> OpenRLHF SFT/DPO/RewardModel/PPOトレーナーは`--packing_samples`をサポートしています [`--flash_attn`に基づく](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing)

### 報酬モデルのトレーニング
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
   --use_wandb {wandb_token}

```

報酬モデルの`--value_prefix_head`オプションを`score`に設定することをお勧めします。これにより、`AutoModelForSequenceClassification`を使用してモデルをロードできます：

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

### Rayを使用しないPPO

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
  --use_wandb {wandb_token}

# リモート報酬モデルのサポート（HTTP）
# --remote_rm_url http://localhost:5000/get_reward
```

### RayとvLLMを使用したPPO/REINFORCE++

RLHFトレーニング速度を向上させるか、70Bモデルをサポートするために、RayとvLLM加速を使用したPPOを使用できます

```bash
# コンテナ内でRayのマスターノードを起動
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# さらに多くのノードでRayを起動する場合は
ray start --address {MASTER-NODE-ADDRESS}:6379  --num-gpus 8

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
  --micro_rollout_batch_size 16 \
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
  --use_wandb {wandb_token}

# REINFORCE++ | RLOOのサポート | REINFORCE++-baseline | GRPO
# --advantage_estimator reinforce | rloo | reinforce_baseline | group_norm

# リモート報酬モデルのサポート（HTTP）
# --remote_rm_url http://localhost:5000/get_reward


# Nサンプルのサポート
# --n_samples_per_prompt 4
```
> [!NOTE]
> `--vllm_num_engines`を設定しない場合は、vLLMエンジンを使用しないことを意味します。
> `setup_commands`を使用してRayが自動的に環境をデプロイすることもできます。例えば、`--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`。

[!NOTE]
OPENRLHFのRLOOは、REINFORCE++を基に改良されたものであり、オリジナル版とは異なります。

> [!NOTE]
> deepspeedがGPUデバイスを設定する際にインデックスが範囲外に関連するエラーが発生した場合、環境変数[`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`](openrlhf/trainer/ray/utils.py)を設定して回避策を試すことができます。
>   ```bash
>   # NVIDIA GPUの場合:
>   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
>   ```

サポートされているアルゴリズムの起動スクリプトとドキュメントは[example/scripts](./examples/scripts/)および[Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)にあります。

### Reinforced Fine-tuning

OpenRLHFは、便利で効率的なReinforced Fine-tuningをサポートしています。カスタム `reward_func` 関数を含む[ファイル](./examples/scripts/reward_func.py)を実装し、そのパスを `remote_rm_url` パラメータに渡すだけです。例えば：

```python
# reward_func.py
import torch

def reward_func(queries, prompts, labels):
    # queries は prompts + responses
    # labels は answers
    print(queries)
    return torch.randn(len(queries))
```

そして、以下のように設定するだけです：

```shell 
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/openrlhf"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  ...
  --remote_rm_url /path/to/reward_func.py
```

### LoRA  
`LoRA (Low-Rank Adaptation)` を使用する場合、`OpenRLHF` はデフォルトで `LoRA Adapter` のみを保存し、フルウェイトは保存しません。タスクを通常どおり続行するには、`Adapter` をベースモデルのウェイトと統合する必要があります。  

```bash
python -m openrlhf.cli.lora_combiner \
    --model_path meta-llama/Meta-Llama-3-8B \
    --lora_path ./checkpoint/llama3-8b-rm \
    --output_path ./checkpoint/llama-3-8b-rm-combined \
    --is_rm \
    --bf16
```

## パフォーマンス

Adamオフロードの有効化、報酬モデル（RM）および参照モデル（Ref）オフロードなどの技術を使用して、DSChatのパフォーマンスを最大限に最適化し、推論段階でのマイクロバッチサイズを増やし、メモリ不足の問題を回避しました。LLaMA2のハイブリッドエンジン（HE）を有効にするために、DSChatのいくつかのバグも修正しました。Optimized DSChatとOpenRLHFを使用して1024のプロンプトを1つのPPOエポックでトレーニングするのにかかる平均時間（秒）は次のとおりです：

| **サイズ** | **NVIDIA A800-80GB GPU** | **Optimized DSChat（ハイブリッドエンジン付き）** | **OpenRLHF** | **スピードアップ** |
| :---: | :---: | :---: | :---: | :---: |
| 7B | 16 | 855.09 | 471.11 | 1.82x |
| 13B | 32 | 1528.93 | 608.93 | 2.5x |
| 34B | 32 | 3634.98 | 1526.4 | 2.4x |
| 70B | 32 | 10407.0 | 4488.53 | 2.3x |

> [!NOTE]
> データは古くなっています。再テストのためにパフォーマンスチューニングセクションを参照してください。

### パフォーマンスチューニングガイド

最適なパフォーマンスを達成するために、vLLMエンジンにより多くのノードを割り当てることをお勧めします。例えば、32個のA100 GPUを持つ70Bモデルの場合、16個のA100 GPUをvLLMエンジンに割り当て、8個のGPUをActorモデルに、残りの8個のGPUをCriticモデルに割り当てることをお勧めします。さらに、`--colocate_critic_reward`、`--colocate_actor_ref`オプションを有効にしてノードをマージします。最後に、`rollout_micro_batch_size`（およびvLLMエンジンのTPサイズを最小化）を可能な限り増やすべきです。トレーニングフェーズでは、より大きな`--micro_train_batch_size`が望ましく、`--packing_samples`を有効にします。十分なGPUがある場合、`--adam_offload`を無効にし、`--overlap_comm`を有効にします。マルチノードRLHFの場合、vLLM 0.6.4+で`--vllm_sync_backend nccl`を使用してください。

## OpenRLHFを使用している企業と組織

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

## 参加方法

**参加方法**

1. janhu9527@gmail.comにメールを送るか、[GitHub Organization](https://github.com/OpenRLHF)に参加してください。以下の詳細を含めてください：
   - あなたの名前
   - あなたのGitHubユーザー名
   - あなたの興味のある分野
   - NLPおよび/またはAIに関連するスキルと経験
1. 公式GitHub[OpenRLHF ↗](https://github.com/OpenRLHF/OpenRLHF)プロジェクトページを通じて参加することもできます。貢献したい興味についてのissueを作成するだけで、私たちが連絡します。

**何ができるか**

1. チームに参加し、OpenRLHFプロジェクトの開発に参加します。
1. プロジェクトに貢献するためにプルリクエストを提出します。
1. ドキュメントの改善、バグの修正、新機能の作成を手伝います。
1. プロジェクトを共有し、コミュニティの成長を支援します。

## スポンサー

スポンサーシップは、OpenRLHFの維持と改善に役立ちます。このプロジェクトが役立つと感じた場合は、スポンサーを検討してください。[Open Collective ↗](https://opencollective.com/OpenRLHF)でスポンサーになることができます。

## スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=OpenRLHF/OpenRLHF&type=Date)](https://star-history.com/#OpenRLHF/OpenRLHF&Date)

## 貢献者

すべての貢献者に感謝します！貢献したい場合は、プルリクエストを作成するか、issueを作成してください。

<a href="https://github.com/OpenRLHF/OpenRLHF/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=OpenRLHF/OpenRLHF" />
</a>

## 参考文献と謝辞

AIおよびNLP分野への貢献に対して、以下のプロジェクトおよび組織に感謝します：

- [Hugging Face Transformers ↗](https://github.com/huggingface/transformers)
- [OpenAI GPT ↗](https://github.com/openai/gpt-3)
- [LLaMA ↗](https://llama.meta.com/)
- [DeepSpeed ↗](https://github.com/microsoft/DeepSpeed)
- [Ray ↗](https://github.com/ray-project/ray)

私たちのプロジェクトは、[ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)および[DeepSpeedChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)にも感謝します。プロジェクトの初期段階で、彼らのコードデザインを参考にしました。

(2024/7) 私たちのGitHub組織はOpenLLMAIからOpenRLHFに変更されました。

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

*OpenRLHF © 2025 OpenRLHF. All Rights Reserved.*
