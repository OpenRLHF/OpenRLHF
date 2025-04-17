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
- **Hybrid Engine**: OpenRLHFはHybrid Engineもサポートしており、すべてのトレーニングエンジンと推論エンジンがGPUを共有してリソースのアイドル状態を防ぎます。

詳細は[スライド](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit?usp=sharing) | [技術報告](https://arxiv.org/abs/2405.11143) | [ドキュメント](https://openrlhf.readthedocs.io/)をご覧ください。

## ニュース
- [2025/3] CMUの[2025年春の高度自然言語処理コース](https://cmu-l3.github.io/anlp-spring2025/)がOpenRLHFをRLHFフレームワークの教育事例として採用。
- [2025/2] [Logic-RL](https://arxiv.org/abs/2502.14768) と [PRIME](https://arxiv.org/abs/2502.01456) は、REINFORCE++ が訓練の安定性において GRPO より優れ、PPO より高速であることを示した。
- [2025/2] [LMM-R1](https://github.com/TideDra/lmm-r1) は OpenRLHF のフォークで、マルチモーダルタスクでの DeepSeek-R1 の再現のための高性能 RL インフラストラクチャを提供することを目的としています。
- [2025/2] MIT & Microsoft は OpenRLHF を使用して [On the Emergence of Thinking in LLMs I: Searching for the Right Intuition](https://arxiv.org/pdf/2502.06773) を提案しました。
- [2025/1] HKUSTは [OpenRLHF を使用して小規模モデルでの DeepSeek-R1-Zero と DeepSeek-R1 のトレーニング](https://github.com/hkust-nlp/simpleRL-reason)を再現しました。
- [2024/12] 私たちは😊 [REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models](https://www.researchgate.net/publication/387487679_REINFORCE_A_SIMPLE_AND_EFFICIENT_APPROACH_FOR_ALIGNING_LARGE_LANGUAGE_MODELS)を「提案」しました。
- [2024/12] [Notionブログ](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights#147d9a33ecc9806090f3d5c749d31f05)でPPO、REINFORCE++、GRPO、およびRLOOを分析しました。

## 特徴

- Rayに基づく分散[ PPO](./examples/scripts/train_ppo_llama_ray.sh)および[EINFORCE++/REINFORCE++-baseline/GRPO/RLOO](./examples/scripts/train_reinforce_llama_ray.sh)の実装。
- [Ray-based Reinforced Finetuning](./examples/scripts/train_ppo_llama_with_reward_fn.sh)
- RayとHybrid Engineに基づく[PPO](./examples/scripts/train_ppo_llama_ray_hybrid_engine.sh)および[REINFORCE++/REINFORCE++-baseline/GRPO/RLOO](./examples/scripts/train_reinforce_llama_ray_hybrid_engine.sh)のサポート (`--colocate_all_models`, `--vllm_enable_sleep` and `--vllm_gpu_memory_utilization 0.5`)
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
- Wandb（`--use_wandb`）およびTensorBoard（`--use_tensorboard`）によるログ記録のサポート。
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

# vLLM加速を使用する場合（vLLM 0.8.3をインストール）
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
>vLLM 0.8.3以降の使用をお勧めします。
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

トレーニングおよびテストデータセットの指定方法

テストデータセットのパスは ``--eval_dataset {name or path}`` を使用して設定してください。

> [!NOTE]
> ``JSON key`` オプションは特定のデータセットに依存します。詳細は [Reward Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/reward_dataset.py#L10) および [SFT Dataset](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/datasets/sft_dataset.py#L9) を参照してください。

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

### RayとvLLMを使用したPPO/REINFORCE++

RLHFトレーニング速度を向上させるか、70Bモデルをサポートするために、RayとvLLM加速を使用したPPOを使用できます (Hybrid Engine)

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
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 2 \
  --colocate_all_models \
  --vllm_gpu_memory_utilization 0.5 \
  --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
  --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
  --save_path /openrlhf/examples/test_scripts/final/llama3-8b-rlhf \
  --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf \
  --save_hf_ckpt \
  --micro_train_batch_size 8 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 1024 \
  --n_samples_per_prompt 1 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --max_samples 100000 \
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
  --gradient_checkpointing \
  --packing_samples \
  --vllm_sync_backend nccl \
  --enforce_eager \
  --vllm_enable_sleep \
  --deepspeed_enable_sleep \
  --use_wandb {wandb_token}

# REINFORCE++  | RLOO | REINFORCE++-baseline | GRPO | Dr. GRPO をサポート
# --advantage_estimator reinforce | rloo | reinforce_baseline | group_norm | dr_grpo

# --init_kl_coef を 0 に設定すると参照モデルが起動しません

# リモート報酬モデル（HTTP）をサポート
# --remote_rm_url http://localhost:5000/get_reward

# N個のサンプルをサポート
# --n_samples_per_prompt 4
```
> [!NOTE]
> また、``setup_commands``を使用してRayに環境を自動的にデプロイさせることもできます。例：`--runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}'`

> [!NOTE]
> OpenRLHFのRLOOとREINFORCE++-baselineはREINFORCE++に基づく修正版です：
> - REINFORCE++は、PPOの主要な最適化技術（アドバンテージ正規化やPPO-clipロスなど）を統合し、criticネットワークの必要性を排除します。
> - REINFORCE++-baselineは、`同じプロンプトから生成された複数のサンプルの平均報酬`をベースラインとして報酬を再形成します（グローバルバッチ正規化 `/std` を使用）。
> - OpenRLHFのRLOOは、`トークンごとのKL報酬`を導入し、`PPO-clipロス`を使用することで元のバージョンを修正しています。
> - Dr. GRPOは、GRPOのグループ正規化 `/std` を削除します。

> [!NOTE]
> deepspeedがGPUデバイスをセットアップする際にインデックス範囲外のエラーが発生した場合は、環境変数 [`RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES`](openrlhf/trainer/ray/utils.py) を設定することで一時的な解決策として対応できます。
>   ```bash
>   # NVIDIA GPUの場合：
>   export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
>   ```

サポートされているアルゴリズムの起動スクリプトとドキュメントは[example/scripts](./examples/scripts/)および[Documents - Usage](https://openrlhf.readthedocs.io/en/latest/usage.html)にあります。

### Reinforced Fine-tuning

OpenRLHFは、便利で効率的なReinforced Fine-tuningをサポートしています。カスタム `reward_func` 関数を含む[ファイル](./examples/scripts/reward_func.py)を実装し、そのパスを `