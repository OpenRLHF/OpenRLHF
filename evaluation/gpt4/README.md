# GPT-4 Evaluation

We evaluated the models (see [checkpoints]( https://huggingface.co/OpenLLMAI/openrlhf_checkpoint)) trained on OpenRLHF using GPT-4 based on 160 [prompts](./benchmark.jsonl) from LMSys. The basic idea was to compare the RLHF fine-tuned model against the OCRA SFT model, using the following method:

1. Generate responses

```shell
deepspeed batch_inference.py
    --eval_task generate \
    --pretrain OpenLLMAI/Llama-2-7b-sft-model-ocra-500k \
    --bf16 \
    --max_len 2048 \
    --dataset ./data/benchmark.jsonl \
    --dataset_probs 1.0 \
    --max_samples 160 \
    --zero_stage 0 \
    --micro_batch_size 16 \
    --greedy_sampling \
    --output_path {output_json_path}
    # --enable_dt (for Decision Transformer models)
```

2. GPT-4 evaluation

```shell
pip install alpaca_eval

# convert .jsonl to .json
sed -i '1s/^/[/; $!s/$/,/; $s/$/]/' {output_json_path}
sed -i 's/"input"/"instruction"/g' {output_json_path}

alpaca_eval --model_outputs {output_json_path} --annotators_config alpaca_eval_gpt4 --reference_outputs sft.json
```

## Results

Below are the results compared to the OCRA SFT model

|        | Win | Lose  | Tie  | 
|  ----  | ----  |  ----  | ----  | 
| PPO  |  |   |  | 
| DPO  |  |   |  | 
| DT  |  |   |  |