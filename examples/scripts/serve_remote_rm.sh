set -x

python -m openrlhf.cli.serve_rm \
    --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
    --port 5000 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --reward.normalize_enable \
    --data.max_len 8192 \
    --batch_size 16