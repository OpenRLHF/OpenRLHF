set -x

python -m openrlhf.cli.serve_rm \
    --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
    --port 5000 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 8192 \
    --batch_size 16