set -x

# Default bind is 127.0.0.1. Multi-node deploys must pass --host <reachable-ip>.
# Optional auth: --api-key or OPENRLHF_RM_API_KEY.
python -m openrlhf.cli.serve_rm \
    --reward.model_name_or_path OpenRLHF/Llama-3-8b-rm-700k \
    --host 127.0.0.1 \
    --port 5000 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation flash_attention_2 \
    --reward.normalize_enable \
    --data.max_len 8192 \
    --batch_size 16
