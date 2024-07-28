set -x

python -m openrlhf.cli.serve_rm \
    --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
    --port 5000