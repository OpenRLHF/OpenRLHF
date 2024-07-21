set -x
export CUDA_VISIBLE_DEVICES=0

reward_pretrain=./ckpt/tiny_llama/tiny_llama_rm

read -r -d '' training_commands <<EOF
openrlhf.cli.utils_for_server \
    --reward_pretrain ${reward_pretrain} \
    --port 5000

EOF

if [[ ${1} != "slurm" ]]; then
  export PATH=$HOME/.local/bin/:$PATH
  deepspeed --module $training_commands
fi