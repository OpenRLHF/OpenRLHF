

cd /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/OpenRLHF
. /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/env/openrlhf/bin/activate
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/data


ROLLOUT_BS=256
N_SAMPLES_PER_PROMPT=16
TEMPERATURE=0.4
NUM_EPISODES=1
KL_COEF=0.001
BS=128
EP=1
LR=5e-7


TRIAL_NAME=rl.reinforce_qwen.math.7b.ins_gair.3k_rbs${ROLLOUT_BS}.n${N_SAMPLES_PER_PROMPT}.t${TEMPERATURE}es${NUM_EPISODES}.kl${KL_COEF}_bs${BS}.ep${EP}.lr${LR}

DATA_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/data/training_data/rl_prompt/gair_3k
POLICY_MODEL_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/model/Qwen2.5-Math-7B-Instruct
SAVE_PATH=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/hyzou/math/ckpts/qwen2.5-7b-instruct/$TRIAL_NAME


# start rm
if [ "$PET_NODE_RANK" -eq 0 ]; then
    python -m openrlhf.cli.serve_rm \
        --mode rule \
        --data_path $DATA_PATH \
        --port 5000 \
        --host $MASTER_ADDR &
fi

set -x
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265
if [ "$PET_NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node on $(hostname)"
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    echo "Starting Ray worker node on $(hostname)"
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi


sleep 10s


# start rl
if [ "$PET_NODE_RANK" -eq 0 ]; then
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://$MASTER_ADDR:$MASTER_PORT" \
    --runtime-env-json='{"working_dir": "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/liupengfei-24025/xfli/o1/OpenRLHF"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 2 \
    --pretrain $POLICY_MODEL_PATH \
    --remote_rm_url http://$MASTER_ADDR:5000/get_reward \
    --save_path $SAVE_PATH \
    --micro_train_batch_size 1 \
    --train_batch_size $BS \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size $ROLLOUT_BS \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $EP \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len 2048 \
    --generate_max_len 3072 \
    --advantage_estimator reinforce \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate $LR \
    --init_kl_coef $KL_COEF \
    --prompt_data $DATA_PATH \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --flash_attn \
    --vllm_sync_backend gloo \
    --gradient_checkpointing \
    --temperature $TEMPERATURE \
    --use_wandb "1badc41f0d258400b42ad079d39f9d58376dabf0" \
    --wandb_project Wiles \
    --wandb_group rl.reinforce \
    --wandb_run_name $TRIAL_NAME   
fi
