# 2-GPU sampled-mode OPD (--opd.topk == 0) on GSM8k, with held-out eval every 10 steps
# Substitude the model path and data
set -x
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
TEACHER_PATH=${TEACHER_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
DATA_DIR=${DATA_DIR:-"./data"}
TRAIN_DATASET=${TRAIN_DATASET:-"${DATA_DIR}/gsm8k_train.jsonl"}
EVAL_DATASET=${EVAL_DATASET:-"${DATA_DIR}/gsm8k_eval.jsonl"}
SAVE_PATH=${SAVE_PATH:-"./exp/gsm8k_sampled_opd_k3"}

[[ -n ${PYTHON_ENV} ]] && export PATH="${PYTHON_ENV}/bin:${PATH}"

deepspeed --num_gpus 2 --module openrlhf.cli.train_opd \
    --model.model_name_or_path ${MODEL_PATH} \
    --teacher.model_name_or_path ${TEACHER_PATH} \
    --data.prompt_dataset ${TRAIN_DATASET} \
    --data.input_key question \
    --data.label_key answer \
    --data.apply_chat_template \
    --data.max_samples 400 \
    --data.prompt_max_len 512 \
    --data.max_len 1024 \
    --generate.max_new_tokens 512 \
    --generate.temperature 1.0 \
    --generate.top_p 1.0 \
    --eval.dataset ${EVAL_DATASET} \
    --eval.split train \
    --eval.steps 10 \
    --opd.topk 0 \
    --opd.kl_estimator k3 \
    --train.micro_batch_size 4 \
    --train.batch_size 8 \
    --train.max_epochs 1 \
    --adam.lr 1e-6 \
    --ds.zero_stage 2 \
    --ds.param_dtype bf16 \
    --ds.attn_implementation sdpa \
    --model.gradient_checkpointing_enable \
    --ckpt.output_dir ${SAVE_PATH} \
    --ckpt.path "${SAVE_PATH}/ckpt" \
    --ckpt.save_steps 1000 \
    --logger.tensorboard_dir "${SAVE_PATH}/runs" \
    --logger.logging_steps 1