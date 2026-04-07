set -x

# Run from the OpenRLHF repo root so this path resolves.
readonly HEURISTICS="examples/python/es_countdown_heuristic.py"

readonly SAVE_PATH=".checkpoint"
readonly ES_BATCH_SIZE=200
readonly VLLM_MAX_NUM_SEQS="${ES_BATCH_SIZE}"
module purge
module load StdEnv 2>/dev/null || true
module load gcc/12.2.0 2>/dev/null || true
module load cuda/13.0.1


if [[ ${1} != "slurm" ]]; then
   VLLM_MAX_SEQ_ARG=()
   if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
      VLLM_MAX_SEQ_ARG=(--vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}")
   fi
   uv run --no-project python -m openrlhf.cli.train_es_ray \
      --save_path "${SAVE_PATH}" \
      --use_tensorboard "${SAVE_PATH}/tensorboard" \
      --pretrain Qwen/Qwen2.5-3B-Instruct \
      --prompt_data "dkoplow/ZKL-cd" \
      --eval_dataset "dkoplow/ZKL-cd" \
      --input_key prompt \
      --label_key label \
      --apply_chat_template \
      --heuristics "${HEURISTICS}" \
      --vllm_num_engines 4 \
      --vllm_tensor_parallel_size 1 \
      --vllm_gpu_memory_utilization 0.9 \
      "${VLLM_MAX_SEQ_ARG[@]}" \
      --population_size 11 \
      --es_std 0.001 \
      --es_optimizer SGD \
      --es_optimizer_params '{"lr": 0.001}' \
      --es_clip_grad_norm 0.0 \
      --batch_size "${ES_BATCH_SIZE}" \
      --max_samples 204800 \
      --max_eval_samples 200 \
      --max_len 1536 \
      --temperature 0.0 \
      --top_k -1 \
      --min_new_tokens 1 \
      --no-skip_special_tokens \
      --stop_token "</answer>" \
      --n_samples_per_prompt 1 \
      --max_epochs 1 \
      --save_steps 8 \
      --logging_steps 1 \
      --eval_steps 2 \
      --tensorboard_text_samples 3 \
      --tensorboard_text_max_chars 12000 \
      --seed 42
fi
