#!/bin/bash
# VLM (Vision-Language Model) RLHF Training Example
#
# This script demonstrates RLHF training for VLM models (tested with Qwen3.5)
# using the geometry3k multimodal math dataset.
#
# Key VLM features:
# - Auto-detects VLM models via vision_config in HuggingFace config
# - Loads full VLM (AutoModelForImageTextToText)
# - Uses AutoProcessor for correct image token insertion
# - Passes images to vLLM for multimodal generation
# - --freeze_visual_encoder: freeze vision encoder, only sync language model weights to vLLM
#
# Dataset format (JSONL):
#   {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Find x."}]}],
#    "images": ["/path/to/image.png"],
#    "label": "3"}
#
# VLM-specific parameters:
#   --image_key images               # Dataset key for image paths/URLs
#   --max_images_per_prompt 1        # Max images per prompt for vLLM
#   --freeze_visual_encoder          # Freeze vision encoder (only train language model)

set -x

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Model and Dataset Configuration
MODEL_PATH="Qwen/Qwen3.5-2B"
DATASET_PATH="hiyouga/geometry3k"   # or local JSONL file with images
SAVE_PATH="${WORK_DIR}/exp/Qwen3.5-2B-VLM-RLHF"

# Math reward function for answer verification
REWARD_FUNC_PATH="${WORK_DIR}/examples/python/math_reward_func.py"

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.7 \
   --init_kl_coef 0 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator reinforce_baseline \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url ${REWARD_FUNC_PATH} \
   --save_path ${SAVE_PATH} \
   --ckpt_path "${SAVE_PATH}/ckpt" \
   --save_hf_ckpt \
   --save_steps 5 \
   --eval_steps -1 \
   --micro_train_batch_size 1 \
   --micro_rollout_batch_size 1 \
   --train_batch_size 1024 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --num_episodes 100 \
   --max_len 4096 \
   --max_new_tokens 2048 \
   --zero_stage 3 \
   --param_dtype bf16 \
   --actor_learning_rate 2e-6 \
   --prompt_data ${DATASET_PATH} \
   --input_key prompt \
   --label_key label \
   --image_key images \
   --max_images_per_prompt 1 \
   --freeze_visual_encoder \
   --apply_chat_template \
   --gradient_checkpointing \
   --attn_implementation eager \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# VLM Notes:
#
# Tested VLM models:
#   - Qwen3.5 (0.8B/2B/4B/9B/27B) - hybrid linear+full attention
#
# Other VLMs with ForConditionalGeneration architecture (e.g. Gemma4,
# LLaVA, InternVL) are auto-detected via vision_config but not yet tested.
#
# With --freeze_visual_encoder, only language model is trained.
# Weight sync to vLLM only transfers trainable parameters.
#
# Dataset format:
#   The dataset should have an image field (configurable via --image_key)
#   containing paths or URLs to images. Each prompt can reference images
#   using {"type": "image"} content items in the chat message format.
#
# Multi-image support:
#   Set --max_images_per_prompt N for prompts with multiple images.
#   Images in the dataset should be a list of paths/URLs.
#
# Attention implementation:
#   Use --attn_implementation eager for models with linear attention (Qwen3.5).
#   Flash attention may not support packed sequences with linear attention layers.
