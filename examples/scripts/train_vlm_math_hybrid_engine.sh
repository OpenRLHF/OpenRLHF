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
#   --data.image_key images               # Dataset key for image paths/URLs
#   --data.max_images_per_prompt 1        # Max images per prompt for vLLM
#   --actor.freeze_visual_encoder          # Freeze vision encoder (only train language model)

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
   --ref.num_nodes 1 \
   --ref.num_gpus_per_node 4 \
   --actor.num_nodes 1 \
   --actor.num_gpus_per_node 4 \
   --vllm.num_engines 4 \
   --vllm.tensor_parallel_size 1 \
   --train.colocate_all \
   --vllm.gpu_memory_utilization 0.7 \
   --algo.kl.init_coef 0 \
   --algo.advantage.gamma 1.0 \
   --algo.kl.use_loss \
   --algo.kl.estimator k2 \
   --algo.advantage.estimator reinforce_baseline \
   --actor.model_name_or_path ${MODEL_PATH} \
   --reward.remote_url ${REWARD_FUNC_PATH} \
   --ckpt.output_dir ${SAVE_PATH} \
   --ckpt.path "${SAVE_PATH}/ckpt" \
   --ckpt.save_hf \
   --ckpt.save_steps 5 \
   --eval.steps -1 \
   --train.micro_batch_size 1 \
   --rollout.micro_batch_size 1 \
   --train.batch_size 1024 \
   --rollout.batch_size 128 \
   --rollout.n_samples_per_prompt 8 \
   --train.num_episodes 100 \
   --data.max_len 4096 \
   --rollout.max_new_tokens 2048 \
   --ds.zero_stage 3 \
   --ds.param_dtype bf16 \
   --actor.adam.lr 2e-6 \
   --data.prompt_dataset ${DATASET_PATH} \
   --data.input_key prompt \
   --data.label_key label \
   --data.image_key images \
   --data.max_images_per_prompt 1 \
   --actor.freeze_visual_encoder \
   --data.apply_chat_template \
   --actor.gradient_checkpointing_enable \
   --ds.attn_implementation eager \
   --vllm.sync_backend nccl \
   --vllm.enforce_eager \
   --vllm.enable_sleep \
   --ds.enable_sleep

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
#   Set --data.max_images_per_prompt N for prompts with multiple images.
#   Images in the dataset should be a list of paths/URLs.
#
# Attention implementation:
#   Use --ds.attn_implementation eager for models with linear attention (Qwen3.5).
#   Flash attention may not support packed sequences with linear attention layers.
