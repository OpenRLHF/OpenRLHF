import argparse

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
            packing_samples=args.packing_samples,
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries, prompts):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normalization")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--packing_samples", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    # server
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        prompts = data.get("prompts")
        rewards = reward_model.get_reward(queries, prompts)
        result = {"rewards": rewards, "scores": rewards, "extra_logs": {"dummy_scores": rewards}}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
