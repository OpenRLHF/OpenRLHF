import argparse
import hmac
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.utils.config import hierarchize
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def build_parser():
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward.model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument(
        "--reward.normalize_enable", action="store_true", default=False, help="Enable Reward Normalization"
    )
    parser.add_argument("--ds.value_head_prefix", type=str, default="score")
    parser.add_argument("--data.max_len", type=int, default=2048)

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Multi-node deploys must set --host to a reachable address explicitly.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Shared secret for POST /get_reward (or set OPENRLHF_RM_API_KEY). "
        "When set, require Authorization: Bearer <key> or X-Api-Key.",
    )

    # Performance
    parser.add_argument("--ds.load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--ds.param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type",
    )
    parser.add_argument(
        "--ds.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument(
        "--ds.experts_implementation",
        type=str,
        default=None,
        choices=["eager", "batched_mm", "grouped_mm", "deepgemm"],
        help="MoE expert computation strategy passed to transformers from_pretrained (default: auto — transformers picks grouped_mm when supported, else eager)",
    )
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)
    return parser


class RewardModelProxy:
    def __init__(self, args):
        from openrlhf.models import get_llm_for_sequence_regression
        from openrlhf.utils import get_tokenizer

        self.reward_model = get_llm_for_sequence_regression(
            args.reward.model_name_or_path,
            "reward",
            normalize_reward=args.reward.normalize_enable,
            attn_implementation=args.ds.attn_implementation,
            experts_implementation=args.ds.experts_implementation,
            param_dtype=args.ds.param_dtype,  # default: bf16
            load_in_4bit=args.ds.load_in_4bit,
            value_head_prefix=args.ds.value_head_prefix,
            device_map="auto",
            packing_samples=args.ds.packing_samples,
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward.model_name_or_path,
            self.reward_model,
            "left",
            None,
            use_fast=not args.data.disable_fast_tokenizer,
        )
        self.max_length = args.data.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        import torch

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


def create_app(reward_model, api_key=None):
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        if api_key:
            x_key = request.headers.get("x-api-key") or ""
            auth = request.headers.get("authorization") or ""
            bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            ok = (bool(bearer) and hmac.compare_digest(bearer, api_key)) or (
                bool(x_key) and hmac.compare_digest(x_key, api_key)
            )
            if not ok:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards, "scores": rewards, "extra_logs": {"dummy_scores": rewards}}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    return app


if __name__ == "__main__":
    import uvicorn

    args = hierarchize(build_parser().parse_args())

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    api_key = args.api_key or os.environ.get("OPENRLHF_RM_API_KEY")
    if not api_key:
        logger.warning("POST /get_reward is unauthenticated; set --api-key or OPENRLHF_RM_API_KEY")

    reward_model = RewardModelProxy(args)
    app = create_app(reward_model, api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
