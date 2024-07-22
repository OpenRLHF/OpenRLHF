import argparse
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer, get_strategy
# from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RMServer():
    """RM server"""

    def __init__(self):
        self.reward_model, self.tokenizer = self.init_rm_model(args)

    def init_rm_model(self, args, device='cuda'):
        strategy = get_strategy(args)
        strategy.setup_distributed()
        reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
        )
        reward_model = reward_model.to(device)
        tokenizer = get_tokenizer(args.reward_pretrain, reward_model, "left", strategy,
                                  use_fast=not args.disable_fast_tokenizer)

        return reward_model, tokenizer

    def get_rm_score_with_vllm(self, queries, responses=None):
        """TODO: use vllm to accelerate rm server. see. https://github.com/vllm-project/vllm/issues/6620"""
        pass

    def get_rm_score(self, queries, responses=None, max_length=2048, sep='\n\n'):
        if isinstance(queries, list) and isinstance(responses, list):
            queries = [q + sep + r for q, r in zip(queries, responses)]
        inputs = self.tokenize_fn(queries, max_length, device=self.reward_model.device)
        r = self.reward_model(inputs['input_ids'], inputs['attention_mask'])
        r = r.tolist()
        if len(r) == 1:
            return r[0]
        return r

    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--value_head_prefix", type=str, default="value_head")
    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # server
    args = parser.parse_args()

    app = Flask(__name__)
    rm_server = RMServer()


    @app.route('/get_rm_score', methods=['POST'])
    def get_rm_score():
        data = request.json
        queries = data.get("query")
        responses = data.get("response", None)
        max_length = data.get("max_length", 2048)
        scores = rm_server.get_rm_score(queries, responses, max_length)
        result = {'score': scores}
        return jsonify(result)


    app.run(port=args.port)
