from openrlhf.models import Actor
from beamsearch import search as run_beamsearch
from transformers import AutoTokenizer
from openrlhf.utils import get_strategy
import json
import torch
from model import get_llm_for_sequence_regression

data_fpath = "../dataset/train_gsm8k_math.jsonl"

policy_fpath = "/apdcephfs_cq11/share_1603164/data/antewang/trained_models/Qwen-2.5-Math-1.5B_sft_star_fp32/checkpoint-1738"
critic_fpath = "/apdcephfs_cq11/share_1603164/data/antewang/trained_models/Qwen2.5-Math-1.5B_sft_value_ep1_bsz64_lr5e-6_fp32/checkpoint-1489"

tokenizer = AutoTokenizer.from_pretrained(policy_fpath)

actor = Actor(
        policy_fpath,
        bf16=False,
    )

critic = get_llm_for_sequence_regression(
        critic_fpath,
        "critic",
        normalize_reward=True,
        bf16=False,
        value_head_prefix="score",
        init_value_head=False,
    )

actor.model = actor.model.to(torch.device("cuda:0"))
critic = critic.to(torch.device("cuda:1"))

actor.model = actor.model.eval()
critic = critic.eval()

with open(data_fpath, "r") as f:
    data = [json.loads(line) for line in f.readlines()]

question = data[-1]["input"]
run_beamsearch(question, tokenizer, actor, critic)