import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import deepspeed

from transformers import AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_ds_config():
    zero_opt_dict = {
        "stage": 3,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {
            "device": "none",
            "pin_memory": True,
        },
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "train_micro_batch_size_per_gpu": 8,
        "train_batch_size": 128,
    }


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def main(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

    ds_config = get_ds_config()
    if args.enable_hf_deespeed and ds_config["zero_optimization"]["stage"] == 3:
        print("==========enable HfDeepSpeedConfig============")
        dschf = HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    engine, *_ = deepspeed.initialize(
        model=model,
        args={"local_rank": args.local_rank},
        config=ds_config,
        dist_init_required=True,
    )
    engine.eval()

    state_dict = torch.load(f"experience.pt", map_location="cpu")
    input_ids, attention_mask = state_dict["sequences"].to("cuda"), state_dict["attention_mask"].to("cuda")
    with torch.no_grad():
        output = engine(input_ids, attention_mask=attention_mask)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], input_ids[:, 1:])

    if torch.distributed.get_rank() == 0:
        print(log_probs)
        torch.save({"log_probs": log_probs}, f"result_{args.enable_hf_deespeed}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # For test-only, we use Llama-2-7b-hf instead of meta-llama/Llama-2-70b-hf
    parser.add_argument("--pretrain", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--enable_hf_deespeed", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
