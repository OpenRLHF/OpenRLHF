import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Tuple, Dict
from difflib import SequenceMatcher


def reward_normalization(objs):
    rewards = [float(obj["reward"]) for obj in objs]
    rewards = torch.tensor(rewards, dtype=torch.float64)
    rewards = (rewards - rewards.mean()) / rewards.std()
    for i, obj in enumerate(objs):
        obj["reward"] = rewards[i].item()


# Conditional SFT
# See https://arxiv.org/abs/2308.12050
DEFAULT_REWARD_PROMPT = "{input} <rm_score>: {reward} "


def conditional_sft_processor(args, objs):
    if "reward_template" not in args or args.reward_template is None:
        reward_template = DEFAULT_REWARD_PROMPT
    else:
        reward_template = args.reward_template
    assert "{input}" in reward_template
    assert "{reward}" in reward_template

    if args.normalize_reward:
        reward_normalization(objs)

    for obj in tqdm(objs, desc="Conditional SFT process..."):
        input = obj["input"]
        reward = "{:.2f}".format(float(obj["reward"]))
        input = reward_template.replace("{reward}", reward).replace("{input}", input)
        obj["input"] = input

    return objs


# Rejection Sampling
# See https://arxiv.org/abs/2307.09288
def rejection_sampling_processor(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {"output": output, "reward": reward}
        elif reward > out[input]["reward"]:
            out[input]["reward"] = reward
            out[input]["output"] = output

    return [{"input": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]


# Iterative DPO
# See https://github.com/RLHFlow/Online-RLHF/blob/main/run_loop.sh
def iterative_dpo_processor(args, objs):
    out = {}
    for obj in tqdm(objs, desc="Iterative DPO process...."):
        input = obj["input"]
        output = obj["output"]
        reward = float(obj["reward"])

        if input not in out:
            out[input] = {
                "output": output,
                "chosen": output,
                "chosen_reward": reward,
                "rejected": output,
                "rejected_reward": reward,
            }
        elif reward > out[input]["chosen_reward"]:
            out[input]["chosen_reward"] = reward
            out[input]["chosen"] = output
        elif reward < out[input]["rejected_reward"]:
            out[input]["rejected_reward"] = reward
            out[input]["rejected"] = output

    return [
        {
            "prompt": k,
            "chosen": v["chosen"],
            "chosen_reward": v["chosen_reward"],
            "rejected": v["rejected"],
            "rejected_reward": v["rejected_reward"],
        }
        for k, v in out.items()
    ]


PROCESSORS = {
    "rs": rejection_sampling_processor,
    "csft": conditional_sft_processor,
    "iter_dpo": iterative_dpo_processor,
}


def get_processor(name):
    if name in PROCESSORS:
        return PROCESSORS[name]
    else:
        raise ValueError(f"Processor {name} does not exist.")


def get_changed_unchanged_mask(
    chosen_tokens_batch: torch.Tensor,
    rejected_tokens_batch: torch.Tensor,
) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Compute boolean masks for changed and unchanged token indices in a batch,
    optimized for tensor operations.

    Args:
        chosen_tokens_batch: List of token-id lists for 'chosen' responses, length B.
        rejected_tokens_batch: List of token-id lists for 'rejected' responses, length B.

    Returns:
        changed_mask: BoolTensor of shape [B, L], True where chosen tokens were changed.
        unchanged_mask: BoolTensor of shape [B, L], True where chosen tokens unchanged.
    """
    padded_seq_len = chosen_tokens_batch.shape[-1]
    batch_size = chosen_tokens_batch.shape[0]
    changed_mask = torch.zeros((batch_size, padded_seq_len), dtype=torch.bool)
    unchanged_mask = torch.zeros((batch_size, padded_seq_len), dtype=torch.bool)

    chosen_list = chosen_tokens_batch.tolist()
    rejected_list = rejected_tokens_batch.tolist()
    for b in range(batch_size):
        lc = chosen_list[b]
        lr = rejected_list[b]
        mathcer = SequenceMatcher(None, lc, lr)
        for tag, i1, i2, _, _ in mathcer.get_opcodes():
            if tag == 'equal':
                unchanged_mask[b, i1:i2] = True
            else:
                changed_mask[b, i1:i2] = True

    return changed_mask, unchanged_mask