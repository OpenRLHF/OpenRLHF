from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from chatgpt.experience_maker.base import Experience


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    returns: (1)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    batch_size = experience.sequences.size(0)
    batch_kwargs = [{} for _ in range(batch_size)]
    keys = ('sequences', 'action_log_probs', 'values', 'returns', 'advantages', 'attention_mask', 'action_mask')
    for key in keys:
        value = getattr(experience, key)
        vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]['info'] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            batch_kwargs[i]['info'][k] = vv.item()

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = 'left') -> torch.Tensor:
    assert side in ('left', 'right')
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == 'left' else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    keys = ('sequences', 'action_log_probs', 'values', 'returns', 'advantages', 'attention_mask', 'action_mask')
    for key in keys:
        vals = [getattr(item, key) for item in items]
        batch_data = zero_pad_sequences(vals, 'left')
        kwargs[key] = batch_data

    kwargs['info'] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs['info'][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, value, ret, adv, att_mask, act_mask = \
            item.sequences, item.action_log_probs, item.values, item.returns, \
            item.advantages, item.attention_mask, item.action_mask
        right_pad = (1 - act_mask.int()).sum()
        left_pad = (1 - att_mask.int()).sum() - right_pad

        right_pad = None if right_pad == 0 else -right_pad
        item.sequences, item.action_log_probs, item.values, item.returns, \
        item.advantages, item.attention_mask, item.action_mask = \
            seq[left_pad:right_pad], act_log_prob[:right_pad], value[:right_pad], ret[:right_pad], \
            adv[:right_pad], att_mask[left_pad:right_pad], act_mask[:right_pad]
    return items
