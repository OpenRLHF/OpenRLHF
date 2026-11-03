from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from openrlhf.datasets import RewardDataset


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, max_length, **_kwargs):
        token_ids = {
            "prompt": 1,
            "common": 2,
            "chosen": 3,
            "rejected": 4,
            self.eos_token: self.eos_token_id,
        }
        input_ids = torch.tensor([[token_ids[token] for token in text.split()[:max_length]]])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _build_dataset(chosen, rejected, is_dpo):
    data = Dataset.from_list([{"prompt": "prompt ", "chosen": chosen, "rejected": rejected}])
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(
                prompt_key="prompt",
                chosen_key="chosen",
                rejected_key="rejected",
                apply_chat_template=False,
            )
        )
    )
    return RewardDataset(data, _Tokenizer(), 5, strategy, is_dpo=is_dpo, num_processors=None)


@pytest.mark.parametrize("is_dpo", [False, True])
def test_filters_pair_whose_difference_is_truncated(is_dpo):
    dataset = _build_dataset("common common common common chosen", "common common common common rejected", is_dpo)

    assert len(dataset) == 0


@pytest.mark.parametrize("is_dpo", [False, True])
def test_filters_pair_whose_last_difference_is_replaced_by_eos(is_dpo):
    dataset = _build_dataset("common common common chosen", "common common common rejected", is_dpo)

    assert len(dataset) == 0


@pytest.mark.parametrize("is_dpo", [False, True])
def test_keeps_pair_with_difference_inside_retained_tokens(is_dpo):
    dataset = _build_dataset("chosen common", "rejected common", is_dpo)

    assert len(dataset) == 1
    chosen_ids, _, rejected_ids, _, _ = dataset[0]
    assert not torch.equal(chosen_ids, rejected_ids)
