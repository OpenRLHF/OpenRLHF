import copy
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from openrlhf.datasets.sft_dataset import SFTDataset


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "user": 3, "assistant": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tokenizer


@pytest.mark.parametrize("response_format", ["dict", "list", "conversation"])
@pytest.mark.parametrize("max_length", [24, 128])
def test_split_multiturn_matches_full_conversation(tokenizer, response_format, max_length):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": "long answer " * 20},
    ]
    split_at = 1 if response_format == "conversation" else 3
    response = messages[split_at:]
    if response_format == "dict":
        response = response[0]
    row = {"messages": messages[:split_at], "answer": response}
    original = copy.deepcopy(row)
    split = SFTDataset(
        Dataset.from_list([row]),
        tokenizer,
        max_length,
        SimpleNamespace(
            args=SimpleNamespace(
                data=SimpleNamespace(input_key="messages", output_key="answer", apply_chat_template=True)
            )
        ),
        num_processors=None,
        multiturn=True,
    )
    full = SFTDataset(
        Dataset.from_list([{"messages": messages}]),
        tokenizer,
        max_length,
        SimpleNamespace(
            args=SimpleNamespace(data=SimpleNamespace(input_key="messages", output_key=None, apply_chat_template=True))
        ),
        num_processors=None,
        multiturn=True,
    )
    assert len(split) == len(full) == 1
    for actual, expected in zip(split[0], full[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert split.response_ranges == full.response_ranges
    assert split[0][2].sum() > 0
    first = split.process_data(row)
    assert split.process_data(row) == first
    assert row == original
    assert split.output_key == "answer"


@pytest.mark.parametrize("pretrain", [False, True])
def test_split_fix_preserves_existing_text_modes(tokenizer, pretrain):
    row = {"prompt": "user question", "answer": "assistant answer"}
    original = copy.deepcopy(row)
    dataset = SFTDataset(
        Dataset.from_list([row]),
        tokenizer,
        32,
        SimpleNamespace(
            args=SimpleNamespace(
                data=SimpleNamespace(input_key="prompt", output_key="answer", apply_chat_template=False)
            )
        ),
        num_processors=None,
        pretrain_mode=pretrain,
    )
    processed = dataset.process_data(row)
    assert processed["prompt"] == row["prompt"]
    assert processed["response"] == row["answer"]
    assert processed["prompt_ids_len"] == (0 if pretrain else 2)
    ids, attention, loss_mask = dataset[0]
    assert attention.all() and loss_mask.sum() > 0
    assert ids.shape == attention.shape == loss_mask.shape
    assert row == original
