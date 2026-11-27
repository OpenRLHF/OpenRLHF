from argparse import Namespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from openrlhf.datasets import SFTDataset


@pytest.mark.parametrize("multiturn", [False, True])
@pytest.mark.parametrize("long_final_prompt", [False, True])
@pytest.mark.parametrize("earlier_answer", [False, True])
def test_sft_keeps_trainable_turns_after_truncation(multiturn, long_final_prompt, earlier_answer):
    vocabulary = ["[UNK]", "[EOS]", "[PAD]", "user", "assistant", "hi", "good", "long", "answer"]
    backend = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} [EOS] {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    strategy = Namespace(
        args=Namespace(data=Namespace(input_key="messages", output_key=None, apply_chat_template=True))
    )
    messages = []
    if earlier_answer:
        messages.extend([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good"}])
    messages.extend(
        [
            {"role": "user", "content": "long " * (40 if long_final_prompt else 1)},
            {"role": "assistant", "content": "answer"},
        ]
    )
    dataset = SFTDataset(
        Dataset.from_list([{"messages": messages}]),
        tokenizer,
        max_length=16,
        strategy=strategy,
        num_processors=None,
        multiturn=multiturn,
    )
    expected = not long_final_prompt or (multiturn and earlier_answer)
    assert len(dataset) == int(expected)
    if expected:
        input_ids, attention_mask, loss_mask = dataset[0]
        assert input_ids.shape == attention_mask.shape == loss_mask.shape
        target_ids = input_ids.roll(-1, dims=-1)
        trained_ids = target_ids[loss_mask.bool()].tolist()
        assert tokenizer.convert_tokens_to_ids("long") not in trained_ids
        if multiturn and earlier_answer:
            assert tokenizer.convert_tokens_to_ids("good") in trained_ids
        if not long_final_prompt:
            assert tokenizer.convert_tokens_to_ids("answer") in trained_ids
        batch = dataset.collate_fn([dataset[0], dataset[0]])
        for original, collated in zip((input_ids, attention_mask, loss_mask), batch):
            torch.testing.assert_close(collated, torch.cat([original, original]))
