"""Check SFT chat rendering using an actual tokenizer without the training stack."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


@pytest.mark.parametrize(
    "prompt,completion",
    [
        ("question", "answer"),
        ([{"role": "user", "content": "question"}], "answer"),
        ("question", [{"role": "assistant", "content": "answer"}]),
        ([{"role": "user", "content": "question"}], [{"role": "assistant", "content": "answer"}]),
    ],
)
def test_sft_chat_input_and_output_can_use_different_formats(monkeypatch, prompt, completion):
    utils = ModuleType("openrlhf.utils.utils")
    utils.zero_pad_sequences = None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    source = Path(__file__).resolve().parents[1] / "openrlhf" / "datasets" / "sft_dataset.py"
    spec = importlib.util.spec_from_file_location("_sft_chat_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template=(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        ),
    )
    rendered_prompt, rendered_completion = module.preprocess_data(
        {"input": prompt, "output": completion}, output_key="output", apply_chat_template=tokenizer.apply_chat_template
    )

    assert rendered_prompt == "user: question\nassistant: "
    assert rendered_completion == "answer\n"
