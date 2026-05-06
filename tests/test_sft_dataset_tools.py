import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_sft_dataset_module():
    # Keep these unit tests focused on sft_dataset instead of package-level imports.
    module_path = Path(__file__).parents[1] / "openrlhf" / "datasets" / "sft_dataset.py"
    module_name = "_sft_dataset_under_test"
    stubbed_module_names = ("openrlhf.utils", "openrlhf.utils.utils")
    sentinel = object()
    previous_modules = {name: sys.modules.get(name, sentinel) for name in stubbed_module_names}

    utils_package = types.ModuleType("openrlhf.utils")
    utils_package.__path__ = []
    utils_module = types.ModuleType("openrlhf.utils.utils")

    def zero_pad_sequences(*_args, **_kwargs):
        raise AssertionError("zero_pad_sequences is not exercised by these tests")

    utils_module.zero_pad_sequences = zero_pad_sequences

    sys.modules["openrlhf.utils"] = utils_package
    sys.modules["openrlhf.utils.utils"] = utils_module
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module

    return module


sft_dataset = _load_sft_dataset_module()
SFTDataset = sft_dataset.SFTDataset
_get_chat_template_kwargs = sft_dataset._get_chat_template_kwargs
preprocess_data = sft_dataset.preprocess_data


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up weather by city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


class RecordingTemplate:
    def __init__(self):
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))

        rendered_parts = []
        if "tools" in kwargs:
            rendered_parts.append("<tools>")
        for message in messages:
            rendered_parts.append(f"<{message['role']}>{message.get('content', '')}")
        if kwargs.get("add_generation_prompt"):
            rendered_parts.append("<assistant>")
        return "".join(rendered_parts)


class NoToolsTemplate:
    def __call__(self, messages, **kwargs):
        assert "tools" not in kwargs
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        if kwargs.get("add_generation_prompt"):
            rendered += "<assistant>"
        return rendered


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, apply_chat_template):
        self.apply_chat_template = apply_chat_template

    def __call__(self, text, **kwargs):
        return {"attention_mask": torch.ones((1, len(text)), dtype=torch.long)}


def test_get_chat_template_kwargs_omits_missing_and_none_tools():
    assert _get_chat_template_kwargs({}) == {}
    assert _get_chat_template_kwargs({"tools": None}) == {}


def test_get_chat_template_kwargs_passes_empty_and_populated_tools():
    empty_tools = []
    assert _get_chat_template_kwargs({"tools": empty_tools})["tools"] is empty_tools
    assert _get_chat_template_kwargs({"tools": TOOLS})["tools"] is TOOLS


def test_preprocess_data_passes_tools_for_split_prompt_response():
    template = RecordingTemplate()

    prompt, response = preprocess_data(
        {"input": "What is the weather?", "output": "Sunny.", "tools": TOOLS},
        input_key="input",
        output_key="output",
        apply_chat_template=template,
    )

    assert prompt == "<tools><user>What is the weather?<assistant>"
    assert response == "Sunny."
    assert len(template.calls) == 2
    assert all(call_kwargs["tools"] is TOOLS for _, call_kwargs in template.calls)


def test_preprocess_data_passes_tools_for_full_conversation():
    template = RecordingTemplate()
    row = {
        "input": [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "Sunny."},
        ],
        "tools": TOOLS,
    }

    prompt, response = preprocess_data(row, input_key="input", apply_chat_template=template)

    assert prompt == "<tools><user>What is the weather?<assistant>"
    assert response == "Sunny."
    assert len(template.calls) == 2
    assert all(call_kwargs["tools"] is TOOLS for _, call_kwargs in template.calls)


def test_preprocess_data_does_not_pass_tools_when_field_is_absent():
    prompt, response = preprocess_data(
        {"input": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        input_key="input",
        apply_chat_template=NoToolsTemplate(),
    )

    assert prompt == "<user>Hi<assistant>"
    assert response == "Hello"


def test_preprocess_data_passes_empty_tools_list():
    template = RecordingTemplate()

    prompt, response = preprocess_data(
        {"input": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], "tools": []},
        input_key="input",
        apply_chat_template=template,
    )

    assert prompt == "<tools><user>Hi<assistant>"
    assert response == "Hello"
    assert len(template.calls) == 2
    assert all(call_kwargs["tools"] == [] for _, call_kwargs in template.calls)


def test_process_data_multiturn_passes_tools_for_response_ranges():
    template = RecordingTemplate()
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.tokenizer = FakeTokenizer(template)
    dataset.pretrain_mode = False
    dataset.max_length = 1024
    dataset.multiturn = True
    dataset.input_template = None
    dataset.input_key = "input"
    dataset.output_key = None
    dataset.apply_chat_template = template

    processed = dataset.process_data(
        {
            "input": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "One"},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": "Two"},
            ],
            "tools": TOOLS,
        }
    )

    assert len(processed["response_ranges"]) == 2
    assert len(template.calls) == 6
    assert all(call_kwargs["tools"] is TOOLS for _, call_kwargs in template.calls)

    dataset.response_ranges = [processed["response_ranges"]]
    input_ids = torch.ones((1, len(processed["prompt"] + processed["response"])), dtype=torch.long)
    loss_mask = dataset.get_loss_mask(input_ids, 0)
    assert loss_mask.sum().item() > 0
