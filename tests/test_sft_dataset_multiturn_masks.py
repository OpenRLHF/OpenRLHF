import copy
import importlib.util
import sys
import types
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _zero_pad_sequences(sequences, side="left", value=0, stack=False):
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0) if stack else torch.cat(padded_sequences, dim=0)


fake_utils = types.ModuleType("openrlhf.utils")
fake_utils_utils = types.ModuleType("openrlhf.utils.utils")
fake_utils_utils.zero_pad_sequences = _zero_pad_sequences

_SFT_DATASET_PATH = Path(__file__).resolve().parents[1] / "openrlhf" / "datasets" / "sft_dataset.py"
_SFT_SPEC = importlib.util.spec_from_file_location("sft_dataset_under_test", _SFT_DATASET_PATH)
_SFT_MODULE = importlib.util.module_from_spec(_SFT_SPEC)
_ORIGINAL_UTILS = sys.modules.get("openrlhf.utils")
_ORIGINAL_UTILS_UTILS = sys.modules.get("openrlhf.utils.utils")
try:
    sys.modules["openrlhf.utils"] = fake_utils
    sys.modules["openrlhf.utils.utils"] = fake_utils_utils
    _SFT_SPEC.loader.exec_module(_SFT_MODULE)
finally:
    if _ORIGINAL_UTILS is None:
        sys.modules.pop("openrlhf.utils", None)
    else:
        sys.modules["openrlhf.utils"] = _ORIGINAL_UTILS

    if _ORIGINAL_UTILS_UTILS is None:
        sys.modules.pop("openrlhf.utils.utils", None)
    else:
        sys.modules["openrlhf.utils.utils"] = _ORIGINAL_UTILS_UTILS
SFTDataset = _SFT_MODULE.SFTDataset


class CharTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(
        self,
        text,
        max_length=None,
        padding=False,
        truncation=False,
        return_tensors=None,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del padding, add_special_tokens
        ids = [ord(ch) for ch in text]
        offsets = [(idx, idx + 1) for idx in range(len(text))]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]

        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            result = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
            if return_offsets_mapping:
                result["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
            return result

        result = {"input_ids": ids, "attention_mask": attention_mask}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class QwenLikeTokenizer(CharTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        del kwargs, return_assistant_tokens_mask
        rendered = self._render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return rendered

        encoded = self(
            rendered,
            max_length=max_length,
            truncation=truncation,
            return_tensors=None,
            add_special_tokens=False,
        )
        return encoded if return_dict else encoded["input_ids"]

    def _render(self, messages, add_generation_prompt=False):
        last_query_index = len(messages) - 1
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            content = message.get("content", "")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and not content.startswith("<tool_response>")
            ):
                last_query_index = index
                break

        rendered = []
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            content = content if isinstance(content, str) else ""
            if role in ("system", "user"):
                rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            elif role == "assistant":
                reasoning_content = ""
                if isinstance(message.get("reasoning_content"), str):
                    reasoning_content = message["reasoning_content"]
                elif "</think>" in content:
                    reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n")

                if index > last_query_index:
                    if index == len(messages) - 1 or reasoning_content:
                        rendered.append(
                            "<|im_start|>assistant\n"
                            f"<think>\n{reasoning_content.strip(chr(10))}\n</think>\n\n"
                            f"{content.lstrip(chr(10))}"
                            "<|im_end|>\n"
                        )
                    else:
                        rendered.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
                else:
                    rendered.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")
        return "".join(rendered)


class CountingQwenLikeTokenizer(QwenLikeTokenizer):
    def __init__(self):
        self.marked_render_calls = 0

    def apply_chat_template(self, messages, *args, **kwargs):
        if any("__OPENRLHF_ASSISTANT_SPAN_" in str(message) for message in messages):
            self.marked_render_calls += 1
        return super().apply_chat_template(messages, *args, **kwargs)


class SimpleTemplateTokenizer(CharTokenizer):
    eos_token = "</s>"
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        del return_assistant_tokens_mask, kwargs
        rendered = "".join(f"<{message['role']}>\n{message.get('content', '')}</s>\n" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>\n"
        if not tokenize:
            return rendered

        encoded = self(
            rendered,
            max_length=max_length,
            truncation=truncation,
            return_tensors=None,
            add_special_tokens=False,
        )
        return encoded if return_dict else encoded["input_ids"]


class NativeMaskTokenizer(SimpleTemplateTokenizer):
    def __init__(self):
        self.native_mask_calls = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        del kwargs
        rendered = []
        assistant_mask = []
        for message in messages:
            prefix = f"<{message['role']}>\n"
            content = message.get("content", "")
            suffix = "</s>\n"
            rendered.append(prefix + content + suffix)
            if message["role"] == "assistant":
                assistant_mask.extend([0] * len(prefix))
                assistant_mask.extend([1] * (len(content) + len(suffix.rstrip("\n"))))
                assistant_mask.extend([0])
            else:
                assistant_mask.extend([0] * len(prefix + content + suffix))

        if add_generation_prompt:
            rendered.append("<assistant>\n")
            assistant_mask.extend([0] * len("<assistant>\n"))

        rendered = "".join(rendered)
        if not tokenize:
            return rendered

        encoded = self(
            rendered,
            max_length=max_length,
            truncation=truncation,
            return_tensors=None,
            add_special_tokens=False,
        )
        if return_dict:
            if return_assistant_tokens_mask:
                self.native_mask_calls += 1
                encoded["assistant_masks"] = assistant_mask[: len(encoded["input_ids"])]
            return UserDict(encoded)
        return encoded["input_ids"]


class EmptyNativeMaskTokenizer(NativeMaskTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        encoded = super().apply_chat_template(*args, **kwargs)
        if kwargs.get("tokenize") and kwargs.get("return_dict") and "assistant_masks" in encoded:
            encoded["assistant_masks"] = [0] * len(encoded["assistant_masks"])
        return encoded


def _make_dataset(tokenizer, max_length=4096, output_key=None):
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.tokenizer = tokenizer
    dataset.strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(
                input_key="input",
                output_key=output_key,
                apply_chat_template=True,
                tokenizer_chat_template=None,
            )
        )
    )
    dataset.pretrain_mode = False
    dataset.max_length = max_length
    dataset.multiturn = True
    dataset.input_template = None
    dataset.input_key = "input"
    dataset.output_key = output_key
    dataset.apply_chat_template = tokenizer.apply_chat_template
    dataset._warned_offset_mapping_unavailable = False
    return dataset


def _process_and_get_masked_targets(dataset, row):
    processed = dataset.process_data(copy.deepcopy(row))
    dataset.prompts = [processed["prompt"]]
    dataset.responses = [processed["response"]]
    dataset.prompt_ids_lens = [processed["prompt_ids_len"]]
    dataset.response_ranges = [processed["response_ranges"]]

    _, _, loss_mask = dataset[0]
    text = (processed["prompt"] + processed["response"]).rstrip("\n")
    if not text.endswith(dataset.tokenizer.eos_token):
        text += " " + dataset.tokenizer.eos_token

    mask = loss_mask[0].tolist()
    masked_targets = "".join(text[idx + 1] for idx, value in enumerate(mask[:-1]) if value and idx + 1 < len(text))
    return processed, text, masked_targets


def test_qwen3_history_think_not_masked():
    dataset = _make_dataset(QwenLikeTokenizer())
    row = {
        "input": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "reasoning_content": "hidden first", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "reasoning_content": "visible second", "content": "answer two"},
        ]
    }

    _, rendered_text, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert "hidden first" not in rendered_text
    assert "hidden first" not in masked_targets
    assert "question one" not in masked_targets
    assert "question two" not in masked_targets
    assert "answer one" in masked_targets
    assert "visible second" in masked_targets
    assert "answer two" in masked_targets


def test_qwen3_in_band_history_think_masks_visible_answer_only():
    dataset = _make_dataset(QwenLikeTokenizer())
    row = {
        "input": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "<think>\nhidden first\n</think>\n\nanswer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "content": "<think>\nvisible second\n</think>\n\nanswer two"},
        ]
    }

    _, rendered_text, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert "hidden first" not in rendered_text
    assert "hidden first" not in masked_targets
    assert "question two" not in masked_targets
    assert "answer one" in masked_targets
    assert "<think>\nvisible second" in masked_targets
    assert "answer two" in masked_targets


def test_simple_template_masks_assistant_turns_without_user_text():
    dataset = _make_dataset(SimpleTemplateTokenizer())
    row = {
        "input": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "content": "answer two"},
        ]
    }

    _, _, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert "answer one" in masked_targets
    assert "answer two" in masked_targets
    assert "question one" not in masked_targets
    assert "question two" not in masked_targets


def test_truncation_clips_response_ranges():
    dataset = _make_dataset(SimpleTemplateTokenizer(), max_length=80)
    row = {
        "input": [
            {"role": "user", "content": "short question"},
            {"role": "assistant", "content": "a" * 200},
        ]
    }

    processed, _, _ = _process_and_get_masked_targets(dataset, row)

    assert processed["response_ranges"]
    for start_idx, end_idx in processed["response_ranges"]:
        assert 0 <= start_idx <= end_idx < dataset.max_length


def test_output_key_multiturn_does_not_mutate_input_row():
    dataset = _make_dataset(SimpleTemplateTokenizer(), output_key="output")
    row = {
        "input": [{"role": "user", "content": "question"}],
        "output": {"role": "assistant", "content": "answer"},
    }
    original = copy.deepcopy(row)

    _, _, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert row == original
    assert "answer" in masked_targets
    assert "question" not in masked_targets


def test_native_assistant_mask_path_when_available():
    tokenizer = NativeMaskTokenizer()
    dataset = _make_dataset(tokenizer)
    row = {
        "input": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    _, _, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert tokenizer.native_mask_calls == 1
    assert "answer" in masked_targets
    assert "question" not in masked_targets


def test_marker_fallback_when_native_mask_empty():
    tokenizer = EmptyNativeMaskTokenizer()
    dataset = _make_dataset(tokenizer)
    row = {
        "input": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    _, _, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert tokenizer.native_mask_calls == 1
    assert "answer" in masked_targets
    assert "question" not in masked_targets


def test_marker_fallback_marks_all_assistant_turns_in_one_render():
    tokenizer = CountingQwenLikeTokenizer()
    dataset = _make_dataset(tokenizer)
    row = {
        "input": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "reasoning_content": "hidden first", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "reasoning_content": "visible second", "content": "answer two"},
        ]
    }

    _, _, masked_targets = _process_and_get_masked_targets(dataset, row)

    assert tokenizer.marked_render_calls == 1
    assert "answer one" in masked_targets
    assert "visible second" in masked_targets
    assert "answer two" in masked_targets
