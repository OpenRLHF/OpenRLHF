import copy
import warnings
from collections.abc import Mapping
from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiturn=False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        self._warned_offset_mapping_unavailable = False

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.output_key = getattr(self.strategy.args.data, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args.data, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def _normalize_multiturn_messages(self, data):
        messages = copy.deepcopy(data[self.input_key])
        if self.output_key and data.get(self.output_key):
            output_message = copy.deepcopy(data[self.output_key])
            if isinstance(output_message, list):
                messages.extend(output_message)
            else:
                messages.append(output_message)
        return messages

    def _render_chat(self, messages, add_generation_prompt=False):
        return self.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _to_flat_list(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            value = value[0]
        return list(value)

    @staticmethod
    def _mask_to_ranges(mask):
        ranges = []
        start_idx = None
        for idx, value in enumerate(mask):
            if value and start_idx is None:
                start_idx = idx
            elif not value and start_idx is not None:
                ranges.append((start_idx, idx - 1))
                start_idx = None
        if start_idx is not None:
            ranges.append((start_idx, len(mask) - 1))
        return ranges

    @staticmethod
    def _merge_char_spans(spans):
        if not spans:
            return []

        spans = sorted(spans)
        merged = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    def _tokenize_text(self, text, return_offsets_mapping=False):
        kwargs = {
            "max_length": self.max_length,
            "padding": False,
            "truncation": True,
            "return_tensors": None,
            "add_special_tokens": False,
        }
        if return_offsets_mapping:
            kwargs["return_offsets_mapping"] = True
        return self.tokenizer(text, **kwargs)

    def _build_multiturn_response_ranges(self, messages, rendered_text):
        native_ranges = self._try_native_assistant_ranges(messages, rendered_text)
        if native_ranges is not None:
            return native_ranges
        return self._build_marker_assistant_ranges(messages, rendered_text)

    def _try_native_assistant_ranges(self, messages, rendered_text):
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=self.max_length,
            )
        except Exception:
            return None

        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            return None

        mask = None
        for key in ("assistant_masks", "assistant_tokens_mask", "assistant_mask"):
            if key in encoded:
                mask = encoded[key]
                break
        if mask is None:
            return None

        native_input_ids = self._to_flat_list(encoded["input_ids"])
        assistant_mask = self._to_flat_list(mask)
        if len(native_input_ids) != len(assistant_mask) or not any(assistant_mask):
            return None

        tokenized_render = self._tokenize_text(rendered_text)
        rendered_input_ids = self._to_flat_list(tokenized_render["input_ids"])
        if native_input_ids != rendered_input_ids:
            return None

        return self._mask_to_ranges(assistant_mask)

    def _build_marker_assistant_ranges(self, messages, rendered_text):
        marked_messages = copy.deepcopy(messages)
        collision_text = rendered_text + "\n" + str(marked_messages)
        marker_pairs_by_message = []
        for message_idx, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue

            marker_pairs = self._mark_assistant_message(marked_messages, message_idx, collision_text)
            if not marker_pairs:
                continue
            marker_pairs_by_message.append((message_idx, marker_pairs))

        if not marker_pairs_by_message:
            return []

        try:
            marked_render = self._render_chat(marked_messages).rstrip("\n")
        except Exception:
            return []

        all_marker_pairs = [marker_pair for _, marker_pairs in marker_pairs_by_message for marker_pair in marker_pairs]
        stripped_render, marker_occurrences = self._strip_marker_occurrences(marked_render, all_marker_pairs)
        if stripped_render != rendered_text:
            warnings.warn(
                "Skipping multi-turn assistant mask construction because the marked chat-template render does not "
                "match the unmarked prompt + response after marker removal.",
                stacklevel=2,
            )
            return []

        char_spans = []
        for _, marker_pairs in marker_pairs_by_message:
            visible_spans = []
            for start_marker, end_marker in marker_pairs:
                start_marker_pos = marked_render.find(start_marker)
                if start_marker_pos == -1:
                    continue

                content_start = start_marker_pos + len(start_marker)
                end_marker_pos = marked_render.find(end_marker, content_start)
                if end_marker_pos == -1:
                    continue

                start_char = self._marked_to_unmarked_pos(content_start, marker_occurrences)
                end_char = self._marked_to_unmarked_pos(end_marker_pos, marker_occurrences)
                if start_char < end_char:
                    visible_spans.append((start_char, end_char))

            if visible_spans:
                start_char = min(start for start, _ in visible_spans)
                end_char = max(end for _, end in visible_spans)
                char_spans.append(self._extend_assistant_span(rendered_text, start_char, end_char))

        return self._char_spans_to_token_ranges(rendered_text, self._merge_char_spans(char_spans))

    def _mark_assistant_message(self, messages, message_idx, collision_text):
        message = messages[message_idx]
        marker_pairs = []
        marker_index = 0

        def make_marker_pair():
            nonlocal marker_index
            while True:
                start_marker = f"__OPENRLHF_ASSISTANT_SPAN_{message_idx}_{marker_index}_START__"
                end_marker = f"__OPENRLHF_ASSISTANT_SPAN_{message_idx}_{marker_index}_END__"
                marker_index += 1
                if start_marker not in collision_text and end_marker not in collision_text:
                    return start_marker, end_marker

        def wrap_segment(text):
            start_marker, end_marker = make_marker_pair()
            marker_pairs.append((start_marker, end_marker))
            return f"{start_marker}{text}{end_marker}"

        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            message["reasoning_content"] = self._mark_newline_stripped_segment(
                reasoning_content,
                wrap_segment,
                strip_left=True,
                strip_right=True,
            )

        content = message.get("content")
        if isinstance(content, str) and content:
            message["content"] = self._mark_content_segments(content, wrap_segment)

        return marker_pairs

    @staticmethod
    def _mark_newline_stripped_segment(text, wrap_segment, strip_left=False, strip_right=False):
        start = 0
        end = len(text)
        if strip_left:
            while start < end and text[start] == "\n":
                start += 1
        if strip_right:
            while end > start and text[end - 1] == "\n":
                end -= 1
        if start >= end:
            return text
        return text[:start] + wrap_segment(text[start:end]) + text[end:]

    def _mark_content_segments(self, content, wrap_segment):
        think_start = "<think>"
        think_end = "</think>"
        first_think_start = content.find(think_start)
        first_think_end = content.find(think_end, first_think_start + len(think_start))
        if first_think_start == -1 or first_think_end == -1:
            return self._mark_newline_stripped_segment(content, wrap_segment, strip_left=True)

        reasoning_start = first_think_start + len(think_start)
        reasoning_text = content[reasoning_start:first_think_end]
        marked_reasoning = self._mark_newline_stripped_segment(
            reasoning_text,
            wrap_segment,
            strip_left=True,
            strip_right=True,
        )

        last_think_end = content.rfind(think_end)
        answer_start = last_think_end + len(think_end)
        while answer_start < len(content) and content[answer_start] == "\n":
            answer_start += 1

        answer_text = content[answer_start:]
        marked_answer = wrap_segment(answer_text) if answer_text else answer_text
        return content[:reasoning_start] + marked_reasoning + content[first_think_end:answer_start] + marked_answer

    @staticmethod
    def _strip_marker_occurrences(marked_render, marker_pairs):
        markers = {marker for marker_pair in marker_pairs for marker in marker_pair}
        occurrences = []
        for marker in markers:
            search_start = 0
            while True:
                marker_pos = marked_render.find(marker, search_start)
                if marker_pos == -1:
                    break
                occurrences.append((marker_pos, marker))
                search_start = marker_pos + len(marker)

        occurrences.sort(key=lambda item: item[0])
        stripped_parts = []
        last_pos = 0
        for marker_pos, marker in occurrences:
            stripped_parts.append(marked_render[last_pos:marker_pos])
            last_pos = marker_pos + len(marker)
        stripped_parts.append(marked_render[last_pos:])
        return "".join(stripped_parts), occurrences

    @staticmethod
    def _marked_to_unmarked_pos(position, marker_occurrences):
        removed_chars = sum(len(marker) for marker_pos, marker in marker_occurrences if marker_pos < position)
        return position - removed_chars

    def _extend_assistant_span(self, rendered_text, start_char, end_char):
        think_prefix = "<think>\n"
        previous_text = rendered_text[start_char - len(think_prefix) : start_char]
        if start_char >= len(think_prefix) and previous_text == think_prefix:
            start_char -= len(think_prefix)

        think_suffix = "\n</think>"
        if rendered_text.startswith(think_suffix, end_char):
            end_char += len(think_suffix)

        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token and rendered_text.startswith(eos_token, end_char):
            end_char += len(eos_token)

        return start_char, end_char

    def _char_spans_to_token_ranges(self, rendered_text, spans):
        if not spans:
            return []

        try:
            encoded = self._tokenize_text(rendered_text, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
        except Exception:
            return self._char_spans_to_token_ranges_without_offsets(rendered_text, spans)

        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        ranges = []
        for start_char, end_char in spans:
            token_indices = [
                idx
                for idx, offset in enumerate(offsets)
                if len(offset) == 2 and offset[0] != offset[1] and offset[0] < end_char and offset[1] > start_char
            ]
            if token_indices:
                ranges.append((token_indices[0], token_indices[-1]))
        return self._merge_char_spans(ranges)

    def _char_spans_to_token_ranges_without_offsets(self, rendered_text, spans):
        if not self._warned_offset_mapping_unavailable:
            warnings.warn(
                "Tokenizer does not provide offset mappings; falling back to prefix token counts for multi-turn "
                "assistant masks.",
                stacklevel=2,
            )
            self._warned_offset_mapping_unavailable = True

        ranges = []
        for start_char, end_char in spans:
            start_idx = len(self._to_flat_list(self._tokenize_text(rendered_text[:start_char])["input_ids"]))
            end_idx = len(self._to_flat_list(self._tokenize_text(rendered_text[:end_char])["input_ids"])) - 1
            clipped_end_idx = min(end_idx, self.max_length - 1)
            if start_idx <= clipped_end_idx:
                ranges.append((start_idx, clipped_end_idx))
        return self._merge_char_spans(ranges)

    def process_data(self, data):
        response_ranges = []
        effective_output_key = self.output_key
        data_for_preprocess = data

        if self.multiturn:
            messages = self._normalize_multiturn_messages(data)
            data_for_preprocess = dict(data)
            data_for_preprocess[self.input_key] = messages
            effective_output_key = None

        prompt, response = preprocess_data(
            data_for_preprocess,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            effective_output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        if self.multiturn and not self.pretrain_mode:
            rendered_text = (prompt + response).rstrip("\n")
            response_ranges = self._build_multiturn_response_ranges(messages, rendered_text)

        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges if self.multiturn else None,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, idx)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, idx):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            response_ranges = self.response_ranges[idx]
            for start_idx, end_idx in response_ranges:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks
