from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openrlhf.utils.utils import convert_token_to_id
from .utils import zero_pad_sequences


class ProcessRewardDataset(Dataset):
    """
    Dataset for process reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of

        # chat_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.placeholder_token = getattr(self.strategy.args, "placeholder_token", None)
        self.reward_tokens = getattr(self.strategy.args, "reward_tokens", None)

        self.placeholder_token_id = convert_token_to_id(self.placeholder_token, self.tokenizer)

        # Store the processed data in class attributes
        self.inputs = dataset[self.input_key]
        self.labels = dataset[self.label_key]

    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        input_token = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = input_token["input_ids"]
        label_values = self.labels[idx]
        assert isinstance(label_values, list), "labels should be a list of strings or numbers"
        if isinstance(label_values[0], str):
            label_tokens = []
            for label in label_values:
                assert (
                    self.reward_tokens is None or label in self.reward_tokens
                ), f"label should be in reward tokens {self.reward_tokens}, got {label}"
                label_tokens.append(convert_token_to_id(label, self.tokenizer))

            # label_tokens is list of token id (for '+', '-', etc)
            label_tensor = torch.tensor(label_tokens, dtype=input_ids.dtype)
        else:
            # label_values is list of float numbers (for reward values)
            label_tensor = torch.tensor(label_values, dtype=torch.float)
        # Motivation: inputs_ids maybe truncated to self.max_length, where placeholder_tokens at the end may be removed.
        # We should also truncate the labels to match the length of input_ids
        # Step 1: Create a mask for placeholder token positions
        mask = input_ids == self.placeholder_token_id
        # Step 2: Ensure that label_tensor is truncated along the last dimension
        # Find the length of the last dimension of the mask
        num_placeholders = mask.sum(dim=-1)
        # Truncate label_tensor along the last dimension to match num_placeholders
        truncated_labels = label_tensor[..., : num_placeholders.max()]
        # Step 3: Update labels at placeholder token positions
        labels = torch.full_like(input_ids, -100)
        labels[mask] = truncated_labels

        return (
            input_ids,
            input_token["attention_mask"],
            labels,
        )

    def collate_fn(self, item_list):
        input_ids = []
        input_masks = []
        label_ids = []
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id)
            input_masks.append(input_mask)
            label_ids.append(label_id)

        padding_side = "right"
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        input_masks = zero_pad_sequences(input_masks, side=padding_side)
        label_ids = zero_pad_sequences(label_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        return input_ids, input_masks, label_ids

    def packing_collate_fn(self, item_list):
        input_ids = []
        input_att_masks = []
        input_seq_lens = []
        label_ids = []
        index = 1
        for input_id, input_mask, label_id in item_list:
            input_ids.append(input_id.flatten())
            input_att_masks.append(torch.full_like(input_id.flatten(), index))
            input_seq_lens.append(len(input_id.flatten()))

            label_ids.append(label_id.flatten())
            index += 1

        packed_input_ids = torch.cat(input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(input_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = input_seq_lens
        packed_label_ids = torch.cat(label_ids, dim=0).unsqueeze(0)

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, packed_label_ids
