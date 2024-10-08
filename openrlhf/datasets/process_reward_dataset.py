from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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

        label_token = self.tokenizer(
            self.labels[idx],
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
        )

        assert input_token["input_ids"].numel() == label_token["input_ids"].numel()

        return (
            input_token["input_ids"],
            input_token["attention_mask"],
            label_token["input_ids"],
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
