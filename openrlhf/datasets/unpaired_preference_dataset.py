from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data, input_template=None, input_key=None, output_key=None, label_key=None, apply_chat_template=None
):
    """
    Preprocess data from raw dataset to prompt, response, label

    Args:
        data: raw data from dataset
    """
    label = data[label_key]

    if apply_chat_template:
        prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
        response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        response = data[output_key]
        if input_template:
            prompt = input_template.format(prompt)
    return prompt, response, label


class UnpairedPreferenceDataset(Dataset):
    """
    Unpaired preference dataset for algorithm, like KTO

    Args:
        dataset: raw dataset
        self.tokenizer: self.tokenizer for model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy, input_template=None) -> None:
        super().__init__()
        self.prompts = []
        self.prompt_ids_lens = []
        self.responses = []
        self.labels = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        input_key = getattr(self.strategy.args, "input_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response, label = preprocess_data(
                data, input_template, input_key, output_key, label_key, apply_chat_template
            )
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                continue
            else:
                self.prompt_ids_lens.append(prompt_ids_len)

            self.prompts.append(prompt)
            self.responses.append(response)
            self.labels.append(label)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index], self.responses[index], self.labels[index], self.prompt_ids_lens[index]

    def collate_fn(self, item_list):
        def tokenizer(prompt, response):
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        tot_ids, tot_masks, tot_labels, prompt_ids_lens = [], [], [], []
        for prompt, response, label, prompt_ids_len in item_list:
            input_ids, attention_mask = tokenizer(prompt, response)
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(label)
            prompt_ids_lens.append(prompt_ids_len)

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list)):
            next_idx = (idx + 1) % len(item_list)
            input_ids, attention_mask = tokenizer(item_list[idx][0], item_list[next_idx][1])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(-1)
            prompt_ids_lens.append(item_list[idx][3])

        input_ids = zero_pad_sequences(tot_ids, side="right", value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(tot_masks, side="right")
        return input_ids, attention_mask, torch.LongTensor(tot_labels), prompt_ids_lens
