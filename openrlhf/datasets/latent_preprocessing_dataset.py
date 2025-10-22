from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(data, input_key="query", output_key="response"):
    prompt = data[input_key]
    response = data[output_key]
    return prompt, response


class Latent_preprocessing_Dataset(Dataset):
    """
    Dataset for latent preprocessing

    Args:
        dataset: dataset for latent preprocessing
        tokenizer: tokenizer for latent preprocessing
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        num_processors=8,  # Specify the number of processors you want to use
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        # chat template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)

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

    def process_data(self, data):

        prompt, response = preprocess_data(
            data,
            self.input_key,
            self.output_key,
        )

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

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
        }

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        prompt_ids_len = self.prompt_ids_lens[idx]

        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_token["input_ids"]
        prompt_attention_mask = prompt_token["attention_mask"]

        response_token = self.tokenizer(
            response,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        response_ids = response_token["input_ids"]
        response_attention_mask = response_token["attention_mask"]

        return prompt_ids, prompt_attention_mask, response_ids, response_attention_mask, prompt_ids_len

        input_token = self.tokenizer(
            prompt,
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

        prompt_ids = []
        prompt_attention_masks = []
        response_ids = []
        response_attention_masks = []
        prompt_ids_lens = []

        for prompt_id, prompt_attention_mask, response_id, response_attention_mask, prompt_ids_len in item_list:
            prompt_ids.append(prompt_id)
            prompt_attention_masks.append(prompt_attention_mask)
            response_ids.append(response_id)
            response_attention_masks.append(response_attention_mask)
            prompt_ids_lens.append(prompt_ids_len)

        # prompt_ids = zero_pad_sequences(prompt_ids, "left", self.tokenizer.pad_token_id)
        # prompt_attention_masks = zero_pad_sequences(prompt_attention_masks, "left")
        response_ids = zero_pad_sequences(response_ids, "right", self.tokenizer.pad_token_id)
        response_attention_masks = zero_pad_sequences(response_attention_masks, "right")
        return prompt_ids, prompt_attention_masks, response_ids, response_attention_masks, prompt_ids_lens
        # input_ids = []
        # attention_masks = []
        # loss_masks = []

        # for input_id, attention_mask, loss_mask in item_list:
        #     input_ids.append(input_id)
        #     attention_masks.append(attention_mask)
        #     loss_masks.append(loss_mask)

        # input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        # attention_masks = zero_pad_sequences(attention_masks, "right")
        # loss_masks = zero_pad_sequences(loss_masks, "right")
        # return input_ids, attention_masks, loss_masks
