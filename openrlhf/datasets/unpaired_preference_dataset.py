from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler
from tqdm import tqdm

from .reward_dataset import RewardDataset
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, prompt_key=None, output_key=None, label_key=None):
    """
    Preprocess data from raw dataset to prompt, response, label

    Args:
        data: raw data from dataset
    """
    # custom dataset
    if output_key and label_key:
        if prompt_key:
            prompt = data[prompt_key]
        else:
            prompt = ""
            input_template = None  # do not modified with input template again
        response = data[output_key]
        label = data[label_key]
    else:
        # Dylan2048/ultrafeedback-unpaired-preferences
        if exist_and_not_none(data, "score"):
            prompt = data["instruction"]
            response = data["response"]
            label = data["score"]
        else:
            raise ValueError("Unknown dataset")

    # input template
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

    def __init__(
        self, dataset, tokenizer: Callable, max_length: int, strategy, input_template="Human: {}\nAssistant: "
    ) -> None:
        super().__init__()
        self.prompts = []
        self.responses = []
        self.labels = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        prompt_key = getattr(self.strategy.args, "prompt_key", None)
        output_key = getattr(self.strategy.args, "output_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, response, label = preprocess_data(data, input_template, prompt_key, output_key, label_key)
            self.prompts.append(prompt)
            self.responses.append(response)
            self.labels.append(label)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index], self.responses[index], self.labels[index]

    def collate_fn(self, item_list):
        def tokenizer(prompt, response):
            inputs = self.tokenizer(
                prompt + response + " " + self.tokenizer.eos_token,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )

            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
            inputs["attention_mask"][0][-1] = True
            return inputs["input_ids"], inputs["attention_mask"]

        tot_ids, tot_masks, tot_labels = [], [], []
        for prompt, response, label in item_list:
            input_ids, attention_mask = tokenizer(prompt, response)
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(label)

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        for idx in range(len(item_list)):
            next_idx = (idx + 1) % len(item_list)
            input_ids, attention_mask = tokenizer(item_list[idx][0], item_list[next_idx][1])
            tot_ids.append(input_ids)
            tot_masks.append(attention_mask)
            tot_labels.append(-1)

        input_ids = zero_pad_sequences(tot_ids, value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(tot_masks)
        return input_ids, attention_mask, torch.LongTensor(tot_labels)


class UnpairedRewardDataset(Dataset):
    """
    Dataset for KTO training, init from RewardDataset

    Args:
        dataset: dataset for training
        self.tokenizer: self.tokenizer for model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        vanilla_loss=False,
    ) -> None:
        super().__init__()
        # directly init from reward dataset
        assert isinstance(dataset, RewardDataset)
        self.prompts = dataset.prompts * 2
        self.responses = dataset.chosens + dataset.rejects
        self.labels = [1] * len(dataset.chosens) + [0] * len(dataset.rejects)

        self.tokenizer = dataset.tokenizer
        self.strategy = dataset.strategy
        self.max_length = dataset.max_length
        self.vanilla_loss = vanilla_loss

    def __getitem__(self, index):
        return self.prompts[index], self.responses[index], self.labels[index]

    def __len__(self):
        return len(self.prompts)

    def collate_fn(self, item_list):
        def concat_to_tensor(prompt, response):
            response = prompt + response + " " + self.tokenizer.eos_token
            response_token = self.tokenizer(
                response,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            # to avoid EOS_token truncation
            response_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            response_token["attention_mask"][0][-1] = True
            return response_token["input_ids"], response_token["attention_mask"]

        response_ids = []
        response_masks = []
        labels = []
        for response_id, response_mask, label in item_list:
            input_ids, attention_mask = concat_to_tensor(response_id, response_mask)
            response_ids.append(input_ids)
            response_masks.append(attention_mask)
            labels.append(label)

        # add unmatched y'| x (used to estimate the KL divergence between policy and reference)
        if not self.vanilla_loss:
            for prompt_idx in range(len(item_list)):
                response_idx = (prompt_idx + 1) % len(item_list)
                input_ids, attention_mask = concat_to_tensor(item_list[prompt_idx][0], item_list[response_idx][1])
                response_ids.append(input_ids)
                response_masks.append(attention_mask)
                labels.append(-1)

        response_ids = zero_pad_sequences(response_ids, value=self.tokenizer.pad_token_id)
        response_masks = zero_pad_sequences(response_masks)
        return response_ids, response_masks, torch.LongTensor(labels)


class DistributedVanillaKTOSampler(DistributedSampler):
    """
    Sampler for even number of +/- samples in batch
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=True)
        if self.num_samples % 2 == 1:
            self.num_samples -= 1
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        The index ensures that desirable and undesirable samples are interleaved.
        """
        labels = np.array(self.dataset.labels)
        unique_labels = np.unique(labels)
        self.label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        for label in self.label_to_indices:
            rng = np.random.default_rng(np.random.SeedSequence(self.seed + self.epoch))
            rng.shuffle(self.label_to_indices[label])

        assert len(self.label_to_indices[0]) == len(
            self.label_to_indices[1]
        ), f"Desirable and undesirable samples should be balanced, but {len(self.label_to_indices[0])} != {len(self.label_to_indices[1])}"
        desirable_indices = self.label_to_indices[1][self.rank : self.total_size // 2 : self.num_replicas]
        undesirable_indices = self.label_to_indices[0][self.rank : self.total_size // 2 : self.num_replicas]

        # drop out the last few samples(or it should be filled up to the longer)
        size_per_cls = min(len(desirable_indices), len(undesirable_indices))

        even_indices = []
        for idx in range(size_per_cls):
            even_indices.append(desirable_indices[idx])
            even_indices.append(undesirable_indices[idx])
        assert len(even_indices) == self.num_samples

        return iter(even_indices)

    def __len__(self):
        return self.num_samples
