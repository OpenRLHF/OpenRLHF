from typing import Callable

import torch
from torch.utils.data import Dataset

from .utils import (
    zero_pad_sequences,
    process_vision,
    padding_vision_token,
    convert_conversations,
)

IGNORE_INDEX = -100

class VLSFTDataset(Dataset):
    """
    Dataset for VL sft tasks.
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        pretrain_mode=False,
        num_processors=8,  # Specify the number of processors you want to use
        multiple_of=1,
        processor=None,
        args=None,
    ) -> None:
        super().__init__()
        assert args is not None
        self.args = args
        self.tokenizer = tokenizer
        self.strategy = strategy
        assert not pretrain_mode, f"{pretrain_mode=} is not supported currently"
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of

        self.special_tokens_map = {
            k: v
            for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
        }

        self.image_token = self.args.image_token
        self.video_token = self.args.video_token

        self.image_processor = processor.image_processor if processor is not None else None

        processed_dataset = dataset.map(self.process_data,
                                        remove_columns=dataset.column_names,
                                        num_proc=num_processors)
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None,
                                                     num_proc=num_processors,
                                                     batch_size=1000)
        self.processed_dataset = processed_dataset

    def __len__(self):
        length = len(self.processed_dataset)
        return length

    def __getitem__(self, idx):
        item = self.processed_dataset[idx]
        prompt_text = item["prompt"]
        all_text = item["all_text"]
        media_info = item["media_info"]

        if not self.pretrain_mode:
            if not all_text.endswith(self.tokenizer.eos_token):
                all_text += " " + self.tokenizer.eos_token
        else:
            text = prompt_text

        input_ids, attention_mask, labels = self.tokenize_text(all_text, prompt_text)
        pixel_values = torch.tensor(media_info["pixel_values"], dtype=torch.float32)
        image_grid_thw = torch.tensor(media_info["image_grid_thw"], dtype=torch.int64)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[-1] = self.tokenizer.eos_token_id
            attention_mask[-1] = True

        return (
            input_ids,
            attention_mask,
            labels,
            pixel_values,
            image_grid_thw,
        )

    def collate_fn(self, item_list):
        new_item_list = []
        tokens_list = []
        attn_masks_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thws_list = []
        for item in item_list:
            input_ids, attention_mask, labels, pixel_values, image_grid_thw = item
            tokens_list.append(input_ids)
            attn_masks_list.append(attention_mask)
            labels_list.append(labels)
            pixel_values_list.append(pixel_values)
            image_grid_thws_list.append(image_grid_thw)

        padding_side = "right"
        input_ids = zero_pad_sequences(tokens_list,
                                       side=padding_side,
                                       value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(attn_masks_list, side=padding_side)
        labels = zero_pad_sequences(labels_list, side=padding_side, value=IGNORE_INDEX)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thws = torch.cat(image_grid_thws_list, dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thws,
        }

    def packing_collate_fn(self, item_list):
        raise NotImplementedError("unsupport packing example currently")

    def process_data(self, data):
        imgs = data[self.args.image_tag]
        conversations = convert_conversations(self.args, data)
        assert len(conversations) > 1

        media_info = process_vision(self.image_processor, imgs)
        image_grid_thw = media_info["image_grid_thw"]

        all_text = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        all_text = all_text.rstrip("\n")

        prompt_text = self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=False,
        )

        assert len(all_text) > len(prompt_text)

        all_text = padding_vision_token(self.image_processor, self.image_token, self.video_token,
                                        all_text, image_grid_thw)
        prompt_text = padding_vision_token(self.image_processor, self.image_token, self.video_token,
                                           prompt_text, image_grid_thw)

        # todo
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt_text = None
        else:
            raise NotImplementedError(f"pretrain_mode {self.pretrain_mode} is not yet implemented.")

        return {
            "prompt": prompt_text,
            "all_text": all_text,
            "media_info": media_info,
        }

    def tokenize_text(self, all_text, prompt_text):
        assert len(all_text) > len(prompt_text)
        answer = all_text[len(prompt_text):]

        prompt_tokenized = self.tokenizer(prompt_text, padding=False)
        answer_tokenized = self.tokenizer(answer, padding=False)

        input_ids = prompt_tokenized.input_ids + answer_tokenized.input_ids
        attention_mask = prompt_tokenized.attention_mask + answer_tokenized.attention_mask
        labels = [IGNORE_INDEX] * len(prompt_tokenized.input_ids) + answer_tokenized.input_ids

        # truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            labels = labels[-self.max_length:]

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        return input_ids, attention_mask, labels
