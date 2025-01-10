from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.datasets.vl_sft_dataset import IGNORE_INDEX
from .utils import (
    exist_and_not_none,
    zero_pad_sequences,
    process_vision,
    padding_vision_token,
    convert_conversations,
)


class VLRewardDataset(Dataset):
    """
    Dataset for VL dpo tasks.
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
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
        self.max_length = max_length
        self.multiple_of = multiple_of

        self.special_tokens_map = {
            k: v
            for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
        }

        self.image_token = self.args.image_token
        self.video_token = self.args.video_token

        assert input_template is None, f"input_template is not supported currently"
        assert is_dpo, f"Only dpo is supported currently"
        #TODO: support more tasks such as rm
        self.input_template = input_template
        self.is_dpo = is_dpo

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
        prompt = item["prompt"]
        chosen = item["chosen"]
        reject = item["reject"]
        extra = item["extra"]
        media_info = item["media_info"]

        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token

        input_ids, attention_mask, labels = self.tokenize_text(prompt, chosen, reject)

        chosen_token, reject_token = input_ids
        chosen_attn_mask, reject_attn_mask = attention_mask
        chosen_labels, reject_labels = labels
        pixel_values = torch.tensor(media_info["pixel_values"], dtype=torch.float32)
        image_grid_thw = torch.tensor(media_info["image_grid_thw"], dtype=torch.int64)

        # to avoid EOS_token truncation
        chosen_token[-1] = self.tokenizer.eos_token_id
        reject_token[-1] = self.tokenizer.eos_token_id
        chosen_attn_mask[-1] = True
        reject_attn_mask[-1] = True

        return (
            chosen_token,
            chosen_attn_mask,
            chosen_labels,
            reject_token,
            reject_attn_mask,
            reject_labels,
            extra,
            pixel_values,
            image_grid_thw,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        chosen_labels = []
        chosen_pixels = []
        chosen_image_grid_thw = []
        reject_ids = []
        rejects_masks = []
        reject_labels = []
        reject_pixels = []
        reject_image_grid_thw = []
        extras = []

        for item in item_list:
            chosen_id, chosen_mask, chosen_label, reject_id, rejects_mask, reject_label, extra, pixel_values, image_grid_thw = item
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            chosen_labels.append(chosen_label)
            chosen_pixels.append(pixel_values)
            chosen_image_grid_thw.append(image_grid_thw)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            reject_labels.append(reject_label)
            reject_pixels.append(pixel_values)
            reject_image_grid_thw.append(image_grid_thw)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            raise NotImplementedError(f"Only dpo is supported currently")

        input_ids = chosen_ids + reject_ids
        attention_mask = chosen_masks + rejects_masks
        labels = chosen_labels + reject_labels
        pixel_values = chosen_pixels + reject_pixels
        image_grid_thws = chosen_image_grid_thw + reject_image_grid_thw
        input_ids = zero_pad_sequences(input_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(attention_mask, side=padding_side)
        labels = zero_pad_sequences(labels, side=padding_side, value=IGNORE_INDEX)

        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thws = torch.cat(image_grid_thws, dim=0)
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
        chosen, rejected = convert_conversations(self.args, data)
        media_info = process_vision(self.image_processor, imgs)
        image_grid_thw = media_info["image_grid_thw"]

        chosen_text = self.tokenizer.apply_chat_template(
            chosen,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        rejected_text = self.tokenizer.apply_chat_template(
            rejected,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        chosen_text = chosen_text.rstrip("\n")
        rejected_text = rejected_text.rstrip("\n")

        prompt_text = self.tokenizer.apply_chat_template(
            chosen[:-1],
            tokenize=False,
            add_generation_prompt=True,  # must set true for prompt
            add_vision_id=False,
        )

        assert len(chosen_text) > len(prompt_text)
        assert len(rejected_text) > len(prompt_text)
        prompt_text = padding_vision_token(self.image_processor, self.image_token, self.video_token,
                                           prompt_text, image_grid_thw)
        chosen_text = padding_vision_token(self.image_processor, self.image_token, self.video_token,
                                           chosen_text, image_grid_thw)
        rejected_text = padding_vision_token(self.image_processor, self.image_token,
                                             self.video_token, rejected_text, image_grid_thw)

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt_text = None

        # margin loss
        margin = data["margin"] if exist_and_not_none(data, "margin") else 0

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "reject": rejected_text,
            "extra": prompt_ids_len if self.is_dpo else margin,
            "media_info": media_info,
        }

    def tokenize_text(self, prompt_text, chosen_text, rejected_text):
        prompt_tokenized = self.tokenizer(prompt_text, padding=False)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for content in [chosen_text, rejected_text]:
            answer = content[len(prompt_text):]
            answer_tokenized = self.tokenizer(answer, padding=False)

            input_ids = prompt_tokenized.input_ids + answer_tokenized.input_ids
            attention_mask = prompt_tokenized.attention_mask + answer_tokenized.attention_mask
            labels = [IGNORE_INDEX] * len(prompt_tokenized.input_ids) + answer_tokenized.input_ids

            # truncate
            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                labels = labels[-self.max_length:]

            input_ids_list.append(torch.tensor(input_ids, dtype=torch.int64))
            attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.int64))
            labels_list.append(torch.tensor(labels, dtype=torch.int64))

        return input_ids_list, attention_mask_list, labels_list
