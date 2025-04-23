from typing import List, Dict

import torch
from qwen_vl_utils import process_vision_info
from ..base.data_processor import BaseDataProcessor, MMInputs


class Qwen2_5_VLDataProcessor(BaseDataProcessor):
    def __call__(
        self,
        messages,
        max_length,
        padding=True,
        device=None,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
    ) -> MMInputs:
        messages = self._format_messages(messages)
        processor = self.processor
        texts = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=padding,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_tensors=return_tensors,
        )
        emb_inputs, extra_info = self._split_input_dict(batch)
        return MMInputs(emb_inputs=emb_inputs,extra_info=extra_info).to(device)
    
    def _split_input_dict(self, input_dict: Dict) -> tuple[Dict, Dict]:
        extra_info = {}
        if "input_ids" in input_dict:
            extra_info["input_ids"] = input_dict.pop("input_ids")
        if "attention_mask" in input_dict:
            extra_info["attention_mask"] = input_dict.pop("attention_mask")
        return input_dict, extra_info

    def make_input_batch(self, inputs: List[MMInputs]) -> MMInputs:
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_grid_thw"]:
                # qwen2vl concat all patches of all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Qwen2VLDataProcessor")

        emb_inputs, extra_info = self._split_input_dict(batch)
        return MMInputs(emb_inputs=emb_inputs,extra_info=extra_info)

    def split_input_batch(self, batch: MMInputs) -> List[MMInputs]:
        batch_size = len(batch["input_ids"])
        batch_kwargs = [{} for _ in range(batch_size)]
        # first process None values
        keys = []
        for k, v in batch.items():
            if v is not None:
                keys.append(k)
            else:
                for i in range(batch_size):
                    batch_kwargs[i][k] = None

        if "pixel_values" in keys and (
            "input_ids" not in keys or "image_grid_thw" not in keys
        ):
            raise ValueError(
                "Cannot split batch with pixel_values without input_ids and image_grid_thw"
            )
        if "image_grid_thw" in keys and ("input_ids" not in keys):
            raise ValueError("Cannot split batch with image_grid_thw without input_ids")
        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v
        if "pixel_values" in keys:
            thws = batch["image_grid_thw"]  # (total_img_num, (t,h,w))
            pixel_values = batch["pixel_values"]
            vision_start_id = self.processor.tokenizer("<|vision_start|>")["input_ids"][0]
            vision_end_id = self.processor.tokenizer("<|vision_end|>")["input_ids"][0]
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                vision_start_num = (input_ids_i == vision_start_id).sum().item()
                vision_end_num = (input_ids_i == vision_end_id).sum().item()
                assert vision_start_num == vision_end_num
                img_num = vision_start_num
                if img_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_grid_thw"] = None
                    continue
                thws_i = thws[:img_num]
                assert len(thws_i) == img_num
                thws = thws[img_num:]
                if not isinstance(thws_i, torch.Tensor):
                    thws_i = torch.stack(thws_i)
                batch_kwargs[i]["image_grid_thw"] = thws_i
                patchs_num = thws_i.prod(dim=1).sum().item()
                pixel_values_i = pixel_values[:patchs_num]
                assert len(pixel_values_i) == patchs_num
                pixel_values = pixel_values[patchs_num:]
                batch_kwargs[i]["pixel_values"] = pixel_values_i
            assert len(thws) == 0
            assert len(pixel_values) == 0
        mm_inputs_list = []
        for b in batch_kwargs:
            emb_inputs, extra_info = self._split_input_dict(b)
            mm_inputs_list.append(MMInputs(emb_inputs=emb_inputs,extra_info=extra_info))
        return mm_inputs_list
    
DataProcessor = Qwen2_5_VLDataProcessor

__all__ = ["DataProcessor"]