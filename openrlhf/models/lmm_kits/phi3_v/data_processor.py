from typing import List, Dict, Union
from ..base.data_processor import BaseDataProcessor, MMInputs
import torch

class Phi3_VDataProcessor(BaseDataProcessor):
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
        texts = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = self.get_images_from_messages(messages)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=padding,
            max_length=max_length,
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

    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        messages = self._convert_message_format(messages)
        return self.processor.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def _convert_message_format(self, messages: List[List[Dict]]) -> List[List[Dict]]:
        converted_messages = []
        
        for message in messages:
            new_message = []
            image_counter = 1
            for msg in message:
                role = msg["role"]
                content = msg["content"]

                # Process the content to combine text and images
                processed_content = ""
                
                if isinstance(content,list):
                    for content_item in content:
                        if content_item["type"] == "text":
                            processed_content += content_item["text"]
                        elif content_item["type"] == "image":
                            image_placeholder = f"<|image_{image_counter}|>"
                            processed_content += image_placeholder
                            image_counter += 1
                else:
                    processed_content += content

                new_message.append({"role": role, "content": processed_content})
            converted_messages.append(new_message)
        return converted_messages

    def make_input_batch(self, inputs: List[MMInputs]) -> MMInputs:
        # each element has no batch dimension
        batch = {}
        # collect all keys
        for inp in inputs:
            batch.update({k:None for k,v in inp.items() if v is not None})
        for k in batch.keys():
            if k in ["input_ids", "attention_mask"]:
                batch[k] = torch.stack([inp[k] for inp in inputs if k in inp], dim=0)
            elif k in ["pixel_values", "image_sizes"]:
                # The first dimension is the image count dimension
                # concat all images in a batch in the first dimension
                batch[k] = torch.cat([inp[k] for inp in inputs if k in inp], dim=0)
            else:
                raise ValueError(f"Unknown key {k} for Phi3_VDataProcessor")
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

        if "pixel_values" in keys and ("input_ids" not in keys or "image_sizes" not in keys):
            raise ValueError("Cannot split batch with pixel_values without input_ids and image_sizes")
        
        for k in ["input_ids", "attention_mask"]:
            if k in keys:
                vals = batch[k]
                if isinstance(vals, torch.Tensor):
                    vals = torch.unbind(vals)
                assert batch_size == len(vals)
                for i, v in enumerate(vals):
                    batch_kwargs[i][k] = v

        if "pixel_values" in keys:
            image_sizes = batch["image_sizes"]  # (image_num, 2)
            pixel_values = batch["pixel_values"]  # (image_num, ...)

            image_sizes = list(image_sizes)
            pixel_values = list(pixel_values)
            # Calculate number of image tokens for each image in the batch
            # copy from processing_phi3_v.py
            num_img_tokens_per_image = [int(((h//336)*(w//336)+1)*144 + 1 + (h//336+1)*12) for h, w in image_sizes]

            # Split pixel values and image sizes for each sample. Each sample can have multiple images.
            image_token_id = self.processor.tokenizer.encode("<|image|>", add_special_tokens=False)[0] #vllm use <|image|> as image token
            for i in range(batch_size):
                input_ids_i = batch_kwargs[i]["input_ids"]
                if not isinstance(input_ids_i, torch.Tensor):
                    input_ids_i = torch.tensor(input_ids_i)
                rest_image_token_num = (input_ids_i == image_token_id).sum().item()
                if rest_image_token_num == 0:
                    batch_kwargs[i]["pixel_values"] = None
                    batch_kwargs[i]["image_sizes"] = None
                    continue
                image_sizes_i = []
                pixel_values_i = []
                while rest_image_token_num > 0:
                    if len(num_img_tokens_per_image) == 0:
                        raise ValueError("Mismatch in total number of image tokens")
                    rest_image_token_num -= num_img_tokens_per_image.pop(0)
                    image_sizes_i.append(image_sizes.pop(0))
                    pixel_values_i.append(pixel_values.pop(0))
                assert rest_image_token_num == 0, "Mismatch in total number of image tokens"
                batch_kwargs[i]["pixel_values"] = torch.stack(pixel_values_i, dim=0)
                batch_kwargs[i]["image_sizes"] = torch.stack(image_sizes_i, dim=0)
            assert len(image_sizes) == 0
            assert len(pixel_values) == 0
        mm_inputs_list = []
        for b in batch_kwargs:
            emb_inputs, extra_info = self._split_input_dict(b)
            mm_inputs_list.append(MMInputs(emb_inputs=emb_inputs,extra_info=extra_info))
        return mm_inputs_list

DataProcessor = Phi3_VDataProcessor

__all__ = ["DataProcessor"]