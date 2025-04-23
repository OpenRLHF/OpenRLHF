import json
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Dict
import torch
import transformers
from transformers.processing_utils import ProcessorMixin
from qwen_vl_utils import process_vision_info
from PIL.Image import Image

class BatchFeature(transformers.feature_extraction_utils.BatchFeature):
    def pin_memory(self):
        new_data = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v.pin_memory()
            else:
                new_data[k] = v
        self.data = new_data
        return self

@dataclass
class MMInputs:
    emb_inputs: Optional[BatchFeature | dict] = None # used for getting the multimodal input_embeds
    forward_inputs: Optional[BatchFeature | dict] = None # some models need extra inputs for forward, even if given input_embeds
    extra_info: Optional[BatchFeature | dict] = None # Reserved item for other usages. Now used for batching and splitting.

    def __post_init__(self):
        if isinstance(self.emb_inputs,(dict,type(None))):
            self.emb_inputs = BatchFeature(self.emb_inputs)
        if isinstance(self.forward_inputs,(dict,type(None))):
            self.forward_inputs = BatchFeature(self.forward_inputs)
        if isinstance(self.extra_info,(dict,type(None))):
            self.extra_info = BatchFeature(self.extra_info)

    def to(self, *args, **kwargs):
        self.emb_inputs = self.emb_inputs.to(*args, **kwargs)
        self.forward_inputs = self.forward_inputs.to(*args, **kwargs)
        self.extra_info = self.extra_info.to(*args, **kwargs)
        return self
    
    def pin_memory(self):
        self.emb_inputs = self.emb_inputs.pin_memory()
        self.forward_inputs = self.forward_inputs.pin_memory()
        self.extra_info = self.extra_info.pin_memory()
        return self

    def _merge_to_dict(self):
        result = {**self.emb_inputs.data,**self.forward_inputs.data,**self.extra_info.data}
        # if two items have the same key, the values should be the same.
        for key in result.keys():
            if key in self.emb_inputs.keys():
                # same value should be the same object
                assert result[key] is self.emb_inputs[key]
            if key in self.forward_inputs.keys():
                assert result[key] is self.forward_inputs[key]
            if key in self.extra_info.keys():
                assert result[key] is self.extra_info[key]
        return result
    
    def keys(self):
        return self._merge_to_dict().keys()
    
    def items(self):
        return self._merge_to_dict().items()
    
    def __contains__(self, key):
        return key in self._merge_to_dict()

    def __getitem__(self, key):
        return self._merge_to_dict()[key]


class BaseDataProcessor(ABC):
    def __init__(self, processor: ProcessorMixin,processor_kwargs:Dict):
        super().__init__()
        self.processor = processor
        self.processor_kwargs = processor_kwargs
        # We use process_vision_info of qwen_vl_utils to get the image inputs for all model,
        # To be compatible with Qwen2VLImageProcessor, we always set the min_pixels and max_pixels for the processor
        self.min_pixels = processor_kwargs["min_pixels"]
        self.max_pixels = processor_kwargs["max_pixels"]
    @abstractmethod
    def __call__(
        self,
        messages: Union[Dict, List[str], str],
        max_length: int,
        padding: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: Optional[bool] = False,
        truncation: Optional[bool] = True,
    ) -> MMInputs:
        """
        We mainly use this function to get the visual inputs for the model.
        """
        raise NotImplementedError

    def _add_pixel_bounds(self,messages:List[List[Dict]]) -> List[List[Dict]]:
       DEFAULT_MIN_PIXELS = self.min_pixels
       DEFAULT_MAX_PIXELS = self.max_pixels

       def process_content(content):
           if isinstance(content, list):
               for item in content:
                   if isinstance(item, dict) and item.get("type") == "image":
                       if "min_pixels" not in item:
                           item["min_pixels"] = DEFAULT_MIN_PIXELS
                       if "max_pixels" not in item:
                           item["max_pixels"] = DEFAULT_MAX_PIXELS
           return content

       for message in messages:
           for msg in message:
               msg["content"] = process_content(msg["content"])
       return messages

    @abstractmethod
    def make_input_batch(self, inputs: List[MMInputs]) -> MMInputs:
        raise NotImplementedError

    @abstractmethod
    def split_input_batch(self, batch: MMInputs) -> List[MMInputs]:
        raise NotImplementedError

    def _format_messages(self, messages: Union[Dict, List[str], str]) -> List[List[Dict]]:
        messages = deepcopy(messages)
        if isinstance(messages, list) and isinstance(messages[0], str):
            formated_messages = [json.loads(m) for m in messages]
        elif isinstance(messages, str):
            formated_messages = [json.loads(messages)]
        elif isinstance(messages, dict):
            formated_messages = [[messages]]
        else:
            raise ValueError("Invalid messages format, must be a list of strings or a string or a dict")
        return self._add_pixel_bounds(formated_messages)

    def apply_chat_template(
        self,
        messages: Union[Dict, List[str], str],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> List[str]:
        messages = self._format_messages(messages)
        
        return self.processor.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def get_images_from_messages(
        self, messages: Union[Dict, List[str], str]
    ) -> List[Image]:
        messages = self._format_messages(messages)
        image_inputs, _ = process_vision_info(messages)
        return image_inputs


    @property
    def pad_token_id(self) -> int:
        return self.processor.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.processor.tokenizer.eos_token_id

    @property
    def tokenizer(self):
        return self.processor.tokenizer