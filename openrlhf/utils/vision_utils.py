import os
import re
from io import BytesIO
from abc import ABC, abstractmethod
from typing_extensions import override
from typing import Optional, Sequence, Union, Tuple, Any, Dict, Set, List
import math
import json
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject

import torch
from transformers import AutoConfig, AutoProcessor


IGNORE_INDEX = -100
ImageInput = Union[str, bytes, ImageObject]
SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]
IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")
VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")
DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)


def tool_formatter(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    tool_names = []
    for tool in tools:
        param_text = ""
        for name, param in tool["parameters"]["properties"].items():
            required, enum, items = "", "", ""
            if name in tool["parameters"].get("required", []):
                required = ", required"

            if param.get("enum", None):
                enum = ", should be one of [{}]".format(", ".join(param["enum"]))

            if param.get("items", None):
                items = ", where each item should be {}".format(param["items"].get("type", ""))

            param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                name=name,
                type=param.get("type", ""),
                required=required,
                desc=param.get("description", ""),
                enum=enum,
                items=items,
            )

        tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
            name=tool["name"], desc=tool.get("description", ""), args=param_text
        )
        tool_names.append(tool["name"])

    return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))


@dataclass
class DatasetAttr:
    ranking: bool = False
    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = None
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None

    def set_attr(self, key: str, obj, default = None) -> None:
        setattr(self, key, obj.get(key, default))


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[str] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        r"""
        Forms a list of slots according to the inputs to encode.
        """
        ...

    def extract(self, content) -> Union[str, List]:
        r"""
        Extract a list of tuples from the response message if using tools.

        Each tuple consists of function name and function arguments.
        """
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}")

        return elements


@dataclass
class FunctionFormatter(Formatter):
    def __post_init__(self):
        function_slots = ["Action: {{name}}\nAction Input: {{arguments}}\n"]
        self.slots = function_slots + self.slots

    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        functions: List[Tuple[str, str]] = []
        try:
            tool_calls = json.loads(content)
            if not isinstance(tool_calls, list):  # parallel function call
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                functions.append((tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False)))

        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in function message: {str([content])}")  # flat string

        elements = []
        for name, arguments in functions:
            for slot in self.slots:
                if isinstance(slot, str):
                    slot = slot.replace("{{name}}", name).replace("{{arguments}}", arguments)
                    elements.append(slot)
                elif isinstance(slot, (dict, set)):
                    elements.append(slot)
                else:
                    raise RuntimeError(f"Input must be string, set[str] or dict[str, str], got {type(slot)}")

        return elements


@dataclass
class ToolFormatter(Formatter):
    @override
    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return [tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON format in tool description: {str([content])}")  # flat string

    @override
    def extract(self, content: str) -> Union[str, List]:
        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append((tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content
        return results

# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py
class BasePlugin:
    def __init__(self, image_token, video_token) -> None:
        self.image_token = image_token
        self.video_token = video_token

    def _validate_input(self, images, videos) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")

    def _preprocess_image(self, image, **kwargs) -> ImageObject:
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if (image.width * image.height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _get_video_sample_frames(self, video_stream, **kwargs) -> int:
        r"""
        Computes video sample frames according to fps.
        """
        video_fps: float = kwargs.get("video_fps")
        video_maxlen: int = kwargs.get("video_maxlen")
        total_frames = video_stream.frames
        sample_frames = float(video_stream.duration * video_stream.time_base) * video_fps
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return math.floor(sample_frames)

    def _regularize_images(self, images, **kwargs) -> List[ImageObject]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, bytes):
                image = Image.open(BytesIO(image))
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError(f"Expect input is a list of Images, but got {type(image)}.")

            results.append(self._preprocess_image(image, **kwargs))

        return results

    def _regularize_videos(self, videos, **kwargs) -> List[List[ImageObject]]:
        r"""
        Regularizes videos to avoid error. Including reading, resizing and converting.
        """
        results = []
        for video in videos:
            container = av.open(video, "r")
            video_stream = next(stream for stream in container.streams if stream.type == "video")
            total_frames = video_stream.frames
            sample_frames = self._get_video_sample_frames(video_stream, **kwargs)
            sample_indices = np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)
            frames: List["ImageObject"] = []
            container.seek(0)
            for frame_idx, frame in enumerate(container.decode(video_stream)):
                if frame_idx in sample_indices:
                    frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)
            results.append(frames)

        return results

    def _get_mm_inputs(self, images, videos, processor) -> Dict[str, torch.Tensor]:
        r"""
        Processes visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)
        """
        image_processor = getattr(processor, "image_processor")
        video_processor = getattr(processor, "video_processor", image_processor)
        input_dict = {"images": None}  # default key
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512 * 512),
            )
            input_dict["images"] = images

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_resolution=getattr(processor, "video_resolution", 128 * 128),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 64),
            )
            input_dict["videos"] = videos

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:
            # qwen2-vl has same processor.
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs

    def process_messages(self, messages, images, videos, processor) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, tokenizer, processor
                          ) -> Tuple[List[int], Optional[List[int]]]:
        r"""
        Pre-processes token ids after tokenization for VLMs.
        """
        self._validate_input(images, videos)
        return input_ids, labels

    def get_mm_inputs(self, images, videos, imglens, vidlens, batch_ids, processor
                      ) -> Dict[str, Union[List[int], torch.Tensor]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            batch_ids: input ids of samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return {}

# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py
class Qwen2vlPlugin(BasePlugin):
    @override
    def _preprocess_image(self, image, **kwargs) -> ImageObject:
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)
        return image

    @override
    def _get_video_sample_frames(self, video_stream, **kwargs) -> int:
        sample_frames = super()._get_video_sample_frames(video_stream, **kwargs)
        sample_frames = sample_frames // 2 * 2
        return sample_frames

    @override
    def process_messages(self, messages, images, videos, processor
                         ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        self._validate_input(images, videos)
        image_processor = getattr(processor, "image_processor")
        merge_length = getattr(image_processor, "merge_size") ** 2
        mm_inputs = self._get_mm_inputs(images, videos, processor)
        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError(f"`len(images)` is less than the number of {IMAGE_PLACEHOLDER} tokens.")

                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError(f"`len(videos)` is less than the number of {VIDEO_PLACEHOLDER} tokens.")

                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError(f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens.")

        if len(videos) != num_video_tokens:
            raise ValueError(f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens.")

        return messages

    @override
    def get_mm_inputs(self, images, videos, imglens, vidlens, batch_ids, processor
                      ) -> Dict[str, Union[List[int], torch.Tensor]]:
        r"""
        Builds batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            batch_ids: input ids of samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos
        """
        self._validate_input(images, videos)
        return self._get_mm_inputs(images, videos, processor)


@dataclass
class VisionEncoderUtils:
    format_user: Formatter
    format_assistant: Formatter
    format_system: Formatter
    format_function: Formatter
    format_observation: Formatter
    format_tools: Formatter
    format_separator: Formatter
    format_prefix: Formatter
    default_system: str
    stop_words: List[str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    mm_plugin: Qwen2vlPlugin

    def encode_oneturn(
        self,
        tokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        answer_ids = encoded_messages[-1]
        return prompt_ids, answer_ids

    def encode_multiturn(
        self,
        tokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content) -> Union[str, List[Tuple[str, str]]]:
        r"""
        Extracts tool message.
        """
        return self.format_tools.extract(content)

    def _encode(
        self,
        tokenizer,
        messages: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[List[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query                    resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == "user":
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == "assistant":
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == "observation":
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == "function":
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _convert_elements_to_ids(self, tokenizer, elements: SLOTS) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids


def get_dataset_attr(config_path):
    dataset_info = None
    try:
        with open(config_path) as f:
            dataset_info = json.load(f)
    except Exception as e:
        raise ValueError(f"Cannot open {config_path} due to {str(e)}.")
    dataset_attr = DatasetAttr()
    dataset_attr.set_attr("ranking", dataset_info, default=False)
    if "columns" in dataset_info:
        column_names = ["messages", "system", "tools", "images", "videos", "chosen", "rejected"]
        for column_name in column_names:
            dataset_attr.set_attr(column_name, dataset_info["columns"])
    if "tags" in dataset_info:
        tag_names = (
            "role_tag", "content_tag", "user_tag", "assistant_tag",
            "observation_tag", "function_tag", "system_tag",
        )
        for tag in tag_names:
            dataset_attr.set_attr(tag, dataset_info["tags"])
    return dataset_attr

def get_image_seqlen(config):
    r"""
    Computes the number of special tokens per image.
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "llava":
        image_seqlen = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        if getattr(config, "vision_feature_select_strategy", "default") == "full":  # add [CLS] token
            image_seqlen += 1
    elif model_type == "paligemma":
        image_seqlen = config.vision_config.num_image_tokens
    else:
        image_seqlen = -1

    return image_seqlen

def get_patch_size(config, processor):
    r"""
    Computes the patch size of the vit.
    """
    patch_size = getattr(config.vision_config, "patch_size", getattr(processor, "patch_size", -1))
    return patch_size

def get_vision_feature_select_strategy(config, processor):
    r"""
    Get the vision_feature_select_strategy.
    """
    vision_feature_select_strategy = getattr(
        config, "vision_feature_select_strategy", getattr(processor, "vision_feature_select_strategy", "default")
    )
    return vision_feature_select_strategy

def get_vision_processor(args, processor_path, tokenizer):
    def patch_processor(processor, config, tokenizer):
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_seqlen", get_image_seqlen(config))
        setattr(processor, "image_resolution", args.image_resolution)
        setattr(processor, "patch_size", get_patch_size(config, processor))
        setattr(processor, "video_resolution", args.video_resolution)
        setattr(processor, "video_fps", args.video_fps)
        setattr(processor, "video_maxlen", args.video_maxlen)
        setattr(processor, "vision_feature_select_strategy",
                get_vision_feature_select_strategy(config, processor))

    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "token": None,
    }
    config = AutoConfig.from_pretrained(processor_path, **init_kwargs)
    vision_processor = AutoProcessor.from_pretrained(processor_path, **init_kwargs)
    patch_processor(vision_processor, config, tokenizer)
    if vision_processor is not None and "Processor" not in vision_processor.__class__.__name__:
        vision_processor = None
    return vision_processor

def get_qwen2_vl_utils(args):
    eos_slots = [] if args.efficient_eos else [{"eos_token"}]
    encoder_utils = VisionEncoderUtils(
        format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_assistant=StringFormatter(slots=["{{content}}"] + eos_slots),
        format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
        format_function=FunctionFormatter(slots=eos_slots, tool_format="default"),
        format_observation=StringFormatter(slots=["<|im_start|>tool\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
        format_tools=ToolFormatter(tool_format="default"),
        format_separator=EmptyFormatter(slots=["\n"]),
        format_prefix=EmptyFormatter(),
        default_system="You are a helpful assistant.",
        stop_words=["<|im_end|>"],
        efficient_eos=args.efficient_eos,
        replace_eos=True,
        replace_jinja_template=False,
        mm_plugin=Qwen2vlPlugin(image_token="<|image_pad|>", video_token="<|video_pad|>")
    )
    return encoder_utils
