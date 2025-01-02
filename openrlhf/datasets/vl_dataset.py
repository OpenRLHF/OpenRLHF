import os
from typing import Any,  Dict, Literal, Optional, Sequence, Union, List, Tuple
from functools import partial
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import DataCollatorForSeq2Seq, ProcessorMixin

from openrlhf.utils.utils import blending_datasets
from openrlhf.utils.vision_utils import (
    IGNORE_INDEX, ImageInput,
    VisionEncoderUtils, DatasetAttr,
    get_dataset_attr,
)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _convert_images(
    images: Union[ImageInput, Sequence[ImageInput]],
    dataset_attr: DatasetAttr,
) -> Optional[List[ImageInput]]:
    r"""
    Optionally concatenates image path to dataset dir when loading from local disk.
    """
    if not isinstance(images, list):
        images = [images]
    elif len(images) == 0:
        return None
    else:
        images = images[:]

    return images


def convert_sharegpt(
    example: Dict[str, Any],
    dataset_attr: DatasetAttr
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: "user",
        dataset_attr.assistant_tag: "assistant",
        dataset_attr.observation_tag: "observation",
        dataset_attr.function_tag: "function",
        dataset_attr.system_tag: "system",
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            print(f"Invalid role tag in {messages}.")
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
        dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        print(f"Invalid message count in {messages}.")
        broken_data = True

    if (
        dataset_attr.ranking
        and isinstance(example[dataset_attr.chosen], dict)
        and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (
            chosen[dataset_attr.role_tag] not in accept_tags[-1]
            or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            print(f"Invalid role tag in {[chosen, rejected]}.")
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        print("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": None,
    }
    return output


def _encode_supervised_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence[ImageInput],
    videos: Sequence,
    encoder: VisionEncoderUtils,
    tokenizer,
    processor,
    cutoff_len: int,
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    messages = encoder.mm_plugin.process_messages(prompt + response, images, videos, processor)
    input_ids, labels = encoder.mm_plugin.process_token_ids([], [], images, videos, tokenizer, processor)
    encoded_pairs = encoder.encode_multiturn(tokenizer, messages, system, tools)
    total_length = len(input_ids) + (1 if encoder.efficient_eos else 0)
    if mask_history:
        encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        if total_length >= cutoff_len:
            break

        source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), cutoff_len - total_length)
        source_ids = source_ids[:source_len]
        target_ids = target_ids[:target_len]
        total_length += source_len + target_len

        if train_on_prompt:
            source_label = source_ids
        elif encoder.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != 0:  # train on the last turn only
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        if mask_history:  # reversed sequences
            input_ids = source_ids + target_ids + input_ids
            labels = source_label + target_label + labels
        else:
            input_ids += source_ids + target_ids
            labels += source_label + target_label

    if encoder.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    return input_ids, labels


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence[ImageInput],
    videos: Sequence,
    encoder: VisionEncoderUtils,
    tokenizer,
    processor,
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = encoder.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = encoder.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_ids, chosen_ids = encoder.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = encoder.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if encoder.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = encoder.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    encoder: VisionEncoderUtils,
    tokenizer,
    processor,
    data_args,
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            print(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_supervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            encoder=encoder,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.max_len,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    encoder: VisionEncoderUtils,
    tokenizer,
    processor,
    data_args,
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            print(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            encoder=encoder,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.max_len,
        )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def get_preprocessed_dataset(args, data_list, encoder, tokenizer, processor):
    train_data, eval_data = data_list

    dataset_attr = get_dataset_attr(args.dataset_config_path)

    kwargs = dict(
        num_proc=args.processing_num_workers,
        load_from_cache_file=(not args.overwrite_cache) or (args.local_process_index != 0),
        desc="Converting format of dataset",
    )
    convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr)
    column_names = list(next(iter(train_data)).keys())
    train_data = train_data.map(convert_func, batched=False, remove_columns=column_names,
                                **kwargs)
    if eval_data is not None:
        eval_data = eval_data.map(convert_func, batched=False, remove_columns=column_names,
                                **kwargs)

    if args.task_type == "sft":
        process_dataset_class = preprocess_supervised_dataset
    elif args.task_type == "dpo":
        process_dataset_class = preprocess_pairwise_dataset
    else:
        raise NotImplementedError(f"Unknown task_type: {args.task_type}")

    preprocess_func = partial(
        process_dataset_class,
        encoder=encoder,
        tokenizer=tokenizer,
        processor=processor,
        data_args=args,
    )
    kwargs.update({"desc": "Running tokenizer on dataset"})
    column_names = list(next(iter(train_data)).keys())
    train_data = train_data.map(preprocess_func, batched=True,
                                batch_size=args.preprocessing_batch_size,
                                remove_columns=column_names, **kwargs)
    if eval_data is not None:
        eval_data = eval_data.map(preprocess_func, batched=True,
                                  batch_size=args.preprocessing_batch_size,
                                  remove_columns=column_names, **kwargs)
    return train_data, eval_data


# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py
@dataclass
class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Data collator that supports VLMs.

    Features should contain input_ids, attention_mask, labels and images.
    """

    encoder_utils: Optional[VisionEncoderUtils] = None
    vision_processor: Optional[ProcessorMixin] = None

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids = [], [], [], [], []
        for feature in features:
            images = feature.pop("images", None) or []
            videos = feature.pop("videos", None) or []
            batch_images.extend(images)
            batch_videos.extend(videos)
            batch_imglens.append(len(images))
            batch_vidlens.append(len(videos))
            batch_input_ids.append(feature["input_ids"])

        mm_inputs = self.encoder_utils.mm_plugin.get_mm_inputs(
            batch_images, batch_videos, batch_imglens, batch_vidlens, batch_input_ids, self.vision_processor
        )
        if "token_type_ids" in mm_inputs:
            token_type_ids = mm_inputs.pop("token_type_ids")
            for i, feature in enumerate(features):
                feature["token_type_ids"] = token_type_ids[i]

        features: Dict[str, torch.Tensor] = super().__call__(features)
        features.update(mm_inputs)
        if isinstance(features.get("pixel_values"), list):  # for pixtral inputs
            features = features.data  # use default_collate() instead of BatchEncoding.to()
        return features

# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py
@dataclass
class SFTDataCollatorWith4DAttentionMask(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = self.prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)
        return features

    def prepare_4d_attention_mask(self, attention_mask_with_indices, dtype) -> torch.Tensor:
        r"""
        Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
        while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

        e.g.
        ```python
        # input
        [[1, 1, 2, 2, 2, 0]]
        # output
        [
            [
                [
                    [o, x, x, x, x, x],
                    [o, o, x, x, x, x],
                    [x, x, o, x, x, x],
                    [x, x, o, o, x, x],
                    [x, x, o, o, o, x],
                    [x, x, x, x, x, x],
                ]
            ]
        ]
        ```
        where `o` equals to `0.0`, `x` equals to `min_dtype`.
        """
        bsz, seq_len = attention_mask_with_indices.size()
        min_dtype = torch.finfo(dtype).min
        expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
        # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
        padding_mask = torch.where(expanded_mask != 0, 1, 0)
        # Create a block-diagonal mask.
        attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
        # Use the lower triangular mask to zero out the upper triangular part
        attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
        # Invert the attention mask.
        attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
        return attention_mask_4d

# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/collator.py
@dataclass
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature[f"{key}_input_ids"],
                    "attention_mask": feature[f"{key}_attention_mask"],
                    "labels": feature[f"{key}_labels"],
                    "images": feature["images"],
                    "videos": feature["videos"],
                }
                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)
 

def build_train_and_valid_datasets(args, tokenizer, processor, encoder_utils, strategy):
    train_ds, eval_ds = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )

    return get_preprocessed_dataset(args, [train_ds, eval_ds], encoder_utils, tokenizer, processor)


def build_data_collator(args, tokenizer, encoder_utils, vision_processor):
    collator_class = None
    kwargs = {}
    if args.task_type == "dpo":
        collator_class = PairwiseDataCollatorWithPadding
    elif args.task_type == "sft":
        collator_class = SFTDataCollatorWith4DAttentionMask
        kwargs = {
            "block_diag_attn": args.neat_packing,
            "attn_implementation": "flash_attention_2" if args.flash_attn else None,
            "compute_dtype": torch.bfloat16 if args.bf16 else torch.float16
        }
    else:
        raise NotImplementedError(f"Task type {args.task_type} not supported.")

    data_collator = collator_class(
        encoder_utils=encoder_utils,
        vision_processor=vision_processor,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        tokenizer=tokenizer,
        **kwargs
    )
    return data_collator
