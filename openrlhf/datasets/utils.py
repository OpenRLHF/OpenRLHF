import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None


def convert_conversations(args, example):
    r"""
    Convert conversations
    """
    def create_mapping():
        return {
            args.original_user_tag: args.processed_user_tag,
            args.original_assistant_tag: args.processed_assistant_tag,
        }

    tag_mapping = create_mapping()
    content_tag = args.content_tag
    conversation_tag = args.conversation_tag
    role_tag = args.role_tag

    odd_tags = (args.original_user_tag)
    even_tags = (args.original_assistant_tag)
    accept_tags = (odd_tags, even_tags)

    aligned_messages = []
    messages = example[conversation_tag]
    for idx, message in enumerate(messages):
        assert message[role_tag] in accept_tags[idx % 2]

        content = convert_to_visual_content(args.model_arch, message[content_tag])
        aligned_messages.append(
            {"role": tag_mapping[message[role_tag]], "content": content}
        )

    if args.task_type == "dpo":
        assert isinstance(example[args.chosen_key], dict)
        assert isinstance(example[args.rejected_key], dict)

        chosen = example[args.chosen_key]
        rejected = example[args.rejected_key]
        chosen_content = convert_to_visual_content(args.model_arch, chosen[content_tag])
        rejected_content = convert_to_visual_content(args.model_arch, rejected[content_tag])

        chosen_messages = [{"role": tag_mapping[chosen[role_tag]], "content": chosen_content}]
        reject_messages = [{"role": tag_mapping[rejected[role_tag]], "content": rejected_content}]

        return aligned_messages + chosen_messages, aligned_messages + reject_messages

    return aligned_messages


def process_vision(image_processor, images, videos=None):
    if images is not None:
        image_inputs = image_processor(
            images=images,
            videos=None,
            return_tensors="pt",
        )
    else:
        image_inputs = {}

    if videos is not None:
        videos_inputs = image_processor(
            images=None,
            videos=videos,
            return_tensors="pt",
        )
    else:
        videos_inputs = {}

    return BatchFeature(data={**image_inputs, **videos_inputs})


def padding_vision_token(image_processor, image_token, video_token, text, image_grid_thw, video_grid_thw=None):
    merge_length = image_processor.merge_size**2
    if image_grid_thw is not None:
        index = 0
        while image_token in text:
            text = text.replace(
                image_token,
                "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1)
            index += 1
        text = text.replace("<|placeholder|>", image_token)

    if video_grid_thw is not None:
        index = 0
        while video_token in text:
            text = text.replace(
                video_token,
                "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1)
            index += 1
        text = text.replace("<|placeholder|>", video_token)

    return text


def convert_to_visual_content(model_arch,
                              user_input: str,
                              image_pattern: str = '<image>',
                              video_pattern: str = '<video>'):
    if model_arch in ["qwen2_vl"]:
        return convert_to_qwen2vl_content(user_input, image_pattern, video_pattern)
    else:
        raise NotImplementedError(f"model_arch {model_arch} is not implemented yet")


# Reference by: https://github.com/alibaba/Pai-Megatron-Patch/blob/main/examples/qwen2_vl/dataset_helpers.py#L75
def convert_to_qwen2vl_content(user_input: str,
                               image_pattern: str = '<image>',
                               video_pattern: str = '<video>'):
    """
        Split user input into format Qwen2VL tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start].strip()})

        contents.append({
            "type":
            matched.string[start:end][1:-1],
            matched.string[start:end][1:-1]:
            str(mm_idx[matched.string[start:end][1:-1]])
        })

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:len(user_input)].strip()})

    return contents
