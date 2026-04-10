"""VLM (Vision-Language Model) utilities for multimodal RLHF training.

Handles image loading, processor-based tokenization, and multimodal tensor
extraction for models like Qwen2.5-VL, Gemma3/4, LLaVA, InternVL etc.
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image


def load_images(image_refs: Union[str, List[str], Image.Image, List[Any]]) -> List[Image.Image]:
    """Load PIL images from paths, URLs, or pass-through PIL objects."""
    if image_refs is None:
        return []
    if not isinstance(image_refs, list):
        image_refs = [image_refs]

    pil_images = []
    for img in image_refs:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        elif isinstance(img, str):
            if img.startswith(("http://", "https://")):
                import requests

                pil_images.append(Image.open(io.BytesIO(requests.get(img).content)))
            else:
                pil_images.append(Image.open(img))
    return pil_images


def process_prompt_with_images(
    processor, prompt: str, images: Any
) -> Tuple[List[int], Optional[Dict], List[Image.Image]]:
    """Tokenize a prompt with images using a VLM processor.

    Handles image loading internally — *images* can be paths, URLs, or PIL objects.

    Returns:
        (token_ids, mm_train_inputs, pil_images)
        - mm_train_inputs: dict of tensors (pixel_values, image_grid_thw …) or None
        - pil_images: loaded PIL images (needed for vLLM multi_modal_data)
    """
    pil_images = load_images(images)
    if not pil_images:
        token_ids = processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids, None, []

    proc_out = processor(text=[prompt], images=pil_images, return_tensors="pt")
    token_ids = proc_out["input_ids"][0].tolist()

    # Extract multimodal tensors only (pixel_values, image_grid_thw, …).
    # mm_token_type_ids is excluded — it is sequence-length dependent and
    # reconstructed from input_ids during training (see Actor.forward).
    _skip_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    mm_train_inputs = {k: v for k, v in proc_out.items() if k not in _skip_keys}

    return token_ids, (mm_train_inputs or None), pil_images


def merge_mm_train_inputs(mm_train_inputs_list: list, device) -> Dict[str, torch.Tensor]:
    """Merge per-sample multimodal tensor dicts into a single batched dict.

    Each element is a dict like {"pixel_values": (N, D), "image_grid_thw": (N, 3)}.
    Tensors are concatenated along dim=0.  None entries are skipped.
    """
    merged: Dict[str, list] = {}
    for item in mm_train_inputs_list:
        for mm_dict in (item if isinstance(item, list) else [item]):
            if mm_dict is None:
                continue
            for key, val in mm_dict.items():
                merged.setdefault(key, []).append(val if isinstance(val, torch.Tensor) else torch.tensor(val))

    return {k: torch.cat(v, dim=0).to(device) for k, v in merged.items()} if merged else {}
