"""VLM (Vision-Language Model) utilities for multimodal RLHF training.

Handles image loading, processor-based tokenization, and multimodal tensor
extraction for models like Qwen3.5, Gemma4, LLaVA, InternVL etc.
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union

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


def process_prompt_with_images(processor, prompt: str, images: Any) -> Tuple[List[int], Optional[Dict]]:
    """Use a VLM processor to tokenize a prompt with images.

    Returns:
        (prompt_token_ids, mm_train_inputs) where mm_train_inputs is a dict
        of tensors (pixel_values, image_grid_thw, etc.) for the training
        forward pass, or None if no images.
    """
    pil_images = load_images(images)
    if not pil_images:
        token_ids = processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids, None

    # Call processor with text + images — this correctly inserts image tokens
    # and produces pixel_values etc.
    proc_out = processor(
        text=[prompt],
        images=pil_images,
        return_tensors="pt",
    )

    token_ids = proc_out["input_ids"][0].tolist()

    # Extract multimodal tensors (everything except text tokenizer outputs)
    _text_keys = {"input_ids", "attention_mask", "token_type_ids"}
    mm_train_inputs = {k: v for k, v in proc_out.items() if k not in _text_keys}

    return token_ids, mm_train_inputs if mm_train_inputs else None
