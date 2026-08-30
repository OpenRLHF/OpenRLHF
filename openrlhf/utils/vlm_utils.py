"""VLM (Vision-Language Model) utilities for multimodal RLHF training.

Handles image loading, processor-based tokenization, and multimodal tensor
extraction for Qwen3.5 and Gemma4 vision-language models.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _is_base64_image(s: str) -> bool:
    """Heuristic check for base64-encoded image data."""
    if s.startswith("data:image"):
        return True
    # Bare base64: must be reasonably long and contain only valid chars.
    if len(s) > 256:
        import re

        return bool(re.fullmatch(r"[A-Za-z0-9+/\n\r]+=*", s[:512]))
    return False


def load_images(image_refs: Union[str, List[str], Image.Image, List[Any]]) -> List[Image.Image]:
    """Load PIL images from paths, URLs, base64 strings, raw bytes, or PIL objects.

    Invalid entries (missing files, broken URLs, unsupported types) are
    skipped with a warning instead of crashing the training run.
    """
    if image_refs is None:
        return []
    if not isinstance(image_refs, list):
        image_refs = [image_refs]

    pil_images = []
    for img in image_refs:
        try:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            elif isinstance(img, bytes):
                pil_images.append(Image.open(io.BytesIO(img)))
            elif isinstance(img, str):
                if img.startswith(("http://", "https://")):
                    import requests

                    pil_images.append(Image.open(io.BytesIO(requests.get(img, timeout=30).content)))
                elif _is_base64_image(img):
                    import base64

                    # Strip optional data-URI header: "data:image/png;base64,..."
                    raw = img.split(",", 1)[-1] if img.startswith("data:") else img
                    pil_images.append(Image.open(io.BytesIO(base64.b64decode(raw))))
                else:
                    pil_images.append(Image.open(img))
            else:
                logger.warning(f"Skipping unsupported image type: {type(img)}")
        except Exception as e:
            logger.warning(f"Failed to load image {img!r}: {e}")
    return pil_images


def process_prompt_with_images(
    processor, prompt: str, images: Any
) -> Tuple[List[int], Optional[Dict], List[Image.Image]]:
    """Tokenize a prompt with images using a VLM processor (AutoProcessor).

    Returns:
        (token_ids, mm_train_inputs, pil_images)
        - mm_train_inputs: dict of multimodal tensors (pixel_values, image_grid_thw, ...)
          or None when no images are present.
        - pil_images: loaded PIL images (reused for vLLM multi_modal_data).
    """
    pil_images = load_images(images)

    # No images → text-only tokenization (valid for text-only samples in mixed datasets).
    # Filter out None entries before checking — datasets may use [None] for text-only samples.
    non_none_refs = [img for img in (images if isinstance(images, list) else [images])] if images else []
    non_none_refs = [img for img in non_none_refs if img is not None]
    if not pil_images:
        if non_none_refs:
            # Caller provided real image references but none loaded successfully.
            # Falling through to text-only would leave image placeholder
            # tokens in the prompt with no matching pixel_values.
            raise ValueError(
                f"All images failed to load ({images!r}).  The prompt likely "
                "contains image placeholder tokens that require pixel_values."
            )
        token_ids = processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids, None, []

    # Warn on partial load failures — the prompt may expect more images
    # than were actually loaded, which can misalign placeholder tokens.
    expected = len(non_none_refs)
    if len(pil_images) < expected:
        logger.warning(
            f"Only {len(pil_images)}/{expected} images loaded successfully. "
            "Image placeholder tokens in the prompt may not match pixel_values."
        )

    proc_out = processor(text=[prompt], images=pil_images, add_special_tokens=False, return_tensors="pt")
    token_ids = proc_out["input_ids"][0].tolist()

    # Keep only multimodal tensors (pixel_values, image_grid_thw, ...).
    # Token-type fields are excluded because they are sequence-length
    # dependent and get reconstructed from input_ids during training
    # (see Actor.forward) — the training sequence includes the response
    # while the processor only saw the prompt.
    _skip_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    mm_train_inputs = {k: v for k, v in proc_out.items() if k not in _skip_keys}

    return token_ids, (mm_train_inputs or None), pil_images


def dedup_media_tokens(token_ids: List[int], pad_token_ids: set) -> List[int]:
    """Collapse consecutive image/video pad tokens to a single placeholder.

    vLLM expects one placeholder per image and expands it internally.
    Passing already-expanded token IDs causes double-expansion.
    """
    ids = np.asarray(token_ids)
    is_pad = np.isin(ids, list(pad_token_ids))
    keep = np.ones(len(ids), dtype=bool)
    keep[1:] &= ~(is_pad[1:] & is_pad[:-1])
    return ids[keep].tolist()


def accumulate_mm_inputs(existing: Optional[Dict], new: Optional[Dict]) -> Optional[Dict]:
    """Incrementally merge a new step's multimodal tensors into the running accumulator.

    Handles key asymmetry: keys present in only one dict are preserved as-is,
    keys present in both are concatenated along dim=0.
    """
    if new is None:
        return existing
    if existing is None:
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in new.items()}
    merged = {}
    for k in set(existing) | set(new):
        if k in existing and k in new:
            merged[k] = torch.cat([existing[k], new[k]], dim=0)
        elif k in existing:
            merged[k] = existing[k]
        else:
            merged[k] = new[k]
    return merged


def merge_mm_train_inputs(mm_train_inputs_list: list, device) -> Dict[str, torch.Tensor]:
    """Merge per-sample multimodal tensor dicts into a single batched dict.

    ``mm_train_inputs_list`` comes from Experience: each element is either a
    list of per-sample dicts (one dict or None per sample) or a single dict/None.
    Tensors are concatenated along dim=0 and moved to *device*.
    """
    merged: Dict[str, list] = {}
    for item in mm_train_inputs_list:
        # Each item is a list of per-sample dicts (from Experience.mm_train_inputs).
        for mm_dict in (item if isinstance(item, list) else [item]):
            if mm_dict is None:
                continue
            for key, val in mm_dict.items():
                merged.setdefault(key, []).append(val if isinstance(val, torch.Tensor) else torch.tensor(val))

    return {k: torch.cat(v, dim=0).to(device) for k, v in merged.items()} if merged else {}
