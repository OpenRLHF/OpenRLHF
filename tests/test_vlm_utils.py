import importlib.util
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VLM_UTILS_PATH = _REPO_ROOT / "openrlhf" / "utils" / "vlm_utils.py"

spec = importlib.util.spec_from_file_location("vlm_utils", _VLM_UTILS_PATH)
vlm_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vlm_utils)

accumulate_mm_inputs = vlm_utils.accumulate_mm_inputs
dedup_media_tokens = vlm_utils.dedup_media_tokens


def test_dedup_media_tokens_collapses_consecutive_pads():
    token_ids = [5, 1, 1, 1, 6, 1, 7]
    pad_token_ids = {1}

    assert dedup_media_tokens(token_ids, pad_token_ids) == [5, 1, 6, 1, 7]


def test_dedup_media_tokens_preserves_non_pad_tokens():
    token_ids = [2, 3, 4, 5]
    pad_token_ids = {1}

    assert dedup_media_tokens(token_ids, pad_token_ids) == token_ids


def test_accumulate_mm_inputs_concatenates_matching_keys_and_preserves_new_keys():
    existing = {
        "pixel_values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    new = {
        "pixel_values": torch.tensor([[5.0, 6.0]]),
        "image_grid_thw": torch.tensor([[7, 8, 9]]),
    }

    merged = accumulate_mm_inputs(existing, new)

    assert torch.equal(
        merged["pixel_values"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert torch.equal(merged["image_grid_thw"], torch.tensor([[7, 8, 9]]))


def test_accumulate_mm_inputs_clones_new_tensors_when_existing_is_none():
    new = {
        "pixel_values": torch.tensor([[1.0, 2.0]]),
    }

    merged = accumulate_mm_inputs(None, new)
    new["pixel_values"][0, 0] = 99.0

    assert merged["pixel_values"][0, 0].item() == 1.0


def test_accumulate_mm_inputs_returns_existing_when_new_is_none():
    existing = {
        "pixel_values": torch.tensor([[1.0, 2.0]]),
    }

    merged = accumulate_mm_inputs(existing, None)

    assert merged is existing