from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch

_EVAL_UTILS_PATH = Path(__file__).resolve().parents[1] / "openrlhf" / "trainer" / "ppo_utils" / "eval_utils.py"
_SPEC = spec_from_file_location("test_eval_utils", _EVAL_UTILS_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

aggregate_eval_metrics = _MODULE.aggregate_eval_metrics
build_eval_records = _MODULE.build_eval_records
get_eval_sample_preview = _MODULE.get_eval_sample_preview
save_eval_samples = _MODULE.save_eval_samples


def _make_sample(prompt, label, reward, response_length, truncated, datasource="math"):
    return SimpleNamespace(
        prompts=[prompt],
        labels=[label],
        rewards=torch.tensor([reward]),
        info={
            "response_length": torch.tensor([response_length]),
            "total_length": torch.tensor([response_length + 3]),
            "truncated": torch.tensor([truncated]),
            "reward": torch.tensor([reward]),
        },
    )


def test_build_eval_records_and_metrics_with_duplicate_prompts(tmp_path):
    samples_list = [
        _make_sample("prompt-a", "label-a", 1.0, 10, False),
        _make_sample("prompt-a", "label-b", 0.0, 12, True),
        _make_sample("prompt-b", "label-c", 0.5, 8, False, datasource="code"),
        _make_sample("prompt-b", "label-d", 1.0, 9, False, datasource="code"),
    ]
    prompt_metadata = {
        "prompt-a": [
            {"datasource": "math", "label": "label-a", "eval_index": 0},
            {"datasource": "math", "label": "label-b", "eval_index": 1},
        ],
        "prompt-b": [
            {"datasource": "code", "label": "label-c", "eval_index": 2},
            {"datasource": "code", "label": "label-d", "eval_index": 3},
        ],
    }

    records = build_eval_records(samples_list, prompt_metadata)
    assert len(records) == 4
    assert records[0]["datasource"] == "math"
    assert records[2]["datasource"] == "code"

    metrics = aggregate_eval_metrics(records, n_samples_per_prompt=2)
    assert metrics["reward_mean"] == 0.625
    assert metrics["response_length_max"] == 12.0
    assert metrics["truncated_rate"] == 0.25
    assert metrics["pass1"] == 0.625
    assert metrics["pass2"] == 1.0
    assert metrics["math_pass2"] == 1.0
    assert metrics["code_pass2"] == 1.0

    preview = get_eval_sample_preview(records, max_samples=2)
    assert len(preview) == 2
    assert preview[0]["prompt"] == "prompt-a"

    save_path = save_eval_samples(records, str(tmp_path), global_step=12)
    assert save_path is not None
    content = (tmp_path / "eval_samples_step_12.jsonl").read_text(encoding="utf-8")
    assert "prompt-a" in content
