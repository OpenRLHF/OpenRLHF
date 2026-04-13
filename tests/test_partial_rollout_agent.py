from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest
import torch

PROJECT_ROOT = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").exists())
AGENT_PATH = PROJECT_ROOT / "openrlhf/utils/agent.py"


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):  # noqa: ARG002
        token_ids = [ord(ch) for ch in text]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(chr(token_id) for token_id in token_ids)


class _SamplingParams:
    def __init__(self, max_tokens: int, logprobs: int | None = None):
        self.max_tokens = max_tokens
        self.logprobs = logprobs


class _FakeRequestOutput:
    def __init__(self, token_ids: list[int], finish_reason: str):
        logprobs = [{token_id: SimpleNamespace(logprob=-0.1)} for token_id in token_ids]
        self.outputs = [
            SimpleNamespace(
                token_ids=token_ids,
                logprobs=logprobs,
                finish_reason=finish_reason,
                text="".join(chr(token_id) for token_id in token_ids),
            )
        ]


class _FakeLLMEngine:
    def __init__(self, chunks, *, partial_rollout: bool, mask_offpolicy_in_partial_rollout: bool):
        self._chunks = list(chunks)
        self.partial_rollout = partial_rollout
        self.mask_offpolicy_in_partial_rollout = mask_offpolicy_in_partial_rollout
        self.weight_version = self._chunks[0]["version"]
        self.observed_prompts: list[list[int]] = []

    def get_weight_version(self) -> int:
        return self._chunks[0]["version"] if self._chunks else self.weight_version

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None):  # noqa: ARG002
        self.observed_prompts.append(list(prompt_token_ids))
        chunk = self._chunks.pop(0)
        self.weight_version = chunk["version"]
        return _FakeRequestOutput(chunk["token_ids"], chunk["finish_reason"])


def _load_agent_module():
    logging_utils_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_utils_module.init_logger = lambda name: logging.getLogger(name)

    openrlhf_package = types.ModuleType("openrlhf")
    openrlhf_package.__path__ = []
    utils_package = types.ModuleType("openrlhf.utils")
    utils_package.__path__ = []

    original_modules = {
        "openrlhf": sys.modules.get("openrlhf"),
        "openrlhf.utils": sys.modules.get("openrlhf.utils"),
        "openrlhf.utils.logging_utils": sys.modules.get("openrlhf.utils.logging_utils"),
    }
    sys.modules["openrlhf"] = openrlhf_package
    sys.modules["openrlhf.utils"] = utils_package
    sys.modules["openrlhf.utils.logging_utils"] = logging_utils_module

    try:
        spec = importlib.util.spec_from_file_location("_agent_under_test", AGENT_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_turn_executor_continues_partial_rollout_and_masks_old_tokens() -> None:
    module = _load_agent_module()
    executor = module.SingleTurnAgentExecutor()
    tokenizer = _Tokenizer()
    llm_engine = _FakeLLMEngine(
        [
            {"version": 0, "token_ids": [ord("a"), ord("b")], "finish_reason": "abort"},
            {"version": 1, "token_ids": [ord("c"), ord("d")], "finish_reason": "stop"},
        ],
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
    )

    output = await executor.execute(
        prompt="P",
        label="L",
        sampling_params=_SamplingParams(max_tokens=4, logprobs=1),
        max_length=8,
        hf_tokenizer=tokenizer,
        llm_engine=llm_engine,
    )

    assert output["observation_tokens"] == [ord("P"), ord("a"), ord("b"), ord("c"), ord("d")]
    assert output["action_loss_mask"] == [0, 0, 1, 1]
    assert output["min_weight_version"] == 0
    assert output["max_weight_version"] == 1
    assert output["partial_old_token_count"] == 2
    assert llm_engine.observed_prompts == [
        [ord("P")],
        [ord("P"), ord("a"), ord("b")],
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_single_turn_executor_keeps_old_tokens_when_masking_disabled() -> None:
    module = _load_agent_module()
    executor = module.SingleTurnAgentExecutor()
    tokenizer = _Tokenizer()
    llm_engine = _FakeLLMEngine(
        [
            {"version": 0, "token_ids": [ord("x")], "finish_reason": "abort"},
            {"version": 1, "token_ids": [ord("y")], "finish_reason": "stop"},
        ],
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=False,
    )

    output = await executor.execute(
        prompt="Q",
        label="L",
        sampling_params=_SamplingParams(max_tokens=2, logprobs=1),
        max_length=4,
        hf_tokenizer=tokenizer,
        llm_engine=llm_engine,
    )

    assert output["action_loss_mask"] == [1, 1]
    assert output["partial_old_token_count"] == 1
