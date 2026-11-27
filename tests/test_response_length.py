"""
Tests for response_length computation in _process_response_into_experience.

response_length must equal action_mask.sum() (only model-generated tokens),
not the span from first to last 1 (which over-counts in multi-turn agentic
training because tool-call tokens sit in the gaps between turns).
"""

import sys
import types


def _load_samples_generator():
    # samples_generator imports vllm (directly and via vllm_engine); stub those
    # out when vllm is not installed so the test runs on CPU-only machines.
    inserted_stubs = []
    try:
        import vllm  # noqa: F401
    except ImportError:
        vllm_stub = types.ModuleType("vllm")
        vllm_stub.SamplingParams = object
        engine_stub = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
        engine_stub.batch_vllm_engine_call = None
        for name, stub in (("vllm", vllm_stub), ("openrlhf.trainer.ray.vllm_engine", engine_stub)):
            if name not in sys.modules:
                sys.modules[name] = stub
                inserted_stubs.append(name)

    try:
        from openrlhf.trainer.ppo_utils import samples_generator
    finally:
        # Drop the import-time stubs so later tests see the real (or missing) packages.
        for name in inserted_stubs:
            del sys.modules[name]

    return samples_generator


def _process(action_ranges, num_tokens, max_len=2048):
    """Run the production conversion on a synthetic agent response."""
    response = {
        "observation_tokens": list(range(num_tokens)),
        "action_ranges": action_ranges,
        "rollout_log_probs": None,
        "prompt": "p",
        "label": "l",
    }
    SamplesGenerator = _load_samples_generator().SamplesGenerator
    # The method does not touch instance state, so no generator is constructed.
    return SamplesGenerator._process_response_into_experience(None, response, max_len=max_len)


def test_single_turn():
    # [prompt(3)] [response(5)]
    experience = _process([(3, 8)], num_tokens=8)
    assert experience.response_length.item() == 5


def test_multiturn_excludes_tool_tokens():
    # [prompt(3)] [turn1(4)] [tool(3)] [turn2(5)]
    experience = _process([(3, 7), (10, 15)], num_tokens=15)
    assert experience.response_length.item() == 9  # span would give 12


def test_multiturn_single_token_turns():
    experience = _process([(1, 2), (3, 4)], num_tokens=5)
    assert experience.response_length.item() == 2  # span would give 3


def test_truncation_counts_only_kept_action_tokens():
    # [prompt(3)] [turn1(4)] [tool(3)] [turn2(5)], truncated to 12 tokens:
    # turn2 keeps tokens 10 and 11 only.
    experience = _process([(3, 7), (10, 15)], num_tokens=15, max_len=12)
    assert experience.response_length.item() == 6


def test_empty_action_ranges():
    experience = _process([], num_tokens=8)
    assert experience.response_length.item() == 0
