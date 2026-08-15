import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_tokenspeed_engine_module():
    root = Path(__file__).resolve().parents[1]
    package_name = "_tokenspeed_engine_test"
    module_name = f"{package_name}.tokenspeed_engine"

    package = types.ModuleType(package_name)
    package.__path__ = [str(root / "openrlhf" / "trainer" / "ray")]

    utils = types.ModuleType(f"{package_name}.utils")
    utils.ray_noset_visible_devices = lambda: False

    agent = types.ModuleType("openrlhf.utils.agent")
    agent.AgentExecutorBase = type("AgentExecutorBase", (), {})
    agent.SingleTurnAgentExecutor = MagicMock()

    fake_ray = types.ModuleType("ray")
    fake_ray.remote = lambda cls: cls
    fake_ray.get_gpu_ids = lambda: []
    fake_ray.get = lambda refs: refs

    placement_group_module = types.ModuleType("ray.util.placement_group")
    placement_group_module.placement_group = MagicMock()

    scheduling_module = types.ModuleType("ray.util.scheduling_strategies")
    scheduling_module.PlacementGroupSchedulingStrategy = MagicMock()

    originals = {}
    fakes = {
        package_name: package,
        f"{package_name}.utils": utils,
        "openrlhf.utils.agent": agent,
        "ray": fake_ray,
        "ray.util": types.ModuleType("ray.util"),
        "ray.util.placement_group": placement_group_module,
        "ray.util.scheduling_strategies": scheduling_module,
    }
    for name, module in fakes.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "openrlhf" / "trainer" / "ray" / "tokenspeed_engine.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    return module


tokenspeed_engine = _load_tokenspeed_engine_module()


def test_select_ipc_handle_returns_matching_handle():
    handle = object()

    assert tokenspeed_engine._select_ipc_handle({2: handle}, 2) is handle


def test_select_ipc_handle_fails_when_gpu_id_is_missing():
    with pytest.raises(RuntimeError, match="GPU ID 3 not found"):
        tokenspeed_engine._select_ipc_handle({0: object(), 1: object()}, 3)


def test_adapt_tokenspeed_output_matches_vllm_shape():
    output = tokenspeed_engine._adapt_tokenspeed_output(
        {
            "text": "hello",
            "output_ids": [11, 12],
            "meta_info": {
                "finish_reason": {"type": "length", "length": 2},
                "output_token_logprobs": [(-0.1, 11, None), (-0.2, 12, None)],
            },
        },
        require_logprobs=True,
    )

    completion = output.outputs[0]
    assert completion.text == "hello"
    assert completion.token_ids == [11, 12]
    assert completion.finish_reason == "length"
    assert completion.logprobs[0][11].logprob == pytest.approx(-0.1)
    assert completion.logprobs[1][12].logprob == pytest.approx(-0.2)


def test_adapt_tokenspeed_output_fails_when_requested_logprobs_are_missing():
    with pytest.raises(RuntimeError, match="did not return output token logprobs"):
        tokenspeed_engine._adapt_tokenspeed_output(
            {"text": "hello", "output_ids": [11], "meta_info": {"finish_reason": {"type": "stop"}}},
            require_logprobs=True,
        )


def test_tokenspeed_sampling_params_requires_generation_budget():
    sampling_params = types.SimpleNamespace(max_tokens=None)

    with pytest.raises(ValueError, match="max_tokens"):
        tokenspeed_engine._tokenspeed_sampling_params(sampling_params)


def test_tokenspeed_sampling_params_maps_vllm_names():
    sampling_params = types.SimpleNamespace(
        max_tokens=7,
        min_tokens=2,
        temperature=0.8,
        top_p=0.9,
        top_k=40,
        skip_special_tokens=False,
        stop_token_ids=[1, 2],
    )

    assert tokenspeed_engine._tokenspeed_sampling_params(sampling_params) == {
        "max_new_tokens": 7,
        "min_new_tokens": 2,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "skip_special_tokens": False,
        "stop_token_ids": [1, 2],
    }
