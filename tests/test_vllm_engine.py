"""Tests for vLLM engine environment setup."""

import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_vllm_engine_dependencies():
    originals = {}

    fake_ray = ModuleType("ray")
    fake_ray.remote = lambda obj=None, **_kwargs: obj if obj is not None else lambda decorated: decorated
    fake_ray.get = MagicMock()
    fake_ray.get_gpu_ids = MagicMock(return_value=[0])
    fake_ray.util = SimpleNamespace(placement_group_table=MagicMock(return_value={"bundles_to_node_id": {}}))

    fake_placement_group = ModuleType("ray.util.placement_group")
    fake_placement_group.placement_group = MagicMock()

    fake_scheduling_strategies = ModuleType("ray.util.scheduling_strategies")
    fake_scheduling_strategies.PlacementGroupSchedulingStrategy = MagicMock()

    fake_vllm = ModuleType("vllm")
    fake_vllm.__version__ = "0.11.0"
    fake_vllm.AsyncEngineArgs = MagicMock()
    fake_vllm.AsyncLLMEngine = MagicMock()

    fake_vllm_inputs = ModuleType("vllm.inputs")
    fake_vllm_inputs.TokensPrompt = dict

    fake_vllm_utils = ModuleType("vllm.utils")
    fake_vllm_utils.random_uuid = MagicMock(return_value="uuid")

    fake_agent = ModuleType("openrlhf.utils.agent")
    fake_agent.AgentExecutorBase = object
    fake_agent.SingleTurnAgentExecutor = MagicMock()

    modules = {
        "ray": fake_ray,
        "ray.util.placement_group": fake_placement_group,
        "ray.util.scheduling_strategies": fake_scheduling_strategies,
        "vllm": fake_vllm,
        "vllm.inputs": fake_vllm_inputs,
        "vllm.utils": fake_vllm_utils,
        "openrlhf.utils.agent": fake_agent,
    }

    for name, module in modules.items():
        originals[name] = sys.modules.get(name)
        sys.modules[name] = module

    sys.modules.pop("openrlhf.trainer.ray.vllm_engine", None)

    yield

    sys.modules.pop("openrlhf.trainer.ray.vllm_engine", None)
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def _clean_vllm_env():
    env_vars = ("VLLM_RAY_PER_WORKER_GPUS", "VLLM_RAY_BUNDLE_INDICES")
    original = {name: os.environ.get(name) for name in env_vars}
    for name in env_vars:
        os.environ.pop(name, None)

    yield

    for name in env_vars:
        if original[name] is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original[name]


def _new_actor():
    vllm_engine = importlib.import_module("openrlhf.trainer.ray.vllm_engine")
    return object.__new__(vllm_engine.RolloutRayActor)


class TestConfigureDeviceEnv:
    def test_bundle_indices_full_gpu(self):
        actor = _new_actor()

        actor._configure_device_env(backend="ray", bundle_indices=[0, 1], num_gpus=1)

        assert os.environ["VLLM_RAY_PER_WORKER_GPUS"] == "1"
        assert os.environ["VLLM_RAY_BUNDLE_INDICES"] == "0,1"

    def test_bundle_indices_fractional_gpu(self):
        actor = _new_actor()

        actor._configure_device_env(backend="ray", bundle_indices=[0, 1], num_gpus=0.2)

        assert os.environ["VLLM_RAY_PER_WORKER_GPUS"] == "0.2"
        assert os.environ["VLLM_RAY_BUNDLE_INDICES"] == "0,1"

    def test_no_bundle_indices_skips_vllm_env_vars(self):
        actor = _new_actor()

        actor._configure_device_env(backend="ray", bundle_indices=None, num_gpus=0.2)

        assert "VLLM_RAY_PER_WORKER_GPUS" not in os.environ
        assert "VLLM_RAY_BUNDLE_INDICES" not in os.environ
