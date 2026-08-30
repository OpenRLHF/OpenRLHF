"""Tests for Ray runtime env_vars handling in train_ppo_ray.

Verifies that user-set environment variables (e.g. NCCL_DEBUG=INFO) are
respected and not overridden by hardcoded defaults.

These tests mock the ``ray`` module so they run without a Ray installation.
"""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_ray_module():
    """Inject a fake ``ray`` module into sys.modules before each test.

    This allows ``import ray`` inside train_ppo_ray.py to succeed even when
    Ray is not installed.  We also mock ``ray.util.placement_group`` which is
    imported at module level.
    """
    fake_ray = MagicMock()
    fake_ray.util = MagicMock()
    fake_ray.util.placement_group = MagicMock()
    fake_ray.util.placement_group.placement_group = MagicMock()

    originals = {}
    modules_to_fake = [
        "ray",
        "ray.util",
        "ray.util.placement_group",
    ]
    for mod in modules_to_fake:
        originals[mod] = sys.modules.get(mod)
        sys.modules[mod] = (
            fake_ray if mod == "ray" else getattr(fake_ray, mod.split(".", 1)[-1].replace(".", "_"), MagicMock())
        )

    # Also fake the sub-modules that train_ppo_ray.py imports transitively
    # through openrlhf.trainer.ray.*
    _ensure_fake_modules(originals)

    yield fake_ray

    # Restore original modules
    for mod, orig in originals.items():
        if orig is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = orig

    # Force re-import on next test
    sys.modules.pop("openrlhf.cli.train_ppo_ray", None)


def _ensure_fake_modules(originals: dict):
    """Pre-populate sys.modules with MagicMock for heavy dependencies.

    train_ppo_ray.py imports from openrlhf.trainer.ray.* which may pull in
    torch, deepspeed, vllm, etc.  We mock the entire openrlhf.trainer.ray
    subtree so those imports don't fail.
    """
    mock_modules = [
        "openrlhf.trainer.ray",
        "openrlhf.trainer.ray.launcher",
        "openrlhf.trainer.ray.ppo_actor",
        "openrlhf.trainer.ray.ppo_critic",
        "openrlhf.trainer.ray.vllm_engine",
        "openrlhf.utils",
    ]
    for mod in mock_modules:
        if mod not in sys.modules:
            originals[mod] = sys.modules.get(mod)
            sys.modules[mod] = MagicMock()


def _reload_train_module():
    """(Re)load openrlhf.cli.train_ppo_ray and return the module."""
    mod_name = "openrlhf.cli.train_ppo_ray"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


def _get_ray_init_env_vars(
    env_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Call train() with mocked Ray and return the env_vars passed to ray.init().

    Args:
        env_overrides: Environment variables to set before calling train().
            These simulate the user setting env vars via
            ``ray job submit --runtime-env-json``.

    Returns:
        The ``env_vars`` dict that was passed to ``ray.init(runtime_env=...)``.
    """
    env_overrides = env_overrides or {}
    mock_ray_init = MagicMock()

    # Patch os.environ so user overrides are visible
    with patch.dict(os.environ, env_overrides, clear=False):
        mod = _reload_train_module()

        # Configure the fake ray that is already in sys.modules
        fake_ray = sys.modules["ray"]
        fake_ray.is_initialized.return_value = False
        fake_ray.init = mock_ray_init

        # ray.init() should raise to short-circuit the rest of train()
        mock_ray_init.side_effect = SystemExit("stop after ray.init")

        try:
            mod.train(MagicMock())
        except SystemExit:
            pass

    mock_ray_init.assert_called_once()
    _, kwargs = mock_ray_init.call_args
    return kwargs["runtime_env"]["env_vars"]


# ---------------------------------------------------------------------------
# Tests: defaults (user has NOT set the env var)
# ---------------------------------------------------------------------------


class TestRayEnvVarsDefaults:
    """When user has NOT set env vars, sensible defaults should be used."""

    def test_nccl_debug_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NCCL_DEBUG", None)
            env_vars = _get_ray_init_env_vars()
        assert env_vars["NCCL_DEBUG"] == "WARN"

    def test_tokenizers_parallelism_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
            env_vars = _get_ray_init_env_vars()
        assert env_vars["TOKENIZERS_PARALLELISM"] == "true"

    def test_ray_zero_copy_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", None)
            env_vars = _get_ray_init_env_vars()
        assert env_vars["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] == "1"


# ---------------------------------------------------------------------------
# Tests: user overrides
# ---------------------------------------------------------------------------


class TestRayEnvVarsUserOverride:
    """When user HAS set env vars, their values must be preserved."""

    def test_nccl_debug_user_info(self):
        env_vars = _get_ray_init_env_vars({"NCCL_DEBUG": "INFO"})
        assert env_vars["NCCL_DEBUG"] == "INFO"

    def test_nccl_debug_user_trace(self):
        env_vars = _get_ray_init_env_vars({"NCCL_DEBUG": "TRACE"})
        assert env_vars["NCCL_DEBUG"] == "TRACE"

    def test_tokenizers_parallelism_user_false(self):
        env_vars = _get_ray_init_env_vars({"TOKENIZERS_PARALLELISM": "false"})
        assert env_vars["TOKENIZERS_PARALLELISM"] == "false"

    def test_ray_zero_copy_user_disabled(self):
        env_vars = _get_ray_init_env_vars({"RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "0"})
        assert env_vars["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] == "0"

    def test_multiple_user_overrides(self):
        env_vars = _get_ray_init_env_vars(
            {
                "NCCL_DEBUG": "INFO",
                "TOKENIZERS_PARALLELISM": "false",
                "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": "0",
            }
        )
        assert env_vars["NCCL_DEBUG"] == "INFO"
        assert env_vars["TOKENIZERS_PARALLELISM"] == "false"
        assert env_vars["RAY_ENABLE_ZERO_COPY_TORCH_TENSORS"] == "0"


# ---------------------------------------------------------------------------
# Tests: ray.init() skipped when already initialized
# ---------------------------------------------------------------------------


class TestRayInitSkippedWhenAlreadyInitialized:
    """When Ray is already initialized, ray.init() should NOT be called."""

    def test_ray_init_not_called_when_initialized(self):
        mod = _reload_train_module()

        fake_ray = sys.modules["ray"]
        fake_ray.is_initialized.return_value = True
        mock_init = MagicMock()
        fake_ray.init = mock_init

        # get_strategy is called right after ray.init; mock it to avoid
        # going deeper into training logic
        mod.get_strategy = MagicMock(side_effect=SystemExit("stop"))

        try:
            mod.train(MagicMock())
        except SystemExit:
            pass

        mock_init.assert_not_called()
