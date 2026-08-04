import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_logging_utils_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "openrlhf.utils.logging_utils", root / "openrlhf" / "utils" / "logging_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTable:
    def __init__(self, columns=None, data=None):
        self.columns = columns or []
        self.data = data or []


class _FakeWandb(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")
        self.api = SimpleNamespace(api_key="configured")
        self.metric_calls = []
        self.logged = []
        self.Table = _FakeTable

    def init(self, **kwargs):
        pass

    def define_metric(self, *args, **kwargs):
        self.metric_calls.append((args, kwargs))

    def log(self, payload):
        self.logged.append(payload)


def test_wandb_eval_metrics_use_the_logged_global_step(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    WandbLogger = _load_logging_utils_module().WandbLogger
    args = SimpleNamespace(
        logger=SimpleNamespace(
            wandb=SimpleNamespace(key="key", org="org", project="project", group="group", run_name="run")
        )
    )

    logger = WandbLogger(args)
    logger.log_eval(7, {"accuracy": 0.5})

    assert (("eval/global_step",), {}) in fake_wandb.metric_calls
    assert (("eval/*",), {"step_metric": "eval/global_step", "step_sync": True}) in fake_wandb.metric_calls
    assert fake_wandb.logged[-1] == {"eval/accuracy": 0.5, "eval/global_step": 7}
