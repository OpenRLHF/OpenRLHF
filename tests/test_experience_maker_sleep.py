from types import SimpleNamespace

from openrlhf.trainer.ppo_utils import experience_maker as experience_maker_module
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker


class FakeGroup:
    def __init__(self):
        self.calls = []

    def async_run_method(self, method_name, *args, **kwargs):
        self.calls.append(("method", method_name))
        return [f"{method_name}-ref"]

    def async_run_method_batch(self, method_name, **kwargs):
        self.calls.append(("batch", method_name))
        return [f"{method_name}-ref"]


def _make_maker(cpu_offload):
    maker = object.__new__(RemoteExperienceMaker)
    maker.strategy = SimpleNamespace(cpu_offload=cpu_offload)
    maker.args = SimpleNamespace(fsdp=SimpleNamespace(enable_sleep=True))
    return maker


def _call_names(group):
    return [name for _, name in group.calls]


def test_cpu_offload_sleep_uses_phase_level_lp_inference(monkeypatch):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda refs: refs)
    maker = _make_maker(cpu_offload=True)
    group = FakeGroup()
    other_group = FakeGroup()

    maker._prepare_lp_inference_phase([group, group, other_group])
    maker._dispatch_forward(group, sync_condition=True, sequences=[object()])
    maker._finish_lp_inference_phase([group, other_group])

    assert _call_names(group) == [
        "prepare_for_lp_inference",
        "forward",
        "empty_cache",
        "offload_after_refit",
    ]
    assert _call_names(other_group) == ["prepare_for_lp_inference", "offload_after_refit"]


def test_non_cpu_offload_sleep_keeps_per_group_bracket(monkeypatch):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda refs: refs)
    maker = _make_maker(cpu_offload=False)
    group = FakeGroup()

    maker._dispatch_forward(group, sync_condition=False, sequences=[object()])

    assert _call_names(group) == [
        "prepare_for_lp_inference",
        "forward",
        "empty_cache",
        "offload_after_refit",
    ]
