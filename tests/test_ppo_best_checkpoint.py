from types import SimpleNamespace

class _DummyTrainer:
    def __init__(self, args):
        self.args = args
        self.best_eval_metric = None
        self.saved = []

    def _save_checkpoint(self, tag, client_states=None):
        self.saved.append((tag, client_states or {}))

    def _is_better_metric(self, metric_value: float) -> bool:
        if self.best_eval_metric is None:
            return True
        if getattr(self.args, "greater_is_better", True):
            return metric_value > self.best_eval_metric
        return metric_value < self.best_eval_metric

    def maybe_save_best_checkpoint(self, global_step, eval_logs, client_states=None):
        if not getattr(self.args, "save_best_only", False):
            return False

        metric_name = getattr(self.args, "metric_for_best_model", None)
        if not metric_name:
            return False

        metric_value = eval_logs.get(metric_name)
        if metric_value is None:
            return False

        metric_value = float(metric_value)
        if not self._is_better_metric(metric_value):
            return False

        self.best_eval_metric = metric_value
        best_tag = f"best_global_step{global_step}"
        best_client_states = dict(client_states or {})
        best_client_states.update(
            {
                "best_eval_metric": self.best_eval_metric,
                "best_model_global_step": global_step,
                "best_model_metric_name": metric_name,
            }
        )
        self._save_checkpoint(best_tag, best_client_states)
        return True


def test_maybe_save_best_checkpoint_higher_is_better():
    args = SimpleNamespace(save_best_only=True, metric_for_best_model="eval_math_pass4", greater_is_better=True)
    trainer = _DummyTrainer(args)

    assert trainer.maybe_save_best_checkpoint(10, {"eval_math_pass4": 0.6}, {"foo": "bar"}) is True
    assert trainer.best_eval_metric == 0.6
    assert trainer.saved[0][0] == "best_global_step10"

    assert trainer.maybe_save_best_checkpoint(20, {"eval_math_pass4": 0.4}, {}) is False
    assert len(trainer.saved) == 1

    assert trainer.maybe_save_best_checkpoint(30, {"eval_math_pass4": 0.8}, {}) is True
    assert trainer.best_eval_metric == 0.8
    assert len(trainer.saved) == 2


def test_maybe_save_best_checkpoint_lower_is_better():
    args = SimpleNamespace(save_best_only=True, metric_for_best_model="eval_loss", greater_is_better=False)
    trainer = _DummyTrainer(args)

    assert trainer.maybe_save_best_checkpoint(10, {"eval_loss": 1.2}, {}) is True
    assert trainer.maybe_save_best_checkpoint(20, {"eval_loss": 1.5}, {}) is False
    assert trainer.maybe_save_best_checkpoint(30, {"eval_loss": 0.9}, {}) is True
    assert trainer.best_eval_metric == 0.9
