try:
    from openrlhf.utils.constrained_reward import AllTrueJudge, BaseBinaryJudge, shape_constrained_rewards
except ModuleNotFoundError:
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "openrlhf" / "utils" / "constrained_reward.py"

    spec = importlib.util.spec_from_file_location("_constrained_reward_example_module", module_path)
    constrained_reward_module = importlib.util.module_from_spec(spec)
    sys.modules["_constrained_reward_example_module"] = constrained_reward_module
    spec.loader.exec_module(constrained_reward_module)

    AllTrueJudge = constrained_reward_module.AllTrueJudge
    BaseBinaryJudge = constrained_reward_module.BaseBinaryJudge
    shape_constrained_rewards = constrained_reward_module.shape_constrained_rewards


class NonEmptyJudge(BaseBinaryJudge):
    def judge(self, prompts, completions, gold_completions=None):
        return [int(bool(completion and str(completion).strip())) for completion in completions]


class ContainsGoldJudge(BaseBinaryJudge):
    def judge(self, prompts, completions, gold_completions=None):
        if gold_completions is None:
            return [-1 for _ in completions]

        judgments = []
        for completion, gold in zip(completions, gold_completions):
            completion = str(completion).strip().lower() if completion is not None else ""
            gold = str(gold).strip().lower() if gold is not None else ""
            judgments.append(int(bool(gold) and gold in completion))

        return judgments


judge = AllTrueJudge(
    [
        NonEmptyJudge(),
        ContainsGoldJudge(),
    ]
)


def _completion_from_query(query: str, prompt: str) -> str:
    if query.startswith(prompt):
        return query[len(prompt) :]
    return query


def _format_reward_output(result):
    if len(result.rewards) == 1:
        return {
            "rewards": result.rewards[0],
            "scores": result.scores[0] if result.scores is not None else None,
            "extra_logs": {key: values[0] for key, values in result.extra_logs.items()},
        }

    return {
        "rewards": result.rewards,
        "scores": result.scores,
        "extra_logs": result.extra_logs,
    }


def reward_func(queries, prompts, labels):
    """Example constrained reward function.

    OpenRLHF passes labels from --data.label_key into this function.
    Here labels are used as gold/reference completions.
    """

    completions = [_completion_from_query(query, prompt) for query, prompt in zip(queries, prompts)]
    labels = ["" if label is None else str(label) for label in labels]

    raw_rewards = [
        1.0 if label.strip().lower() and label.strip().lower() in completion.strip().lower() else 0.0
        for completion, label in zip(completions, labels)
    ]

    baseline_rewards = [1.0 for _ in labels]

    result = shape_constrained_rewards(
        prompts=prompts,
        completions=completions,
        rewards=raw_rewards,
        gold_completions=labels,
        baseline_rewards=baseline_rewards,
        judge=judge,
        constraint_penalty=1.0,
        calibrate=True,
    )

    return _format_reward_output(result)