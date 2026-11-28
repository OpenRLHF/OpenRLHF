import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "openrlhf/utils/math_utils.py"
_spec = importlib.util.spec_from_file_location("_math_utils_test", _path)
math_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_utils)


@pytest.mark.parametrize("response", [r"so the answer is $\boxed{7}$.", r"so the answer is $\fbox{7}$."])
def test_extract_boxed_answer_accepts_boxed_and_fbox(response):
    answer = math_utils.extract_boxed_answer(response)
    assert answer == "7"
    assert math_utils.grade_answer(answer, "7")


@pytest.mark.parametrize(
    "given,expected",
    [
        (r"\sqrt[3]{2}", r"\sqrt{2}"),
        (r"\sqrt[4]{2}", r"\sqrt[3]{2}"),
        (r"\sqrt[3]{x}", r"\sqrt{x}"),
        (r"\frac{1}{\sqrt[3]{2}}", r"\frac{1}{\sqrt{2}}"),
        (r"\sqrt{1+\sqrt[3]{2}}", r"\sqrt{1+\sqrt{2}}"),
    ],
)
def test_radical_indices_distinguish_different_answers(given, expected):
    assert not math_utils.grade_answer(given, expected)
    assert not math_utils.grade_answer(expected, given)


@pytest.mark.parametrize(
    "given,expected",
    [
        (r"\sqrt[2]{2}", r"\sqrt{2}"),
        (r"\sqrt[3]{2}", r"\sqrt[6]{4}"),
        (r"\sqrt{\sqrt[3]{2}}", r"\sqrt[6]{2}"),
        (r"\sqrt[2+1]{2}", r"\sqrt[3]{2}"),
        (r"2\sqrt[3]{2}", r"\sqrt[3]{16}"),
        (r"\sqrt2", r"\sqrt{2}"),
        (r"\frac{1}{2}", "0.5"),
    ],
)
def test_equivalent_radical_expressions_keep_their_value(given, expected):
    assert math_utils.grade_answer(given, expected)
    assert math_utils.grade_answer(expected, given)


def test_math_reward_preserves_radical_indices():
    import torch

    from examples.python.math_reward_func import reward_func

    result = reward_func(
        queries=[r"Answer: \boxed{\sqrt[3]{2}}", r"Answer: \boxed{\sqrt[6]{4}}"],
        prompts=["Answer: ", "Answer: "],
        labels=[r"\sqrt{2}", r"\sqrt[3]{2}"],
    )

    torch.testing.assert_close(result["rewards"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(result["scores"], result["rewards"])
    torch.testing.assert_close(result["extra_logs"]["math_accuracy"], torch.tensor(0.5))
