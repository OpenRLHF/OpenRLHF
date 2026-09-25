import pytest

from openrlhf.utils.math_utils import grade_answer


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("5 minutes", "5 hours"),
        (r"5\text{ minutes}", r"5\text{ hours}"),
        ("10 miles", "10 meters"),
        ("60 feet per second", "60 miles per hour"),
    ],
)
def test_grade_answer_rejects_different_units(given, ground_truth):
    assert not grade_answer(given, ground_truth)


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("5 hours", "5 hours"),
        ("5", "5 hours"),
        ("5 hours", "5"),
        ("5 centimeters", "5 cm"),
        ("3 feet", "3 foot"),
        (r"\frac{1}{2}", "0.5"),
    ],
)
def test_grade_answer_keeps_equivalent_answers(given, ground_truth):
    assert grade_answer(given, ground_truth)
