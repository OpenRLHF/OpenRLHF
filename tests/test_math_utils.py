import pytest

from openrlhf.utils.math_utils import grade_answer


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("5 minutes", "5 hours"),
        ("5 weeks", "5 hours"),
        (r"5\text{ minutes}", r"5\text{ hours}"),
        ("10 miles", "10 meters"),
        ("10 centimeters", "10 meters"),
        ("100 inches", "100 feet"),
        ("5 years", "5 days"),
        ("60 feet per second", "60 miles per hour"),
        ("60 miles per year", "60 miles per hour"),
        ("60 hours per mile", "60 miles per hour"),
        ("3 cm, 4 cm, 5 meter", "3 cm, 4 meter, 5 cm"),
        ("5 per year", "$5 per day"),
        (r"12\text{ inches}^2", r"12\text{ cm}^2"),
        (r"12 \text{ m}^2", r"12 \text{ cm}^2"),
        ("minutes", "hours"),
        ("meter", "cm"),
        ("", "hours"),
        ("5minutes", "5 hours"),
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
        ("5 by 3 cm", "5 cm by 3 cm"),
        ("3 by 4 feet", "3 feet by 4 feet"),
        ("(3, 4) cm", "(3 cm, 4 cm)"),
        (r"3 \times 4\text{ cm}", r"3\text{ cm} \times 4\text{ cm}"),
        ("2, 3 hours", "2 hours, 3 hours"),
        (r"(m-1)(m+1)", r"m^2 - 1"),
        (r"\frac{1}{2}", "0.5"),
    ],
)
def test_grade_answer_keeps_equivalent_answers(given, ground_truth):
    assert grade_answer(given, ground_truth)
