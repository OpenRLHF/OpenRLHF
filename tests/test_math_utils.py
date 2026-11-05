import pytest

from openrlhf.utils.math_utils import grade_answer


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("\\frac{2}{4}", "\\frac{1}{2}"),
        ("2/4", "1/2"),
        ("3/6", "1/2"),
        ("-2/4", "-1/2"),
        ("6/4", "3/2"),
    ],
)
def test_equivalent_fractions_are_correct(given, ground_truth):
    assert grade_answer(given, ground_truth)
    assert grade_answer(ground_truth, given)


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("2*2", "4"),
        ("1+1", "2"),
        ("8/2", "4"),
        ("\\frac{4}{2}", "2"),
        ("0.5+0.5", "1"),
        ("\\sqrt{4}", "2"),
        ("sqrt(4)", "2"),
    ],
)
def test_expressions_equal_to_an_integer_are_correct(given, ground_truth):
    assert grade_answer(given, ground_truth)
    assert grade_answer(ground_truth, given)


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("1", "2"),
        ("1/2", "0.6"),
        ("3/6", "1/3"),
        ("2*2", "5"),
        ("sqrt(4)", "3"),
        ("(1,2)", "(2,1)"),
        ("x", "y"),
    ],
)
def test_different_answers_are_still_rejected(given, ground_truth):
    assert not grade_answer(given, ground_truth)
    assert not grade_answer(ground_truth, given)


@pytest.mark.parametrize(
    "given,ground_truth",
    [
        ("(1,2)", "(1,2)"),
        ("1,000", "1000"),
        ("\\frac{1}{2}", "0.5"),
        ("\\pi", "pi"),
        ("3.0", "3"),
    ],
)
def test_previously_supported_forms_still_match(given, ground_truth):
    assert grade_answer(given, ground_truth)
