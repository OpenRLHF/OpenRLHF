import pytest

from openrlhf.utils.math_utils import grade_answer


@pytest.mark.parametrize(
    ("given_answer", "ground_truth"),
    [
        ("[1,2]", "(1,2)"),
        ("(1,2)", "[1,2]"),
        ("(1,2]", "(1,2)"),
        ("[1,2)", "[1,2]"),
    ],
)
def test_grade_answer_rejects_mismatched_tuple_delimiters(given_answer, ground_truth):
    assert not grade_answer(given_answer, ground_truth)


def test_grade_answer_accepts_equivalent_tuple_with_matching_delimiters():
    assert grade_answer("(1.0,2.0)", "(1,2)")
