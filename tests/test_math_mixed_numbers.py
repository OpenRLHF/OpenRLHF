import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "openrlhf/utils/math_utils.py"
_spec = importlib.util.spec_from_file_location("_math_utils_test", _path)
math_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_utils)


@pytest.mark.parametrize(
    "answer,expected,is_correct",
    [
        (r"-2\frac{1}{2}", "-2.5", True),
        ("-2 1/2", "-2.5", True),
        (r"-12\frac{3}{4}", "-12.75", True),
        (r"-0\frac{1}{2}", "-0.5", True),
        (r"-2\frac{1}{2}", "-1.5", False),
        ("-2 1/2", "-1.5", False),
        (r"-12\frac{3}{4}", "-11.25", False),
        (r"-0\frac{1}{2}", "0.5", False),
        ("-2 1/2", "-21/2", False),
        ("2 1/2", "21/2", False),
        ("2 1/2", "2.5", True),
        ("12 3/4", "12.75", True),
        (r"2\frac{1}{2}", "2.5", True),
        ("-2+1/2", "-1.5", True),
        ("-5/2", "-2.5", True),
        ("-2.5", "-2.5", True),
    ],
)
def test_grade_answer_preserves_mixed_number_sign_and_spacing(answer, expected, is_correct):
    assert math_utils.grade_answer(answer, expected) is is_correct
