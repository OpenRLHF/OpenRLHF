import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "openrlhf/utils/math_utils.py"
_spec = importlib.util.spec_from_file_location("_math_utils_test", _path)
math_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_utils)


@pytest.mark.parametrize(
    "response",
    [
        r"so the answer is $\boxed{7}$.",
        r"so the answer is $\fbox{7}$.",
        r"Initially $\boxed{4}$, but the final answer is $\fbox{7}$.",
        r"Initially $\fbox{4}$, but the final answer is $\boxed{7}$.",
    ],
)
def test_extract_boxed_answer_accepts_boxed_and_fbox(response):
    answer = math_utils.extract_boxed_answer(response)
    assert answer == "7"
    assert math_utils.grade_answer(answer, "7")


def test_extract_boxed_answer_does_not_fall_back_past_an_unclosed_final_fbox():
    assert math_utils.extract_boxed_answer(r"Initially \boxed{4}, finally \fbox{7") is None
