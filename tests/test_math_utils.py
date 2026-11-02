import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

MATH_UTILS_PATH = Path(__file__).resolve().parents[1] / "openrlhf/utils/math_utils.py"
spec = importlib.util.spec_from_file_location("_math_utils_test", MATH_UTILS_PATH)
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)


def test_grade_answer_rejects_oversized_numeric_expressions():
    answers = [
        "9**9**9",
        "9**+9999999",
        "9**-9999999",
        "9**9_999_999",
        "9**(9**9)",
        "x+9**9**9",
        "(9**9)**(9**9)",
        "9**(9999999/1)",
        "(9/2)**999999",
        "(1/2)**1000000000",
        "2**(x-x+1000000000)",
        "sqrt(2)**1000000000",
        "1000!**100000",
        "(5!)!",
        "9999999!",
        "9999999!!",
        "(9**9)!",
    ]
    script = f"""
import importlib.util

spec = importlib.util.spec_from_file_location("mu", {str(MATH_UTILS_PATH)!r})
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)
for answer in {answers!r}:
    print(mu.grade_answer(answer, r"\\frac{{1}}{{2}}"), flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.splitlines() == ["False"] * len(answers)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("(n+1)!", "(1+n)!"),
        ("(2^3)^x", "8^x"),
        ("(2^3)^9", "8^9"),
        ("2^-10", "1/1024"),
        ("1001!", "1001!+0"),
        ("5!", "5!+0"),
        ("7!!", "7!!+0"),
        ("x**9999999", "x**9999999+0"),
        ("(3/2)^2", "(3/2)^2+0"),
    ],
)
def test_grade_answer_preserves_equivalent_expressions(given, expected):
    assert math_utils.grade_answer(given, expected)


def test_should_allow_large_but_safe_numeric_and_symbolic_powers():
    assert math_utils.should_allow_eval("3**600000")
    assert math_utils.should_allow_eval("(3/2)**100")
    assert math_utils.should_allow_eval("x**" + "9" * 1_100)


def test_should_reject_power_with_unrepresentably_large_exponent():
    assert not math_utils.should_allow_eval("2**(2**100000)")
    assert not math_utils.should_allow_eval("2**(x-x+1000000000)")
    assert not math_utils.should_allow_eval("2**(x+1000000000)")
    assert not math_utils.should_allow_eval("2**(x+1000*1000*1000)")
    assert not math_utils.should_allow_eval("sqrt(2)**1000000000")
    assert not math_utils.should_allow_eval("1000!**100000")
    assert not math_utils.should_allow_eval("(5!)!")
    assert math_utils.should_allow_eval("8**x")
    assert math_utils.should_allow_eval("2**(x+1)")
    assert math_utils.should_allow_eval("2**(x+x)")
    assert math_utils.should_allow_eval("2**(2*x)")


def test_should_allow_equivalent_compound_symbolic_powers():
    assert math_utils.are_equal_under_sympy("2**(x+1)", "2*2**x")
    assert math_utils.are_equal_under_sympy("2**(x+x)", "4**x")
    assert math_utils.are_equal_under_sympy("2**(2*x)", "4**x")


def test_guarded_grading_reuses_preflight_parse(monkeypatch):
    parse_expr = math_utils.sympy_parser.parse_expr
    parse_count = 0

    def count_parse(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        return parse_expr(*args, **kwargs)

    monkeypatch.setattr(math_utils.sympy_parser, "parse_expr", count_parse)

    assert math_utils.are_equal_under_sympy("2**10", "1024")
    assert parse_count == 1


def test_should_stop_evaluating_addition_after_size_limit():
    expression = "+".join(["2**999999"] * 100)

    assert not math_utils.should_allow_eval(expression)


def test_should_scan_numeric_addends_after_symbolic_terms():
    assert not math_utils.should_allow_eval("x+9**9**9")


def test_should_check_factorial_size_without_materializing_result(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("preflight should not construct factorial results")

    monkeypatch.setattr(math_utils.math, "factorial", fail_if_called)

    assert math_utils.should_allow_eval("1001!")


def test_should_allow_products_with_cancelling_rational_factors():
    assert math_utils.are_equal_under_sympy("1", "(2**500000)*(1/2**500000)")


def test_should_bound_products_before_multiplying(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("oversized product should be rejected before multiplication")

    monkeypatch.setattr(math_utils.math, "prod", fail_if_called)

    assert not math_utils.should_allow_eval("(2**500000)*(2**500000)")


def test_should_reject_oversized_rational_before_conversion(monkeypatch):
    oversized = math_utils.sympy.Rational(1 << (math_utils.MAX_NUMERIC_BITS + 1), 3)
    monkeypatch.setattr(math_utils.sympy_parser, "parse_expr", lambda *args, **kwargs: oversized)

    assert not math_utils.should_allow_eval("x**2")


def test_should_check_integer_size_before_conversion(monkeypatch):
    class OversizedInteger:
        is_Integer = True
        is_Rational = False
        p = 1 << (math_utils.MAX_NUMERIC_BITS + 1)
        q = 1
        args = ()

        def __int__(self):
            raise AssertionError("oversized integer should not be converted")

    monkeypatch.setattr(math_utils.sympy_parser, "parse_expr", lambda *args, **kwargs: OversizedInteger())

    assert not math_utils.should_allow_eval("x**2")
