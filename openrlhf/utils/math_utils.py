"""Utilities for checking mathematical answer equivalence.

Based on https://github.com/Freder-chen/OpenRLHF-Agent/blob/main/src/openrlhf_agent/agentkit/rewards/result_rewards/hub/math_utils.py
"""

from __future__ import annotations

import math
import re
import signal
import threading
from fractions import Fraction

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


class _SympyTimeout(Exception):
    pass


def _sympy_timeout_handler(signum, frame):
    raise _SympyTimeout()


def _strip_string(string: str) -> str:
    def _fix_fracs(expr: str) -> str:
        substrs = expr.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return expr
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        return new_str

    def _fix_a_slash_b(expr: str) -> str:
        if len(expr.split("/")) != 2:
            return expr
        a = expr.split("/")[0]
        b = expr.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert expr == f"{a}/{b}"
            return "\\frac{" + str(a) + "}{" + str(b) + "}"
        except Exception:
            return expr

    def _remove_right_units(expr: str) -> str:
        if "\\text{ " in expr:
            splits = expr.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        return expr

    def _fix_sqrt(expr: str) -> str:
        if "\\sqrt" not in expr:
            return expr
        splits = expr.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def mathd_normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if match is not None:
            answer = match.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except Exception:
        return False


def _is_int(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = pattern.sub(r"\1\3\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _str_is_int(value: str) -> bool:
    try:
        value = _strip_properly_formatted_commas(value)
        value = float(value)
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _str_to_int(value: str) -> int:
    value = value.replace(",", "")
    value = float(value)
    return int(value)


def _inject_implicit_mixed_number(step: str) -> str:
    pattern = re.compile(r"([0-9]) +([0-9])")
    return pattern.sub(r"\1+\2", step)


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if match is not None:
        expr = match.group("text")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")
    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"
MAX_NUMERIC_BITS = 1_000_000


def _parse_if_safe(expr: str) -> sympy.Expr | None:
    if count_unknown_letters_in_expr(expr) > 2:
        return None
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return None
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return None

    if not any(operator in expr for operator in ("**", "^", "!")):
        return _sympy_parse(expr)

    # Keep SymPy from evaluating numeric powers and factorials during this preflight.
    factorial = sympy.Function("factorial")
    factorial2 = sympy.Function("factorial2")
    try:
        parsed = sympy_parser.parse_expr(
            expr.replace("^", "**"),
            local_dict={"factorial": factorial, "factorial2": factorial2},
            transformations=sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application, sympy_parser.rationalize),
            evaluate=False,
        )
    except (SyntaxError, TypeError, ValueError):
        return None

    evaluated: dict[sympy.Expr, sympy.Expr] = {}

    def bit_size(value: int | Fraction) -> int:
        if isinstance(value, Fraction):
            return max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        return abs(value).bit_length()

    def bounded_value(node: sympy.Expr) -> tuple[int | Fraction | None, bool]:
        if node.is_Integer or node.is_Rational:
            # Inspect components before converting the SymPy value.
            numerator, denominator = node.p, node.q
            if max(abs(numerator).bit_length(), denominator.bit_length()) > MAX_NUMERIC_BITS:
                return None, True
            value = numerator if denominator == 1 else Fraction(numerator, denominator)
            return value, False

        if node.func is sympy.Add:
            value = 0
            symbolic = False
            for child in node.args:
                item, too_large = bounded_value(child)
                if too_large:
                    return None, True
                if item is None:
                    symbolic = True
                    continue
                value += item
                if bit_size(value) > MAX_NUMERIC_BITS:
                    return None, True
            if symbolic:
                return None, False
            evaluated[node] = (
                sympy.Rational(value.numerator, value.denominator)
                if isinstance(value, Fraction)
                else sympy.Integer(value)
            )
            return value, False

        children = [bounded_value(child) for child in node.args]
        if any(too_large for _, too_large in children):
            return None, True
        values = [value for value, _ in children]
        if node.func is sympy.Pow and values[0] is None and not node.args[0].free_symbols and values[1] is not None:
            return None, True
        if node.func is sympy.Pow and values[0] is not None and values[1] is None:
            exponent = node.args[1]
            if exponent.free_symbols and not exponent.is_number:
                numeric_terms = exponent.args if exponent.is_Add else (exponent,)
                numeric_offset = 0
                for term in numeric_terms:
                    if term.is_number:
                        value, too_large = bounded_value(term)
                        if too_large or value is None:
                            return None, True
                        numeric_offset += value
                if abs(numeric_offset) > MAX_NUMERIC_BITS:
                    return None, True
                if any(
                    term.is_Integer and abs(int(term)) > MAX_NUMERIC_BITS
                    for term in sympy.preorder_traversal(exponent)
                ):
                    return None, True
                return None, False
            return None, True
        if node.func in (factorial, factorial2) and values[0] is None:
            return None, not node.args[0].free_symbols
        if not values or any(value is None for value in values):
            return None, node.is_number is True

        # Bound each numeric operation before constructing its exact result.
        if node.func in (factorial, factorial2):
            argument = values[0]
            if isinstance(argument, Fraction) and argument.denominator == 1:
                argument = argument.numerator
            if not isinstance(argument, int):
                return None, False
            if argument < 0:
                return None, False
            if argument > MAX_NUMERIC_BITS:
                return None, True
            if node.func is factorial:
                result_bits = math.lgamma(argument + 1) / math.log(2)
            elif argument % 2 == 0:
                half = argument // 2
                result_bits = half + math.lgamma(half + 1) / math.log(2)
            else:
                half = (argument + 1) // 2
                result_bits = (math.lgamma(argument + 1) - half * math.log(2) - math.lgamma(half + 1)) / math.log(2)
            if result_bits > MAX_NUMERIC_BITS:
                return None, True
            return None, False
        elif node.func is sympy.Mul:
            numerators, denominators = [], []
            sign = 1
            for item in values:
                factor = Fraction(item)
                sign *= -1 if factor.numerator < 0 else 1
                numerators.append(abs(factor.numerator))
                denominators.append(factor.denominator)
            for i, numerator in enumerate(numerators):
                for j, denominator in enumerate(denominators):
                    common = math.gcd(numerator, denominator)
                    numerators[i] //= common
                    denominators[j] //= common
                    numerator //= common
                    if numerator == 1:
                        break
            if sum(n.bit_length() for n in numerators) > MAX_NUMERIC_BITS:
                return None, True
            if sum(d.bit_length() for d in denominators) > MAX_NUMERIC_BITS:
                return None, True
            value = Fraction(sign * math.prod(numerators), math.prod(denominators))
        elif node.func is sympy.Pow:
            base, exponent = values
            if isinstance(exponent, Fraction) and exponent.denominator == 1:
                exponent = exponent.numerator
            if not isinstance(exponent, int):
                return None, False
            base = Fraction(base)
            if exponent < 0 and base == 0:
                return None, True
            if abs(base) != 1 and exponent:
                # Every non-unit rational adds at least one result bit per exponent.
                if abs(exponent) > MAX_NUMERIC_BITS + 2:
                    return None, True
                numerator_bits = math.log2(abs(base.numerator)) if abs(base.numerator) > 1 else 0
                denominator_bits = math.log2(base.denominator) if base.denominator > 1 else 0
                if abs(exponent) * max(numerator_bits, denominator_bits) > MAX_NUMERIC_BITS + 2:
                    return None, True
            value = base**exponent
        else:
            return None, node.is_number is True

        evaluated[node] = (
            sympy.Rational(value.numerator, value.denominator) if isinstance(value, Fraction) else sympy.Integer(value)
        )
        return value, bit_size(value) > MAX_NUMERIC_BITS

    _, too_large = bounded_value(parsed)
    if too_large:
        return None

    factorials = {
        node: (sympy.factorial if node.func is factorial else sympy.factorial2)(node.args[0])
        for node in sympy.preorder_traversal(parsed)
        if node.func in (factorial, factorial2)
    }
    return parsed.xreplace(evaluated).xreplace(factorials)


def should_allow_eval(expr: str) -> bool:
    return _parse_if_safe(expr) is not None


def count_unknown_letters_in_expr(expr: str) -> int:
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = {x for x in expr if x.isalpha()}
    return len(letters_in_expr)


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str, timeout: int = 120) -> bool:
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        sympy_diff = _parse_if_safe(expr)
        if sympy_diff is not None:

            # Use signal.alarm to enforce a hard timeout on sympy.simplify,
            # which can hang indefinitely on certain expressions.
            # signal.alarm only works in the main thread; in worker threads
            # (e.g. Ray actors) we fall back to no timeout.
            use_alarm = threading.current_thread() is threading.main_thread()
            if use_alarm:
                prev_handler = signal.signal(signal.SIGALRM, _sympy_timeout_handler)
                signal.alarm(timeout)
            try:
                simplified = sympy.simplify(sympy_diff)
            except _SympyTimeout:
                return False
            finally:
                if use_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, prev_handler)

            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(value: str) -> str | None:
    left = "\\boxed{"
    try:
        assert value[: len(left)] == left
        assert value[-1] == "}"
        return value[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the answer from inside a LaTeX \\boxed{} command."""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution) if solution is not None else None
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if not given_normalized:
        return False
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        # Bracket/delimiter mismatch on a tuple answer: reject outright. Falling
        # through would let the element-wise check below silently overwrite this.
        return False
    if len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = False
        for gt_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(gt_elem) and _is_frac(given_elem):
                is_correct = gt_elem == given_elem
            elif _str_is_int(gt_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(gt_elem, given_elem)
            if not is_correct:
                break
    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)
    return ground_truth_normalized_mathd == given_answer_normalized_mathd


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Grade if the given answer matches the ground truth.

    Args:
        given_answer: The extracted answer from model (already extracted from \\boxed{})
        ground_truth: The ground truth answer

    Returns:
        True if the answer is correct, False otherwise
    """
    if given_answer is None or ground_truth is None:
        return False
    ground_truth = str(ground_truth)
    given_answer = str(given_answer)
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)
