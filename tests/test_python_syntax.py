import warnings
from pathlib import Path


def test_math_utils_compiles_with_syntax_warnings_as_errors():
    root = Path(__file__).resolve().parents[1]
    path = root / "openrlhf" / "utils" / "math_utils.py"

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        compile(path.read_text(), str(path), "exec")
