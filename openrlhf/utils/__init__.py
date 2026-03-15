from typing import TYPE_CHECKING, Any

__all__ = [
    "convert_to_torch_dtype",
    "extract_boxed_answer",
    "get_processor",
    "get_strategy",
    "get_tokenizer",
    "grade_answer",
    "reward_normalization",
]

if TYPE_CHECKING:
    from .math_utils import extract_boxed_answer, grade_answer
    from .processor import get_processor, reward_normalization
    from .utils import convert_to_torch_dtype, get_strategy, get_tokenizer


def __getattr__(name: str) -> Any:
    if name in {"extract_boxed_answer", "grade_answer"}:
        from .math_utils import extract_boxed_answer, grade_answer

        return {"extract_boxed_answer": extract_boxed_answer, "grade_answer": grade_answer}[name]

    if name in {"get_processor", "reward_normalization"}:
        from .processor import get_processor, reward_normalization

        return {"get_processor": get_processor, "reward_normalization": reward_normalization}[name]

    if name in {"convert_to_torch_dtype", "get_strategy", "get_tokenizer"}:
        from .utils import convert_to_torch_dtype, get_strategy, get_tokenizer

        return {
            "convert_to_torch_dtype": convert_to_torch_dtype,
            "get_strategy": get_strategy,
            "get_tokenizer": get_tokenizer,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
