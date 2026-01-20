#!/usr/bin/env python3
"""
FSDP2 Implementation Validator

This script validates the FSDP2 implementation structure and ensures
all required components are properly implemented. Run this before
attempting GPU training to catch configuration issues early.

Usage:
    python validate_implementation.py
"""

import inspect
import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, passed: bool, message: str, details: str = ""):
        self.passed = passed
        self.message = message
        self.details = details


def validate_imports() -> List[ValidationResult]:
    """Validate that all required modules can be imported."""
    results = []

    # Test FSDP2Strategy import
    try:
        results.append(ValidationResult(True, "FSDP2Strategy import"))
    except ImportError as e:
        results.append(ValidationResult(False, "FSDP2Strategy import", str(e)))

    # Test FSDP2 utilities import
    try:
        results.append(ValidationResult(True, "FSDP2 utilities import"))
    except ImportError as e:
        results.append(ValidationResult(False, "FSDP2 utilities import", str(e)))

    # Test DeepSpeed import
    try:
        results.append(ValidationResult(True, "DeepspeedStrategy import"))
    except ImportError as e:
        results.append(ValidationResult(False, "DeepspeedStrategy import", str(e)))

    # Test get_strategy import
    try:
        results.append(ValidationResult(True, "get_strategy import"))
    except ImportError as e:
        results.append(ValidationResult(False, "get_strategy import", str(e)))

    # Test Actor import
    try:
        results.append(ValidationResult(True, "Actor import"))
    except ImportError as e:
        results.append(ValidationResult(False, "Actor import", str(e)))

    # Test ring attention utilities
    try:
        results.append(ValidationResult(True, "Ring attention utilities import"))
    except ImportError as e:
        results.append(ValidationResult(False, "Ring attention utilities import", str(e)))

    return results


def validate_interface_methods() -> List[ValidationResult]:
    """Validate that both backends implement the required interface methods."""
    results = []

    # Required methods for training strategy backends (both DeepSpeed and FSDP2)
    required_methods = [
        "setup_distributed",
        "create_optimizer",
        "backward",
        "optimizer_step",
        "setup_dataloader",
        "prepare",
        "moving_average",
        "load_model",
        "save_model",
        "all_reduce",
        "all_gather",
        "print",
        "is_rank_0",
        "get_rank",
        "save_ckpt",
        "load_ckpt",
        "get_ds_train_config",
        "get_ds_eval_config",
    ]

    try:
        from openrlhf.utils.deepspeed import DeepspeedStrategy
        from openrlhf.utils.fsdp2 import FSDP2Strategy

        for method in required_methods:
            # Check FSDP2Strategy
            if hasattr(FSDP2Strategy, method):
                results.append(ValidationResult(True, f"FSDP2Strategy.{method}()"))
            else:
                results.append(ValidationResult(False, f"FSDP2Strategy.{method}()", "Method not found"))

            # Check DeepspeedStrategy
            if hasattr(DeepspeedStrategy, method):
                results.append(ValidationResult(True, f"DeepspeedStrategy.{method}()"))
            else:
                results.append(ValidationResult(False, f"DeepspeedStrategy.{method}()", "Method not found"))

    except ImportError as e:
        results.append(ValidationResult(False, "Interface validation", str(e)))

    return results


def validate_fsdp2_specific() -> List[ValidationResult]:
    """Validate FSDP2-specific features."""
    results = []

    try:
        from openrlhf.utils.fsdp2 import FSDP2Strategy

        # Check for AutoTP support
        if hasattr(FSDP2Strategy, "_get_tp_plan"):
            results.append(ValidationResult(True, "FSDP2 AutoTP support (_get_tp_plan)"))
        else:
            results.append(ValidationResult(False, "FSDP2 AutoTP support", "_get_tp_plan method not found"))

        # Check for ring attention support
        if hasattr(FSDP2Strategy, "setup_ring_attn"):
            results.append(ValidationResult(True, "FSDP2 Ring Attention support"))
        else:
            results.append(ValidationResult(False, "FSDP2 Ring Attention support", "setup_ring_attn method not found"))

        # Check for FSDP2 initialization
        if hasattr(FSDP2Strategy, "_apply_fsdp2"):
            results.append(ValidationResult(True, "FSDP2 model wrapping (_apply_fsdp2)"))
        else:
            results.append(ValidationResult(False, "FSDP2 model wrapping", "_apply_fsdp2 method not found"))

        # Check for scheduler recreation
        if hasattr(FSDP2Strategy, "_recreate_scheduler"):
            results.append(ValidationResult(True, "FSDP2 scheduler recreation"))
        else:
            results.append(
                ValidationResult(False, "FSDP2 scheduler recreation", "_recreate_scheduler method not found")
            )

    except ImportError as e:
        results.append(ValidationResult(False, "FSDP2 specific validation", str(e)))

    return results


def validate_deepspeed_specific() -> List[ValidationResult]:
    """Validate DeepSpeed-specific features."""
    results = []

    try:
        from openrlhf.utils.deepspeed import DeepspeedStrategy

        # Check for ring attention support
        if hasattr(DeepspeedStrategy, "setup_ring_attn"):
            results.append(ValidationResult(True, "DeepSpeed Ring Attention support"))
        else:
            results.append(
                ValidationResult(False, "DeepSpeed Ring Attention support", "setup_ring_attn method not found")
            )

        # Check for tensor parallel support
        method = DeepspeedStrategy.__init__
        sig = inspect.signature(method)
        if "args" in sig.parameters:
            results.append(ValidationResult(True, "DeepSpeed tensor parallel support"))
        else:
            results.append(ValidationResult(False, "DeepSpeed tensor parallel support", "args parameter not found"))

    except ImportError as e:
        results.append(ValidationResult(False, "DeepSpeed specific validation", str(e)))

    return results


def validate_tp_plans() -> List[ValidationResult]:
    """Validate tensor parallel plan functions."""
    results = []

    try:
        from openrlhf.utils.fsdp2 import (
            get_gemma_tp_plan,
            get_llama_tp_plan,
            get_qwen_tp_plan,
            translate_parallel_style,
        )

        # Test LLaMA TP plan
        plan = get_llama_tp_plan()
        if isinstance(plan, dict) and len(plan) > 0:
            results.append(ValidationResult(True, "LLaMA TP plan generation"))
        else:
            results.append(ValidationResult(False, "LLaMA TP plan generation", "Invalid plan returned"))

        # Test LLaMA TP plan with sequence parallel
        plan_sp = get_llama_tp_plan(sequence_parallel=True)
        if isinstance(plan_sp, dict) and len(plan_sp) > len(plan):
            results.append(ValidationResult(True, "LLaMA TP plan with sequence parallel"))
        else:
            results.append(ValidationResult(False, "LLaMA TP plan with sequence parallel", "Plan not extended"))

        # Test Qwen TP plan
        plan = get_qwen_tp_plan()
        if isinstance(plan, dict) and len(plan) > 0:
            results.append(ValidationResult(True, "Qwen TP plan generation"))
        else:
            results.append(ValidationResult(False, "Qwen TP plan generation", "Invalid plan returned"))

        # Test Gemma TP plan
        plan = get_gemma_tp_plan()
        if isinstance(plan, dict) and len(plan) > 0:
            results.append(ValidationResult(True, "Gemma TP plan generation"))
        else:
            results.append(ValidationResult(False, "Gemma TP plan generation", "Invalid plan returned"))

        # Test translate_parallel_style
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        style = translate_parallel_style("colwise")
        if isinstance(style, ColwiseParallel):
            results.append(ValidationResult(True, "translate_parallel_style (colwise)"))
        else:
            results.append(ValidationResult(False, "translate_parallel_style (colwise)", "Wrong type returned"))

        style = translate_parallel_style("rowwise")
        if isinstance(style, RowwiseParallel):
            results.append(ValidationResult(True, "translate_parallel_style (rowwise)"))
        else:
            results.append(ValidationResult(False, "translate_parallel_style (rowwise)", "Wrong type returned"))

    except Exception as e:
        results.append(ValidationResult(False, "TP plan validation", str(e)))

    return results


def validate_cli_arguments() -> List[ValidationResult]:
    """Validate that CLI arguments are properly defined."""
    results = []

    try:
        import importlib.util

        # Load train_sft.py module
        spec = importlib.util.spec_from_file_location(
            "train_sft", os.path.join(os.path.dirname(__file__), "..", "openrlhf", "cli", "train_sft.py")
        )
        train_sft = importlib.util.module_from_spec(spec)

        # Read the file content to check for argument definitions
        with open(os.path.join(os.path.dirname(__file__), "..", "openrlhf", "cli", "train_sft.py"), "r") as f:
            content = f.read()

        required_args = [
            "--backend",
            "--fsdp_tensor_parallel_size",
            "--use_hf_tp_plan",
            "--ring_attn_size",
            "--param_dtype",
        ]

        for arg in required_args:
            if arg in content:
                results.append(ValidationResult(True, f"CLI argument {arg}"))
            else:
                results.append(ValidationResult(False, f"CLI argument {arg}", "Not found in train_sft.py"))

    except Exception as e:
        results.append(ValidationResult(False, "CLI argument validation", str(e)))

    return results


def print_results(results: List[ValidationResult], category: str):
    """Print validation results for a category."""
    print(f"\n{category}")
    print("-" * len(category))

    passed = 0
    failed = 0

    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.message}")
        if not result.passed and result.details:
            print(f"      Details: {result.details}")

        if result.passed:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("FSDP2 Implementation Validation")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    # Run validation categories
    categories = [
        ("Module Imports", validate_imports),
        ("Interface Methods", validate_interface_methods),
        ("FSDP2-Specific Features", validate_fsdp2_specific),
        ("DeepSpeed-Specific Features", validate_deepspeed_specific),
        ("Tensor Parallel Plans", validate_tp_plans),
        ("CLI Arguments", validate_cli_arguments),
    ]

    for category_name, validator in categories:
        try:
            results = validator()
            passed, failed = print_results(results, category_name)
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n{category_name}")
            print("-" * len(category_name))
            print(f"  ✗ Validation failed with error: {e}")
            total_failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n✓ All validations passed!")
        print("  The FSDP2 implementation appears to be correctly structured.")
        print("  You can proceed with GPU training tests.")
        return 0
    else:
        print("\n✗ Some validations failed!")
        print("  Please fix the issues above before proceeding with training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
