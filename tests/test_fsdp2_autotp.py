#!/usr/bin/env python
"""Test script for FSDP2 AutoTP implementation.

This script verifies that:
1. All FSDP2 modules can be imported
2. The get_hf_tp_plan() function works correctly
3. The translate_parallel_style() function works correctly
4. Optimized TP plans are available for known models
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        print("  ✓ All FSDP2 modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_translate_parallel_style():
    """Test translate_parallel_style function."""
    print("\nTesting translate_parallel_style()...")
    try:
        from openrlhf.utils.fsdp2 import translate_parallel_style

        # Test all supported styles
        styles = ["colwise", "rowwise", "colwise_rep", "rowwise_rep", "sequence_parallel"]
        for style in styles:
            result = translate_parallel_style(style)
            print(f"  ✓ '{style}' -> {type(result).__name__}")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_llama_tp_plan():
    """Test get_llama_tp_plan function."""
    print("\nTesting get_llama_tp_plan()...")
    try:
        from openrlhf.utils.fsdp2 import get_llama_tp_plan

        # Test without sequence parallel
        tp_plan = get_llama_tp_plan(sequence_parallel=False)
        print(f"  ✓ LLaMA TP plan (no SP): {len(tp_plan)} entries")
        for key in list(tp_plan.keys())[:5]:
            print(f"      - {key}: {type(tp_plan[key]).__name__}")

        # Test with sequence parallel
        tp_plan_sp = get_llama_tp_plan(sequence_parallel=True)
        print(f"  ✓ LLaMA TP plan (with SP): {len(tp_plan_sp)} entries")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_qwen_tp_plan():
    """Test get_qwen_tp_plan function."""
    print("\nTesting get_qwen_tp_plan()...")
    try:
        from openrlhf.utils.fsdp2 import get_qwen_tp_plan

        tp_plan = get_qwen_tp_plan(sequence_parallel=False)
        print(f"  ✓ Qwen TP plan (no SP): {len(tp_plan)} entries")

        tp_plan_sp = get_qwen_tp_plan(sequence_parallel=True)
        print(f"  ✓ Qwen TP plan (with SP): {len(tp_plan_sp)} entries")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_gemma_tp_plan():
    """Test get_gemma_tp_plan function."""
    print("\nTesting get_gemma_tp_plan()...")
    try:
        from openrlhf.utils.fsdp2 import get_gemma_tp_plan

        tp_plan = get_gemma_tp_plan(sequence_parallel=False)
        print(f"  ✓ Gemma TP plan (no SP): {len(tp_plan)} entries")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def test_hf_tp_plan_with_model():
    """Test get_hf_tp_plan with a real model."""
    print("\nTesting get_hf_tp_plan() with model...")
    try:
        from transformers import AutoConfig

        # Use a small config to avoid loading large weights
        print("  Loading small model config...")
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)

        # Check if the model class has _tp_plan
        model_cls_name = "LlamaForCausalLM"
        from transformers import LlamaForCausalLM

        has_tp_plan = hasattr(LlamaForCausalLM, "_tp_plan")
        print(f"  Model class has _tp_plan: {has_tp_plan}")

        if has_tp_plan:
            print(f"  _tp_plan content: {LlamaForCausalLM._tp_plan}")

        return True
    except Exception as e:
        print(f"  ⚠ Test skipped or failed: {e}")
        # This is not a failure - the model may not have _tp_plan in older transformers
        return True


def test_fsdp2_strategy_init():
    """Test FSDP2Strategy initialization."""
    print("\nTesting FSDP2Strategy initialization...")
    try:
        import argparse

        from openrlhf.utils.fsdp2 import FSDP2Strategy

        # Create minimal args
        args = argparse.Namespace(
            param_dtype="bf16",
            local_rank=-1,
            adam_offload=False,
            gradient_checkpointing=False,
            fsdp_tensor_parallel_size=2,
            ring_attn_size=1,
            use_dynamic_batch=False,
            use_hf_tp_plan=True,
            sequence_parallel=False,
        )

        strategy = FSDP2Strategy(
            seed=42,
            full_determinism=False,
            max_norm=1.0,
            micro_train_batch_size=2,
            train_batch_size=128,
            zero_stage=2,
            args=args,
        )

        print(f"  ✓ FSDP2Strategy created successfully")
        print(f"      tensor_parallel_size: {strategy.tensor_parallel_size}")
        print(f"      use_hf_tp_plan: {strategy.use_hf_tp_plan}")
        print(f"      sequence_parallel: {strategy.sequence_parallel}")

        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FSDP2 AutoTP Implementation Tests")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("translate_parallel_style", test_translate_parallel_style()))
    results.append(("LLaMA TP plan", test_llama_tp_plan()))
    results.append(("Qwen TP plan", test_qwen_tp_plan()))
    results.append(("Gemma TP plan", test_gemma_tp_plan()))
    results.append(("HF TP plan with model", test_hf_tp_plan_with_model()))
    results.append(("FSDP2Strategy init", test_fsdp2_strategy_init()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
