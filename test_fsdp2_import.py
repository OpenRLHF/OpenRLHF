#!/usr/bin/env python
"""Quick test to verify FSDP2 implementation imports and basic structure."""

import sys

def test_imports():
    """Test that all FSDP2 modules can be imported."""
    print("Testing FSDP2 imports...")
    
    try:
        from openrlhf.utils.fsdp2 import FSDP2Strategy
        print("✓ FSDP2Strategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FSDP2Strategy: {e}")
        return False
    
    try:
        from openrlhf.utils.fsdp2 import (
            get_optimizer_grouped_parameters,
            get_grad_norm,
            clip_grad_by_total_norm_,
            to_local_if_dtensor,
            get_llama_tp_plan,
        )
        print("✓ FSDP2 utility functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import FSDP2 utilities: {e}")
        return False
    
    try:
        from openrlhf.utils import get_strategy
        print("✓ get_strategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import get_strategy: {e}")
        return False
    
    try:
        from openrlhf.models import Actor
        print("✓ Actor imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Actor: {e}")
        return False
    
    return True


def test_strategy_creation():
    """Test that FSDP2Strategy can be created with dummy args."""
    print("\nTesting FSDP2Strategy creation...")
    
    from openrlhf.utils.fsdp2 import FSDP2Strategy
    from argparse import Namespace
    
    args = Namespace(
        param_dtype="bf16",
        adam_offload=False,
        gradient_checkpointing=False,
        fsdp_tensor_parallel_size=1,
        ring_attn_size=1,
        use_dynamic_batch=False,
        local_rank=-1,
        backend="fsdp2",
    )
    
    try:
        strategy = FSDP2Strategy(
            seed=42,
            full_determinism=False,
            max_norm=1.0,
            micro_train_batch_size=2,
            train_batch_size=128,
            zero_stage=2,
            args=args,
        )
        print("✓ FSDP2Strategy created successfully")
        
        # Check attributes
        assert strategy.param_dtype == "bf16", "param_dtype mismatch"
        assert strategy.max_norm == 1.0, "max_norm mismatch"
        assert strategy.train_batch_size == 128, "train_batch_size mismatch"
        print("✓ Strategy attributes verified")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create FSDP2Strategy: {e}")
        return False


def test_get_strategy_backend():
    """Test that get_strategy returns correct backend based on args."""
    print("\nTesting get_strategy backend selection...")
    
    from openrlhf.utils import get_strategy
    from argparse import Namespace
    
    # Test FSDP2 backend
    args_fsdp2 = Namespace(
        backend="fsdp2",
        param_dtype="bf16",
        adam_offload=False,
        gradient_checkpointing=False,
        fsdp_tensor_parallel_size=1,
        ring_attn_size=1,
        use_dynamic_batch=False,
        local_rank=-1,
        seed=42,
        full_determinism=False,
        max_norm=1.0,
        micro_train_batch_size=2,
        train_batch_size=128,
        zero_stage=2,
    )
    
    try:
        strategy = get_strategy(args_fsdp2)
        from openrlhf.utils.fsdp2 import FSDP2Strategy
        assert isinstance(strategy, FSDP2Strategy), "Expected FSDP2Strategy"
        print("✓ FSDP2 backend selected correctly")
    except Exception as e:
        print(f"✗ Failed to get FSDP2 strategy: {e}")
        return False
    
    # Test DeepSpeed backend
    args_deepspeed = Namespace(
        backend="deepspeed",
        param_dtype="bf16",
        adam_offload=False,
        gradient_checkpointing=False,
        ds_tensor_parallel_size=1,
        ring_attn_size=1,
        use_dynamic_batch=False,
        local_rank=-1,
        seed=42,
        full_determinism=False,
        max_norm=1.0,
        micro_train_batch_size=2,
        train_batch_size=128,
        zero_stage=2,
        zpg=1,
        grad_accum_dtype=None,
        overlap_comm=False,
        use_ds_universal_ckpt=False,
        deepcompile=False,
    )
    
    try:
        strategy = get_strategy(args_deepspeed)
        from openrlhf.utils.deepspeed import DeepspeedStrategy
        assert isinstance(strategy, DeepspeedStrategy), "Expected DeepspeedStrategy"
        print("✓ DeepSpeed backend selected correctly")
    except Exception as e:
        print(f"✗ Failed to get DeepSpeed strategy: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("FSDP2 Implementation Tests")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_strategy_creation():
        all_passed = False
    
    if not test_get_strategy_backend():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
