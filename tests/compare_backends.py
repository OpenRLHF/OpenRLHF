#!/usr/bin/env python3
"""
Backend Comparison Script

This script compares loss values from DeepSpeed and FSDP2 training logs
to verify precision alignment between the two backends.

Usage:
    python compare_backends.py --deepspeed_log logs/deepspeed_test.log --fsdp2_log logs/fsdp2_test.log

The script will:
1. Parse loss values from both log files
2. Calculate statistics (mean, std, min, max)
3. Compare loss curves and report alignment metrics
4. Determine if the backends are aligned (<5% difference)
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LossStats:
    """Statistics for loss values."""

    mean: float
    std: float
    min: float
    max: float
    values: List[float]

    @classmethod
    def from_values(cls, values: List[float]) -> "LossStats":
        """Create LossStats from a list of values."""
        if not values:
            return cls(mean=0, std=0, min=0, max=0, values=[])

        import statistics

        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return cls(mean=mean, std=std, min=min(values), max=max(values), values=values)


def extract_loss_values(log_file: str) -> List[float]:
    """Extract gpt_loss values from a training log file.

    Args:
        log_file: Path to the log file

    Returns:
        List of loss values extracted from the log
    """
    loss_values = []

    # Pattern to match gpt_loss values in the log
    # Examples:
    # - {'gpt_loss': 1.1234, 'lr': 5e-06}
    # - gpt_loss: 1.1234
    patterns = [
        r"'gpt_loss':\s*([0-9]+\.?[0-9]*)",
        r"gpt_loss:\s*([0-9]+\.?[0-9]*)",
        r'"gpt_loss":\s*([0-9]+\.?[0-9]*)',
        r"loss_mean.*?:\s*([0-9]+\.?[0-9]*)",
    ]

    try:
        with open(log_file, "r") as f:
            for line in f:
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        try:
                            loss = float(match)
                            if loss > 0 and loss < 100:  # Sanity check
                                loss_values.append(loss)
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return []

    return loss_values


def compare_loss_curves(
    ds_losses: List[float], fsdp2_losses: List[float], threshold: float = 0.05
) -> Tuple[bool, float, str]:
    """Compare two loss curves for alignment.

    Args:
        ds_losses: DeepSpeed loss values
        fsdp2_losses: FSDP2 loss values
        threshold: Maximum allowed relative difference (default 5%)

    Returns:
        Tuple of (aligned, max_diff, message)
    """
    if not ds_losses or not fsdp2_losses:
        return False, 1.0, "One or both log files have no loss values"

    # Compare initial losses
    initial_diff = abs(ds_losses[0] - fsdp2_losses[0]) / max(ds_losses[0], fsdp2_losses[0])

    # Compare mean losses
    ds_mean = sum(ds_losses) / len(ds_losses)
    fsdp2_mean = sum(fsdp2_losses) / len(fsdp2_losses)
    mean_diff = abs(ds_mean - fsdp2_mean) / max(ds_mean, fsdp2_mean)

    # Compare final losses
    final_diff = abs(ds_losses[-1] - fsdp2_losses[-1]) / max(ds_losses[-1], fsdp2_losses[-1])

    # Maximum difference
    max_diff = max(initial_diff, mean_diff, final_diff)

    aligned = max_diff < threshold

    message = f"""
Loss Comparison Results:
------------------------
Initial Loss Difference: {initial_diff:.4f} ({initial_diff*100:.2f}%)
Mean Loss Difference:    {mean_diff:.4f} ({mean_diff*100:.2f}%)
Final Loss Difference:   {final_diff:.4f} ({final_diff*100:.2f}%)
Maximum Difference:      {max_diff:.4f} ({max_diff*100:.2f}%)

Threshold: {threshold*100:.1f}%
Status: {'ALIGNED' if aligned else 'NOT ALIGNED'}
"""

    return aligned, max_diff, message


def print_loss_comparison(ds_stats: LossStats, fsdp2_stats: LossStats, aligned: bool, message: str) -> None:
    """Print formatted loss comparison results."""

    print("\n" + "=" * 60)
    print("BACKEND COMPARISON RESULTS")
    print("=" * 60)

    print("\nDeepSpeed Statistics:")
    print("-" * 30)
    print(f"  Mean Loss:    {ds_stats.mean:.6f}")
    print(f"  Std Loss:     {ds_stats.std:.6f}")
    print(f"  Min Loss:     {ds_stats.min:.6f}")
    print(f"  Max Loss:     {ds_stats.max:.6f}")
    print(f"  Num Steps:    {len(ds_stats.values)}")

    print("\nFSDP2 Statistics:")
    print("-" * 30)
    print(f"  Mean Loss:    {fsdp2_stats.mean:.6f}")
    print(f"  Std Loss:     {fsdp2_stats.std:.6f}")
    print(f"  Min Loss:     {fsdp2_stats.min:.6f}")
    print(f"  Max Loss:     {fsdp2_stats.max:.6f}")
    print(f"  Num Steps:    {len(fsdp2_stats.values)}")

    print(message)

    if aligned:
        print("\n" + "=" * 60)
        print("SUCCESS: Backends are aligned!")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("WARNING: Backends are NOT aligned within threshold!")
        print("=" * 60 + "\n")


def plot_losses(ds_losses: List[float], fsdp2_losses: List[float], output_file: Optional[str] = None) -> None:
    """Plot loss curves for visual comparison.

    Args:
        ds_losses: DeepSpeed loss values
        fsdp2_losses: FSDP2 loss values
        output_file: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(ds_losses, label="DeepSpeed", alpha=0.8)
        plt.plot(fsdp2_losses, label="FSDP2", alpha=0.8)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss Curves Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Plot difference if lengths match
        min_len = min(len(ds_losses), len(fsdp2_losses))
        if min_len > 0:
            diff = [abs(ds_losses[i] - fsdp2_losses[i]) for i in range(min_len)]
            plt.plot(diff, color="red", alpha=0.8)
            plt.xlabel("Step")
            plt.ylabel("Absolute Difference")
            plt.title("Loss Difference (|DeepSpeed - FSDP2|)")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150)
            print(f"Plot saved to: {output_file}")
        else:
            plt.show()

    except ImportError:
        print("Note: matplotlib not available for plotting")


def main():
    parser = argparse.ArgumentParser(description="Compare loss values between DeepSpeed and FSDP2 backends")
    parser.add_argument(
        "--deepspeed_log", type=str, default="logs/deepspeed_test.log", help="Path to DeepSpeed log file"
    )
    parser.add_argument("--fsdp2_log", type=str, default="logs/fsdp2_test.log", help="Path to FSDP2 log file")
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Maximum allowed relative difference (default: 0.05 = 5%%)"
    )
    parser.add_argument("--plot", type=str, default=None, help="Optional path to save comparison plot")
    parser.add_argument("--verbose", action="store_true", help="Print all loss values")

    args = parser.parse_args()

    # Extract loss values
    print(f"\nExtracting loss values from logs...")
    print(f"DeepSpeed log: {args.deepspeed_log}")
    print(f"FSDP2 log: {args.fsdp2_log}")

    ds_losses = extract_loss_values(args.deepspeed_log)
    fsdp2_losses = extract_loss_values(args.fsdp2_log)

    if not ds_losses:
        print(f"Error: No loss values found in DeepSpeed log")
        sys.exit(1)

    if not fsdp2_losses:
        print(f"Error: No loss values found in FSDP2 log")
        sys.exit(1)

    print(f"Found {len(ds_losses)} DeepSpeed loss values")
    print(f"Found {len(fsdp2_losses)} FSDP2 loss values")

    if args.verbose:
        print("\nDeepSpeed losses (first 10):", ds_losses[:10])
        print("FSDP2 losses (first 10):", fsdp2_losses[:10])

    # Calculate statistics
    ds_stats = LossStats.from_values(ds_losses)
    fsdp2_stats = LossStats.from_values(fsdp2_losses)

    # Compare backends
    aligned, max_diff, message = compare_loss_curves(ds_losses, fsdp2_losses, threshold=args.threshold)

    # Print results
    print_loss_comparison(ds_stats, fsdp2_stats, aligned, message)

    # Plot if requested
    if args.plot:
        plot_losses(ds_losses, fsdp2_losses, args.plot)

    # Exit with appropriate code
    sys.exit(0 if aligned else 1)


if __name__ == "__main__":
    main()
