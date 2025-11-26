import argparse


def add_overlap_comm_arg(parser: argparse.ArgumentParser) -> None:
    """Add the DeepSpeed overlap_comm toggle with a default of True."""
    parser.add_argument(
        "--overlap_comm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DeepSpeed overlap_comm (default: on; use --no-overlap_comm to disable)",
    )
