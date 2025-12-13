from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def normalize_interval_config(args):
    """Normalize eval/save interval configs to simplify callers."""
    if args.eval_steps == -1:
        args.eval_steps = float("inf")
    if args.save_steps == -1:
        args.save_steps = float("inf")


def build_prompt_dataloader(args, strategy, tokenizer, split, *, for_eval=False):
    """Create dataloader for the given split; compute max_steps only for training."""
    max_count = args.max_samples if not for_eval else sys.maxsize
    data = blending_datasets(
        args.eval_dataset if for_eval else args.prompt_data,
        args.prompt_data_probs if not for_eval else None,
        strategy,
        args.seed,
        max_count=max_count,
        dataset_split=split,
    )

    data = data.select(range(min(max_count, len(data))))
    dataset = PromptDataset(data, tokenizer, strategy, input_template=args.input_template)
    # train -> shuffle/drop_last True; eval -> shuffle True keeps prior behavior
    dataloader = strategy.setup_dataloader(dataset, 1, True, not for_eval)

    max_steps = None
    if not for_eval:
        max_steps = (
            len(dataset) * args.n_samples_per_prompt // args.train_batch_size * args.num_episodes * args.max_epochs
        )

    return dataloader, max_steps
