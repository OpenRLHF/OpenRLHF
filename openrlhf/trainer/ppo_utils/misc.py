from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def normalize_interval_config(args):
    """Normalize eval/save interval configs to simplify callers."""
    if args.eval_steps == -1:
        args.eval_steps = float("inf")  # do not evaluate
    if args.save_steps == -1:
        args.save_steps = float("inf")  # do not save ckpt


def build_prompt_dataloader(args, strategy, tokenizer, split):
    # prepare datasets
    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        dataset_split=split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.input_template)
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, 1, True, True)

    max_steps = (
        len(prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.num_episodes * args.max_epochs
    )

    return prompts_dataloader, max_steps


def build_eval_dataloader(args, strategy, tokenizer, split):
    # Create eval dataset if eval data exists
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,  # No probability sampling for eval datasets
            strategy,
            dataset_split=split,
        )
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(eval_data, tokenizer, strategy, input_template=args.input_template)
        eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
    else:
        eval_dataloader = None

    return eval_dataloader
