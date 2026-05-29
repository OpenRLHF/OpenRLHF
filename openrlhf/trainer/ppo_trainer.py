import ray

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.utils.utils import get_tokenizer


def prepare_datasets(strategy, tokenizer):
    args = strategy.args

    # prepare datasets
    train_data = blending_datasets(
        args.data.prompt_dataset,
        args.data.prompt_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.prompt_split,
    )

    # Create train dataset
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    prompts_dataset = PromptDataset(train_data, tokenizer, strategy, input_template=args.data.input_template)

    max_steps = (
        len(prompts_dataset)
        * args.rollout.n_samples_per_prompt
        // args.train.batch_size
        * args.train.num_episodes
        * args.train.max_epochs
    )
    return max_steps


@ray.remote
class PPOTrainer:
    def __init__(
        self,
        pretrain: str,
        strategy,
        actor_model_group,
        critic_model_group,
        reward_model_group,
        reference_model_group,
        vllm_engines,
    ) -> None:
        tokenizer = get_tokenizer(
            pretrain, None, "left", strategy, use_fast=not strategy.args.data.disable_fast_tokenizer
        )
        self.max_steps = prepare_datasets(strategy, tokenizer)

    def get_max_steps(self):
        return self.max_steps
