import torch


def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """
    # Print input queries for debugging purposes
    print(queries)

    # Generate random rewards as an example
    # In real applications, this should be replaced with actual reward calculation logic
    reward = torch.randint(0, 2, (len(queries),)).float()

    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "extra_logs": {"dummy_scores": reward},  # Additional logging info for wandb
    }
