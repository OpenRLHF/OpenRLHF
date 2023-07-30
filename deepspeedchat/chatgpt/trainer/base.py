from abc import ABC, abstractmethod

class Trainer(ABC):
    """
        Base class for rlhf trainers.

    Args:
        strategy (Strategy):the strategy to use for training
        experience_maker (ExperienceMaker): the experience maker to use for produce experience to fullfill replay buffer
        replay_buffer (ReplayBuffer): the replay buffer to use for training
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenizer (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        data_loader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, prompts_dataset, pretrain_dataset, num_episodes: int = 1, update_timesteps: int = 32) -> None:
        pass