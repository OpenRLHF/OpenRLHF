from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, process_multi_turn_dialogue


def preprocess_data(data, input_template=None, input_key=None, apply_chat_template=None) -> str:
    # custom dataset
    if input_key:
        if apply_chat_template:
            prompt = apply_chat_template(data[input_key], tokenize=False, add_generation_prompt=True)
            input_template = None
        else:
            prompt = data[input_key]
    else:
        # Open-Orca/OpenOrca
        if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + " " + data["question"]
        # Dahoas/full-hh-rlhf
        elif exist_and_not_none(data, "prompt"):
            prompt = data["prompt"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            input_template = None  # do not modified with input template again
        # RLHFlow/prompt-collection-v0.1
        elif exist_and_not_none(data, "context_messages") and isinstance(data["context_messages"], list):
            prompt = data["context_messages"]
            prompt = process_multi_turn_dialogue(prompt, input_template=input_template)
            input_template = None  # do not modified with input template again
        else:
            raise ValueError("Unknown prompts dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
