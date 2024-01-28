from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none


def preprocess_data(data, input_template, no_template=False, eos_token="</s>") -> str:
    # Open-Orca/OpenOrca
    if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = data["system_prompt"] + "\n" + data["question"]
    # BelleGroup/train_0.5M_CN
    # LLMs/Alpaca-ShareGPT
    # yahma/alpaca-cleaned
    # QingyiSi/Alpaca-CoT
    elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
        input = " " + data["input"] if exist_and_not_none(data, "input") else ""
        prompt = data["instruction"] + input
    # stanfordnlp/SHP
    elif exist_and_not_none(data, "history"):
        prompt = data["history"]
    # lvwerra/stack-exchange-paired
    elif exist_and_not_none(data, "question") and exist_and_not_none(data, "response_j"):
        prompt = data["question"]
    # lmsys/chatbot_arena_conversations
    elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

        def process_chatbot_arena_conversations(lll):
            result = []
            for l in lll:
                if "user" in l["role"]:
                    result.append(input_template.format(l["content"]))
                else:
                    result.append(l["content"] + eos_token)
            return "\n".join(result)

        prompt = data["conversation_a"][:-1]
        prompt = process_chatbot_arena_conversations(prompt)
        no_template = True
    # openai/webgpt_comparisons
    elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
        prompt = data["question"]["full_text"]
    # Dahoas/full-hh-rlhf
    elif exist_and_not_none(data, "prompt"):
        prompt = data["prompt"]
        # tasksource/oasst1_pairwise_rlhf_reward
        if prompt.startswith("prompter:"):
            prompt = prompt.replace("prompter:", "").replace("assistant:", "\nAssistant: ")
    # JSON files for batch inference
    elif exist_and_not_none(data, "input"):
        prompt = data["input"]
    else:
        raise ValueError("prompt dataset key error")
    if not no_template:
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
        self, dataset, strategy, input_template="Human: {} \nAssistant: ", no_template=False, eos_token="</s>"
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.input_template = input_template
        self.no_template = no_template
        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, no_template, eos_token)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
