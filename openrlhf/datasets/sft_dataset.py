from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template, no_template=False, eos_token="</s>"):
    # Dahoas/full-hh-rlhf
    # iamketan25/open-assistant-instructions
    if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
        prompt = data["prompt"]
        target = data["chosen"]
        no_template = True  # do not modified with input template again
    # pvduy/sharegpt_alpaca_oa_vicuna_format
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
        prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
        target = data["label"].replace("</s>", "")
    # BelleGroup/train_0.5M_CN
    # LLMs/Alpaca-ShareGPT
    # yahma/alpaca-cleaned
    # QingyiSi/Alpaca-CoT
    elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
        input = " " + data["input"] if exist_and_not_none(data, "input") else ""
        prompt = data["instruction"] + input
        target = data["output"]
    # Open-Orca/OpenOrca
    elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = data["system_prompt"] + "\n" + data["question"]
        target = data["response"]
    # crumb/gpt4all-clean
    # nomic-ai/gpt4all-j-prompt-generations
    elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
        prompt = data["prompt"]
        target = data["response"]
    # EleutherAI/pile [pretrain !!!]
    elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
        assert no_template  # pretrain_mode
        prompt = ""
        target = data["text"]
    # custom datasets
    elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
        prompt = data["input"]
        target = data["output"]
    else:
        raise ValueError("Unknown SFT dataset")

    # input template
    if not no_template:
        prompt = input_template.format(prompt)
    return prompt, target


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, target = preprocess_data(data, input_template, pretrain_mode, eos_token=self.tokenizer.eos_token)

            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not target:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.targets.append(target)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]

        input_token = self.tokenizer(
            prompt + target + " " + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "output": target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos
