import copy

from torch.utils.data import Dataset
from tqdm import tqdm


def _str_to_content_list(text: str):
    """Convert a string with ``<image>`` tags to a VLM content list.

    E.g. ``"<image>Find x."`` → ``[{"type": "image"}, {"type": "text", "text": "Find x."}]``
    Returns the original string unchanged when no ``<image>`` tags are present.
    """
    if "<image>" not in text:
        return text
    parts = text.split("<image>")
    content = []
    for i, part in enumerate(parts):
        if i > 0:
            content.append({"type": "image"})
        stripped = part.strip()
        if stripped:
            content.append({"type": "text", "text": stripped})
    return content


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": _str_to_content_list(chat)}]
        elif isinstance(chat, list):
            # Deep copy to avoid mutating the original dataset entries.
            chat = copy.deepcopy(chat)
            for msg in chat:
                if isinstance(msg.get("content"), str):
                    msg["content"] = _str_to_content_list(msg["content"])
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


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
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.image_key = getattr(self.strategy.args, "image_key", "images")

        self.prompts = []
        self.labels = []
        self.images = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.images.append(data.get(self.image_key, None))
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx], self.images[idx]

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        images = []
        for datasource, prompt, label, img in item_list:
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            images.append(img)

        return datasources, prompts, labels, images
