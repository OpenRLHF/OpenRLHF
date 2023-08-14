from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data):
    # stanfordnlp/SHP
    if exist_and_not_none(data, 'human_ref_A'):
        prompt = "Human:\n" +  data['history'] + "\nAssistant:\n"
        preferA = bool(data['labels'])
        chosen = data['human_ref_A'] if preferA else data['human_ref_B']
        reject = data['human_ref_B'] if preferA else data['human_ref_A']
    # Anthropic/hh-rlhf
    # tasksource/oasst1_pairwise_rlhf_reward
    elif exist_and_not_none(data, 'chosen') and exist_and_not_none(data, 'rejected'):
        prompt = data['prompt'] if exist_and_not_none(data, 'prompt') else ""
        if prompt.startswith('prompter:'):
            prompt = prompt.replace('prompter:', 'Human:\n').replace('assistant:', '\nAssistant:\n') + '\nAssistant:\n'

        chosen = data['chosen']
        reject = data['rejected']
    # lvwerra/stack-exchange-paired
    elif exist_and_not_none(data, 'response_j'):
        prompt = "Human:\n" +  data['question'] + "\nAssistant:\n"
        chosen = data['response_j']
        reject = data['response_k']
    # lmsys/chatbot_arena_conversations
    elif exist_and_not_none(data, 'winner') and exist_and_not_none(data, 'conversation_a'):
        def process_chatbot_arena_conversations(lll):
            result = []
            for l in lll:
                result.append(l['role'].replace('user', 'Human:').replace('assistant', 'Assistant:'))
                result.append(l['content'])
            return "\n".join(result)
        prompt = ""
        chosen = data['conversation_a'] if data['winner'] == 'model_a' else data['conversation_b']
        reject = data['conversation_b'] if data['winner'] == 'model_a' else data['conversation_a']
        chosen = process_chatbot_arena_conversations(chosen)
        reject = process_chatbot_arena_conversations(reject)
    # openai/webgpt_comparisons
    elif exist_and_not_none(data, 'answer_0') and exist_and_not_none(data, 'answer_1'):
        prompt = "Human:\n" +  data['question']['full_text'] + "\nAssistant:\n"
        chosen = data['answer_0'] if data['score_0'] > data['score_1'] else data['answer_1']
        reject = data['answer_1'] if data['score_0'] > data['score_1'] else data['answer_0']
    else:
        raise ValueError("reward_dataset key error")
    return prompt, chosen, reject


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject = preprocess_data(data)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)


    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]

        chosen = prompt + chosen + " " + self.tokenizer.eos_token 
        chosen_token = self.tokenizer(chosen,
                                    max_length=self.max_length,
                                    padding=False,
                                    truncation=True,
                                    return_tensors="pt")
        
        reject = prompt + reject + " " + self.tokenizer.eos_token 
        reject_token = self.tokenizer(reject,
                                    max_length=self.max_length,
                                    padding=False,
                                    truncation=True,
                                    return_tensors="pt")
        
        return chosen_token["input_ids"], chosen_token["attention_mask"], reject_token[\
            "input_ids"], reject_token["attention_mask"]

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        for chosen_id, chosen_mask, reject_id, rejects_mask in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks