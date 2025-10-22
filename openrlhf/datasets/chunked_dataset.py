from typing import Callable
import torch
from torch.utils.data import Dataset
from openrlhf.datasets.sft_dataset import preprocess_data
from openrlhf.utils.utils import zero_pad_sequences


class ChunkedDataset(Dataset):
    """Dataset that chunks a dataset into chunks of a given size."""

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        strategy,
        chunk_size: int,
        input_template=None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.chunk_size = chunk_size

        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        self.prompts: List[List[int]] = []
        self.responses: List[List[List[int]]] = []

        for data in dataset:
            prompt, response = preprocess_data(
                data,
                self.input_template,
                self.input_key,
                self.output_key,
                apply_chat_template=self.apply_chat,
                multiturn=self.multiturn,
            )

            if not prompt or not response:
                continue

            prompt_ids = self.tokenizer(
                prompt,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].tolist()
            self.prompts.append(prompt_ids)

            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            resp_ids = self.tokenizer(
                response,
                padding=False,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].tolist()

            # Chunking
            response_chunks = []
            for i in range(0, len(resp_tokens), self.chunk_size):
                chunk_ids = resp_ids[i : i + self.chunk_size]
                if not chunk_ids:
                    continue
                if len(chunk_ids) < self.chunk_size:
                    pad = self.chunk_size - len(chunk_ids)
                    chunk_ids = chunk_ids + [self.tokenizer.pad_token_id] * pad

                response_chunks.append(chunk_ids)

            self.responses.append(response_chunks)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompts = self.prompts[idx]
        response_chunks = self.responses[idx]

        # Build tensors directly from precomputed token ids
        prompt_ids = torch.tensor([prompts], dtype=torch.long)
        prompt_am = torch.ones_like(prompt_ids, dtype=torch.long)

        # Combine chunks along batch dimension
        response_ids = []
        response_am = []
        for chunk_ids in response_chunks:
            r_ids = torch.tensor([chunk_ids], dtype=torch.long)
            r_am = (r_ids != self.tokenizer.pad_token_id).long()
            response_ids.append(r_ids)
            response_am.append(r_am)

        # Return prompt + response chunks separately
        return (
            prompt_ids,
            prompt_am,
            response_ids,
            response_am,
        )

    def collate_fn(self, batch):
        # Unpack
        prompt_ids_list: List[torch.Tensor] = []
        prompt_attn_list: List[torch.Tensor] = []
        all_chunk_ids: List[torch.Tensor] = []
        all_chunk_attn: List[torch.Tensor] = []
        chunk_owner_indices: List[int] = []

        for owner_idx, (
            p_ids,
            p_attn,
            r_ids_list,
            r_attn_list,
        ) in batch:
            prompts_input_ids.append(p_ids)
            prompts_attention_masks.append(p_attn)
            for c_ids, c_attn in zip(r_ids_list, r_attn_list):
                all_chunk_ids.append(c_ids)
                all_chunk_attn.append(c_attn)
                chunk_owner_indices.append(owner_idx)

        # Batch prompts with right padding
        batched_prompt_ids = zero_pad_sequences(prompts_input_ids, "right", self.tokenizer.pad_token_id)
        batched_prompt_attn = zero_pad_sequences(prompts_attention_masks, "right")

        chunk_input_ids = torch.cat(all_chunk_ids, dim=0)  # [B_chunks, T_chunk]
        chunk_attn = torch.cat(all_chunk_attn, dim=0)  # [B_chunks, T_chunk]
        chunk2sample = torch.tensor(chunk_owner_indices, dtype=torch.long)  # [B_chunks]

        return (
            batched_prompt_ids,
            batched_prompt_attn,
            chunk_input_ids,
            chunk_attn,
            chunk2sample,
        )
