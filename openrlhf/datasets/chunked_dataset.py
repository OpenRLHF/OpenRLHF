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

        self.prompts = []
        self.responses = []
        
        for data in dataset:
            prompt, response = preprocess_data(
                data,
                self.input_template,
                self.input_key,
                self.output_key,
                apply_chat_template=None if self.pretrain_mode else self.apply_chat,
                multiturn=self.multiturn,
            )
            
            if not prompt or not response:
                continue
            
            prompt_ids = self.tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=False,
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
                chunk_ids = resp_ids[i:i+self.chunk_size]
                if not chunk_ids:
                    continue
                
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
        prompt_labels = torch.zeros_like(prompt_ids, dtype=torch.float32)
        
        response_ids = []
        response_am = []
        response_labels = []
        for chunk_ids in response_chunks:
            r_ids = torch.tensor([chunk_ids], dtype=torch.long)
            r_am = torch.ones_like(r_ids, dtype=torch.long)
            r_labels = torch.zeros_like(r_ids, dtype=torch.float32)
            response_ids.append(r_ids)
            response_am.append(r_am)
            response_labels.append(r_labels)

        # Return prompt + response chunks separately
        return (
            prompt_input_ids,
            prompt_attention_mask,
            prompt_loss_mask,
            response_input_ids,
            response_attention_masks,
            response_loss_masks,
        )

    def collate_fn(self, item_list):
        # Unpack
        prompts_input_ids = []
        prompts_attention_masks = []
        prompts_loss_masks = []
        responses_input_ids_grouped = []
        responses_attention_masks_grouped = []
        responses_loss_masks_grouped = []

        for (
            p_ids,
            p_attn,
            p_loss,
            r_ids_list,
            r_attn_list,
            r_loss_list,
        ) in item_list:
            prompts_input_ids.append(p_ids)
            prompts_attention_masks.append(p_attn)
            prompts_loss_masks.append(p_loss)
            responses_input_ids_grouped.append(r_ids_list)
            responses_attention_masks_grouped.append(r_attn_list)
            responses_loss_masks_grouped.append(r_loss_list)

        # Batch prompts with right padding
        batched_prompt_ids = zero_pad_sequences(prompts_input_ids, "right", self.tokenizer.pad_token_id)
        batched_prompt_attn = zero_pad_sequences(prompts_attention_masks, "right")
        batched_prompt_loss = zero_pad_sequences(prompts_loss_masks, "right")

        # Responses: keep ragged (list of lists of tensors) so you can process chunks separately
        return (
            batched_prompt_ids,
            batched_prompt_attn,
            batched_prompt_loss,
            responses_input_ids_grouped,
            responses_attention_masks_grouped,
            responses_loss_masks_grouped,
        )