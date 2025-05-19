from typing import List

import ray
import torch

from openrlhf.trainer.ppo_utils.experience_maker import Samples, SamplesGenerator


class SamplesGeneratorAsync(SamplesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        llms = self.vllm_engines
        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )
        truncate_length = self.prompt_max_len + kwargs.get("max_new_tokens", 1024)

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop("n_samples_per_prompt", args.n_samples_per_prompt)
        all_prompts = sum([[prompt] * n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])

        # Distribute requests to engines and collect responses to outputs
        refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size : (i + 1) * batch_size]
            labels = all_labels[i * batch_size : (i + 1) * batch_size]
            refs.append(
                llm.add_requests.remote(
                    sampling_params=sampling_params,
                    prompts=prompts,
                    labels=labels,
                    max_length=truncate_length,
                    hf_tokenizer=self.tokenizer,
                )
            )
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        # Group outputs by prompt
        prompt_groups = {}
        for output in all_outputs:
            prompt = output["prompt"]
            prompt_groups.setdefault(prompt, []).append(output)

        # Reorder outputs to keep same prompts together
        # This is very important for REINFORCE++-baseline/GRPO/RLOO
        all_outputs = []
        for prompt in prompt_groups.keys():
            all_outputs.extend(prompt_groups[prompt])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        # Process outputs one by one
        samples_list = []
        for output in all_outputs:
            # Tokenize state
            state_tokens = self.tokenizer(output["state"], add_special_tokens=False, return_tensors="pt")["input_ids"][
                0
            ]
            tokenized_state = state_tokens.tolist()

            # Convert action ranges to token indices
            tokenized_ranges = []
            for start, end in output["action_ranges"]:
                # Get token indices for the entire state up to end
                full_tokens = self.tokenizer(output["state"][:end], add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ][0]
                # Get token indices for the entire state up to start
                start_tokens = self.tokenizer(output["state"][:start], add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ][0]
                # Calculate token indices
                tokenized_ranges.append((len(start_tokens), len(full_tokens)))

            # Create tensors
            sequences = torch.tensor([tokenized_state])
            attention_mask = torch.tensor([[1] * len(tokenized_state)])

            # Create action mask based on tokenized action_ranges
            action_mask = torch.zeros_like(attention_mask)
            # Mark action positions in the mask
            for start, end in tokenized_ranges:
                action_mask[0, start:end] = 1

            # Apply length limit
            sequences = sequences[:, :truncate_length].to("cpu")
            attention_mask = attention_mask[:, :truncate_length].to("cpu")
            action_mask = action_mask[:, 1:truncate_length].to("cpu")
            response_length = action_mask.float().sum(dim=-1)
            total_length = attention_mask.float().sum(dim=-1)

            rollout_samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                response_length=response_length,
                total_length=total_length,
                prompts=[output["prompt"]],
                labels=[output["label"]],
                rewards=[output["reward"]],
            )
            samples_list.append(rollout_samples)

        return samples_list
