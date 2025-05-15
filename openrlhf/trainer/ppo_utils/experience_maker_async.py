from typing import List

import ray
import torch

from openrlhf.models.utils import process_sequences
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker, Samples


class RemoteExperienceMakerAsync(RemoteExperienceMaker):
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
            refs.append(llm.add_requests.remote(sampling_params=sampling_params, prompts=prompts, labels=labels))
        ray.get(refs)

        # Retrieve and combine results from all outputs
        all_output_refs = []
        for i, llm in enumerate(llms):
            all_output_refs.append(llm.get_responses.remote())
        all_outputs = sum(ray.get(all_output_refs), [])

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id

        # Group outputs by micro_rollout_batch_size
        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            batch_outputs = all_outputs[i : i + args.micro_rollout_batch_size]
            batch_prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            batch_labels = all_labels[i : i + args.micro_rollout_batch_size]

            # Calculate max lengths for this batch only
            batch_max_input_len = max(len(output.prompt_token_ids) for output in batch_outputs)
            batch_max_output_len = max(len(output.outputs[0].token_ids) for output in batch_outputs)

            sequences = []
            for output in batch_outputs:
                # left padding input
                input_len = len(output.prompt_token_ids)
                input_ids = [pad_token_id] * (batch_max_input_len - input_len) + list(output.prompt_token_ids)

                # right padding output
                output_len = len(output.outputs[0].token_ids)
                output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (batch_max_output_len - output_len)

                # concat input and output
                sequences.append(input_ids + output_ids)

            sequences = torch.tensor(sequences)
            sequences, attention_mask, action_mask = process_sequences(
                sequences, batch_max_input_len, eos_token_id, pad_token_id
            )
            sequences = sequences.to("cpu")
            attention_mask = attention_mask.to("cpu")
            action_mask = action_mask.to("cpu")
            response_length = action_mask.float().sum(dim=-1)
            total_length = attention_mask.float().sum(dim=-1)

            rollout_samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                response_length=response_length,
                total_length=total_length,
                prompts=batch_prompts,
                labels=batch_labels,
            )
            samples_list.append(rollout_samples)

        return samples_list
