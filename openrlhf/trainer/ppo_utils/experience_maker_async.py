from typing import List

import ray
import torch

from openrlhf.trainer.ppo_utils.experience_maker import Experience, SamplesGenerator


class SamplesGeneratorAsync(SamplesGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_vllm(self, all_prompts: List[str], all_labels, **kwargs) -> List[Experience]:
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
            logprobs=1 if self.strategy.args.use_vllm_logprobs else None,
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
        experiences_list = []
        for output in all_outputs:
            # Tokenize observation
            observation_tokens = self.tokenizer(output["observation"], add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0]
            tokenized_observation = observation_tokens.tolist()
            if observation_tokens[-1] != eos_token_id:
                tokenized_observation.append(eos_token_id)

            # Convert action ranges to token indices
            tokenized_ranges = []
            for start, end in output["action_ranges"]:
                # Get token indices for the entire observation up to end
                full_tokens = self.tokenizer(
                    output["observation"][:end], add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
                # Get token indices for the entire observation up to start
                start_tokens = self.tokenizer(
                    output["observation"][:start], add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]
                # Calculate token indices
                tokenized_ranges.append((len(start_tokens), len(full_tokens)))
            if observation_tokens[-1] != eos_token_id:
                tokenized_ranges[-1] = (tokenized_ranges[-1][0], tokenized_ranges[-1][1] + 1)

            # Create tensors
            sequences = torch.tensor(tokenized_observation)
            attention_mask = torch.tensor([1] * len(tokenized_observation))

            # Create action mask based on tokenized action_ranges
            action_mask = torch.zeros_like(attention_mask)
            # Mark action positions in the mask
            for start, end in tokenized_ranges:
                action_mask[start:end] = 1

            # Extract logprobs from vllm output (only when using vLLM logprobs)
            if args.use_vllm_logprobs:
                rollout_log_probs_tensor = None
                # Check if logprobs are available in the output
                if "logprobs" in output and output["logprobs"] is not None:
                    # Initialize logprobs for all tokens in the observation
                    full_logprobs = [0.0] * len(tokenized_observation)

                    # Map logprobs to their corresponding token positions
                    logprobs_idx = 0
                    response_logprobs = output["logprobs"]

                    # Process each action range and assign logprobs
                    for start_pos, end_pos in tokenized_ranges:
                        # Get the number of tokens in this action range
                        num_tokens = end_pos - start_pos

                        # Assign logprobs for this action range
                        for i in range(num_tokens):
                            if logprobs_idx < len(response_logprobs) and (start_pos + i) < len(full_logprobs):
                                full_logprobs[start_pos + i] = response_logprobs[logprobs_idx]
                                logprobs_idx += 1

                    # Create tensor and apply truncation
                    if len(full_logprobs) > 0:
                        rollout_log_probs_tensor = torch.tensor(full_logprobs, dtype=torch.float)
                        rollout_log_probs_tensor = rollout_log_probs_tensor[1:truncate_length].to("cpu")
                    else:
                        rollout_log_probs_tensor = None
                else:
                    rollout_log_probs_tensor = None
            else:
                rollout_log_probs_tensor = None

            # Apply length limit
            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")

            # Calculate response length (distance between first and last 1)
            ones_indices = torch.where(action_mask)[0]
            response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
            total_length = attention_mask.float().sum()
            is_clipped = total_length >= truncate_length

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
                "reward": torch.tensor([output["reward"]]),
                "score": torch.tensor([output["scores"]]),
            }

            # Process extra_logs
            extra_logs = output.get("extra_logs", {})
            for key, value in extra_logs.items():
                info[key] = torch.tensor([value.item()])

            experience = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                action_log_probs=(
                    rollout_log_probs_tensor.unsqueeze(0) if rollout_log_probs_tensor is not None else None
                ),
                prompts=[output["prompt"]],
                labels=[output["label"]],
                rewards=torch.tensor([output["reward"]]),
                scores=torch.tensor([output["scores"]]),
                info=info,
            )
            experiences_list.append(experience)

        return experiences_list
