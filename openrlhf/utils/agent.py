from abc import ABC, abstractmethod
from copy import deepcopy


class AgentInstanceBase(ABC):
    @abstractmethod
    async def __init__(self, *args, **kwargs):
        pass

    async def reset(self, states: dict, **kwargs):
        return states

    @abstractmethod
    async def step(self, state_dict: dict, **kwargs):
        raise NotImplementedError("AgentInstance.step is not implemented")


class AgentExecutorBase(ABC):
    def __init__(self, agent_instance_cls, max_length, llm_engine, hf_tokenizer):
        assert issubclass(agent_instance_cls, AgentInstanceBase), "AgentInstance must inherit from AgentInstanceBase"
        self.agent_instance = agent_instance_cls()

        self.llm_engine = llm_engine
        self.hf_tokenizer = hf_tokenizer
        self.max_length = max_length

    async def execute(self, prompt, label, sampling_params, request_id=None):
        # Initialize with reset function
        initial_states = {"observation": prompt, "label": label}
        reset_result = await self.agent_instance.reset(initial_states)
        observation_text = reset_result["observation"]

        # Tokenize the initial observation
        current_obs_tokens = self.hf_tokenizer(observation_text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].tolist()

        # Initialize tracking variables
        action_ranges = []
        total_reward = 0
        final_scores = 0
        extra_logs = {}

        if sampling_params.logprobs is not None:
            rollout_log_probs = [0.0] * len(current_obs_tokens)
        else:
            rollout_log_probs = None

        # Execute multiple steps of interaction
        while True:
            # Next sampling budget
            sampling_params.max_tokens = self.max_length - len(current_obs_tokens)
            # No budget to generate, break
            if sampling_params.max_tokens <= 0:
                break

            # Generate response asynchronously (input and output are token ids)
            request_output = await self.llm_engine.generate.remote(current_obs_tokens, deepcopy(sampling_params))
            action_tokens = request_output.outputs[0].token_ids
            action_text = request_output.outputs[0].text

            # Record action range in token space
            action_start = len(current_obs_tokens)
            action_end = action_start + len(action_tokens)
            action_ranges.append((action_start, action_end))

            # Call step function to get environment feedback
            states = {
                "observation_text": observation_text,
                "action_text": action_text,
                "label": label,
                "sampling_params": sampling_params,
            }
            step_result = await self.agent_instance.step(states)

            total_reward += step_result["rewards"].item()
            final_scores = step_result.get("scores", total_reward)
            environment_feedback_text = step_result["environment_feedback"]
            done = step_result["done"]
            extra_logs = step_result.get("extra_logs", {})

            # Concatenate observation, action, and environment_feedback, then tokenize
            observation_text = observation_text + action_text + environment_feedback_text
            current_obs_tokens = (
                current_obs_tokens
                + action_tokens
                + self.hf_tokenizer(environment_feedback_text, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ][0].tolist()
            )

            # Calculate rollout log probs
            if sampling_params.logprobs is not None:
                # action tokens logprobs
                for i, logprob in enumerate(request_output.outputs[0].logprobs):
                    rollout_log_probs.append(logprob[action_tokens[i]].logprob)
                # dummy logprobs for the env feedback tokens
                rollout_log_probs.extend([0.0] * (len(current_obs_tokens) - len(rollout_log_probs)))

            # Get sampling params from the environment step
            if step_result.get("sampling_params", None):
                sampling_params = step_result["sampling_params"]

            if done:
                break

        # Store the final response when agent execution is complete
        final_response = {
            "prompt": prompt,
            "label": label,
            "reward": total_reward,
            "scores": final_scores,
            "observation_tokens": current_obs_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "request_id": request_id,
            "extra_logs": extra_logs,
        }
        return final_response
