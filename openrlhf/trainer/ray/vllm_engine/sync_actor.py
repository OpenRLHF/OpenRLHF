import ray

from .base import BaseLLMRayActor


@ray.remote
class LLMRayActorSync(BaseLLMRayActor):
    """Synchronous vLLM actor (legacy)."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        import vllm

        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        self.llm = vllm.LLM(*args, **self.kwargs)

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend, use_ray):
        return self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend, use_ray),
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.collective_rpc("update_weight", args=(name, dtype, shape, empty_cache))

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles, empty_cache=False):
        return self.llm.collective_rpc("update_weight_cuda_ipc", args=(name, dtype, shape, ipc_handles, empty_cache))

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.llm.sleep(level=level)

    def wake_up(self):
        self.llm.wake_up()

    def add_requests(self, sampling_params, prompt_token_ids):
        """Generate responses synchronously from tokenized prompts."""
        from vllm.inputs import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        self.response_queues.put(responses)

    def get_responses(self):
        """Return responses for the actor."""
        return self.response_queues.get()
