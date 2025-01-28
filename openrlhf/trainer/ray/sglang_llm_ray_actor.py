import os

import ray
import sglang
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from openrlhf.trainer.ray.utils import ray_noset_visible_devices
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():

    return os.environ


@ray.remote
class SGLangLLMRayActor:
    def __init__(self, *args, **kwargs):
        # Some of the parameters leads to error in token-in-token-out mode
        self.llm = sglang.Engine(
            model_path=args[0],
            trust_remote_code=kwargs.get("trust_remote_code", True),
            dtype=kwargs.get("dtype", "auto"),
            tp_size=kwargs.get("tensor_parallel_size", 1),
            device="cuda",
            random_seed=kwargs.get("seed", 42),
            # disable_radix_cache=not kwargs.get("enable_prefix_caching", False),
            # disable_cuda_graph=not kwargs.get("enforce_eager", False),
            # disable_cuda_graph_padding=not kwargs.get("enable_prefix_caching", False),
            # context_length=kwargs.get("max_model_len", None),
            log_level="info",
            skip_tokenizer_init=True,
        )

    def generate(self, sampling_params, prompt_token_ids, stop_token_ids):

        # min_tokens, include_stop_str_in_output is not used in sglang

        sampling_params = dict(
            max_new_tokens=sampling_params.get("max_tokens", 1024),
            top_p=sampling_params.get("top_p", 1),
            top_k=sampling_params.get("top_k", 50),
            temperature=sampling_params.get("temperature", 1),
            repetition_penalty=sampling_params.get("repetition_penalty", 1),
            skip_special_tokens=sampling_params.get("skip_special_tokens", False),
            stop_token_ids=stop_token_ids,
        )
        outputs = self.llm.generate(input_ids=prompt_token_ids, sampling_params=sampling_params)
        return outputs

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.llm.init_weights_update_group(
            master_address, master_port, rank_offset, world_size, group_name, backend="nccl"
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        return self.llm.update_weights_from_distributed(name, dtype, shape)

    def stop_remote_worker_execution_loop(self):
        # SGLang does not needs this function
        pass


def create_llm_ray_actor_sglang(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
):
    inference_engines = []
    # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES will always be set in current context,
    # So we need to get env variables from ray process to check if it is set.
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    for i in range(num_engines):
        # When tensor_parallel_size=1 and RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is not set
        # (SGLang mp backend will work smoothly only when *_VISIBLE_DEVICES is modified),
        # SGLang init model in LLMEngine directly, assign 1 GPU for it.
        num_gpus = int(tensor_parallel_size == 1 and not noset_visible_devices)
        scheduling_strategy = None
        if tensor_parallel_size > 1 or noset_visible_devices:
            bundles = [{"GPU": 1, "CPU": 1}] * tensor_parallel_size
            pg = placement_group(bundles)
            ray.get(pg.ready())
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0
            )

        actor_cls = SGLangLLMRayActor

        inference_engines.append(
            actor_cls.options(
                num_cpus=1,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                pretrain,
                noset_visible_devices=noset_visible_devices,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                seed=seed + i,
                enable_prefix_caching=enable_prefix_caching,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                backend="sglang",
            )
        )
    return inference_engines
