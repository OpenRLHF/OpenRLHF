import json
import os
import queue
from typing import Any, List

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm.inputs import TokensPrompt
from vllm.sampling_params import BeamSearchParams, SamplingParams

from openrlhf.utils.logging_utils import init_logger

from .utils import get_bundle_indices, ray_noset_visible_devices

logger = init_logger(__name__)


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


class BaseLLMRayActor:
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        kwargs.pop("agent_func_path", None)
        noset_visible_devices = ray_noset_visible_devices()
        if kwargs.get("distributed_executor_backend") == "ray":
            # a hack to make the script work.
            # stop ray from manipulating *_VISIBLE_DEVICES
            # at the top-level when the distributed_executor_backend is ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        elif noset_visible_devices:
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")
        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        # Number of actors that will send prompt to this engine
        self.requests = {}
        self.response_queues = queue.Queue()

        full_determinism = kwargs.pop("full_determinism", False)
        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.kwargs = kwargs

        import vllm

        if vllm.__version__ >= "0.9.0":
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@ray.remote
class LLMRayActor(BaseLLMRayActor):
    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        super().__init__(*args, bundle_indices=bundle_indices, **kwargs)

        import vllm

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

    def get_tool_call_results(self, prompt_token_ids, responses, tokenizer, remote_reward_model):
        def validate_tool_call_args(args):
            if not isinstance(args, dict):
                return False
            if "smiles" not in args:
                return False
            if not isinstance(args["smiles"], list):
                return False
            for smiles in args["smiles"]:
                if not isinstance(smiles, str):
                    return False
            return True

        # Perform tool calls if detected
        idx_tool_call = []
        tool_calls_res_map = {}  # Map from tool call result index to the original response index
        prompt_list = []
        query_list = []
        last_idx = 0

        errors = {}
        for i, prompt_token_id in enumerate(prompt_token_ids):
            response = responses[i]
            resp = response.outputs[0].text
            if len(resp.split("<tool_call>")) > 1 and len(resp.split("</tool_call>")) > 1:
                # This response contains tool calls, so we need to handle it
                # If it is the 2nd tool call return an error as a result
                if resp.count("<tool_call>") > 1:
                    errors[i] = "ERROR: Max number of tool calls exceeded."
                    continue
                elif "<tool_response>" in resp.split("<tool_call>")[-1]:  # Tool call already answered
                    continue
                # Extract the tool call arguments from the response
                json_args = resp.split("<tool_call>")[-1].split("</tool_call>")[0].replace("\n", "")
                try:
                    tool_call = json.loads(json_args)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode tool call JSON: {json_args}")
                    continue
                args = tool_call["arguments"]
                # Append the tool call result to the response
                assert tokenizer is not None
                current_completion = prompt_token_id + response.outputs[0].token_ids
                if not current_completion[-1] == tokenizer.eos_token_id:
                    continue
                assert remote_reward_model is not None, "remote_reward_model must be provided for tool calls"
                # Validate the tool call arguments
                if isinstance(args, list) and len(args) == 1:
                    args = args[0]
                if not validate_tool_call_args(args):
                    continue

                tool_calls_res_map[i] = []
                for smiles in args["smiles"]:
                    query_list.append("<answer> " + smiles + " </answer>")
                    prompt_list.append(tokenizer.decode(prompt_token_id))
                    tool_calls_res_map[i].append(last_idx)

                    last_idx += 1
                idx_tool_call.append(i)

        tool_calls_res = remote_reward_model.get_rewards.remote(
            queries_list=query_list,
            prompts_list=prompt_list,
            labels_list=[None] * len(prompt_list),
        )
        tool_calls_res = ray.get(tool_calls_res)
        tool_calls_res = sum(
            [res["rewards"].tolist() for res in tool_calls_res if res is not None], []
        )  # Flatten the list of results for each idx
        tool_calls_res = {
            i: [tool_calls_res[idx] for idx in tool_calls_res_map[i]] for i in idx_tool_call
        }  # Get the results for each idx_tool_call (with possible multiple results)
        for i in errors:
            tool_calls_res[i] = [errors[i]]
        return tool_calls_res

    def get_prompt_with_tool_results(self, prompt_token_ids, responses, mm_datas, tokenizer, remote_reward_model):
        """
        Process the responses to extract tool call results and append them to the original prompt.
        """
        tool_calls_res = self.get_tool_call_results(prompt_token_ids, responses, tokenizer, remote_reward_model)

        second_generation_request = []
        second_mm_datas = [mm_datas[i] for i in tool_calls_res] if mm_datas else None

        for i in tool_calls_res:
            prompt_token_id = prompt_token_ids[i]
            response = responses[i]
            assert tool_calls_res[i] != []
            result = tool_calls_res[i]
            for i, r in enumerate(result):
                if isinstance(r, (int, float)):
                    # Turn to str with at most 2 decimal places
                    result[i] = f"{float(r):.2f}"
                else:
                    logger.warning(f"Tool call result is not a number: {r}")
            result = ";".join(result)
            tool_response = f"\n<|im_start|>user\n<tool_response>\n{str(result)}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            tool_response_token_ids = tokenizer(
                tool_response,
            )["input_ids"]

            # Append the tool response to the response
            new_token_ids = prompt_token_id + response.outputs[0].token_ids + tool_response_token_ids
            if not second_mm_datas:
                second_generation_request.append(TokensPrompt(prompt_token_ids=new_token_ids))
            else:
                second_generation_request.append(
                    TokensPrompt(prompt_token_ids=new_token_ids, mm_data=second_mm_datas[i])
                )
        return second_generation_request, list(tool_calls_res.keys())

    def apply_tool_calls(
        self,
        prompt_token_ids,
        responses,
        mm_datas=None,
        tokenizer=None,
        remote_reward_model=None,
        sampling_params=None,
    ):
        second_generation_request, idx_tool_call = self.get_prompt_with_tool_results(
            prompt_token_ids, responses, mm_datas, tokenizer, remote_reward_model
        )
        if len(idx_tool_call) > 0:
            # If there are tool calls, we need to generate a second round of responses
            second_generation_responses = self.llm.generate(
                prompts=second_generation_request, sampling_params=sampling_params
            )
            for i, idx in enumerate(idx_tool_call):
                n = len(responses[idx].prompt_token_ids)
                final_output = (
                    second_generation_responses[i].prompt_token_ids[n:]
                    + second_generation_responses[i].outputs[0].token_ids
                )

                decoded_output = tokenizer.batch_decode([final_output])[0]
                responses[idx].outputs[0].token_ids = final_output
                responses[idx].outputs[0].text = decoded_output
        return responses

    def add_requests(self, sampling_params, prompt_token_ids, mm_datas=None, tokenizer=None, remote_reward_model=None):
        """
        Process requests from rank0 and generate responses.
        Since only rank0 will send requests, we don't need to track actor ranks.
        """

        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.tolist()

        if not mm_datas:
            requests = [TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids]
        else:
            assert len(prompt_token_ids) == len(mm_datas), "prompt_token_ids and mm_datas must have the same length"
            requests = [
                TokensPrompt(prompt_token_ids=r, mm_data=mm_data) for r, mm_data in zip(prompt_token_ids, mm_datas)
            ]
        if isinstance(sampling_params, SamplingParams):
            responses = self.llm.generate(prompts=requests, sampling_params=sampling_params)
        elif isinstance(sampling_params, BeamSearchParams):
            responses = self.llm.beam_search(prompts=requests, sampling_params=sampling_params)
        for _ in range(2):
            responses = self.apply_tool_calls(
                prompt_token_ids,
                responses,
                mm_datas=mm_datas,
                tokenizer=tokenizer,
                remote_reward_model=remote_reward_model,
                sampling_params=sampling_params,
            )
        self.response_queues.put(responses)

    def get_responses(self):
        """
        Return the responses for the actor with the given rank
        """
        return self.response_queues.get()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    llm_actor_cls=LLMRayActor,
    agent_func_path=None,
):
    import vllm

    assert vllm.__version__ > "0.8.2", "OpenRLHF only supports vllm > 0.8.2"

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = get_bundle_indices(shared_pg, i, tensor_parallel_size)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        vllm_engines.append(
            llm_actor_cls.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="openrlhf.trainer.ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype="bfloat16",
                trust_remote_code=True,
                full_determinism=full_determinism,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,
                agent_func_path=agent_func_path,
            )
        )

    if vllm_enable_sleep:
        batch_vllm_engine_call(vllm_engines, "sleep")

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """
    Batch call a method on multiple vLLM engines.
    Args:
        engines: List of vLLM engine instances
        method_name: Name of the method to call
        rank_0_only: Only execute on rank 0 if True
        *args: Positional arguments to pass to the method
        **kwargs: Keyword arguments to pass to the method
    Returns:
        List of results from ray.get() if on rank 0, None otherwise
    """
    import torch

    if torch.distributed.is_initialized():
        if rank_0_only and torch.distributed.get_rank() != 0:
            return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return ray.get(refs)
