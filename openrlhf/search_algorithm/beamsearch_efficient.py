# TODO: observe OOM, perhaps due to the traj is too long, we should stop some trajs when necessary

import torch
import re
from vllm import SamplingParams
import ray
import numpy as np
import jsonlines
from tqdm import tqdm
from openrlhf.search_algorithm.search_utils import Tree, Node, \
    DEFAULT_TEMPERATURE, DEFAULT_N, DEFAULT_BEAM_SIZE, DEFAULT_SEARCH_STEPS, DEFAULT_MAX_LENGTH, DEFAULT_MAX_STEP_LENGTH

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor, greedy=False):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=not greedy, max_new_tokens=1024,
                             temperature=DEFAULT_TEMPERATURE, tokenizer=tokenizer,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences[0]

def get_next_steps(trajs, tokenizer, actor):
    input_ids = tokenizer(trajs, padding=True, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, stop_strings=END_OF_STEP, max_new_tokens=MAX_NEW_TOKENS,
                             temperature=DEFAULT_TEMPERATURE, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    input_len = input_ids["input_ids"].shape[1]
    sequences = tokenizer.batch_decode(outputs[:, input_len:], keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_step_scores(trajs, tokenizer, critic):
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(critic.device) for k, v in input_ids.items()}
    outputs = critic.compute_value(**input_ids, return_dict=False)
    return (torch.clamp(outputs[0].squeeze(), min=-1, max=1) + 1) / 2

def search(query, tokenizer, actor, critic=None, search_args=None):

    beam_size = search_args.get("beam_size", DEFAULT_BEAM_SIZE)
    candidate_size = search_args.get("candidate_size", DEFAULT_N)
    assert candidate_size % beam_size == 0
    expand_size = candidate_size // beam_size
    search_steps = search_args.get("search_steps", DEFAULT_SEARCH_STEPS)
    max_step_length = search_args.get("max_step_length", DEFAULT_MAX_STEP_LENGTH)
    max_length = search_args.get("max_length", DEFAULT_MAX_LENGTH)
    add_greedy = search_args["add_greedy"]

    tree = Tree(query)
    query = tree.question
    for search_iter in range(search_steps):
        actions = tree.get_beam_to_expand(beam_size)
        if search_iter < 1:
            actions = actions * beam_size
        if actions:
            trajs = [action.print_path() for action in actions]
            trajs, anchors = trajs * expand_size, actions * expand_size
            with torch.no_grad():
                next_steps = get_next_steps(trajs, tokenizer, actor)
                next_values = get_step_scores([traj + next_step for traj, next_step in zip(trajs, next_steps)], tokenizer, critic)
            for anchor, traj, next_step, next_value in zip(anchors, trajs, next_steps, next_values):
                state = tree.add_node(next_step, next_value.item(), anchor, next_step.endswith(tokenizer.eos_token))
                if len(next_step) == 0 or len(next_step) > max_step_length or len(traj + next_step) > max_length:
                    state.value = -1
                # print((search_iter, traj, next_step, next_value))
        else:
            break

    # return the best traj
    terminal_nodes = [node for node in tree.all_nodes if node.is_leaf]
    final_traj = None
    if terminal_nodes:
        best_node = max(terminal_nodes, key=lambda x: x.value)
        final_traj = best_node.print_path()
    else:
        with torch.no_grad():
            final_traj = get_full_traj(query, tokenizer, actor)
        # return None

    if add_greedy:
        with torch.no_grad():
            greedy_traj = get_full_traj(query, tokenizer, actor, greedy=True)
        return [final_traj, greedy_traj]
    else:
        return [final_traj]


def get_full_traj_vllm(traj, tokenizer, actor):
    llms = actor
    trajs = [traj]
    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        top_p=1,
        top_k=-1,
        max_tokens=DEFAULT_MAX_LENGTH,
        min_tokens=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        logprobs = 0,
    )

    # Expand prompt list based on the number of samples per prompt
    # all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in trajs], [])
    all_prompts = trajs
    all_prompt_token_ids = tokenizer(all_prompts, add_special_tokens=False, max_length=1024, truncation=True,)["input_ids"]

    # Distribute requests to engines and collect responses to outputs
    all_output_refs = []
    batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        if prompt_token_ids:
            all_output_refs.append(
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=False)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    return sequences

def get_next_steps_vllm(trajs, tokenizer, actor):
    llms = actor
    sampling_params = SamplingParams(
        temperature=DEFAULT_TEMPERATURE,
        top_p=1,
        top_k=-1,
        max_tokens=DEFAULT_MAX_STEP_LENGTH,
        min_tokens=1,
        stop=["\n", "\n\n"],
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        logprobs = 0,
    )

    # Expand prompt list based on the number of samples per prompt
    # all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in trajs], [])
    all_prompts = trajs
    all_prompt_token_ids = tokenizer(all_prompts, add_special_tokens=False, max_length=1024, truncation=True,)["input_ids"]

    # Distribute requests to engines and collect responses to outputs
    all_output_refs = []
    batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        if prompt_token_ids:
            all_output_refs.append(
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids, use_tqdm=False)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    cumulative_logprobs = [output.outputs[0].cumulative_logprob for output in all_outputs]
    avg_logprobs = [sum(list(item.values())[0].logprob for item in output.outputs[0].logprobs) / len(output.outputs[0].logprobs) for output in all_outputs] # if the same logprob, prefer longer sequence
    return sequences, cumulative_logprobs, avg_logprobs


# multi-threads
def process_single_query(query, args):
    """处理单个查询的线程函数"""
    tree = Tree(query)
    processed_query = tree.question

    for search_iter in range(args["LIMIT"]):
        actions = tree.get_beam_to_expand(args["BEAM"])
        if search_iter < 1:
            actions = actions * args["BEAM"]

        if not actions:
            break

        trajs = [action.print_path() for action in actions]
        expanded_trajs = trajs * (args["N"] // args["BEAM"])
        expanded_anchors = actions * (args["N"] // args["BEAM"])

        with torch.no_grad():
            next_steps, sum_logps, avg_logps = get_next_steps_vllm(
                expanded_trajs, args["tokenizer"], args["actor"]
            )
            next_values = avg_logps if args["search_args"]["compute_reward_strategy"] == "average" else sum_logps

        for anchor, traj, next_step, next_value in zip(expanded_anchors, expanded_trajs, next_steps, next_values):
            next_value += anchor.value if anchor.parent is not None else 0
            state = tree.add_node(
                next_step, 
                next_value, 
                anchor, 
                next_step.endswith(args["tokenizer"].eos_token)
            )

    # 返回最佳轨迹
    terminal_nodes = [node for node in tree.all_nodes if node.is_leaf]
    if terminal_nodes:
        best_node = max(terminal_nodes, key=lambda x: x.value)
        return best_node.print_path()
    else:
        with torch.no_grad():
            return get_full_traj_vllm(processed_query, args["tokenizer"], args["actor"])[0]

from concurrent.futures import ThreadPoolExecutor
from functools import partial

def search_vllm(queries, tokenizer, actor, critic=None, search_args=None):
    """多线程处理主函数"""

    rank = torch.distributed.get_rank()  # 注意：需要确认多线程环境下的rank行为

    worker_args = {
        "tokenizer": tokenizer,
        "actor": actor,
        "critic": critic,
        "BEAM": DEFAULT_BEAM_SIZE,
        "N": DEFAULT_N,
        "LIMIT": 32,
        "search_args": search_args
    }

    # 创建线程池
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 使用偏函数固定参数
        process_fn = partial(process_single_query, args=worker_args)
        
        # 创建进度条
        with tqdm(total=len(queries), desc=f"Rank {rank}") as pbar:
            # 提交所有任务
            futures = []
            for query in queries:
                future = executor.submit(process_fn, query)
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append(future)
            
            # 收集结果（保持原始顺序）
            results = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                    results.append(None)  # 或者根据需求处理异常
    
    return results