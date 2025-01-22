# TODO: observe OOM, perhaps due to the traj is too long, we should stop some trajs when necessary
from vllm import SamplingParams
import ray

import random
import torch
import re

# temporal hard code
LIMIT=32
N=4
BEAM=2
TEMPERATURE=1
MAX_REPEAT=2
END_OF_STEP=["\n\n", "\n", "<|endoftext|>"]
MAX_CHAR_PER_STEP = 512
MAX_CHAR_PER_PATH = 2048
MAX_NEW_TOKENS=256

#### Search Tree ####
class Node:
    def __init__(self, content, value, parent, timestep, tree, is_leaf=False):
        self.content = content
        self.value = value
        self.parent = parent
        self.children = []
        self.timestep = timestep
        self.tree = tree
        self.is_leaf = is_leaf

    def get_depth(self):
        return len(self.return_path()) + 1

    def return_path(self):
        if self.parent is None:
            return [self.content]
        return self.parent.return_path() + [self.content]

    def print_path(self):
        return "".join(self.return_path())

class Tree:
    def __init__(self, question):
        self.question = question
        self.all_nodes = []
        self.root = Node(question, 0, None, 0, self)
        self.all_nodes.append(self.root)

    def return_timestep(self):
        return max([node.timestep for node in self.all_nodes])

    def add_node(self, content, value, parent, is_leaf=False):
        node = Node(content, value, parent, parent.timestep + 1, self, is_leaf)
        parent.children.append(node)
        self.all_nodes.append(node)
        return node

    def get_beam_to_expand(self, beam_size=5):
        curr_timestep = self.return_timestep()
        latest_nodes = [node for node in self.all_nodes if node.is_leaf or node.timestep == curr_timestep]
        # beam = sorted(latest_nodes, key=lambda x: x.value, reverse=True)[:beam_size]
        content_dict = {}
        beam = []
        for node in sorted(latest_nodes, key=lambda x: x.value, reverse=True):
            if content_dict.get(node.content, 0) >= MAX_REPEAT and node.value < 0:
                continue
            beam.append(node)
            if len(beam) >= beam_size:
                break
            if not node.is_leaf:
                if node.content not in content_dict:
                    content_dict[node.content] = 1
                else:
                    content_dict[node.content] += 1
        return [node for node in beam if not node.is_leaf]
########

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor):
    llms = actor
    trajs = [traj]
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1,
        top_k=-1,
        max_tokens=1024,
        min_tokens=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
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
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    return sequences

def get_next_steps(trajs, tokenizer, actor):
    llms = actor
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1,
        top_k=-1,
        max_tokens=1024,
        min_tokens=1,
        stop=END_OF_STEP,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
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
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    return sequences

def get_step_scores(trajs, tokenizer, critic):
    # return [random.randint(0, 1) for _ in range(len(trajs))] # debug
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    # outputs = critic.compute_value.remote(**input_ids)
    outputs = critic.forward.remote(**input_ids)
    outputs = ray.get(outputs)
    # return [value.item() for value in (torch.clamp(outputs.squeeze(), min=-1, max=1) + 1) / 2]
    return [random.randint(0, 1) for _ in range(len(trajs))] # debug

def search(query, tokenizer, actor, critic):
    rank = torch.distributed.get_rank()
    tree = Tree(query)
    query = tree.question
    for search_iter in range(LIMIT):
        actions = tree.get_beam_to_expand(BEAM)
        if search_iter < 1:
            actions = actions * BEAM
        if actions:
            trajs = [action.print_path() for action in actions]
            trajs, anchors = trajs * (N // BEAM), actions * (N // BEAM)
            with torch.no_grad():
                next_steps = get_next_steps(trajs, tokenizer, actor)
                next_values = get_step_scores([traj + next_step for traj, next_step in zip(trajs, next_steps)], tokenizer, critic)
            for anchor, traj, next_step, next_value in zip(anchors, trajs, next_steps, next_values):
                state = tree.add_node(next_step, next_value, anchor, next_step.endswith(tokenizer.eos_token))
                if len(next_step) == 0 or len(next_step) > MAX_CHAR_PER_STEP or len(traj + next_step) > MAX_CHAR_PER_PATH:
                    state.value = -1
                print((rank, search_iter, traj, next_step, next_value))
        else:
            break
    
    # return the best traj
    terminal_nodes = [node for node in tree.all_nodes if node.is_leaf]
    if terminal_nodes:
        best_node = max(terminal_nodes, key=lambda x: x.value)
        return best_node.print_path()
    else:
        with torch.no_grad():
            return get_full_traj(query, tokenizer, actor)[0]
        # return None