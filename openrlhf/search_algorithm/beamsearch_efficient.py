# TODO: observe OOM, perhaps due to the traj is too long, we should stop some trajs when necessary

import torch
import re
from openrlhf.search_algorithm.search_utils import Tree, Node, \
    DEFAULT_TEMPERATURE, DEFAULT_BEAM_SIZE, DEFAULT_MAX_LENGTH, DEFAULT_MAX_STEP_LENGTH

# temporal hard code
LIMIT=32
N=2
BEAM=1
MAX_REPEAT=2
END_OF_STEP=["\n\n", "\n", "<|endoftext|>"]
MAX_CHAR_PER_STEP = 512
MAX_CHAR_PER_PATH = 2048
MAX_NEW_TOKENS=256
ADD_GREEDY = False


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

def search(query, tokenizer, actor, critic):
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
                state = tree.add_node(next_step, next_value.item(), anchor, next_step.endswith(tokenizer.eos_token))
                if len(next_step) == 0 or len(next_step) > MAX_CHAR_PER_STEP or len(traj + next_step) > MAX_CHAR_PER_PATH:
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
    
    if ADD_GREEDY:
        with torch.no_grad():
            greedy_traj = get_full_traj(query, tokenizer, actor, greedy=True)
        return [final_traj, greedy_traj]
    else:
        return [final_traj]