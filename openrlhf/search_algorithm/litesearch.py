import torch
import re
import math

# temporal hard code
LIMIT=64
N=8
EXCEPTED = 0.95
TEMPERATURE=1
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

    def select_next_node(self):
        available_nodes = [node for node in self.all_nodes if len(node.children) < 1] # have not been explored
        best_node = max(available_nodes, key=lambda x: x.value)
        return best_node
########

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, max_new_tokens=512,
                             temperature=TEMPERATURE, tokenizer=tokenizer, 
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_next_steps(trajs, tokenizer, actor):
    input_ids = tokenizer(trajs, padding=True, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, stop_strings=END_OF_STEP, max_new_tokens=MAX_NEW_TOKENS,
                             temperature=TEMPERATURE, tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    input_len = input_ids["input_ids"].shape[1]
    sequences = tokenizer.batch_decode(outputs[:, input_len:], keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_step_scores(trajs, tokenizer, critic):
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(critic.device) for k, v in input_ids.items()}
    outputs = critic.compute_value(**input_ids, return_dict=False)
    return (torch.clamp(outputs[0].squeeze(-1), min=-1, max=1) + 1) / 2

def fix_value(state, eos_token):
    if state.parent is not None: # repeat
        if state.parent.content == state.content:
            state.value = -100
    if state.content is not None: # too short or too long
        if len(state.content) == 0:
            state.value = -100
        elif len(state.content) > MAX_CHAR_PER_STEP:
            state.value = -100
    if not any([state.content.endswith(end_str) for end_str in STEP_STOP_TOKENS]):
        state.value = -100
    if state.is_leaf and "The answer is:" not in state.content: # invalid
        state.value = -100

def search(query, tokenizer, actor, critic):
    tree = Tree(query)
    query = tree.question
    tree.root.value = get_step_scores([query], tokenizer, critic)[0]
    for search_iter in range(LIMIT):
        action = tree.select_next_node()
        if action.is_leaf:
            break
        traj = action.print_path()
        budget = min(N, max(2, math.ceil(math.log(1 - EXCEPTED) / action.get_depth() ** 0.5 / math.log(min(max(1 - action.value, 1e-2), 1 - 1e-2)))))
        trajs, anchors = [traj] * budget, [action] * budget
        with torch.no_grad():
            next_steps = get_next_steps(trajs, tokenizer, actor)
            next_values = get_step_scores([traj + next_step for traj, next_step in zip(trajs, next_steps)], tokenizer, critic)
        for anchor, traj, next_step, next_value in zip(anchors, trajs, next_steps, next_values):
            state = tree.add_node(next_step, next_value.item(), anchor, next_step.endswith(tokenizer.eos_token))
            if len(next_step) == 0 or len(next_step) > MAX_CHAR_PER_STEP or len(traj + next_step) > MAX_CHAR_PER_PATH:
                state.value = -1
            # print((search_iter, traj, next_step, next_value))
    
    # return the best traj
    terminal_nodes = [node for node in tree.all_nodes if node.is_leaf]
    if terminal_nodes:
        best_node = max(terminal_nodes, key=lambda x: x.value)
        return [best_node.print_path()]
    else:
        # expand the most valuable node
        action = tree.select_next_node()
        traj = action.print_path()
        with torch.no_grad():
            return [get_full_traj(traj, tokenizer, actor)[0]]
        # return None