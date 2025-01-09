import torch
import re

# temporal hard code
LIMIT=32
N=8
BEAM=4
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
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, max_new_tokens=512,
                             temperature=TEMPERATURE, tokenizer=tokenizer, 
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    return sequences

def get_next_steps(traj, tokenizer, actor):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids, do_sample=True, stop_strings=END_OF_STEP, max_new_tokens=MAX_NEW_TOKENS,
                             temperature=TEMPERATURE, num_return_sequences=N // BEAM,
                             tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)
    sequences = tokenizer.batch_decode(outputs, keep_special_tokens=True)
    sequences = [clean_pad_token(seq[len(traj):], tokenizer.pad_token) for seq in sequences]
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
            for action in actions:
                traj = action.print_path()
                with torch.no_grad():
                    next_steps = get_next_steps(traj, tokenizer, actor)
                    next_values = get_step_scores([traj + next_step for next_step in next_steps], tokenizer, critic)
                for next_step, next_value in zip(next_steps, next_values):
                    state = tree.add_node(next_step, next_value.item(), action, next_step.endswith(tokenizer.eos_token))
                    if len(next_step) == 0 or len(next_step) > MAX_CHAR_PER_STEP or len(traj + next_step) > MAX_CHAR_PER_PATH:
                        state.value = -1
                    # print((search_iter, next_step, next_value))
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