
import jsonlines

DEFAULT_N = 8
DEFAULT_BEAM_SIZE = 1
DEFAULT_TEMPERATURE = 1
DEFAULT_MAX_LENGTH = 1024
DEFAULT_MAX_STEP_LENGTH = 1024
strategy = "best"

#### Search Tree Data Structure ####
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

def initialize_question_answer_map():
    data_fpath_list = [
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/train.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/test.jsonl"
    ]

    dataset = []
    for data_fpath in data_fpath_list:
        with jsonlines.open(data_fpath, 'r') as reader:
            for item in reader:
                if "gsm8k" in data_fpath:
                    question = item["question"]
                    answer = item["answer"].split("####")[-1].strip()
                    dataset.append({"question": question, "answer": answer, "type": "gsm8k"})
                else:
                    question = item["problem"]
                    answer = item["answer"]
                    dataset.append({"question": question, "answer": answer, "type": "math"})

    answer_dict = {item["question"].strip(): {"ref": item["answer"], "type": item["type"]} for item in dataset}
    return answer_dict
