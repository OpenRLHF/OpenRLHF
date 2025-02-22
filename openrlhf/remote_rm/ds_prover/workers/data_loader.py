import os
import copy

import torch
import torch.multiprocessing as mp

from openrlhf.remote_rm.ds_prover.utils import load_jsonl_objects


class DataLoader(object):
    def __init__(self, data_path, data_split, data_repeat, node_rank, world_size, log_dir):
        self.manager = mp.Manager()
        self.queue = self.manager.Queue()
        self.lock = mp.Lock()
        self.finished_flag_filename = 'finished_running.txt'

        done_set = set()
        for dirname in os.listdir(log_dir):
            run_dir = os.path.join(log_dir, dirname)
            if os.path.isdir(run_dir):
                for subdirname in os.listdir(run_dir):
                    if subdirname.startswith('run') and os.path.exists(os.path.join(run_dir, subdirname, self.finished_flag_filename)):
                        done_set.add(os.path.join(dirname, subdirname))

        todo_count = 0
        if isinstance(data_split, str):
            data_split = [data_split]
        dataset = load_jsonl_objects(data_path)
        for _repeat in range(data_repeat):
            for prob_idx, prob in enumerate(dataset):
                prob_runname = os.path.join(prob['name'], f'run{_repeat}')
                if f'{prob_idx}_{prob_runname}' in done_set:
                    continue
                if data_split is not None and prob['split'] not in data_split:
                    continue
                if todo_count % world_size == node_rank:
                    self.queue.put((prob_idx, prob_runname, copy.deepcopy(prob)))
                todo_count += 1
        print('Number of TODO Problems: {}'.format(self.queue.qsize()))
    
    def size(self):
        return self.queue.qsize()
    
    def get(self):
        with self.lock:
            if self.queue.qsize() > 0:
                return self.queue.get()
        return None, None, None
