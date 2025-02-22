import os
import time

import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

from openrlhf.remote_rm.ds_prover.utils import AttrDict, MODEL_FORMAT


class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_path, task_queue, request_statuses, lock, args):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_path = model_path
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            n=1,
        )
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']

    def run(self):
        seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000
        os.environ['LOCAL_RANK'] = str(self.local_rank)
        llm = LLM(model=self.model_path, max_num_batched_tokens=8192, seed=seed, trust_remote_code=True)
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            model_inputs = [
                ''.join([
                    item.get('_extra_header', str()),
                    self.prompt_func(item),
                    item.get('_extra_prompt', str()),
                ]) for _, _, item in inputs
            ]
            model_outputs = llm.generate(
                model_inputs,
                self.sampling_params,
                use_tqdm=False,
            )
            outputs = [self.output_func(_output.outputs[0].text) for _output in model_outputs]
            with self.lock:
                for (_, request_id, _), output in zip(inputs, outputs):
                    self.request_statuses[request_id] = output
