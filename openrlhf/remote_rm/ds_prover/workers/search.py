import os
import time
import copy
import json
import pickle
from pathlib import Path

import torch
import torch.multiprocessing as mp
import numpy as np

from openrlhf.remote_rm.ds_prover.utils import AttrDict, get_datetime


class SearchProcess(mp.Process):
    def __init__(self, idx, log_dir, tokenizer_path, scheduler, data_loader, cfg):
        self.idx = idx
        self.log_dir = Path(log_dir)
        self.scheduler = scheduler
        self.data_loader = data_loader
        super().__init__()

        self._current_prob_idx = None
        sampler_cls = cfg.sampler['algorithm']
        self.sampler = sampler_cls(
            scheduler=self.scheduler,
            tokenizer_path=tokenizer_path,
            process_print=self.process_print,
            cfg=AttrDict({
                **cfg.sampler,
                'mode': cfg.model_args.mode,
                'max_tokens': cfg.model_args.max_tokens,
            })
        )
    
    def _post_process(self, data: dict, proof_code: str):
        header = data.get('header', str())
        tailer = data.get('tailer', str())
        formal_statement = data['formal_statement']
        return dict(
            statement_proposal=f'{header}{formal_statement}{proof_code}{tailer}',
            proof_code=proof_code,
        )
    
    def process_print(self, logs, **kwargs):
        print('Process ID: {:3d}    Problem ID: {}    {}'.format(self.idx, self._current_prob, logs), **kwargs)

    def run(self):
        while True:
            prob_idx, prob_runname, data = self.data_loader.get()
            if prob_idx is None: break
            
            sample_start_time = time.time()
            # build a yield-iterator object to generate samples
            self._current_prob = f'{prob_idx}_{prob_runname}'
            prob_log_dir = self.log_dir / self._current_prob
            os.makedirs(prob_log_dir, exist_ok=True)
            sample_generator = self.sampler.sample(
                data=data,
                prob_log_dir=prob_log_dir,
            )
            # submit requests to the verification server when receiving from the generator
            candidate_list, info_list, request_id_list = [], [], []
            for sample, info in sample_generator:
                candidate = self._post_process(data, sample)
                candidate_list.append(candidate)
                info_list.append(copy.deepcopy(info))
                request_id = self.scheduler.verifier_submit_request(candidate['statement_proposal'])
                request_id_list.append(request_id)
            sample_timecost = time.time() - sample_start_time

            verification_start_wait_time = time.time()
            result_list = self.scheduler.verifier_get_all_request_outputs(request_id_list)
            verification_timecost = time.time() - verification_start_wait_time

            success_count = sum([int(result['complete']) for result in result_list])
            self.process_print('Success: {} / {}    Generation: {:.2f} secs    Verfication: {:.2f} secs'.format(
                success_count, len(candidate_list), sample_timecost, verification_timecost,
            ))
            

            summary_dict = dict(success=[], failure=[])
            for _idx, (candidate, result, info) in enumerate(zip(candidate_list, result_list, info_list)):
                success_flag = 'success' if result['complete'] else 'failure'
                summary_dict[success_flag].append(dict(
                    problem_name=data['name'],
                    sample_info=info,
                    formal_statement=data['formal_statement'],
                    proof_code=candidate['proof_code'],
                    result=result,
                ))
            
            prob_name, run_id = prob_runname.split('/')
            prob_log_basedir = self.log_dir / f'{prob_idx}_{data["name"]}'
            log_tag = f'{self.sampler.algorithm_name}-{run_id}'
            # separately save success and failure results
            for success_flag, summary_list in summary_dict.items():
                if len(summary_list) > 0:
                    with open(prob_log_basedir / f'{success_flag}-{log_tag}-{get_datetime()}.pkl', 'wb') as pkl_f:
                        pickle.dump(summary_list, pkl_f)
            # create a 'finished' placeholder
            with open(prob_log_dir / self.data_loader.finished_flag_filename, 'w') as f:
                print('finished', file=f)
