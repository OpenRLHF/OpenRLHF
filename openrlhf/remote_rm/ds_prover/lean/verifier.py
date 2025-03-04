import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
from pprint import pprint

import numpy as np

from openrlhf.remote_rm.ds_prover.lean.ast_parser import lean4_parser
from openrlhf.remote_rm.ds_prover.workers import ProcessScheduler
from openrlhf.remote_rm.ds_prover.utils import AttrDict


HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'

DEFAULT_LEAN_WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mathlib4')


def verify_lean4_file(code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=True, timeout=300, allTactics=False, ast=False, premises=False, tactics=False):
    test_file = os.path.join(lean_workspace, 'test_proof.lean')
    try:
        full_code = f"""
import Mathlib.Tactic.Basic
import Mathlib.Tactic.NormNum

{code}
"""
        with open(test_file, 'w') as f:
            f.write(full_code)
        
        if verbose:
            print(f"Created test file: {test_file}")
            print(f"Code content:\n{full_code}")
        
        try:
            outputs = subprocess.run(
                [lake_path, 'env', 'lean', os.path.basename(test_file)],
                cwd=lean_workspace,
                capture_output=True,
                timeout=timeout
            )
            
            if verbose:
                print(f"Command output:\nstdout: {outputs.stdout}\nstderr: {outputs.stderr}")
            
            result = {
                'pass': outputs.returncode == 0,
                'complete': outputs.returncode == 0,
                'system_messages': outputs.stderr.decode(),
                'output': outputs.stdout.decode()
            }
            
        except Exception as e:
            result = {
                'pass': False,
                'complete': False,
                'system_messages': str(e)
            }
    finally:
        if not verbose:
            os.remove(test_file)
            
    return result


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args

        self.timeout = extra_args.get('timeout', 300)
        self.memory_limit = extra_args.get('memory_limit', -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)
    
    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if 'timeout' not in task:
                    task['timeout'] = self.timeout
                result = verify_lean4_file(**task)
                if len(result['system_messages']) > 0:
                    retry_start_time = time.time()
                    while ('lean::exception: failed to create thread' in result['system_messages'] or
                           'std::bad_alloc: std::bad_alloc' in result['system_messages'] or
                           'Cannot allocate memory' in result['system_messages']) \
                          and time.time() - retry_start_time < self.timeout:
                        time.sleep(0.1)
                        result = verify_lean4_file(**task)
                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1


class Lean4ServerScheduler(ProcessScheduler):
    def __init__(self, max_concurrent_requests=64, timeout=300, memory_limit=-1, name='verifier'):
        super().__init__(batch_size=1, name=name)
        
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f'Complete launching {len(self.processes)} LeanServerProcesses')

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(['killall', 'repl', f'--older-than={int(self.timeout) + 10}s'], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f'All {len(self.processes)} LeanServerProcesses stopped')


if __name__ == '__main__':
    code = open('mathlib4/.lake/packages/REPL/test/aime_1983_p9.code.in').read()
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')
    request_id_list = lean4_scheduler.submit_all_request([dict(code=code, ast=True, tactics=True)])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()
    pprint(outputs_list)
