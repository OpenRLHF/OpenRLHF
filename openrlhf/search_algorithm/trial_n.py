import re
import torch
import random
import ray
from typing import List
import numpy as np
from openrlhf.utils.remote_rm_utils import request_api_wrapper
from openrlhf.search_algorithm.bestofn import get_full_trajs_vllm
from openrlhf.search_algorithm.search_utils import DEFAULT_N, DEFAULT_TEMPERATURE, initialize_question_answer_map

"""
## Correct case
curl -X POST "http://localhost:1234/predict_de
tail" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "formal_statement": "theorem lean_workbook_plus_65302 (x y : ℝ) (h : x * y + y * x + x 
= 1) : 2 * x * y + x = 1",
        "proof": ":= by\n  rw [mul_comm] at h\n  linarith\n"
      }
    ]
  }'
{"rewards":[1.0],"details":[{"formal_statement":"theorem lean_workbook_plus_65302 (x y : ℝ) (h 
: x * y + y * x + x = 1) : 2 * x * y + x = 1","verification_time":121.857168674469,"errors":[],
"status":"unknown","complete":true,"pass":true}]}

## Wrong case
curl -X POST "http://localhost:1234/predict_detail" \ail" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {
        "formal_statement": "theorem lean_workbook_plus_65302 (x y : ℝ) (h : x * y + y * x + x = 1) : 2 * x * y + x = 1",
        "proof": ":= by\n  rw [mul_comm] at h\n  linarit\n"
      }
    ]
  }'
{"rewards":[-1.0],"details":[{"formal_statement":"theorem lean_workbook_plus_65302 (x y : ℝ) (h : x * y + y * x + x = 1) : 2 * x * y + x = 1","verification_time":18.54811954498291,"errors":[],"status":"unknown","complete":false,"pass":false}]}


"""
def verify_by_lean(api_url, statements: List[str], proofs: List[str]):
    queries = [{"formal_statement": s, "proof": p} for s, p in zip(statements, proofs)]
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)

def longest_partial_trajectory(traj, lean_outcome):
    pass

def search_vllm(queries, tokenizer, actor, critic=None, search_args=None):
    if not isinstance(queries, list):
        queries = [queries]

    trial_budget = search_args.get("trial_budget", DEFAULT_N)
    all_prompts, all_indices = zip(*enumerate(queries))
    sequences = ["" for _ in queries]
    for n in range(trial_budget):
        trajs, cumulative_logprobs, avg_logprobs = get_full_trajs_vllm(all_prompts, tokenizer, actor)
        lean_outcomes = verify_by_lean(trajs)

        all_prompts, all_indices = [], []
        for i, traj in enumerate(trajs):
            # TODO (mukai): you can decide the format of lean_outcomes, not necessarily str
            if lean_outcomes[i]['rewards'] == 1:
                sequences[all_indices[i]] = traj
            elif lean_outcomes[i]["rewards"] != 1:
                # TODO: get the longest partial trajectory before the compile error
                partial_traj = longest_partial_trajectory()
                all_prompts.append(partial_traj)
                all_indices.append(all_indices[i])
                sequences[all_indices[i]] = partial_traj
            else:
                all_prompts.append(queries[i])
                all_indices.append(all_indices[i])
                sequences[all_indices[i]] = traj

    return sequences
