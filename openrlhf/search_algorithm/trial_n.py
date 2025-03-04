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
{"rewards":[-1.0],"details":[{"formal_statement":"theorem lean_workbook_plus_65302 (x y : ℝ) (h : x * y + y * x + x = 1) : 2 * x * y + x = 1","verification_time":17.850919723510742,"errors":[],"status":"unknown","complete":false,"pass":false,"output":"test_proof.lean:14:3: error: unknown tactic\ntest_proof.lean:12:93: error: unsolved goals\nx y : ℝ\nh :                           OpenRLHF  data  lean_proof y * x + y * x + x = 1\n⊢ 2 * x * y + x = 1\n","system_messages":null,"error_positions":[{"line":3,"file_line":14,"column":3,"position":30,"me9543838501,ssage":"unknown tactic","content":"  linarit"},{"line":1,"file_line":12tmpn12rn8zl.lean:14,"column":93,"position":93,"message":"unsolved goals","content":":= by"s\nx y : ℝ\nh : y *}],"proof_segments":[]}]}


"""
def verify_by_lean(api_url, statements: List[str], proofs: List[str]):
    queries = [{"formal_statement": s, "proof": p} for s, p in zip(statements, proofs)]
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)

def longest_partial_trajectory(traj: str, error_positions: List[dict]) -> str:
    """Returns the proof content up to the first error position"""
    if not error_positions:
        return traj
    
    lines = traj.split('\n')
    # Get the first error line
    first_error = min(error_positions, key=lambda x: (x['line'], x['column']))
    error_line = first_error['line']
    error_col = first_error['column']
    
    # Keep lines before error
    valid_lines = lines[:error_line-1]
    
    # For the error line, keep content before error column
    if error_line <= len(lines):
        error_line_content = lines[error_line-1][:error_col-1]
        valid_lines.append(error_line_content)
    
    return '\n'.join(valid_lines)

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
                partial_traj = longest_partial_trajectory(
                    traj, lean_outcomes[i]["details"][0]['error_positions']
                )
                all_prompts.append(partial_traj)
                all_indices.append(all_indices[i])
                sequences[all_indices[i]] = partial_traj
            else:
                all_prompts.append(queries[i])
                all_indices.append(all_indices[i])
                sequences[all_indices[i]] = traj

    return sequences
