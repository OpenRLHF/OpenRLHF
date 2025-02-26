import heapq
import numpy as np
import torch
import torch.distributed
from openrlhf.search_algorithm.search_utils import DEFAULT_N, DEFAULT_EXPAND_SIZE, DEFAULT_SEARCH_STEPS, trajectory_finished
from openrlhf.search_algorithm.beamsearch_efficient import get_next_steps_vllm, get_step_scores
from openrlhf.search_algorithm.bestofn import get_full_trajs_vllm

def search_vllm(queries, tokenizer, actor, critic=None, search_args=None):
    if not isinstance(queries, list):
        queries = [queries]

    beam_size = search_args.get("beam_size", DEFAULT_N)
    expand_size = search_args.get("expand_size", DEFAULT_EXPAND_SIZE)
    candidate_size = beam_size * expand_size
    search_steps = search_args.get("search_steps", DEFAULT_SEARCH_STEPS)

    full_trajs = get_full_trajs_vllm(queries, tokenizer, actor, search_args)
    finished_trajs = [[] for _ in queries]  # [batch_size, list of `(traj, score)` tuples]
    prev_beam = sum([[prompt] * beam_size for prompt in queries], [])  # [batch_size * beam_size]
    for _ in range(search_steps):
        # vllm --> next steps
        inputs =  sum([[traj] * expand_size for traj in prev_beam], [])  # [batch_size * beam_size * expand_size]
        trajs, cumulative_logprobs, avg_logprobs = get_next_steps_vllm(inputs, tokenizer, actor)

        # decide the scores for guidance
        if search_args["search_guidance"] == "cum_logp":
            scores = cumulative_logprobs
        elif search_args["search_guidance"] == "avg_logp":
            scores = avg_logprobs
        elif search_args["search_guidance"] == "critic":
            assert critic is not None
            torch.distributed.barrier()
            scores = get_step_scores(trajs, tokenizer, critic)
        else:
            raise Exception("Invalid compute_reward_strategy")

        prev_beam = []
        for i in range(0, len(inputs), candidate_size):
            n = i // candidate_size  # which query
            _trajs, _scores = [], []  # unfinished trajectories for the query
            for j in range(i, i + candidate_size):
                if trajectory_finished(trajs[j]):
                    finished_trajs[n].append((trajs[j], scores[j]))
                else:
                    _trajs.append(trajs[j])
                    _scores.append(scores[j])
            _indices = heapq.nlargest(beam_size, range(len(_scores)), key=_scores.__getitem__)
            _beam = [_trajs[idx] for idx in _indices]
            if len(_beam) < beam_size:
                _beam = _beam + [queries[n]] * (beam_size - len(_beam))
            prev_beam.extend(_beam)

    chosen_trajs = []
    for n, trajs in enumerate(finished_trajs):
        if trajs == []:
            chosen_trajs.append(full_trajs[n])
        else:
            trajs.sort(key = lambda x: x[1], reverse=True)
            chosen_trajs.append(trajs[0][0])
    return chosen_trajs
