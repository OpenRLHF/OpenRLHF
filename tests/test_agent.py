import asyncio
import importlib.util
import logging
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.test_experience import Experience


@pytest.mark.parametrize("query_count", [0, 1, 4, 6])
@pytest.mark.parametrize("rounds", [1, 6])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("replicas", [1, 3])
def test_remote_rewards_distribute_nonempty_shards(monkeypatch, query_count, rounds, concurrent, replicas):
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", SimpleNamespace(init_logger=logging.getLogger))
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_agent_test", Path(__file__).resolve().parents[1] / "openrlhf" / "utils" / "agent.py"
    )
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

    async def run():
        received = []
        requests_per_replica = Counter()

        async def get_reward(request):
            payload = await request.json()
            received.append(payload)
            requests_per_replica[request.match_info["replica"]] += 1
            # The shipped reward server requires a nonempty query batch.
            if not payload["query"]:
                raise web.HTTPBadRequest(text="empty query batch")
            return web.json_response({"rewards": [float(q) for q in payload["query"]]})

        app = web.Application()
        app.router.add_post("/{replica}", get_reward)
        async with TestServer(app) as server:
            executor = agent.SingleTurnAgentExecutor([str(server.make_url(f"/{i}")) for i in range(replicas)])
            queries = [str(i) for i in range(query_count * rounds)]
            prompts = [f"prompt-{i}" for i in range(query_count * rounds)]
            labels = [f"label-{i}" for i in range(query_count * rounds)]
            calls = [
                executor._fetch_rewards_via_http(
                    queries[i * query_count : (i + 1) * query_count],
                    prompts[i * query_count : (i + 1) * query_count],
                    labels[i * query_count : (i + 1) * query_count],
                )
                for i in range(rounds)
            ]
            results = await asyncio.gather(*calls) if concurrent else [await call for call in calls]

        assert all(payload["query"] for payload in received)
        counts = [requests_per_replica[str(i)] for i in range(replicas)]
        assert max(counts) - min(counts) <= 1
        assert [reward for batch in results for result in batch for reward in result["rewards"]] == list(
            range(query_count * rounds)
        )
        received.sort(key=lambda payload: int(payload["query"][0]))
        for key, expected in (("query", queries), ("prompts", prompts), ("labels", labels)):
            assert [value for payload in received for value in payload[key]] == expected

    asyncio.run(run())


@pytest.fixture
def agent_modules(monkeypatch):
    # Exercise the executor and reward pipeline without launching GPU workers.
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", SimpleNamespace(init_logger=logging.getLogger))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", SimpleNamespace(Experience=Experience))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.vllm_engine", MagicMock())
    monkeypatch.setitem(sys.modules, "ray", MagicMock())
    monkeypatch.setitem(sys.modules, "vllm", MagicMock())
    root = Path(__file__).resolve().parents[1]
    modules = []
    for path in ("utils/agent.py", "trainer/ppo_utils/samples_generator.py", "trainer/ppo_utils/length_penalty.py"):
        spec = importlib.util.spec_from_file_location(Path(path).stem, root / "openrlhf" / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.mark.parametrize(
    "finish_reasons,feedback,done_at_end,expected_truncated",
    [
        (("stop",), " ", True, False),
        (("length",), " ", True, True),
        (("stop", "stop"), " ", True, False),
        (("stop", "length"), " ", True, True),
        # Six prompt tokens + two action tokens leave 56 tokens for feedback.
        (("stop",), " " * 56, False, True),
        (("stop",), " " * 57, False, True),
        (("stop",), " " * 56, True, False),
        (("stop",), " " * 57, True, False),
        (("stop", "stop"), " " * 27, False, True),
        (("stop", "stop"), " " * 28, False, True),
    ],
)
@pytest.mark.parametrize("with_logprobs", [False, True])
@pytest.mark.parametrize("penalty_coef", [0.0, 0.5, -0.5])
def test_multiturn_truncation_reaches_reward_penalty(
    agent_modules, finish_reasons, feedback, done_at_end, expected_truncated, with_logprobs, penalty_coef
):
    agent, samples, penalties = agent_modules

    class Environment(agent.AgentInstanceBase):
        def __init__(self):
            self.steps = 0

        async def step(self, states):
            self.steps += 1
            return {
                "rewards": torch.tensor(1.0),
                "environment_feedback": feedback,
                "done": done_at_end and self.steps == len(finish_reasons),
            }

    class Engine:
        calls = 0

        async def generate(self, token_ids, params, **kwargs):
            reason = finish_reasons[self.calls]
            self.calls += 1
            count = params.max_tokens if reason == "length" else 2
            return SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        token_ids=[65] * count,
                        text="A" * count,
                        finish_reason=reason,
                        logprobs=[{65: SimpleNamespace(logprob=-0.1)}] * count if with_logprobs else None,
                    )
                ]
            )

    def tokenizer(text, **kwargs):
        return {"input_ids": torch.tensor([[ord(char) for char in text]])}

    engine = Engine()
    response = asyncio.run(
        agent.MultiTurnAgentExecutor(Environment).execute(
            "prompt",
            "label",
            SimpleNamespace(max_tokens=None, logprobs=1 if with_logprobs else None),
            64,
            tokenizer,
            engine,
        )
    )
    assert response.get("truncated", False) is expected_truncated
    assert engine.calls == len(finish_reasons)
    assert response["reward"] == len(finish_reasons)

    generator = samples.SamplesGenerator.__new__(samples.SamplesGenerator)
    experience = generator._process_response_into_experience(response, max_len=64)
    assert experience.truncated.item() is expected_truncated
    assert experience.sequences.shape[1] <= 64
    if with_logprobs:
        assert experience.rollout_log_probs.shape == experience.action_mask.shape
        mask = experience.action_mask.bool()
        assert torch.allclose(
            experience.rollout_log_probs[mask], torch.full_like(experience.rollout_log_probs[mask], -0.1)
        )
        assert (experience.rollout_log_probs[~mask] == 0).all()
    else:
        assert experience.rollout_log_probs is None
    for start, end in response["action_ranges"]:
        assert response["observation_tokens"][start:end] == [65] * (end - start)

    count = penalties.apply_stop_properly_penalty([experience], penalty_coef)
    expected_reward = float(len(finish_reasons))
    if expected_truncated:
        expected_reward = penalty_coef if penalty_coef < 0 else expected_reward * penalty_coef
    assert count == int(expected_truncated)
    assert experience.rewards.item() == expected_reward
