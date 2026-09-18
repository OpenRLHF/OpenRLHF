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


@pytest.mark.parametrize("finish_reasons", [("stop",), ("length",), ("stop", "stop"), ("stop", "length")])
@pytest.mark.parametrize("with_logprobs", [False, True])
@pytest.mark.parametrize("penalty_coef", [0.0, 0.5, -0.5])
def test_multiturn_truncation_reaches_reward_penalty(agent_modules, finish_reasons, with_logprobs, penalty_coef):
    agent, samples, penalties = agent_modules

    class Environment(agent.AgentInstanceBase):
        def __init__(self):
            self.steps = 0

        async def step(self, states):
            self.steps += 1
            return {
                "rewards": torch.tensor(1.0),
                "environment_feedback": " ",
                "done": self.steps == len(finish_reasons),
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
    expected_truncated = finish_reasons[-1] == "length"
    assert response.get("truncated", False) is expected_truncated
    assert engine.calls == len(finish_reasons)
    assert response["reward"] == len(finish_reasons)

    generator = samples.SamplesGenerator.__new__(samples.SamplesGenerator)
    experience = generator._process_response_into_experience(response, max_len=64)
    assert experience.truncated.item() is expected_truncated
    assert experience.sequences.shape[1] <= 64
    if with_logprobs:
        assert experience.rollout_log_probs.shape == experience.action_mask.shape

    count = penalties.apply_stop_properly_penalty([experience], penalty_coef)
    expected_reward = float(len(finish_reasons))
    if expected_truncated:
        expected_reward = penalty_coef if penalty_coef < 0 else expected_reward * penalty_coef
    assert count == int(expected_truncated)
    assert experience.rewards.item() == expected_reward


@pytest.fixture
def openai_executor(agent_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf.utils.agent", agent_modules[0])
    pytest.importorskip("fastapi")
    for name in ("uvicorn", "openai"):
        monkeypatch.setitem(sys.modules, name, MagicMock())
    path = Path(__file__).resolve().parents[1] / "examples/python/agent_func_openai_server_executor.py"
    spec = importlib.util.spec_from_file_location("_openai_executor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AgentExecutor()


@pytest.mark.parametrize("image_kwargs", [{}, {"images": None}, {"images": []}])
@pytest.mark.parametrize("num_samples", [1, 4])
def test_openai_executor_text_rollouts(openai_executor, monkeypatch, image_kwargs, num_samples):
    executor = openai_executor
    executor.client = object()
    executor._token_traces = {}
    executor._session_buffers = {}
    session_ids = []

    async def run_agent(prompt, label, session_id):
        session_ids.append(session_id)
        executor._token_traces[session_id] = [
            {"prompt_token_ids": [10, 20], "completion_token_ids": [30], "finish_reason": "stop", "logprobs": [-0.5]}
        ]
        executor._session_buffers[session_id] = [10, 20, 30]
        await asyncio.sleep(0)
        return {"reward": 1.0, "scores": 1.0}

    monkeypatch.setattr(executor, "run_agent", run_agent)

    async def run():
        return await asyncio.gather(
            *[
                executor.execute("prompt", "label", SimpleNamespace(logprobs=1), 64, None, None, **image_kwargs)
                for _ in range(num_samples)
            ]
        )

    results = asyncio.run(run())
    assert len(set(session_ids)) == num_samples
    for result in results:
        assert result["prompt"] == "prompt"
        assert result["label"] == "label"
        assert result["observation_tokens"] == [10, 20, 30]
        assert result["action_ranges"] == [(2, 3)]
        assert result["rollout_log_probs"] == [0.0, 0.0, -0.5]
        assert result["reward"] == result["scores"] == 1.0
        assert result["truncated"] is False
    assert not executor._token_traces
    assert not executor._session_buffers


@pytest.mark.parametrize("images", ["image.png", ["image.png"]])
def test_openai_executor_rejects_images_before_starting_server(openai_executor, monkeypatch, images):
    init_server = MagicMock()
    monkeypatch.setattr(openai_executor, "_init_server", init_server)
    with pytest.raises(ValueError, match="only supports text"):
        asyncio.run(openai_executor.execute("prompt", "label", None, 64, None, None, images=images))
    init_server.assert_not_called()


@pytest.mark.parametrize(
    "request_budget, default_budget, expected",
    [
        ({}, None, 6),
        ({"max_tokens": None}, 8, 6),
        ({}, 3, 3),
        ({"max_tokens": 2}, None, 2),
        ({"max_tokens": 20}, None, 6),
    ],
)
@pytest.mark.parametrize("max_length", [8, 2])
def test_openai_executor_http_token_budget(
    openai_executor, monkeypatch, request_budget, default_budget, expected, max_length
):
    httpx = pytest.importorskip("httpx")
    executor = openai_executor
    globals_dict = executor._start_server.__func__.__globals__
    from fastapi import FastAPI

    app = FastAPI()
    monkeypatch.setitem(globals_dict, "FastAPI", lambda: app)
    monkeypatch.setitem(globals_dict, "SamplingParams", SimpleNamespace)
    monkeypatch.setitem(globals_dict, "urlopen", MagicMock())
    monkeypatch.setattr(globals_dict["threading"], "Thread", MagicMock())
    executor.model_name = "test-model"
    executor.host, executor.port = "127.0.0.1", 0
    executor.max_length = max_length
    executor.sampling_params = SimpleNamespace(max_tokens=default_budget, temperature=1.0, top_p=1.0)
    executor.hf_tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [10, 20])
    executor._session_buffers = {}
    executor._token_traces = {}
    budgets = []

    async def generate(token_ids, params):
        budgets.append(params.max_tokens)
        return SimpleNamespace(outputs=[SimpleNamespace(text="OK", token_ids=[30], finish_reason="stop")])

    executor.llm_engine = SimpleNamespace(generate=generate)
    executor._start_server()

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hello"}], **request_budget}
            )

    response = asyncio.run(run())
    if max_length == 2:
        assert response.status_code == 400
        assert not budgets
    else:
        assert response.status_code == 200
        assert budgets == [expected]
        assert response.json()["token_ids"] == {"prompt_token_ids": [10, 20], "completion_token_ids": [30]}
