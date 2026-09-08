import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest.mark.parametrize("query_count", [0, 1, 4, 6])
def test_remote_rewards_skip_empty_shards(monkeypatch, query_count):
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", SimpleNamespace(init_logger=logging.getLogger))
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_agent_test", Path(__file__).resolve().parents[1] / "openrlhf" / "utils" / "agent.py"
    )
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

    async def run():
        received = []

        async def get_reward(request):
            payload = await request.json()
            received.append(payload)
            # The shipped reward server requires a nonempty query batch.
            if not payload["query"]:
                raise web.HTTPBadRequest(text="empty query batch")
            return web.json_response({"rewards": [float(q) for q in payload["query"]]})

        app = web.Application()
        app.router.add_post("/{replica}", get_reward)
        async with TestServer(app) as server:
            executor = agent.SingleTurnAgentExecutor([str(server.make_url(f"/{i}")) for i in range(3)])
            queries = [str(i) for i in range(query_count)]
            prompts = [f"prompt-{i}" for i in range(query_count)]
            labels = [f"label-{i}" for i in range(query_count)]
            results = await executor._fetch_rewards_via_http(queries, prompts, labels)

        assert all(payload["query"] for payload in received)
        assert [reward for result in results for reward in result["rewards"]] == list(range(query_count))
        received.sort(key=lambda payload: int(payload["query"][0]))
        for key, expected in (("query", queries), ("prompts", prompts), ("labels", labels)):
            assert [value for payload in received for value in payload[key]] == expected

    asyncio.run(run())
