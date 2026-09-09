import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Avoid openrlhf.utils.__init__ (sympy/pylatexenc) so these stay CPU-light.
_UTILS = Path(__file__).resolve().parents[1] / "openrlhf" / "utils"
if "openrlhf.utils" not in sys.modules:
    _utils_pkg = types.ModuleType("openrlhf.utils")
    _utils_pkg.__path__ = [str(_UTILS)]
    sys.modules["openrlhf.utils"] = _utils_pkg


def test_serve_rm_default_host_is_loopback():
    pytest.importorskip("fastapi")
    from openrlhf.cli.serve_rm import build_parser

    assert build_parser().get_default("host") == "127.0.0.1"


def test_missing_key_rejected_when_server_has_key():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from openrlhf.cli.serve_rm import create_app

    class FakeRewardModel:
        def get_reward(self, queries):
            return [1.0] * len(queries)

    client = TestClient(create_app(FakeRewardModel(), api_key="secret"))
    response = client.post("/get_reward", json={"query": ["hello"]})
    assert response.status_code == 401


def test_wrong_length_rewards_rejected_on_client():
    from openrlhf.utils.agent import SingleTurnAgentExecutor

    executor = SingleTurnAgentExecutor("http://127.0.0.1:9/get_reward")

    class FakeResponse:
        def raise_for_status(self):
            return None

        async def json(self):
            return {"rewards": [1.0, 2.0], "scores": [1.0, 2.0]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            return FakeResponse()

    async def run():
        with patch("openrlhf.utils.agent.aiohttp.ClientSession", return_value=FakeSession()):
            await executor._fetch_rewards_via_http(["q"], ["p"], ["l"])

    with pytest.raises(ValueError, match="malformed rewards"):
        asyncio.run(run())
