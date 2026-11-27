"""
Agent Executor with OpenAI-Compatible vLLM Server.

Wraps vLLM as a local OpenAI-compatible HTTP server, collecting token-level
traces (IDs, logprobs) for RL training. Supports multi-turn agents via the
overridable run_agent() method.

Sessions preserve the exact sampled tokens, including earlier reasoning that
chat templates may remove from history. Requests must extend the same history
with the returned assistant content and new feedback; edits need a new session.

Usage:
    python -m openrlhf.cli.train_ppo \
        --train.agent_func_path examples/python/agent_func_openai_server_executor.py ...
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from copy import deepcopy
from urllib.request import urlopen
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI
from vllm import SamplingParams

from openrlhf.utils.agent import AgentExecutorBase

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    return "\n".join(f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in messages) + "\n"


class AgentExecutor(AgentExecutorBase):
    """
    Wraps vLLM as an OpenAI-compatible server for RL training.

    Exposes /v1/chat/completions with token-level traces. Multi-turn sessions
    accumulate traces keyed by session_id. execute() returns a single dict
    with the stitched RL training sample.
    """

    def _init_server(self, llm_engine, hf_tokenizer):
        self.host = "127.0.0.1"
        self.port = _find_open_port()
        self.llm_engine = llm_engine
        self.hf_tokenizer = hf_tokenizer
        self.model_name = getattr(hf_tokenizer, "name_or_path", "policy-model")
        self._token_traces = {}  # session_id -> [trace, ...]
        self._session_buffers = {}  # session_id -> list[int]

        self._start_server()
        self.client = AsyncOpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key="EMPTY",
        )
        logger.info(
            f"OpenAI-compatible server ready at http://{self.host}:{self.port}/v1 " f"(model={self.model_name})"
        )

    def _start_server(self):
        app = FastAPI()
        executor = self

        @app.get("/health")
        async def health():
            return {"status": "healthy", "model": executor.model_name}

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{"id": executor.model_name, "object": "model", "owned_by": "openrlhf"}],
            }

        @app.post("/tokenize")
        async def tokenize(request: Request):
            body = await request.json()
            text = _apply_chat_template(executor.hf_tokenizer, body.get("messages", []), add_generation_prompt=False)
            token_ids = executor.hf_tokenizer.encode(text)
            return {"tokens": token_ids, "count": len(token_ids), "max_model_len": executor.max_length}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            messages = body.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            session_id = body.get("session_id", uuid4().hex)
            max_tokens = body.get("max_tokens", executor.sampling_params.max_tokens)
            temperature = body.get("temperature", executor.sampling_params.temperature)
            top_p = body.get("top_p", executor.sampling_params.top_p)
            include_logprobs = body.get("logprobs", False)
            top_logprobs = body.get("top_logprobs", 1)

            prompt_text = _apply_chat_template(executor.hf_tokenizer, messages, add_generation_prompt=True)
            full_prompt_ids = executor.hf_tokenizer.encode(prompt_text, add_special_tokens=False)

            buffer = executor._session_buffers.get(session_id)
            if buffer is None or full_prompt_ids[: len(buffer)] == buffer:
                prompt_token_ids = full_prompt_ids
            else:
                previous = executor._token_traces[session_id][-1]
                assistant_index = len(previous["messages"])
                if messages[:assistant_index] != previous["messages"] or len(messages) <= assistant_index:
                    raise HTTPException(status_code=400, detail="Session history changed; use a new session_id")
                assistant = messages[assistant_index]
                if (
                    assistant.get("role") != "assistant"
                    or assistant.get("content") != previous["completion_text"]
                    or assistant.get("tool_calls")
                    or assistant.get("reasoning_content")
                    or assistant.get("name")
                ):
                    raise HTTPException(status_code=400, detail="Replay the returned assistant content unchanged")

                # Locate the last reply's end in this render, without assuming a
                # model-specific delimiter or re-encoding any sampled tokens.
                marker = f"__openrlhf_boundary_{uuid4().hex}__"
                marked_messages = deepcopy(messages)
                marked_messages[assistant_index]["content"] += marker
                marked_text = _apply_chat_template(executor.hf_tokenizer, marked_messages)
                if marked_text.count(marker) != 1:
                    raise HTTPException(status_code=400, detail="Chat template cannot locate the reply boundary")
                prefix, suffix = marked_text.split(marker)
                if prompt_text not in (prefix + suffix, prefix.rstrip() + suffix):
                    raise HTTPException(status_code=400, detail="Chat template changed around the reply boundary")
                delta_ids = executor.hf_tokenizer.encode(suffix, add_special_tokens=False)
                eos_id = executor.hf_tokenizer.eos_token_id
                sampled_ids = previous["completion_token_ids"]
                if not delta_ids or eos_id is None or delta_ids[0] != eos_id:
                    raise HTTPException(
                        status_code=400, detail="Chat template must close the reply with its EOS token"
                    )
                if previous["finish_reason"] == "stop" and (not sampled_ids or sampled_ids[-1] != eos_id):
                    raise HTTPException(status_code=400, detail="Cannot identify the sampled reply's EOS token")
                # A sampled EOS already closes the reply. After a length stop,
                # the template's closure instead belongs to observation.
                if sampled_ids and sampled_ids[-1] == eos_id:
                    delta_ids = delta_ids[1:]
                prompt_token_ids = buffer + delta_ids

            remaining = executor.max_length - len(prompt_token_ids)
            if remaining <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt ({len(prompt_token_ids)}) exceeds max_length ({executor.max_length})",
                )

            sp = SamplingParams(
                max_tokens=min(max_tokens, remaining),
                temperature=temperature,
                top_p=top_p,
                logprobs=top_logprobs if include_logprobs else None,
            )
            output = await executor.llm_engine.generate(prompt_token_ids, sp)
            gen = output.outputs[0]
            gen_text = gen.text or ""
            gen_ids = list(gen.token_ids)

            # Build OpenAI-format logprobs and trace logprobs in one pass
            logprobs_content = None
            trace_logprobs = None
            if include_logprobs and gen.logprobs:
                logprobs_content = []
                trace_logprobs = []
                for i, lp_dict in enumerate(gen.logprobs):
                    tid = gen_ids[i]
                    token_lp = lp_dict.get(tid)
                    lp_val = token_lp.logprob if token_lp else 0.0
                    trace_logprobs.append(lp_val)
                    logprobs_content.append(
                        {
                            "token": executor.hf_tokenizer.decode([tid]),
                            "logprob": lp_val,
                            "bytes": None,
                            "token_id": tid,
                            "top_logprobs": [
                                {"token": executor.hf_tokenizer.decode([t]), "token_id": t, "logprob": lp.logprob}
                                for t, lp in lp_dict.items()
                            ],
                        }
                    )

            trace = {
                "messages": deepcopy(messages),
                "completion_text": gen_text,
                "prompt_token_ids": prompt_token_ids,
                "completion_token_ids": gen_ids,
                "finish_reason": gen.finish_reason or "stop",
            }
            if trace_logprobs is not None:
                trace["logprobs"] = trace_logprobs
            executor._session_buffers[session_id] = prompt_token_ids + gen_ids
            executor._token_traces.setdefault(session_id, []).append(trace)

            return {
                "id": f"chatcmpl-{uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", executor.model_name),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": gen_text},
                        "finish_reason": gen.finish_reason or "stop",
                        "logprobs": {"content": logprobs_content} if logprobs_content else None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(gen_ids),
                    "total_tokens": len(prompt_token_ids) + len(gen_ids),
                },
                "token_ids": {
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": gen_ids,
                },
            }

        thread = threading.Thread(
            target=lambda: uvicorn.run(app, host=executor.host, port=executor.port, log_level="info", loop="asyncio"),
            daemon=True,
        )
        thread.start()

        for _ in range(60):
            try:
                urlopen(f"http://{self.host}:{self.port}/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("vLLM OpenAI server failed to start within 60s")

    async def run_agent(self, prompt: str, label: str, session_id: str):
        """
        Override for multi-turn workflows. Default: single-turn completion.

        Use self.client with extra_body={"session_id": session_id} so traces
        are accumulated for RL training.
        """
        await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.sampling_params.max_tokens,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            logprobs=self.sampling_params.logprobs is not None,
            top_logprobs=self.sampling_params.logprobs or 1,
            extra_body={"session_id": session_id},
        )
        return {}

    def _stitch_session(self, session_id, prompt, label, sampling_params, agent_result):
        """Stitch token traces into a single RL training sample (prefix stability assumed)."""
        traces = self._token_traces.pop(session_id, [])
        self._session_buffers.pop(session_id, None)
        if not traces:
            raise RuntimeError(f"No token traces for session {session_id}")

        # Verify prefix consistency: each trace's prompt must extend the previous trace's full sequence
        for i in range(1, len(traces)):
            prev_full = traces[i - 1]["prompt_token_ids"] + traces[i - 1]["completion_token_ids"]
            curr_prompt = traces[i]["prompt_token_ids"]
            assert len(curr_prompt) >= len(prev_full) and curr_prompt[: len(prev_full)] == prev_full, (
                f"Session {session_id}: prefix break at call {i} — "
                f"prev_full({len(prev_full)}) not a prefix of curr_prompt({len(curr_prompt)})"
            )

        last = traces[-1]
        obs_tokens = last["prompt_token_ids"] + last["completion_token_ids"]
        action_ranges = [
            (len(t["prompt_token_ids"]), len(t["prompt_token_ids"]) + len(t["completion_token_ids"])) for t in traces
        ]

        rollout_log_probs = None
        if sampling_params.logprobs is not None:
            rollout_log_probs = [0.0] * len(obs_tokens)
            for t in traces:
                if "logprobs" in t:
                    start = len(t["prompt_token_ids"])
                    for j, lp in enumerate(t["logprobs"]):
                        if start + j < len(rollout_log_probs):
                            rollout_log_probs[start + j] = lp

        return {
            "prompt": prompt,
            "label": label,
            "observation_tokens": obs_tokens,
            "action_ranges": action_ranges,
            "rollout_log_probs": rollout_log_probs,
            "truncated": last["finish_reason"] == "length",
            "reward": agent_result.get("reward"),
            "scores": agent_result.get("scores"),
            "extra_logs": agent_result.get("extra_logs", {}),
        }

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine):
        """Execute an agent episode and return RL training samples."""
        self.sampling_params = sampling_params
        self.max_length = max_length
        if not hasattr(self, "client"):
            self._init_server(llm_engine, hf_tokenizer)

        session_id = uuid4().hex
        try:
            result = await self.run_agent(prompt, label, session_id)
            return self._stitch_session(session_id, prompt, label, sampling_params, result or {})
        finally:
            self._token_traces.pop(session_id, None)
            self._session_buffers.pop(session_id, None)
