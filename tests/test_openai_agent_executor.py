from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from openai import BadRequestError, InternalServerError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTOR_SOURCE = ROOT / "examples" / "python" / "agent_func_openai_server_executor.py"
SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")

HISTORY_TEMPLATE = """
{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'tool' %}
{{- '<|im_start|>user\\n<tool_response>\\n' + message['content'] + '\\n</tool_response><|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and not enable_thinking %}
{{- '<think>\\n\\n</think>\\n\\n' }}
{%- else %}
{{- '<think>\\n' }}
{%- endif %}
{%- endif %}
"""

STABLE_TEMPLATE = """
{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}
{%- elif message['role'] == 'tool' %}
{{- '<|im_start|>user\\n<tool_response>\\n' + message['content'] + '\\n</tool_response><|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}
"""


class _ScriptedEngine:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, prompt_token_ids, sampling_params):
        self.calls.append((list(prompt_token_ids), deepcopy(sampling_params)))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        token_ids = list(output["token_ids"])
        assert len(token_ids) <= sampling_params.max_tokens
        logprobs = [{token_id: SimpleNamespace(logprob=lp)} for token_id, lp in zip(token_ids, output["logprobs"])]
        return SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=token_ids,
                    text=output["text"],
                    finish_reason=output["finish_reason"],
                    logprobs=logprobs,
                )
            ]
        )


class _TemplateTokenizer:
    def __init__(self, tokenizer, enable_thinking, template):
        self._tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self._tokenizer.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("enable_thinking", self.enable_thinking)
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def _build_bpe_tokenizer(enable_thinking, template):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(ByteLevel.alphabet())
    vocab = {token: index for index, token in enumerate(alphabet)}
    for token in ("<unk>", "oo", "foo", "<|endoftext|>"):
        if token not in vocab:
            vocab[token] = len(vocab)
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[("o", "o"), ("f", "oo")], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=list(SPECIAL_TOKENS[:1] + SPECIAL_TOKENS[2:]),
    )
    fast.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKENS[1]]})
    return _TemplateTokenizer(fast, enable_thinking, template)


def _make_tokenizer(enable_thinking, template_kind):
    template = HISTORY_TEMPLATE if template_kind == "history" else STABLE_TEMPLATE
    tokenizer_dir = os.environ.get("OPENRLHF_TEST_TOKENIZER_DIR")
    if tokenizer_dir:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if template_kind == "stable":
            tokenizer.chat_template = template
        return _TemplateTokenizer(tokenizer, enable_thinking, tokenizer.chat_template)
    return _build_bpe_tokenizer(enable_thinking, template)


def _load_executor(monkeypatch):
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    source = Path(os.environ.get("OPENRLHF_TEST_EXECUTOR_SOURCE", DEFAULT_EXECUTOR_SOURCE))
    name = f"_openrlhf_executor_test_{id(source)}_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _patch_server(monkeypatch, module):
    state = {"app": None, "ready": threading.Event()}

    def fake_run(app, **_kwargs):
        state["app"] = app
        state["ready"].set()

    def fake_urlopen(*_args, **_kwargs):
        assert state["ready"].wait(2), "uvicorn app was not captured"
        return SimpleNamespace()

    real_async_openai = module.AsyncOpenAI

    def make_client(*args, **kwargs):
        assert state["ready"].wait(2), "uvicorn app was not captured"
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.ASGITransport(app=state["app"]))
        return real_async_openai(*args, max_retries=0, **kwargs)

    monkeypatch.setattr(module, "_find_open_port", lambda: 28123)
    monkeypatch.setattr(module.uvicorn, "run", fake_run)
    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    monkeypatch.setattr(module, "AsyncOpenAI", make_client)
    return state


def _request_kwargs(executor, messages, session_id):
    return {
        "model": executor.model_name,
        "messages": messages,
        "max_tokens": executor.sampling_params.max_tokens,
        "temperature": executor.sampling_params.temperature,
        "top_p": executor.sampling_params.top_p,
        "logprobs": executor.sampling_params.logprobs is not None,
        "top_logprobs": executor.sampling_params.logprobs or 1,
        "extra_body": {"session_id": session_id},
    }


def _three_round_executor(module):
    class ThreeRoundExecutor(module.AgentExecutor):
        async def run_agent(self, prompt, label, session_id):
            del label
            messages = [{"role": "user", "content": prompt}]
            for index in range(3):
                response = await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
                if index == 2:
                    continue
                messages.append({"role": "assistant", "content": response.choices[0].message.content})
                if index == 0:
                    messages.append({"role": "user", "content": "feedback <|im_start|> literal"})
                else:
                    messages.extend(
                        [
                            {"role": "tool", "content": "tool result <|im_start|> literal"},
                            {"role": "user", "content": "final feedback"},
                        ]
                    )
            return {"reward": 3.0}

    return ThreeRoundExecutor


def _run_and_close(executor, *args, **kwargs):
    async def run():
        try:
            return await executor.execute(*args, **kwargs)
        finally:
            if hasattr(executor, "client"):
                await executor.client.close()

    return asyncio.run(run())


def _answer_ids(tokenizer, noncanonical):
    if not noncanonical:
        return tokenizer.encode("foo", add_special_tokens=False)
    ids = [tokenizer.convert_tokens_to_ids("f"), tokenizer.convert_tokens_to_ids("oo")]
    assert tokenizer.decode(ids, skip_special_tokens=False) == "foo"
    assert tokenizer.encode("foo", add_special_tokens=False) != ids
    return ids


def _outputs(
    tokenizer, thinking, finish_reason="stop", include_eos=None, noncanonical=True, trailing_whitespace=False
):
    if include_eos is None:
        include_eos = finish_reason == "stop"
    prefix = tokenizer.encode("reasoning</think>\n\n", add_special_tokens=False) if thinking else []
    answer = _answer_ids(tokenizer, noncanonical)
    whitespace = " \n" if trailing_whitespace else ""
    eos = [tokenizer.eos_token_id] if include_eos else []
    token_ids = prefix + answer + tokenizer.encode(whitespace, add_special_tokens=False) + eos
    return {
        "token_ids": token_ids,
        "text": ("reasoning</think>\n\n" if thinking else "") + "foo" + whitespace,
        "finish_reason": finish_reason,
        "logprobs": [-(index + 1) / 10 for index in range(len(token_ids))],
    }


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("trailing_whitespace", [False, True])
def test_three_round_history_preserves_sampled_tokens_and_logprobs(monkeypatch, thinking, trailing_whitespace):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(thinking, "history")
    outputs = [_outputs(tokenizer, thinking, trailing_whitespace=trailing_whitespace) for _ in range(3)]
    engine = _ScriptedEngine(outputs)
    executor = _three_round_executor(module)()
    sampling = SimpleNamespace(max_tokens=64, temperature=0.0, top_p=1.0, logprobs=1)

    result = _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    observed = result["observation_tokens"]
    assert len(engine.calls) == 3
    assert not executor._token_traces
    assert not executor._session_buffers

    expected_logprobs = [0.0] * len(observed)
    for call_index, ((prompt_ids, _params), output) in enumerate(zip(engine.calls, outputs)):
        assert observed[: len(prompt_ids)] == prompt_ids
        start, end = result["action_ranges"][call_index]
        assert (start, end) == (len(prompt_ids), len(prompt_ids) + len(output["token_ids"]))
        assert observed[start:end] == output["token_ids"]
        for offset, value in enumerate(output["logprobs"]):
            expected_logprobs[start + offset] = value
    assert result["rollout_log_probs"] == expected_logprobs
    assert all(
        value != 0.0
        for value in result["rollout_log_probs"][result["action_ranges"][0][0] : result["action_ranges"][0][1]]
    )

    first_delta_start = len(engine.calls[0][0]) + len(outputs[0]["token_ids"])
    second_delta_start = len(engine.calls[1][0]) + len(outputs[1]["token_ids"])
    first_delta = tokenizer.decode(engine.calls[1][0][first_delta_start:], skip_special_tokens=False)
    second_delta = tokenizer.decode(engine.calls[2][0][second_delta_start:], skip_special_tokens=False)
    assert "feedback <|im_start|> literal" in first_delta
    assert "tool result <|im_start|> literal" in second_delta
    assert "final feedback" in second_delta
    assert second_delta.index("tool result") < second_delta.index("final feedback")


@pytest.mark.parametrize("closure", ["eos", "alternate", "none"])
def test_stable_template_is_a_positive_control(monkeypatch, closure):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "stable")
    end_token = {"eos": tokenizer.eos_token, "alternate": tokenizer.pad_token, "none": ""}[closure]
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    tokenizer._tokenizer.chat_template = STABLE_TEMPLATE.replace("<|im_end|>", end_token)
    outputs = [_outputs(tokenizer, False, include_eos=False, noncanonical=False) for _ in range(3)]
    for output in outputs:
        output["token_ids"] += tokenizer.encode(end_token, add_special_tokens=False)
        output["logprobs"] = [-(index + 1) / 10 for index in range(len(output["token_ids"]))]
    engine = _ScriptedEngine(outputs)
    executor = _three_round_executor(module)()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)

    result = _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    assert len(engine.calls) == 3
    assert all(
        observed_prompt == result["observation_tokens"][: len(observed_prompt)] for observed_prompt, _ in engine.calls
    )


def test_stop_eos_is_not_duplicated(monkeypatch):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "history")
    outputs = [_outputs(tokenizer, False, include_eos=True) for _ in range(3)]
    engine = _ScriptedEngine(outputs)
    executor = _three_round_executor(module)()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)

    result = _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    eos_id = tokenizer.eos_token_id
    for index, output in enumerate(outputs[:-1]):
        start, end = result["action_ranges"][index]
        assert result["observation_tokens"][end - 1] == eos_id
        assert result["observation_tokens"][end] != eos_id
        assert output["token_ids"][-1] == eos_id


def test_length_stop_adds_non_action_closure_with_zero_logprob(monkeypatch):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "history")
    outputs = [
        _outputs(tokenizer, False, finish_reason="length"),
        _outputs(tokenizer, False),
        _outputs(tokenizer, False),
    ]
    engine = _ScriptedEngine(outputs)
    executor = _three_round_executor(module)()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)

    result = _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    first_end = result["action_ranges"][0][1]
    assert result["observation_tokens"][first_end] == tokenizer.eos_token_id
    assert result["rollout_log_probs"][first_end] == 0.0
    assert all(first_end < start for start, _ in result["action_ranges"][1:])


@pytest.mark.parametrize("missing_eos", ["template", "sample"])
def test_unknown_reply_closure_is_rejected(monkeypatch, missing_eos):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "history")
    if missing_eos == "template":
        tokenizer._tokenizer.chat_template = HISTORY_TEMPLATE.replace("<|im_end|>", "|")
    engine = _ScriptedEngine([_outputs(tokenizer, False, include_eos=missing_eos != "sample")])
    executor = _three_round_executor(module)()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)

    with pytest.raises(BadRequestError, match="EOS"):
        _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    assert len(engine.calls) == 1
    assert not executor._token_traces
    assert not executor._session_buffers


def test_rewritten_history_is_rejected_and_session_is_cleaned(monkeypatch):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "history")
    engine = _ScriptedEngine([_outputs(tokenizer, False)])

    class RewrittenHistoryExecutor(module.AgentExecutor):
        async def run_agent(self, prompt, label, session_id):
            del label
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
            rewritten = [
                {"role": "user", "content": "rewritten initial prompt"},
                {"role": "assistant", "content": response.choices[0].message.content},
                {"role": "user", "content": "new feedback"},
            ]
            await self.client.chat.completions.create(**_request_kwargs(self, rewritten, session_id))

    executor = RewrittenHistoryExecutor()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)
    with pytest.raises(BadRequestError, match="history"):
        _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    assert not executor._token_traces
    assert not executor._session_buffers


def test_failed_generation_keeps_last_successful_session_state(monkeypatch):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "history")
    outputs = [_outputs(tokenizer, False) for _ in range(2)]
    engine = _ScriptedEngine([outputs[0], module.HTTPException(503, "temporary engine error"), outputs[1]])

    class RetryExecutor(module.AgentExecutor):
        async def run_agent(self, prompt, label, session_id):
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
            messages.extend(
                [
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": "feedback"},
                ]
            )
            buffer = list(self._session_buffers[session_id])
            traces = deepcopy(self._token_traces[session_id])
            with pytest.raises(InternalServerError, match="temporary engine error"):
                await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
            assert self._session_buffers[session_id] == buffer
            assert self._token_traces[session_id] == traces
            await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
            return {"reward": 1.0}

    executor = RetryExecutor()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)
    result = _run_and_close(executor, "initial prompt", "label", sampling, 256, tokenizer, engine)
    assert len(engine.calls) == 3
    assert engine.calls[1][0] == engine.calls[2][0]
    assert len(result["action_ranges"]) == 2
    for (start, end), output in zip(result["action_ranges"], outputs):
        assert result["observation_tokens"][start:end] == output["token_ids"]
        assert result["rollout_log_probs"][start:end] == output["logprobs"]
    assert not executor._token_traces
    assert not executor._session_buffers


def test_max_length_overflow_is_rejected_without_truncation(monkeypatch):
    module = _load_executor(monkeypatch)
    _patch_server(monkeypatch, module)
    tokenizer = _make_tokenizer(False, "stable")
    engine = _ScriptedEngine([_outputs(tokenizer, False, noncanonical=False)])

    class OverflowExecutor(module.AgentExecutor):
        async def run_agent(self, prompt, label, session_id):
            del label
            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))
            messages.extend(
                [
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": "overflow " + ("alpha beta gamma delta epsilon " * 50)},
                ]
            )
            await self.client.chat.completions.create(**_request_kwargs(self, messages, session_id))

    executor = OverflowExecutor()
    sampling = SimpleNamespace(max_tokens=16, temperature=0.0, top_p=1.0, logprobs=1)
    with pytest.raises(BadRequestError, match="exceeds max_length"):
        _run_and_close(executor, "initial prompt", "label", sampling, 64, tokenizer, engine)
    assert len(engine.calls) == 1
    assert not executor._token_traces
    assert not executor._session_buffers
