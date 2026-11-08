import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


class _Pbar:
    def __init__(self, *_args, **_kwargs):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


class _StatefulLoader:
    """Minimal stand-in for StatefulDataLoader: a cursor that survives a restart."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.ids):
            raise StopIteration
        item = self.ids[self.position]
        self.position += 1
        return item

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state_dict):
        self.position = state_dict["position"]


@pytest.fixture
def samples_generator_module(monkeypatch):
    fake_ray = types.ModuleType("ray")
    fake_ray.cancelled = []
    fake_ray.wait = lambda refs, **_: (refs[:1], refs[1:])
    fake_ray.get = lambda ref: ref
    fake_ray.cancel = fake_ray.cancelled.append
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=_Pbar))

    experience_module = types.ModuleType("openrlhf.trainer.ppo_utils.experience")
    experience_module.Experience = object
    monkeypatch.setitem(sys.modules, experience_module.__name__, experience_module)

    engine_module = types.ModuleType("openrlhf.trainer.ray.vllm_engine")
    engine_module.batch_vllm_engine_call = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)

    logging_module = types.ModuleType("openrlhf.utils.logging_utils")
    logging_module.init_logger = lambda *_args, **_kwargs: SimpleNamespace(info=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, logging_module.__name__, logging_module)

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_samples_generator_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, fake_ray


def _fake_generate_vllm(dataloader_iter, num_prompts, dynamic_filtering, **kwargs):
    """Stand-in for vLLM generation: pull up to num_prompts prompt ids from the loader."""
    del dynamic_filtering, kwargs
    generated = []
    for _ in range(num_prompts):
        try:
            generated.append(next(dataloader_iter))
        except StopIteration:
            break
    return generated, len(generated), len(generated) < num_prompts


def _generator(module, loader, batch_size=2, vllm_generate_batch_size=4):
    args = SimpleNamespace(
        rollout=SimpleNamespace(
            batch_size=batch_size, n_samples_per_prompt=1, vllm_generate_batch_size=vllm_generate_batch_size
        ),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        vllm=SimpleNamespace(enable_sleep=False),
    )
    generator = module.SamplesGenerator(SimpleNamespace(args=args), loader, None, None, [])
    generator._generate_vllm = _fake_generate_vllm
    return generator


def _drain(generator):
    """Collect training chunks until the generator reports the episode exhausted."""
    chunks = []
    while True:
        chunk, _, _, exhausted = generator.generate_samples()
        chunks.append(chunk)
        if exhausted:
            return chunks


def test_terminal_oversampling_buffer_is_drained_before_next_episode(samples_generator_module):
    module, _ = samples_generator_module

    loader = MagicMock()
    loader.__iter__.side_effect = lambda: iter([0, 1, 2])
    generator = _generator(module, loader)

    first, _, first_prompts, first_exhausted = generator.generate_samples()
    first_buffer = list(generator._sample_buffer)
    second, _, second_prompts, second_exhausted = generator.generate_samples()
    third, _, third_prompts, third_exhausted = generator.generate_samples()

    assert (first, first_prompts, first_exhausted) == ([0, 1], 3, False)
    assert first_buffer == [2]
    assert (second, second_prompts, second_exhausted) == ([2], 0, True)
    assert (third, third_prompts, third_exhausted) == ([0, 1], 3, False)
    assert loader.__iter__.call_count == 2


def test_async_checkpoint_resume_serves_buffered_rollouts(samples_generator_module):
    """Resuming at a training boundary must not skip oversampled rollouts (#1322)."""
    module, _ = samples_generator_module
    assert _drain(_generator(module, _StatefulLoader(range(8)))) == [[0, 1], [2, 3], [4, 5], [6, 7], []]

    generator = _generator(module, _StatefulLoader(range(8)))
    trained_before_checkpoint, _, _, _ = generator.generate_samples()
    assert trained_before_checkpoint == [0, 1]
    assert list(generator._sample_buffer) == [2, 3]

    # Simulate a restart: a fresh loader and a fresh generator restored from the checkpoint.
    checkpoint = generator.state_dict()
    resumed = _generator(module, _StatefulLoader(range(8)))
    resumed.load_state_dict(checkpoint)

    chunks_after_checkpoint = _drain(resumed)

    assert chunks_after_checkpoint == [[2, 3], [4, 5], [6, 7], []]
    assert trained_before_checkpoint + [item for chunk in chunks_after_checkpoint for item in chunk] == list(range(8))


def test_resume_after_drained_generation_serves_buffered_rollouts(samples_generator_module):
    """A checkpoint taken after the loader drained must still train its leftovers (#1322)."""
    module, _ = samples_generator_module

    generator = _generator(module, _StatefulLoader(range(3)))
    first, _, _, first_exhausted = generator.generate_samples()
    assert (first, first_exhausted) == ([0, 1], False)
    checkpoint = generator.state_dict()
    assert checkpoint["dataloader_exhausted"] is True

    resumed = _generator(module, _StatefulLoader(range(3)))
    resumed.load_state_dict(checkpoint)
    resumed._generate_vllm = lambda *_args, **_kwargs: pytest.fail("a drained episode must not re-iterate")

    chunk, _, prompts_consumed, exhausted = resumed.generate_samples()

    assert (chunk, prompts_consumed, exhausted) == ([2], 0, True)


def test_dynamic_filtering_preserves_terminal_accepted_rollouts(samples_generator_module):
    module, fake_ray = samples_generator_module
    generator = module.SamplesGenerator.__new__(module.SamplesGenerator)
    generator.args = SimpleNamespace(algo=SimpleNamespace(dynamic_filtering_range=(0.0, 1.0)))

    def group(prefix, scores):
        return [
            SimpleNamespace(name=f"{prefix}-{index}", scores=torch.tensor([score]))
            for index, score in enumerate(scores)
        ]

    refs = [group("accepted", [0.0, 1.0]), group("rejected", [0.0, 0.0]), group("pending", [0.0, 1.0])]
    generator._dispatch_prompts_to_vllm = lambda *_args, **_kwargs: refs
    generator._process_response_into_experience = lambda response, **_kwargs: response

    dataloader = iter((None, [f"prompt-{index}"], [None], [None]) for index in range(3))
    experiences, prompts_consumed, exhausted = generator._generate_vllm(
        dataloader, num_prompts=3, dynamic_filtering=True
    )

    assert [experience.name for experience in experiences] == [
        "accepted-0",
        "accepted-1",
        "pending-0",
        "pending-1",
    ]
    assert prompts_consumed == 3
    assert exhausted is True
    assert fake_ray.cancelled == []


def test_async_checkpoint_advances_episode_after_final_rollout(monkeypatch, samples_generator_module):
    stateful = pytest.importorskip("torchdata.stateful_dataloader")
    module, fake_ray = samples_generator_module
    monkeypatch.setattr(_Pbar, "close", lambda self: None, raising=False)
    fake_ray.remote = lambda obj=None, **_kw: obj if obj is not None else lambda wrapped: wrapped
    fake_ray.get = lambda value: value
    fake_util = types.ModuleType("ray.util")
    fake_queue = types.ModuleType("ray.util.queue")
    fake_queue.Queue = object
    monkeypatch.setitem(sys.modules, "ray.util", fake_util)
    monkeypatch.setitem(sys.modules, "ray.util.queue", fake_queue)
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_trainer", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.launcher", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.deepspeed", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.utils", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.samples_generator", module)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_async_actor_test", root / "openrlhf/trainer/ppo_trainer_async.py")
    actor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actor_module)

    def make_actor(episode_count, count=3):
        loader = stateful.StatefulDataLoader(list(range(count)), batch_size=None, num_workers=0)
        actor = actor_module.GenerateSamplesActor.__new__(actor_module.GenerateSamplesActor)
        actor.args = SimpleNamespace(train=SimpleNamespace(num_episodes=episode_count))
        actor.prompts_dataloader = loader
        actor.samples_generator = _generator(module, loader)
        actor._partial_rollout = True
        actor._should_eval = lambda _: False
        actor._eval_just_done = False
        actor.generate_kwargs = {}
        actor.rollout_slots = MagicMock()
        actor.rollout_slots.get.return_value = 0
        actor.rollout_queue = MagicMock()
        return actor

    actor = make_actor(2)
    actor.fit(episode=0, total_consumed_prompts=0)
    payloads = [call.args[0] for call in actor.rollout_queue.put.call_args_list if call.args[0] != "done"]
    assert [payload[0] for payload in payloads] == [[0, 1], [2], [0, 1], [2]]
    first_state, boundary_state, final_state = (payloads[index][1] for index in (0, 1, 3))
    assert first_state["episode"] == 0
    assert first_state["samples_generator_state_dict"]["sample_buffer"] == [2]
    assert boundary_state["episode"] == 1
    assert final_state["episode"] == 2

    # A producer already ran ahead, but each queued checkpoint belongs to its rollout.
    replay = make_actor(2)
    replay.load_state_dict(first_state["samples_generator_state_dict"])
    replay.fit(first_state["episode"], first_state["total_consumed_prompts"])
    replay_payloads = [call.args[0] for call in replay.rollout_queue.put.call_args_list if call.args[0] != "done"]
    assert [payload[0] for payload in replay_payloads] == [[2], [0, 1], [2]]

    replay = make_actor(2)
    replay.load_state_dict(boundary_state["samples_generator_state_dict"])
    replay.fit(boundary_state["episode"], boundary_state["total_consumed_prompts"])
    replay_payloads = [call.args[0] for call in replay.rollout_queue.put.call_args_list if call.args[0] != "done"]
    assert [payload[0] for payload in replay_payloads] == [[0, 1], [2]]

    replay = make_actor(2)
    replay.load_state_dict(final_state["samples_generator_state_dict"])
    replay.fit(final_state["episode"], final_state["total_consumed_prompts"])
    assert [call.args[0] for call in replay.rollout_queue.put.call_args_list] == ["done"]

    # An exact chunk multiple reports exhaustion on a later empty generation.
    exact = make_actor(1, count=4)
    exact.fit(episode=0, total_consumed_prompts=0)
    exact_payloads = [call.args[0] for call in exact.rollout_queue.put.call_args_list if call.args[0] != "done"]
    exact_state = exact_payloads[-1][1]
    replay = make_actor(1, count=4)
    replay.load_state_dict(exact_state["samples_generator_state_dict"])
    replay.fit(exact_state["episode"], exact_state["total_consumed_prompts"])
    assert [call.args[0] for call in replay.rollout_queue.put.call_args_list] == ["done"]


def test_legacy_dataloader_only_state_resumes(samples_generator_module):
    module, _ = samples_generator_module
    source = _generator(module, _StatefulLoader(range(8)))
    source.generate_samples()
    old_state = {"data_loader_state_dict": source.prompts_dataloader.state_dict()}
    resumed = _generator(module, _StatefulLoader(range(8)))
    resumed.load_state_dict(old_state)
    assert resumed.generate_samples()[0] == [4, 5]
