import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_terminal_oversampling_buffer_is_drained_before_next_episode(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", MagicMock())
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=object))
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ppo_utils.experience", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.trainer.ray.vllm_engine", MagicMock())
    monkeypatch.setitem(sys.modules, "openrlhf.utils.logging_utils", MagicMock())

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_openrlhf_samples_generator_test",
        root / "openrlhf" / "trainer" / "ppo_utils" / "samples_generator.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1, vllm_generate_batch_size=4),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        vllm=SimpleNamespace(enable_sleep=False),
    )
    loader = MagicMock()
    loader.__iter__.side_effect = lambda: iter([0, 1, 2])
    generator = module.SamplesGenerator(SimpleNamespace(args=args), loader, None, None, [])

    def generate(dataloader_iter, num_prompts, dynamic_filtering, **kwargs):
        del dynamic_filtering, kwargs
        generated = []
        for _ in range(num_prompts):
            try:
                generated.append(next(dataloader_iter))
            except StopIteration:
                break
        return generated, len(generated), len(generated) < num_prompts

    generator._generate_vllm = generate

    first, _, first_prompts, first_exhausted = generator.generate_samples()
    first_buffer = list(generator._sample_buffer)
    second, _, second_prompts, second_exhausted = generator.generate_samples()
    third, _, third_prompts, third_exhausted = generator.generate_samples()

    assert (first, first_prompts, first_exhausted) == ([0, 1], 3, False)
    assert first_buffer == [2]
    assert (second, second_prompts, second_exhausted) == ([2], 0, True)
    assert (third, third_prompts, third_exhausted) == ([0, 1], 3, False)
    assert loader.__iter__.call_count == 2
