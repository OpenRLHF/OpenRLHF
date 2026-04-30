from types import SimpleNamespace

from openrlhf.trainer.ppo_utils.samples_generator import SamplesGenerator


def test_generate_samples_fills_training_chunk_across_generation_batches():
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=2),
        vllm=SimpleNamespace(enable_sleep=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    generator.prompts_dataloader = [object()]
    calls = [
        ([object(), object()], 2, False),
        ([object(), object()], 2, False),
    ]

    def fake_generate_vllm(**kwargs):
        return calls.pop(0)

    generator._generate_vllm = fake_generate_vllm

    samples, filter_pass_rate, prompts_consumed, exhausted = generator.generate_samples()

    assert len(samples) == 4
    assert prompts_consumed == 4
    assert filter_pass_rate is None
    assert exhausted is False
    assert calls == []


def test_generate_samples_allows_short_chunk_when_dataloader_exhausted():
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=2),
        vllm=SimpleNamespace(enable_sleep=False),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    generator.prompts_dataloader = [object()]
    calls = [([object(), object()], 2, True)]

    def fake_generate_vllm(**kwargs):
        return calls.pop(0)

    generator._generate_vllm = fake_generate_vllm

    samples, _, prompts_consumed, exhausted = generator.generate_samples()

    assert len(samples) == 2
    assert prompts_consumed == 2
    assert exhausted is True
