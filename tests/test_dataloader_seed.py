from copy import deepcopy

import pytest
import torch

from tests.test_deepspeed_save_model import ds_module  # noqa: F401


@pytest.fixture
def strategy(ds_module, monkeypatch):
    monkeypatch.setattr(ds_module.dist, "is_initialized", lambda: False)
    strategy = object.__new__(ds_module.DeepspeedStrategy)
    strategy.seed = 42
    with torch.random.fork_rng(devices=[]):
        yield strategy


@pytest.mark.parametrize("num_workers", [0, 2])
def test_prompt_shuffle_uses_configured_seed_across_epochs(strategy, num_workers):
    orders = []
    for ambient_seed, train_seed in [(101, 42), (202, 42), (101, 43)]:
        torch.manual_seed(ambient_seed)
        strategy.seed = train_seed
        loader = strategy.setup_dataloader(list(range(32)), 1, num_workers=num_workers)
        orders.append([[batch.item() for batch in loader] for _ in range(2)])

    assert orders[0] == orders[1]
    assert orders[0][0] != orders[2][0]
    assert orders[0][0] != orders[0][1]
    for run in orders:
        for epoch in run:
            assert sorted(epoch) == list(range(32))


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("legacy", [False, True])
def test_prompt_shuffle_resumes_remaining_samples(strategy, ds_module, num_workers, legacy):
    if legacy:
        # Older checkpoints used the loader's default, unseeded stateful sampler.
        loader = ds_module.StatefulDataLoader(
            list(range(32)), batch_size=1, shuffle=True, num_workers=num_workers, persistent_workers=num_workers > 0
        )
    else:
        loader = strategy.setup_dataloader(list(range(32)), 1, num_workers=num_workers)
    iterator = iter(loader)
    consumed = [next(iterator).item() for _ in range(7)]
    state = deepcopy(loader.state_dict())
    expected = [batch.item() for batch in iterator]
    expected_next_epoch = [batch.item() for batch in loader]

    torch.manual_seed(987)
    resumed = strategy.setup_dataloader(list(range(32)), 1, num_workers=num_workers)
    resumed.load_state_dict(state)
    actual = [batch.item() for batch in resumed]

    assert actual == expected
    assert sorted(consumed + actual) == list(range(32))
    assert [batch.item() for batch in resumed] == expected_next_epoch


def test_unshuffled_and_explicit_sampler_order_is_preserved(strategy):
    dataset = list(range(8))
    sequential = strategy.setup_dataloader(dataset, 1, shuffle=False)
    assert [batch.item() for batch in sequential] == dataset

    sampler = [6, 1, 4, 0]
    explicit = strategy.setup_dataloader(dataset, 1, sampler=sampler)
    assert explicit.sampler is sampler
    assert [batch.item() for batch in explicit] == sampler
