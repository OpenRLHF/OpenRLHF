from types import SimpleNamespace

import torch

from openrlhf.models import PolicyLoss, ValueLoss
from openrlhf.trainer.ppo_utils.experience import Experience, balance_experiences, get_model_parallel_size


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_includes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 24


def test_balance_experiences_uses_ep_for_effective_dp():
    exp = Experience(
        sequences=torch.arange(8).view(4, 2),
        attention_mask=torch.ones(4, 2, dtype=torch.long),
        total_length=torch.tensor([8, 7, 6, 5]),
    )

    balanced = balance_experiences([exp], _args(ep=2, actor_gpus=4))

    assert len(balanced) == 2
    assert [len(item.sequences) for item in balanced] == [2, 2]


def _cp_rank_shards(tensor):
    # AutoModel CP rank 0 owns chunks [0, 3], rank 1 owns chunks [1, 2].
    chunks = torch.chunk(tensor, chunks=4, dim=1)
    return torch.cat([chunks[0], chunks[3]], dim=1), torch.cat([chunks[1], chunks[2]], dim=1)


def test_cp_local_value_loss_matches_global_token_mean():
    values = torch.tensor([[0.2, 0.4, -0.1, 0.8]])
    old_values = torch.zeros_like(values)
    returns = torch.tensor([[0.0, 0.6, 0.3, 0.2]])
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)
    loss_fn = ValueLoss(clip_eps=None)

    full_loss = loss_fn(values, old_values, returns, action_mask=mask)
    value_shards = _cp_rank_shards(values)
    old_shards = _cp_rank_shards(old_values)
    return_shards = _cp_rank_shards(returns)
    mask_shards = _cp_rank_shards(mask)
    shard_losses = [
        loss_fn(v, o, r, action_mask=m, dp_size=2, batch_num_tokens=mask.sum())
        for v, o, r, m in zip(value_shards, old_shards, return_shards, mask_shards)
    ]

    torch.testing.assert_close(sum(shard_losses) / 2, full_loss)


def test_cp_local_policy_loss_matches_global_token_mean():
    log_probs = torch.log(torch.tensor([[0.5, 0.6, 0.7, 0.8]]))
    old_log_probs = torch.log(torch.tensor([[0.4, 0.7, 0.6, 0.75]]))
    advantages = torch.tensor([[1.0, -0.5, 0.25, 0.75]])
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.bool)
    loss_fn = PolicyLoss()

    full_loss, *_ = loss_fn(log_probs, old_log_probs, advantages, action_mask=mask)
    log_prob_shards = _cp_rank_shards(log_probs)
    old_log_prob_shards = _cp_rank_shards(old_log_probs)
    advantage_shards = _cp_rank_shards(advantages)
    mask_shards = _cp_rank_shards(mask)
    shard_losses = [
        loss_fn(lp, old_lp, adv, action_mask=m, dp_size=2, batch_num_tokens=mask.sum())[0]
        for lp, old_lp, adv, m in zip(log_prob_shards, old_log_prob_shards, advantage_shards, mask_shards)
    ]

    torch.testing.assert_close(sum(shard_losses) / 2, full_loss)
