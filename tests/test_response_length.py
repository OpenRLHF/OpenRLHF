"""
Tests for response_length computation in _process_response_into_experience.

response_length must equal action_mask.sum() (only model-generated tokens),
not the span from first to last 1 (which over-counts in multi-turn agentic
training because tool-call tokens sit in the gaps between turns).
"""

import torch


def _compute_response_length_span(action_mask: torch.Tensor) -> int:
    """Old (buggy) approach: span from first to last action token."""
    ones_indices = torch.where(action_mask)[0]
    return (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0


def _compute_response_length_sum(action_mask: torch.Tensor) -> int:
    """New (correct) approach: count of action tokens only."""
    return action_mask.sum().item()


def test_single_turn_approaches_agree():
    """Single-turn: contiguous span, so both methods return the same value."""
    # [prompt(3)] [response(5)] [padding(2)]  → action_mask over response only
    action_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=torch.long)
    assert _compute_response_length_span(action_mask) == _compute_response_length_sum(action_mask) == 5


def test_multiturn_sum_excludes_tool_tokens():
    """Multi-turn agentic: tool tokens sit between turns, span overcounts them."""
    # [prompt(3)] [turn1(4)] [tool(3)] [turn2(5)]
    action_mask = torch.zeros(15, dtype=torch.long)
    action_mask[3:7] = 1    # turn 1: 4 tokens
    action_mask[10:15] = 1  # turn 2: 5 tokens

    assert _compute_response_length_sum(action_mask) == 9   # correct: 4+5
    assert _compute_response_length_span(action_mask) == 12  # wrong: 14-3+1=12, includes 3 tool tokens


def test_multiturn_single_token_turns():
    """Edge case: each turn is exactly one token."""
    action_mask = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    assert _compute_response_length_sum(action_mask) == 2
    assert _compute_response_length_span(action_mask) == 3  # wrong: 3-1+1=3, includes the gap token


def test_empty_action_mask():
    """No action tokens: both approaches return 0."""
    action_mask = torch.zeros(8, dtype=torch.long)
    assert _compute_response_length_sum(action_mask) == 0
    assert _compute_response_length_span(action_mask) == 0
