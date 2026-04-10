import itertools
from dataclasses import dataclass, field, fields
from typing import Any, List, Union

import torch

from openrlhf.utils.utils import zero_pad_sequences


def tensor_field(role: str, **kwargs):
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["tensor_role"] = role
    return field(metadata=metadata, **kwargs)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """A batch of RL experience for policy optimization.

    Fields are grouped by RL semantics:
    - Trajectory: token-level state-action sequences and masks (B, T)
    - Policy: per-step log-probabilities under different policies (B, A)
    - Value: per-step value estimates, returns, and advantages (B, A)
    - Outcome: per-episode rewards and generation metadata (B,)
    - Metadata: non-tensor fields for logging and data tracking
    """

    # ── Trajectory: state-action sequences ──
    sequences: torch.Tensor = tensor_field("step", default=None)  # (B, T) token ids [prompt + response]
    attention_mask: torch.LongTensor = tensor_field("step", default=None)  # (B, T)
    action_mask: torch.BoolTensor = tensor_field("step", default=None)  # (B, A) mask over action (response) tokens

    # ── Policy: log π(a|s) under different policies ──
    action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, A) log π_θ(a|s)  current policy
    base_action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, A) log π_ref(a|s) reference policy
    rollout_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, A) log π_old(a|s) rollout policy

    # ── Value estimation ──
    values: torch.Tensor = tensor_field("step", default=None)  # (B, A) V(s)
    returns: torch.Tensor = tensor_field("step", default=None)  # (B, A) G_t
    advantages: torch.Tensor = tensor_field("step", default=None)  # (B, A) Â(s,a)
    kl: torch.Tensor = tensor_field("step", default=None)  # (B, A) D_KL(π_θ ‖ π_ref)

    # ── Episode outcomes (per-sample scalars) ──
    rewards: torch.Tensor = tensor_field("episode", default=None)  # (B,) R, used for advantage calculation
    scores: torch.Tensor = tensor_field("episode", default=None)  # (B,) binary score for dynamic sampling
    response_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) number of generated tokens
    truncated: torch.Tensor = tensor_field("episode", default=None)  # (B,) whether generation was truncated
    total_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) prompt + response length

    # ── Metadata (not part of RL computation) ──
    index: list[int] = None
    prompts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    images: list = field(default_factory=list)  # per-sample image paths/URLs for VLM (None entries for text-only)
    mm_train_inputs: list = field(default_factory=list)  # per-sample processor outputs (pixel_values dicts) for VLM
    info: dict = field(default_factory=dict)  # per-sample metrics for logging

    @classmethod
    def is_step_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "step"

    @classmethod
    def is_episode_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "episode"

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, field, to(value, device))

        return self

    def pin_memory(self):
        """Pin memory for all tensor fields."""
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, field, {key: pin_memory(val) for key, val in value.items()})
            else:
                setattr(self, field, pin_memory(value))

        return self

    @staticmethod
    def select(experiences: List["Experience"], fields: List[str]) -> List["Experience"]:
        """Select specific fields from a list of Experience instances to create new Experience instances.

        Args:
            experiences: List of Experience instances
            fields: List of field names to select

        Returns:
            A list of new Experience instances containing only the selected fields
        """
        new_experiences = []
        for exp in experiences:
            new_exp = Experience()
            for field in fields:
                if hasattr(exp, field):
                    setattr(new_exp, field, getattr(exp, field))
            new_experiences.append(new_exp)
        return new_experiences

    @staticmethod
    def _merge_item(items: List, pad_value: int = 0) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            return zero_pad_sequences(items, side="right", value=pad_value)
        elif isinstance(items[0], list):
            return list(itertools.chain.from_iterable(items))
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {key: Experience._merge_item(values, pad_value) for key, values in result.items()}
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(experiences_list: List["Experience"], pad_token_id) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


# ── Batch manipulation utilities ──


def split_experience_batch(experience: Experience) -> List[Experience]:
    """Split a batched Experience into individual single-sample Experiences."""
    batch_size = len(experience.sequences)
    experience.index = None

    items = []
    for i in range(batch_size):
        kwargs = {}
        for f in fields(Experience):
            value = getattr(experience, f.name)
            if value is None:
                kwargs[f.name] = None
            elif isinstance(value, torch.Tensor):
                if len(value) != batch_size:
                    raise ValueError(f"Size of {f.name} ({len(value)}) does not match batch_size ({batch_size})")
                kwargs[f.name] = value[i]
            elif isinstance(value, dict):
                d = {}
                for k, v in value.items():
                    if isinstance(v, (torch.Tensor, list)):
                        if len(v) != batch_size:
                            raise ValueError(
                                f"Size of {f.name}[{k}] ({len(v)}) does not match batch_size ({batch_size})"
                            )
                        d[k] = v[i]
                    else:
                        raise TypeError(f"Unsupported type for {f.name}[{k}]: {type(v)}")
                kwargs[f.name] = d
            elif isinstance(value, list):
                kwargs[f.name] = [value[i]] if len(value) == batch_size else value
        items.append(Experience(**kwargs))

    return items


def make_experience_batch(items: List[Experience], packing_samples=False) -> Experience:
    """Combine individual single-sample Experiences into a batched Experience."""
    if not items:
        raise ValueError("Empty items list")

    kwargs = {}
    for f in fields(Experience):
        first = getattr(items[0], f.name)
        if first is None:
            kwargs[f.name] = None
        elif isinstance(first, torch.Tensor):
            tensors = [getattr(item, f.name) for item in items]
            if Experience.is_step_tensor_field(f.name):
                kwargs[f.name] = zero_pad_sequences(tensors, "right", stack=True)
            elif Experience.is_episode_tensor_field(f.name) or first.dim() == 0:
                kwargs[f.name] = torch.stack(tensors)
            else:
                raise ValueError(f"Unsupported tensor field batching rule for {f.name}")
        elif isinstance(first, dict):
            kwargs[f.name] = {}
            for key in first.keys():
                vals = [getattr(item, f.name)[key] for item in items]
                if not vals:
                    continue
                first_type = type(vals[0])
                if not all(isinstance(v, first_type) for v in vals):
                    raise TypeError(f"Inconsistent types in {f.name}[{key}]")
                if all(isinstance(v, (int, float)) for v in vals):
                    kwargs[f.name][key] = torch.tensor(vals)
                else:
                    kwargs[f.name][key] = vals
        elif isinstance(first, list):
            kwargs[f.name] = list(itertools.chain.from_iterable(getattr(item, f.name) for item in items))

    return Experience(**kwargs)


def remove_padding_in_sequences(items: List[Experience]) -> List[Experience]:
    """Remove right padding from per-step fields of single-sample Experiences."""
    for item in items:
        right_pad = item.attention_mask.flip(0).argmax()
        right_pad = None if right_pad == 0 else -right_pad

        for f in fields(Experience):
            value = getattr(item, f.name)
            if isinstance(value, torch.Tensor) and Experience.is_step_tensor_field(f.name):
                setattr(item, f.name, value[:right_pad])

    return items


def balance_experiences(experiences, args):
    """
    Balance experience across dp ranks by interleaving long and short sequences.

    Example:
        sorted lengths: [8,7,6,5,4,3,2,1], effective_num: 2
        first_half: [[8,7], [6,5]], last_half: [[3,4], [1,2]], interval_items: [[8,7], [1,2], [6,5], [3,4]]
        interval_merged: [[8,1,6,3], [7,2,5,4]]
    """
    # split experience, sort by total_length
    items_all = []
    for item in experiences:
        items_all.extend(split_experience_batch(item))
    items_all.sort(key=lambda x: x.total_length, reverse=True)

    # split experience into chunks
    effective_num = (
        args.actor_num_nodes * args.actor_num_gpus_per_node // args.ring_attn_size // args.ds_tensor_parallel_size
    )
    split_items = [items_all[i : i + effective_num] for i in range(0, len(items_all), effective_num)]
    half = len(split_items) // 2
    first_half = split_items[:half]
    last_half = [item[::-1] for item in split_items[half:]]

    # balance distribution by interleaving chunks
    interval_items = []
    for i in range(half):
        interval_items.append(first_half[i])
        interval_items.append(last_half[-(i + 1)])
    if len(last_half) > len(first_half):
        interval_items.append(last_half[0])

    interval_merged = list(zip(*interval_items))
    return [make_experience_batch(items) for items in interval_merged]
