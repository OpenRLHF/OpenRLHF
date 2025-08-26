import torch
import torch.distributed as dist
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from flash_attn.utils.distributed import all_gather

STAR_ATTN_GROUP = None


def set_star_attn_group(group):
    global STAR_ATTN_GROUP
    STAR_ATTN_GROUP = group


def get_star_attn_group():
    return STAR_ATTN_GROUP


def _get_tensor_in_current_star_rank(tensors: list[torch.Tensor] | torch.Tensor, star_attn_group, pad_id):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    star_rank = dist.get_rank(group=star_attn_group)
    star_size = dist.get_world_size(group=star_attn_group)
    seqlen = tensors[0].shape[-1]
    total_seq_len = tensors[0].numel()
    star_attn_pad_len = (star_size - seqlen % star_size) % star_size
    output_tensors = []
    for tensor in tensors:
        if tensor.numel() != total_seq_len:
            raise ValueError(f"tensor.numel() {tensor.numel()} != total_seq_len {total_seq_len}")
        tensor = torch.nn.functional.pad(tensor, (0, star_attn_pad_len), value=pad_id)
        local_seq_len = tensor.numel() // star_size
        start, end = star_rank * local_seq_len, (star_rank + 1) * local_seq_len
        tensor = tensor[:, start:end]
        output_tensors.append(tensor)
    if len(output_tensors) == 1:
        output_tensors = output_tensors[0]
    return output_tensors, star_attn_pad_len


def unpad_and_slice_tensor(sequences, attention_mask, star_attn_group):
    rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
    sequences, indices, cu_seqlens, _, _ = unpad_input(sequences.unsqueeze(-1), attention_mask)
    sequences = sequences.transpose(0, 1)
    rolled_sequences = index_first_axis(
        rearrange(rolled_sequences.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None)
    position_ids = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(
        0, 1
    )
    star_attn_pad_len = 0
    if star_attn_group is not None:
        (sequences, position_ids, rolled_sequences), star_attn_pad_len = _get_tensor_in_current_star_rank(
            [sequences, position_ids, rolled_sequences], star_attn_group, 0
        )
        cu_seqlens[-1] += star_attn_pad_len
    return sequences, position_ids, rolled_sequences, star_attn_pad_len, indices


def gather_and_pad_tensor(tensor, star_attn_group, star_attn_pad_len, indices, batch, seqlen):
    if star_attn_group is not None:
        tensor = all_gather(tensor.transpose(0, 1), star_attn_group).transpose(0, 1)
        if star_attn_pad_len > 0:
            tensor = tensor[:, :-star_attn_pad_len]
    tensor = pad_input(tensor.transpose(0, 1), indices, batch, seqlen).squeeze(-1)
    return tensor
