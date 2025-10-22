import argparse
import json
import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from openrlhf.datasets import Latent_preprocessing_Dataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models import TextVQVAE


def add_special_tokens(indices, selected_mask, attn_masks, tokenizer, special_token_id_list):
    """
    add special tokens at the beginning and end of each sequence
    """

    device = indices.device
    pad_id = tokenizer.pad_token_id
    bol_id = special_token_id_list[0]
    eol_id = special_token_id_list[1]
    valid_len = selected_mask.sum(dim=-1)

    # 1. generate result tensor with pad_id, add bol_id at the beginning, add eol_id at the end of each sequence
    batch_size, seq_len = indices.shape
    result = torch.full((batch_size, seq_len + 2), pad_id, device=device, dtype=indices.dtype)

    # 2. fill the left side of result with bol_id
    result[:, 0] = bol_id

    # 3. append valid tokens to the result
    for i in range(batch_size):
        valid_count = valid_len[i].item()
        if valid_count > 0:
            # copy valid tokens to the result
            result[i, 1 : 1 + valid_count] = indices[i, :valid_count]
            # add eol_id to the end of valid tokens
            result[i, 1 + valid_count] = eol_id
            # copy invalid tokens to the result
            if valid_count < seq_len:
                result[i, 2 + valid_count :] = indices[i, valid_count:]

    attention_mask = result != pad_id

    return result, attention_mask


## TODO: need to check
def replace_indices_with_latent_tokens(indices, selected_mask, tokenizer, latent_vocab_indices):
    """
    indices: (B, L) int tensor
    selected_mask: (B, L) bool tensor
    tokenizer: huggingface-style tokenizer (encode() available)
    latent_vocab_size: LATENT token size (default 1024)
    """

    device = indices.device
    pad_id = tokenizer.pad_token_id
    # (latent_vocab_size,) tensor

    # 2️⃣ initialize with pad_id
    replaced_indices = torch.full_like(indices, pad_id)

    # 3️⃣ map indices to special_token_ids
    valid_indices = indices[selected_mask].to(torch.long)  # (num_valid,)
    mapped_values = latent_vocab_indices[valid_indices]  # (num_valid,)

    # 4️⃣ fill the result
    replaced_indices[selected_mask] = mapped_values

    return replaced_indices


def integrate_vq_indices(
    extended_inputs: torch.Tensor,  # (B, N)
    indices: torch.Tensor,  # (B, seq_len)
    selected_mask: torch.Tensor,  # (B, seq_len), bool
    r: int,
    chunk_num: list[int] | torch.Tensor,
    m_list: list[int] | torch.Tensor,
    pad_value: int = 0,
):
    """
    for each batch i:
      valid_len = selected_mask[i].sum()
      replaced_len = valid_len * r

      result[i, :valid_len] = indices[i, :valid_len]
      result[i, valid_len : valid_len + (N - replaced_len)] = extended_inputs[i, replaced_len:]
      the rest is padded with pad_value
    """
    device = extended_inputs.device
    B, N = extended_inputs.shape

    # initialize: pad with pad_value
    result = torch.full_like(extended_inputs, pad_value)

    # for each batch
    for i in range(B):
        valid_len = int(selected_mask[i].sum().item())  # valid tokens from VQ
        replaced_len = valid_len * r
        if chunk_num:
            assert chunk_num[i] == valid_len
        if m_list:
            assert m_list[i] == valid_len * r  # replaced tokens in original sequence

        # ① replace the valid part of indices
        if valid_len > 0:
            result[i, :valid_len] = indices[i, :valid_len]

        # ② append original tail
        tail_len = N - replaced_len
        if tail_len > 0:
            # tail is the part of extended_inputs after replaced_len
            result[i, valid_len : valid_len + tail_len] = extended_inputs[i, replaced_len : replaced_len + tail_len]

    attention_mask = result != pad_value

    return result, attention_mask


def _preprocess_original_dataset(
    dataloader, tokenizer, vqvae: TextVQVAE, max_M_list: list, m_count_per_sample: int, r: int, latent_vocab_size: int
):
    """
    preprocess original dataset - sample max M, uniform m, slicing and chunking with m.

    1. for each sample, sample max M from M_max list, and sample m in uniform distribution between 1 and M, multiple of L.
    2. slicing and chunking with m. (batch, m/L, L), m is different for each sample.
    3. caching list of m in batch, m_list.



    Return:
        - preprocessed dataset
        (batch, m/L, L)
        - cached m list in batch
        (batch,)
    """

    latent_vocab_size = vqvae.codebook_size
    latent_token_id_list = []
    for i in range(latent_vocab_size):
        token_str = f"<LATENT_{i}>"
        token_id = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_id) != 1:
            raise ValueError(f"Token '{token_str}' did not map to exactly one token: {token_id}")
        latent_token_id_list.append(token_id[0])

    special_token_id_list = [tokenizer.encode("<|bol|>"), tokenizer.encode("<|eol|>")]

    latent_token_id_list = torch.tensor(latent_token_id_list)
    special_token_id_list = torch.tensor(special_token_id_list)

    preprocessed_response_ids = []
    origin_prompt_ids = []
    for prompt_ids, prompt_attention_masks, response_ids, response_attention_masks, prompt_ids_lens in dataloader:
        # prompts have left padding, responses have right padding
        inputs = response_ids
        attn_masks = response_attention_masks

        sampled_max_Ms = random.sample(max_M_list, inputs.shape[0])

        b, n = inputs.shape
        device = inputs.device

        if not isinstance(sampled_max_Ms, torch.Tensor):
            sampled_max_Ms = torch.tensor(sampled_max_Ms, dtype=torch.long, device=device)
        else:
            sampled_max_Ms = sampled_max_Ms.to(device=device, dtype=torch.long)

        # (b, k): for each sample, sample k uniform random [0,1)
        rand_vals = torch.rand(b, m_count_per_sample, device=device)

        # possible candidates for each sample = (max_M / r) + 1
        num_candidates = (sampled_max_Ms // r) + 1  # (b,)

        # (b, k): for each sample, sample one candidate from the possible candidates (0 ~ num_candidates-1)
        rand_indices = (rand_vals * num_candidates.unsqueeze(1)).long().clamp(max=num_candidates.unsqueeze(1) - 1)

        # (b, k): calculate the actual m value for each sample
        m_list = rand_indices * r  # (b, k)

        # (b*k,): flatten
        m_list = m_list.reshape(-1)
        chunk_nums = m_list // r

        # (b, k, n): replicate each sample k times
        extended_inputs = inputs.unsqueeze(1).expand(-1, m_count_per_sample, -1)
        extended_attn_masks = attn_masks.unsqueeze(1).expand(-1, m_count_per_sample, -1)

        # (b*k, n)
        extended_inputs = extended_inputs.reshape(b * m_count_per_sample, n)
        extended_attn_masks = extended_attn_masks.reshape(b * m_count_per_sample, n)

        indices, selected_mask = vqvae.encode_with_chunk_num(extended_inputs, extended_attn_masks, chunk_nums)

        # indices = integrate_vq_indices(extended_inputs, indices, selected_mask, r, chunk_nums, m_list, tokenizer.pad_token_id)

        latent_token_indices = replace_indices_with_latent_tokens(
            indices, selected_mask, tokenizer, latent_token_id_list
        )
        origin_with_latent, origin_with_latent_attn_masks = integrate_vq_indices(
            extended_inputs, latent_token_indices, selected_mask, r, chunk_nums, m_list, tokenizer.pad_token_id
        )

        # add bol, eol special tokens to the origin_with_latent
        origin_with_latent, origin_with_latent_attn_masks = add_special_tokens(
            origin_with_latent, selected_mask, origin_with_latent_attn_masks, tokenizer, special_token_id_list
        )

        origin_prompt_ids.extend(prompt_ids)
        preprocessed_response_ids.extend(origin_with_latent.tolist())

    return origin_prompt_ids, preprocessed_response_ids


def init_distributed():
    """Initialize distributed training if available"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0


def build_latent_dataset(args):
    """
    Build latent dataset from original dataset.

    1. get vqvae model, tokenizer, dataset
    2. preprocess original dataset - sample max M, uniform m, slicing and chunking with m.
    3. get batch codebook indices
    4. replace original tokens with codebook indices
    5. add bol, eol special tokens
    6. save dataset
    """
    rank, world_size, local_rank = init_distributed()

    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print if rank == 0 else lambda *args, **kwargs: None
    dummy_strategy.is_rank_0 = lambda: rank == 0
    dummy_strategy.args = args

    # maybe change load config method
    model_config = json.load(open(os.path.join(args.model_path, "config.json")))

    vqvae = TextVQVAE(**model_config)

    # need to save trained model like this:
    # torch.save(model.state_dict(), "save_model_path/vqvae.bin")
    state_dict = torch.load(os.path.join(args.model_path, "vqvae.bin"), map_location="cpu")
    vqvae.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = blending_datasets(
        args.dataset,
        None,
        dummy_strategy,
        args.seed,
    )

    # split dataset to each rank
    if world_size > 1:
        # Calculate chunk size for each rank
        total_samples = len(dataset)
        samples_per_rank = total_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank < world_size - 1 else total_samples

        # Create subset for this rank
        dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))

        if rank == 0:
            dummy_strategy.print(f"Total samples: {total_samples}, samples per rank: {samples_per_rank}")
            dummy_strategy.print(f"Rank {rank} processing samples {start_idx} to {end_idx-1}")

    dataset = Latent_preprocessing_Dataset(dataset, tokenizer, args.max_length, dummy_strategy)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    prompt_ids, preprocessed_response_ids = _preprocess_original_dataset(
        dataloader,
        tokenizer,
        vqvae,
        args.max_M_list,
        args.m_count_per_sample,
        vqvae.compression_rate,
        vqvae.codebook_size,
    )

    # gather prompt_ids and preprocessed_response_ids to rank 0
    if world_size > 1:
        # Gather all prompt_ids and preprocessed_response_ids to rank 0
        gathered_prompt_ids = [None] * world_size
        gathered_response_ids = [None] * world_size

        dist.all_gather_object(gathered_prompt_ids, prompt_ids)
        dist.all_gather_object(gathered_response_ids, preprocessed_response_ids)

        if rank == 0:
            # Flatten the gathered lists
            all_prompt_ids = []
            all_response_ids = []
            for i in range(world_size):
                all_prompt_ids.extend(gathered_prompt_ids[i])
                all_response_ids.extend(gathered_response_ids[i])

            prompt_ids = all_prompt_ids
            preprocessed_response_ids = all_response_ids

    if rank == 0:
        import jsonlines

        with jsonlines.open(args.save_path, mode="w") as writer:
            for prompt_id, latent_response in zip(prompt_ids, preprocessed_response_ids):
                writer.write({f"{args.input_key}": prompt_id, f"{args.output_key}": latent_response})

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    ## requirements:
    # model_path: path to the vqvae model, must have config.json, extended_tokenizer, and vqvae.bin
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--max_M_list", type=list, required=True)
    parser.add_argument("--m_count_per_sample", type=int, required=True)
    parser.add_argument("--input_key", type=str, required=True)
    parser.add_argument("--output_key", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    build_latent_dataset(args)
