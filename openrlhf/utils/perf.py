# Modified from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/perf.py
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch


# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= 1 - k

        hf_model._num_params = sum(
            [p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in hf_model.parameters()]
        )
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )


# Enhanced version of the function above that provides calculations and printing for Step 3
# NOTE: We assume that actor and critic model have same model size
def print_throughput_step3(actor_model, args, e2e_time, gen_exp_time, train_time, rank=0):
    if rank <= 0:
        # Actor model passed here is a HF model.
        actor_hf_config = actor_model.config
        actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(actor_hf_config)

        # We should account all models' GPU
        gpus_per_model = (
            args.ref_num_nodes * args.ref_num_gpus_per_node
            + args.reward_num_nodes * args.reward_num_gpus_per_node
            + args.critic_num_nodes * args.critic_num_gpus_per_node
            + args.actor_num_nodes * args.actor_num_gpus_per_node
            + args.vllm_num_engines * args.vllm_tensor_parallel_size
        )
        seq_length = args.generate_max_len + args.prompt_max_len
        batch_size = args.rollout_batch_size
        samples_per_second = batch_size / e2e_time

        actor_checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3

        actor_model._num_params = sum(
            [p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in actor_model.parameters()]
        )
        actor_params_in_billions = actor_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        actor_train_flops_per_iteration = calculate_flops(
            actor_checkpoint_activations_factor, batch_size, seq_length, actor_hf_config
        )

        # Actor model + Critic model
        total_train_flops = actor_train_flops_per_iteration * 2
        train_tflops = total_train_flops / (train_time * gpus_per_model * (10**12))

        # Modified formula for calculating flops in the forward pass only
        gen_flops_per_iteration = (24 * batch_size * seq_length * actor_num_layers * (actor_hidden_size**2)) * (
            1.0
            + (seq_length / (6.0 * actor_hidden_size))
            + (actor_vocab_size / (16.0 * actor_num_layers * actor_hidden_size))
        )

        gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model * (10**12))

        if actor_hf_config.torch_dtype == torch.float16:
            num_bytes = 2
        elif actor_hf_config.torch_dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = -1

        pertok_lat = gen_exp_time / args.generate_max_len
        gen_bw = 1 / pertok_lat * actor_model._num_params * num_bytes / 1e9

        total_flops_per_iteration = total_train_flops + gen_flops_per_iteration
        total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))

        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {args.generate_max_len}"
        )
        print(f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}")
        actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"
        print(f"Actor Model Parameters => {actor_param_string}")


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length, hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (
        24 * checkpoint_activations_factor * batch_size * seq_length * num_layers * (hidden_size**2)
    ) * (1.0 + (seq_length / (6.0 * hidden_size)) + (vocab_size / (16.0 * num_layers * hidden_size)))
    return flops_per_iteration


def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size
