import argparse
import multiprocessing as mp
import os
from pathlib import Path

import jsonlines
import torch
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def load_config_from_yaml(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _worker_vllm_generate(
    llm: LLM,
    prompts: list,
    metadatas: list,
    sampling_params: SamplingParams,
    save_prop: int = 10,
    use_tqdm: bool = False,
):
    # Divide prompts into batches and yield
    batch_size = max(1, len(prompts) // save_prop)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + min(batch_size, len(prompts) - i)]
        batch_metadatas = metadatas[i : i + min(batch_size, len(metadatas) - i)]
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=use_tqdm)
        yield outputs, batch_metadatas


def batch_generate_vllm(args):
    N = args.best_of_n

    # configure strategy
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = lambda: True
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    kwargs = {}
    is_mistral = (
        "magistral" in args.pretrain.lower()
        or "ministral" in args.pretrain.lower()
        or "molstral" in args.pretrain.lower()
        or "mistral" in args.pretrain.lower()
    )
    if is_mistral:
        if "molstral" not in args.pretrain.lower():
            kwargs = dict(
                reasoning_parser="mistral", tokenizer_mode="mistral", config_format="mistral", load_format="mistral"
            )
        else:
            kwargs = dict(tokenizer_mode="mistral")
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        max_logprobs=256,
        **kwargs,
    )

    # Create a sampling params object.
    skip_spe_toks = is_mistral

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        max_count=args.max_samples,
    )
    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    prompts_dataset = PromptDataset(
        prompts_data, tokenizer, dummy_strategy, input_template=args.input_template, return_tokens=True
    )
    # if outpout path exists, open it and extract prompt_id to avoid duplicate generation
    if os.path.exists(args.output_path + ".jsonl"):
        existing_prompt_ids_count = {}
        with jsonlines.open(args.output_path + ".jsonl", mode="r") as reader:
            for obj in reader:
                if "metadata" in obj and "prompt_id" in obj["metadata"]:
                    prompt_id = obj["metadata"]["prompt_id"]
                    if prompt_id not in existing_prompt_ids_count:
                        existing_prompt_ids_count[prompt_id] = 0
                    existing_prompt_ids_count[prompt_id] += 1
        existing_prompt_ids = set([k for k, v in existing_prompt_ids_count.items() if v == N])

        print(f"Found {len(existing_prompt_ids)} existing prompt_ids in {args.output_path + '.jsonl'}")
        # filter prompts_dataset to remove existing prompt_ids
        prompts_dataset = [
            item for item in prompts_dataset if item[2].get("prompt_id", None) not in existing_prompt_ids
        ]
        print(f"{len(prompts_dataset)} prompts left after filtering existing prompt_ids")
    prompts = []
    metadatas = []
    for _, prompt, metadata in prompts_dataset:
        prompts.append(TokensPrompt(prompt_token_ids=prompt))
        metadatas.append(metadata)

    # Conditional SFT inference
    if args.enable_csft:
        for i in range(len(prompts)):
            prompts[i] += args.csft_prompt.strip() + " "

    # best of n
    if "<end_of_turn>" in tokenizer.vocab:
        stop_tokens_ids = [tokenizer.convert_tokens_to_ids("<end_of_turn>"), tokenizer.eos_token_id]
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            skip_special_tokens=skip_spe_toks,
            truncate_prompt_tokens=args.prompt_max_len,
            include_stop_str_in_output=True,
            stop_token_ids=stop_tokens_ids,
            n=N,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            skip_special_tokens=skip_spe_toks,
            truncate_prompt_tokens=args.prompt_max_len,
            include_stop_str_in_output=True,
            n=N,
        )
    for batched_outputs, batched_metadatas in _worker_vllm_generate(
        llm, prompts, metadatas, sampling_params, save_prop=args.save_prop, use_tqdm=True
    ):
        output_dataset = []
        for output, metadata in zip(batched_outputs, batched_metadatas):
            prompt = output.prompt
            for i in range(len(output.outputs)):
                out_text = output.outputs[i].text
                output_dataset.append({"input": prompt, "output": out_text, "metadata": metadata})
        with jsonlines.open(args.output_path + ".jsonl", mode="a") as writer:
            writer.write_all(output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--eval_task", type=str, default=None, help="Set to generate_vllm, generate (HF generate) or rm"
    )
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO Stage")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed cli")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16 for deepspeed")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAtten2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--full_determinism",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")
    parser.add_argument(
        "--value_head_prefix", type=str, default="value_head", help="value_head prefix for Reward Model"
    )

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default="output", help="JSON dataset key")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="HF tokenizer apply_chat_template"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output_path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for prompt")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens in generation")
    parser.add_argument("--greedy_sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--best_of_n", type=int, default=1, help="Number of responses to generate per prompt")
    parser.add_argument(
        "--post_processor",
        type=str,
        default=None,
        help="set to rs (Rejection Sampling), csft (Conditional SFT), iter_dpo (Iterative DPO) or None",
    )

    # For vllm
    parser.add_argument("--tp_size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size",
    )
    parser.add_argument("--rollout_batch_size", type=int, default=2048, help="Number of samples to generate")

    # For Conditional SFT
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--reward_template", type=str, default=None)
    parser.add_argument("--enable_csft", action="store_true", default=False)
    parser.add_argument("--csft_prompt", type=str, default="<rm_score>: 5.00", help="Conditional SFT prompt")

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    parser.add_argument(
        "--use_tool_calls",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_prop",
        type=int,
        default=10,
        help="Proportion of generations to checkpoint during generation, e.g. save_prop=10 means saving every 1/10 generations",
    )

    args = parser.parse_args()

    # Load config from YAML file if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")

        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        yaml_config = load_config_from_yaml(args.config)

        # Create a namespace object from the YAML config
        config_args = argparse.Namespace(**yaml_config)

        # Merge: CLI arguments override YAML config values
        for key, value in vars(args).items():
            if key == "config":
                continue
            if key not in yaml_config:
                setattr(config_args, key, value)

        args = config_args

    # If output_path has {iter}, replace it with iter number
    args.output_path = args.output_path.format(iter=args.iter if args.iter is not None else "final")
    # Create all output Path
    path = Path(args.output_path).parent
    path.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {args.output_path}.jsonl")

    if args.eval_task and args.eval_task == "generate":
        raise NotImplementedError
    if args.eval_task and args.eval_task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.eval_task and args.eval_task == "rm":
        raise NotImplementedError
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
