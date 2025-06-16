# ddpo_trainer.py
import sys
sys.modules
import os
import argparse
import torch
import json

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    default_data_collator,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_config,
    get_peft_model,
    TaskType
)
import deepspeed
from openrlhf.models import DDPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from tqdm import tqdm
from PIL import Image
import requests

from utils import get_changed_unchanged_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Dense DPO Trainer for Qwen/Qwen-VL-Chat with DeepSpeed and LoRA")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="../model_local")
    parser.add_argument("--dataset_name", type=str, default="openbmb/RLHF-V-Dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=2,
                        help="Weight of changed tokens contribute to DDPO Loss.")
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=-1, help="for deepspeed")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DDPOTrainer:
    def __init__(self, args, gamma, N, model, ref_model, tokenizer, train_loader, eval_loader, optimizer, scheduler):
        self.args = args
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = DDPOLoss(gamma)
        self.global_step = 0

        # TensorBoard
        if dist.get_rank() == 0:
            tb_dir = os.path.join(args.output_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.tb_writer = None

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        # 与 OpenRLHF 原版一样，合并 chosen/rejected 一次性前向
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            return torch.cat(
                [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
            )

        max_len_ids = max(chosen_ids.size(1), reject_ids.size(1))
        chosen_ids_padded = pad_to_length(chosen_ids, max_len_ids, self.tokenizer.pad_token_id)
        reject_ids_padded = pad_to_length(reject_ids, max_len_ids, self.tokenizer.pad_token_id)
        input_ids = torch.cat([
            chosen_ids_padded,
            reject_ids_padded
        ], dim=0)

        max_len_mask = max(c_mask.size(1), r_mask.size(1))
        attn_mask = torch.cat([
            pad_to_length(c_mask, max_len_mask, 0),
            pad_to_length(r_mask, max_len_mask, 0)
        ], dim=0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
            return_loss=False,
        )
        # 输出 log-probs: 需要手动从 logits->log_probs
        logits = outputs.logits  # [2*batch_size, seq_len, vocab_size]
        B2, _, _ = logits.size()
        B = B2//2
        shifted_logits = logits[:, :-1, :].contiguous()  # [2*bsz, seq_len-1, vocab_size]
        shift_mask = attn_mask[:, 1:].contiguous()  # [2*bsz, seq_len-1]

        return shifted_logits[:B], shifted_logits[B:], shift_mask, chosen_ids_padded, reject_ids_padded

    def train(self):
        self.model.train()
        self.ref_model.eval()
        total_acc = 0.0
        total_loss = 0.0
        for epoch in range(self.args.max_epochs):
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch} (rank {dist.get_rank()})")
            for batch in loop:
                chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
                c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
                reject_ids = batch["rejected_input_ids"].to(torch.cuda.current_device())
                r_mask = batch["rejected_attention_mask"].to(torch.cuda.current_device())
                prompt_id_lens = batch["prompt_length"].to(torch.cuda.current_device())
                diff_indexes = batch["diff_indexes"].to(torch.cuda.current_device())

                # Forward
                policy_chosen_logps, policy_rejected_logps, policy_pad_mask, \
                policy_chosen_ids_padded, policy_reject_ids_padded = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )
                with torch.no_grad():
                    ref_chosen_logps, ref_rejected_logps, _, \
                    _, _ = self.concatenated_forward(
                        self.model, chosen_ids, c_mask, reject_ids, r_mask
                    )
                
                changed_mask, unchanged_mask = get_changed_unchanged_mask(policy_chosen_ids_padded, policy_reject_ids_padded)
                loss = self.loss_fn(torch.cat([policy_chosen_logps, policy_rejected_logps], dim=0),
                                    torch.cat([ref_chosen_logps, ref_rejected_logps], dim=0),
                                    policy_pad_mask,
                                    changed_mask,
                                    unchanged_mask)

                self.model.backward(loss)
                self.model.step()
                self.global_step += 1

                total_loss += loss.item()

                # Logging
                if self.global_step % self.args.logging_steps == 0 and dist.get_rank() == 0:
                    avg_loss = total_loss / self.args.logging_steps
                    self.tb_writer.add_scalar("train/loss", avg_loss, self.global_step)
                    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    self.tb_writer.add_scalar("train/max_mem_GB", mem_alloc, self.global_step)
                    total_loss = 0.0

                # Eval
                if self.global_step % self.args.eval_steps == 0:
                    self.evaluate()

                # Save
                if self.global_step % self.args.save_steps == 0 and dist.get_rank() == 0:
                    ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-step{self.global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    self.model.save_checkpoint(ckpt_path, client_state={"step": self.global_step})

            # End epoch
        if dist.get_rank() == 0:
            self.tb_writer.close()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        loop = tqdm(self.eval_loader, desc=f"Eval (rank {dist.get_rank()})", leave=False)
        for batch in loop:
            chosen_ids = batch["chosen_input_ids"].to(torch.cuda.current_device())
            c_mask = batch["chosen_attention_mask"].to(torch.cuda.current_device())
            reject_ids = batch["rejected_input_ids"].to(torch.cuda.current_device())
            r_mask = batch["rejected_attention_mask"].to(torch.cuda.current_device())
            prompt_id_lens = batch["prompt_length"].to(torch.cuda.current_device())

            chosen_logps, rejected_logps, _, _ = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )
            ref_chosen_logps, ref_rejected_logps, _, _ = self.concatenated_forward(
                self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
            )

            pref_loss, chosen_reward, reject_reward = self.loss_fn(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
            acc = (chosen_reward > reject_reward).float().mean().item()
            total_loss += pref_loss.item()
            total_acc += acc
            cnt += 1

        avg_loss = total_loss / max(cnt, 1)
        avg_acc = total_acc / max(cnt, 1)
        if dist.get_rank() == 0:
            self.tb_writer.add_scalar("eval/loss", avg_loss, self.global_step)
            self.tb_writer.add_scalar("eval/acc", avg_acc, self.global_step)
        self.model.train()

def main():
    args = parse_args()
    set_seed(args.seed)

    # 设置环境变量
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '100'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    # 初始化分布式环境
    deepspeed.init_distributed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(args.local_rank)

    # 加载 tokenizer & processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.pad_token = '<|endoftext|>'
    processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, trust_remote_code=True)
    
    
    # 加载数据集
    raw_datasets = load_dataset(args.dataset_name)
    def preprocess_fn(examples):
        images = [processor(
            images=img,
            text=json.loads(text)['question'],  # 添加问题文本
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )['input_ids'][0]
            for img, text in zip(examples['image'], examples['text'])]
        
        # 解析JSON字符串并提取文本
        texts = [json.loads(text) for text in examples['text']]
        chosen_texts = [text['chosen'] for text in texts]
        rejected_texts = [text['rejected'] for text in texts]
        chosen_texts_tokenized = [tokenizer.tokenize(seq) for seq in chosen_texts]
        rejected_texts_tokenized = [tokenizer.tokenize(seq) for seq in rejected_texts]
        diff_indexes = get_changed_unchanged_mask(chosen_texts_tokenized, rejected_texts_tokenized)
        
        chosen_enc = tokenizer(chosen_texts, padding="max_length", truncation=True, max_length=512)
        rejected_enc = tokenizer(rejected_texts, padding="max_length", truncation=True, max_length=512)

        # 计算prompt长度
        prompt_lengths = [len(tokenizer.encode(text['question'])) + len(processor(
            images=img,
            text=text['question'],
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )['input_ids'][0])
            for text, img in zip(texts, examples['image'])]
        
        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "prompt_length": prompt_lengths,
            "pixel_values": images,
            "diff_indexes": diff_indexes
        }

    tokenized_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # DataLoader + DistributedSampler
    train_sampler = DistributedSampler(tokenized_datasets["train"], shuffle=True)
    eval_sampler = DistributedSampler(tokenized_datasets["validation"], shuffle=False)
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=default_data_collator,
        num_workers=4,
        pin_memory=True
    )
    eval_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        num_workers=4,
        pin_memory=True
    )

    # 配置 DeepSpeed
    ds_config_path = "openrlhf/configs/ddpo_ds_config.json"
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型 & 参考模型
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
        resume_download=True,
        force_download=False,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
        resume_download=True,
        force_download=False,
    )
    # 模型启用gradient checkpoint
    model.gradient_checkpointing_enable()
    ref_model.gradient_checkpointing_enable()

    # 应用 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"] # TODO(chuanwei.tang): 模块名称需要看一下Qwen-VL-Chat的模块名称
    )
    model = get_peft_model(model, peft_config)
    ref_model = get_peft_model(ref_model, peft_config)

    # 将 ref_model 冻结，并确保不更新 LoRA 权重
    for param in ref_model.parameters():
        param.requires_grad = False


    # 构建 optimizer & scheduler
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "lora" in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # 初始化 DeepSpeed 模型
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=optimizer_grouped_parameters,
        config_params=ds_config_path,
        training_data=tokenized_datasets["train"],
        train_batch_size=args.train_batch_size,
        train_micro_batch_size_per_gpu=args.train_batch_size,
        dataloader=train_loader
    )

    # 参考模型无需 DeepSpeed 包装，直接放到 GPU0
    ref_model = ref_model.half().to(torch.device(f"cuda:{args.local_rank}"))

    trainer = DDPOTrainer(
        args=args,
        model=model_engine,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    trainer.train()


if __name__ == "__main__":
    main()
