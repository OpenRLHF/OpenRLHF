import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models.loss import aggregate_loss
from openrlhf.models.utils import compute_approx_kl
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.loss_utils import get_loss_batch_info


def build_generation_mask(sequences, prompt_len, prompt_attention_mask, eos_token_id):
    """ Build a full attention mask and the response ``action_mask`` for the generated tokens. 
    ``sequences`` is (B, prompt_len + gen_len) from model.generate. Anything after first EOS is masked
    """
    B = sequences.size(0)
    device = sequences.device
    gen = sequences[:, prompt_len:]
    gen_len = gen.size(1)
    
    is_eos = gen == eos_token_id
    has_eos = is_eos.any(dim=1)
    
    first_eos = torch.where(
        has_eos,
        is_eos.float().argmax(dim=1).long(),
        torch.full((B, ), gen_len - 1, device=device, dtype=torch.long)
    )
    positions = torch.arange(gen_len, device=device).unsqueeze(0).expand(B, -1)
    action_mask = (positions <= first_eos.unsqueeze(1)).long()
    
    attention_mask = torch.cat([prompt_attention_mask.to(device), action_mask], dim=1)
    return attention_mask, action_mask

def align_response_logits(logits, num_actions):
    return logits[:, -num_actions -1:-1, :]


def compute_opd_loss(
    student_action_log_probs, 
    teacher_action_log_probs,
    action_mask, 
    kl_estimator="k3",
    loss_batch_info=None, 
):
    teacher_action_log_probs = teacher_action_log_probs.detach()
    log_ratio = student_action_log_probs - teacher_action_log_probs
    kl = compute_approx_kl(student_action_log_probs, teacher_action_log_probs, kl_estimator=kl_estimator)
    
    surrogate = log_ratio.detach() * student_action_log_probs
    loss = kl.detach() + surrogate - surrogate.detach()
    return aggregate_loss(loss, action_mask, **(loss_batch_info or {}))
    
    
def compute_topk_token_kl(
    student_resp_logits, 
    teacher_resp_logits, 
    topk=5,
    temperature=1.0,
):
    topk = min(topk, student_resp_logits.size(-1))
    student_log_probs = torch.log_softmax(student_resp_logits / temperature, dim=-1)
    with torch.no_grad():
        teacher_log_probs = torch.log_softmax(teacher_resp_logits.detach() / temperature, dim=-1)
    # Compute the top-k KD loss
    _, top_idx = torch.topk(student_log_probs, k=topk, dim=-1)
    log_q = torch.gather(student_log_probs, dim=-1, index=top_idx)
    log_p = torch.gather(teacher_log_probs, dim=-1, index=top_idx)
    
    q = log_q.exp()
    p = log_p.exp()
    kl = (q * (log_q - log_p)).sum(dim=-1)
    if topk < student_resp_logits.size(-1):
        q_tail = (1.0 - q.sum(dim=-1)).clamp(min=1e-7, max=1.0)
        p_tail = (1.0 - p.sum(dim=-1)).clamp(min=1e-7, max=1.0)
        kl = kl + (q_tail * (q_tail.log() - p_tail.log()))
    
    return kl

def compute_topk_kd_loss(
    student_resp_logits, 
    teacher_resp_logits, 
    action_mask, 
    topk=5,
    temperature=1.0,
    loss_batch_info=None, 
):
    kl =  compute_topk_token_kl(student_resp_logits, teacher_resp_logits, topk, temperature)
    return aggregate_loss(kl, action_mask, **(loss_batch_info or {}))

class OPDTrainer(ABC):
    def __init__(
        self, 
        model, 
        teacher_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        scheduler,
        max_norm = 1.0,
        max_epochs: int = 1,
        eval_dataloader=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.max_norm = max_norm
        self.epochs = max_epochs
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.model = model
        self.teacher_model = teacher_model
        
        self.kl_estimator = self.args.opd.kl_estimator
        self.topk = self.args.opd.topk
        self.kd_temperature = self.args.opd.kd_temperature
        self.prompt_max_len = self.args.data.prompt_max_len
        self.generate_kwargs = dict(
            max_new_tokens=self.args.generate.max_new_tokens,
            do_sample=True,
            temperature=self.args.generate.temperature,
            top_p=self.args.generate.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        self.aux_loss = self.args.model.aux_loss_coef > 1e-8
        
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.logger.wandb.key and self.strategy.is_rank_0():
            import wandb
            
            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.logger.wandb.key)
            wandb.init(
                entity=strategy.args.logger.wandb.org,
                project=strategy.args.logger.wandb.project,
                group=strategy.args.logger.wandb.group,
                name=strategy.args.logger.wandb.run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        
        if self.strategy.args.logger.tensorboard_dir and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter
            
            os.makedirs(self.strategy.args.logger.tensorboard_dir, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.logger.tensorboard_dir, self.strategy.args.logger.wandb.run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)
            
    @torch.no_grad()
    def generate_samples(self, prompts):
        device = next(self.model.parameters()).device
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.prompt_max_len, 
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(device)
        prompt_attention_mask = inputs["attention_mask"].to(device)
        prompt_len = input_ids.size(1)
        
        was_training = self.model.training
        self.model.eval()
        
        try:
            generation_model = self.strategy._unwrap_model(self.model)
            sequences = generation_model.generate(
                input_ids=input_ids,
                attention_mask=prompt_attention_mask,
                **self.generate_kwargs
            )
        finally:
            self.model.train(was_training)
            
        attention_mask, action_mask = build_generation_mask(sequences, prompt_len, prompt_attention_mask, self.tokenizer.eos_token_id)
        return sequences, attention_mask, action_mask
    
    @torch.no_grad()
    def batched_teacher_log_probs(self, sequences, attention_mask, action_mask):
        micro = self.args.train.micro_batch_size
        chunks = []
        for i in range(0, sequences.size(0), micro):
            sl = slice(i, i + micro)
            chunks.append(
                self.teacher_model(
                    sequences[sl],
                    attention_mask=attention_mask[sl],
                    action_mask=action_mask[sl],
                    ring_attn_group=self.strategy.ring_attn_group,
                )
            )
        return torch.cat(chunks, dim=0)
    
    def compute_loss(self, sequences, attention_mask, action_mask, teacher_log_probs=None, loss_batch_info=None):
        student_log_probs, student_output =  self.model(
            sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            return_output=True,
        )
        
        teacher_output = None
        if self.topk > 0 or teacher_log_probs is None:
            with torch.no_grad():
                teacher_log_probs, teacher_output = self.teacher_model(
                    sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    ring_attn_group=self.strategy.ring_attn_group,
                    return_output=True,
                )
                
        if self.topk > 0:
            num_actions = action_mask.size(1)
            opd_loss = compute_topk_kd_loss(
                align_response_logits(student_output['logits'], num_actions),
                align_response_logits(teacher_output['logits'], num_actions),
                action_mask,
                self.topk,
                self.kd_temperature,
                loss_batch_info=loss_batch_info,
            )
        else:
            opd_loss = compute_opd_loss(
                student_log_probs,
                teacher_log_probs,
                action_mask,
                self.kl_estimator,
                loss_batch_info=loss_batch_info,
            )
        
        aux_loss = student_output.aux_loss if (self.aux_loss and "aux_loss" in student_output) else 0
        return opd_loss, aux_loss
    
    @torch.no_grad()
    def compute_token_kl(self, sequences, attention_mask, action_mask):
        if self.topk > 0:
            _, student_output = self.model(
                sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                ring_attn_group=self.strategy.ring_attn_group,
                return_output=True,
            )
            _, teacher_output = self.teacher_model(
                sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                ring_attn_group=self.strategy.ring_attn_group,
                return_output=True,
            )
            num_actions = action_mask.size(1)
            return compute_topk_token_kl(
                align_response_logits(student_output['logits'], num_actions),
                align_response_logits(teacher_output['logits'], num_actions),
                topk=self.topk,
                temperature=self.kd_temperature,
            )
        student_log_probs = self.model(
            sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            ring_attn_group=self.strategy.ring_attn_group,
        )
        teacher_log_probs = self.teacher_model(
            sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            ring_attn_group=self.strategy.ring_attn_group,
        )
        return compute_approx_kl(student_log_probs, teacher_log_probs, kl_estimator=self.kl_estimator)
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader, steps=0):
        was_training = self.model.training
        self.teacher_model.eval()
        step_bar = tqdm(range(len(eval_dataloader)), desc="Eval Step", disable=not self.strategy.is_rank_0())
        
        device = next(self.model.parameters()).device
        kl_sum = torch.zeros((), dtype=torch.float32, device=device)
        token_sum = torch.zeros((), dtype=torch.float32, device=device)
        seq_sum = torch.zeros((), dtype=torch.float32, device=device)
        
        try:
            for _datasources, prompts, _labels, _images in eval_dataloader:
                sequences, attention_mask, action_mask = self.generate_samples(prompts)
                self.model.eval()
                token_kl = self.compute_token_kl(sequences, attention_mask, action_mask)
                
                mask = action_mask.float()
                
                kl_sum += (token_kl.float() * mask).sum()
                token_sum += mask.sum()
                seq_sum += mask.size(0)
                step_bar.update()
        finally:
            self.model.train(was_training)
            
        totals = self.strategy.all_reduce(torch.stack([kl_sum, token_sum, seq_sum]), op="sum")
        kl_total, token_total, seq_total = totals.tolist()
        logs = {
            "eval_kl": kl_total / max(token_total, 1.0),
            "eval_mean_response_length": token_total / max(seq_total, 1.0),
        }
        step_bar.set_postfix(logs)
        
        if self.strategy.is_rank_0():
            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        return logs

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if num_update_steps_per_epoch is None:
            num_update_steps_per_epoch = len(self.train_dataloader)
        if num_update_steps_per_epoch <= 0:
            raise ValueError("num_update_steps_per_epoch must be a positive integer")
        
        if args.eval.steps == -1:
            args.eval.steps = num_update_steps_per_epoch if self.eval_dataloader is not None else float('inf')
        if args.ckpt.save_steps == -1:
            args.ckpt.save_steps = float('inf')
            
        step = consumed_samples // args.train.batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train.batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train.batch_size)
        
        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train Epoch", disable=not self.strategy.is_rank_0())
        loss_sum = 0.0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples)
            
            step_bar = tqdm(range(len(self.train_dataloader)), desc="Train Step of epoch %d" % epoch, disable=not self.strategy.is_rank_0())    
            
            self.model.train()
            self.teacher_model.eval()
            
            accum = self.strategy.accumulated_gradient
            group, offsets = [], []
            for _datasources, prompts, _labels, _images in self.train_dataloader:
                offsets.append((len(group), len(group) + len(prompts)))
                group.extend(prompts)
                if len(offsets) < accum:
                    continue
            
                sequences, attention_mask, action_mask = self.generate_samples(group)
                
                loss_batch_info = get_loss_batch_info(self.strategy, action_mask)
                loss_batch_info["batch_num_tokens"] /= accum
                loss_batch_info["global_batch_size"] /= accum
                
                cached_teacher_lp = (
                    self.batched_teacher_log_probs(sequences, attention_mask, action_mask) if self.topk == 0 else None
                )
                
                for start, end in offsets:
                    sl = slice(start, end)
                    teacher_lp = cached_teacher_lp[sl] if cached_teacher_lp is not None else None
                    opd_loss, aux_loss = self.compute_loss(
                        sequences[sl],
                        attention_mask[sl],
                        action_mask[sl],
                        teacher_log_probs=teacher_lp,
                        loss_batch_info=loss_batch_info,
                    )
                    loss = opd_loss + aux_loss * self.args.model.aux_loss_coef
                    
                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                    
                    log_keys = ["opd_loss", "response_length", "lr", "grad_norm"]
                    log_values = [
                        opd_loss.detach(),
                        action_mask[sl].float().sum(dim=-1).mean(),
                        self.scheduler.get_last_lr()[0],
                        self.strategy.get_grad_norm(self.model),
                    ]
                    if self.aux_loss:
                        log_keys.append("aux_loss")
                        log_values.append(aux_loss.detach())
                    device = opd_loss.device
                    log_tensor = torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32, device=device) for v in log_values]
                    )
                    logs_dict = dict(zip(log_keys, self.strategy.all_reduce(log_tensor).tolist()))
                    loss_sum += logs_dict["opd_loss"]
                    
                    optimizer_boundary = step % accum == 0
                    if optimizer_boundary:
                        logs_dict["loss_mean"] = loss_sum / accum
                        loss_sum = 0.0
                    
                    step_bar.set_postfix(logs_dict)
                    step_bar.update()
                    
                    if optimizer_boundary:
                        global_step = step // accum
                        client_states = {"consumed_samples": global_step * args.train.batch_size}
                        self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                    
                    step += 1
                group, offsets = [], []
            
            epoch_bar.update()
        
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()
                    

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        client_states = client_states or {}
        if global_step % args.logger.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
        elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                    
        if global_step % args.eval.steps == 0 and self.eval_dataloader is not None:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        
        if global_step % args.ckpt.save_steps == 0:
            tag = f"global_step_{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model,
                    args.ckpt.path,
                    tag,
                    args.ckpt.max_num,
                    args.ckpt.max_mem,
                    client_states
                )
            if self.save_hf_ckpt:
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.ckpt.path, f"{tag}_hf"))