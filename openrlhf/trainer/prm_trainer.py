from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import PRMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.utils import convert_token_to_id


class ProcessRewardModelTrainer(ABC):
    """
    Trainer for training a process reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        # set placeholder token
        self.placeholder_token_id = convert_token_to_id(strategy.args.placeholder_token, self.tokenizer)
        self.reward_token_ids = self.args.reward_tokens
        if self.reward_token_ids is not None:
            self.reward_token_ids = [convert_token_to_id(token, self.tokenizer) for token in self.reward_token_ids]

        self.ignore_index = -100
        self.loss_fn = PRMLoss(self.placeholder_token_id, self.reward_token_ids)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb setting
        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0
        acc_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            for data in self.train_dataloader:
                if not self.packing_samples:
                    inputs, attention_masks, labels = data
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                    labels = labels.to(torch.cuda.current_device())
                    packed_seq_lens = None
                else:
                    inputs, attention_masks, packed_seq_lens, labels = data
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                    labels = labels.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=packed_seq_lens,
                )

                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0

                prm_loss, acc = self.loss_fn(inputs, output.logits, labels, return_acc=True)
                loss = prm_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += loss.item()
                acc_sum += acc.item()
                logs_dict = {
                    "prm_loss": prm_loss.item(),
                    "acc": acc.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            )

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            acc_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for data in eval_dataloader:
                if not self.packing_samples:
                    inputs, attention_masks, labels = data
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                    labels = labels.to(torch.cuda.current_device())
                    packed_seq_lens = None
                else:
                    inputs, attention_masks, packed_seq_lens, labels = data
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                    labels = labels.to(torch.cuda.current_device()).squeeze(1)

                output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=packed_seq_lens,
                )

                loss, acc = self.loss_fn(inputs, output.logits, labels, return_acc=True)

                times += 1
                loss_sum += loss.item()
                acc_sum += acc.item()
                bar_dict = {"eval prm_loss": loss_sum / times, "eval acc": acc_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()  # reset model state
