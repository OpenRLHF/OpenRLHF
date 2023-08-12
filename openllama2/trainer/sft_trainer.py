from abc import ABC
import math

import torch
from openllama2.datasets import SFTDataset
from openllama2.models import GPTLMLoss
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from torch import nn
from transformers.trainer import get_scheduler

from .strategies import Strategy


class SFTTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy: Strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer = None,
        accumulated_gradient: int = 1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.accumulated_gradient = accumulated_gradient
        self.gradient_checkpointing = gradient_checkpointing

        # misc
        self.loss_fn = GPTLMLoss()

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def fit(self, use_lora):
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(range(self.train_dataloader.__len__()),
            desc='Train step of epoch %d' % epoch,
            disable=not self.strategy.is_rank_0())

            # train
            self.model.train()
            for prompts_id_len, inputs, attention_masks in self.train_dataloader:
                inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                logits = self.model(inputs, attention_mask=attention_mask, return_output=True)['logits']

                labels = torch.where(inputs.eq(self.tokenizer.pad_token_id), self.loss_fn.IGNORE_INDEX, inputs)
                if not self.pretrain_mode:
                    for label, source_len in zip(labels, prompts_id_len):
                        label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(logits, labels)
                # a patch for empty loss (the lenght of prompt > max_length)
                if torch.isnan(loss):
                    loss *= 0
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                step_bar.update()
                bar_dict = {'train loss': loss.item()}
                step_bar.set_postfix(self.strategy.all_reduce(bar_dict))

            # eval
            times= 0
            self.model.eval()
            with torch.no_grad():
                loss_sum = 0
                step_bar = tqdm(range(self.eval_dataloader.__len__()),
                desc='Eval stage + ',
                disable=not self.strategy.is_rank_0())

                for prompts_id_len, inputs, attention_masks in self.eval_dataloader:
                    inputs = inputs.squeeze(1).to(torch.cuda.current_device())
                    attention_mask = attention_masks.squeeze(1).to(torch.cuda.current_device())
                    logits = self.model(inputs, attention_mask=attention_mask, return_output=True)['logits']

                    labels = torch.where(inputs.eq(self.tokenizer.pad_token_id), self.loss_fn.IGNORE_INDEX, inputs)
                    if not self.pretrain_mode:
                        for label, source_len in zip(labels, prompts_id_len):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX
                    loss = self.loss_fn(logits, labels)

                    times += 1
                    loss_sum += loss.item()
                    bar_dict ={'eval loss': loss_sum / times}

                    step_bar.update()
                    step_bar.set_postfix(self.strategy.all_reduce(bar_dict))
                    
            epoch_bar.update()

