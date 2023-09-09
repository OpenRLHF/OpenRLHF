import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openllama2.models import DPOLoss


class DPOTrainer(ABC):
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
        ref_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.gradient_checkpointing = gradient_checkpointing

        self.beta = beta
        self.loss_fn = DPOLoss(self.beta)

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
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
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

    def fit(self, use_lora):
        global_step = 0
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_logps = self.model(chosen_ids, attention_mask=c_mask)
                rejected_logps = self.model(reject_ids, attention_mask=r_mask)
                reference_chosen_logps = self.ref_model(chosen_ids, attention_mask=c_mask)
                reference_rejected_logps = self.ref_model(reject_ids, attention_mask=r_mask)

                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )

                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                step_bar.update()
                global_step += 1
                logs = {"train_loss": loss.item()}
                logs["chosen_reward"] = chosen_reward.item()
                logs["reject_reward"] = reject_reward.item()
                logs = self.strategy.all_reduce(logs)
                step_bar.set_postfix(logs)
                if (
                    self._wandb is not None
                    and self.strategy.is_rank_0()
                    and global_step % self.strategy.accumulated_gradient == 0
                ):
                    logs = {"train/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                    self._wandb.log(logs)

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
