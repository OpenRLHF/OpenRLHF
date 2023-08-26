import math
from abc import ABC

import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openllama2.models import LogExpLoss, PairWiseLoss


class RewardModelTrainer(ABC):
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
            strategy,
            optim: Optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            max_norm=0.5,
            batch_size: int = 1,
            max_epochs: int = 2,
            only_evaluate=False,
            loss="sigmoid",
            gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs if not only_evaluate else 1
        self.only_evaluate = only_evaluate
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.gradient_checkpointing = gradient_checkpointing

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print('LogSigmoid Loss')
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print('LogExp Loss')

        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._wandb = None
        if self.strategy.args.use_wandb:
            import wandb
            self._wandb = wandb
            wandb.login(key=strategy.args.use_wandb)
            wandb.init(entity=strategy.args.wandb_org, project=strategy.args.wandb_project,
                       group=strategy.args.wandb_group, name=strategy.args.wandb_run_name,
                       config=strategy.args.__dict__, reinit=True)

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

    def fit(self, use_lora):
        global_step = 0
        epoch_bar = tqdm(range(self.epochs), desc='Train epoch', disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            if not self.only_evaluate:
                step_bar = tqdm(range(self.train_dataloader.__len__()),
                                desc='Train step of epoch %d' % epoch,
                                disable=not self.strategy.is_rank_0())

                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)

                self.model.train()
                for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                    reject_reward = self.model(reject_ids, attention_mask=r_mask)
                    loss = self.loss_fn(chosen_reward, reject_reward)

                    self.strategy.backward(loss, self.model, self.optimizer)
                    self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                    step_bar.update()
                    bar_dict = {'train loss': loss.item()}
                    logs = self.strategy.all_reduce(bar_dict)
                    step_bar.set_postfix(logs)
                    global_step += 1
                    if self._wandb is not None and self.strategy.is_rank_0() and global_step % 1000 == 0:
                        logs = {'train/%s' % k: v for k, v in {**logs, "global_step": global_step}.items()}
                        self._wandb.log(logs)

            step_bar = tqdm(range(self.eval_dataloader.__len__()),
                            desc='Eval stage of epoch %d' % epoch,
                            disable=not self.strategy.is_rank_0())
            # eval
            self.model.eval()
            with torch.no_grad():
                acc = 0
                rewards = []
                loss_sum = 0
                for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                    chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                    c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                    reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                    r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                    chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                    reject_reward = self.model(reject_ids, attention_mask=r_mask)
                    loss = self.loss_fn(chosen_reward, reject_reward)

                    rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                    acc += (chosen_reward > reject_reward).float().mean().item()
                    loss_sum += loss.item()
                    step_bar.update()

                acc_mean = acc / self.eval_dataloader.__len__()
                loss_mean = loss_sum / self.eval_dataloader.__len__()

                rewards = torch.cat(rewards).float()
                rewards = self.strategy.all_gather(rewards)
                reward_mean = torch.mean(rewards)
                reward_std = torch.std(rewards).clamp(min=1e-8)

                # save mean std
                self.strategy.print("Set reward mean std")
                self.model.mean[0] = reward_mean
                self.model.std[0] = reward_std

                bar_dict = {'eval loss': loss_mean, 'acc_mean': acc_mean, 'reward_mean': reward_mean.item(),
                            'reward_std': reward_std.item()}
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

                histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
                self.strategy.print('histgram')
                self.strategy.print(histgram)

                if self._wandb is not None and self.strategy.is_rank_0():
                    logs = {'eval/%s' % k: v for k, v in {**logs, "epoch": epoch}.items()}
                    self._wandb.log(logs)

            epoch_bar.update()
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
