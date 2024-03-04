from abc import ABC

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from openrlhf.models import KTOLoss, VanillaKTOLoss


class KTOTrainer(ABC):
    """
        Trainer for KTO algorithms

    Args:
        model (torch.nn.Module): the model to train
        ref_model (torch.nn.Module): the reference model to provide reference logits
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
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm=0.5,
        beta=0.01,
        max_epochs: int = 2,
        vanilla_loss=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.vanilla_loss = vanilla_loss

        self.beta = beta
        if self.vanilla_loss:
            self.loss_fn = VanillaKTOLoss(self.beta)
        else:
            self.loss_fn = KTOLoss(self.beta, 1.0, 1.0, self.strategy.world_size, torch.cuda.current_device())

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

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

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            self.ref_model.eval()
            loss_mean = 0

            # train
            for input_ids, attention_mask, labels in self.train_dataloader:
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())

                if self.vanilla_loss:
                    policy_chosen_logps, policy_reject_logps, aux_loss = self.compute_model_logps(
                        self.model, input_ids, attention_mask, labels
                    )
                    with torch.no_grad():
                        ref_chosen_logps, ref_reject_logps, _ = self.compute_model_logps(
                            self.ref_model, input_ids, attention_mask, labels
                        )

                    kto_loss, chosen_rewards, rejected_rewards = self.loss_fn(
                        policy_chosen_logps, policy_reject_logps, ref_chosen_logps, ref_reject_logps
                    )
                else:
                    # make sure local batch size >= 2 (to pack unmatched examples)
                    policy_returns = self.compute_model_logps_with_KL(self.model, input_ids, attention_mask, labels)
                    aux_loss = policy_returns[3]

                    with torch.no_grad():
                        ref_returns = self.compute_model_logps_with_KL(
                            self.ref_model, input_ids, attention_mask, labels
                        )

                    kto_loss, chosen_rewards, rejected_rewards, KL = self.loss_fn(
                        policy_returns[0],
                        policy_returns[1],
                        policy_returns[2],
                        ref_returns[0],
                        ref_returns[1],
                        ref_returns[2],
                    )

                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = kto_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_mean = loss_mean * 0.9 + 0.1 * loss.item()

                logs_dict = {
                    "kto_loss": loss.item(),
                    "chosen_reward": chosen_rewards.mean().item() if len(chosen_rewards) != 0 else 0,
                    "reject_reward": rejected_rewards.mean().item() if len(rejected_rewards) != 0 else 0,
                    "loss_mean": loss_mean,
                }
                if not self.vanilla_loss:
                    logs_dict["kl"] = KL

                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        # logs
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(self.eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            loss_sum = 0
            chosen_reward, reject_reward = 0, 0
            for input_ids, attention_mask, labels in self.eval_dataloader:
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())

                if self.vanilla_loss:
                    policy_chosen_logps, policy_reject_logps, _ = self.compute_model_logps(
                        self.model, input_ids, attention_mask, labels
                    )
                    ref_chosen_logps, ref_reject_logps, _ = self.compute_model_logps(
                        self.ref_model, input_ids, attention_mask, labels
                    )

                    kto_loss, chosen_rewards, rejected_rewards = self.loss_fn(
                        policy_chosen_logps, policy_reject_logps, ref_chosen_logps, ref_reject_logps
                    )
                else:
                    # make sure local batch size >= 2 (to pack unmatched examples)
                    policy_returns = self.compute_model_logps_with_KL(self.model, input_ids, attention_mask, labels)
                    aux_loss = policy_returns[3]

                    with torch.no_grad():
                        ref_returns = self.compute_model_logps_with_KL(
                            self.ref_model, input_ids, attention_mask, labels
                        )

                    kto_loss, chosen_rewards, rejected_rewards, KL = self.loss_fn(
                        policy_returns[0],
                        policy_returns[1],
                        policy_returns[2],
                        ref_returns[0],
                        ref_returns[1],
                        ref_returns[2],
                    )

                chosen_reward += chosen_rewards.mean().item()
                reject_reward += rejected_rewards.mean().item()
                loss_sum += kto_loss.item()
                step_bar.update()

            loss_mean = loss_sum / self.eval_dataloader.__len__()
            chosen_reward = chosen_reward / self.eval_dataloader.__len__()
            reject_reward = reject_reward / self.eval_dataloader.__len__()

            logs = {"eval_loss": loss_mean, "chosen_reward": chosen_reward, "reject_reward": reject_reward}
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)
        self.model.train()

    def compute_model_logps_with_KL(self, model, input_ids, attention_mask, labels):
        """
        the front half is matched for spv, the latter half is unmatched for KL
        """
        hsize = input_ids.shape[0] // 2

        # front half
        chosen_logps, reject_logps, aux_loss = self.compute_model_logps(
            model, input_ids[:hsize], attention_mask[:hsize], labels[:hsize]
        )

        # latter half
        output = model(input_ids[hsize:], attention_mask=attention_mask[hsize:], return_output=True)
        all_logits = output["logits"]
        KL_logps = self._get_batch_logps(
            all_logits, input_ids[hsize:], attention_mask=attention_mask[hsize:], average_log_prob=False
        )
        return chosen_logps, reject_logps, KL_logps, aux_loss

    def compute_model_logps(self, model, input_ids, attention_mask, labels):
        output = model(input_ids, attention_mask=attention_mask, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(all_logits, input_ids, attention_mask=attention_mask, average_log_prob=False)
        chosen_logps = all_logps[labels == 1]
        reject_logps = all_logps[labels == 0]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, reject_logps, aux_loss

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = attention_mask[:, 1:].bool()
        # dummy token; we'll ignore the losses on these tokens later
        labels[~loss_mask] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        return (per_token_logps * loss_mask).sum(-1)
