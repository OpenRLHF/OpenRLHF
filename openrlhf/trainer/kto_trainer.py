import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import KTOLoss
from openrlhf.models.utils import log_probs_from_logits
from openrlhf.utils.distributed_sampler import DistributedSampler


class KTOTrainer(ABC):
    """
    Trainer for KTO training.

    Args:
        model (torch.nn.Module): The primary model to be trained.
        ref_model (torch.nn.Module): The reference model for comparing and guiding preference.
        strategy (Strategy): The strategy to use for training.
        tokenizer (Tokenizer): The tokenizer for processing input data.
        optim (Optimizer): The optimizer for training the model.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to control learning rate during training.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        beta (float, defaults to 0.01): Coefficient for regularizing the preference loss.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
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

        self.beta = beta
        self.loss_fn = KTOLoss(
            self.beta,
            self.args.desirable_loss_weight,
            self.args.undesirable_loss_weight,
            self.strategy.world_size,
            torch.cuda.current_device(),
        )

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
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

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

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

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        loss_sum = 0
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

            self.model.train()
            self.ref_model.eval()

            # train
            for input_ids, attention_mask, labels, prompt_ids_lens in self.train_dataloader:
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())

                # make sure local batch size >= 2 (to pack unmatched examples)
                policy_returns = self.compute_model_logps_with_KL(
                    self.model, input_ids, attention_mask, labels, prompt_ids_lens
                )
                aux_loss = policy_returns[3]

                with torch.no_grad():
                    ref_returns = self.compute_model_logps_with_KL(
                        self.ref_model, input_ids, attention_mask, labels, prompt_ids_lens
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

                loss_sum += loss.item()
                logs_dict = {
                    "kto_loss": loss.item(),
                    "chosen_reward": chosen_rewards.mean().item() if len(chosen_rewards) != 0 else 0,
                    "reject_reward": rejected_rewards.mean().item() if len(rejected_rewards) != 0 else 0,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                logs_dict["kl"] = KL.item()
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            loss_sum = 0
            chosen_reward, reject_reward = 0, 0
            for input_ids, attention_mask, labels, prompt_ids_lens in eval_dataloader:
                input_ids = input_ids.squeeze(1).to(torch.cuda.current_device())
                attention_mask = attention_mask.squeeze(1).to(torch.cuda.current_device())

                # make sure local batch size >= 2 (to pack unmatched examples)
                policy_returns = self.compute_model_logps_with_KL(
                    self.model, input_ids, attention_mask, labels, prompt_ids_lens
                )
                aux_loss = policy_returns[3]

                with torch.no_grad():
                    ref_returns = self.compute_model_logps_with_KL(
                        self.ref_model, input_ids, attention_mask, labels, prompt_ids_lens
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

            loss_mean = loss_sum / eval_dataloader.__len__()
            chosen_reward = chosen_reward / eval_dataloader.__len__()
            reject_reward = reject_reward / eval_dataloader.__len__()

            logs = {"eval_loss": loss_mean, "chosen_reward": chosen_reward, "reject_reward": reject_reward}
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()

    def compute_model_logps_with_KL(self, model, input_ids, attention_mask, labels, prompt_id_lens):
        """
        the front half is matched for spv, the latter half is unmatched for KL
        """
        hsize = input_ids.shape[0] // 2

        # front half
        chosen_logps, reject_logps, aux_loss = self.compute_model_logps(
            model, input_ids[:hsize], attention_mask[:hsize], labels[:hsize], prompt_id_lens[:hsize]
        )

        # latter half
        output = model(input_ids[hsize:], attention_mask=attention_mask[hsize:], return_output=True)
        all_logits = output["logits"]
        KL_logps = self._get_batch_logps(
            all_logits,
            input_ids[hsize:],
            attention_mask=attention_mask[hsize:],
            average_log_prob=False,
            prompt_id_lens=prompt_id_lens[hsize:],
        )
        return chosen_logps, reject_logps, KL_logps, aux_loss

    def compute_model_logps(self, model, input_ids, attention_mask, labels, prompt_id_lens):
        output = model(input_ids, attention_mask=attention_mask, return_output=True)
        all_logits = output["logits"]
        all_logps = self._get_batch_logps(
            all_logits, input_ids, attention_mask=attention_mask, average_log_prob=False, prompt_id_lens=prompt_id_lens
        )
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
        prompt_id_lens=[],
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

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[~loss_masks] = 0
        per_token_logps = log_probs_from_logits(logits, labels)

        if average_log_prob:
            return (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return (per_token_logps * loss_masks).sum(-1)
