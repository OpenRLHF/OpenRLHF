import os
from abc import ABC

import torch
from flash_attn.utils.distributed import all_gather
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import DPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class DPOTrainer(ABC):
    """
    Trainer for Direct Preference Optimization (DPO) training.

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
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
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
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
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
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.beta = beta
        self.loss_fn = DPOLoss(self.beta, self.args.label_smoothing, self.args.ipo)

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.nll_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

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

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
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
            for data in self.train_dataloader:
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, aux_loss, nll_loss = self.model_forward(
                    data)

                # loss function
                preference_loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef + nll_loss * self.args.nll_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # dpo logs
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                grad_norm = self.model.model.get_global_grad_norm()
                if grad_norm is not None:
                    logs_dict.update({
                        "grad_norm":
                        grad_norm.detach().item()
                        if isinstance(grad_norm, torch.Tensor) else grad_norm
                    })

                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
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

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def model_forward(self, data):
        if not self.packing_samples:
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                    self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens)
        else:
            packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens = data
            packed_input_ids, packed_attention_masks = packed_input_ids.to(
                torch.cuda.current_device()), packed_attention_masks.to(torch.cuda.current_device())
            chosen_logps, rejected_logps, aux_loss, nll_loss = self.packed_samples_forward(
                self.model, packed_input_ids, packed_attention_masks, packed_seq_lens,
                prompt_id_lens)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, _, _ = self.packed_samples_forward(
                    self.ref_model, packed_input_ids, packed_attention_masks, packed_seq_lens,
                    prompt_id_lens)
        return chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, aux_loss, nll_loss

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        # logs
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
        if global_step % args.eval_steps == 0:
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
            acc_sum = 0
            loss_sum = 0
            times = 0
            for data in eval_dataloader:
                chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, aux_loss, _ = self.model_forward(
                    data)

                loss, chosen_reward, reject_reward = self.loss_fn(
                    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                acc_sum += (chosen_reward > reject_reward).float().mean().item()
                loss_sum += loss.item()
                times += 1
                step_bar.update()

            logs = {
                "eval_loss": loss_sum / times,
                "acc_mean": acc_sum / times,
            }
            logs = self.strategy.all_reduce(logs)
            step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )
        output = model(input_ids, attention_mask=att_masks, return_output=True)
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, input_ids, att_masks, prompt_id_lens, average_log_prob=False
        )
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
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
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[loss_masks == False] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means

    def packed_samples_forward(self, model, packed_input_ids, packed_attention_masks, packed_seq_lens, prompt_id_lens):
        output = model(
            packed_input_ids,
            attention_mask=packed_attention_masks,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._packed_get_batch_logps(
            all_logits,
            packed_input_ids,
            packed_attention_masks,
            prompt_id_lens * 2,
            packed_seq_lens,
            average_log_prob=False,
        )
        chosen_logps = all_logps_sum[: len(packed_seq_lens) // 2]
        rejected_logps = all_logps_sum[len(packed_seq_lens) // 2 :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: len(packed_seq_lens) // 2].mean()

    def _packed_get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
        packed_seq_lens,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        assert average_log_prob == False

        if self.strategy.ring_attn_group is None:
            assert logits.shape[:-1] == labels.shape
            labels = labels[:, 1:]
            logits = logits[:, :-1, :]
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        else:
            rank = self.strategy.ring_attn_rank
            total_seq_len = labels.numel()
            local_seq_len = total_seq_len // self.strategy.ring_attn_size
            local_slice = slice(rank * local_seq_len + 1, (rank + 1) * local_seq_len + 1)
            local_label = labels[:, local_slice]
            if rank == self.strategy.ring_attn_size - 1:
                # add a dummy label to the last logit
                local_label = F.pad(local_label, (0, 1), value=0)
            local_per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=local_label.unsqueeze(2)
            ).squeeze(2)
            # we may not need to all_gather the entire tensor, but it's easier to implement.
            # use the flash_attn all_gather so that the all_gather has correct backward.
            per_token_logps = all_gather(local_per_token_logps, self.strategy.ring_attn_group).reshape((1, -1))
            per_token_logps = per_token_logps[:, :-1]

        loss_masks = attention_mask.clone().bool()

        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            loss_masks[0, index : index + prompt_id_lens[i]] = False
            index = index + seq_len

        loss_masks = loss_masks[:, 1:]

        logprobs_sums = []
        logprobs_means = []
        index = 0
        for i, seq_len in enumerate(packed_seq_lens):
            seq = per_token_logps[0, index : index + seq_len - 1]
            mask = loss_masks[0, index : index + seq_len - 1]
            logprobs_sums.append((seq * mask).sum())
            logprobs_means.append((seq * mask).sum() / mask.sum())
            index = index + seq_len

        return torch.stack(logprobs_sums), torch.stack(logprobs_means)


class VLDPOTrainer(DPOTrainer):
    """
    Trainer for visual Direct Preference Optimization (DPO) training.

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
    def data_to_device(self, input_data):
        for key, value in input_data.items():
            input_data[key] = value.to(torch.cuda.current_device())
        return input_data

    def concatenated_forward(self, model, input_ids, attn_masks, labels, pixel_values, image_grid_thw):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        output = model(input_ids,
                       attention_mask=attn_masks,
                       return_output=True,
                       pixel_values=pixel_values,
                       image_grid_thw=image_grid_thw)

        all_logits = output["logits"]
        all_logps_sum, all_logps_mean = self._get_batch_logps(
            all_logits, labels, attn_masks, None, average_log_prob=False
        )
        assert input_ids.shape[0] % 2 == 0
        batch_size = input_ids.shape[0] // 2
        chosen_logps = all_logps_sum[: batch_size]
        rejected_logps = all_logps_sum[batch_size :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: batch_size].mean()

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        attention_mask,
        prompt_id_lens,
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
        assert average_log_prob == False
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_masks = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means

    def model_forward(self, data):
        data = self.data_to_device(data)

        chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
            self.model,
            data["input_ids"],
            data["attention_mask"],
            data["labels"],
            data["pixel_values"],
            data["image_grid_thw"],
        )
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                self.ref_model,
                data["input_ids"],
                data["attention_mask"],
                data["labels"],
                data["pixel_values"],
                data["image_grid_thw"],
            )
        return chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps, aux_loss, nll_loss
