"""ES Worker Extension for vLLM.

Extends WorkerWrap with Evolutionary Strategies (ES) capabilities:
- model_mutate: Apply deterministic noise perturbations to model parameters
- apply_es_gradient: Compute and apply ES gradient from seed/score pairs
- get_mutation_seed: Return current mutation seed for response tagging
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .vllm_worker_wrap import WorkerWrap

STABILIZE_SEED = -1


# This function is run population * num_params times, but only takes a few microseconds so it's never a bottleneck.
def _noise(p: torch.Tensor, name: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=p.device)
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    g.manual_seed((seed ^ h) & 0xFFFFFFFF)
    n = torch.normal(
        0, 1, size=p.shape, device=p.device, generator=g, dtype=p.dtype
    )  # the noise needs to be fp16 or bf16 for addition/subtraction to be reversible in float32.

    return n


class ESWorkerWrap(WorkerWrap):
    """vLLM worker extension with ES mutation and gradient support."""

    current_seed: Optional[int] = None
    current_std: float = 0.0
    optimizer: Optional[torch.optim.Optimizer] = None

    def revert_mutation(self, lazy: bool = False):
        if not lazy and isinstance(self.current_seed, int) and self.current_std:
            for n, p in self.model_runner.model.named_parameters():
                orig_dtype = p.data.dtype
                p.data.copy_(
                    (p.data.float() - _noise(p.data, n, self.current_seed).float() * self.current_std).to(
                        orig_dtype
                    )  # Note this is perfectly reversible since we upcast to float32 prior to doing addition/subtraction.
                )
        self.current_seed, self.current_std = None, 0.0
        return True

    def apply_mutation(self, seed: Optional[int] = None, std: float = 0.0) -> Optional[int]:
        if isinstance(seed, int) and std:
            for n, p in self.model_runner.model.named_parameters():
                orig_dtype = p.data.dtype
                p.data.copy_((p.data.float() + _noise(p.data, n, seed).float() * std).to(orig_dtype))
        self.current_seed, self.current_std = seed, std
        return seed

    def update_weight(self, **kwargs):
        self.revert_mutation(lazy=True)
        super().update_weight(**kwargs)

    def update_weight_cuda_ipc(self, **kwargs):
        self.revert_mutation(lazy=True)
        super().update_weight_cuda_ipc(**kwargs)

    def model_mutate(self, seed: Optional[int] = None, std: float = 0.0) -> Optional[int]:

        self.revert_mutation()
        if seed == STABILIZE_SEED:
            self.current_seed = STABILIZE_SEED
            self.current_std = 0.0
            return seed
        self.apply_mutation(seed, std)

        return seed

    def get_mutation_seed(self) -> Optional[int]:
        """Return current mutation seed for response tagging."""
        return self.current_seed

    def _get_or_create_optimizer(self) -> torch.optim.Optimizer:
        """Lazy-initialize optimizer from environment variables."""
        if self.optimizer is not None:
            return self.optimizer

        # Read optimizer config from environment
        optimizer_name = os.getenv("ES_OPTIMIZER", "SGD")
        optimizer_params_str = os.getenv("ES_OPTIMIZER_PARAMS", '{"lr": 0.001}')

        try:
            optimizer_params = json.loads(optimizer_params_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid --es_optimizer_params JSON: {optimizer_params_str}")

        # Get model parameters
        params = [p for n, p in self.model_runner.model.named_parameters()]

        # Create optimizer
        optimizer_cls = getattr(torch.optim, optimizer_name)
        self.optimizer = optimizer_cls(params, **optimizer_params)

        return self.optimizer

    def apply_es_gradient(self, updates: List[Tuple[int, float, float]]) -> bool:
        """Compute and apply ES gradient from seed/score pairs.

        Args:
            updates: List of (seed, normalized_score, sigma) tuples.
                    Scores should already be normalized (zero mean, optionally unit std).

        Returns:
            True on success.
        """
        # Convert from dataclass if needed
        updates = [(u.seed, u.score, u.sigma) if hasattr(u, "seed") else u for u in updates]

        # Revert any current mutation before applying gradient
        self.revert_mutation()

        # We stage weight updates and perform an optimizer step per parameter to minimize memory usage.

        clip = float(os.getenv("ES_CLIP_GRAD_NORM", "0.0"))
        optimizer = self._get_or_create_optimizer()
        optimizer.zero_grad(set_to_none=True)
        for name, p in self.model_runner.model.named_parameters():

            # Move optimizer state to GPU if needed
            st = optimizer.state.get(p)
            if st:
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to(p.device)

            # Compute ES gradient: weighted sum of noise vectors
            g = torch.zeros_like(p, dtype=torch.float32)
            for seed, w, _ in updates:
                if isinstance(seed, int) and seed != STABILIZE_SEED:
                    # Note we don't need to divide by sigma since noise has std=1, all sigmas are the same, and weights are already normalized.
                    g.add_(_noise(p.data, name, seed).float(), alpha=float(w))
            g.div_(max(1, len(updates)))

            # ES uses negative gradient (we want to move in direction of positive scores)
            p.grad = (-g).to(p.dtype)

            if clip > 0:
                # NOTE: This is not equivalent to PyTorch's clip_grad_norm_(), though we can't do torcher's version since it would make this code too slow.
                torch.nn.utils.clip_grad_norm_([p], clip)

            optimizer.step()
            p.grad = None

            # Move optimizer state back to CPU to save memory. This is slightly slower than keeping it on the GPU, but is more than offset by allowing vLLM to use all of the GPU for inference.
            st = optimizer.state.get(p)
            if st:
                for k, v in list(st.items()):
                    if isinstance(v, torch.Tensor):
                        st[k] = v.to("cpu")

        self.current_seed, self.current_std = None, 0.0
        return True

    def save_hf_checkpoint(self, output_dir: str, tokenizer_path: str) -> bool:
        """Persist current weights and tokenizer to a HuggingFace-compatible folder (rank-0 / TP=1)."""
        self.revert_mutation()
        os.makedirs(output_dir, exist_ok=True)

        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True).save_pretrained(output_dir)

        model = self.model_runner.model
        saved = False
        for candidate in (model, getattr(model, "model", None), getattr(model, "llm", None)):
            if candidate is not None and hasattr(candidate, "save_pretrained"):
                try:
                    candidate.save_pretrained(output_dir, safe_serialization=True)
                    saved = True
                    break
                except Exception:
                    continue

        if not saved:
            try:
                from safetensors.torch import save_file

                cfg_src = Path(tokenizer_path) / "config.json"
                if cfg_src.is_file():
                    shutil.copy2(cfg_src, Path(output_dir) / "config.json")
                state_dict = {k: v.detach().cpu() for k, v in model.named_parameters()}
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
                saved = True
            except Exception as exc:
                print(f"[ESWorkerWrap] save_hf_checkpoint fallback failed: {exc}")
                return False

        return saved
