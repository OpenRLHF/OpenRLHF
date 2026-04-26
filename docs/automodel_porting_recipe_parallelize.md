> **⚠ SUPERSEDED 2026-04-26.** User redirect: "你不用导入 nemo rl fsdp 相关的路径
> 只需要兼容 automodel". OpenRLHF's FSDP backend will not port NeMo-RL's
> parallelize layer; it wraps `nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained`
> directly. Per-architecture TP plans, ColwiseParallel/RowwiseParallel dicts,
> custom ParallelStyle subclasses, etc. all live inside Automodel and are
> consumed via its public API rather than copied. This document is kept for
> historical reference only — see `automodel_migration_plan.md` §3 for the
> current architecture.

---

# (ARCHIVED) NeMo-RL `parallelize.py` → OpenRLHF `openrlhf/utils/fsdp/parallelize.py` Porting Recipe

**Source:** `/home/scratch.jianh_gpu/projects/NeMo-RL/nemo_rl/models/dtensor/parallelize.py` (882 lines)  
**Target:** `openrlhf/utils/fsdp/parallelize.py` (to be created)  
**Key Constraint:** NO Hydra (`get_class`), NO complex dataclass hierarchies, plain functions taking ints/booleans/torch types, single file.

---

## Overview

NeMo-RL's `parallelize.py` handles DTensor tensor parallelism (TP), FSDP2 data parallelism (DP), sequence parallelism (SP), activation checkpointing, and gradient clipping across Llama, Qwen, Gemma3, Mistral, and VLM architectures. The port must:

1. **Drop Hydra:** Replace `get_class(custom_parallel_plan)` with direct dict/function handling
2. **Flatten inputs:** Convert config hierarchies (`cfg.dtensor_cfg.tensor_parallel_size`) to flat function args (int `tp_size`, etc.)
3. **Keep correctness:** Preserve DTensor parallelization plans, FSDP2 mesh logic, checkpoint wrapper patterns
4. **Single file:** No sub-modules; all functions and classes in one file
5. **Support DP-only Phase 1:** Initially Llama + Qwen, no VLM, no NemotronH

---

## Section-by-Section Breakdown

### 1. Custom ParallelStyle Classes (Lines 72–168)

#### `RotaryEmbedParallel` (Lines 72–108)
- **Purpose:** Handle sequence parallelism for Qwen2/Gemma3 rotary embeddings (tuple input/output).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None—this is pure PyTorch DTensor logic, no config dependency.
- **Line Range:** 72–108

#### `ColwiseParallelWithGather` (Lines 110–168)
- **Purpose:** Custom colwise parallelism with optional all-gather output.
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None—config-independent.
- **Line Range:** 110–168

**Subtotal for Phase 1:** Keep both classes as-is. ~100 lines.

---

### 2. Architecture-Specific TP Plans (Lines 170–338)

Three architecture functions return `dict[str, ParallelStyle]`. Each maps model submodules to TP strategies (ColwiseParallel, RowwiseParallel, SequenceParallel, etc.).

#### `_parallelize_gemma3()` (Lines 170–218)
- **Purpose:** Gemma3ForCausalLM + Gemma3ForConditionalGeneration TP plan.
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None—takes only `model` and bool `sequence_parallel`, returns dict.
- **Line Range:** 170–218
- **Phase 1 Skip:** Gemma3 is not a priority for Phase 1; defer to Phase 2.

#### `_parallelize_llama()` (Lines 220–256)
- **Purpose:** LlamaForCausalLM TP plan (base + optional sequence parallel).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None—signature is already clean (`model`, `sequence_parallel` bool).
- **Line Range:** 220–256
- **Phase 1 Priority:** YES—core for Phase 1 DP-only baseline.

#### `_parallelize_qwen()` (Lines 258–326)
- **Purpose:** Qwen2/Qwen3ForCausalLM TP plan with inner `Qwen3QKNorm` class for q/k normalization.
- **Port Verdict:** [Keep verbatim, move inner class to module level]
- **Adaptations:** 
  - Inline inner class `Qwen3QKNorm` (lines 264–280) → move to module level (before `_parallelize_qwen`).
  - Signature is clean; no Hydra dependency.
- **Line Range:** 258–326
- **Phase 1 Priority:** YES—Qwen2 is critical for Phase 1.

#### `PARALLIZE_FUNCTIONS` dict (Lines 328–338)
- **Purpose:** Router mapping model class → parallelization function.
- **Port Verdict:** [Adapt minor]
- **Adaptations:**
  - Llama4, VL models (Qwen2-VL, Llava, Mistral3) removed for Phase 1.
  - Phase 1 dict: `{Qwen2ForCausalLM: _parallelize_qwen, Qwen3ForCausalLM: _parallelize_qwen, LlamaForCausalLM: _parallelize_llama}`
  - Phase 2 extends with Gemma3, VLM classes, NemotronH.
- **Line Range:** 328–338
- **Phase 1:** Core three models.

**Subtotal for Phase 1:** Llama, Qwen, router. ~120 lines.

---

### 3. Parallel Style Translation (Lines 341–365)

#### `translate_parallel_style()` (Lines 341–365)
- **Purpose:** String → ParallelStyle enum (colwise, rowwise, colwise_rep, rowwise_rep, colwise_gather_output, sequence_parallel).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None—pure string-to-object mapping, no config.
- **Caching:** `@lru_cache` remains.
- **Line Range:** 341–365

**Subtotal for Phase 1:** 25 lines.

---

### 4. HuggingFace TP Plan Extraction (Lines 367–463)

#### `get_hf_tp_plan()` (Lines 367–463)
- **Purpose:** Extract `_tp_plan` attribute from HF model class/instance/inner model; fall back to hardcoded strategy strings.
- **Port Verdict:** [Adapt major]
- **Adaptations:**
  1. **Lines 389–425 (VL model handling):** Drop Llava, Mistral3, Llama4, Qwen2_5VL for Phase 1. Keep only base causal LM logic.
  2. **Lines 429–462 (plan resolution):** Keep string-to-ParallelStyle mapping (lines 453–460).
  3. **Lines 444–447 (assertion):** Relax error message for Phase 1 (no custom plans).
  4. **Signature:** No change—takes `model: PreTrainedModel`, returns dict.
- **Line Range:** 367–463
- **Phase 1 Subset:** Lines 427–462 (base case + plan resolution), drop VL branches.

**Subtotal for Phase 1:** ~80 lines (condensed from 97).

---

### 5. Core Parallelization Logic (Lines 465–768)

#### `_parallelize_nm5_h()` (Lines 465–540)
- **Purpose:** Special-case Mamba/NemotronH architecture.
- **Port Verdict:** [Drop — not for Phase 1]
- **Rationale:** NemotronH is custom (trust_remote_code), not a standard HF model. Phase 1 skips.
- **Line Range:** 465–540

#### `_parallelize_model()` (Lines 543–768) — **CORE FUNCTION**
- **Purpose:** Main entry point. Routes model by class, applies TP plan via `parallelize_module()`, wraps with activation checkpointing, applies FSDP2 `fully_shard()`.
- **Port Verdict:** [Adapt major]
- **Adaptations:**
  1. **Signature (lines 543–558):**
     - Keep: `model`, `dp_mesh`, `tp_mesh`, `param_dtype`, `sequence_parallel`, `activation_checkpointing`, `cpu_offload`.
     - Remove: `custom_parallel_plan` (Hydra-dependent).
     - Result: `_parallelize_model(model, dp_mesh, tp_mesh, param_dtype, sequence_parallel=False, activation_checkpointing=False, cpu_offload=False)` — clean, no config object.
  
  2. **Lines 580–664 (model structure detection):**
     - Drop NemotronH check (line 588).
     - Drop all VL model branches (lines 602–658).
     - Keep only: Gemma3ForConditionalGeneration (lines 583–586, defer for Phase 2) + default causal LM (lines 659–663).
     - Phase 1: Only line 661–663 (`model.model.layers`, `config.num_attention_heads`, etc.).
  
  3. **Lines 665–671 (TP validation):**
     - Keep verbatim (divisibility checks).
  
  4. **Lines 673–720 (TP plan resolution):**
     - Remove custom_parallel_plan handling (lines 674–691, uses Hydra `get_class`).
     - Simplify: Try `PARALLIZE_FUNCTIONS` (Phase 1: Qwen, Llama), fallback to `get_hf_tp_plan()`.
     - Phase 1: No custom plans; always use built-in or HF extraction.
     - New code:
       ```python
       if model_cls in PARALLIZE_FUNCTIONS:
           try:
               func = PARALLIZE_FUNCTIONS[model_cls]
               model_parallel_plan = func(model, sequence_parallel)
               print("Using optimized parallel plan.")
           except Exception as e:
               print(f"Optimized plan failed: {e}. Falling back to HF TP plan.")
               assert not sequence_parallel, "Sequence parallelism not supported in HF plan."
               model_parallel_plan = get_hf_tp_plan(model)
       else:
           print(f"No optimized plan for {model_cls}. Using HF TP plan.")
           assert not sequence_parallel, "Sequence parallelism not supported in HF plan."
           model_parallel_plan = get_hf_tp_plan(model)
       ```
     - Delete Hydra `get_class` import and lines 678–689.
  
  5. **Lines 722–744 (activation checkpointing):**
     - Keep verbatim; wraps layer MLP/attention/norm with `checkpoint_wrapper()`.
  
  6. **Lines 745–758 (FSDP2 policies):**
     - Keep verbatim:
       - `MixedPrecisionPolicy(param_dtype, reduce_dtype=float32, output_dtype=float32)`
       - `CPUOffloadPolicy(pin_memory=False)` if `cpu_offload`, else `OffloadPolicy()`
  
  7. **Lines 755–768 (FSDP wrapping):**
     - Keep verbatim: per-layer `fully_shard(layer, ...)`, then root `fully_shard(model, reshard_after_forward=False, ...)`.

- **Line Range:** 543–768 (condensed for Phase 1: ~500 lines with VL/NemotronH dropped).
- **Phase 1 Subset:** Llama + Qwen, no VL. ~350 lines.

**Subtotal for Phase 1:** ~350 lines (from 226 original).

---

### 6. Utility Functions (Lines 771–883)

#### `to_local_if_dtensor()` (Lines 771–778)
- **Purpose:** Convert DTensor to local tensor if needed (grad clipping, grad norm).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None.
- **Line Range:** 771–778

#### `clip_grad_by_total_norm_()` (Lines 780–814)
- **Purpose:** Clip gradients by total norm (DTensor-aware).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None.
- **Line Range:** 780–814

#### `get_grad_norm()` (Lines 816–883)
- **Purpose:** Compute gradient norm across DP/TP groups (DTensor-aware).
- **Port Verdict:** [Keep verbatim]
- **Adaptations:** None.
- **Line Range:** 816–883

**Subtotal for Phase 1:** 113 lines; all kept.

---

### 7. DeviceMesh Creation Pattern (No explicit function in original)

- **Pattern:** `DeviceMesh` is passed as argument (`dp_mesh`, `tp_mesh`).
- **Port Note:** Caller (OpenRLHF trainer) creates meshes:
  ```python
  from torch.distributed.device_mesh import init_device_mesh
  dp_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
  tp_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))
  ```
- **Action:** Document mesh creation in docstring; do NOT create in `parallelize.py`.

---

### 8. FSDP2 Configuration Pattern

**Fully-Shard Patterns Preserved:**
- `MixedPrecisionPolicy(param_dtype, reduce_dtype=float32, output_dtype=float32)` — standard across all models.
- `CPUOffloadPolicy(pin_memory=False)` when enabled; else `OffloadPolicy()` (default).
- `reshard_after_forward=False` on root model (params used immediately in backward).

**No Changes Needed** — patterns copied from NeMo-RL.

---

## Import Refactoring

**Removes:**
```python
from hydra.utils import get_class  # DELETE
```

**Keeps:**
```python
from functools import lru_cache, partial
from types import FunctionType  # DELETE (no longer needed)
from typing import Callable, Optional, Union, cast
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel, parallelize_module
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.modeling_utils import PreTrainedModel
# Model imports: LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM
```

---

## Phase 1: Minimum Viable Port (DP-only, Llama + Qwen)

**Lines to port immediately:**

1. **Custom classes (lines 72–168):** `RotaryEmbedParallel`, `ColwiseParallelWithGather` — 100 lines.
2. **Llama TP plan (lines 220–256):** `_parallelize_llama()` — 37 lines.
3. **Qwen TP plan + inner class (lines 258–326 + move inner `Qwen3QKNorm`):** `_parallelize_qwen()`, `Qwen3QKNorm` — 69 lines.
4. **Router (lines 328–338):** Phase 1 subset only (Llama, Qwen, Qwen3) — 10 lines.
5. **Style translation (lines 341–365):** `translate_parallel_style()` — 25 lines.
6. **HF plan extraction (lines 367–463):** `get_hf_tp_plan()` base case only — 60 lines.
7. **Core function (lines 543–768):** `_parallelize_model()` with VL/NemotronH dropped — 350 lines.
8. **Utils (lines 771–883):** `to_local_if_dtensor()`, `clip_grad_by_total_norm_()`, `get_grad_norm()` — 113 lines.

**Phase 1 Total:** ~764 lines (vs 882 original).

---

## Suggested PR Breakdown

### PR 1: Foundation (Custom Classes + Utilities)
- Files: `openrlhf/utils/fsdp/parallelize.py` (new)
- Content: RotaryEmbedParallel, ColwiseParallelWithGather, to_local_if_dtensor, clip_grad_by_total_norm_, get_grad_norm
- Tests: Unit tests for DTensor → local conversion, gradient norm across TP/DP groups
- ~200 lines

### PR 2: TP Plans (Architecture-Specific)
- Depends on PR 1
- Content: _parallelize_llama, _parallelize_qwen, Qwen3QKNorm, PARALLIZE_FUNCTIONS router
- Tests: Integration test with Llama/Qwen models (check TP plan keys match expected module paths)
- ~120 lines added

### PR 3: HF Plan Extraction + Core Parallelization
- Depends on PR 2
- Content: translate_parallel_style, get_hf_tp_plan, _parallelize_model (with dropped VL/NemotronH)
- Tests: Full end-to-end test (model → DeviceMesh → parallelize → check FSDP2 wrapping)
- ~450 lines added

### PR 4: Phase 2 Extensions (Optional, later)
- Content: Gemma3, VL models, NemotronH, custom_parallel_plan support (Hydra-free)
- ~150 lines added

---

## Key Implementation Notes

1. **No Hydra:** The biggest change is removing `get_class(custom_parallel_plan)`. In Phase 1, no custom plans; Phase 2 could accept function pointers directly (no string resolution).

2. **Function Argument Style:** All functions take raw types (int, bool, torch.dtype, DeviceMesh) not config objects. Example:
   ```python
   def _parallelize_model(
       model: PreTrainedModel,
       dp_mesh: DeviceMesh,
       tp_mesh: DeviceMesh,
       param_dtype: torch.dtype,
       sequence_parallel: bool = False,
       activation_checkpointing: bool = False,
       cpu_offload: bool = False,
   ) -> torch.nn.Module:
   ```

3. **Docstrings:** Keep NeMo-RL's docstrings as-is; no Hydra references in them anyway. Update `_parallelize_model()` docstring to remove `custom_parallel_plan` mention.

4. **DeviceMesh creation:** Caller responsibility. Document in module-level docstring how trainer should create meshes.

5. **Gradient clipping:** Functions `clip_grad_by_total_norm_()` and `get_grad_norm()` are trainer utilities; kept as-is.

6. **Error messages:** Update any references to "custom_parallel_plan" (lines 445–446, 685–689) to be generic or omit.

---

## Testing Strategy

- **Unit:** ParallelStyle objects, DTensor helpers
- **Integration:** Load Llama/Qwen, verify TP plan keys, apply parallelize_module, check fully_shard wraps all layers
- **Smoke:** Distributed run (2 GPUs) with dp_size=2, tp_size=1; verify no errors
- **Correctness:** Compare grad norms before/after parallelize_model with NeMo-RL baseline (if possible)

---

## Summary Table

| Item | NeMo-RL Lines | Port Status | Phase 1? | Approx. Lines |
|------|---|---|---|---|
| RotaryEmbedParallel | 72–108 | Keep | Yes | 37 |
| ColwiseParallelWithGather | 110–168 | Keep | Yes | 59 |
| _parallelize_gemma3 | 170–218 | Keep (defer) | No | — |
| _parallelize_llama | 220–256 | Keep | Yes | 37 |
| _parallelize_qwen | 258–326 | Keep | Yes | 69 |
| Qwen3QKNorm | 264–280 | Move to module | Yes | 17 |
| PARALLIZE_FUNCTIONS | 328–338 | Adapt (Phase 1 subset) | Yes | 10 |
| translate_parallel_style | 341–365 | Keep | Yes | 25 |
| get_hf_tp_plan | 367–463 | Adapt (drop VL) | Yes | 60 |
| _parallelize_nm5_h | 465–540 | Drop | No | — |
| _parallelize_model | 543–768 | Adapt major | Yes | 350 |
| to_local_if_dtensor | 771–778 | Keep | Yes | 8 |
| clip_grad_by_total_norm_ | 780–814 | Keep | Yes | 35 |
| get_grad_norm | 816–883 | Keep | Yes | 68 |
| **Total (Phase 1)** | 882 | — | — | **~764** |

