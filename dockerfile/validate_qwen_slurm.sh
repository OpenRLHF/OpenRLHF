#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

IMAGE_PATH="${PROJECT_DIR}/openrlhf-vllm0.27.1-ds0.19.5-tf5.15.0-full.sqsh"
CACHE_ROOT="${PROJECT_DIR}/.image-cache"
mkdir -p \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/triton" \
    "${CACHE_ROOT}/ray" \
    "${CACHE_ROOT}/tmp" \
    "${CACHE_ROOT}/vllm" \
    "${CACHE_ROOT}/xdg" \
    "${CACHE_ROOT}/torch-extensions" \
    "${CACHE_ROOT}/home"

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Missing validated image candidate: ${IMAGE_PATH}" >&2
    exit 1
fi

# The base image is built from a raw OCI export (see import_slurm_base_image.sh),
# which drops the NGC image's NVIDIA_VISIBLE_DEVICES/NVIDIA_DRIVER_CAPABILITIES
# env. Enroot's NVIDIA hook reads those at container start to inject the driver
# utility libraries (nvidia-smi, libnvidia-ml.so). Without them NVML is missing
# and vLLM platform detection falls back to UnspecifiedPlatform ("Device string
# must not be empty"). Re-export them so the hook injects NVML.
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

srun \
    --container-image="${IMAGE_PATH}" \
    --container-readonly \
    --container-mounts="${PROJECT_DIR}:/openrlhf" \
    --container-workdir=/openrlhf \
    bash -lc '
        set -euxo pipefail
        # The container is read-only, so HOME=/root cannot be written. Point HOME
        # at a writable mount so tools that ignore the cache envs below (e.g.
        # flashinfer, which mkdirs ~/.cache/flashinfer on import) do not fail with
        # "No space left on device".
        export HOME=/openrlhf/.image-cache/home
        export HF_HOME=/openrlhf/.image-cache/huggingface
        export TRITON_CACHE_DIR=/openrlhf/.image-cache/triton
        export RAY_TMPDIR=/openrlhf/.image-cache/ray
        export TMPDIR=/openrlhf/.image-cache/tmp
        export VLLM_CACHE_ROOT=/openrlhf/.image-cache/vllm
        export XDG_CACHE_HOME=/openrlhf/.image-cache/xdg
        export TORCH_EXTENSIONS_DIR=/openrlhf/.image-cache/torch-extensions
        set +e
        source /etc/shinit_v2
        compat_status=${_CUDA_COMPAT_STATUS:-}
        set -e
        test "${compat_status}" = "CUDA Driver OK"
        readlink -f /usr/local/cuda/compat/lib/libcuda.so.1
        python -m pip check
        python - <<"PY"
import torch
import transformers
import vllm
import deepspeed
from transformers import Qwen3Config, Qwen3ForCausalLM

# vLLM 0.27.x ships kernels as the "_C_stable_libtorch" extension and registers
# them into the torch.ops._C namespace; the old top-level "vllm._C" module no
# longer exists, so assert the op namespace instead.
assert hasattr(torch.ops, "_C"), "vLLM C++ ops (torch.ops._C) not registered"
# NVML must be reachable or vLLM cannot detect the CUDA platform at runtime.
import pynvml
pynvml.nvmlInit()
from vllm.platforms import current_platform
assert current_platform.device_type == "cuda", current_platform.device_type

expected = {"vllm": "0.27.1", "deepspeed": "0.19.5", "transformers": "5.15.0"}
actual = {"vllm": vllm.__version__, "deepspeed": deepspeed.__version__, "transformers": transformers.__version__}
assert actual == expected, (actual, expected)

config = Qwen3Config(
    vocab_size=128,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = Qwen3ForCausalLM(config).cuda().to(torch.bfloat16)
tokens = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
loss = model(input_ids=tokens, labels=tokens).loss
loss.backward()
assert torch.isfinite(loss)
print({**actual, "torch": torch.__version__, "cuda": torch.version.cuda, "qwen_loss": float(loss)})
PY
        torchrun --standalone --nproc-per-node=2 -m pytest -q tests/test_deepspeed_runtime_compat.py

        # Warm the DeepSpeed op cache from a single process before launching the
        # multi-rank training below. If the ops are JIT-compiled concurrently by
        # every ZeRO rank sharing TORCH_EXTENSIONS_DIR they deadlock on the
        # PyTorch FileBaton lock. Clearing stale locks first guards against a
        # previous interrupted run leaving a held lock behind.
        find "${TORCH_EXTENSIONS_DIR}" \( -name lock -o -name .ninja_lock \) -delete 2>/dev/null || true
        python -c "from deepspeed.ops.op_builder import FusedAdamBuilder, CPUAdamBuilder; FusedAdamBuilder().load(); CPUAdamBuilder().load(); print('deepspeed ops ready')"

        ray stop --force || true
        trap "ray stop --force || true" EXIT
        ray start --head --node-ip-address=127.0.0.1 --num-gpus=8 --dashboard-host=0.0.0.0 --disable-usage-stats

        # Use the repository Qwen default script on all eight GPUs. Limit the
        # dataset to one complete default rollout/update batch so validation
        # terminates after exercising the full training path.
        script=$(sed "s/--data.max_samples 128000/--data.max_samples 128/" \
            examples/scripts/train_reinforce_baseline_hybrid_engine.sh)
        bash -c "${script}" examples/scripts/train_reinforce_baseline_hybrid_engine.sh
    '
