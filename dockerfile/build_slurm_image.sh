#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
IMAGE_BASENAME=${IMAGE_BASENAME:-openrlhf-vllm0.27.1-ds0.19.5-tf5.15.0.sqsh}
IMAGE_PATH="${PROJECT_DIR}/${IMAGE_BASENAME}"
BASE_IMAGE_PATH="${PROJECT_DIR}/nvidia-pytorch-26.03-py3.sqsh"
PIP_CACHE_DIR="${PROJECT_DIR}/.image-cache/pip"
TRITON_CACHE_DIR="${PROJECT_DIR}/.image-cache/triton"

if [[ ! -f "${BASE_IMAGE_PATH}" ]]; then
    echo "Missing local base image: ${BASE_IMAGE_PATH}" >&2
    exit 1
fi
if [[ -e "${IMAGE_PATH}" ]]; then
    echo "Refusing to overwrite existing image: ${IMAGE_PATH}" >&2
    exit 1
fi
mkdir -p "${PIP_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# The raw OCI export drops the NGC image's NVIDIA env; re-export it so enroot's
# NVIDIA hook injects the driver utility libs (nvidia-smi, libnvidia-ml.so) that
# NVML-based platform detection needs. See validate_qwen_slurm.sh for details.
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

srun \
    --container-image="${BASE_IMAGE_PATH}" \
    --container-writable \
    --container-save="${IMAGE_PATH}" \
    --container-mounts="${PROJECT_DIR}:/openrlhf,${PIP_CACHE_DIR}:/root/.cache/pip,${TRITON_CACHE_DIR}:/root/.triton" \
    --container-workdir=/openrlhf \
    bash -lc '
        set -euxo pipefail
        export TRITON_CACHE_DIR=/root/.triton
        set +e
        source /etc/shinit_v2
        compat_status=${_CUDA_COMPAT_STATUS:-}
        set -e
        test "${compat_status}" = "CUDA Driver OK"
        python -m pip uninstall -y xgboost transformer_engine flash_attn pynvml opencv-python-headless || true
        python -m pip install vllm==0.27.1 transformers==5.15.0
        # Ahead-of-time compile the DeepSpeed CUDA ops (FusedAdam/CPUAdam) into the
        # image. Otherwise DeepSpeed JIT-compiles them on first use at runtime, and
        # when every ZeRO rank shares one TORCH_EXTENSIONS_DIR they deadlock on the
        # PyTorch FileBaton build lock (each waiting rank spins in time.sleep).
        DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 MAX_JOBS=32 \
            python -m pip install deepspeed==0.19.5 --no-build-isolation
        MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
        python -m pip install ring-flash-attn
        python -m pip install -e /openrlhf
        python -m pip check
        python -c "import deepspeed, torch, transformers, vllm; assert hasattr(torch.ops, \"_C\"), \"vLLM torch.ops._C missing\"; x = torch.ones(1024, device=\"cuda\"); assert x.sum().item() == 1024; print({\"torch\": torch.__version__, \"cuda\": torch.version.cuda, \"gpu\": torch.cuda.get_device_name(), \"vllm\": vllm.__version__, \"deepspeed\": deepspeed.__version__, \"transformers\": transformers.__version__})"
    '

echo "Saved validated image to ${IMAGE_PATH}"
