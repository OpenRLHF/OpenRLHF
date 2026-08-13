#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

SOURCE_IMAGE="${PROJECT_DIR}/openrlhf-vllm0.27.1-ds0.19.5-tf5.15.0.sqsh"
FINAL_IMAGE="${PROJECT_DIR}/openrlhf-vllm0.27.1-ds0.19.5-tf5.15.0-full.sqsh"
PIP_CACHE="${PROJECT_DIR}/.image-cache/pip"
TRITON_CACHE="${PROJECT_DIR}/.image-cache/triton"
mkdir -p "${PIP_CACHE}" "${TRITON_CACHE}"

test -f "${SOURCE_IMAGE}"
if [[ -e "${FINAL_IMAGE}" ]]; then
    echo "Refusing to overwrite existing image: ${FINAL_IMAGE}" >&2
    exit 1
fi

# The raw OCI export drops the NGC image's NVIDIA env; re-export it so enroot's
# NVIDIA hook injects the driver utility libs (nvidia-smi, libnvidia-ml.so) that
# NVML-based platform detection needs. See validate_qwen_slurm.sh for details.
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

srun \
    --container-image="${SOURCE_IMAGE}" \
    --container-writable \
    --container-save="${FINAL_IMAGE}" \
    --container-mounts="${PROJECT_DIR}:/openrlhf,${PIP_CACHE}:/root/.cache/pip,${TRITON_CACHE}:/root/.triton" \
    --container-workdir=/openrlhf \
    bash -lc '
        set -euxo pipefail
        export TRITON_CACHE_DIR=/root/.triton
        python -m pip install -e /openrlhf
        python -m pip check
        python -c "import deepspeed, flash_attn, ray, torch, transformers, vllm; assert hasattr(torch.ops, \"_C\"), \"vLLM torch.ops._C missing\"; x = torch.ones(1024, device=\"cuda\"); assert x.sum().item() == 1024; print({\"torch\": torch.__version__, \"cuda\": torch.version.cuda, \"vllm\": vllm.__version__, \"deepspeed\": deepspeed.__version__, \"transformers\": transformers.__version__, \"flash_attn\": flash_attn.__version__, \"ray\": ray.__version__})"
    '

echo "Saved completed image to ${FINAL_IMAGE}"
