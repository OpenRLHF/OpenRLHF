#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
    PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi

IMAGE_PATH="${PROJECT_DIR}/nvidia-pytorch-26.03-py3.sqsh"
TOOLS_DIR="${PROJECT_DIR}/.image-tools"
WORK_DIR="${PROJECT_DIR}/.image-work/nvidia-pytorch-26.03-py3"
ROOTFS_DIR="${WORK_DIR}/rootfs"
CRANE="${TOOLS_DIR}/crane"

if [[ ! -x "${CRANE}" ]]; then
    echo "Missing project-local crane binary: ${CRANE}" >&2
    exit 1
fi
if [[ -e "${IMAGE_PATH}" || -e "${WORK_DIR}" ]]; then
    echo "Refusing to overwrite an existing image or workspace" >&2
    exit 1
fi
mkdir -p "${ROOTFS_DIR}" "${WORK_DIR}/tmp"

export TMPDIR="${WORK_DIR}/tmp"
"${CRANE}" config nvcr.io/nvidia/pytorch:26.03-py3 >"${WORK_DIR}/config.json"
"${CRANE}" export --platform=linux/amd64 nvcr.io/nvidia/pytorch:26.03-py3 - \
    | tar --extract --preserve-permissions --no-same-owner --file=- --directory="${ROOTFS_DIR}"

# A raw OCI export does not carry Docker Config.Env. Login shells used by the
# Slurm scripts source this generated profile fragment and recover those values.
mkdir -p "${ROOTFS_DIR}/etc/profile.d"
jq -r '.config.Env[] | capture("^(?<key>[^=]+)=(?<value>.*)$") | "export \(.key)=\(.value|@sh)"' \
    "${WORK_DIR}/config.json" >"${ROOTFS_DIR}/etc/profile.d/oci-image-env.sh"

mksquashfs "${ROOTFS_DIR}" "${IMAGE_PATH}" \
    -noappend -all-root -no-xattrs -comp zstd -Xcompression-level 3 -processors 32

echo "Saved base image to ${IMAGE_PATH}"
