# CUDA 13.0 is the version the PyTorch build that vLLM pins is compiled for.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

WORKDIR /app

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668): let pip install into it.
# The CUDA base image does not set CUDA_HOME, which just-in-time builds such as DeepSpeed's ops read.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN apt install -y tzdata

# cuda-compat-13-0 is CUDA forward compatibility: a userspace libcuda that lets this CUDA 13
# stack run on data center hosts whose NVIDIA driver is older than 580. It lives in
# /usr/local/cuda/compat, off the loader path; docker-entrypoint.sh adds it only when needed.
RUN apt-get -y install build-essential git ca-certificates cuda-compat-13-0 python3-dev python3-pip python-is-python3 libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
# Debian's pip, setuptools and wheel cannot be upgraded in place, so current ones are layered over them.
RUN pip install --ignore-installed pip setuptools wheel
# Keep the training and rollout stacks on one resolver-selected PyTorch/CUDA ABI.
RUN pip install vllm==0.34.0 deepspeed==0.20.1 transformers==5.22.2
# Build FlashAttention after vLLM has installed its exact PyTorch dependency, so the
# extension uses the same ABI. flash-attn/ builds the release that PyPI's own build cannot.
COPY flash-attn /tmp/flash-attn
RUN MAX_JOBS=8 pip install --no-build-isolation /tmp/flash-attn && rm -rf /tmp/flash-attn

# Fail the build, not a user's first run, when the installed stack does not import.
RUN python -c "import torch, vllm, deepspeed, transformers, flash_attn"

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
