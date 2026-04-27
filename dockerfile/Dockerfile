FROM nvcr.io/nvidia/pytorch:26.03-py3

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        gdb \
        git \
        gosu \
        libglew-dev \
        libglfw3-dev \
        libomp-dev \
        libopenexr-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        python3-dev \
        python3-pip \
        sudo \
        tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

COPY requirements.txt /tmp/openrlhf-requirements.txt

RUN set -eux && \
    python -m pip install --upgrade pip setuptools wheel && \
    (python -m pip uninstall -y xgboost pynvml opencv-python-headless || true) && \
    python -m pip install "dion @ git+https://github.com/microsoft/dion.git@28311d93686275f1baa7f0af57e6c4c61b75440b" && \
    python -m pip install vllm==0.20.0 && \
    python -m pip install --no-deps https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl && \
    sed '/^dion @/d' /tmp/openrlhf-requirements.txt > /tmp/openrlhf-requirements-no-dion.txt && \
    python -m pip install -r /tmp/openrlhf-requirements-no-dion.txt

COPY dockerfile/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
