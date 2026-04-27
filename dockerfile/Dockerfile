FROM nvcr.io/nvidia/pytorch:25.11-py3

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
    python -m pip uninstall -y xgboost pynvml opencv-python-headless || true && \
    python -m pip install -r /tmp/openrlhf-requirements.txt && \
    python -m pip install vllm==0.20.0 && \
    python -m pip install "dion @ git+https://github.com/microsoft/dion.git"

COPY dockerfile/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
