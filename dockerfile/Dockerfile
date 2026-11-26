FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /app

RUN set -eux && \
    apt-get update && \
    apt-get install -y gosu && \
    rm -rf /var/lib/apt/lists/* && \
    gosu nobody true

RUN apt-get update && apt-get -y install sudo
RUN sudo su -

RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata

RUN apt-get -y install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev gdb
RUN pip uninstall xgboost transformer_engine torchao flash_attn pynvml opencv-python-headless -y
# Keep the training and rollout stacks on one resolver-selected PyTorch/CUDA ABI.
RUN pip install vllm==0.34.0 deepspeed==0.20.1 transformers==5.22.2
# Rebuild the existing FlashAttention dependency after vLLM has installed its
# exact PyTorch dependency, so the extension uses the same ABI. PyTorch 2.14
# headers need C++20, while flash-attn 2.8.3 still passes -std=c++17.
RUN git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    sed -i 's/-std=c++17/-std=c++20/g' /tmp/flash-attention/setup.py && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install --no-build-isolation --no-deps /tmp/flash-attention && \
    rm -rf /tmp/flash-attention

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
