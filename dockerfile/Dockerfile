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
RUN pip install vllm==0.27.1 deepspeed==0.19.5 transformers==5.15.0
# Rebuild the existing FlashAttention dependency after vLLM has installed its
# exact PyTorch dependency, so the extension uses the same ABI.
RUN MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
