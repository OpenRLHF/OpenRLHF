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
RUN pip uninstall xgboost transformer_engine flash_attn pynvml opencv-python-headless -y
# vLLM 0.22.1 pulls in its matching torch==2.11.0 (cu13) build.
RUN pip install vllm==0.22.1
# Build flash-attn 2.8.3 against the torch 2.11 pulled in above (no prebuilt
# torch2.11 x86_64 wheel is published yet). MAX_JOBS caps build memory usage.
RUN MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
