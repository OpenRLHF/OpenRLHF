FROM nvcr.io/nvidia/pytorch:25.02-py3

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
RUN pip install flash-attn==2.8.2
RUN pip install vllm==0.10.0

COPY docker-entrypoint.sh .
RUN chmod a+x docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
