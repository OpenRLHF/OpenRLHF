set -x

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
# Or the image built from dockerfile/Dockerfile: docker build -t openrlhf dockerfile
IMAGE_NAME="pytorch/pytorch:2.14.0-cuda13.0-cudnn9-devel"

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
	-e PIP_BREAK_SYSTEM_PACKAGES=1 -v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash