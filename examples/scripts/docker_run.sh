set -x

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
# Or the image built from dockerfile/Dockerfile: docker build -t openrlhf dockerfile
IMAGE_NAME="nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
	-v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash