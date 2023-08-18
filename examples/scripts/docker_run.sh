set -x

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.05-py3"

docker run --runtime=nvidia -it --rm --shm-size="1g" --cap-add=SYS_ADMIN \
	-u $(id -u):$(id -g) \
 	-v $PROJECT_PATH:/openllama2 -v  $HOME/.cache:/.cache -v  $HOME/.bash_history2:/.bash_history \
	-v $HOME/.local:/.local -v $HOME/.triton:/.triton \
	$IMAGE_NAME bash