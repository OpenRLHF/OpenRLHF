set -x

PROJECT_PATH=$(cd $(dirname $0)/../; pwd)
IMAGE_NAME="nvcr.io/nvidia/pytorch:23.07-py3"

docker run --runtime=nvidia -it --rm --shm-size="1g" --cap-add=SYS_ADMIN \
 	-v $PROJECT_PATH:/root/chatgpt -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash