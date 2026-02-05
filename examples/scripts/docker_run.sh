set -x

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
IMAGE_NAME="vllm/vllm-openai:v0.15.1-cu130"

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
	-v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash