set -x

PROJECT_PATH=$(cd $(dirname $0)/../../; pwd)
IMAGE_NAME="${IMAGE_NAME:-openrlhf-fsdp2:latest}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
	docker build -t "$IMAGE_NAME" -f "$PROJECT_PATH/dockerfile/Dockerfile" "$PROJECT_PATH"
fi

docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
	-v $PROJECT_PATH:/openrlhf -v  $HOME/.cache:/root/.cache -v  $HOME/.bash_history2:/root/.bash_history \
	$IMAGE_NAME bash -lc "cd /openrlhf && exec bash"
