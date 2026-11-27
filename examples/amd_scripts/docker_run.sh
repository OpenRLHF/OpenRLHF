NAME=openrlhf_release
DOCKER=openrlhf-rocm:validated

docker run -it --name $NAME --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $DOCKER \
  /bin/bash