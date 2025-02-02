# execute in advance
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MEGATRON_REPO=/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/OpenRLHF/
# export LD_LIBRARY_PATH=/root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${MEGATRON_REPO}:$PYTHONPATH
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8