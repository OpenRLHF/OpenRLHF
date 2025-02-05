# Setup

### Docker Image:

```
mirrors.tencent.com/youngyli/openrlhf:3.0
```

### Enviroment setup:

```
mv /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12 /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12.bak
ln -s /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12 /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12
pip install deepspeed==0.15.0
pip install pylatexenc
```
