# Setup

### Docker image:

```
mirrors.tencent.com/youngyli/openrlhf:3.0
```

### Enviroment setup:

```
mv /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12 /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12.bak
ln -s /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12 /root/miniconda3/envs/rlhf/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12
pip install deepspeed==0.15.0
pip install pylatexenc latex2sympy2 word2number
```

# Run

### How to kill job

For killing the background training script only, do ``killall pt_main_thread``

For killing both the ray server and the training script, do ``kill python gcs_server``
