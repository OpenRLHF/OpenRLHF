# OpenRLHF Tutorial
## Prepare envs and training
### Verified envs 
You can build openrlhf from nvidia docker(recomended) or from conda envs. 

* Python: 3.8/3.9/3.10/3.11
* Torch: 2.0.0/2.0.1
* CUDA: 12.0+(recomended)/11.8/11.7

### Single-node training with nvidia-docker 
```
cd examples/scripts

# install nvidia-docker (Optional)
./nvidia_docker_install.sh

# launch nvidia container
./docker_run.sh

# cd in container
cd /openrlhf/examples/scripts

# build OpenRLHF (i.e, pip install)
./build_openrlhf.sh

# huggingface login 
~/.local/bin/huggingface-cli login

# train SFT model
./train_sft_llama.sh

# train RM model
./train_rm_llama.sh

# train PPO model
./train_ppo_llama.sh
```

### Multi-nodes training on Slurm 
```
cd examples/scripts

# huggingface login on Slurm 
pip install transformers
huggingface-cli login

# Moidfy the Slurm Account/Nodes ... in `train_llama_slurm.sh`

# For SFT, RM, and PPO training stage:
# Modify the variable `training_script` in `train_llama_slurm.sh` to
readonly training_script="train_sft_llama.sh"
readonly training_script="train_rm_llama.sh"
readonly training_script="train_ppo_llama.sh"

# set `GPUS_PER_NODE` in `train_llama_slurm.sh`
readonly GPUS_PER_NODE=8

# run multi-nodes training script
# train_llama_slurm.sh will load the training args from `training_script`
sbatch ./train_llama_slurm.sh
```

### build openrlhf from conda envs 
If you really don't want to use nvidia docker, we also provide tutorials for building openrlhf from a conda environment.  
```
# we need conda
conda create -n llama2 python=3.8
# now requirements.txt seems incomplete for a conda env
pip install -r requirements.txt
# so, we need install some package manualy: when installing torch, you may need to match the corresponding cuda version.
pip install packaging ninja
pip install torch --index-url https://download.pytorch.org/whl/cu118
# check ninjia
ninja --version
echo $? 
# install flash-attn: may take some time
pip install flash-attn --no-build-isolation
# build: may need to remove --use-feature=in-tree-build
./build_openrlhf.sh
# enjoy it!
```

## Infer 
After completing the training, you can evaluate your model by using the inference script: 

```
./inference_llama.sh { model_path } "Please introduce the GTA5 game."
```

## Eval 
Now, we support ceval. Other eval methods are under construction. 

```
sh run_ceval.sh
```

## RLHF pipeline 
The following table shows the improvement on ceval after simple SFT and PPO training. 

| Model                      | 5shot-no_prompt     |
| -------------------------- | ------------------- |
| Llama-2-7b-hf              | 0.3261515601783061  |
| ckpt/7b_llama/sft_model.pt | 0.32838038632986627 |
| ckpt/7b_llama/ppo_model.pt | 0.3328380386329866  |