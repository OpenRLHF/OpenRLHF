# This code is modified from https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

pretrain_model_path='meta-llama/Llama-2-7b-hf' #HF format model
ckpt_path=${pretrain_model_path}  #no ckpt
#ckpt_path=path/to/ckpt #pt model
output_path='cmmlu_results/llama_5shot_no_prompt'

python eval.py \
    --pretrain_model_path ${pretrain_model_path} \
    --ckpt_path ${ckpt_path} \
    --few_shot True \
    --with_prompt False \
    --constrained_decoding True \
    --output_dir ${output_path} \
    --input_dir cmmlu_data