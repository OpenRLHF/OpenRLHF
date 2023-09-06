# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval
# and https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

pretrain_model_path='meta-llama/Llama-2-7b-hf' #HF format model
ckpt_path=${pretrain_model_path}               #no_pt_model, you can use pretrain_model_path or just remove it
#ckpt_path='ckpt/7b_llama/sft_model.pt' #pt model
output_path='ceval_results/llama_5shot_no_prompt'

python ceval.py \
  --pretrain_model_path ${pretrain_model_path} \
  --ckpt_path ${ckpt_path} \
  --cot False \
  --few_shot True \
  --with_prompt False \
  --constrained_decoding True \
  --temperature 0.2 \
  --n_times 1 \
  --ntrain 5 \
  --do_save_csv False \
  --do_test False \
  --output_dir ${output_path}
