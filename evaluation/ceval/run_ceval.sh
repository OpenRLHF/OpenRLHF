# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval
# and https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

pretrain_model_path='OpenLLMAI/Llama-2-7b-sft-model-ocra-500k' #HF format model
output_path='ceval_results/llama_5shot_no_prompt'

python ceval.py \
  --pretrain_model_path ${pretrain_model_path} \
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
