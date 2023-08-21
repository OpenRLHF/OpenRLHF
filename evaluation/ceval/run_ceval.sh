model_path='ckpt' #hf format model
output_path='ceval_results'

python ceval.py \
    --model_path ${model_path} \
    --cot False \
    --few_shot True \
    --with_prompt False \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \