# OpenRLHF Benchmark

## Setting
We basically follow [deepspeed-chat's benchmark settings](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/BenckmarkSetting.md) with some differences
- max_prompt_seq_len=1024 and max_answer_seq_len=1024
- init reward model from actor model with same model size

For all model sizes and comparisions, we use a fixed global train_batch_size=1024, then we adjust micro_train_batch_size and micro_rollout_batch_size to the largest size that doesn't cause out of memory error.