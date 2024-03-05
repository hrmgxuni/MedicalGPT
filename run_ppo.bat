@echo off
@REM set CUDA_VISIBLE_DEVICES=0
@REM torchrun --nproc_per_node 1
python ppo_training.py ^
    --model_type bloom ^
    --model_name_or_path D:/TPHY/bigscience/bloomz-560m ^
    --reward_model_name_or_path D:/TPHY/bigscience/bloomz-560m ^
    --torch_dtype float16 ^
    --device_map auto ^
    --train_file_dir ./data/finetune ^
    --validation_file_dir ./data/finetune ^
    --batch_size 8 ^
    --max_source_length 256 ^
    --max_target_length 256 ^
    --max_train_samples 1000 ^
    --use_peft True ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --lora_dropout 0.05 ^
    --do_train ^
    --max_steps 100 ^
    --learning_rate 1e-5 ^
    --save_steps 50 ^
    --output_dir outputs-rl-bloom-v1 ^
    --early_stopping True ^
    --target_kl 0.1 ^
    --reward_baseline 0.0 ^
