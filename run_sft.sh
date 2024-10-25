python train_llm.py --output_dir ./results \
--run_name test \
--data_files /home/ma.y/Research/swallow/lmsys-chat-1m/release_candidates/dataset/lmsys-chat-1m-synth-sft.jsonl.gz \
--model_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-v0.1 \
--tokenizer_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1 \
--fp16 \
--num_train_epochs 1 \
--per_device_train_batch 2 \
--gradient_accumulation_steps 64 \
--gradient_checkpointing \
--optim paged_adamw_8bit \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--max_grad_norm 0.5 \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 300 \

