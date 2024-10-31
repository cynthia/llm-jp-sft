#! /bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00

HUGGINGFACE_CACHE=/gs/bs/tga-okazaki/ma/cache

export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE

module load nccl/2.20.5

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate llm-jp-sft

LR=2.5e-5
MINLR=2.5e-6
WD=0.1
NAME=$1

accelerate launch --config_file configs/my_accelerate_config_zero1.yaml train_llm.py --output_dir /gs/bs/tga-okazaki/ma/$NAME \
--run_name $NAME \
--data_files /gs/bs/tga-okazaki/ma/lmsys-chat-1m/lmsys-chat-1m-synth-ja-wo-pii-and-template-instructions.jsonl.gz /gs/bs/tga-okazaki/ma/lmsys-chat-1m/lmsys-chat-1m-synth-en-wo-pii-and-template-instructions.jsonl.gz  \
--model_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-v0.1 \
--tokenizer_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1 \
--bf16 \
--num_train_epochs 2 \
--per_device_train_batch 4 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--optim adamw_torch \
--adam_beta2 0.95 \
--learning_rate ${LR} \
--lr_scheduler_type cosine_with_min_lr \
--lr_scheduler_kwargs '{"min_lr":'${MINLR}'}' \
--weight_decay ${WD} \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 300 \

