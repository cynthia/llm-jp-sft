#! /bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00

HUGGINGFACE_CACHE=/gs/bs/tga-okazaki/$USER/cache

export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE

# Memory optimization for H100 GPUs (95830MiB VRAM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# CPU optimization - dynamically detect CPU threads from sysfs
# Get total online CPUs from sysfs
TOTAL_CPUS=$(nproc)
# Divide by number of GPU processes (4 for Zero-1 config)
THREADS_PER_RANK=$((TOTAL_CPUS / 4))
export OMP_NUM_THREADS=$THREADS_PER_RANK
export MKL_NUM_THREADS=$THREADS_PER_RANK
export NUMEXPR_NUM_THREADS=$THREADS_PER_RANK

# W&B configuration
export WANDB_PROJECT=llama-swallow-dpo
export WANDB_ENTITY=${WANDB_ENTITY:-titech-org}

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate llm-jp-sft

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-gemma2_binarized}

LR=2.5e-5
MINLR=2.5e-6
WD=0.1
BETA=0.1
NAME=llama-3.1-swallow-instruct-v0.2_dpo_${DATASET_NAME}_adamw_0.95_LR_${LR}_MINLR_${MINLR}_WD_${WD}_BETA_${BETA}_EPOCH_2_zero1


accelerate launch --config_file configs/my_accelerate_config_zero1.yaml train_llm_dpo.py --output_dir /gs/bs/tga-okazaki/$USER/$NAME \
--run_name $NAME \
--data_files /gs/bs/tga-okazaki/$USER/swallow-preference-tune/${DATASET_NAME}.parquet  \
--model_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2 \
--tokenizer_name_or_path tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2 \
--bf16 \
--num_train_epochs 2 \
--per_device_train_batch 3 \
--gradient_accumulation_steps 21 \
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
--beta ${BETA} \
--report_to wandb