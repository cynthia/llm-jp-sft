#! /bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00

module load cuda/12.1.0

source ~/venv/vllm/bin/activate

vllm serve google/gemma-2-27b-it --port 8000 \
--dtype bfloat16 \
--tensor-parallel-size 2 \
--device cuda \
--max-num-seqs 1024
