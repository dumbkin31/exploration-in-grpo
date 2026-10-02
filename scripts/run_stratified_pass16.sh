#!/bin/bash
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=qwen_pass16
#SBATCH --output=qwen_pass16_%j.log

cd ~/ANLP-QWEN
source ~/.bashrc
conda activate mt_env

# Prevent hanging on cluster compute nodes that have no internet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Use local model folder so Hugging Face doesn't try to fetch from web
MODEL_DIR="./Qwen3-1.7B"
if [ ! -d "$MODEL_DIR" ]; then
    MODEL_DIR="/scratch/arushishukla/anlp_project/Qwen3-1.7B"
fi

python -u evaluate_stratified_pass16.py \
    --model_path "$MODEL_DIR" \
    --split train \
    --samples_per_category 20 \
    --num_samples 16 \
    --chunk_size 4 \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_new_tokens 2048 \
    --output_file "results_qwen3_base_stratified_pass16.json"
