#!/bin/bash
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=qwen_strat350
#SBATCH --output=qwen_strat350_%j.log

cd ~/ANLP-QWEN
source ~/.bashrc
conda activate mt_env

# Check for local model weights
MODEL_DIR="./Qwen3-1.7B"
if [ ! -d "$MODEL_DIR" ]; then
    MODEL_DIR="/scratch/arushishukla/anlp_project/Qwen3-1.7B"
fi

# Run stratified 350-sample Pass@1 baseline (50 questions across all 7 categories)
python -u evaluate_stratified_math.py \
    --model_path "$MODEL_DIR" \
    --split train \
    --samples_per_category 50 \
    --output_file "results_qwen3_base_stratified_350.json"
