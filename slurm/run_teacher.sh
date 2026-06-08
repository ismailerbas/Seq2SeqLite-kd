#!/bin/bash
#SBATCH --job-name=teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── CRITICAL: must be set BEFORE python starts ────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA=/gpfs/path/to/your/nmi          # <-- edit this
SAVE=/gpfs/path/to/your/runs         # <-- edit this

python train_teacher.py \
    --data-dir  "$DATA" \
    --save-dir  "$SAVE" \
    --n-total-m 8 \
    --seq-len   135 \
    --n-out     3 \
    --teacher-units  128 \
    --teacher-layers 2 \
    --batch-size 2048 \
    --epochs    300 \
    --patience  20 \
    --lr        1e-3 \
    --lr-factor 0.5 \
    --lr-patience 8 \
    --lr-min    1e-6 \
    --split-seed 42 \
    --pipeline-workers 16 \
    --prefetch-batches 32
