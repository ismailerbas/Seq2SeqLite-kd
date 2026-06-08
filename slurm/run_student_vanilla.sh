#!/bin/bash
#SBATCH --job-name=student_vanilla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATA=/gpfs/path/to/your/nmi          # <-- edit this
SAVE=/gpfs/path/to/your/runs         # <-- edit this
CKPT="$DATA/teacher_training/teacher_best.weights.h5"

python train_student_vanilla_kd.py \
    --data-dir      "$DATA" \
    --teacher-ckpt  "$CKPT" \
    --save-dir      "$SAVE" \
    --student-units 32 \
    --teacher-units 128 \
    --teacher-layers 2 \
    --bits-kernel     4 \
    --bits-recurrent  4 \
    --bits-bias       4 \
    --bits-activation 4 \
    --bits-state      4 \
    --seq-len   135 \
    --n-out     3 \
    --temperature 4.0 \
    --alpha     0.7 \
    --batch-size 1024 \
    --epochs    300 \
    --patience  15 \
    --lr        1e-4 \
    --lr-factor 0.5 \
    --lr-patience 8 \
    --lr-min    1e-6 \
    --infer-batch   8192 \
    --pipeline-workers 4 \
    --prefetch-batches 32 \
    --split-seed 42
