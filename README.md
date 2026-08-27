# seq2seqlite-kd

Quantized seq2seq GRU training pipeline for TCSPC/FLIM lifetime retrieval, with three
knowledge-distillation strategies running across 8 GPUs via TensorFlow `MirroredStrategy`.

---

## What this is

A teacher–student distillation framework for compressing a full-precision stacked-GRU
sequence-to-sequence model into a QKeras-quantized (4-bit) single-layer GRU that is small
enough to deploy on FPGA/edge hardware, while recovering teacher-level accuracy on
fluorescence lifetime (τ₁, τ₂) and FRET fraction estimation.

Three training scripts are provided:

| Script | Method | Loss |
|---|---|---|
| `train_teacher.py` | Full-precision baseline | MSE |
| `train_student.py` | **FW-QATD-RAC** | L_GT + α·L_KD + β·L_traj + γ·L_RAC |
| `train_student_vanilla_kd.py` | Vanilla KD (ablation) | α·T²·MSE(student,teacher) + (1-α)·MSE(student,GT) |

**FW-QATD-RAC** = Fisher-Weighted Quantization-Aware Trajectory Distillation with
Recurrent Accumulator Consistency regularisation.

---

## Architecture

```
Encoder input  (N, 135, 1)  ──► Teacher: stacked GRU [128, 128]  ──► Dense(3)
                                 Student: single QGRU [32], 4-bit  ──► QDense(3)
Decoder input  (N, 135, 1) ─┘
```

Teacher layers (weight-loading names): `encinput`, `decinput`, `encrnn`, `decrnn`, `decdense`
Student layers: `sencinput`, `sdecinput`, `sencgru`, `sdecgru`, `sdecdense`

---

## Quick start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the teacher

```bash
python train_teacher.py \
    --data-dir /gpfs/.../nmi \
    --n-total-m 8 \
    --seq-len 135 --n-out 3 \
    --teacher-units 128 --teacher-layers 2 \
    --batch-size 2048 --epochs 300 --patience 20 \
    --lr 1e-3
```

Outputs land in `<data-dir>/teacher_training/`:
- `teacher_best.weights.h5` — best checkpoint (load this for student training)
- `teacher_training_history.png` / `.csv`
- `teacher_scatter_tau1/tau2/fret.png`
- `teacher_residuals.png`
- `teacher_test_metrics.json`

### 3 — Delete stale student caches

After **any** teacher retrain, run:

```bash
rm <data-dir>/teacherPred_L135*.npy
rm <data-dir>/teacherHidden_L135*.npy
rm <data-dir>/fisherDiag_L135*.npy
```

### 4 — Train the FW-QATD-RAC student

```bash
python train_student.py \
    --data-dir     /gpfs/.../nmi \
    --teacher-ckpt /gpfs/.../nmi/teacher_training/teacher_best.weights.h5 \
    --student-units 32 --teacher-units 128 --teacher-layers 2 \
    --bits-kernel 4 --bits-recurrent 4 --bits-bias 4 \
    --bits-activation 4 --bits-state 4 \
    --batch-size 1024 --epochs 300 --patience 15 \
    --lr 1e-4 --alpha 0.5 --beta 0.05 --gamma 1e-3
```

Add `--resume` to continue from a preempted SLURM job.

### 5 — Train the vanilla KD student (ablation)

```bash
python train_student_vanilla_kd.py \
    --data-dir     /gpfs/.../nmi \
    --teacher-ckpt /gpfs/.../nmi/teacher_training/teacher_best.weights.h5 \
    --student-units 32 --teacher-units 128 --teacher-layers 2 \
    --bits-kernel 4 --bits-recurrent 4 --bits-bias 4 \
    --bits-activation 4 --bits-state 4 \
    --batch-size 1024 --epochs 300 --patience 15 \
    --lr 1e-4 --temperature 4.0 --alpha 0.7
```

---

## Cluster / SLURM

Ready-to-edit batch scripts are in `slurm/`. Update the `DATA=` and `SAVE=` paths then:

```bash
sbatch slurm/run_teacher.sh
sbatch slurm/run_student_fwqatd.sh
sbatch slurm/run_student_vanilla.sh
```

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` is set inside each script before Python starts.
All three scripts auto-detect available GPUs and fall back to CPU gracefully.

---

## Data format

See `docs/data_format.md` for a full description of the expected `.npy` files.

Summary:

| File | Shape | Notes |
|---|---|---|
| `tpsf_seq_L135_8M.npy` | (N, 135, 1) | Normalised TCSPC decay — encoder input |
| `res_L135_8M.npy` | (N, 135, 3) | Decoder target (channel reconstructions) |
| `labels_3ch_L135_8M.npy` | (N, 3) | Ground-truth [τ₁, τ₂, FRET] |
| `trainidx.npy` | (N_train,) | Train split indices |
| `validx.npy` | (N_val,) | Val split indices |
| `testidx.npy` | (N_test,) | Test split indices |

Split files are created automatically on first run if they do not already exist.

---

## Configs

`configs/` holds JSON snapshots of the default hyperparameters for each script. They are
**reference only** — the scripts do not load them automatically. Copy and edit one to record
the exact flags used in a given experiment run (the scripts also write `*_args.json` into
the output directory automatically).

---



---

## Output directory layout

Each student run creates:

```
<save-dir>/results/<job-name>/
    student_best.weights.h5
    student_final.weights.h5
    student_args.json
    training_history.csv
    training_history.png
    test_metrics.json
    test_scatter_tau1.png
    test_scatter_tau2.png
    test_scatter_fret.png
    test_residuals.png
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.13
- QKeras ≥ 0.9
- NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.7, tqdm ≥ 4.65
- 8× NVIDIA GPU node (tested on A100 80 GB); single-GPU and CPU fallback work

---

## License

See `LICENSE`.
