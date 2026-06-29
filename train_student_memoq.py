#!/usr/bin/env python3
"""
train_student_memoq_full.py  —  MemoQ: Memory-Preserving 4-bit Recurrent
Quantization for GRU-based Sequence-to-Sequence Knowledge Distillation.

ROOT CAUSE TARGETED:
  Small 4-bit GRUs fail through two coupled recurrent mechanisms:
  (1) Memory-gate saturation: simultaneous 4-bit quantization of z/r/h gates
      destabilises the update gate. Saturation toward 1 makes the cell
      feedforward (overwrites memory); saturation toward 0 freezes
      quantisation error in the hidden state.
  (2) Recurrent state-error accumulation: per-step quantisation errors
      accumulate through the recurrence, driving the hidden state to the
      4-bit boundary before output KD can recover temporal memory.

METHOD — THREE PHASES:
  Phase 1  (float/8-bit warm-up, 30-50 ep):
    Float student, same topology+layer names as final QKeras student.
    L = (1-alpha)*HuberCN(s, gt) + alpha*HuberCN(s, teacher)  [alpha=0.7]
    Channel scales from teacher predictions over training split.

  Phase 2  (gate-decoupled 4-bit hardening, stages 2A/2B/2C):
    Custom SplitGateQGRUCell with separate W_z, W_r, W_h / U_z, U_r, U_h.
    Quantise gates in causal order: h first, then r, then z (NOT packed order).
    Stage 2A: quantise h-gate (candidate). L += 0.01 L_mem + 0.0005 L_rail
              After 5 epochs activate L_innov (lambda=0.005).
    Stage 2B: quantise r-gate. L weights escalated.
    Stage 2C: quantise z-gate. Activate L_zsat (logit barrier).
    Full MemoQ loss:
      L = L_seq + alpha*L_KD + lm*L_mem + li*L_innov + lz*L_zsat + lr*L_rail

  Phase 3  (hard 4-bit polish):
    All gates/kernels/biases/activations/states hard 4-bit via standard
    QKeras QGRU. Weights transferred by concatenating split-gate variables
    in Keras/QKeras packed order [z|r|h]. Low LR ~5e-6, clipnorm=0.5.

AUXILIARY LOSSES:
  L_mem    — lagged temporal memory kernel distillation (dimension-free)
  L_innov  — temporal innovation-profile matching (log-ratio)
  L_zsat   — update-gate saturation barrier (logit barrier |logit|>3)
  L_rail   — predictive rail-margin regularisation on hidden state

FINAL EXPORT:
  Standard QKeras QGRU encoder (sencgru) + QGRU decoder (sdecgru) +
  QDense head (sdec_dense). No auxiliary parameters at inference.

USAGE:
  python train_student_memoq_full.py \\
      --data-dir /path/to/data \\
      --teacher-ckpt /path/to/teacher_best.weights.h5 \\
      --save-dir /path/to/runs \\
      --bits-kernel 4 --bits-bias 4 --bits-recurrent 4 \\
      --bits-activation 4 --bits-state 4 \\
      --student-units 32 --teacher-units 128 --teacher-layers 2 \\
      --seq-len 135 --n-out 3 --gate-width-ns 0.09 \\
      --batch-size 16384 --lr 1e-4 --ref-batch-size 1024 \\
      --lr-factor 0.5 --lr-patience 8 --lr-min 1e-6 \\
      --temperature 4.0 --alpha 0.7 \\
      --memoq-warmup-epochs 40 \\
      --memoq-stage2a-epochs 30 --memoq-stage2b-epochs 30 \\
      --memoq-stage2c-epochs 30 --memoq-finetune-epochs 170 \\
      --memoq-lambda-mem 0.03 --memoq-lambda-innov 0.005 \\
      --memoq-lambda-zsat 0.002 --memoq-lambda-rail 0.001 \\
      --memoq-huber-delta 0.1 --memoq-rho-z 0.98 \\
      --memoq-rho-rail 0.88 --memoq-mu-rail 0.9 \\
      --patience 30 --log-interval 10 --infer-batch 8192 \\
      --prefetch-batches 4 --pipeline-workers 4 \\
      --split-seed 42

OUTPUTS (all inside --save-dir / results / job_name /):
  phase1_best.weights.h5
  stage2a_best.weights.h5
  stage2b_best.weights.h5
  stage2c_best.weights.h5
  student_best.weights.h5       <- final hard 4-bit QKeras model
  student_final.weights.h5
  student_args.json
  training_history.csv
  training_history.png
  test_metrics.json
  test_scatter_tau1.png  test_scatter_tau2.png  test_scatter_fret.png
  test_residuals.png
"""

import argparse
import glob
import json
import os
import sys
import time
import math

# ── Force GPU visibility BEFORE any TF import ────────────────────────────────
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print("[GPU] CUDA_VISIBLE_DEVICES defaulting to 0,1,2,3,4,5,6,7", flush=True)
else:
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except Exception:
    fastdtw  = None
    euclidean = None
    HAS_FASTDTW = False
from scipy.spatial.distance import euclidean

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GRU, Lambda, Concatenate
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh




# ==============================================================================
# File discovery
# ==============================================================================

def find_data_files(data_dir, seq_len):
    def find_one(patterns, desc):
        for pat in patterns:
            matches = glob.glob(os.path.join(data_dir, pat))
            if matches:
                return sorted(matches)[0]
        raise FileNotFoundError(
            f"Cannot find {desc} in {data_dir}. Tried: {patterns}"
        )

    file_input = find_one(
        [f"tpsf_seq_L{seq_len}_*.npy"],
        "encoder input",
    )
    file_res = find_one(
        [f"res_L{seq_len}_*.npy"],
        "decoder target",
    )
    file_labels = find_one(
        [f"labels_3ch_L{seq_len}_*.npy"],
        "labels_3ch",
    )

    def find_idx(names, desc):
        for name in names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{desc} not found in {data_dir}. Tried: {names}"
        )

    file_train = find_idx(["trainidx.npy", "train_idx.npy"], "train index")
    file_val   = find_idx(["validx.npy",   "val_idx.npy"],   "validation index")
    file_test  = find_idx(["testidx.npy",  "test_idx.npy"],  "test index")

    return file_input, file_res, file_labels, file_train, file_val, file_test





# ==============================================================================
# Teacher prediction + hidden cache
# ==============================================================================

def cache_teacher_predictions_and_hidden(
    teacher_model,
    teacher_hidden_model,
    normalized_input,
    seq_len,
    n_out,
    teacher_units,
    n_samples,
    infer_batch,
    data_dir,
    pf,
):
    """
    Cache ONLY teacher predictions to disk (mmap). Teacher hidden states are
    NOT cached — they are recomputed live inside the train/val step from
    teacher_hidden_model. Writing the full (N, T, teacher_units) hidden cache
    costs hundreds of GB and is never read, so it is skipped entirely.
    Returns (teacher_predictions_memmap, None).
    """
    file_pred = os.path.join(data_dir, f"teacherPred_L{seq_len}{n_samples}.npy")

    if os.path.exists(file_pred):
        pf(f"[CACHE] Teacher pred cache found: {file_pred}")
        teacher_predictions = np.load(file_pred, mmap_mode="r")
        pf(f"[CACHE] Loaded pred {teacher_predictions.shape}")
        return teacher_predictions, None

    pf("[CACHE] Teacher pred cache NOT found — computing predictions only "
       "(hidden is recomputed live and is NOT cached).")
    fp_pred = np.lib.format.open_memmap(
        file_pred, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, n_out),
    )

    pf(f"[CACHE] Building teacher predictions — {n_samples} samples, batch {infer_batch}")
    for start in range(0, n_samples, infer_batch):
        end = min(start + infer_batch, n_samples)
        enc_batch = normalized_input[start:end]
        dec_batch = np.zeros((end - start, seq_len, 1), dtype=np.float32)
        pred = teacher_model.predict(
            [enc_batch, dec_batch], batch_size=len(enc_batch), verbose=0
        )
        fp_pred[start:end] = pred.astype(np.float32)
        del pred, enc_batch, dec_batch
        if (start // infer_batch) % 10 == 0:
            pf(f"  [{start}/{n_samples}] cached")

    fp_pred.flush()
    del fp_pred
    pf("[CACHE] Teacher prediction cache written.")
    teacher_predictions = np.load(file_pred, mmap_mode="r")
    pf(f"[CACHE] Re-opened as memmap: pred {teacher_predictions.shape}")
    return teacher_predictions, None


# ==============================================================================
# Channel-normalised Huber loss
# ==============================================================================

def channel_normalised_huber(y_true, y_pred, channel_scales, huber_delta):
    """
    y_true, y_pred : (B, T, C)
    channel_scales : (C,) tf.constant — per-channel std of teacher preds
    Returns scalar mean loss.
    """
    residual = (y_pred - y_true) / channel_scales
    abs_res  = tf.abs(residual)
    delta    = tf.cast(huber_delta, tf.float32)
    huber    = tf.where(
        abs_res <= delta,
        0.5 * tf.square(residual),
        delta * (abs_res - 0.5 * delta),
    )
    return tf.reduce_mean(huber)


def compute_channel_scales(tpred_train, n_out, pf):
    pf("[SCALES] Per-channel teacher std:")
    scales = []
    for c in range(n_out):
        std_c = float(np.std(tpred_train[:, :, c]))
        std_c = max(std_c, 1e-3)
        scales.append(std_c)
        pf(f"  channel {c}: std={std_c:.6f}")
    sys.stdout.flush()
    return tf.constant(scales, dtype=tf.float32)




# ==============================================================================
# ReduceLROnPlateau
# ==============================================================================

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor, patience, min_lr, min_delta):
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.min_delta = min_delta
        self.best      = float("inf")
        self.wait      = 0

        if callable(optimizer.learning_rate):
            cur = float(optimizer.learning_rate(optimizer.iterations))
        else:
            cur = float(K.get_value(optimizer.learning_rate))

        self.lr_var = tf.Variable(cur, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = self.lr_var

    @property
    def current_lr(self):
        return float(K.get_value(self.lr_var))

    def step(self, val_loss, epoch, pfn):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            old_lr = self.current_lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                self.lr_var.assign(new_lr)
                pfn(f"ReduceLR: {old_lr:.2e} -> {new_lr:.2e}")
            self.wait = 0
            return True
        return False

    def reset(self, new_lr):
        self.best = float("inf")
        self.wait = 0
        self.lr_var.assign(float(new_lr))


# ==============================================================================
# Progress bar
# ==============================================================================

def bar(step, total, metrics: dict, epoch_start_time: float, width=28):
    frac   = step / max(total, 1)
    filled = int(width * frac)
    b      = "█" * filled + "░" * (width - filled)
    stats  = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
    elapsed = time.time() - epoch_start_time
    if step > 0:
        rem = (elapsed / step) * (total - step)
        eta = f"{rem/60:.0f}m{int(rem)%60:02d}s" if rem >= 60 else f"{rem:.0f}s"
        el  = f"{elapsed/60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
        ts  = f"  [{el}<{eta}]"
    else:
        ts = ""
    sys.stdout.write(f"\r{step:5}/{total}  {b}  {frac*100:5.1f}%  {stats}{ts}")
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ==============================================================================
# Phase 1 train/val steps — float student, no auxiliary losses
# ==============================================================================

def train_step_phase1_per_replica(batch_x, batch_y, model, optimizer,
                                  alpha, channel_scales, huber_delta):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    with tf.GradientTape() as tape:
        s_out = model([enc_b, dec_b], training=True)
        l_seq = channel_normalised_huber_memoq(tgt_b,   s_out, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber_memoq(tpred_b, s_out, channel_scales, huber_delta)
        total = (1.0 - alpha) * l_seq + alpha * l_kd

    grads = tape.gradient(total, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    nan_flag = tf.cast(
        tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(g)) for g in grads])),
        tf.float32
    )
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total, l_seq, l_kd, nan_flag


def val_step_phase1_per_replica(batch_x, batch_y, model, alpha, channel_scales, huber_delta):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    s_out = model([enc_b, dec_b], training=False)
    l_seq = channel_normalised_huber_memoq(tgt_b,   s_out, channel_scales, huber_delta)
    l_kd  = channel_normalised_huber_memoq(tpred_b, s_out, channel_scales, huber_delta)
    total = (1.0 - alpha) * l_seq + alpha * l_kd
    mae   = tf.reduce_mean(tf.abs(s_out - tgt_b))
    return total, l_seq, l_kd, mae


def make_dist_phase1_train(strategy, model, optimizer, alpha, channel_scales, huber_delta):
    @tf.function
    def step(bx, by):
        pr = strategy.run(train_step_phase1_per_replica,
                          args=(bx, by, model, optimizer, alpha, channel_scales, huber_delta))
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM,  pr[3], axis=None) > 0.0,
        )
    return step


def make_dist_phase1_val(strategy, model, alpha, channel_scales, huber_delta):
    @tf.function
    def step(bx, by):
        pr = strategy.run(val_step_phase1_per_replica,
                          args=(bx, by, model, alpha, channel_scales, huber_delta))
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[3], axis=None),
        )
    return step

def set_phase2_quantizers(args, enc_cell, dec_cell, stage):
    """
    Assign quantizers to MemoQGRUCell gate variables based on the
    MemoQ curriculum stage.

    Weight/recurrent quantizers now use alpha="auto_po2" to match the final
    QKeras student (build_final_qkeras_student). Using alpha=1.0 here while
    the final model used auto_po2 would create a scale mismatch at the
    P2C->P3 packing boundary; keeping them identical removes that source of
    the cliff. State stays at alpha=1.0 (bounded [-1, 1] hidden state).
    """
    q4  = quantized_bits(args.bits_kernel,     0, 1, alpha="auto_po2")
    q4s = quantized_bits(args.bits_state,      0, 1, alpha=1.0)
    q4a = quantized_tanh(bits=args.bits_activation, symmetric=True)

    if stage == "P2A":
        q_h = q4
        q_r = None
        q_z = None
        q_s = None
        q_a = None
    elif stage == "P2B":
        q_h = q4
        q_r = q4
        q_z = None
        q_s = None
        q_a = None
    elif stage == "P2C":
        q_h = q4
        q_r = q4
        q_z = q4
        q_s = q4s
        q_a = q4a
    elif stage == "P3":
        q_h = q4
        q_r = q4
        q_z = q4
        q_s = q4s
        q_a = q4a
    else:
        raise ValueError(f"Unknown stage: {stage!r}. Expected one of P2A, P2B, P2C, P3.")

    for cell in [enc_cell, dec_cell]:
        cell.quantizer_h          = q_h
        cell.quantizer_r          = q_r
        cell.quantizer_z          = q_z
        cell.quantizer_state      = q_s
        cell.quantizer_activation = q_a


# ==============================================================================
# Main MemoQ training loop
# ==============================================================================
def training_loop_memoq(
    strategy,
    float_student,
    final_qkeras_student,
    enc_cell_p2,
    dec_cell_p2,
    phase2_model,
    args,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    channel_scales,
    epsilon_innov,
    job_dir,
    pf,
    teacher_hidden_model=None,
    evaluate_fn=None,
):
    p1_ckpt         = os.path.join(job_dir, "phase1_best.weights.h5")
    p2a_ckpt        = os.path.join(job_dir, "stage2a_best.weights.h5")
    p2b_ckpt        = os.path.join(job_dir, "stage2b_best.weights.h5")
    p2c_ckpt        = os.path.join(job_dir, "stage2c_best.weights.h5")
    p3_ckpt         = os.path.join(job_dir, "student_best.weights.h5")
    resume_path     = os.path.join(job_dir, "resume_state.json")
    completion_path = os.path.join(job_dir, "training_complete.flag")

    history = {
        "total":     [], "seq":       [], "kd":        [],
        "mem":       [], "innov":     [], "zsat":      [], "rail":      [],
        "val_total": [], "val_seq":   [], "val_kd":    [],
        "val_mem":   [], "val_innov": [], "val_zsat":  [], "val_rail":  [],
        "val_mae":   [], "phase":     [],
    }

    total_planned = (
        args.memoq_warmup_epochs
        + args.memoq_stage2a_epochs
        + args.memoq_stage2b_epochs
        + args.memoq_stage2c_epochs
        + args.memoq_stage3_epochs
    )
    global_epoch = 0
    nan_warn_threshold = max(1, int(train_steps * 0.10))

    resume_stage = "P1"
    resume_epoch_in_stage = 0
    best_vals = {
        "P1": float("inf"), "P2A": float("inf"), "P2B": float("inf"),
        "P2C": float("inf"), "P3": float("inf"),
    }
    patience_cts = {"P1": 0, "P2A": 0, "P2B": 0, "P2C": 0, "P3": 0}

    stage_order_list = ["P1", "P2A", "P2B", "P2C", "P3"]
    max_epochs_per_stage = {
        "P1":  args.memoq_warmup_epochs,
        "P2A": args.memoq_stage2a_epochs,
        "P2B": args.memoq_stage2b_epochs,
        "P2C": args.memoq_stage2c_epochs,
        "P3":  args.memoq_stage3_epochs,
    }

    # ── Check completion sentinel before loading resume state ─────────────────
    # If training_complete.flag exists, all phases are done. Load the final
    # weights and return immediately so the caller can do evaluation/PNG
    # generation without SLURM resubmitting into a training loop.
    if args.resume and os.path.exists(completion_path):
        pf(f"[RESUME] Training already complete — found {completion_path}")
        pf("[RESUME] Skipping all training phases. Loading final weights for evaluation.")
        if os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf(f"[RESUME] Loaded final QKeras weights from {p3_ckpt}")
        sys.stdout.flush()
        _best_final = float("inf")
        if os.path.exists(resume_path):
            try:
                with open(resume_path) as _rf:
                    _rs = json.load(_rf)
                _bv = _rs.get("best_vals", {})
                for _k in ["P3", "P2C", "P2B", "P2A", "P1"]:
                    if _k in _bv:
                        _best_final = float(_bv[_k])
                        break
            except Exception:
                pass
        return history, _best_final

    if args.resume and os.path.exists(resume_path):
        pf(f"[RESUME] {resume_path}")
        with open(resume_path) as f:
            rs = json.load(f)
        resume_stage          = rs.get("stage", "P1")
        resume_epoch_in_stage = int(rs.get("epoch_in_stage", 0))
        best_vals.update({k: float(v) for k, v in rs.get("best_vals", {}).items()})
        patience_cts.update({k: int(v) for k, v in rs.get("patience_cts", {}).items()})
        if "history" in rs:
            for key in history:
                if key in rs["history"]:
                    history[key] = list(rs["history"][key])
        global_epoch = len(history["phase"])
        pf(f"[RESUME] stage={resume_stage} epoch_in_stage={resume_epoch_in_stage} global_epoch={global_epoch}")
        stage_idx = stage_order_list.index(resume_stage)
        if stage_idx >= 1 and os.path.exists(p1_ckpt):
            float_student.load_weights(p1_ckpt)
            pf("[RESUME] Loaded P1 weights for float_student")
        if stage_idx >= 2:
            ckpt = p2a_ckpt if stage_idx == 2 else (p2b_ckpt if stage_idx == 3 else p2c_ckpt)
            if os.path.exists(ckpt):
                phase2_model.load_weights(ckpt)
                pf(f"[RESUME] Loaded P2 weights for phase2_model from {ckpt}")
        if stage_idx >= 4 and os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf("[RESUME] Loaded P3 weights for final_qkeras_student")
        sys.stdout.flush()

    csv_path = os.path.join(job_dir, "training_history.csv")
    if not args.resume or (resume_stage == "P1" and resume_epoch_in_stage == 0):
        with open(csv_path, "w") as f:
            f.write(
                "epoch,phase,total,seq,kd,mem,innov,zsat,rail,"
                "val_total,val_seq,val_kd,val_mem,val_innov,val_zsat,val_rail,"
                "val_mae,lr\n"
            )

    def save_resume(stage_tag, ep_in_stage):
        state = {
            "stage":          stage_tag,
            "epoch_in_stage": ep_in_stage,
            "best_vals":      {k: float(v) for k, v in best_vals.items()},
            "patience_cts":   {k: int(v)   for k, v in patience_cts.items()},
            "history":        {
                k: ([float(x) for x in v] if k != "phase" else list(v))
                for k, v in history.items()
            },
        }
        with open(resume_path, "w") as f:
            json.dump(state, f, indent=2)

    def mark_training_complete():
        with open(completion_path, "w") as f:
            f.write("done\n")
        pf(f"[COMPLETE] All training phases finished — wrote {completion_path}")
        sys.stdout.flush()

    def should_run(stage_tag):
        if not args.resume:
            return True
        resume_idx = stage_order_list.index(resume_stage)
        stage_idx  = stage_order_list.index(stage_tag)
        if stage_idx > resume_idx:
            return True
        if stage_idx == resume_idx:
            return resume_epoch_in_stage < max_epochs_per_stage[stage_tag]
        return False

    def start_ep(stage_tag):
        if args.resume and stage_tag == resume_stage:
            return resume_epoch_in_stage
        return 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Float warm-up
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P1"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 1 — Float warm-up ({args.memoq_warmup_epochs} epochs)")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p1 = args.effective_lr
        opt_p1 = keras.optimizers.Adam(learning_rate=lr_p1)
        sched_p1 = ReduceLROnPlateau(
            opt_p1, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p1.reset(lr_p1)

        dist_train_p1 = make_dist_phase1_train(
            strategy, float_student, opt_p1, args.alpha, channel_scales, args.memoq_huber_delta
        )
        dist_val_p1 = make_dist_phase1_val(
            strategy, float_student, args.alpha, channel_scales, args.memoq_huber_delta
        )

        for ep_in_phase in range(start_ep("P1"), args.memoq_warmup_epochs):
            history, best_vals["P1"], patience_cts["P1"], early_stop = run_epoch(
                phase_tag="P1",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p1,
                dist_val_step_fn=dist_val_p1,
                lr_scheduler=sched_p1,
                effective_warmup_epochs=args.effective_warmup_epochs,
                effective_lr=lr_p1,
                history=history,
                best_val=best_vals["P1"],
                patience_ct=patience_cts["P1"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p1_ckpt,
                model_to_save=float_student,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P1", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p1_ckpt):
            float_student.load_weights(p1_ckpt)
            pf(f"[P1] Best weights loaded: {p1_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P1...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P1")

    # ── Transfer Phase 1 float weights into Phase 2 split-gate model ──────────
    if should_run("P2A"):
        pf("[P1->P2] Transferring float weights to split-gate model...")
        transfer_float_to_phase2(float_student, phase2_model, enc_cell_p2, dec_cell_p2, pf)
        sys.stdout.flush()



    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2A — Candidate gate (h) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2A"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2A — Candidate gate h 4-bit ({args.memoq_stage2a_epochs} epochs)")
        pf("  Quantising W_h, U_h only. z and r gates float.")
        if should_run("P2A"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2A")
            pf("=" * 60)
        sys.stdout.flush()

        qbits_h = quantized_bits(args.bits_kernel, 0, 1, alpha="auto_po2")        
        enc_cell_p2.quantizer_h = qbits_h
        dec_cell_p2.quantizer_h = qbits_h
        enc_cell_p2.quantizer_z = None
        dec_cell_p2.quantizer_z = None
        enc_cell_p2.quantizer_r = None
        dec_cell_p2.quantizer_r = None

        lr_p2a = args.effective_lr * 0.5
        opt_p2a = keras.optimizers.Adam(learning_rate=lr_p2a)
        sched_p2a = ReduceLROnPlateau(
            opt_p2a, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2a.reset(lr_p2a)

        innov_active = False
        ep2a_start = start_ep("P2A")

        innov_active = ep2a_start >= args.memoq_innov_burnin

        lambda_i_2a = args.memoq_lambda_innov if innov_active else 0.0
        dist_train_p2a = make_dist_memoq_train(
            strategy, phase2_model, opt_p2a,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.01, lambda_i_2a, 0.0, 0.0005,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=innov_active,
            use_zsat=False, use_rail=True,
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
        )
        dist_val_p2a = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.01, lambda_i_2a, 0.0, 0.0005,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=innov_active,
            use_zsat=False, use_rail=True,
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
        )

        for ep_in_phase in range(ep2a_start, args.memoq_stage2a_epochs):
            if ep_in_phase == args.memoq_innov_burnin and not innov_active:
                innov_active = True
                lambda_i_2a = args.memoq_lambda_innov
                dist_train_p2a = make_dist_memoq_train(
                    strategy, phase2_model, opt_p2a,
                    args.alpha, channel_scales, args.memoq_huber_delta,
                    0.01, lambda_i_2a, 0.0, 0.0005,
                    epsilon_innov, args.seq_len,
                    args.memoq_rho_rail, args.memoq_mu_rail,
                    use_mem=True, use_innov=True,
                    use_zsat=False, use_rail=True,
                    has_z_logit=True, clipnorm=1.0,
                    teacher_hidden_model=teacher_hidden_model,
                )
                dist_val_p2a = make_dist_memoq_val(
                    strategy, phase2_model,
                    args.alpha, channel_scales, args.memoq_huber_delta,
                    0.01, lambda_i_2a, 0.0, 0.0005,
                    epsilon_innov, args.seq_len,
                    args.memoq_rho_rail, args.memoq_mu_rail,
                    use_mem=True, use_innov=True,
                    use_zsat=False, use_rail=True,
                    has_z_logit=True,
                    teacher_hidden_model=teacher_hidden_model,
                )

            history, best_vals["P2A"], patience_cts["P2A"], early_stop = run_epoch(
                phase_tag="P2A",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2a,
                dist_val_step_fn=dist_val_p2a,
                lr_scheduler=sched_p2a,
                effective_warmup_epochs=0,
                effective_lr=lr_p2a,
                history=history,
                best_val=best_vals["P2A"],
                patience_ct=patience_cts["P2A"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2a_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2A", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2a_ckpt):
            phase2_model.load_weights(p2a_ckpt)
            pf(f"[P2A] Loaded best weights from {p2a_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2A...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2A")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2B — Reset gate (r) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2B"):
        if should_run("P2B"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2B")
            pf("=" * 60)
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2B — Reset gate r 4-bit ({args.memoq_stage2b_epochs} epochs)")
        pf("  Quantising W_r, U_r. z gate float. h gate already 4-bit.")
        pf("=" * 60)
        sys.stdout.flush()

        qbits_r = quantized_bits(args.bits_recurrent, 0, 1, alpha="auto_po2")
        enc_cell_p2.quantizer_r = qbits_r
        dec_cell_p2.quantizer_r = qbits_r

        lr_p2b = args.effective_lr * 0.3
        opt_p2b = keras.optimizers.Adam(learning_rate=lr_p2b)
        sched_p2b = ReduceLROnPlateau(
            opt_p2b, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2b.reset(lr_p2b)

        ep2b_start = start_ep("P2B")

        dist_train_p2b = make_dist_memoq_train(
            strategy, phase2_model, opt_p2b,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.03, 0.005, 0.0, 0.001,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=True,
            use_zsat=False, use_rail=True,
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
        )
        dist_val_p2b = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.03, 0.005, 0.0, 0.001,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=True,
            use_zsat=False, use_rail=True,
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
        )

        for ep_in_phase in range(ep2b_start, args.memoq_stage2b_epochs):

            history, best_vals["P2B"], patience_cts["P2B"], early_stop = run_epoch(
                phase_tag="P2B",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2b,
                dist_val_step_fn=dist_val_p2b,
                lr_scheduler=sched_p2b,
                effective_warmup_epochs=0,
                effective_lr=lr_p2b,
                history=history,
                best_val=best_vals["P2B"],
                patience_ct=patience_cts["P2B"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2b_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2B", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2b_ckpt):
            phase2_model.load_weights(p2b_ckpt)
            pf(f"[P2B] Loaded best weights from {p2b_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2B...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2B")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2C — Update gate (z) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2C"):
        if should_run("P2C"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2C")
            pf("=" * 60)
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2C — Update gate z 4-bit ({args.memoq_stage2c_epochs} epochs)")
        pf("  All gates now 4-bit. Activating L_zsat.")
        pf("=" * 60)
        sys.stdout.flush()

        qbits_z = quantized_bits(args.bits_kernel, 0, 1, alpha="auto_po2")
        enc_cell_p2.quantizer_z = qbits_z
        dec_cell_p2.quantizer_z = qbits_z

        lr_p2c = args.effective_lr * 0.2
        opt_p2c = keras.optimizers.Adam(learning_rate=lr_p2c)
        sched_p2c = ReduceLROnPlateau(
            opt_p2c, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2c.reset(lr_p2c)

        ep2c_start = start_ep("P2C")

        dist_train_p2c = make_dist_memoq_train(
            strategy, phase2_model, opt_p2c,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.05, 0.005, 0.002, 0.0002,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=True,
            use_zsat=True, use_rail=True,
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=0.5 * args.memoq_lambda_shape,
        )
        dist_val_p2c = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            0.05, 0.005, 0.002, 0.0002,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=True, use_innov=True,
            use_zsat=True, use_rail=True,
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=0.5 * args.memoq_lambda_shape,
        )

        for ep_in_phase in range(ep2c_start, args.memoq_stage2c_epochs):

            history, best_vals["P2C"], patience_cts["P2C"], early_stop = run_epoch(
                phase_tag="P2C",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2c,
                dist_val_step_fn=dist_val_p2c,
                lr_scheduler=sched_p2c,
                effective_warmup_epochs=0,
                effective_lr=lr_p2c,
                history=history,
                best_val=best_vals["P2C"],
                patience_ct=patience_cts["P2C"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2c_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2C", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2c_ckpt):
            phase2_model.load_weights(p2c_ckpt)
            pf(f"[P2C] Loaded best weights from {p2c_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2C...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2C")

    # ══════════════════════════════════════════════════════════════════════════
    # Export Phase 2 split-gate weights into final hard QKeras student.
    # CRITICAL: only do this transfer when actually entering Phase 3 fresh.
    # On resume into Phase 3 the QKeras student already has P3 checkpoint
    # weights loaded above — re-running the transfer would overwrite P3
    # progress with P2C weights and restart the quantisation cliff from scratch.
    # ══════════════════════════════════════════════════════════════════════════
    entering_p3_fresh = should_run("P3") and not (
        args.resume
        and resume_stage == "P3"
        and os.path.exists(p3_ckpt)
    )
    if entering_p3_fresh:
        pf("=" * 60)
        pf("[EXPORT] Packing split gate weights into standard QKeras QGRU format...")
        pf("=" * 60)
        sys.stdout.flush()
        transfer_splitgate_to_qkeras(enc_cell_p2, dec_cell_p2, phase2_model, final_qkeras_student, pf)
    else:
        pf("[EXPORT] Skipping transfer_splitgate_to_qkeras — resuming Phase 3 from checkpoint, weights already loaded.")
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Hard 4-bit QKeras polish
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P3"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 3 — Hard 4-bit QKeras polish ({args.memoq_stage3_epochs} epochs)")
        pf("  All QGRU/QDense weights hard 4-bit. LR=5e-6. clipnorm=0.5.")
        pf("  Loss: L_seq+0.5*L_KD+0.03*L_mem+0.002*L_innov+0.001*L_zsat+0.0005*L_rail")
        pf("=" * 60)
        sys.stdout.flush()

        # Use a meaningful starting LR for Phase 3.
        # The default memoq_phase3_lr (5e-6) is too low to recover from the
        # quantisation cliff. We use effective_lr * factor but enforce a
        # floor of 1e-5 so there is enough gradient signal on first entry.
        lr_p3 = max(
            min(args.effective_lr * args.memoq_phase3_lr_factor, args.memoq_phase3_lr),
            1e-5,
        )
        opt_p3 = keras.optimizers.Adam(learning_rate=lr_p3)
        sched_p3 = ReduceLROnPlateau(
            opt_p3, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p3.reset(lr_p3)

        ep3_start = start_ep("P3")

        dist_train_p3 = make_dist_memoq_train_final(
            strategy, final_qkeras_student, opt_p3,
            0.5, channel_scales, args.memoq_huber_delta,
            0.03, 0.00002, 0.0, 0.0,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            clipnorm=0.5,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape,
        )
        dist_val_p3 = make_dist_memoq_val_final(
            strategy, final_qkeras_student,
            0.5, channel_scales, args.memoq_huber_delta,
            0.03, 0.00002, 0.0, 0.0,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape,
        )

        for ep_in_phase in range(ep3_start, args.memoq_stage3_epochs):

            history, best_vals["P3"], patience_cts["P3"], early_stop = run_epoch(
                phase_tag="P3",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p3,
                dist_val_step_fn=dist_val_p3,
                lr_scheduler=sched_p3,
                effective_warmup_epochs=0,
                effective_lr=lr_p3,
                history=history,
                best_val=best_vals["P3"],
                patience_ct=patience_cts["P3"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p3_ckpt,
                model_to_save=final_qkeras_student,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P3", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf(f"[P3] Loaded best phase3 weights from {p3_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P3...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P3")

    mark_training_complete()

    return history, best_vals.get(
        "P3", best_vals.get(
            "P2C", best_vals.get(
                "P2B", best_vals.get(
                    "P2A", best_vals.get("P1", float("inf"))
                )
            )
        )
    )

# ==============================================================================
# transfer_splitgate_to_qkeras:
# Pack enc_cell_p2 and dec_cell_p2 split gate variables into the
# standard Keras/QKeras GRU packed layout and transfer to final_qkeras_student.
#
# Keras/QKeras GRU packed kernel shape: (input_dim, 3*units)
#   columns [0:units]     -> z gate (update)
#   columns [units:2H]    -> r gate (reset)
#   columns [2H:3H]       -> h gate (candidate)
#
# Keras/QKeras GRU packed recurrent_kernel shape: (units, 3*units)
# same column ordering.
#
# Keras/QKeras GRU packed bias shape: (2, 3*units) for reset_after=True
#   bias[0] = input bias [z|r|h]
#   bias[1] = recurrent bias [z|r|h]
# ==============================================================================

def transfer_splitgate_to_qkeras(enc_cell, dec_cell, phase2_model, final_qkeras_student, pf):
    """
    Pack split-gate MemoQGRUCell weights into standard Keras/QKeras QGRU
    packed format [z | r | h] and load into final_qkeras_student.

    MemoQGRUCell variables:
        W_z, W_r, W_h       (input_dim, H)
        U_z, U_r, U_h       (H,         H)
        b_z_inp, b_r_inp, b_h_inp  (H,) input biases
        b_z_rec, b_r_rec, b_h_rec  (H,) recurrent biases

    Keras/QKeras reset_after=True QGRU weight layouts (version-dependent):
      3-weight, bias shape (3*H,)    -> input bias only, flat
      3-weight, bias shape (2, 3*H)  -> stacked [input | recurrent]
      4-weight                       -> bias_inp (3*H,), bias_rec (3*H,) separate
      6-weight (quantizer scale vars)-> first 3 or 4 are the actual weight arrays

    This function probes the actual shape of the bias tensor and dispatches
    to the correct packing strategy so it never fails on a shape mismatch.
    """
    pf("[EXPORT] transfer_splitgate_to_qkeras: packing MemoQGRUCell -> QKeras QGRU [z|r|h]...")

    def pack_and_load(cell, layer_name, target_model):
        W_z = cell.W_z.numpy()
        W_r = cell.W_r.numpy()
        W_h = cell.W_h.numpy()

        U_z = cell.U_z.numpy()
        U_r = cell.U_r.numpy()
        U_h = cell.U_h.numpy()

        b_z_inp = cell.b_z_inp.numpy()
        b_r_inp = cell.b_r_inp.numpy()
        b_h_inp = cell.b_h_inp.numpy()

        b_z_rec = cell.b_z_rec.numpy()
        b_r_rec = cell.b_r_rec.numpy()
        b_h_rec = cell.b_h_rec.numpy()

        packed_kernel    = np.concatenate([W_z, W_r, W_h], axis=1)
        packed_recurrent = np.concatenate([U_z, U_r, U_h], axis=1)
        packed_bias_inp  = np.concatenate([b_z_inp, b_r_inp, b_h_inp], axis=0)
        packed_bias_rec  = np.concatenate([b_z_rec, b_r_rec, b_h_rec], axis=0)
        packed_bias_2row = np.stack([packed_bias_inp, packed_bias_rec], axis=0)

        try:
            target_layer = target_model.get_layer(layer_name)
        except ValueError:
            pf(f"[EXPORT]   SKIP {layer_name} — not found in target model")
            return

        q_weights = target_layer.get_weights()
        n_w = len(q_weights)

        pf(f"[EXPORT]   {layer_name}: layer has {n_w} weight tensors, shapes: {[w.shape for w in q_weights]}")

        if n_w >= 4:
            bias_inp_shape = q_weights[2].shape
            bias_rec_shape = q_weights[3].shape
            if bias_inp_shape == packed_bias_inp.shape and bias_rec_shape == packed_bias_rec.shape:
                target_layer.set_weights([
                    packed_kernel,
                    packed_recurrent,
                    packed_bias_inp,
                    packed_bias_rec,
                ] + list(q_weights[4:]))
                pf(
                    f"[EXPORT]   {layer_name}: kernel={packed_kernel.shape}  "
                    f"recurrent={packed_recurrent.shape}  "
                    f"bias_inp={packed_bias_inp.shape}  bias_rec={packed_bias_rec.shape} (4-weight mode)"
                )
                return
            pf(
                f"[EXPORT]   {layer_name}: 4-weight mode shape mismatch, "
                f"expected bias_inp={packed_bias_inp.shape} bias_rec={packed_bias_rec.shape} "
                f"got {bias_inp_shape} {bias_rec_shape}, falling through to 3-weight probe"
            )

        if n_w >= 3:
            bias_slot_shape = q_weights[2].shape
            if bias_slot_shape == packed_bias_2row.shape:
                target_layer.set_weights([
                    packed_kernel,
                    packed_recurrent,
                    packed_bias_2row,
                ] + list(q_weights[3:]))
                pf(
                    f"[EXPORT]   {layer_name}: kernel={packed_kernel.shape}  "
                    f"recurrent={packed_recurrent.shape}  bias={packed_bias_2row.shape} (3-weight 2-row bias mode)"
                )
                return
            if bias_slot_shape == packed_bias_inp.shape:
                target_layer.set_weights([
                    packed_kernel,
                    packed_recurrent,
                    packed_bias_inp,
                ] + list(q_weights[3:]))
                pf(
                    f"[EXPORT]   {layer_name}: kernel={packed_kernel.shape}  "
                    f"recurrent={packed_recurrent.shape}  bias={packed_bias_inp.shape} (3-weight flat bias mode, recurrent bias dropped)"
                )
                pf(
                    f"[EXPORT]   WARNING: {layer_name} QKeras QGRU only exposes 1 bias tensor of shape "
                    f"{bias_slot_shape}. Recurrent bias b_{{z,r,h}}_rec is not transferred. "
                    f"This is a QKeras version limitation (reset_after bias not separately stored)."
                )
                return
            pf(
                f"[EXPORT]   ERROR: {layer_name} bias slot shape {bias_slot_shape} does not match "
                f"packed_bias_2row {packed_bias_2row.shape} or packed_bias_inp {packed_bias_inp.shape}. "
                f"Attempting raw set_weights with 2-row bias to let Keras raise a clear error."
            )
            raise ValueError(
                f"transfer_splitgate_to_qkeras: layer '{layer_name}' has {n_w} weight tensors, "
                f"bias slot shape is {bias_slot_shape}, expected either "
                f"{packed_bias_inp.shape} (flat) or {packed_bias_2row.shape} (2-row). "
                f"All weight shapes: {[w.shape for w in q_weights]}. "
                f"packed_kernel={packed_kernel.shape}, packed_recurrent={packed_recurrent.shape}."
            )

        pf(
            f"[EXPORT]   ERROR: {layer_name} has only {n_w} weight tensors — too few to transfer. "
            f"Expected at least 3. Shapes: {[w.shape for w in q_weights]}"
        )
        raise ValueError(
            f"transfer_splitgate_to_qkeras: layer '{layer_name}' has {n_w} weight tensors, "
            f"cannot transfer (need >= 3). Shapes: {[w.shape for w in q_weights]}"
        )

    pack_and_load(enc_cell, "sencgru", final_qkeras_student)
    pack_and_load(dec_cell, "sdecgru", final_qkeras_student)

    pf("[EXPORT] Dense head (sdec_dense): transferring from phase2_model...")
    try:
        src_dense = phase2_model.get_layer("sdec_dense")
        dst_dense = final_qkeras_student.get_layer("sdec_dense")
        src_w = src_dense.get_weights()
        dst_w = dst_dense.get_weights()
        if len(src_w) < 2:
            pf(
                f"[EXPORT]   sdec_dense: source has only {len(src_w)} tensors, "
                f"cannot transfer kernel+bias — SKIPPED"
            )
        else:
            # Overwrite by index: slot 0 = kernel, slot 1 = bias.
            # Any trailing tensors (QKeras per-tensor scale vars) are kept
            # at their default values so quantizer state is not corrupted.
            new_w = list(dst_w)
            if new_w[0].shape != src_w[0].shape:
                pf(
                    f"[EXPORT]   sdec_dense: kernel shape mismatch "
                    f"src={src_w[0].shape} dst={new_w[0].shape} — SKIPPED"
                )
            elif new_w[1].shape != src_w[1].shape:
                pf(
                    f"[EXPORT]   sdec_dense: bias shape mismatch "
                    f"src={src_w[1].shape} dst={new_w[1].shape} — SKIPPED"
                )
            else:
                new_w[0] = src_w[0]
                new_w[1] = src_w[1]
                dst_dense.set_weights(new_w)
                pf(
                    f"[EXPORT]   sdec_dense: OK 2 tensors transferred "
                    f"(dst had {len(dst_w)} total slots, trailing slots preserved)"
                )
    except Exception as exc:
        pf(f"[EXPORT]   sdec_dense: FAILED ({exc}) — skipped")

    sys.stdout.flush()
# ==============================================================================
# MemoQGRUCell — custom training-time GRU cell with split gate variables.
#
# Variables:
#   W_z, W_r, W_h  : input kernels, shape (input_dim, units)
#   U_z, U_r, U_h  : recurrent kernels, shape (units, units)
#   b_z_inp, b_r_inp, b_h_inp : input biases, shape (units,)
#   b_z_rec, b_r_rec, b_h_rec : recurrent biases, shape (units,) [reset_after=True]
#
# Quantizers:
#   quantizer_z, quantizer_r, quantizer_h : applied per gate independently.
#   None means float (no quantization for that gate).
#   quantizer_state : applied to hidden state h_t. None means float.
#
# Outputs during training:
#   (h_t, [h_t, z_logit_t])
#   z_logit_t : pre-sigmoid update gate logit for L_zsat computation.
#   h_t is the hidden state (quantized if quantizer_state is set).
#
# At inference (in final QKeras model): this cell is NOT used.
# ==============================================================================

class MemoQGRUCell(keras.layers.Layer):
    def __init__(
        self,
        units,
        input_dim,
        quantizer_z=None,
        quantizer_r=None,
        quantizer_h=None,
        quantizer_state=None,
        quantizer_activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self._quantizer_z = quantizer_z
        self._quantizer_r = quantizer_r
        self._quantizer_h = quantizer_h
        self._quantizer_state = quantizer_state
        self._quantizer_activation = quantizer_activation
        # state_size must be a list so Keras RNN allocates two state tensors:
        #   states[0] = h_prev (B, units)
        #   states[1] = z_logit_prev (B, units)  -- carried for extraction only
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        d = self.input_dim
        H = self.units

        init  = keras.initializers.GlorotUniform()
        orth  = keras.initializers.Orthogonal()
        zeros = keras.initializers.Zeros()

        self.W_z     = self.add_weight(name="W_z",     shape=(d, H), initializer=init,  trainable=True)
        self.W_r     = self.add_weight(name="W_r",     shape=(d, H), initializer=init,  trainable=True)
        self.W_h     = self.add_weight(name="W_h",     shape=(d, H), initializer=init,  trainable=True)
        self.U_z     = self.add_weight(name="U_z",     shape=(H, H), initializer=orth,  trainable=True)
        self.U_r     = self.add_weight(name="U_r",     shape=(H, H), initializer=orth,  trainable=True)
        self.U_h     = self.add_weight(name="U_h",     shape=(H, H), initializer=orth,  trainable=True)
        self.b_z_inp = self.add_weight(name="b_z_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_r_inp = self.add_weight(name="b_r_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_h_inp = self.add_weight(name="b_h_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_z_rec = self.add_weight(name="b_z_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.b_r_rec = self.add_weight(name="b_r_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.b_h_rec = self.add_weight(name="b_h_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.built = True

    def _apply_quantizer(self, q, w):
        if q is None:
            return w
        return q(w)

    def call(self, inputs, states, training=None):
            h_prev = states[0]   # (B, units)
            # states[1] is z_logit_prev -- carried but not used in the forward pass

            # state_quantizer: quantize the incoming recurrent state (quantized_bits),
            # matching QKeras QGRU state_quantizer applied to h_tm1 at the step input.
            if self._quantizer_state is not None:
                h_prev_q = self._quantizer_state(h_prev)
            else:
                h_prev_q = h_prev

            W_z = self._apply_quantizer(self._quantizer_z, self.W_z)
            W_r = self._apply_quantizer(self._quantizer_r, self.W_r)
            W_h = self._apply_quantizer(self._quantizer_h, self.W_h)
            U_z = self._apply_quantizer(self._quantizer_z, self.U_z)
            U_r = self._apply_quantizer(self._quantizer_r, self.U_r)
            U_h = self._apply_quantizer(self._quantizer_h, self.U_h)

            z_logit = (
                tf.matmul(inputs, W_z) + self.b_z_inp
                + tf.matmul(h_prev_q, U_z) + self.b_z_rec
            )
            r_logit = (
                tf.matmul(inputs, W_r) + self.b_r_inp
                + tf.matmul(h_prev_q, U_r) + self.b_r_rec
            )

            z = tf.sigmoid(z_logit)
            r = tf.sigmoid(r_logit)

            # Candidate pre-activation (reset_after=True form).
            cand_preact = (
                tf.matmul(inputs, W_h) + self.b_h_inp
                + r * (tf.matmul(h_prev_q, U_h) + self.b_h_rec)
            )

            # Candidate activation. When quantizer_activation is set it IS
            # quantized_tanh, which computes tanh(x) then snaps to the 4-bit grid --
            # exactly what QKeras QGRU does via activation=quantized_tanh.
            # P2A/P2B (quantizer_activation None): plain tanh.
            if self._quantizer_activation is not None:
                h_candidate = self._quantizer_activation(cand_preact)
            else:
                h_candidate = tf.tanh(cand_preact)

            # Final state: convex combination. Do NOT re-quantize here -- the next
            # step's state_quantizer regrids h_t. QKeras never applies the cell
            # activation to the combined state.
            h_t = z * h_prev_q + (1.0 - z) * h_candidate

            return h_t, [h_t, z_logit]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.float32
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype),
        ]

    @property
    def quantizer_z(self):
        return self._quantizer_z

    @quantizer_z.setter
    def quantizer_z(self, q):
        self._quantizer_z = q

    @property
    def quantizer_r(self):
        return self._quantizer_r

    @quantizer_r.setter
    def quantizer_r(self, q):
        self._quantizer_r = q

    @property
    def quantizer_h(self):
        return self._quantizer_h

    @quantizer_h.setter
    def quantizer_h(self, q):
        self._quantizer_h = q

    @property
    def quantizer_state(self):
        return self._quantizer_state

    @quantizer_state.setter
    def quantizer_state(self, q):
        self._quantizer_state = q

    @property
    def quantizer_activation(self):
        return self._quantizer_activation

    @quantizer_activation.setter
    def quantizer_activation(self, q):
        self._quantizer_activation = q
# ==============================================================================
# build_phase2_model:
# Constructs the training-time phase2 model using MemoQGRUCell instances.
# Returns (model, enc_cell, dec_cell).
# The model outputs (seq_output, dec_h_seq, dec_z_logit_seq).
#   seq_output     : (batch, T, n_out) — student predictions for KD loss
#   dec_h_seq      : (batch, T, units) — decoder hidden trajectory for mem/innov/rail
#   dec_z_logit_seq: (batch, T, units) — decoder z gate logits for L_zsat
# ==============================================================================

class MemoQRNNUnroll(keras.layers.Layer):
    """
    Manually unrolls a MemoQGRUCell over a sequence and exposes
    both the hidden trajectory and the z_logit trajectory.

    Inputs : (batch, T, input_dim)
    Outputs: (hidden_seq (B,T,units), z_logit_seq (B,T,units), final_h (B,units))
    """
    def __init__(self, cell, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell = cell

    def call(self, inputs, initial_state=None, training=None):
        seq_len_static = inputs.shape[1]
        if seq_len_static is None:
            raise ValueError(
                "MemoQRNNUnroll requires a statically known sequence length. "
                "Use Input(shape=(seq_len, input_dim), ...) with a concrete integer."
            )

        batch_size = tf.shape(inputs)[0]
        units      = self.cell.units

        if initial_state is None:
            h     = tf.zeros((batch_size, units), dtype=tf.float32)
            z_log = tf.zeros((batch_size, units), dtype=tf.float32)
        else:
            h     = initial_state[0]
            z_log = initial_state[1]

        h_list = []
        z_list = []

        for t in range(seq_len_static):
            x_t = inputs[:, t, :]
            h_out, new_states = self.cell(x_t, [h, z_log], training=training)
            h     = new_states[0]
            z_log = new_states[1]
            h_list.append(h_out)
            z_list.append(z_log)

        hidden_seq  = tf.stack(h_list, axis=1)   # (B, T, units)
        z_logit_seq = tf.stack(z_list, axis=1)   # (B, T, units)
        final_h     = hidden_seq[:, -1, :]
        return hidden_seq, z_logit_seq, final_h

    def get_config(self):
        cfg = super().get_config()
        return cfg


def build_phase2_model(seq_len, n_out, student_units, input_dim=1):
    enc_cell = MemoQGRUCell(
        units=student_units,
        input_dim=input_dim,
        name="memoq_enc_cell",
    )
    dec_cell = MemoQGRUCell(
        units=student_units,
        input_dim=input_dim,
        name="memoq_dec_cell",
    )

    enc_unroll = MemoQRNNUnroll(enc_cell, name="sencgru_unroll")
    dec_unroll = MemoQRNNUnroll(dec_cell, name="sdecgru_unroll")

    enc_inputs = keras.layers.Input(shape=(seq_len, input_dim), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(seq_len, input_dim), name="sdec_input")

    enc_hidden_seq, enc_z_logit_seq, enc_final_h = enc_unroll(enc_inputs)

    enc_initial_z = keras.layers.Lambda(
        lambda x: tf.zeros_like(x),
        name="enc_initial_z_zero",
    )(enc_final_h)

    dec_hidden_seq, dec_z_logit_seq, _ = dec_unroll(
        dec_inputs,
        initial_state=[enc_final_h, enc_initial_z],
    )

    seq_output = keras.layers.Dense(n_out, activation="linear", name="sdec_dense")(dec_hidden_seq)

    model = keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[seq_output, dec_hidden_seq, dec_z_logit_seq],
        name="memoq_phase2_model",
    )

    return model, enc_cell, dec_cell

# ==============================================================================
# build_final_qkeras_student:
# Standard hard 4-bit QKeras QGRU/QDense student with identical layer names
# to what vanilla_kd uses. Output is (batch, T, n_out) only.
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense.
# ==============================================================================

def build_final_qkeras_student(
    seq_len, n_out, student_units,
    bits_kernel, bits_recurrent, bits_bias,
    bits_activation, bits_state,
):
    # auto_po2 lets each weight tensor learn a per-tensor power-of-2 scale,
    # so all 2^bits levels land where the weights actually live instead of
    # being wasted on an unused +/-1 range. State stays at alpha=1.0 because
    # the convex-combination hidden state is genuinely bounded to [-1, 1].
    def qwk():
        return quantized_bits(bits_kernel, 0, 1, alpha="auto_po2")

    def qwr():
        return quantized_bits(bits_recurrent, 0, 1, alpha="auto_po2")

    def qwb():
        return quantized_bits(bits_bias, 0, 1, alpha="auto_po2")

    def qa():
        return quantized_tanh(bits=bits_activation, symmetric=True)

    def qs():
        return quantized_bits(bits_state, 0, 1, alpha=1.0)

    def qd():
        return quantized_bits(bits_kernel, 0, 1, alpha="auto_po2")

    enc_inputs = keras.layers.Input(shape=(None, 1), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(None, 1), name="sdec_input")

    s_enc_out, s_enc_state = QGRU(
        units=student_units,
        activation=qa(),
        recurrent_activation="sigmoid",
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        reset_after=True,
        name="sencgru",
    )(enc_inputs)

    s_dec_hid_seq, _ = QGRU(
        units=student_units,
        activation=qa(),
        recurrent_activation="sigmoid",
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        reset_after=True,
        name="sdecgru",
    )(dec_inputs, initial_state=s_enc_state)

    s_output = QDense(
        n_out,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(s_dec_hid_seq)

    return keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="memoq_final_qkeras_student",
    )

# ==============================================================================
# MemoQ auxiliary losses.
# All losses are tf.function-traceable.
# ==============================================================================

def loss_mem(h_student, h_teacher, seq_len):
    """
    Memory-kernel distillation loss L_mem.

    h_student : (batch, T, Hs) — student decoder hidden sequence
    h_teacher : (batch, T, Ht) — teacher decoder hidden sequence
    seq_len   : T (integer, used for lag weight computation)

    Returns scalar tf.Tensor.
    """
    lags = [1, 2, 4, 8, 16, 32, 64]
    T = tf.cast(seq_len, tf.float32)

    def cosine_mean(h, lag):
        a = h[:, lag:, :]
        b = h[:, :-lag, :]
        dot = tf.reduce_sum(a * b, axis=-1)
        norm_a = tf.norm(a, axis=-1) + 1e-8
        norm_b = tf.norm(b, axis=-1) + 1e-8
        cos = dot / (norm_a * norm_b)
        return tf.reduce_mean(cos)

    total = tf.constant(0.0, dtype=tf.float32)
    weight_sum = tf.constant(0.0, dtype=tf.float32)

    for lag in lags:
        lag_f = tf.cast(lag, tf.float32)
        w = (1.0 / tf.sqrt(lag_f)) * ((T - lag_f) / T)
        M_s = cosine_mean(h_student, lag)
        M_t = cosine_mean(h_teacher, lag)
        total = total + w * tf.square(M_s - M_t)
        weight_sum = weight_sum + w

    return total / (weight_sum + 1e-8)


def compute_innovation_profile(h):
    """
    Per-timestep mean squared temporal innovation of hidden sequence.

    h : (batch, T, H)
    Returns v : (T-1,) tf.Tensor, mean over batch and H dimensions.
    """
    diff = h[:, 1:, :] - h[:, :-1, :]
    v = tf.reduce_mean(tf.square(diff), axis=[0, 2])
    return v


def loss_innov(h_student, h_teacher, epsilon_innov):
    """
    Temporal innovation-profile matching loss L_innov.

    Matches the per-timestep innovation curve between teacher and student
    using a log-scale squared difference. The per-timestep log-ratio is now
    clipped to [-3, 3] so a transiently frozen quantised state (v_s -> 0)
    cannot inject an unbounded gradient. The unbounded form was producing
    raw values of 15-25 in P3 and destabilising the hard fine-tune.

    h_student     : (batch, T, Hs)
    h_teacher     : (batch, T, Ht)
    epsilon_innov : float scalar, = 0.1 * median(v_t(teacher)), precomputed

    Returns scalar tf.Tensor.
    """
    eps = tf.cast(epsilon_innov, tf.float32)

    diff_s = h_student[:, 1:, :] - h_student[:, :-1, :]
    diff_t = h_teacher[:, 1:, :] - h_teacher[:, :-1, :]

    v_s = tf.reduce_mean(tf.square(diff_s), axis=[0, 2])
    v_t = tf.reduce_mean(tf.square(diff_t), axis=[0, 2])

    log_diff = tf.math.log(v_s + eps) - tf.math.log(v_t + eps)
    log_diff = tf.clip_by_value(log_diff, -3.0, 3.0)
    return tf.reduce_mean(tf.square(log_diff))


def loss_zsat_logit(z_logit_seq, logit_threshold=3.0):
    """
    Update-gate saturation barrier loss using logit form L_zsat.

    z_logit_seq : (batch, T, units) — pre-sigmoid update gate logit
    logit_threshold : float, default 3.0 (sigmoid(3) ≈ 0.95)

    L_zsat = mean(ReLU(|z_logit| - threshold)^2)

    Returns scalar tf.Tensor.
    """
    threshold = tf.cast(logit_threshold, tf.float32)
    excess = tf.nn.relu(tf.abs(z_logit_seq) - threshold)
    return tf.reduce_mean(tf.square(excess))

def loss_zsat_value(z_values, rho_z=0.98):
    """
    Update-gate saturation barrier loss using gate value form L_zsat.

    z_values : (batch, T, units) — post-sigmoid update gate values in [0, 1]
    rho_z    : float, default 0.98 — only penalises near-dead gates (z > rho or (1-z) > rho)

    L_zsat = mean(ReLU(z - rho)^2 + ReLU((1-z) - rho)^2)

    Returns scalar tf.Tensor.
    """
    rho = tf.cast(rho_z, tf.float32)
    penalty_hi = tf.nn.relu(z_values - rho)
    penalty_lo = tf.nn.relu((1.0 - z_values) - rho)
    return tf.reduce_mean(tf.square(penalty_hi) + tf.square(penalty_lo))


def loss_railpred(h_student, rho_rail=0.97, mu_rail=0.0):
    """
    Hidden-state rail barrier.

    Only penalises states inside the top (1 - rho) fraction of the quantiser
    range, i.e. about to clip at the +/-1 boundary. The previous version also
    penalised the one-step velocity |h_t - h_{t-1}| (mu_rail=0.9), which
    actively suppressed the fast hidden-state transitions required to encode
    short and long decay constants and therefore COMPRESSED the predicted-tau
    dynamic range toward the mean. The velocity term is removed (mu_rail=0.0
    by default) and the magnitude threshold is raised to 0.97 so only genuine
    saturation is discouraged, never normal use of the state range.

    h_student : (batch, T, H)
    Returns scalar tf.Tensor.
    """
    rho = tf.cast(rho_rail, tf.float32)
    mu  = tf.cast(mu_rail,  tf.float32)

    h_curr = h_student[:, 1:, :]
    h_prev = h_student[:, :-1, :]

    predicted_mag = tf.abs(h_curr) + mu * tf.abs(h_curr - h_prev)
    excess = tf.nn.relu(predicted_mag - rho)
    return tf.reduce_mean(tf.square(excess))


def loss_shape(y_true, y_pred, huber_delta, amp_floor_frac=0.05, eps=1e-6):
    """
    Amplitude-decoupled decay-shape loss.

    For the two lifetime-bearing channels (ch1 -> tau1, ch2 -> tau2) each
    per-sample curve is normalised by its own t=0 amplitude. Because
    tau = trapz(curve) / curve[t=0], the trapezoidal integral of the
    t0-normalised curve is EXACTLY the lifetime tau. Matching the normalised
    curves therefore forces the student to reproduce the teacher/target
    lifetimes AND the full decay shape, WITHOUT ever computing tau in the
    graph and WITHOUT a division-by-tiny-amplitude instability (a relative
    amplitude floor guards the denominator).

    For decay curves the peak is at t=0, so denom == amp0 and the floor
    almost never binds; it only protects the rare dim-signal sample.

    y_true, y_pred : (B, T, C) with C >= 3
    Returns scalar tf.Tensor.
    """
    delta = tf.cast(huber_delta, tf.float32)
    floor = tf.cast(amp_floor_frac, tf.float32)
    eps_f = tf.cast(eps, tf.float32)

    def normalise(seq):
        ch   = seq[:, :, 1:3]                                   # (B, T, 2)
        amp0 = ch[:, 0:1, :]                                    # (B, 1, 2) value at t=0
        peak = tf.reduce_max(tf.abs(ch), axis=1, keepdims=True) # (B, 1, 2)
        denom = tf.maximum(tf.abs(amp0), floor * peak)
        denom = tf.maximum(denom, eps_f)
        return ch / denom

    n_true   = normalise(y_true)
    n_pred   = normalise(y_pred)
    residual = n_pred - n_true
    abs_res  = tf.abs(residual)
    huber = tf.where(
        abs_res <= delta,
        0.5 * tf.square(residual),
        delta * (abs_res - 0.5 * delta),
    )
    return tf.reduce_mean(huber)

def channel_normalised_huber_memoq(y_true, y_pred, channel_scales, huber_delta):
    """
    Channel-normalised Huber loss.

    y_true, y_pred  : (batch, T, C)
    channel_scales  : (C,) tf.constant
    huber_delta     : float

    Returns scalar tf.Tensor.
    """
    residual = (y_pred - y_true) / channel_scales
    abs_res = tf.abs(residual)
    delta = tf.cast(huber_delta, tf.float32)
    huber = tf.where(
        abs_res <= delta,
        0.5 * tf.square(residual),
        delta * (abs_res - 0.5 * delta),
    )
    return tf.reduce_mean(huber)


# ==============================================================================
# make_dist_memoq_train:
# Factory that returns a distributed training step function for
# phase2 model. The phase2 model outputs (seq_out, dec_h_seq, dec_z_logit_seq).
# Teacher hidden cache is passed through the dataset as batch_x["teacher_hidden"].
# ==============================================================================

def make_dist_memoq_train(
    strategy,
    phase2_model,
    optimizer,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    use_mem,
    use_innov,
    use_zsat,
    use_rail,
    has_z_logit,
    clipnorm,
    teacher_hidden_model=None,
    lambda_s=0.0,
):
    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    lambda_s_f   = tf.cast(lambda_s,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    clipnorm_f   = tf.cast(clipnorm,      tf.float32)
    seq_len_int  = int(seq_len)

    def train_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        if teacher_hidden_model is not None:
            h_teacher = teacher_hidden_model([enc_b, dec_b], training=False)
        else:
            h_teacher = batch_x.get("teacher_hidden", None)

        with tf.GradientTape() as tape:
            model_out = phase2_model([enc_b, dec_b], training=True)

            if has_z_logit:
                s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
            else:
                s_pred, s_hid = model_out[0], model_out[1]
                z_logits = None

            l_seq = channel_normalised_huber_memoq(tgt_b,   s_pred, channel_scales, huber_delta)
            l_kd  = channel_normalised_huber_memoq(tpred_b, s_pred, channel_scales, huber_delta)
            total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

            if use_mem and h_teacher is not None:
                l_m = loss_mem(s_hid, h_teacher, seq_len_int)
                total = total + lambda_m_f * l_m
            else:
                l_m = tf.constant(0.0, dtype=tf.float32)

            if use_innov and h_teacher is not None:
                l_i = loss_innov(s_hid, h_teacher, eps_innov_f)
                total = total + lambda_i_f * l_i
            else:
                l_i = tf.constant(0.0, dtype=tf.float32)

            if use_zsat and has_z_logit and z_logits is not None:
                z_vals = tf.sigmoid(z_logits)
                l_z = loss_zsat_value(z_vals, rho_z=0.98)
                total = total + lambda_z_f * l_z
            else:
                l_z = tf.constant(0.0, dtype=tf.float32)

            if use_rail:
                l_r = loss_railpred(s_hid, rho_rail, mu_rail)
                total = total + lambda_r_f * l_r
            else:
                l_r = tf.constant(0.0, dtype=tf.float32)

            l_shape = (
                (1.0 - alpha_f) * loss_shape(tgt_b,   s_pred, huber_delta)
                + alpha_f       * loss_shape(tpred_b, s_pred, huber_delta)
            )
            total = total + lambda_s_f * l_shape

        grads = tape.gradient(total, phase2_model.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, phase2_model.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) for g in grads
        ]))
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, phase2_model.trainable_variables))

        return (
            total,
            l_seq,
            l_kd,
            l_m,
            l_i,
            l_shape,
            l_r,
            tf.cast(nan_in_grads, tf.float32),
        )

    @tf.function
    def dist_step(batch_x, batch_y):
        per = strategy.run(train_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_step

def make_dist_memoq_val(
    strategy,
    phase2_model,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    use_mem,
    use_innov,
    use_zsat,
    use_rail,
    has_z_logit,
    teacher_hidden_model=None,
    lambda_s=0.0,
):
    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    lambda_s_f   = tf.cast(lambda_s,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    seq_len_int  = int(seq_len)

    def val_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        if teacher_hidden_model is not None:
            h_teacher = teacher_hidden_model([enc_b, dec_b], training=False)
        else:
            h_teacher = batch_x.get("teacher_hidden", None)

        model_out = phase2_model([enc_b, dec_b], training=False)

        if has_z_logit:
            s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
        else:
            s_pred, s_hid = model_out[0], model_out[1]
            z_logits = None

        l_seq = channel_normalised_huber_memoq(tgt_b,   s_pred, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber_memoq(tpred_b, s_pred, channel_scales, huber_delta)
        total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

        if use_mem and h_teacher is not None:
            l_m = loss_mem(s_hid, h_teacher, seq_len_int)
            total = total + lambda_m_f * l_m
        else:
            l_m = tf.constant(0.0, dtype=tf.float32)

        if use_innov and h_teacher is not None:
            l_i = loss_innov(s_hid, h_teacher, eps_innov_f)
            total = total + lambda_i_f * l_i
        else:
            l_i = tf.constant(0.0, dtype=tf.float32)

        if use_zsat and has_z_logit and z_logits is not None:
            z_vals = tf.sigmoid(z_logits)
            l_z = loss_zsat_value(z_vals, rho_z=0.98)
            total = total + lambda_z_f * l_z
        else:
            l_z = tf.constant(0.0, dtype=tf.float32)

        if use_rail:
            l_r = loss_railpred(s_hid, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r
        else:
            l_r = tf.constant(0.0, dtype=tf.float32)

        l_shape = (
            (1.0 - alpha_f) * loss_shape(tgt_b,   s_pred, huber_delta)
            + alpha_f       * loss_shape(tpred_b, s_pred, huber_delta)
        )
        total = total + lambda_s_f * l_shape

        mae = tf.reduce_mean(tf.abs(s_pred - tgt_b))

        return (
            total,
            l_seq,
            l_kd,
            l_m,
            l_i,
            l_shape,
            l_r,
            mae,
        )

    @tf.function
    def dist_val_step(batch_x, batch_y):
        per = strategy.run(val_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_val_step

# ==============================================================================
# make_dist_memoq_train_final / make_dist_memoq_val_final:
# Phase 3 — hard QKeras final student.
# The final_qkeras_student outputs only seq_out (batch, T, n_out).
# For hidden trajectory we build a side hidden model once and call it
# inside the gradient tape. The side model shares weights with
# final_qkeras_student.
# ==============================================================================
def build_final_hidden_model(final_qkeras_student):
    """
    Build a side model from the hard QKeras student that returns
    (seq_output, dec_hidden_seq) so Phase 3 recurrent losses can fire.
    Uses explicit layer re-call from shared input tensors so the graph
    stays connected and gradients flow through final_qkeras_student.trainable_variables.
    sdec_dense is now a QDense with linear activation — no Lambda wrapper.
    """
    enc_input  = final_qkeras_student.input[0]
    dec_input  = final_qkeras_student.input[1]

    enc_layer   = final_qkeras_student.get_layer("sencgru")
    dec_layer   = final_qkeras_student.get_layer("sdecgru")
    dense_layer = final_qkeras_student.get_layer("sdec_dense")

    enc_result = enc_layer(enc_input)
    if isinstance(enc_result, (list, tuple)):
        enc_final = enc_result[1]
    else:
        enc_final = enc_result[:, -1, :]

    dec_result = dec_layer(dec_input, initial_state=enc_final)
    if isinstance(dec_result, (list, tuple)):
        dec_hidden_seq = dec_result[0]
    else:
        dec_hidden_seq = dec_result

    seq_output = dense_layer(dec_hidden_seq)

    return keras.models.Model(
        inputs=[enc_input, dec_input],
        outputs=[seq_output, dec_hidden_seq],
        name="memoq_final_hidden_model",
    )


def make_dist_memoq_train_final(
    strategy,
    final_qkeras_student,
    optimizer,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    clipnorm,
    teacher_hidden_model=None,
    lambda_s=0.0,
):
    final_hidden_model = build_final_hidden_model(final_qkeras_student)

    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    lambda_s_f   = tf.cast(lambda_s,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    clipnorm_f   = tf.cast(clipnorm,      tf.float32)
    seq_len_int  = int(seq_len)

    def train_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        if teacher_hidden_model is not None:
            teacher_hid_b = teacher_hidden_model([enc_b, dec_b], training=False)
        else:
            teacher_hid_b = batch_x.get("teacher_hidden", None)

        with tf.GradientTape() as tape:
            seq_out, dec_h_seq = final_hidden_model([enc_b, dec_b], training=True)

            l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
            l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)
            total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

            if teacher_hid_b is not None:
                l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len_int)
                total = total + lambda_m_f * l_m

                l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
                total = total + lambda_i_f * l_i
            else:
                l_m = tf.constant(0.0, dtype=tf.float32)
                l_i = tf.constant(0.0, dtype=tf.float32)

            l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r

            l_shape = (
                (1.0 - alpha_f) * loss_shape(tgt_b,   seq_out, huber_delta)
                + alpha_f       * loss_shape(tpred_b, seq_out, huber_delta)
            )
            total = total + lambda_s_f * l_shape

            l_z = tf.constant(0.0, dtype=tf.float32)

        grads = tape.gradient(total, final_qkeras_student.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, final_qkeras_student.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) for g in grads
        ]))
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, final_qkeras_student.trainable_variables))

        return total, l_seq, l_kd, l_m, l_i, l_shape, l_r, tf.cast(nan_in_grads, tf.float32)

    @tf.function
    def dist_step(batch_x, batch_y):
        per = strategy.run(train_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_step

def make_dist_memoq_val_final(
    strategy,
    final_qkeras_student,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    teacher_hidden_model=None,
    lambda_s=0.0,
):
    final_hidden_model_val = build_final_hidden_model(final_qkeras_student)

    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    lambda_s_f   = tf.cast(lambda_s,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    seq_len_int  = int(seq_len)

    def val_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        if teacher_hidden_model is not None:
            teacher_hid_b = teacher_hidden_model([enc_b, dec_b], training=False)
        else:
            teacher_hid_b = batch_x.get("teacher_hidden", None)

        seq_out, dec_h_seq = final_hidden_model_val([enc_b, dec_b], training=False)

        l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)
        total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

        if teacher_hid_b is not None:
            l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len_int)
            total = total + lambda_m_f * l_m

            l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
            total = total + lambda_i_f * l_i
        else:
            l_m = tf.constant(0.0, dtype=tf.float32)
            l_i = tf.constant(0.0, dtype=tf.float32)

        l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
        total = total + lambda_r_f * l_r

        l_shape = (
            (1.0 - alpha_f) * loss_shape(tgt_b,   seq_out, huber_delta)
            + alpha_f       * loss_shape(tpred_b, seq_out, huber_delta)
        )
        total = total + lambda_s_f * l_shape

        l_z = tf.constant(0.0, dtype=tf.float32)

        mae = tf.reduce_mean(tf.abs(seq_out - tgt_b))
        return total, l_seq, l_kd, l_m, l_i, l_shape, l_r, mae

    @tf.function
    def dist_val_step(batch_x, batch_y):
        per = strategy.run(val_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_val_step


# ==============================================================================
# run_epoch for MemoQ — extended to handle 8-output step functions.
# Identical calling convention to run_epoch in train_student_sqkd.py but
# handles (total, l_seq, l_kd, l_m, l_i, l_z, l_r, nan_flag) tuples.
# ==============================================================================
def run_epoch(
    phase_tag,
    epoch,
    total_epochs,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    dist_train_step_fn,
    dist_val_step_fn,
    lr_scheduler,
    effective_warmup_epochs,
    effective_lr,
    history,
    best_val,
    patience_ct,
    patience_max,
    min_delta,
    best_ckpt_path,
    model_to_save,
    csv_path,
    log_interval,
    nan_warn_threshold,
    pf,
    epoch_in_phase,
):
    """
    Run one epoch of training + validation for any MemoQ phase.

    Returns: (history, best_val, patience_ct, early_stop)
      history      : updated dict
      best_val     : updated best validation total loss
      patience_ct  : updated patience counter
      early_stop   : bool — True if patience exceeded
    """
    t_epoch = time.time()

    # ── Warmup LR override ────────────────────────────────────────────────────
    if effective_warmup_epochs > 0 and epoch_in_phase < effective_warmup_epochs:
        warmup_frac = (epoch_in_phase + 1) / effective_warmup_epochs
        warmup_lr   = effective_lr * warmup_frac
        lr_scheduler.lr_var.assign(float(warmup_lr))

    # ── Train ─────────────────────────────────────────────────────────────────
    train_acc   = [0.0] * 8
    nan_batches = 0
    step_count  = 0

    for step, (bx, by) in enumerate(dist_train_dataset):
        result = dist_train_step_fn(bx, by)
        result_np = [float(r) for r in result]

        nan_flag = result_np[7] if len(result_np) > 7 else 0.0
        if nan_flag > 0.0:
            nan_batches += 1

        for i in range(min(7, len(result_np))):
            train_acc[i] += result_np[i]

        step_count += 1
        if step_count >= train_steps:
            break

        if step % log_interval == 0:
            metrics_disp = {
                "total": train_acc[0] / step_count,
                "seq":   train_acc[1] / step_count,
                "kd":    train_acc[2] / step_count,
            }
            bar(step_count, train_steps, metrics_disp, t_epoch)

    if step_count > 0:
        train_acc = [v / step_count for v in train_acc]

    bar(step_count, train_steps, {"total": train_acc[0], "seq": train_acc[1], "kd": train_acc[2]}, t_epoch)

    if nan_batches >= nan_warn_threshold:
        pf(f"  [NaN] WARNING: {nan_batches}/{step_count} batches had NaN gradients in {phase_tag} epoch {epoch}")
        sys.stdout.flush()

    # ── Validate ──────────────────────────────────────────────────────────────
    val_acc    = [0.0] * 8
    val_count  = 0

    for step, (bx, by) in enumerate(dist_val_dataset):
        result = dist_val_step_fn(bx, by)
        result_np = [float(r) for r in result]
        for i in range(min(8, len(result_np))):
            val_acc[i] += result_np[i]
        val_count += 1
        if val_count >= val_steps:
            break

    if val_count > 0:
        val_acc = [v / val_count for v in val_acc]

    val_total = val_acc[0]
    val_mae   = val_acc[7] if len(val_acc) > 7 else 0.0

    # ── LR scheduler step ─────────────────────────────────────────────────────
    if epoch_in_phase >= effective_warmup_epochs:
        lr_scheduler.step(val_total, epoch, pf)

    current_lr = lr_scheduler.current_lr

    # ── History ───────────────────────────────────────────────────────────────
    history["total"].append(train_acc[0])
    history["seq"].append(train_acc[1])
    history["kd"].append(train_acc[2])
    history["mem"].append(train_acc[3])
    history["innov"].append(train_acc[4])
    history["zsat"].append(train_acc[5])
    history["rail"].append(train_acc[6])
    history["val_total"].append(val_acc[0])
    history["val_seq"].append(val_acc[1])
    history["val_kd"].append(val_acc[2])
    history["val_mem"].append(val_acc[3])
    history["val_innov"].append(val_acc[4])
    history["val_zsat"].append(val_acc[5])
    history["val_rail"].append(val_acc[6])
    history["val_mae"].append(val_mae)
    history["phase"].append(phase_tag)

    # ── CSV append ────────────────────────────────────────────────────────────
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch},{phase_tag},"
            f"{train_acc[0]:.6f},{train_acc[1]:.6f},{train_acc[2]:.6f},"
            f"{train_acc[3]:.6f},{train_acc[4]:.6f},{train_acc[5]:.6f},{train_acc[6]:.6f},"
            f"{val_acc[0]:.6f},{val_acc[1]:.6f},{val_acc[2]:.6f},"
            f"{val_acc[3]:.6f},{val_acc[4]:.6f},{val_acc[5]:.6f},{val_acc[6]:.6f},"
            f"{val_mae:.6f},{current_lr:.2e}\n"
        )

    # ── Log ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_epoch
    pf(
        f"[{phase_tag}] ep {epoch_in_phase:4d}/{total_epochs}  "
        f"loss={train_acc[0]:.5f}  val={val_total:.5f}  mae={val_mae:.5f}  "
        f"mem={train_acc[3]:.5f}  innov={train_acc[4]:.5f}  "
        f"zsat={train_acc[5]:.5f}  rail={train_acc[6]:.5f}  "
        f"lr={current_lr:.2e}  {elapsed:.1f}s"
    )
    sys.stdout.flush()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    early_stop = False
    if val_total < best_val - min_delta:
        best_val   = val_total
        patience_ct = 0
        model_to_save.save_weights(best_ckpt_path)
        pf(f"  [CKPT] Saved best {phase_tag} -> {best_ckpt_path}  val={best_val:.6f}")
        sys.stdout.flush()
    else:
        patience_ct += 1
        if patience_ct >= patience_max:
            pf(f"  [EARLY STOP] {phase_tag} patience={patience_ct}>={patience_max}")
            sys.stdout.flush()
            early_stop = True

    return history, best_val, patience_ct, early_stop
# ==============================================================================
# cache_teacher_hidden:
# Caches the decoder hidden sequence of the teacher model to a mmap file.
# teacherHidden_L{seq_len}{N}.npy — shape (N, T, teacher_units)
# ==============================================================================

def cache_teacher_hidden(
    teacher_hidden_model,
    normalized_input,
    seq_len,
    teacher_units,
    n_samples,
    infer_batch,
    data_dir,
    pf,
):
    file_hidden = os.path.join(
        data_dir, f"teacherHidden_L{seq_len}{n_samples}.npy"
    )

    if os.path.exists(file_hidden):
        pf(f"[CACHE-HID] Teacher hidden cache found — loading mmap:")
        pf(f"  {file_hidden}")
        sys.stdout.flush()
        teacher_hidden = np.load(file_hidden, mmap_mode="r")
        pf(
            f"[CACHE-HID] teacher_hidden : {teacher_hidden.shape}  "
            f"dtype={teacher_hidden.dtype}"
        )
        sys.stdout.flush()
        return teacher_hidden

    pf("=" * 60)
    pf("[CACHE-HID] Teacher hidden cache NOT found — running full-dataset inference...")
    pf(f"  hidden → {file_hidden}")
    pf("=" * 60)
    sys.stdout.flush()

    teacher_hidden = np.lib.format.open_memmap(
        file_hidden, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, teacher_units),
    )

    @tf.function(reduce_retracing=True)
    def teacher_hidden_forward(enc_b, dec_b):
        return teacher_hidden_model([enc_b, dec_b], training=False)

    wu_size = min(infer_batch, n_samples)
    enc_wu = tf.constant(normalized_input[:wu_size], dtype=tf.float32)
    dec_wu = tf.zeros((wu_size, seq_len, 1), dtype=tf.float32)
    _h = teacher_hidden_forward(enc_wu, dec_wu)
    _ = _h.numpy()
    del enc_wu, dec_wu, _h

    n_batches = int(np.ceil(n_samples / infer_batch))
    print_every = max(1, n_batches // 20)
    t0 = time.time()

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_samples)
        enc_b = tf.constant(normalized_input[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        hid = teacher_hidden_forward(enc_b, dec_b)
        teacher_hidden[s:e] = hid.numpy()
        teacher_hidden.flush()
        del enc_b, dec_b, hid
        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed = time.time() - t0
            pct = 100.0 * e / n_samples
            pf(f"[CACHE-HID] batch {b + 1:>4d}/{n_batches}  {pct:5.1f}%  elapsed={elapsed / 60:.1f}min")
            sys.stdout.flush()

    teacher_hidden.flush()
    del teacher_hidden
    teacher_hidden = np.load(file_hidden, mmap_mode="r")
    pf(
        f"[CACHE-HID] Done  shape={teacher_hidden.shape}  "
        f"dtype={teacher_hidden.dtype}"
    )
    sys.stdout.flush()
    return teacher_hidden


# ==============================================================================
# compute_epsilon_innov:
# Compute epsilon_innov = 0.1 * median(v_t(teacher)) over cached teacher hidden.
# Runs once before training.
# ==============================================================================

def compute_epsilon_innov_from_model(
    teacher_hidden_model,
    enc_train,
    seq_len,
    infer_batch,
    pf,
):
    """
    Compute epsilon_innov = 0.1 * median(v_t(teacher)) by running a small
    inference pass over at most 50000 training samples.
    Used when teacher hidden is not pre-materialised.
    """
    pf("[EPS_INNOV] Computing epsilon_innov via teacher inference on 50k sample subset...")
    n = min(50000, enc_train.shape[0])
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(enc_train.shape[0], n, replace=False)
    enc_sub = enc_train[idx].astype(np.float32)
    dec_sub = np.zeros((n, seq_len, 1), dtype=np.float32)

    h_list = []
    for s in range(0, n, infer_batch):
        e = min(s + infer_batch, n)
        enc_b = tf.constant(enc_sub[s:e], dtype=tf.float32)
        dec_b = tf.constant(dec_sub[s:e], dtype=tf.float32)
        h_chunk = teacher_hidden_model([enc_b, dec_b], training=False).numpy()
        h_list.append(h_chunk)
        del enc_b, dec_b, h_chunk

    h = np.concatenate(h_list, axis=0).astype(np.float32)
    delta = h[:, 1:, :] - h[:, :-1, :]
    v = np.mean(delta ** 2, axis=(0, 2))
    eps = 0.1 * float(np.median(v))
    eps = max(eps, 1e-6)
    pf(f"[EPS_INNOV] epsilon_innov={eps:.2e}  median_vt={float(np.median(v)):.2e}")
    sys.stdout.flush()
    return eps

# ==============================================================================
# make_memoq_kd_dataset:
# Extends make_kd_dataset to include teacher_hidden in batch_x.
# batch_x keys: enc_input, dec_input, tpred, teacher_hidden
# batch_y: ground truth labels
# ==============================================================================

def make_memoq_dataset(
    enc_arr,
    tgt_arr,
    tpred_arr,
    batch_size,
    seq_len,
    n_out,
    shuffle,
    seed,
    prefetch_batches,
    pipeline_workers,
):
    """
    tf.data pipeline for MemoQ training.
    Teacher hidden states (thid) are NOT included in the dataset —
    they are computed on-the-fly inside the train step from the live
    teacher_hidden_model. This avoids the ~443 GB RAM cost of storing
    (N, 135, 128) float32 for the full training split.
    """
    n = len(enc_arr)
    dec_arr = np.zeros_like(enc_arr)

    ds_enc   = tf.data.Dataset.from_tensor_slices(enc_arr)
    ds_dec   = tf.data.Dataset.from_tensor_slices(dec_arr)
    ds_tpred = tf.data.Dataset.from_tensor_slices(tpred_arr)
    ds_tgt   = tf.data.Dataset.from_tensor_slices(tgt_arr)

    ds = tf.data.Dataset.zip((ds_enc, ds_dec, ds_tpred, ds_tgt))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(n, 200_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=True)

    def set_shapes(enc_b, dec_b, tpred_b, tgt_b):
        enc_b.set_shape([batch_size, seq_len, 1])
        dec_b.set_shape([batch_size, seq_len, 1])
        tpred_b.set_shape([batch_size, seq_len, n_out])
        tgt_b.set_shape([batch_size, seq_len, n_out])
        batchx = {
            "enc_input": enc_b,
            "dec_input": dec_b,
            "tpred":     tpred_b,
        }
        return batchx, tgt_b

    ds = ds.map(set_shapes, num_parallel_calls=pipeline_workers)
    ds = ds.prefetch(prefetch_batches)
    return ds

# ==============================================================================
# materialise_memoq_buffers:
# Like materialise_enc_tgt_tpred but also materialises teacher_hidden.
# ==============================================================================
class _MemmapSubset:
    """
    Lightweight wrapper around a np.memmap that provides integer-indexed access
    to a pre-sorted subset without loading the full array into RAM.

    Reads are sequential in on-disk order (sorted_idx) to minimise random I/O.
    Accessed via plain integer index into the logical (original) subset order
    using the stored unsort_order permutation.

    Parameters
    ----------
    mmap        : np.memmap  full memory-mapped array, shape (N_total, T, H)
    sorted_idx  : np.ndarray int64, sorted indices into the full array
    unsort_order: np.ndarray int64, argsort(argsort(original_idx))
                  maps logical position → sorted position
    """
    def __init__(self, mmap, sorted_idx, unsort_order):
        self._mmap        = mmap
        self._sorted_idx  = sorted_idx
        self._unsort      = unsort_order

    def __len__(self):
        return len(self._sorted_idx)

    def __getitem__(self, i):
        """Return sample at logical position i as a float32 ndarray (T, H)."""
        disk_pos = self._unsort[i]
        return self._mmap[self._sorted_idx[disk_pos]].astype(np.float32)

    @property
    def shape(self):
        return (len(self._sorted_idx),) + self._mmap.shape[1:]

    @property
    def dtype(self):
        return np.float32


def materialise_memoq_buffers(
    normalized_input,
    res,
    teacher_predictions,
    idx,
    seq_len,
    n_out,
    label,
    pf,
):
    """
    Materialise encoder, decoder-target, and teacher-prediction buffers
    for a single data split (train / val / test).

    This function is called once per split. It performs a fancy-index
    copy of only the rows required for that split, converting memmap
    arrays into contiguous float32 arrays that tf.data can consume
    without triggering repeated disk I/O during training.

    Parameters
    ----------
    normalized_input : np.ndarray, shape (N, seq_len, 1)
        Normalised encoder input for the full dataset (memmap or ndarray).
    res : np.ndarray, shape (N, seq_len, n_out) or (N, n_out)
        Decoder regression targets for the full dataset (memmap or ndarray).
    teacher_predictions : np.ndarray, shape (N, seq_len, n_out)
        Cached teacher output predictions for the full dataset (memmap).
    idx : np.ndarray, shape (n_split,)
        Integer indices selecting the rows that belong to this split.
    seq_len : int
        Sequence length (used for shape-assertion logging only).
    n_out : int
        Number of output channels (used for shape-assertion logging only).
    label : str
        Human-readable split name used in log messages ('train', 'val', 'test').
    pf : callable
        Print function (e.g. lambda msg: print(msg, flush=True)).

    Returns
    -------
    enc_buf : np.ndarray, shape (n_split, seq_len, 1), dtype float32
        Encoder inputs for this split.
    tgt_buf : np.ndarray, shape (n_split, ...), dtype float32
        Decoder targets for this split.
    tpred_buf : np.ndarray, shape (n_split, seq_len, n_out), dtype float32
        Teacher predictions for this split.
    """
    n_split = len(idx)
    pf(f"[BUFFERS:{label}] Materialising {n_split} samples "
       f"(enc seq_len={seq_len}, n_out={n_out})...")

    enc_buf   = np.asarray(normalized_input[idx],    dtype=np.float32)
    tgt_buf   = np.asarray(res[idx],                 dtype=np.float32)
    tpred_buf = np.asarray(teacher_predictions[idx], dtype=np.float32)

    pf(f"[BUFFERS:{label}] enc_buf   shape={enc_buf.shape}   dtype={enc_buf.dtype}")
    pf(f"[BUFFERS:{label}] tgt_buf   shape={tgt_buf.shape}   dtype={tgt_buf.dtype}")
    pf(f"[BUFFERS:{label}] tpred_buf shape={tpred_buf.shape} dtype={tpred_buf.dtype}")

    return enc_buf, tgt_buf, tpred_buf

# ==============================================================================
# transfer_float_to_phase2:
# Transfer weights from phase1 float student (standard Keras GRU) into the
# MemoQGRUCell split-gate variables of phase2_model.
# The float GRU stores packed kernel (input_dim, 3*H) and
# recurrent_kernel (H, 3*H) in columns [z, r, h] order.
# ==============================================================================

def transfer_float_to_phase2(float_student, phase2_model, enc_cell_p2, dec_cell_p2, pf):
    """
    Transfer weights from phase1 float student (standard Keras GRU, reset_after=True)
    into the MemoQGRUCell split-gate variables of phase2_model.

    Keras GRU reset_after=True weight layout:
      kernel           (input_dim, 3*H)  columns: [z | r | h]
      recurrent_kernel (H,         3*H)  columns: [z | r | h]
      bias             (2,         3*H)  row 0 = input bias [z|r|h], row 1 = recurrent bias [z|r|h]

    MemoQGRUCell variables:
      W_z, W_r, W_h       (input_dim, H)
      U_z, U_r, U_h       (H,         H)
      b_z_inp, b_r_inp, b_h_inp  (H,)
      b_z_rec, b_r_rec, b_h_rec  (H,)
    """
    pf("[P1->P2 TRANSFER] Unpacking float GRU weights into MemoQGRUCell split-gate variables...")

    def unpack_and_set(gru_layer_name, cell):
        try:
            layer = float_student.get_layer(gru_layer_name)
        except ValueError:
            pf(f"  SKIP {gru_layer_name} — not found in float_student")
            return

        weights = layer.get_weights()
        if len(weights) < 3:
            pf(f"  SKIP {gru_layer_name} — only {len(weights)} weight tensors (expected 3)")
            return

        kernel           = weights[0]   # (input_dim, 3*H)
        recurrent_kernel = weights[1]   # (H, 3*H)
        bias             = weights[2]   # (2, 3*H) for reset_after=True

        H = cell.units

        if kernel.shape[1] != 3 * H:
            pf(f"  SKIP {gru_layer_name} — kernel column count {kernel.shape[1]} != 3*{H}")
            return

        # Keras GRU packed column order: [z | r | h]
        W_z = kernel[:, 0:H]
        W_r = kernel[:, H:2*H]
        W_h = kernel[:, 2*H:3*H]

        U_z = recurrent_kernel[:, 0:H]
        U_r = recurrent_kernel[:, H:2*H]
        U_h = recurrent_kernel[:, 2*H:3*H]

        if bias.ndim == 2 and bias.shape[0] == 2:
            # reset_after=True: row 0 = input bias, row 1 = recurrent bias
            b_z_inp = bias[0, 0:H]
            b_r_inp = bias[0, H:2*H]
            b_h_inp = bias[0, 2*H:3*H]
            b_z_rec = bias[1, 0:H]
            b_r_rec = bias[1, H:2*H]
            b_h_rec = bias[1, 2*H:3*H]
        elif bias.ndim == 1 and bias.shape[0] == 3 * H:
            # reset_after=False fallback: single flat bias, all recurrent biases = 0
            b_z_inp = bias[0:H]
            b_r_inp = bias[H:2*H]
            b_h_inp = bias[2*H:3*H]
            b_z_rec = np.zeros(H, dtype=np.float32)
            b_r_rec = np.zeros(H, dtype=np.float32)
            b_h_rec = np.zeros(H, dtype=np.float32)
        else:
            pf(f"  SKIP {gru_layer_name} — unexpected bias shape {bias.shape}")
            return

        cell.W_z.assign(W_z)
        cell.W_r.assign(W_r)
        cell.W_h.assign(W_h)
        cell.U_z.assign(U_z)
        cell.U_r.assign(U_r)
        cell.U_h.assign(U_h)
        cell.b_z_inp.assign(b_z_inp)
        cell.b_r_inp.assign(b_r_inp)
        cell.b_h_inp.assign(b_h_inp)
        cell.b_z_rec.assign(b_z_rec)
        cell.b_r_rec.assign(b_r_rec)
        cell.b_h_rec.assign(b_h_rec)

        pf(
            f"  OK {gru_layer_name}: "
            f"W_z={W_z.shape} W_r={W_r.shape} W_h={W_h.shape}  "
            f"U_z={U_z.shape} U_r={U_r.shape} U_h={U_h.shape}  "
            f"b_*_inp={b_z_inp.shape} b_*_rec={b_z_rec.shape}"
        )
        sys.stdout.flush()

    unpack_and_set("sencgru", enc_cell_p2)
    unpack_and_set("sdecgru", dec_cell_p2)

    try:
        src_dense = float_student.get_layer("sdec_dense")
        dst_dense = phase2_model.get_layer("sdec_dense")
        src_w = src_dense.get_weights()
        dst_w = dst_dense.get_weights()
        if len(src_w) == len(dst_w) and all(s.shape == d.shape for s, d in zip(src_w, dst_w)):
            dst_dense.set_weights(src_w)
            pf("  OK sdec_dense")
        else:
            pf(
                f"  SKIP sdec_dense — shape mismatch "
                f"src={[w.shape for w in src_w]} dst={[w.shape for w in dst_w]}"
            )
    except Exception as exc:
        pf(f"  SKIP sdec_dense: {exc}")

    sys.stdout.flush()

# ==============================================================================
# save_loss_curves_memoq:
# Multi-panel PNG with all 7 loss components across all phases.
# ==============================================================================

def save_loss_curves_memoq(history, best_val_loss, args, job_dir, pf):
    phases_arr = history.get("phase", [])
    n_ep = len(history.get("total", []))

    if n_ep == 0:
        pf("[CURVES] history is empty (completion-flag early return) — skipping plot to preserve existing training_history.png")
        sys.stdout.flush()
        return

    epochs_arr = list(range(1, n_ep + 1))

    phase_colors = {
        "P1":  ("tab:blue",   "tab:cyan"),
        "P2A": ("tab:orange", "tab:red"),
        "P2B": ("tab:purple", "tab:pink"),
        "P2C": ("tab:brown",  "tab:olive"),
        "P3":  ("tab:green",  "limegreen"),
    }

    fig, axes = plt.subplots(2, 5, figsize=(36, 8))
    axes = axes.flatten()

    loss_pairs = [
        ("total",   "val_total",  "Total Loss"),
        ("seq",     "val_seq",    "L_seq (HuberCN GT)"),
        ("kd",      "val_kd",     "L_KD (HuberCN Teacher)"),
        ("mem",     "val_mem",    "L_mem (Memory Kernel)"),
        ("innov",   "val_innov",  "L_innov (Innovation)"),
        ("zsat",    "val_zsat",   "L_zsat (Gate Sat)"),
        ("rail",    "val_rail",   "L_railpred"),
        ("val_mae", None,         "Val MAE"),
    ]

    for ax_idx, (train_key, val_key, title) in enumerate(loss_pairs):
        if ax_idx >= len(axes) - 2:
            break
        ax = axes[ax_idx]
        for phase_tag, (ctr, cva) in phase_colors.items():
            mask = [i for i, ph in enumerate(phases_arr) if ph == phase_tag]
            if not mask:
                continue
            ep_ph   = [epochs_arr[i] for i in mask]
            tr_vals = [history.get(train_key, [0] * n_ep)[i] for i in mask]
            ax.plot(ep_ph, tr_vals, color=ctr, label=f"{phase_tag} train", linewidth=1.2)
            if val_key and val_key in history and len(history[val_key]) == n_ep:
                va_vals = [history[val_key][i] for i in mask]
                ax.plot(ep_ph, va_vals, color=cva, linestyle="--", label=f"{phase_tag} val", linewidth=1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[8].axis("off")
    axes[8].text(
        0.05, 0.5,
        f"MemoQ  SEQLEN={args.seq_len}\n"
        f"alpha={args.alpha}  huber_delta={args.memoq_huber_delta}\n"
        f"bits: k={args.bits_kernel} r={args.bits_recurrent} "
        f"b={args.bits_bias} a={args.bits_activation} s={args.bits_state}\n"
        f"Student QGRU hidden={args.student_units}\n"
        f"Teacher GRU hidden={args.teacher_units} x {args.teacher_layers}\n"
        f"P1={args.memoq_stage1_epochs}  2A={args.memoq_stage2a_epochs}  "
        f"2B={args.memoq_stage2b_epochs}  2C={args.memoq_stage2c_epochs}  "
        f"P3={args.memoq_stage3_epochs}\n"
        f"Batch={args.batch_size}  EffLR={args.effective_lr:.2e}\n"
        f"rho_rail={args.memoq_rho_rail}  mu_rail={args.memoq_mu_rail}\n"
        f"innov_burnin={args.memoq_innov_burnin}\n"
        f"Best val (P3)={best_val_loss:.6f}\n"
        f"Total epochs run={n_ep}",
        fontsize=8,
        verticalalignment="center",
        transform=axes[8].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    axes[9].axis("off")

    plt.tight_layout()
    curves_path = os.path.join(job_dir, "training_history.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"Loss curves saved: {curves_path}")
    sys.stdout.flush()

# ==============================================================================
# evaluate_and_save — identical to vanilla_kd but uses final_qkeras_student.
# ==============================================================================
def evaluate_and_save(
    model,
    enc_test,
    tgt_test,
    labels_test,
    gate_width_ns,
    n_out,
    seq_len,
    infer_batch,
    job_dir,
    args,
    pf,
    phase_tag=None,
):
    """
    Run inference on the test split, compute lifetime scatter metrics, and
    save scatter PNGs in the SAME visual style as train_student_vanilla_kd.py
    (hexbin, Blues/Greens/Oranges, log count colorbar, y=x line, RMSE/r/cov box).

    `model` is whichever Keras model currently holds the trained weights for
    this phase:
        P1            -> float_student            (output: preds)
        P2A/P2B/P2C   -> phase2_model             (output: [preds, hidden, zlogits])
        P3 / final    -> final_qkeras_student      (output: preds)

    The function auto-detects multi-output models and uses output 0 as the
    prediction tensor, so all three model types route through one code path.

    phase_tag : str or None.  When provided (e.g. "P1", "P2A", "P2B", "P2C",
                "P3", "final") the PNGs are written as
                test_scatter_tau1_<phase_tag>.png etc in addition to the
                canonical names (which are always written / overwritten).
    """

    TAU1_MAX_PHYS = 3.0   # ns — physical label range, same as vanilla_kd
    TAU2_MAX_PHYS = 3.0   # ns
    FRET_MAX_PHYS = 1.0

    pf(f"[EVAL] Running evaluate_and_save (phase_tag={phase_tag!r}) ...")
    sys.stdout.flush()

    n_test   = enc_test.shape[0]
    dec_test = np.zeros((n_test, seq_len, 1), dtype=np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────
    pf(f"[EVAL] Inference on {n_test} test samples (batch={infer_batch}) ...")
    sys.stdout.flush()
    preds_list = []
    for s in range(0, n_test, infer_batch):
        e = min(s + infer_batch, n_test)
        out = model([enc_test[s:e], dec_test[s:e]], training=False)
        if isinstance(out, (list, tuple)):
            batch_pred = out[0]
        else:
            batch_pred = out
        if hasattr(batch_pred, "numpy"):
            batch_pred = batch_pred.numpy()
        preds_list.append(np.asarray(batch_pred, dtype=np.float32))
    preds = np.concatenate(preds_list, axis=0).astype(np.float32)

    # ── MAE on raw seq targets ─────────────────────────────────────────────
    mae_seq = float(np.mean(np.abs(preds - tgt_test)))
    pf(f"[EVAL] Sequence MAE (test): {mae_seq:.6f}")
    sys.stdout.flush()

    # ── Lifetime extraction ────────────────────────────────────────────────
    t_ns = np.arange(seq_len, dtype=np.float32) * float(gate_width_ns)

    def extract_lifetimes(seq_arr, t_axis):
        """
        Extract tau1, tau2, fret_eff from a (N, T, 3) predicted sequence.

        Channel layout (matches vanilla_kd extract_lifetimes):
          ch0 = full decay (unused for tau/fret)
          ch1 = short decay  -> tau1
          ch2 = long decay   -> tau2

        Integration: trapezoidal rule over the physical time axis t_axis (ns),
        identical to vanilla_kd (np.trapz(ch, t, axis=1)).
        Amplitude = value at t=0 (first gate).
        tau = integral / amplitude when amplitude > 1e-6 else 0.0.
        fret = amp1 / (amp1 + amp2) when denom > 1e-6 else 0.5.

        This mirrors vanilla_kd EXACTLY so student and teacher scatters are
        directly comparable in the paper. No physical clipping, no relative
        amplitude guard — vanilla does neither.
        """
        ch1 = seq_arr[:, :, 1]
        ch2 = seq_arr[:, :, 2]

        int1 = np.trapz(ch1, t_axis, axis=1)
        int2 = np.trapz(ch2, t_axis, axis=1)

        amp1 = ch1[:, 0]
        amp2 = ch2[:, 0]

        tau1 = np.where(amp1 > 1e-6, int1 / amp1, 0.0).astype(np.float32)
        tau2 = np.where(amp2 > 1e-6, int2 / amp2, 0.0).astype(np.float32)

        denom = amp1 + amp2
        fret  = np.where(denom > 1e-6, amp1 / denom, 0.5).astype(np.float32)
        return tau1, tau2, fret

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(preds, t_ns)

    # Ground truth = physical scalar labels.
    if labels_test is not None and labels_test.ndim == 2 and labels_test.shape[1] >= 3:
        tau1_gt = labels_test[:, 0].astype(np.float32)
        tau2_gt = labels_test[:, 1].astype(np.float32)
        fret_gt = labels_test[:, 2].astype(np.float32)
    else:
        pf("[EVAL] WARNING: labels_test missing or !=3ch — falling back to target-derived GT")
        tau1_gt, tau2_gt, fret_gt = extract_lifetimes(tgt_test, t_ns)

    # ── Metrics: RMSE, Pearson r, 1-sigma coverage (vanilla compute_metrics) ──
    def compute_metrics(gt, pred):
        mask = np.isfinite(gt) & np.isfinite(pred)
        if mask.sum() < 5:
            return float("nan"), float("nan"), float("nan")
        gt_m   = gt[mask].astype(float)
        pred_m = pred[mask].astype(float)
        rmse = float(np.sqrt(np.mean((gt_m - pred_m) ** 2)))
        r    = float(pearsonr(gt_m, pred_m)[0])
        residuals = pred_m - gt_m
        sigma = residuals.std()
        cov   = float(np.mean(np.abs(residuals) <= sigma) * 100.0)
        return rmse, r, cov

    rmse_tau1, r_tau1, cov_tau1 = compute_metrics(tau1_gt, tau1_pred)
    rmse_tau2, r_tau2, cov_tau2 = compute_metrics(tau2_gt, tau2_pred)
    rmse_fret, r_fret, cov_fret = compute_metrics(fret_gt, fret_pred)

    pf(f"[EVAL] tau1  RMSE={rmse_tau1:.4f}  r={r_tau1:.4f}  1σ-cov={cov_tau1:.1f}%")
    pf(f"[EVAL] tau2  RMSE={rmse_tau2:.4f}  r={r_tau2:.4f}  1σ-cov={cov_tau2:.1f}%")
    pf(f"[EVAL] fret  RMSE={rmse_fret:.4f}  r={r_fret:.4f}  1σ-cov={cov_fret:.1f}%")
    sys.stdout.flush()

    title_suffix = f"  [{phase_tag}]" if phase_tag else ""

    # ── Scatter plots — vanilla style: hexbin, log bins, colored cmap, y=x ────
    panels = [
        (tau1_gt, tau1_pred, rmse_tau1, r_tau1, cov_tau1,
         "τ₁ (ns)", "GT τ₁ (ns)", "Pred τ₁ (ns)",
         (0.0, TAU1_MAX_PHYS), "Blues",   "test_scatter_tau1"),
        (tau2_gt, tau2_pred, rmse_tau2, r_tau2, cov_tau2,
         "τ₂ (ns)", "GT τ₂ (ns)", "Pred τ₂ (ns)",
         (0.0, TAU2_MAX_PHYS), "Greens",  "test_scatter_tau2"),
        (fret_gt, fret_pred, rmse_fret, r_fret, cov_fret,
         "FRET (f)", "GT FRET (f)", "Pred FRET (f)",
         (0.0, FRET_MAX_PHYS), "Oranges", "test_scatter_fret"),
    ]

    for (gt_c, pred_c, rmse_v, r_v, cov_v,
         title, xlabel, ylabel, lims, cmap, base_name) in panels:
        lo, hi = lims
        mask = np.isfinite(gt_c) & np.isfinite(pred_c)
        gt_plot   = gt_c[mask]
        pred_plot = pred_c[mask]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if len(gt_plot) > 0:
            hb = ax.hexbin(
                gt_plot, pred_plot,
                gridsize=80, bins="log", cmap=cmap,
                extent=(lo, hi, lo, hi), mincnt=1,
            )
            fig.colorbar(hb, ax=ax, pad=0.02).set_label("log₁₀(count)", fontsize=9)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        phase_title = f"{phase_tag}" if phase_tag else "memoq"
        ax.set_title(f"{title}  {phase_title}", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.03, 0.97,
            f"RMSE={rmse_v:.4f}\nr={r_v:.4f}\n1σ-cov={cov_v:.1f}%",
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()

        canonical_path = os.path.join(job_dir, f"{base_name}.png")
        plt.savefig(canonical_path, dpi=150, bbox_inches="tight")
        pf(f"[EVAL] Saved {canonical_path}")

        if phase_tag is not None:
            tagged_path = os.path.join(job_dir, f"{base_name}_{phase_tag}.png")
            plt.savefig(tagged_path, dpi=150, bbox_inches="tight")
            pf(f"[EVAL] Saved {tagged_path}")

        plt.close(fig)

    sys.stdout.flush()

    # ── Residuals plot — vanilla style (3-panel histogram, mu/sigma box) ──────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, gt_c, pred_c, label, color in zip(
        axes,
        [tau1_gt, tau2_gt, fret_gt],
        [tau1_pred, tau2_pred, fret_pred],
        ["τ₁ (ns)", "τ₂ (ns)", "FRET (f)"],
        ["steelblue", "seagreen", "darkorange"],
    ):
        mask = np.isfinite(gt_c) & np.isfinite(pred_c)
        residuals = pred_c[mask] - gt_c[mask]
        ax.hist(residuals, bins=100, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel(f"Residual {label}", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Residuals {label}{title_suffix}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.97, 0.97,
            f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    phase_title = f"{phase_tag}" if phase_tag else "memoq"
    fig.suptitle(f"Residuals  {phase_title}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    residuals_path = os.path.join(job_dir, "test_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    if phase_tag is not None:
        plt.savefig(os.path.join(job_dir, f"test_residuals_{phase_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"[EVAL] Saved {residuals_path}")
    sys.stdout.flush()

    # ── Metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "mae_seq":          mae_seq,
        "rmse_tau1":        rmse_tau1,
        "rmse_tau2":        rmse_tau2,
        "rmse_fret":        rmse_fret,
        "r_tau1":           r_tau1,
        "r_tau2":           r_tau2,
        "r_fret":           r_fret,
        "cov1sigma_tau1":   cov_tau1,
        "cov1sigma_tau2":   cov_tau2,
        "cov1sigma_fret":   cov_fret,
        "n_test":           n_test,
        "n_valid_tau1":     int(np.isfinite(tau1_pred).sum()),
        "n_valid_tau2":     int(np.isfinite(tau2_pred).sum()),
        "n_valid_fret":     int(np.isfinite(fret_pred).sum()),
        "phase_tag":        phase_tag,
    }
    metrics_path = os.path.join(job_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    if phase_tag is not None:
        phase_metrics_path = os.path.join(job_dir, f"test_metrics_{phase_tag}.json")
        with open(phase_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    pf(f"[EVAL] Metrics: {metrics}")
    pf(f"[EVAL] Saved {metrics_path}")
    sys.stdout.flush()

    return metrics
# ==============================================================================
# parse_args — all MemoQ-specific hyperparameters
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MemoQ: memory-preserving 4-bit recurrent quantization KD for QGRU Seq2Seq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data-dir",      type=str, required=True)
    p.add_argument("--save-dir",      type=str, default=None)
    p.add_argument("--seq-len",       type=int, default=135)
    p.add_argument("--n-out",         type=int, default=3)
    p.add_argument("--gate-width-ns", type=float, default=0.09)

    p.add_argument("--teacher-ckpt",   type=str, required=True)
    p.add_argument("--teacher-units",  type=int, default=128)
    p.add_argument("--teacher-layers", type=int, default=2)

    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha",       type=float, default=0.7)

    p.add_argument("--bits-kernel",     type=int, default=4)
    p.add_argument("--bits-bias",       type=int, default=4)
    p.add_argument("--bits-recurrent",  type=int, default=4)
    p.add_argument("--bits-activation", type=int, default=4)
    p.add_argument("--bits-state",      type=int, default=4)

    p.add_argument("--student-units", type=int, default=32)

    p.add_argument(
        "--memoq-warmup-epochs", "--memoq-stage1-epochs",
        dest="memoq_warmup_epochs", type=int, default=40,
    )
    p.add_argument("--memoq-stage2a-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2b-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2c-epochs", type=int, default=30)
    p.add_argument(
        "--memoq-finetune-epochs", "--memoq-stage3-epochs",
        dest="memoq_stage3_epochs", type=int, default=170,
    )

    p.add_argument("--memoq-lambda-mem",   type=float, default=0.03)
    p.add_argument("--memoq-lambda-innov", type=float, default=0.005)
    p.add_argument("--memoq-lambda-zsat",  type=float, default=0.002)
    p.add_argument("--memoq-lambda-rail",  type=float, default=0.001)
    p.add_argument("--memoq-lambda-shape", type=float, default=0.15,
                   help="Weight on the amplitude-decoupled decay-shape loss "
                        "(t0-normalised curve = exact tau surrogate). "
                        "P2C uses half this value, P3 uses the full value.")

    p.add_argument("--memoq-huber-delta",    type=float, default=0.1)
    p.add_argument("--memoq-rho-z",          type=float, default=0.98)
    p.add_argument("--memoq-rho-rail",       type=float, default=0.97)
    p.add_argument("--memoq-mu-rail",        type=float, default=0.0)
    p.add_argument("--memoq-innov-burnin",   type=int,   default=5)

    p.add_argument("--memoq-phase3-lr",        type=float, default=5e-6)
    p.add_argument("--memoq-phase3-lr-factor", type=float, default=1.0)

    p.add_argument("--batch-size",        type=int,   default=1024)
    p.add_argument("--epochs",            type=int,   default=330)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--ref-batch-size",    type=int,   default=1024)
    p.add_argument("--no-lr-scaling",     action="store_true", default=False)
    p.add_argument("--lr-factor",         type=float, default=0.5)
    p.add_argument("--lr-patience",       type=int,   default=8)
    p.add_argument("--lr-min",            type=float, default=1e-6)
    p.add_argument("--patience",          type=int,   default=30)
    p.add_argument("--min-delta",         type=float, default=1e-5)
    p.add_argument("--infer-batch",       type=int,   default=8192)
    p.add_argument("--mixed-precision",   action="store_true", default=False)
    p.add_argument("--log-interval",      type=int,   default=10)
    p.add_argument("--prefetch-batches",  type=int,   default=32)
    p.add_argument("--pipeline-workers",  type=int,   default=4)
    p.add_argument("--split-seed",        type=int,   default=42)
    p.add_argument("--warmup-epochs",     type=int,   default=5)
    p.add_argument("--accumulation-steps", type=int,  default=1)
    p.add_argument("--resume",            action="store_true", default=False)

    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = args.data_dir

    args.memoq_stage1_epochs   = args.memoq_warmup_epochs
    args.memoq_finetune_epochs = args.memoq_stage3_epochs

    scale = float(args.batch_size) / float(args.ref_batch_size)
    if args.no_lr_scaling:
        args.effective_lr             = args.lr
        args.effective_lr_patience    = args.lr_patience
        args.effective_warmup_epochs  = args.warmup_epochs
    else:
        args.effective_lr             = args.lr * scale
        args.effective_lr_patience    = max(1, int(round(args.lr_patience / scale)))
        args.effective_warmup_epochs  = max(1, int(round(args.warmup_epochs * scale)))

    return args

# ==============================================================================
# make_job_name
# ==============================================================================

def make_job_name(args) -> str:
    effective_batch = args.batch_size
    micro_batch     = args.batch_size
    if args.no_lr_scaling:
        effective_lr = args.lr
    else:
        effective_lr = args.lr * (args.batch_size / args.ref_batch_size)
    lr_str = f"{effective_lr:.0e}"
    return (
        f"memoq"
        f"_b{args.bits_kernel}"
        f"k{args.bits_bias}"
        f"r{args.bits_recurrent}"
        f"a{args.bits_activation}"
        f"s{args.bits_state}"
        f"_gru{args.student_units}"
        f"_dense{args.n_out}"
        f"_effbs{effective_batch}"
        f"_microbs{micro_batch}"
        f"_lr{lr_str}"
        f"_p1-{args.memoq_warmup_epochs}"
        f"_2a{args.memoq_stage2a_epochs}"
        f"_2b{args.memoq_stage2b_epochs}"
        f"_2c{args.memoq_stage2c_epochs}"
        f"_p3-{args.memoq_finetune_epochs}"
    )

# ==============================================================================
# setup_gpus_and_strategy — identical to vanilla_kd
# ==============================================================================

def setup_gpus_and_strategy(mixed_precision_flag, pf=print):
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        pf("[GPU] No GPUs found — running on CPU.")
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pf(f"[GPU]   WARNING set_memory_growth: {e}")

    logical_gpus = tf.config.list_logical_devices("GPU")
    pf(f"[GPU] Physical={len(physical_gpus)}  Logical={len(logical_gpus)}")

    keras.mixed_precision.set_global_policy("float32")
    pf("[GPU] Policy: float32 (QKeras requirement)")

    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    pf(f"[GPU] MirroredStrategy replicas={strategy.num_replicas_in_sync}")
    sys.stdout.flush()
    return strategy


# ==============================================================================
# build_teacher — exactly as in vanilla_kd
# ==============================================================================

def build_teacher(seq_len, n_out, teacher_units, teacher_layers):
    LAYERS_TEACHER = [teacher_units] * teacher_layers

    encoder_inputs = keras.layers.Input(shape=(None, 1), name="enc_input")
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells, return_state=True, name="enc_rnn",
    )
    encoder_out_states = encoder_rnn(encoder_inputs)
    encoder_states = encoder_out_states[1:]

    decoder_inputs = keras.layers.Input(shape=(None, 1), name="dec_input")
    decoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"dec_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    decoder_rnn = keras.layers.RNN(
        decoder_cells,
        return_sequences=True,
        return_state=True,
        name="dec_rnn",
    )
    decoder_out_states = decoder_rnn(decoder_inputs, initial_state=encoder_states)
    decoder_hidden_sequence = decoder_out_states[0]

    decoder_dense = keras.layers.Dense(n_out, activation="linear", name="dec_dense")
    decoder_output = decoder_dense(decoder_hidden_sequence)

    teacher_model = keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_output,
        name="teacher_seq2seq",
    )
    return teacher_model


def build_teacher_hidden_model(teacher_model):
    """
    Returns a model that exposes the decoder hidden sequence from teacher.
    Shares all weights with teacher_model (same layer objects).
    """
    dec_rnn_layer = teacher_model.get_layer("dec_rnn")
    dec_rnn_out = dec_rnn_layer.output
    if isinstance(dec_rnn_out, (list, tuple)):
        dec_hidden_seq = dec_rnn_out[0]
    else:
        dec_hidden_seq = dec_rnn_out

    teacher_hidden_model = keras.models.Model(
        inputs=teacher_model.inputs,
        outputs=dec_hidden_seq,
        name="teacher_hidden_seq2seq",
    )
    return teacher_hidden_model


# ==============================================================================
# build_float_student — phase 1 standard Keras GRU student.
# Same topology and same layer names as final QKeras student so that
# transfer_float_to_phase2 can unpack by name.
# ==============================================================================

def build_float_student(seq_len, n_out, student_units):
    enc_inputs = keras.layers.Input(shape=(None, 1), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(None, 1), name="sdec_input")

    enc_out, enc_state = keras.layers.GRU(
        units=student_units,
        return_state=True,
        reset_after=True,
        name="sencgru",
    )(enc_inputs)

    dec_hid_seq, _ = keras.layers.GRU(
        units=student_units,
        return_sequences=True,
        return_state=True,
        reset_after=True,
        name="sdecgru",
    )(dec_inputs, initial_state=enc_state)

    s_output = keras.layers.Dense(
        n_out, activation="linear", name="sdec_dense"
    )(dec_hid_seq)

    return keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="float_student_memoq",
    )


# ==============================================================================
# find_data_files — identical to vanilla_kd
# ==============================================================================


# ==============================================================================
# main
# ==============================================================================

def main():
    args = parse_args()

    # ── OOM guard: cap prefetch so pipeline never stages more than 128k samples ──
    MAX_PREFETCH_RAM_SAMPLES = 131_072
    safe_prefetch = max(1, MAX_PREFETCH_RAM_SAMPLES // args.batch_size)
    if args.prefetch_batches > safe_prefetch:
        print(
            f"[OOM GUARD] Capping prefetch_batches {args.prefetch_batches} -> {safe_prefetch} "
            f"(batch_size={args.batch_size})",
            flush=True,
        )
        args.prefetch_batches = safe_prefetch

    pf = lambda msg: print(msg, flush=True)

    job_name = make_job_name(args)
    job_dir  = os.path.join(args.save_dir, "results", job_name)
    os.makedirs(job_dir, exist_ok=True)
    pf(f"[MAIN] Job dir: {job_dir}")

    with open(os.path.join(job_dir, "student_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    strategy = setup_gpus_and_strategy(args.mixed_precision)

    # ── Load data ─────────────────────────────────────────────────────────────
    pf("[MAIN] Discovering data files...")
    file_input, file_res, file_labels, file_train, file_val, file_test = find_data_files(
        args.data_dir, args.seq_len
    )
    pf(f"  enc_input : {file_input}")
    pf(f"  res       : {file_res}")
    pf(f"  labels    : {file_labels}")

    raw_input  = np.load(file_input,  mmap_mode="r")
    res_data   = np.load(file_res,    mmap_mode="r")
    train_idx  = np.load(file_train)
    val_idx    = np.load(file_val)
    test_idx   = np.load(file_test)

    n_total = raw_input.shape[0]
    pf(f"[MAIN] n_total={n_total}  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ── Normalise encoder input ───────────────────────────────────────────────
    pf("[MAIN] Normalising encoder input...")
    inp_mean = float(np.mean(raw_input[train_idx]))
    inp_std  = float(np.std(raw_input[train_idx]))
    inp_std  = max(inp_std, 1e-6)
    pf(f"  mean={inp_mean:.6f}  std={inp_std:.6f}")
    normalized_input = (raw_input - inp_mean) / inp_std

    # ── Build teacher ─────────────────────────────────────────────────────────
    pf("[MAIN] Building teacher model...")
    with strategy.scope():
        teacher_model = build_teacher(
            args.seq_len, args.n_out, args.teacher_units, args.teacher_layers
        )
    teacher_model.load_weights(args.teacher_ckpt)
    pf(f"[MAIN] Teacher weights loaded from {args.teacher_ckpt}")
    teacher_hidden_model = build_teacher_hidden_model(teacher_model)

    # ── Cache teacher predictions only (NOT hidden — hidden computed live) ────
    # teacher_hidden is NOT materialised here. It is computed on-the-fly inside
    # make_dist_memoq_train / make_dist_memoq_val via teacher_hidden_model.
    # This eliminates the (N, T, teacher_units) float32 RAM footprint entirely.
    teacher_predictions, _ = cache_teacher_predictions_and_hidden(
        teacher_model        = teacher_model,
        teacher_hidden_model = teacher_hidden_model,
        normalized_input     = normalized_input,
        seq_len              = args.seq_len,
        n_out                = args.n_out,
        teacher_units        = args.teacher_units,
        n_samples            = n_total,
        infer_batch          = args.infer_batch,
        data_dir             = args.data_dir,
        pf                   = pf,
    )
    # teacher_predictions is a read-only memmap — shape (N, T, n_out), float32

    # ── Materialise enc/tgt/tpred splits — NO teacher hidden ─────────────────
    # materialise_memoq_buffers does NOT include teacher_hidden.
    # The hidden is computed live in the train/val step from teacher_hidden_model.
    pf("[MAIN] Materialising train buffers...")
    enc_tr, tgt_tr, tpred_tr = materialise_memoq_buffers(
        normalized_input    = normalized_input,
        res                 = res_data,
        teacher_predictions = teacher_predictions,
        idx                 = train_idx,
        seq_len             = args.seq_len,
        n_out               = args.n_out,
        label               = "train",
        pf                  = pf,
    )

    pf("[MAIN] Materialising val buffers...")
    enc_va, tgt_va, tpred_va = materialise_memoq_buffers(
        normalized_input    = normalized_input,
        res                 = res_data,
        teacher_predictions = teacher_predictions,
        idx                 = val_idx,
        seq_len             = args.seq_len,
        n_out               = args.n_out,
        label               = "val",
        pf                  = pf,
    )

    # ── Channel scales from teacher train predictions ─────────────────────────
    pf("[MAIN] Computing channel scales from teacher train predictions...")
    channel_scales_np = np.array(
        [max(float(np.std(tpred_tr[:, :, c])), 1e-3) for c in range(args.n_out)],
        dtype=np.float32,
    )
    pf(f"  channel_scales={channel_scales_np}")
    channel_scales = tf.constant(channel_scales_np, dtype=tf.float32)

    # ── Epsilon innov — computed from teacher hidden model on 50k subset ──────
    # enc_tr is already materialised in RAM so this is a fast inference pass.
    epsilon_innov = compute_epsilon_innov_from_model(
        teacher_hidden_model = teacher_hidden_model,
        enc_train            = enc_tr,
        seq_len              = args.seq_len,
        infer_batch          = args.infer_batch,
        pf                   = pf,
    )

    teacher_hidden_dim = args.teacher_units
    pf(f"[MAIN] teacher_hidden_dim={teacher_hidden_dim}")

    # ── Build tf.data datasets — teacher_hidden NOT in dataset ────────────────
    # make_memoq_dataset does not include teacher_hidden in the batch.
    # The live teacher_hidden_model is called inside each train/val step.
    train_batch = args.batch_size
    val_batch   = args.batch_size

    train_ds = make_memoq_dataset(
        enc_arr          = enc_tr,
        tgt_arr          = tgt_tr,
        tpred_arr        = tpred_tr,
        batch_size       = train_batch,
        seq_len          = args.seq_len,
        n_out            = args.n_out,
        shuffle          = True,
        seed             = args.split_seed,
        prefetch_batches = args.prefetch_batches,
        pipeline_workers = args.pipeline_workers,
    )
    val_ds = make_memoq_dataset(
        enc_arr          = enc_va,
        tgt_arr          = tgt_va,
        tpred_arr        = tpred_va,
        batch_size       = val_batch,
        seq_len          = args.seq_len,
        n_out            = args.n_out,
        shuffle          = False,
        seed             = 0,
        prefetch_batches = args.prefetch_batches,
        pipeline_workers = args.pipeline_workers,
    )

    train_steps = max(1, len(train_idx) // train_batch)
    val_steps   = max(1, len(val_idx)   // val_batch)
    pf(f"[MAIN] train_steps={train_steps}  val_steps={val_steps}")

    dist_train_dataset = strategy.experimental_distribute_dataset(train_ds)
    dist_val_dataset   = strategy.experimental_distribute_dataset(val_ds)

    # ── Build models ──────────────────────────────────────────────────────────
    pf("[MAIN] Building float student (Phase 1)...")
    with strategy.scope():
        float_student = build_float_student(args.seq_len, args.n_out, args.student_units)
    float_student.summary(print_fn=pf)

    pf("[MAIN] Building Phase 2 split-gate model...")
    with strategy.scope():
        phase2_model, enc_cell_p2, dec_cell_p2 = build_phase2_model(
            seq_len       = args.seq_len,
            n_out         = args.n_out,
            student_units = args.student_units,
            input_dim     = 1,
        )
    # Attach the live teacher hidden model so make_dist_memoq_train can call it
    phase2_model._teacher_hidden_model = teacher_hidden_model
    phase2_model.summary(print_fn=pf)

    pf("[MAIN] Building final hard QKeras student (Phase 3)...")
    with strategy.scope():
        final_qkeras_student = build_final_qkeras_student(
            seq_len         = args.seq_len,
            n_out           = args.n_out,
            student_units   = args.student_units,
            bits_kernel     = args.bits_kernel,
            bits_recurrent  = args.bits_recurrent,
            bits_bias       = args.bits_bias,
            bits_activation = args.bits_activation,
            bits_state      = args.bits_state,
        )
    # Attach live teacher hidden model for Phase 3 loss computation
    final_qkeras_student._teacher_hidden_model = teacher_hidden_model
    final_qkeras_student.summary(print_fn=pf)

    # ── Pre-materialise test split so _eval_fn closure can use it at any phase ─
    pf("[MAIN] Materialising test buffers for per-phase evaluation...")
    enc_te, tgt_te, _ = materialise_memoq_buffers(
        normalized_input    = normalized_input,
        res                 = res_data,
        teacher_predictions = teacher_predictions,
        idx                 = test_idx,
        seq_len             = args.seq_len,
        n_out               = args.n_out,
        label               = "test",
        pf                  = pf,
    )

    labels_data  = np.load(file_labels, mmap_mode="r")
    labels_test  = labels_data[test_idx].astype(np.float32)

    # ── Run training ──────────────────────────────────────────────────────────
    pf("[MAIN] Starting MemoQ training loop...")

    def _eval_fn(phase_tag):
            pf(f"[EVAL] _eval_fn called with phase_tag={phase_tag!r}")
            sys.stdout.flush()

            # Route each phase to the model that actually holds its trained weights:
            #   P1            -> float_student     (real float predictions)
            #   P2A/P2B/P2C   -> phase2_model      (real split-gate predictions)
            #   P3 / final    -> final_qkeras_student (real hard 4-bit predictions)
            # This is why P1/P2 scatters were previously blank: the old code
            # evaluated final_qkeras_student, which holds random weights until the
            # split-gate->QKeras transfer fires at the start of P3. Evaluating the
            # correct per-phase model produces real, non-noise scatters for every
            # phase, all in the vanilla hexbin style.
            if phase_tag == "P1":
                model_for_eval = float_student
            elif phase_tag in ("P2A", "P2B", "P2C"):
                model_for_eval = phase2_model
            elif phase_tag in ("P3", "final"):
                model_for_eval = final_qkeras_student
            else:
                pf(f"[EVAL] Unknown phase_tag={phase_tag!r} — skipping evaluation.")
                sys.stdout.flush()
                return

            evaluate_and_save(
                model         = model_for_eval,
                enc_test      = enc_te,
                tgt_test      = tgt_te,
                labels_test   = labels_test,
                gate_width_ns = args.gate_width_ns,
                n_out         = args.n_out,
                seq_len       = args.seq_len,
                infer_batch   = args.infer_batch,
                job_dir       = job_dir,
                args          = args,
                pf            = pf,
                phase_tag     = phase_tag,
            )


    history, best_val = training_loop_memoq(
        strategy             = strategy,
        float_student        = float_student,
        final_qkeras_student = final_qkeras_student,
        enc_cell_p2          = enc_cell_p2,
        dec_cell_p2          = dec_cell_p2,
        phase2_model         = phase2_model,
        args                 = args,
        dist_train_dataset   = dist_train_dataset,
        dist_val_dataset     = dist_val_dataset,
        train_steps          = train_steps,
        val_steps            = val_steps,
        channel_scales       = channel_scales,
        epsilon_innov        = epsilon_innov,
        job_dir              = job_dir,
        pf                   = pf,
        teacher_hidden_model = teacher_hidden_model,
        evaluate_fn          = _eval_fn,
    )

    # ── Transfer fidelity check ───────────────────────────────────────────────
    # Run AFTER training_loop_memoq returns so enc_va is in scope.
    # Compares phase2_model output vs final_qkeras_student on 256 val samples.
    # This fires unconditionally so you always get the diff in the log.
    # A large diff with live quantizers is expected (the cliff).
    # If the diff is large even with float/identity quantizers on the QKeras
    # model then it is a packing bug — check recurrent_activation="sigmoid"
    # in build_final_qkeras_student.
    pf("[TRANSFER CHECK] Comparing phase2_model vs final_qkeras_student on 256 val samples...")
    try:
        _xb = enc_va[:256]
        _db = np.zeros((256, args.seq_len, 1), dtype=np.float32)
        _p2_out  = phase2_model([_xb, _db], training=False)
        _p2_pred = _p2_out[0].numpy() if isinstance(_p2_out, (list, tuple)) else _p2_out.numpy()
        _q3_pred = final_qkeras_student([_xb, _db], training=False).numpy()
        _diff    = float(np.abs(_p2_pred - _q3_pred).mean())
        pf(
            f"[TRANSFER CHECK] mean |p2 - q3| = {_diff:.6f}  "
            f"(near 0 = packing faithful, large = packing bug OR quantisation cliff)"
        )
    except Exception as _te:
        pf(f"[TRANSFER CHECK] Failed (non-fatal): {_te}")
    sys.stdout.flush()
    # ── End transfer fidelity check ───────────────────────────────────────────

    # ── Save final weights ────────────────────────────────────────────────────
    final_path = os.path.join(job_dir, "student_final.weights.h5")
    final_qkeras_student.save_weights(final_path)
    pf(f"[MAIN] Final student weights saved: {final_path}")

    # ── Save history plot ─────────────────────────────────────────────────────
    save_loss_curves_memoq(history, best_val, args, job_dir, pf)

    # ── Test evaluation ───────────────────────────────────────────────────────
    # enc_te, tgt_te, labels_test already materialised above before training loop.
    evaluate_and_save(
        model         = final_qkeras_student,
        enc_test      = enc_te,
        tgt_test      = tgt_te,
        labels_test   = labels_test,
        gate_width_ns = args.gate_width_ns,
        n_out         = args.n_out,
        seq_len       = args.seq_len,
        infer_batch   = args.infer_batch,
        job_dir       = job_dir,
        args          = args,
        pf            = pf,
        phase_tag     = "final",
    )

    pf(f"[MAIN] best_val={best_val:.6f}")
    pf("[MAIN] Done.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()