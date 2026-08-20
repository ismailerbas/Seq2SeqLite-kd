#!/usr/bin/env python3
"""
train_student_memoq_full.py  —  MemoQ: Memory-Preserving 4-bit Recurrent
Quantization for GRU-based Sequence-to-Sequence Knowledge Distillation.

CONTROLLED-EXPERIMENT DESIGN:
  The final exported model is the SAME inference graph and quantization config
  as the vanilla 4-bit KD student. MemoQ differs ONLY in the training path, so
  any improvement is attributable to memory-preserving training, not to a
  different architecture, head, or quantizer family. Use --quantizer-alpha 1.0
  and --output-loss mse for the vanilla-identical control.

ROOT CAUSE TARGETED:
  Small 4-bit GRUs fail through two coupled recurrent mechanisms:
  (1) Memory-gate saturation: simultaneous 4-bit quantization of z/r/h gates
      destabilises the update gate.
  (2) Recurrent state-error accumulation: per-step quantisation errors
      accumulate through the recurrence before output KD can recover memory.

PHASES:
  Phase 1  — float warm-up. Loss obeys --output-loss (mse | huber_cn),
             blended teacher/target by --alpha. LR = effective_lr * --memoq-lr-mult-p1.
  Phase 2  — gate-decoupled hardening (2A/2B/2C). When --memoq-gate-curriculum
             True, gates harden causally h -> r -> z (kernel, recurrent, bias,
             then state+activation once all gates are 4-bit). When False, all
             gates harden at 2A (control rung B). QDense head is quantized from
             Phase 2 onward. Per-phase, per-loss lambdas via CLI.
  Phase 3  — hard 4-bit QKeras polish. LR = max(effective_lr * --memoq-lr-mult-p3,
             --memoq-phase3-lr-floor). Per-phase lambdas via the *-p3 args.

AUXILIARY LOSSES (all per-phase args, default 0.0 => disabled):
  L_mem    — lagged temporal memory-kernel distillation
  L_innov  — temporal innovation-profile matching (clipped log-ratio)
  L_zsat   — update-gate saturation barrier (gate-value form; hidden-rail
             proxy in Phase 3)
  L_rail   — hidden-state rail-margin regulariser
  L_shape  — amplitude-decoupled decay-shape match (t0-normalised curve)

EXPERIMENTAL LADDER (no code edits — CLI only):
  A/B control : --output-loss mse --quantizer-alpha 1.0
                --memoq-gate-curriculum false   (all aux lambdas 0.0)
  C gate      : --memoq-gate-curriculum true
  D +memory   : add --memoq-lambda-mem-p2a/p2b/p2c/p3
  E +innov    : add --memoq-lambda-innov-*
  F full      : add small --memoq-lambda-zsat-* / -rail-* / -shape-*

DIAGNOSTICS:
  [EQUIV P1->P2] : float student vs split-gate cell with quantizers off (~<1e-4).
  [EQUIV P2C->P3]: phase2 model vs packed QKeras student (small => faithful export).
  [CONFIG]       : instantiated QGRU/QDense configs for vanilla-identity check.

OUTPUTS (inside --save-dir / results / job_name /):
  phase1_best.weights.h5  stage2a_best.weights.h5  stage2b_best.weights.h5
  stage2c_best.weights.h5  student_best.weights.h5  student_final.weights.h5
  student_args.json  training_history.csv  training_history.png
  test_metrics.json  test_scatter_tau1.png  test_scatter_tau2.png
  test_scatter_fret.png  test_residuals.png
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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

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
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
                                  alpha, channel_scales, huber_delta, output_loss):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    with tf.GradientTape() as tape:
        s_out = model([enc_b, dec_b], training=True)
        l_seq = output_loss_fn(tgt_b,   s_out, channel_scales, huber_delta, output_loss)
        l_kd  = output_loss_fn(tpred_b, s_out, channel_scales, huber_delta, output_loss)
        total = (1.0 - alpha) * l_seq + alpha * l_kd

    grads = tape.gradient(total, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    nan_in_grads = tf.reduce_any(tf.stack([
        tf.reduce_any(tf.math.is_nan(g)) | tf.reduce_any(tf.math.is_inf(g))
        for g in grads
    ]))
    # Skip the update entirely on a bad batch (Fix 3): zero the grads so
    # apply_gradients is a no-op instead of writing NaN/Inf into the weights.
    safe_grads = [
        tf.where(nan_in_grads, tf.zeros_like(g), g) for g in grads
    ]
    safe_grads, _ = tf.clip_by_global_norm(safe_grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(safe_grads, model.trainable_variables))

    zero = tf.constant(0.0, dtype=tf.float32)
    # 9-tuple contract: (total, seq, kd, mem, innov, zsat, rail, shape, nan)
    return total, l_seq, l_kd, zero, zero, zero, zero, zero, tf.cast(nan_in_grads, tf.float32)

def val_step_phase1_per_replica(batch_x, batch_y, model, alpha, channel_scales, huber_delta, output_loss):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    s_out = model([enc_b, dec_b], training=False)
    l_seq = output_loss_fn(tgt_b,   s_out, channel_scales, huber_delta, output_loss)
    l_kd  = output_loss_fn(tpred_b, s_out, channel_scales, huber_delta, output_loss)
    total = (1.0 - alpha) * l_seq + alpha * l_kd
    mae   = tf.reduce_mean(tf.abs(s_out - tgt_b))

    zero = tf.constant(0.0, dtype=tf.float32)
    # 9-tuple contract: (total, seq, kd, mem, innov, zsat, rail, shape, mae)
    return total, l_seq, l_kd, zero, zero, zero, zero, zero, mae

def make_dist_phase1_train(strategy, model, optimizer, alpha, channel_scales, huber_delta, output_loss):
    @tf.function
    def step(bx, by):
        pr = strategy.run(train_step_phase1_per_replica,
                          args=(bx, by, model, optimizer, alpha, channel_scales, huber_delta, output_loss))
        # pr = (total, seq, kd, mem, innov, zsat, rail, shape, nan)
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[3], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[4], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[5], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[6], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[7], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM,  pr[8], axis=None),
        )
    return step

def make_dist_phase1_val(strategy, model, alpha, channel_scales, huber_delta, output_loss):
    @tf.function
    def step(bx, by):
        pr = strategy.run(val_step_phase1_per_replica,
                          args=(bx, by, model, alpha, channel_scales, huber_delta, output_loss))
        # pr = (total, seq, kd, mem, innov, zsat, rail, shape, mae)
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[i], axis=None)
            for i in range(9)
        )
    return step

def set_phase2_quantizers(args, enc_cell, dec_cell, stage):
    """
    Assign quantizers to MemoQGRUCell gate variables based on the curriculum
    stage AND the controlled-experiment quantizer family (args.q_alpha).

    Stage order (curriculum=True):
      P2A : h kernel + recurrent + bias only
      P2B : h + r kernel + recurrent + bias
      P2C : h + r + z kernel + recurrent  (biases still float)
      P2D : h + r + z kernel + recurrent + ALL biases
      P2E : h + r + z kernel + recurrent + ALL biases + activation quantizer
             activation_blend_beta and act_dither_delta are set to their
             STARTING values (beta=0, dither=DELTA_A*0.5) here so the training
             loop can begin the cosine anneal from epoch 0 of P2E.
      P2F : all of above + state quantizer.
             Both activation_blend_beta and state_blend_beta are reset to 0.0
             here so the merged anneal in the training loop starts from 0 for
             BOTH quantizers simultaneously (Fix 2 — joint anneal).
             state_dither_delta is set to DELTA_S*0.5 at anneal start.
      P3  : identical quantizer assignment to P2F. beta and dither are
             enforced to 1.0 / 0.0 externally by the training loop hard tail.

    curriculum=False collapses P2A/P2B/P2C/P2D/P2E/P2F all into P2A
    (all quantizers on immediately — control run).

    Fix 1: LSB noise (old Gaussian) is REPLACED with subtractive uniform dither.
    DELTA_S = DELTA_A = 0.125 for 4-bit. Half-width = 0.0625.
    The Gaussian std 1/(2^bits-1) = 0.0667 was 1.85x the physical rounding std
    and was unbounded. The new dither is bounded in [-DELTA, DELTA] (triangular
    sum), correct scale, and is applied subtractively inside call() so it
    linearises the dead-zone without biasing the expected value.

    Fix 2: activation_blend_beta mirrors state_blend_beta. Both are set here
    to their starting values. The training loop ramps both together with a
    single cosine schedule in the merged P2E+P2F anneal.

    Dither is tapered to 0.0 over the last 10% of the anneal window by the
    training loop. This function sets the INITIAL value at anneal start.
    In P3 / hard tail both deltas are set to 0.0 (exact inference).
    """
    q4k  = quantized_bits(args.bits_kernel,    0, 1, alpha=args.q_alpha)
    q4r  = quantized_bits(args.bits_recurrent, 0, 1, alpha=args.q_alpha)
    q4b  = quantized_bits(args.bits_bias,      0, 1, alpha=args.q_alpha)
    q4s  = quantized_bits(args.bits_state,     0, 1, alpha=1.0)
    q4a  = quantized_tanh(bits=args.bits_activation, symmetric=True)

    # Fix 1: correct physical half-LSB for 4-bit uniform quantizer.
    # Replaces the old 1/(2^bits_state - 1) = 0.0667 Gaussian std.
    DELTA_S = 2.0 ** (-(args.bits_state - 1))       # 0.125 for 4-bit
    DELTA_A = 2.0 ** (-(args.bits_activation - 1))  # 0.125 for 4-bit
    dither_half_s = DELTA_S * 0.5                   # 0.0625
    dither_half_a = DELTA_A * 0.5                   # 0.0625

    curriculum = args.memoq_gate_curriculum

    if not curriculum:
        q_h          = q4k
        q_r          = q4k
        q_z          = q4k
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = q4r
        q_bias       = q4b
        q_a          = q4a
        q_s          = q4s
        # No anneal in non-curriculum control run: hard quantizers immediately.
        # Set betas to 1.0 and dithers to 0.0 so call() is the hard path.
        act_beta     = 1.0
        state_beta   = 1.0
        act_dither   = 0.0
        state_dither = 0.0
    elif stage == "P2A":
        q_h          = q4k
        q_r          = None
        q_z          = None
        rq_h         = q4r
        rq_r         = None
        rq_z         = None
        q_bias       = None
        q_a          = None
        q_s          = None
        act_beta     = 1.0
        state_beta   = 1.0
        act_dither   = 0.0
        # Fix 1: correct bounded uniform dither on state during P2A.
        # Old code: state_lsb_noise_std = 1/(2^bits-1) = 0.0667 Gaussian.
        # New code: subtractive uniform half-width = DELTA_S * 0.5 = 0.0625.
        state_dither = dither_half_s
    elif stage == "P2B":
        q_h          = q4k
        q_r          = q4k
        q_z          = None
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = None
        q_bias       = None
        q_a          = None
        q_s          = None
        act_beta     = 1.0
        state_beta   = 1.0
        act_dither   = 0.0
        state_dither = dither_half_s
    elif stage == "P2C":
        q_h          = q4k
        q_r          = q4k
        q_z          = q4k
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = q4r
        q_bias       = None
        q_a          = None
        q_s          = None
        act_beta     = 1.0
        state_beta   = 1.0
        act_dither   = 0.0
        state_dither = dither_half_s
    elif stage == "P2D":
        q_h          = q4k
        q_r          = q4k
        q_z          = q4k
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = q4r
        q_bias       = q4b
        q_a          = None
        q_s          = None
        act_beta     = 1.0
        state_beta   = 1.0
        act_dither   = 0.0
        state_dither = dither_half_s
    elif stage == "P2E":
        q_h          = q4k
        q_r          = q4k
        q_z          = q4k
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = q4r
        q_bias       = q4b
        q_a          = q4a
        q_s          = None
        # Fix 0+2: activation quantizer is introduced with beta=0 (fully float)
        # and dither active. The training loop ramps beta 0->1 with cosine
        # schedule over memoq_state_anneal_epochs epochs. State still float.
        act_beta     = 0.0
        state_beta   = 1.0
        act_dither   = dither_half_a
        state_dither = dither_half_s
    elif stage in ("P2F", "P3"):
        q_h          = q4k
        q_r          = q4k
        q_z          = q4k
        rq_h         = q4r
        rq_r         = q4r
        rq_z         = q4r
        q_bias       = q4b
        q_a          = q4a
        q_s          = q4s
        # Fix 2: merged joint anneal. Both activation_blend_beta AND
        # state_blend_beta start at 0.0 here. The training loop drives
        # both to 1.0 together with a single cosine schedule.
        # Both dithers are active at the start; the loop tapers them to 0.0
        # before the hard tail. In P3 the loop enforces beta=1, dither=0.
        act_beta     = 0.0
        state_beta   = 0.0
        act_dither   = dither_half_a
        state_dither = dither_half_s
    else:
        raise ValueError(
            f"Unknown stage: {stage!r}. "
            f"Expected one of P2A, P2B, P2C, P2D, P2E, P2F, P3."
        )

    for cell in [enc_cell, dec_cell]:
        cell.quantizer_h              = q_h
        cell.quantizer_r              = q_r
        cell.quantizer_z              = q_z
        cell.quantizer_recurrent_h    = rq_h
        cell.quantizer_recurrent_r    = rq_r
        cell.quantizer_recurrent_z    = rq_z
        cell.quantizer_state          = q_s
        cell.quantizer_activation     = q_a
        cell.quantizer_bias           = q_bias
        cell.activation_blend_beta    = act_beta
        cell.state_blend_beta         = state_beta
        cell.act_dither_delta         = act_dither
        cell.state_dither_delta       = state_dither
        # Legacy field kept for external code that still writes it. No-op
        # inside call() — the new dither fields are what call() reads.
        cell.state_lsb_noise_std      = 0.0
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
    equiv_enc_sample=None,
):
    p1_ckpt         = os.path.join(job_dir, "phase1_best.weights.h5")
    p2a_ckpt        = os.path.join(job_dir, "stage2a_best.weights.h5")
    p2b_ckpt        = os.path.join(job_dir, "stage2b_best.weights.h5")
    p2c_ckpt        = os.path.join(job_dir, "stage2c_best.weights.h5")
    p2d_ckpt        = os.path.join(job_dir, "stage2d_best.weights.h5")
    p2e_ckpt        = os.path.join(job_dir, "stage2e_best.weights.h5")
    p2f_ckpt        = os.path.join(job_dir, "stage2f_best.weights.h5")
    p3_ckpt         = os.path.join(job_dir, "student_best.weights.h5")
    resume_path     = os.path.join(job_dir, "resume_state.json")
    completion_path = os.path.join(job_dir, "training_complete.flag")

    history = {
        "total":     [], "seq":       [], "kd":        [],
        "mem":       [], "innov":     [], "zsat":      [], "rail":      [], "shape": [],
        "val_total": [], "val_seq":   [], "val_kd":    [],
        "val_mem":   [], "val_innov": [], "val_zsat":  [], "val_rail":  [], "val_shape": [],
        "val_mae":   [], "phase":     [],
    }

    total_planned = (
        args.memoq_warmup_epochs
        + args.memoq_stage2a_epochs
        + args.memoq_stage2b_epochs
        + args.memoq_stage2c_epochs
        + args.memoq_stage2d_epochs
        + args.memoq_stage2e_epochs
        + args.memoq_stage2f_epochs
        + args.memoq_stage3_epochs
    )
    global_epoch = 0
    nan_warn_threshold = max(1, int(train_steps * 0.10))

    resume_stage = "P1"
    resume_epoch_in_stage = 0
    best_vals = {
        "P1": float("inf"), "P2A": float("inf"), "P2B": float("inf"),
        "P2C": float("inf"), "P2D": float("inf"), "P2E": float("inf"),
        "P2F": float("inf"), "P3": float("inf"),
    }
    patience_cts = {
        "P1": 0, "P2A": 0, "P2B": 0, "P2C": 0,
        "P2D": 0, "P2E": 0, "P2F": 0, "P3": 0,
    }

    stage_order_list = ["P1", "P2A", "P2B", "P2C", "P2D", "P2E", "P2F", "P3"]
    max_epochs_per_stage = {
        "P1":  args.memoq_warmup_epochs,
        "P2A": args.memoq_stage2a_epochs,
        "P2B": args.memoq_stage2b_epochs,
        "P2C": args.memoq_stage2c_epochs,
        "P2D": args.memoq_stage2d_epochs,
        "P2E": args.memoq_stage2e_epochs,
        "P2F": args.memoq_stage2f_epochs,
        "P3":  args.memoq_stage3_epochs,
    }

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
                for _k in ["P3", "P2F", "P2E", "P2D", "P2C", "P2B", "P2A", "P1"]:
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

        if os.path.exists(p1_ckpt):
            float_student.load_weights(p1_ckpt)
            pf("[RESUME] Loaded P1 weights for float_student")

        # Load the checkpoint for the stage we are actually resuming into.
        _p2_stage_ckpts = {
            "P2A": p2a_ckpt,
            "P2B": p2b_ckpt,
            "P2C": p2c_ckpt,
            "P2D": p2d_ckpt,
            "P2E": p2e_ckpt,
            "P2F": p2f_ckpt,
        }
        _p2_fallback_order = ["P2A", "P2B", "P2C", "P2D", "P2E", "P2F"]

        if resume_stage in _p2_stage_ckpts:
            target_ckpt = _p2_stage_ckpts[resume_stage]
            if os.path.exists(target_ckpt):
                phase2_model.load_weights(target_ckpt)
                pf(f"[RESUME] Loaded {resume_stage} weights for phase2_model from {target_ckpt}")
            else:
                # No checkpoint yet for this stage — fall back to the most recent
                # earlier stage and restart this stage from epoch 0.
                resume_idx_in_p2 = _p2_fallback_order.index(resume_stage)
                fallback_ckpt = None
                for earlier in reversed(_p2_fallback_order[:resume_idx_in_p2]):
                    cand = _p2_stage_ckpts[earlier]
                    if os.path.exists(cand):
                        fallback_ckpt = cand
                        break
                if fallback_ckpt is not None:
                    phase2_model.load_weights(fallback_ckpt)
                    pf(f"[RESUME] {resume_stage} checkpoint missing — loaded {fallback_ckpt} and restarting {resume_stage} from epoch 0")
                    resume_epoch_in_stage = 0

        if resume_stage == "P3" and os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf("[RESUME] Loaded P3 weights for final_qkeras_student")
        sys.stdout.flush()

    csv_path = os.path.join(job_dir, "training_history.csv")
    if not args.resume or (resume_stage == "P1" and resume_epoch_in_stage == 0):
        with open(csv_path, "w") as f:
            f.write(
                "epoch,phase,total,seq,kd,mem,innov,zsat,rail,shape,"
                "val_total,val_seq,val_kd,val_mem,val_innov,val_zsat,val_rail,val_shape,"
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

        lr_p1 = args.effective_lr * args.memoq_lr_mult_p1
        opt_p1 = keras.optimizers.Adam(learning_rate=lr_p1)
        sched_p1 = ReduceLROnPlateau(
            opt_p1, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p1.reset(lr_p1)

        dist_train_p1 = make_dist_phase1_train(
            strategy, float_student, opt_p1, args.alpha, channel_scales,
            args.memoq_huber_delta, args.output_loss
        )
        dist_val_p1 = make_dist_phase1_val(
            strategy, float_student, args.alpha, channel_scales,
            args.memoq_huber_delta, args.output_loss
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
        if equiv_enc_sample is not None:
            run_equivalence_checks(
                float_student, phase2_model, enc_cell_p2, dec_cell_p2,
                final_qkeras_student, equiv_enc_sample, args.seq_len, pf, "P1->P2",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2A — Candidate gate (h) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2A"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2A — Candidate gate h 4-bit ({args.memoq_stage2a_epochs} epochs)")
        pf("  Curriculum=h-only (z,r float) when --memoq-gate-curriculum True; "
           "all gates 4-bit when False.")
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2A")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2a = args.effective_lr * args.memoq_lr_mult_p2a
        opt_p2a = keras.optimizers.Adam(learning_rate=lr_p2a)
        sched_p2a = ReduceLROnPlateau(
            opt_p2a, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2a.reset(lr_p2a)

        ep2a_start = start_ep("P2A")
        innov_active = ep2a_start >= args.memoq_innov_burnin

        def _build_p2a(innov_on):
            lambda_i_2a = args.memoq_lambda_innov_p2a if innov_on else 0.0
            train_fn = make_dist_memoq_train(
                strategy, phase2_model, opt_p2a,
                args.alpha, channel_scales, args.memoq_huber_delta,
                args.memoq_lambda_mem_p2a, lambda_i_2a,
                args.memoq_lambda_zsat_p2a, args.memoq_lambda_rail_p2a,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=(args.memoq_lambda_mem_p2a > 0.0),
                use_innov=(innov_on and args.memoq_lambda_innov_p2a > 0.0),
                use_zsat=(args.memoq_lambda_zsat_p2a > 0.0),
                use_rail=(args.memoq_lambda_rail_p2a > 0.0),
                has_z_logit=True, clipnorm=1.0,
                teacher_hidden_model=teacher_hidden_model,
                lambda_s=args.memoq_lambda_shape_p2a,
                rho_z=args.memoq_rho_z,
                output_loss=args.output_loss,
            )
            val_fn = make_dist_memoq_val(
                strategy, phase2_model,
                args.alpha, channel_scales, args.memoq_huber_delta,
                args.memoq_lambda_mem_p2a, lambda_i_2a,
                args.memoq_lambda_zsat_p2a, args.memoq_lambda_rail_p2a,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=(args.memoq_lambda_mem_p2a > 0.0),
                use_innov=(innov_on and args.memoq_lambda_innov_p2a > 0.0),
                use_zsat=(args.memoq_lambda_zsat_p2a > 0.0),
                use_rail=(args.memoq_lambda_rail_p2a > 0.0),
                has_z_logit=True,
                teacher_hidden_model=teacher_hidden_model,
                lambda_s=args.memoq_lambda_shape_p2a,
                rho_z=args.memoq_rho_z,
                output_loss=args.output_loss,
            )
            return train_fn, val_fn

        dist_train_p2a, dist_val_p2a = _build_p2a(innov_active)

        for ep_in_phase in range(ep2a_start, args.memoq_stage2a_epochs):
            if ep_in_phase == args.memoq_innov_burnin and not innov_active:
                innov_active = True
                dist_train_p2a, dist_val_p2a = _build_p2a(True)

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
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2B")
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2B — Reset gate r 4-bit ({args.memoq_stage2b_epochs} epochs)")
        pf("  Quantising W_r, U_r. z gate float. h gate already 4-bit.")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2b = args.effective_lr * args.memoq_lr_mult_p2b
        opt_p2b = keras.optimizers.Adam(learning_rate=lr_p2b)
        sched_p2b = ReduceLROnPlateau(
            opt_p2b, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2b.reset(lr_p2b)

        ep2b_start = start_ep("P2B")

        dist_train_p2b = make_dist_memoq_train(
            strategy, phase2_model, opt_p2b,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2b, args.memoq_lambda_innov_p2b,
            args.memoq_lambda_zsat_p2b, args.memoq_lambda_rail_p2b,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2b > 0.0), use_innov=(args.memoq_lambda_innov_p2b > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2b > 0.0), use_rail=(args.memoq_lambda_rail_p2b > 0.0),
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2b,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )
        dist_val_p2b = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2b, args.memoq_lambda_innov_p2b,
            args.memoq_lambda_zsat_p2b, args.memoq_lambda_rail_p2b,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2b > 0.0), use_innov=(args.memoq_lambda_innov_p2b > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2b > 0.0), use_rail=(args.memoq_lambda_rail_p2b > 0.0),
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2b,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
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
    # PHASE 2C — Update gate (z) kernel + recurrent 4-bit. Biases still float.
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2C"):
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2C")
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2C — Update gate z kernel+recurrent 4-bit ({args.memoq_stage2c_epochs} epochs)")
        pf("  All gate kernels now 4-bit. Biases, activation, state still float.")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2c = args.effective_lr * args.memoq_lr_mult_p2c
        opt_p2c = keras.optimizers.Adam(learning_rate=lr_p2c)
        sched_p2c = ReduceLROnPlateau(
            opt_p2c, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2c.reset(lr_p2c)

        ep2c_start = start_ep("P2C")

        dist_train_p2c = make_dist_memoq_train(
            strategy, phase2_model, opt_p2c,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2c, args.memoq_lambda_innov_p2c,
            args.memoq_lambda_zsat_p2c, args.memoq_lambda_rail_p2c,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2c > 0.0), use_innov=(args.memoq_lambda_innov_p2c > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2c > 0.0), use_rail=(args.memoq_lambda_rail_p2c > 0.0),
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2c,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )
        dist_val_p2c = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2c, args.memoq_lambda_innov_p2c,
            args.memoq_lambda_zsat_p2c, args.memoq_lambda_rail_p2c,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2c > 0.0), use_innov=(args.memoq_lambda_innov_p2c > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2c > 0.0), use_rail=(args.memoq_lambda_rail_p2c > 0.0),
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2c,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
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
    # PHASE 2D — All biases 4-bit. Activation and state still float.
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2D"):
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2D")
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2D — All biases 4-bit ({args.memoq_stage2d_epochs} epochs)")
        pf("  All kernels + biases 4-bit. Activation and state still float.")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2d = args.effective_lr * args.memoq_lr_mult_p2d
        opt_p2d = keras.optimizers.Adam(learning_rate=lr_p2d)
        sched_p2d = ReduceLROnPlateau(
            opt_p2d, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2d.reset(lr_p2d)

        ep2d_start = start_ep("P2D")

        dist_train_p2d = make_dist_memoq_train(
            strategy, phase2_model, opt_p2d,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2d, args.memoq_lambda_innov_p2d,
            args.memoq_lambda_zsat_p2d, args.memoq_lambda_rail_p2d,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2d > 0.0), use_innov=(args.memoq_lambda_innov_p2d > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2d > 0.0), use_rail=(args.memoq_lambda_rail_p2d > 0.0),
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2d,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )
        dist_val_p2d = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2d, args.memoq_lambda_innov_p2d,
            args.memoq_lambda_zsat_p2d, args.memoq_lambda_rail_p2d,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2d > 0.0), use_innov=(args.memoq_lambda_innov_p2d > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2d > 0.0), use_rail=(args.memoq_lambda_rail_p2d > 0.0),
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2d,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )

        for ep_in_phase in range(ep2d_start, args.memoq_stage2d_epochs):
            history, best_vals["P2D"], patience_cts["P2D"], early_stop = run_epoch(
                phase_tag="P2D",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2d,
                dist_val_step_fn=dist_val_p2d,
                lr_scheduler=sched_p2d,
                effective_warmup_epochs=0,
                effective_lr=lr_p2d,
                history=history,
                best_val=best_vals["P2D"],
                patience_ct=patience_cts["P2D"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2d_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2D", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2d_ckpt):
            phase2_model.load_weights(p2d_ckpt)
            pf(f"[P2D] Loaded best weights from {p2d_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2D...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2D")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2E — Activation quantizer (quantized_tanh) with soft-blend anneal.
    # Fix 0+2: activation_blend_beta is ramped 0->1 with a cosine schedule
    # over memoq_state_anneal_epochs epochs (reusing that arg for both anneal
    # windows). act_dither_delta is tapered to 0 over the last 10% of the
    # anneal, then 0 for the hard tail. state still float in this stage.
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2E"):
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2E")
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2E — Activation quantizer soft-blend anneal ({args.memoq_stage2e_epochs} epochs)")
        pf(f"  anneal_epochs={args.memoq_state_anneal_epochs}  activation_blend_beta 0->1 cosine then hard.")
        pf("  All kernels + biases 4-bit. Activation annealed. State still float.")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2e = args.effective_lr * args.memoq_lr_mult_p2e
        opt_p2e = keras.optimizers.Adam(learning_rate=lr_p2e)
        sched_p2e = ReduceLROnPlateau(
            opt_p2e, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2e.reset(lr_p2e)

        ep2e_start = start_ep("P2E")

        dist_train_p2e = make_dist_memoq_train(
            strategy, phase2_model, opt_p2e,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2e, args.memoq_lambda_innov_p2e,
            args.memoq_lambda_zsat_p2e, args.memoq_lambda_rail_p2e,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2e > 0.0), use_innov=(args.memoq_lambda_innov_p2e > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2e > 0.0), use_rail=(args.memoq_lambda_rail_p2e > 0.0),
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2e,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )
        dist_val_p2e = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2e, args.memoq_lambda_innov_p2e,
            args.memoq_lambda_zsat_p2e, args.memoq_lambda_rail_p2e,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2e > 0.0), use_innov=(args.memoq_lambda_innov_p2e > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2e > 0.0), use_rail=(args.memoq_lambda_rail_p2e > 0.0),
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2e,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )

        anneal_epochs_2e = max(1, args.memoq_state_anneal_epochs)
        # DELTA_A for 4-bit activation quantizer half-LSB
        _DELTA_A_HALF = 2.0 ** (-(args.bits_activation - 1)) * 0.5

        for ep_in_phase in range(ep2e_start, args.memoq_stage2e_epochs):
            # Cosine anneal: beta = 0.5*(1 - cos(pi * frac))
            if ep_in_phase < anneal_epochs_2e:
                frac = float(ep_in_phase + 1) / float(anneal_epochs_2e)
                beta_a = 0.5 * (1.0 - math.cos(math.pi * frac))
            else:
                beta_a = 1.0
            # Dither taper: full dither for first 90% of anneal, linear taper
            # to 0.0 over the last 10% of the anneal window. Hard tail = 0.0.
            if ep_in_phase < anneal_epochs_2e:
                taper_start = 0.9
                frac_raw = float(ep_in_phase + 1) / float(anneal_epochs_2e)
                if frac_raw < taper_start:
                    act_dither = _DELTA_A_HALF
                else:
                    taper_frac = (frac_raw - taper_start) / (1.0 - taper_start)
                    act_dither = _DELTA_A_HALF * max(0.0, 1.0 - taper_frac)
            else:
                act_dither = 0.0
            for cell in [enc_cell_p2, dec_cell_p2]:
                cell.activation_blend_beta = beta_a
                cell.act_dither_delta      = act_dither
            pf(f"  [P2E] ep {ep_in_phase}  activation_blend_beta={beta_a:.4f}  act_dither_delta={act_dither:.5f}")
            sys.stdout.flush()

            history, best_vals["P2E"], patience_cts["P2E"], early_stop = run_epoch(
                phase_tag="P2E",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2e,
                dist_val_step_fn=dist_val_p2e,
                lr_scheduler=sched_p2e,
                effective_warmup_epochs=0,
                effective_lr=lr_p2e,
                history=history,
                best_val=best_vals["P2E"],
                patience_ct=patience_cts["P2E"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2e_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2E", ep_in_phase + 1)
            if early_stop:
                break

        # Hard tail: force beta=1, dither=0 after P2E regardless of early stop.
        for cell in [enc_cell_p2, dec_cell_p2]:
            cell.activation_blend_beta = 1.0
            cell.act_dither_delta      = 0.0

        if os.path.exists(p2e_ckpt):
            phase2_model.load_weights(p2e_ckpt)
            pf(f"[P2E] Loaded best weights from {p2e_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2E...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2E")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2F — Joint activation + state quantizer soft-blend anneal (Fix 2).
    # Both activation_blend_beta and state_blend_beta are set to 0.0 at entry
    # by set_phase2_quantizers("P2F") and ramped jointly with a single cosine
    # schedule over memoq_state_anneal_epochs epochs. Both dithers are active
    # during the anneal and tapered to 0 before the hard tail.
    # This eliminates the double-shock where P2E adapted weights to a float
    # state and P2F then invalidated that by suddenly quantizing the state.
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2F"):
        set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2F")
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2F — Joint activation+state soft-blend anneal ({args.memoq_stage2f_epochs} epochs)")
        pf(f"  anneal_epochs={args.memoq_state_anneal_epochs}  both betas 0->1 cosine jointly then hard.")
        pf(f"  innov={args.memoq_lambda_innov_p2f}  mem={args.memoq_lambda_mem_p2f}  "
           f"zsat={args.memoq_lambda_zsat_p2f}  rail={args.memoq_lambda_rail_p2f}")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p2f = args.effective_lr * args.memoq_lr_mult_p2f
        opt_p2f = keras.optimizers.Adam(learning_rate=lr_p2f)
        sched_p2f = ReduceLROnPlateau(
            opt_p2f, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2f.reset(lr_p2f)

        ep2f_start = start_ep("P2F")

        dist_train_p2f = make_dist_memoq_train(
            strategy, phase2_model, opt_p2f,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2f, args.memoq_lambda_innov_p2f,
            args.memoq_lambda_zsat_p2f, args.memoq_lambda_rail_p2f,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2f > 0.0), use_innov=(args.memoq_lambda_innov_p2f > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2f > 0.0), use_rail=(args.memoq_lambda_rail_p2f > 0.0),
            has_z_logit=True, clipnorm=1.0,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2f,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )
        dist_val_p2f = make_dist_memoq_val(
            strategy, phase2_model,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p2f, args.memoq_lambda_innov_p2f,
            args.memoq_lambda_zsat_p2f, args.memoq_lambda_rail_p2f,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            use_mem=(args.memoq_lambda_mem_p2f > 0.0), use_innov=(args.memoq_lambda_innov_p2f > 0.0),
            use_zsat=(args.memoq_lambda_zsat_p2f > 0.0), use_rail=(args.memoq_lambda_rail_p2f > 0.0),
            has_z_logit=True,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p2f,
            rho_z=args.memoq_rho_z,
            output_loss=args.output_loss,
        )

        anneal_epochs_2f = max(1, args.memoq_state_anneal_epochs)
        _DELTA_A_HALF_2F = 2.0 ** (-(args.bits_activation - 1)) * 0.5
        _DELTA_S_HALF_2F = 2.0 ** (-(args.bits_state - 1)) * 0.5

        for ep_in_phase in range(ep2f_start, args.memoq_stage2f_epochs):
            # Cosine anneal for both betas jointly.
            if ep_in_phase < anneal_epochs_2f:
                frac = float(ep_in_phase + 1) / float(anneal_epochs_2f)
                beta = 0.5 * (1.0 - math.cos(math.pi * frac))
            else:
                beta = 1.0
            # Dither taper: full for first 90% of anneal window, linear taper
            # to 0.0 over the last 10%. Hard tail = 0.0 for both.
            if ep_in_phase < anneal_epochs_2f:
                taper_start = 0.9
                frac_raw = float(ep_in_phase + 1) / float(anneal_epochs_2f)
                if frac_raw < taper_start:
                    act_dither   = _DELTA_A_HALF_2F
                    state_dither = _DELTA_S_HALF_2F
                else:
                    taper_frac   = (frac_raw - taper_start) / (1.0 - taper_start)
                    act_dither   = _DELTA_A_HALF_2F * max(0.0, 1.0 - taper_frac)
                    state_dither = _DELTA_S_HALF_2F * max(0.0, 1.0 - taper_frac)
            else:
                act_dither   = 0.0
                state_dither = 0.0
            for cell in [enc_cell_p2, dec_cell_p2]:
                cell.activation_blend_beta = beta
                cell.state_blend_beta      = beta
                cell.act_dither_delta      = act_dither
                cell.state_dither_delta    = state_dither
            pf(
                f"  [P2F] ep {ep_in_phase}  beta={beta:.4f}  "
                f"act_dither={act_dither:.5f}  state_dither={state_dither:.5f}"
            )
            sys.stdout.flush()

            history, best_vals["P2F"], patience_cts["P2F"], early_stop = run_epoch(
                phase_tag="P2F",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2f,
                dist_val_step_fn=dist_val_p2f,
                lr_scheduler=sched_p2f,
                effective_warmup_epochs=0,
                effective_lr=lr_p2f,
                history=history,
                best_val=best_vals["P2F"],
                patience_ct=patience_cts["P2F"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2f_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2F", ep_in_phase + 1)
            if early_stop:
                break

        # Hard tail: force both betas=1, both dithers=0 after P2F.
        for cell in [enc_cell_p2, dec_cell_p2]:
            cell.activation_blend_beta = 1.0
            cell.state_blend_beta      = 1.0
            cell.act_dither_delta      = 0.0
            cell.state_dither_delta    = 0.0

        if os.path.exists(p2f_ckpt):
            phase2_model.load_weights(p2f_ckpt)
            pf(f"[P2F] Loaded best weights from {p2f_ckpt}")
            sys.stdout.flush()

        if evaluate_fn is not None:
            pf("[EVAL] Running per-phase scatter evaluation after P2F...")
            sys.stdout.flush()
            evaluate_fn(phase_tag="P2F")

    # ══════════════════════════════════════════════════════════════════════════
    # Export last Phase 2 split-gate weights into final hard QKeras student.
    # Supports ablation runs where P2E and/or P2F are skipped (epochs=0).
    # Determines the last completed Phase 2 stage dynamically so the transfer
    # label and loaded checkpoint are always correct regardless of which stages
    # ran. If P2F ran -> use p2f_ckpt. Elif P2E ran -> use p2e_ckpt. Elif P2D
    # ran -> use p2d_ckpt. Falls back through P2C, P2B, P2A in order.
    # ══════════════════════════════════════════════════════════════════════════
    def _last_completed_p2_ckpt_label():
        candidates = [
            ("P2F", p2f_ckpt, args.memoq_stage2f_epochs),
            ("P2E", p2e_ckpt, args.memoq_stage2e_epochs),
            ("P2D", p2d_ckpt, args.memoq_stage2d_epochs),
            ("P2C", p2c_ckpt, args.memoq_stage2c_epochs),
            ("P2B", p2b_ckpt, args.memoq_stage2b_epochs),
            ("P2A", p2a_ckpt, args.memoq_stage2a_epochs),
        ]
        for label, ckpt, epochs in candidates:
            if epochs > 0 and os.path.exists(ckpt):
                return label, ckpt
        return None, None

    entering_p3_fresh = should_run("P3") and not (
        args.resume
        and resume_stage == "P3"
        and os.path.exists(p3_ckpt)
    )
    if entering_p3_fresh:
        last_p2_label, last_p2_ckpt = _last_completed_p2_ckpt_label()
        pf("=" * 60)
        pf(f"[EXPORT] Last completed Phase 2 stage: {last_p2_label} — packing split gate weights into standard QKeras QGRU format...")
        pf("=" * 60)
        sys.stdout.flush()
        if last_p2_ckpt is not None and not os.path.exists(last_p2_ckpt):
            pf(f"[EXPORT] WARNING: expected checkpoint {last_p2_ckpt} not found — proceeding with current in-memory weights.")
        elif last_p2_ckpt is not None:
            phase2_model.load_weights(last_p2_ckpt)
            pf(f"[EXPORT] Loaded {last_p2_label} weights from {last_p2_ckpt} for export to QKeras.")
        transfer_splitgate_to_qkeras(enc_cell_p2, dec_cell_p2, phase2_model, final_qkeras_student, pf)
        export_label = f"{last_p2_label}->P3" if last_p2_label else "P2?->P3"
        if equiv_enc_sample is not None:
            run_equivalence_checks(
                float_student, phase2_model, enc_cell_p2, dec_cell_p2,
                final_qkeras_student, equiv_enc_sample, args.seq_len, pf, export_label,
            )
    else:
        pf("[EXPORT] Skipping transfer_splitgate_to_qkeras — resuming Phase 3 from checkpoint, weights already loaded.")
        sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Hard 4-bit QKeras polish
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P3"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 3 — Hard 4-bit QKeras polish ({args.memoq_stage3_epochs} epochs)")
        pf(f"  output_loss={args.output_loss}  q_alpha={args.quantizer_alpha}")
        pf(f"  lambdas: mem={args.memoq_lambda_mem_p3}  innov={args.memoq_lambda_innov_p3}  "
           f"zsat={args.memoq_lambda_zsat_p3}  rail={args.memoq_lambda_rail_p3}  "
           f"shape={args.memoq_lambda_shape_p3}")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p3 = max(
            args.effective_lr * args.memoq_lr_mult_p3,
            args.memoq_phase3_lr_floor,
        )
        opt_p3 = keras.optimizers.Adam(learning_rate=lr_p3)
        sched_p3 = ReduceLROnPlateau(
            opt_p3, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p3.reset(lr_p3)

        ep3_start = start_ep("P3")

        dist_train_p3 = make_dist_memoq_train_final(
            strategy, final_qkeras_student, opt_p3,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p3, args.memoq_lambda_innov_p3,
            args.memoq_lambda_zsat_p3, args.memoq_lambda_rail_p3,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            clipnorm=0.5,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p3,
            output_loss=args.output_loss,
        )
        dist_val_p3 = make_dist_memoq_val_final(
            strategy, final_qkeras_student,
            args.alpha, channel_scales, args.memoq_huber_delta,
            args.memoq_lambda_mem_p3, args.memoq_lambda_innov_p3,
            args.memoq_lambda_zsat_p3, args.memoq_lambda_rail_p3,
            epsilon_innov, args.seq_len,
            args.memoq_rho_rail, args.memoq_mu_rail,
            teacher_hidden_model=teacher_hidden_model,
            lambda_s=args.memoq_lambda_shape_p3,
            output_loss=args.output_loss,
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
            "P2F", best_vals.get(
                "P2E", best_vals.get(
                    "P2D", best_vals.get(
                        "P2C", best_vals.get(
                            "P2B", best_vals.get(
                                "P2A", best_vals.get("P1", float("inf"))
                            )
                        )
                    )
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

def transfer_splitgate_to_qkeras(
    enc_cell,
    dec_cell,
    phase2_model,
    final_qkeras_student,
    pf,
):
    """
    Pack the reset-before MemoQ split-gate parameters into the final QKeras
    QGRU layout.

    Target layout for reset_after=False:

        kernel:
            (input_dim, 3*H)
            [W_z | W_r | W_h]

        recurrent_kernel:
            (H, 3*H)
            [U_z | U_r | U_h]

        bias:
            (3*H,)
            [b_z | b_r | b_h]

    No recurrent-bias row exists.
    """

    pf(
        "[P2->P3 TRANSFER] Packing QKeras-compatible reset-before "
        "split-gate weights into QGRU..."
    )

    def pack_and_set(
        cell,
        layer_name,
    ):
        try:
            target_layer = final_qkeras_student.get_layer(
                layer_name
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{layer_name} is missing from final_qkeras_student"
            ) from exc

        target_cell = target_layer.cell

        if bool(target_cell.reset_after):
            raise RuntimeError(
                f"{layer_name}: final QGRU reset_after must be False"
            )

        if int(target_cell.implementation) != 1:
            raise RuntimeError(
                f"{layer_name}: final QGRU implementation must be 1"
            )

        probe = tf.constant(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
            ],
            dtype=tf.float32,
        )

        expected_gate = qkeras_hard_sigmoid(
            probe
        )

        actual_gate = tf.cast(
            target_cell.recurrent_activation(
                probe
            ),
            tf.float32,
        )

        activation_error = float(
            tf.reduce_max(
                tf.abs(
                    expected_gate
                    - actual_gate
                )
            ).numpy()
        )

        if activation_error > 1.0e-7:
            raise RuntimeError(
                f"{layer_name}: final QGRU recurrent activation is not "
                f"QKeras hard_sigmoid. "
                f"max_error={activation_error:.9e}"
            )

        W_z = np.asarray(
            cell.W_z.numpy(),
            dtype=np.float32,
        )

        W_r = np.asarray(
            cell.W_r.numpy(),
            dtype=np.float32,
        )

        W_h = np.asarray(
            cell.W_h.numpy(),
            dtype=np.float32,
        )

        U_z = np.asarray(
            cell.U_z.numpy(),
            dtype=np.float32,
        )

        U_r = np.asarray(
            cell.U_r.numpy(),
            dtype=np.float32,
        )

        U_h = np.asarray(
            cell.U_h.numpy(),
            dtype=np.float32,
        )

        b_z = np.asarray(
            cell.b_z_inp.numpy(),
            dtype=np.float32,
        )

        b_r = np.asarray(
            cell.b_r_inp.numpy(),
            dtype=np.float32,
        )

        b_h = np.asarray(
            cell.b_h_inp.numpy(),
            dtype=np.float32,
        )

        packed_kernel = np.concatenate(
            [
                W_z,
                W_r,
                W_h,
            ],
            axis=1,
        ).astype(
            np.float32
        )

        packed_recurrent = np.concatenate(
            [
                U_z,
                U_r,
                U_h,
            ],
            axis=1,
        ).astype(
            np.float32
        )

        packed_bias = np.concatenate(
            [
                b_z,
                b_r,
                b_h,
            ],
            axis=0,
        ).astype(
            np.float32
        )

        target_weights = target_layer.get_weights()

        if len(target_weights) < 3:
            raise RuntimeError(
                f"{layer_name}: expected at least three target weight tensors, "
                f"got {[tuple(weight.shape) for weight in target_weights]}"
            )

        if target_weights[0].shape != packed_kernel.shape:
            raise RuntimeError(
                f"{layer_name}: kernel shape mismatch. "
                f"source={packed_kernel.shape} "
                f"target={target_weights[0].shape}"
            )

        if target_weights[1].shape != packed_recurrent.shape:
            raise RuntimeError(
                f"{layer_name}: recurrent-kernel shape mismatch. "
                f"source={packed_recurrent.shape} "
                f"target={target_weights[1].shape}"
            )

        if target_weights[2].shape != packed_bias.shape:
            raise RuntimeError(
                f"{layer_name}: bias shape mismatch. "
                f"source={packed_bias.shape} "
                f"target={target_weights[2].shape}. "
                "The target must be a reset_after=False QGRU."
            )

        replacement_weights = list(
            target_weights
        )

        replacement_weights[0] = packed_kernel
        replacement_weights[1] = packed_recurrent
        replacement_weights[2] = packed_bias

        target_layer.set_weights(
            replacement_weights
        )

        pf(
            f"  OK {layer_name}: "
            f"kernel={packed_kernel.shape} "
            f"recurrent={packed_recurrent.shape} "
            f"bias={packed_bias.shape}"
        )

    pack_and_set(
        enc_cell,
        "sencgru",
    )

    pack_and_set(
        dec_cell,
        "sdecgru",
    )

    try:
        src_dense = phase2_model.get_layer(
            "sdec_dense"
        )

        dst_dense = final_qkeras_student.get_layer(
            "sdec_dense"
        )

    except ValueError as exc:
        raise RuntimeError(
            "sdec_dense is missing during P2->P3 transfer"
        ) from exc

    src_weights = src_dense.get_weights()
    dst_weights = dst_dense.get_weights()

    if len(src_weights) < 2:
        raise RuntimeError(
            "Phase-2 sdec_dense does not expose kernel and bias"
        )

    if len(dst_weights) < 2:
        raise RuntimeError(
            "Final sdec_dense does not expose kernel and bias"
        )

    if src_weights[0].shape != dst_weights[0].shape:
        raise RuntimeError(
            "sdec_dense kernel shape mismatch: "
            f"src={src_weights[0].shape} "
            f"dst={dst_weights[0].shape}"
        )

    if src_weights[1].shape != dst_weights[1].shape:
        raise RuntimeError(
            "sdec_dense bias shape mismatch: "
            f"src={src_weights[1].shape} "
            f"dst={dst_weights[1].shape}"
        )

    dense_replacement = list(
        dst_weights
    )

    dense_replacement[0] = np.asarray(
        src_weights[0],
        dtype=np.float32,
    )

    dense_replacement[1] = np.asarray(
        src_weights[1],
        dtype=np.float32,
    )

    dst_dense.set_weights(
        dense_replacement
    )

    pf(
        "  OK sdec_dense"
    )

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
def qkeras_hard_sigmoid(x):
    """
    Exact QKeras hard_sigmoid used by QGRU.

    QKeras quantizers.hard_sigmoid implements:

        clip(0.5 * x + 0.5, 0.0, 1.0)

    Do not replace this with tf.keras.activations.hard_sigmoid. The historical
    Keras hard-sigmoid implementation uses a different slope and therefore
    does not reproduce the QKeras recurrent activation used by the vanilla
    QGRU and the HLS implementation.
    """
    x = tf.cast(x, tf.float32)

    return tf.clip_by_value(
        0.5 * x + 0.5,
        0.0,
        1.0,
    )

class MemoQGRUCell(keras.layers.Layer):
    """
    Split-gate GRU cell used by the MemoQ progressive-hardening trajectory.

    The recurrent mathematics intentionally matches the vanilla QKeras QGRU:

        recurrent_activation = QKeras hard_sigmoid
        implementation       = 1
        reset_after           = False

    Hard-inference recurrence:

        h_prev_q = Q_state(h_prev)

        z_logit = x @ W_z + h_prev_q @ U_z + b_z
        r_logit = x @ W_r + h_prev_q @ U_r + b_r

        z = hard_sigmoid(z_logit)
        r = hard_sigmoid(r_logit)

        candidate_preact =
            x @ W_h
            + (r * h_prev_q) @ U_h
            + b_h

        candidate = Q_activation(candidate_preact)

        h_t =
            z * h_prev_q
            + (1.0 - z) * candidate

    The newly produced h_t is deliberately NOT state-quantized before it is
    returned. It is carried at higher precision and Q_state is applied when
    that raw state re-enters the cell on the next recurrent step. This is the
    exact state_quantizer location used by QKeras QGRUCell.call().

    MemoQ's existing training-only activation/state blending and dither fields
    are retained. They change the training path only. At the hard tail and at
    inference, beta=1 and dither=0 recover the exact QKeras recurrence above.
    """

    def __init__(
        self,
        units,
        input_dim,
        quantizer_z=None,
        quantizer_r=None,
        quantizer_h=None,
        quantizer_state=None,
        quantizer_activation=None,
        quantizer_recurrent_z=None,
        quantizer_recurrent_r=None,
        quantizer_recurrent_h=None,
        quantizer_bias=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.units = int(units)
        self.input_dim = int(input_dim)

        self._quantizer_z = quantizer_z
        self._quantizer_r = quantizer_r
        self._quantizer_h = quantizer_h

        self._quantizer_state = quantizer_state
        self._quantizer_activation = quantizer_activation

        self._quantizer_recurrent_z = quantizer_recurrent_z
        self._quantizer_recurrent_r = quantizer_recurrent_r
        self._quantizer_recurrent_h = quantizer_recurrent_h

        self._quantizer_bias = quantizer_bias

        self.state_size = [
            self.units,
            self.units,
        ]

        self.output_size = self.units

        self._state_blend_beta = 1.0
        self._activation_blend_beta = 1.0

        self._act_dither_delta = 0.0
        self._state_dither_delta = 0.0

        self._state_lsb_noise_std = 0.0

    def build(self, input_shape):
        input_dim_from_shape = int(input_shape[-1])

        if input_dim_from_shape != self.input_dim:
            raise ValueError(
                f"{self.name}: configured input_dim={self.input_dim}, "
                f"but received input_shape={input_shape}"
            )

        H = self.units

        glorot = keras.initializers.GlorotUniform()
        orth = keras.initializers.Orthogonal()
        zeros = keras.initializers.Zeros()

        self.W_z = self.add_weight(
            name="W_z",
            shape=(self.input_dim, H),
            initializer=glorot,
            trainable=True,
        )

        self.W_r = self.add_weight(
            name="W_r",
            shape=(self.input_dim, H),
            initializer=glorot,
            trainable=True,
        )

        self.W_h = self.add_weight(
            name="W_h",
            shape=(self.input_dim, H),
            initializer=glorot,
            trainable=True,
        )

        self.U_z = self.add_weight(
            name="U_z",
            shape=(H, H),
            initializer=orth,
            trainable=True,
        )

        self.U_r = self.add_weight(
            name="U_r",
            shape=(H, H),
            initializer=orth,
            trainable=True,
        )

        self.U_h = self.add_weight(
            name="U_h",
            shape=(H, H),
            initializer=orth,
            trainable=True,
        )

        # reset_after=False has one bias vector per gate.
        # There are deliberately NO recurrent-bias variables.
        self.b_z_inp = self.add_weight(
            name="b_z_inp",
            shape=(H,),
            initializer=zeros,
            trainable=True,
        )

        self.b_r_inp = self.add_weight(
            name="b_r_inp",
            shape=(H,),
            initializer=zeros,
            trainable=True,
        )

        self.b_h_inp = self.add_weight(
            name="b_h_inp",
            shape=(H,),
            initializer=zeros,
            trainable=True,
        )

        super().build(input_shape)

    @staticmethod
    def _apply_quantizer(quantizer, tensor):
        if quantizer is None:
            return tf.cast(
                tensor,
                tf.float32,
            )

        return tf.cast(
            quantizer(tensor),
            tf.float32,
        )

    def call(
        self,
        inputs,
        states,
        training=None,
    ):
        inputs = tf.cast(
            inputs,
            tf.float32,
        )

        h_prev_raw = tf.cast(
            states[0],
            tf.float32,
        )

        if training is None:
            is_training = tf.constant(
                False,
                dtype=tf.bool,
            )
        elif isinstance(training, bool):
            is_training = tf.constant(
                training,
                dtype=tf.bool,
            )
        else:
            is_training = tf.cast(
                training,
                tf.bool,
            )

        # ------------------------------------------------------------------
        # Training-only state dither.
        #
        # Inference is unaffected because training=False contributes zero
        # dither. The hard deployment recurrence therefore begins exactly
        # with Q_state(h_prev_raw).
        # ------------------------------------------------------------------

        h_prev_for_state = h_prev_raw

        if self._state_dither_delta > 0.0:
            sd = tf.cast(
                self._state_dither_delta,
                tf.float32,
            )

            u1_s = tf.random.uniform(
                tf.shape(h_prev_raw),
                minval=-sd,
                maxval=sd,
                dtype=tf.float32,
            )

            u2_s = tf.random.uniform(
                tf.shape(h_prev_raw),
                minval=-sd,
                maxval=sd,
                dtype=tf.float32,
            )

            tri_s = u1_s + u2_s

            training_state_dither = tf.cond(
                is_training,
                lambda: tri_s,
                lambda: tf.zeros_like(tri_s),
            )

            h_prev_for_state = (
                h_prev_raw
                + training_state_dither
            )

        # ------------------------------------------------------------------
        # Exact QKeras state-quantizer location:
        #
        #     quantize the PREVIOUS raw state when it re-enters the cell.
        #
        # Do not quantize h_t after the final blend.
        # ------------------------------------------------------------------

        if self._quantizer_state is not None:
            h_prev_q_hard = tf.cast(
                self._quantizer_state(
                    h_prev_for_state
                ),
                tf.float32,
            )

            beta_s = tf.cast(
                self._state_blend_beta,
                tf.float32,
            )

            h_prev_q = (
                beta_s * h_prev_q_hard
                + (1.0 - beta_s) * h_prev_for_state
            )

        else:
            h_prev_q = h_prev_for_state

        # ------------------------------------------------------------------
        # Quantized parameters.
        # ------------------------------------------------------------------

        W_z = self._apply_quantizer(
            self._quantizer_z,
            self.W_z,
        )

        W_r = self._apply_quantizer(
            self._quantizer_r,
            self.W_r,
        )

        W_h = self._apply_quantizer(
            self._quantizer_h,
            self.W_h,
        )

        U_z = self._apply_quantizer(
            self._quantizer_recurrent_z,
            self.U_z,
        )

        U_r = self._apply_quantizer(
            self._quantizer_recurrent_r,
            self.U_r,
        )

        U_h = self._apply_quantizer(
            self._quantizer_recurrent_h,
            self.U_h,
        )

        b_z = self._apply_quantizer(
            self._quantizer_bias,
            self.b_z_inp,
        )

        b_r = self._apply_quantizer(
            self._quantizer_bias,
            self.b_r_inp,
        )

        b_h = self._apply_quantizer(
            self._quantizer_bias,
            self.b_h_inp,
        )

        # ------------------------------------------------------------------
        # Exact QKeras update/reset logits.
        #
        # reset_after=False has no recurrent bias row.
        # ------------------------------------------------------------------

        z_logit = (
            tf.matmul(
                inputs,
                W_z,
            )
            + tf.matmul(
                h_prev_q,
                U_z,
            )
            + b_z
        )

        r_logit = (
            tf.matmul(
                inputs,
                W_r,
            )
            + tf.matmul(
                h_prev_q,
                U_r,
            )
            + b_r
        )

        # Exact QKeras hard_sigmoid, NOT tf.sigmoid.
        z = qkeras_hard_sigmoid(
            z_logit
        )

        r = qkeras_hard_sigmoid(
            r_logit
        )

        # ------------------------------------------------------------------
        # Exact reset-before candidate:
        #
        #     (r * h_prev_q) @ U_h
        #
        # NOT:
        #
        #     r * (h_prev_q @ U_h + recurrent_bias)
        # ------------------------------------------------------------------

        recurrent_candidate = tf.matmul(
            r * h_prev_q,
            U_h,
        )

        candidate_preact = (
            tf.matmul(
                inputs,
                W_h,
            )
            + recurrent_candidate
            + b_h
        )

        # ------------------------------------------------------------------
        # Candidate activation.
        #
        # At hard inference:
        #
        #     h_candidate = QKeras quantized_tanh(candidate_preact)
        #
        # Existing MemoQ training-only soft blending/dither is retained.
        # ------------------------------------------------------------------

        if self._quantizer_activation is not None:
            tanh_f = tf.tanh(
                candidate_preact
            )

            beta_a = tf.cast(
                self._activation_blend_beta,
                tf.float32,
            )

            if self._act_dither_delta > 0.0:
                ad = tf.cast(
                    self._act_dither_delta,
                    tf.float32,
                )

                grid_step = tf.cast(
                    0.125,
                    tf.float32,
                )

                u1_a = tf.random.uniform(
                    tf.shape(tanh_f),
                    minval=-ad,
                    maxval=ad,
                    dtype=tf.float32,
                )

                u2_a = tf.random.uniform(
                    tf.shape(tanh_f),
                    minval=-ad,
                    maxval=ad,
                    dtype=tf.float32,
                )

                tri_a = u1_a + u2_a

                dithered = (
                    tanh_f
                    + tri_a
                )

                rounded_forward = tf.clip_by_value(
                    tf.round(
                        dithered
                        / grid_step
                    )
                    * grid_step,
                    -1.0,
                    1.0 - grid_step,
                )

                rounded_ste = (
                    dithered
                    + tf.stop_gradient(
                        rounded_forward
                        - dithered
                    )
                )

                q_dith_training = (
                    rounded_ste
                    - tri_a
                )

                q_exact_inference = tf.cast(
                    self._quantizer_activation(
                        candidate_preact
                    ),
                    tf.float32,
                )

                q_dith = tf.cond(
                    is_training,
                    lambda: tf.cast(
                        q_dith_training,
                        tf.float32,
                    ),
                    lambda: q_exact_inference,
                )

            else:
                q_dith = tf.cast(
                    self._quantizer_activation(
                        candidate_preact
                    ),
                    tf.float32,
                )

            h_candidate = (
                beta_a * q_dith
                + (1.0 - beta_a) * tanh_f
            )

        else:
            h_candidate = tf.tanh(
                candidate_preact
            )

        # ------------------------------------------------------------------
        # Exact QKeras update equation.
        #
        # z is a RETENTION coefficient.
        #
        # h_t stays raw/high precision. State quantization happens when this
        # value becomes h_prev_raw on the NEXT call.
        # ------------------------------------------------------------------

        h_t = (
            z * h_prev_q
            + (1.0 - z) * h_candidate
        )

        return h_t, [
            h_t,
            z_logit,
        ]

    def get_initial_state(
        self,
        inputs=None,
        batch_size=None,
        dtype=None,
    ):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(
                inputs
            )[0]

        if dtype is None:
            dtype = tf.float32

        return [
            tf.zeros(
                (
                    batch_size,
                    self.units,
                ),
                dtype=dtype,
            ),
            tf.zeros(
                (
                    batch_size,
                    self.units,
                ),
                dtype=dtype,
            ),
        ]

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "units": self.units,
                "input_dim": self.input_dim,
            }
        )

        return config

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

    @property
    def quantizer_recurrent_z(self):
        return self._quantizer_recurrent_z

    @quantizer_recurrent_z.setter
    def quantizer_recurrent_z(self, q):
        self._quantizer_recurrent_z = q

    @property
    def quantizer_recurrent_r(self):
        return self._quantizer_recurrent_r

    @quantizer_recurrent_r.setter
    def quantizer_recurrent_r(self, q):
        self._quantizer_recurrent_r = q

    @property
    def quantizer_recurrent_h(self):
        return self._quantizer_recurrent_h

    @quantizer_recurrent_h.setter
    def quantizer_recurrent_h(self, q):
        self._quantizer_recurrent_h = q

    @property
    def quantizer_bias(self):
        return self._quantizer_bias

    @quantizer_bias.setter
    def quantizer_bias(self, q):
        self._quantizer_bias = q

    @property
    def state_blend_beta(self):
        return self._state_blend_beta

    @state_blend_beta.setter
    def state_blend_beta(self, v):
        self._state_blend_beta = float(v)

    @property
    def activation_blend_beta(self):
        return self._activation_blend_beta

    @activation_blend_beta.setter
    def activation_blend_beta(self, v):
        self._activation_blend_beta = float(v)

    @property
    def act_dither_delta(self):
        return self._act_dither_delta

    @act_dither_delta.setter
    def act_dither_delta(self, v):
        self._act_dither_delta = float(v)

    @property
    def state_dither_delta(self):
        return self._state_dither_delta

    @state_dither_delta.setter
    def state_dither_delta(self, v):
        self._state_dither_delta = float(v)

    @property
    def state_lsb_noise_std(self):
        return self._state_lsb_noise_std

    @state_lsb_noise_std.setter
    def state_lsb_noise_std(self, v):
        self._state_lsb_noise_std = float(v)
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


def build_phase2_model(seq_len, n_out, student_units, input_dim=1, q_alpha=1.0, bits_kernel=4):
    """
    Phase 2 split-gate training model.

    The dense head is now a QDense with the SAME quantizer as the final QKeras
    student's sdec_dense (task-doc item 4). Training the readout under 4-bit
    quantisation from Phase 2 onward removes the float-Dense -> QDense transfer
    cliff that previously destroyed scatter performance at export.

    Outputs: (seq_output, dec_hidden_seq, dec_z_logit_seq).
    """
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

    # QDense head matching the final QKeras student exactly (item 4).
    if q_alpha == 1.0:
        dense_kq = quantized_bits(bits_kernel, 0)
        dense_bq = quantized_bits(bits_kernel, 0)
    else:
        dense_kq = quantized_bits(bits_kernel, 0, 1, alpha=q_alpha)
        dense_bq = quantized_bits(bits_kernel, 0, 1, alpha=q_alpha)

    seq_output = QDense(
        n_out,
        kernel_quantizer=dense_kq,
        bias_quantizer=dense_bq,
        activation="linear",
        name="sdec_dense",
    )(dec_hidden_seq)

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
    seq_len,
    n_out,
    student_units,
    bits_kernel,
    bits_recurrent,
    bits_bias,
    bits_activation,
    bits_state,
    q_alpha=1.0,
):
    """
    Final hard QKeras student.

    This graph explicitly states the QKeras recurrent defaults used by the
    vanilla student rather than relying on implicit defaults:

        recurrent_activation = "hard_sigmoid"
        implementation       = 1
        reset_after           = False

    With q_alpha=1.0 and the same bit widths, this is the same inference
    architecture and quantizer family as the vanilla QKeras student.
    """

    def qwk():
        return quantized_bits(
            bits_kernel,
            0,
            1,
            alpha=q_alpha,
        )

    def qwr():
        return quantized_bits(
            bits_recurrent,
            0,
            1,
            alpha=q_alpha,
        )

    def qwb():
        return quantized_bits(
            bits_bias,
            0,
            1,
            alpha=q_alpha,
        )

    def qa():
        return quantized_tanh(
            bits=bits_activation,
            symmetric=True,
        )

    def qs():
        return quantized_bits(
            bits_state,
            0,
            1,
            alpha=1.0,
        )

    def qd():
        if q_alpha == 1.0:
            return quantized_bits(
                bits_kernel,
                0,
            )

        return quantized_bits(
            bits_kernel,
            0,
            1,
            alpha=q_alpha,
        )

    enc_inputs = keras.layers.Input(
        shape=(None, 1),
        name="senc_input",
    )

    dec_inputs = keras.layers.Input(
        shape=(None, 1),
        name="sdec_input",
    )

    s_enc_out, s_enc_state = QGRU(
        units=student_units,
        activation=qa(),
        recurrent_activation="hard_sigmoid",
        implementation=1,
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        reset_after=False,
        name="sencgru",
    )(
        enc_inputs
    )

    s_dec_hid_seq, _ = QGRU(
        units=student_units,
        activation=qa(),
        recurrent_activation="hard_sigmoid",
        implementation=1,
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        reset_after=False,
        name="sdecgru",
    )(
        dec_inputs,
        initial_state=s_enc_state,
    )

    s_output = QDense(
        n_out,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(
        s_dec_hid_seq
    )

    return keras.models.Model(
        inputs=[
            enc_inputs,
            dec_inputs,
        ],
        outputs=s_output,
        name="memoq_final_qkeras_student",
    )

def log_final_student_config(
    final_qkeras_student,
    pf,
):
    """
    Validate and log the final QKeras architecture.

    The final MemoQ student is only a valid vanilla-matched controlled
    experiment when BOTH recurrent layers satisfy:

        QKeras hard_sigmoid
        implementation=1
        reset_after=False
        state_quantizer present

    A mismatch is fatal.
    """

    probe = tf.constant(
        [
            -3.0,
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
        ],
        dtype=tf.float32,
    )

    expected_gate = qkeras_hard_sigmoid(
        probe
    )

    for layer_name in [
        "sencgru",
        "sdecgru",
    ]:
        try:
            layer = final_qkeras_student.get_layer(
                layer_name
            )
        except ValueError as exc:
            raise RuntimeError(
                f"[CONFIG] Missing required QGRU layer: {layer_name}"
            ) from exc

        cell = layer.cell

        if bool(cell.reset_after):
            raise RuntimeError(
                f"[CONFIG] {layer_name}: reset_after=True. "
                "Expected reset_after=False."
            )

        if int(cell.implementation) != 1:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: "
                f"implementation={cell.implementation}. "
                "Expected implementation=1."
            )

        actual_gate = tf.cast(
            cell.recurrent_activation(
                probe
            ),
            tf.float32,
        )

        gate_error = float(
            tf.reduce_max(
                tf.abs(
                    actual_gate
                    - expected_gate
                )
            ).numpy()
        )

        if gate_error > 1.0e-7:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: recurrent activation does not "
                f"match QKeras hard_sigmoid. "
                f"max_error={gate_error:.9e}"
            )

        if cell.kernel_quantizer_internal is None:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: kernel quantizer is missing"
            )

        if cell.recurrent_quantizer_internal is None:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: recurrent quantizer is missing"
            )

        if cell.bias_quantizer_internal is None:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: bias quantizer is missing"
            )

        if cell.state_quantizer_internal is None:
            raise RuntimeError(
                f"[CONFIG] {layer_name}: state quantizer is missing"
            )

        pf(
            f"[CONFIG] {layer_name}: "
            f"units={cell.units} "
            f"implementation={cell.implementation} "
            f"reset_after={cell.reset_after} "
            f"kernel_quantizer={cell.kernel_quantizer_internal} "
            f"recurrent_quantizer={cell.recurrent_quantizer_internal} "
            f"bias_quantizer={cell.bias_quantizer_internal} "
            f"state_quantizer={cell.state_quantizer_internal} "
            f"activation={cell.activation} "
            f"hard_sigmoid_max_error={gate_error:.3e}"
        )

    try:
        dense_layer = final_qkeras_student.get_layer(
            "sdec_dense"
        )
    except ValueError as exc:
        raise RuntimeError(
            "[CONFIG] Missing required QDense layer: sdec_dense"
        ) from exc

    pf(
        "[CONFIG] sdec_dense: "
        f"kernel_quantizer={dense_layer.kernel_quantizer_internal} "
        f"bias_quantizer={dense_layer.bias_quantizer_internal} "
        f"activation={dense_layer.activation}"
    )

    pf(
        "[CONFIG] PASS: final MemoQ QKeras graph uses "
        "hard_sigmoid / implementation=1 / reset_after=False / "
        "state-on-reentry quantization."
    )

    sys.stdout.flush()

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


def output_loss_fn(y_true, y_pred, channel_scales, huber_delta, output_loss):
    """
    Base output KD loss selector.

    output_loss == "mse"      -> plain MSE, byte-for-byte the vanilla KD objective.
    output_loss == "huber_cn" -> channel-normalised Huber (original MemoQ).

    Passing "mse" with all auxiliary lambdas at 0.0 makes the MemoQ output
    objective identical to vanilla, which is the required control (ladder A/B).
    """
    if output_loss == "mse":
        return tf.reduce_mean(tf.square(y_pred - y_true))
    return channel_normalised_huber_memoq(y_true, y_pred, channel_scales, huber_delta)

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
    rho_z=0.98,
    output_loss="mse",
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
    rho_z_f      = float(rho_z)

    def train_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        # Only run the teacher-hidden forward pass if a hidden loss actually
        # consumes it. For the control run (mem=innov=0) this skips a full
        # teacher forward every batch.
        if (use_mem or use_innov) and teacher_hidden_model is not None:
            h_teacher = teacher_hidden_model([enc_b, dec_b], training=False)
        elif (use_mem or use_innov):
            h_teacher = batch_x.get("teacher_hidden", None)
        else:
            h_teacher = None

        with tf.GradientTape() as tape:
            model_out = phase2_model([enc_b, dec_b], training=True)

            if has_z_logit:
                s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
            else:
                s_pred, s_hid = model_out[0], model_out[1]
                z_logits = None

            l_seq = output_loss_fn(tgt_b,   s_pred, channel_scales, huber_delta, output_loss)
            l_kd  = output_loss_fn(tpred_b, s_pred, channel_scales, huber_delta, output_loss)
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
                l_z = loss_zsat_value(z_vals, rho_z=rho_z_f)
                total = total + lambda_z_f * l_z
            else:
                l_z = tf.constant(0.0, dtype=tf.float32)

            if use_rail:
                l_r = loss_railpred(s_hid, rho_rail, mu_rail)
                total = total + lambda_r_f * l_r
            else:
                l_r = tf.constant(0.0, dtype=tf.float32)

            if lambda_s_f > 0.0:
                l_shape = (
                    (1.0 - alpha_f) * loss_shape(tgt_b,   s_pred, huber_delta)
                    + alpha_f       * loss_shape(tpred_b, s_pred, huber_delta)
                )
                total = total + lambda_s_f * l_shape
            else:
                l_shape = tf.constant(0.0, dtype=tf.float32)

        grads = tape.gradient(total, phase2_model.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, phase2_model.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) | tf.reduce_any(tf.math.is_inf(g))
            for g in grads
        ]))
        # Skip the update on a bad batch: zero every gradient so apply_gradients
        # is a no-op instead of writing NaN/Inf into the weights.
        grads = [tf.where(nan_in_grads, tf.zeros_like(g), g) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, phase2_model.trainable_variables))

        # Return contract: (total, seq, kd, mem, innov, zsat, rail, shape, nan)
        return (
            total,
            l_seq,
            l_kd,
            l_m,
            l_i,
            l_z,
            l_r,
            l_shape,
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
    rho_z=0.98,
    output_loss="mse",
):
    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    lambda_s_f   = tf.cast(lambda_s,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    seq_len_int  = int(seq_len)
    rho_z_f      = float(rho_z)

    def val_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        if (use_mem or use_innov) and teacher_hidden_model is not None:
            h_teacher = teacher_hidden_model([enc_b, dec_b], training=False)
        elif (use_mem or use_innov):
            h_teacher = batch_x.get("teacher_hidden", None)
        else:
            h_teacher = None

        model_out = phase2_model([enc_b, dec_b], training=False)

        if has_z_logit:
            s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
        else:
            s_pred, s_hid = model_out[0], model_out[1]
            z_logits = None

        l_seq = output_loss_fn(tgt_b,   s_pred, channel_scales, huber_delta, output_loss)
        l_kd  = output_loss_fn(tpred_b, s_pred, channel_scales, huber_delta, output_loss)
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
            l_z = loss_zsat_value(z_vals, rho_z=rho_z_f)
            total = total + lambda_z_f * l_z
        else:
            l_z = tf.constant(0.0, dtype=tf.float32)

        if use_rail:
            l_r = loss_railpred(s_hid, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r
        else:
            l_r = tf.constant(0.0, dtype=tf.float32)

        if lambda_s_f > 0.0:
            l_shape = (
                (1.0 - alpha_f) * loss_shape(tgt_b,   s_pred, huber_delta)
                + alpha_f       * loss_shape(tpred_b, s_pred, huber_delta)
            )
            total = total + lambda_s_f * l_shape
        else:
            l_shape = tf.constant(0.0, dtype=tf.float32)

        mae = tf.reduce_mean(tf.abs(s_pred - tgt_b))

        # Return contract: (total, seq, kd, mem, innov, zsat, rail, shape, mae)
        return (
            total,
            l_seq,
            l_kd,
            l_m,
            l_i,
            l_z,
            l_r,
            l_shape,
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
    output_loss="mse",
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
    use_teacher_hidden = (float(lambda_m) > 0.0) or (float(lambda_i) > 0.0)

    def train_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        # Always produce a tensor for teacher_hid_b so AutoGraph never sees None
        # at the end of any branch inside the @tf.function trace.
        if use_teacher_hidden and teacher_hidden_model is not None:
            teacher_hid_b = teacher_hidden_model([enc_b, dec_b], training=False)
            teacher_hid_available = True
        elif use_teacher_hidden:
            _thid = batch_x.get("teacher_hidden", None)
            if _thid is not None:
                teacher_hid_b = _thid
                teacher_hid_available = True
            else:
                teacher_hid_b = tf.zeros(
                    (tf.shape(enc_b)[0], seq_len_int, 1), dtype=tf.float32
                )
                teacher_hid_available = False
        else:
            teacher_hid_b = tf.zeros(
                (tf.shape(enc_b)[0], seq_len_int, 1), dtype=tf.float32
            )
            teacher_hid_available = False

        with tf.GradientTape() as tape:
            seq_out, dec_h_seq = final_hidden_model([enc_b, dec_b], training=True)

            l_seq = output_loss_fn(tgt_b,   seq_out, channel_scales, huber_delta, output_loss)
            l_kd  = output_loss_fn(tpred_b, seq_out, channel_scales, huber_delta, output_loss)
            total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

            if teacher_hid_available and lambda_m_f > 0.0:
                l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len_int)
                total = total + lambda_m_f * l_m
            else:
                l_m = tf.constant(0.0, dtype=tf.float32)

            if teacher_hid_available and lambda_i_f > 0.0:
                l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
                total = total + lambda_i_f * l_i
            else:
                l_i = tf.constant(0.0, dtype=tf.float32)

            if lambda_r_f > 0.0:
                l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
                total = total + lambda_r_f * l_r
            else:
                l_r = tf.constant(0.0, dtype=tf.float32)

            if lambda_z_f > 0.0:
                excess_z = tf.nn.relu(tf.abs(dec_h_seq) - 0.90)
                l_z = tf.reduce_mean(tf.square(excess_z))
                total = total + lambda_z_f * l_z
            else:
                l_z = tf.constant(0.0, dtype=tf.float32)

            if lambda_s_f > 0.0:
                l_shape = (
                    (1.0 - alpha_f) * loss_shape(tgt_b,   seq_out, huber_delta)
                    + alpha_f       * loss_shape(tpred_b, seq_out, huber_delta)
                )
                total = total + lambda_s_f * l_shape
            else:
                l_shape = tf.constant(0.0, dtype=tf.float32)

        grads = tape.gradient(total, final_qkeras_student.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, final_qkeras_student.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) | tf.reduce_any(tf.math.is_inf(g))
            for g in grads
        ]))
        grads = [tf.where(nan_in_grads, tf.zeros_like(g), g) for g in grads]
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, final_qkeras_student.trainable_variables))

        # Return contract: (total, seq, kd, mem, innov, zsat, rail, shape, nan)
        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, l_shape, tf.cast(nan_in_grads, tf.float32)

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
    output_loss="mse",
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
    use_teacher_hidden = (float(lambda_m) > 0.0) or (float(lambda_i) > 0.0)

    def val_step_per_replica(batch_x, batch_y):
        enc_b   = batch_x["enc_input"]
        dec_b   = batch_x["dec_input"]
        tpred_b = batch_x["tpred"]
        tgt_b   = batch_y

        # Always assign a tensor to teacher_hid_b before the conditional block
        # so AutoGraph never sees None at the end of any branch inside
        # the @tf.function trace. This is the fix for the crash:
        #   ValueError: 'teacher_hid_b' is None at the end of the main branch.
        teacher_hid_b = tf.zeros(
            (tf.shape(enc_b)[0], seq_len_int, 1), dtype=tf.float32
        )
        teacher_hid_available = False

        if use_teacher_hidden and teacher_hidden_model is not None:
            teacher_hid_b = teacher_hidden_model([enc_b, dec_b], training=False)
            teacher_hid_available = True
        elif use_teacher_hidden:
            _thid = batch_x.get("teacher_hidden", None)
            if _thid is not None:
                teacher_hid_b = _thid
                teacher_hid_available = True

        seq_out, dec_h_seq = final_hidden_model_val([enc_b, dec_b], training=False)

        l_seq = output_loss_fn(tgt_b,   seq_out, channel_scales, huber_delta, output_loss)
        l_kd  = output_loss_fn(tpred_b, seq_out, channel_scales, huber_delta, output_loss)
        total = (1.0 - alpha_f) * l_seq + alpha_f * l_kd

        if teacher_hid_available and lambda_m_f > 0.0:
            l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len_int)
            total = total + lambda_m_f * l_m
        else:
            l_m = tf.constant(0.0, dtype=tf.float32)

        if teacher_hid_available and lambda_i_f > 0.0:
            l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
            total = total + lambda_i_f * l_i
        else:
            l_i = tf.constant(0.0, dtype=tf.float32)

        if lambda_r_f > 0.0:
            l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r
        else:
            l_r = tf.constant(0.0, dtype=tf.float32)

        if lambda_z_f > 0.0:
            excess_z = tf.nn.relu(tf.abs(dec_h_seq) - 0.90)
            l_z = tf.reduce_mean(tf.square(excess_z))
            total = total + lambda_z_f * l_z
        else:
            l_z = tf.constant(0.0, dtype=tf.float32)

        if lambda_s_f > 0.0:
            l_shape = (
                (1.0 - alpha_f) * loss_shape(tgt_b,   seq_out, huber_delta)
                + alpha_f       * loss_shape(tpred_b, seq_out, huber_delta)
            )
            total = total + lambda_s_f * l_shape
        else:
            l_shape = tf.constant(0.0, dtype=tf.float32)

        mae = tf.reduce_mean(tf.abs(seq_out - tgt_b))

        # Return contract: (total, seq, kd, mem, innov, zsat, rail, shape, mae)
        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, l_shape, mae

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

    Step functions return a 9-tuple:
      (total, seq, kd, mem, innov, zsat, rail, shape, nan/mae)
    Index 5 = zsat (honest), index 7 = shape, index 8 = nan_flag (train) or mae (val).

    Returns: (history, best_val, patience_ct, early_stop)
    """
    t_epoch = time.time()

    # ── Warmup LR override ────────────────────────────────────────────────────
    if effective_warmup_epochs > 0 and epoch_in_phase < effective_warmup_epochs:
        warmup_frac = (epoch_in_phase + 1) / effective_warmup_epochs
        warmup_lr   = effective_lr * warmup_frac
        lr_scheduler.lr_var.assign(float(warmup_lr))

    # ── Train ─────────────────────────────────────────────────────────────────
    train_acc   = [0.0] * 8   # total, seq, kd, mem, innov, zsat, rail, shape
    nan_batches = 0
    step_count  = 0

    for step, (bx, by) in enumerate(dist_train_dataset):
        result = dist_train_step_fn(bx, by)
        result_np = [float(r) for r in result]

        nan_flag = result_np[8] if len(result_np) > 8 else 0.0
        if nan_flag > 0.0:
            nan_batches += 1

        for i in range(min(8, len(result_np))):
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
    val_acc    = [0.0] * 9   # total, seq, kd, mem, innov, zsat, rail, shape, mae
    val_count  = 0

    for step, (bx, by) in enumerate(dist_val_dataset):
        result = dist_val_step_fn(bx, by)
        result_np = [float(r) for r in result]
        for i in range(min(9, len(result_np))):
            val_acc[i] += result_np[i]
        val_count += 1
        if val_count >= val_steps:
            break

    if val_count > 0:
        val_acc = [v / val_count for v in val_acc]

    val_total = val_acc[0]
    val_mae   = val_acc[8] if len(val_acc) > 8 else 0.0

    # ── LR scheduler step ─────────────────────────────────────────────────────
    if epoch_in_phase >= effective_warmup_epochs:
        lr_scheduler.step(val_total, epoch, pf)

    current_lr = lr_scheduler.current_lr

    # ── History (zsat=index5, shape=index7 — both honest now) ─────────────────
    history["total"].append(train_acc[0])
    history["seq"].append(train_acc[1])
    history["kd"].append(train_acc[2])
    history["mem"].append(train_acc[3])
    history["innov"].append(train_acc[4])
    history["zsat"].append(train_acc[5])
    history["rail"].append(train_acc[6])
    history["shape"].append(train_acc[7])
    history["val_total"].append(val_acc[0])
    history["val_seq"].append(val_acc[1])
    history["val_kd"].append(val_acc[2])
    history["val_mem"].append(val_acc[3])
    history["val_innov"].append(val_acc[4])
    history["val_zsat"].append(val_acc[5])
    history["val_rail"].append(val_acc[6])
    history["val_shape"].append(val_acc[7])
    history["val_mae"].append(val_mae)
    history["phase"].append(phase_tag)

    # ── CSV append (added shape + val_shape columns) ──────────────────────────
    with open(csv_path, "a") as f:
        f.write(
            f"{epoch},{phase_tag},"
            f"{train_acc[0]:.6f},{train_acc[1]:.6f},{train_acc[2]:.6f},"
            f"{train_acc[3]:.6f},{train_acc[4]:.6f},{train_acc[5]:.6f},"
            f"{train_acc[6]:.6f},{train_acc[7]:.6f},"
            f"{val_acc[0]:.6f},{val_acc[1]:.6f},{val_acc[2]:.6f},"
            f"{val_acc[3]:.6f},{val_acc[4]:.6f},{val_acc[5]:.6f},"
            f"{val_acc[6]:.6f},{val_acc[7]:.6f},"
            f"{val_mae:.6f},{current_lr:.2e}\n"
        )

    # ── Log ───────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_epoch
    pf(
        f"[{phase_tag}] ep {epoch_in_phase:4d}/{total_epochs}  "
        f"loss={train_acc[0]:.5f}  val={val_total:.5f}  mae={val_mae:.5f}  "
        f"mem={train_acc[3]:.5f}  innov={train_acc[4]:.5f}  "
        f"zsat={train_acc[5]:.5f}  rail={train_acc[6]:.5f}  shape={train_acc[7]:.5f}  "
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

def transfer_float_to_phase2(
    float_student,
    phase2_model,
    enc_cell_p2,
    dec_cell_p2,
    pf,
):
    """
    Transfer reset_after=False Phase-1 Keras GRU parameters into the
    split-gate MemoQGRUCell.

    Keras GRU reset_after=False layout:

        kernel:
            shape (input_dim, 3*H)
            columns [z | r | h]

        recurrent_kernel:
            shape (H, 3*H)
            columns [z | r | h]

        bias:
            shape (3*H,)
            values [b_z | b_r | b_h]

    There is no recurrent-bias row.
    """

    pf(
        "[P1->P2 TRANSFER] Unpacking reset-before float GRU weights "
        "into QKeras-compatible split-gate variables..."
    )

    def unpack_and_set(
        gru_layer_name,
        cell,
    ):
        try:
            layer = float_student.get_layer(
                gru_layer_name
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{gru_layer_name} is missing from float_student"
            ) from exc

        if bool(layer.reset_after):
            raise RuntimeError(
                f"{gru_layer_name}: reset_after must be False"
            )

        if int(layer.implementation) != 1:
            raise RuntimeError(
                f"{gru_layer_name}: implementation must be 1"
            )

        weights = layer.get_weights()

        if len(weights) != 3:
            raise RuntimeError(
                f"{gru_layer_name}: expected exactly three GRU weight tensors "
                f"for reset_after=False, got "
                f"{len(weights)} with shapes "
                f"{[tuple(weight.shape) for weight in weights]}"
            )

        kernel = np.asarray(
            weights[0],
            dtype=np.float32,
        )

        recurrent_kernel = np.asarray(
            weights[1],
            dtype=np.float32,
        )

        bias = np.asarray(
            weights[2],
            dtype=np.float32,
        )

        H = int(
            cell.units
        )

        expected_kernel_shape = (
            cell.input_dim,
            3 * H,
        )

        expected_recurrent_shape = (
            H,
            3 * H,
        )

        expected_bias_shape = (
            3 * H,
        )

        if kernel.shape != expected_kernel_shape:
            raise RuntimeError(
                f"{gru_layer_name}: kernel shape mismatch. "
                f"expected={expected_kernel_shape} "
                f"actual={kernel.shape}"
            )

        if recurrent_kernel.shape != expected_recurrent_shape:
            raise RuntimeError(
                f"{gru_layer_name}: recurrent-kernel shape mismatch. "
                f"expected={expected_recurrent_shape} "
                f"actual={recurrent_kernel.shape}"
            )

        if bias.shape != expected_bias_shape:
            raise RuntimeError(
                f"{gru_layer_name}: bias shape mismatch. "
                f"reset_after=False requires {expected_bias_shape}, "
                f"actual={bias.shape}"
            )

        W_z = kernel[
            :,
            0:H,
        ]

        W_r = kernel[
            :,
            H:2 * H,
        ]

        W_h = kernel[
            :,
            2 * H:3 * H,
        ]

        U_z = recurrent_kernel[
            :,
            0:H,
        ]

        U_r = recurrent_kernel[
            :,
            H:2 * H,
        ]

        U_h = recurrent_kernel[
            :,
            2 * H:3 * H,
        ]

        b_z = bias[
            0:H
        ]

        b_r = bias[
            H:2 * H
        ]

        b_h = bias[
            2 * H:3 * H
        ]

        cell.W_z.assign(
            W_z
        )

        cell.W_r.assign(
            W_r
        )

        cell.W_h.assign(
            W_h
        )

        cell.U_z.assign(
            U_z
        )

        cell.U_r.assign(
            U_r
        )

        cell.U_h.assign(
            U_h
        )

        cell.b_z_inp.assign(
            b_z
        )

        cell.b_r_inp.assign(
            b_r
        )

        cell.b_h_inp.assign(
            b_h
        )

        pf(
            f"  OK {gru_layer_name}: "
            f"kernel={kernel.shape} "
            f"recurrent={recurrent_kernel.shape} "
            f"bias={bias.shape}"
        )

    unpack_and_set(
        "sencgru",
        enc_cell_p2,
    )

    unpack_and_set(
        "sdecgru",
        dec_cell_p2,
    )

    try:
        src_dense = float_student.get_layer(
            "sdec_dense"
        )

        dst_dense = phase2_model.get_layer(
            "sdec_dense"
        )

    except ValueError as exc:
        raise RuntimeError(
            "sdec_dense is missing during P1->P2 transfer"
        ) from exc

    src_weights = src_dense.get_weights()
    dst_weights = dst_dense.get_weights()

    if len(src_weights) < 2:
        raise RuntimeError(
            "Float sdec_dense does not expose kernel and bias"
        )

    if len(dst_weights) < 2:
        raise RuntimeError(
            "Phase-2 sdec_dense does not expose kernel and bias"
        )

    if src_weights[0].shape != dst_weights[0].shape:
        raise RuntimeError(
            "sdec_dense kernel shape mismatch: "
            f"src={src_weights[0].shape} "
            f"dst={dst_weights[0].shape}"
        )

    if src_weights[1].shape != dst_weights[1].shape:
        raise RuntimeError(
            "sdec_dense bias shape mismatch: "
            f"src={src_weights[1].shape} "
            f"dst={dst_weights[1].shape}"
        )

    replacement_weights = list(
        dst_weights
    )

    replacement_weights[0] = np.asarray(
        src_weights[0],
        dtype=np.float32,
    )

    replacement_weights[1] = np.asarray(
        src_weights[1],
        dtype=np.float32,
    )

    dst_dense.set_weights(
        replacement_weights
    )

    pf(
        "  OK sdec_dense"
    )

    sys.stdout.flush()

def run_equivalence_checks(
    float_student,
    phase2_model,
    enc_cell_p2,
    dec_cell_p2,
    final_qkeras_student,
    equiv_enc_sample,
    seq_len,
    pf,
    check_tag,
):
    """
    Numerical equivalence check between float_student and phase2_model
    (P1->P2) or phase2_model and final_qkeras_student (P2F->P3).

    IMPORTANT: runs on at most 512 samples drawn from equiv_enc_sample to
    avoid a GPU OOM when equiv_enc_sample is the full validation set
    (160 000 x T x 1). The original code passed the entire array to
    model.predict() with batch_size equal to the full array length, which
    allocated a ~16 GB contiguous tensor and triggered SIGABRT (exit 134).
    """
    MAX_EQUIV_SAMPLES = 512

    pf(f"[EQUIV {check_tag}] Running equivalence check...")
    sys.stdout.flush()

    n_available = equiv_enc_sample.shape[0]
    n_use = min(MAX_EQUIV_SAMPLES, n_available)
    sample = equiv_enc_sample[:n_use]
    dec_zeros = np.zeros((n_use, seq_len, 1), dtype=np.float32)

    try:
        if check_tag == "P1->P2":
            out_float = float_student.predict(
                [sample, dec_zeros], batch_size=n_use, verbose=0
            )
            out_p2_raw = phase2_model.predict(
                [sample, dec_zeros], batch_size=n_use, verbose=0
            )
            if isinstance(out_p2_raw, (list, tuple)):
                out_p2 = out_p2_raw[0]
            else:
                out_p2 = out_p2_raw
            max_diff = float(np.max(np.abs(out_float - out_p2)))
            mean_diff = float(np.mean(np.abs(out_float - out_p2)))
            pf(
                f"[EQUIV {check_tag}] float_student vs phase2_model "
                f"(n={n_use}): max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}"
            )
            if max_diff > 1e-3:
                pf(
                    f"[EQUIV {check_tag}] WARNING: max_diff={max_diff:.4e} > 1e-3. "
                    f"Weight transfer may be imperfect. Training will continue."
                )
            else:
                pf(f"[EQUIV {check_tag}] OK (max_diff < 1e-3)")

        elif check_tag == "P2F->P3":
            out_p2_raw = phase2_model.predict(
                [sample, dec_zeros], batch_size=n_use, verbose=0
            )
            if isinstance(out_p2_raw, (list, tuple)):
                out_p2 = out_p2_raw[0]
            else:
                out_p2 = out_p2_raw
            out_p3 = final_qkeras_student.predict(
                [sample, dec_zeros], batch_size=n_use, verbose=0
            )
            if isinstance(out_p3, (list, tuple)):
                out_p3 = out_p3[0]
            max_diff = float(np.max(np.abs(out_p2 - out_p3)))
            mean_diff = float(np.mean(np.abs(out_p2 - out_p3)))
            pf(
                f"[EQUIV {check_tag}] phase2_model vs final_qkeras_student "
                f"(n={n_use}): max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}"
            )
            if max_diff > 1e-2:
                pf(
                    f"[EQUIV {check_tag}] WARNING: max_diff={max_diff:.4e} > 1e-2. "
                    f"Export transfer may be lossy. Training will continue."
                )
            else:
                pf(f"[EQUIV {check_tag}] OK (max_diff < 1e-2)")

        else:
            pf(f"[EQUIV {check_tag}] Unknown check_tag — skipping.")

    except Exception as exc:
        pf(f"[EQUIV {check_tag}] ERROR during check: {exc}")
        pf(f"[EQUIV {check_tag}] Skipping equivalence check and continuing training.")

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

    # ── Quantizer scaling family (CONTROLLED-EXPERIMENT KNOB) ─────────────────
    # "1.0"      -> vanilla-identical fixed +/-1 scale (use this for the paper).
    # "auto_po2" -> learned per-tensor power-of-2 scale (raw-performance mode).
    # This single flag sets the alpha for kernel/recurrent/bias/dense quantizers
    # in BOTH the Phase 2 split-gate cell and the final QKeras student, so the
    # two graphs can never disagree and create a P2C->P3 quantisation cliff.
    p.add_argument("--quantizer-alpha", type=str, default="1.0",
                   choices=["1.0", "auto_po2"],
                   help="Weight quantizer scaling family for kernel/recurrent/"
                        "bias/dense in BOTH Phase 2 and the final QKeras student. "
                        "Use 1.0 for the vanilla-identical controlled experiment.")

    # ── Output loss type (CONTROLLED-EXPERIMENT KNOB) ─────────────────────────
    # "mse"     -> exact vanilla KD objective (plain MSE). Use for ladder rung A/B.
    # "huber_cn"-> channel-normalised Huber (original MemoQ). Confounds vs vanilla.
    p.add_argument("--output-loss", type=str, default="mse",
                   choices=["mse", "huber_cn"],
                   help="Base output KD loss. 'mse' reproduces vanilla exactly; "
                        "'huber_cn' uses channel-normalised Huber.")

    p.add_argument(
        "--memoq-warmup-epochs", "--memoq-stage1-epochs",
        dest="memoq_warmup_epochs", type=int, default=40,
    )
    p.add_argument("--memoq-stage2a-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2b-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2c-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2d-epochs",    type=int,   default=10,
                        help="P2D: all biases 4-bit stage epochs")
    p.add_argument("--memoq-stage2e-epochs",    type=int,   default=10,
                        help="P2E: activation quantizer stage epochs")
    p.add_argument("--memoq-stage2f-epochs",    type=int,   default=30,
                        help="P2F: state quantizer soft-blend stage epochs")
    p.add_argument("--memoq-state-anneal-epochs", type=int, default=15,
                        help="Number of P2F epochs over which beta ramps 0->1")
    p.add_argument("--memoq-lr-mult-p2d",       type=float, default=0.3,
                        help="LR multiplier for P2D stage")
    p.add_argument("--memoq-lr-mult-p2e",       type=float, default=0.3,
                        help="LR multiplier for P2E stage")
    p.add_argument("--memoq-lr-mult-p2f",       type=float, default=0.3,
                        help="LR multiplier for P2F state annealing stage")
    p.add_argument("--memoq-lambda-mem-p2d",    type=float, default=0.0)
    p.add_argument("--memoq-lambda-mem-p2e",    type=float, default=0.0)
    p.add_argument("--memoq-lambda-mem-p2f",    type=float, default=0.1)
    p.add_argument("--memoq-lambda-innov-p2d",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p2e",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p2f",  type=float, default=0.05)
    p.add_argument("--memoq-lambda-zsat-p2d",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p2e",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p2f",   type=float, default=0.01)
    p.add_argument("--memoq-lambda-rail-p2d",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p2e",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p2f",   type=float, default=0.01)
    p.add_argument("--memoq-lambda-shape-p2d",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p2e",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p2f",  type=float, default=0.0)
    p.add_argument(
        "--memoq-finetune-epochs", "--memoq-stage3-epochs",
        dest="memoq_stage3_epochs", type=int, default=170,
    )

    # ── Per-phase auxiliary loss weights (ALL ARGS — set 0.0 to disable) ──────
    # Each phase has its own lambda for every auxiliary loss. The experimental
    # ladder is run purely by setting these on the command line; no code edits.
    #   ladder A/B (vanilla / control): leave every aux lambda at 0.0
    #   ladder D: set memX > 0 ; ladder E: add innovX ; ladder F: add zsat/rail/shape
    # P2A
    p.add_argument("--memoq-lambda-mem-p2a",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p2a", type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p2a",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p2a",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p2a", type=float, default=0.0)
    # P2B
    p.add_argument("--memoq-lambda-mem-p2b",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p2b", type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p2b",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p2b",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p2b", type=float, default=0.0)
    # P2C
    p.add_argument("--memoq-lambda-mem-p2c",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p2c", type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p2c",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p2c",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p2c", type=float, default=0.0)
    # P3
    p.add_argument("--memoq-lambda-mem-p3",    type=float, default=0.0)
    p.add_argument("--memoq-lambda-innov-p3",  type=float, default=0.0)
    p.add_argument("--memoq-lambda-zsat-p3",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-rail-p3",   type=float, default=0.0)
    p.add_argument("--memoq-lambda-shape-p3",  type=float, default=0.0)

    # ── Gate-curriculum toggle (ARG) ──────────────────────────────────────────
    # When False, Phase 2 quantises all gates simultaneously at P2A (no causal
    # h->r->z curriculum) — this is ladder rung B (MemoQ-control). When True,
    # gates harden one at a time (rung C onward).
    p.add_argument("--memoq-gate-curriculum", type=lambda s: s.lower() in ("1", "true", "yes", "y"),
                   default=True,
                   help="True: causal h->r->z gate hardening (rung C+). "
                        "False: all gates 4-bit from P2A (rung B control).")

    # ── Innovation burn-in (ARG) ──────────────────────────────────────────────
    p.add_argument("--memoq-innov-burnin",   type=int,   default=5)

    # ── Loss-shape detail args ────────────────────────────────────────────────
    p.add_argument("--memoq-huber-delta", type=float, default=0.1)
    p.add_argument("--memoq-rho-z",       type=float, default=0.98)
    p.add_argument("--memoq-rho-rail",    type=float, default=0.97)
    p.add_argument("--memoq-mu-rail",     type=float, default=0.0)

    # ── Per-phase LR multipliers (ARGS) ───────────────────────────────────────
    # Phase LR = effective_lr * multiplier (Phase 3 additionally floored, see below).
    p.add_argument("--memoq-lr-mult-p1",  type=float, default=1.0)
    p.add_argument("--memoq-lr-mult-p2a", type=float, default=0.5)
    p.add_argument("--memoq-lr-mult-p2b", type=float, default=0.3)
    p.add_argument("--memoq-lr-mult-p2c", type=float, default=0.2)
    p.add_argument("--memoq-lr-mult-p3",  type=float, default=0.1)

    p.add_argument("--memoq-phase3-lr-floor", type=float, default=1e-5,
                   help="Lower floor on the Phase 3 LR so the hard graph has "
                        "enough gradient signal to recover from the transfer.")

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
# Accepted for CLI compatibility but UNUSED — there is no gradient
    # accumulation in this pipeline. Kept so existing sbatch scripts don't break.
    p.add_argument("--accumulation-steps", type=int, default=1,
                   help="UNUSED — accepted for CLI compatibility only.")
    p.add_argument("--resume",            action="store_true", default=False)

    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = args.data_dir

    args.memoq_stage1_epochs   = args.memoq_warmup_epochs
    args.memoq_finetune_epochs = args.memoq_stage3_epochs

    # ── LR + patience scaling — FIXED DIRECTION (task-doc item 5) ─────────────
    # Vanilla MULTIPLIES patience by the batch ratio. The old MemoQ DIVIDED it,
    # so at batch 16384 / ref 1024 the scheduler dropped LR after ~1 epoch and
    # crippled training. Now patience and warmup scale UP with batch size,
    # exactly like vanilla.
    scale = float(args.batch_size) / float(args.ref_batch_size)
    if args.no_lr_scaling:
        args.effective_lr             = args.lr
        args.effective_lr_patience    = args.lr_patience
        args.effective_warmup_epochs  = args.warmup_epochs
    else:
        args.effective_lr             = args.lr * scale
        args.effective_lr_patience    = max(1, int(round(args.lr_patience   * scale)))
        args.effective_warmup_epochs  = max(0, int(round(args.warmup_epochs * scale)))

    # Resolve the quantizer alpha into the literal value qkeras expects.
    args.q_alpha = 1.0 if args.quantizer_alpha == "1.0" else "auto_po2"

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

def build_float_student(
    seq_len,
    n_out,
    student_units,
):
    """
    Phase-1 float student using the same GRU recurrence convention as the
    vanilla QKeras QGRU, but without low-bit parameter/state/candidate
    quantization.

    Exact recurrent semantics:
        recurrent_activation = QKeras hard_sigmoid
        implementation       = 1
        reset_after           = False
        candidate activation  = float tanh
    """

    enc_inputs = keras.layers.Input(
        shape=(None, 1),
        name="senc_input",
    )

    dec_inputs = keras.layers.Input(
        shape=(None, 1),
        name="sdec_input",
    )

    enc_out, enc_state = keras.layers.GRU(
        units=student_units,
        activation="tanh",
        recurrent_activation=qkeras_hard_sigmoid,
        implementation=1,
        return_state=True,
        reset_after=False,
        name="sencgru",
    )(
        enc_inputs
    )

    dec_hid_seq, _ = keras.layers.GRU(
        units=student_units,
        activation="tanh",
        recurrent_activation=qkeras_hard_sigmoid,
        implementation=1,
        return_sequences=True,
        return_state=True,
        reset_after=False,
        name="sdecgru",
    )(
        dec_inputs,
        initial_state=enc_state,
    )

    s_output = keras.layers.Dense(
        n_out,
        activation="linear",
        name="sdec_dense",
    )(
        dec_hid_seq
    )

    return keras.models.Model(
        inputs=[
            enc_inputs,
            dec_inputs,
        ],
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

    normalized_input = np.load(
        file_input,
        mmap_mode="r",
    )

    res_data = np.load(
        file_res,
        mmap_mode="r",
    )

    train_idx = np.load(
        file_train
    )

    val_idx = np.load(
        file_val
    )

    test_idx = np.load(
        file_test
    )

    n_total = int(
        normalized_input.shape[0]
    )

    if normalized_input.ndim != 3:
        raise RuntimeError(
            f"Expected encoder input rank 3, got shape "
            f"{normalized_input.shape}"
        )

    if normalized_input.shape[1] != args.seq_len:
        raise RuntimeError(
            f"Expected seq_len={args.seq_len}, "
            f"got encoder shape={normalized_input.shape}"
        )

    if normalized_input.shape[2] != 1:
        raise RuntimeError(
            f"Expected encoder input_dim=1, "
            f"got encoder shape={normalized_input.shape}"
        )

    pf(
        f"[MAIN] n_total={n_total} "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    pf(
        "[MAIN] Encoder preprocessing matches vanilla: "
        "using tpsf_seq input directly with no additional MemoQ z-score."
    )

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
    matched_cache_dir = os.path.join(
        args.data_dir,
        "memoq_qkeras_matched_cache",
    )

    os.makedirs(
        matched_cache_dir,
        exist_ok=True,
    )

    pf(
        f"[MAIN] QKeras-matched teacher cache directory: "
        f"{matched_cache_dir}"
    )
    teacher_predictions, _ = cache_teacher_predictions_and_hidden(
        teacher_model=teacher_model,
        teacher_hidden_model=teacher_hidden_model,
        normalized_input=normalized_input,
        seq_len=args.seq_len,
        n_out=args.n_out,
        teacher_units=args.teacher_units,
        n_samples=n_total,
        infer_batch=args.infer_batch,
        data_dir=matched_cache_dir,
        pf=pf,
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
            q_alpha       = args.q_alpha,
            bits_kernel   = args.bits_kernel,
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
            q_alpha         = args.q_alpha,
        )
    # Attach live teacher hidden model for Phase 3 loss computation
    final_qkeras_student._teacher_hidden_model = teacher_hidden_model
    final_qkeras_student.summary(print_fn=pf)
    log_final_student_config(final_qkeras_student, pf)
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
            elif phase_tag in ("P2A", "P2B", "P2C", "P2D", "P2E", "P2F"):
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
        equiv_enc_sample    = enc_va,
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