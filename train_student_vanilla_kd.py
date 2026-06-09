#!/usr/bin/env python3
"""
train_student_vanilla_kd.py — Vanilla Knowledge Distillation Student Training

Architecture: Quantized GRU Seq2Seq (same as fw-qatd-rac student).
Loss: total = alpha * MSE(student_output || teacher_output)
              + (1 - alpha) * MSE(student_output || hard_target)

Advanced infrastructure (same as fw-qatd-rac):
  - CUDA_VISIBLE_DEVICES set before TF import if not already in env
  - Explicit MirroredStrategy device list (never auto-guess)
  - Pre-flight GPU timing check
  - Teacher prediction cache: compute once, checkpoint-guarded, mmap reload
  - @tf.function placed on distributed wrappers only (not per-replica fns)
  - Loss scaling: MEAN reduce — no manual local/global scaling
  - NaN guard: skip apply_gradients, count, warn if >10% of batches
  - ReduceLROnPlateau and EarlyStopping wired in manual loop
  - find_data_files: glob-based discovery + both index filename conventions
  - Pre-saved split indices (trainidx.npy / validx.npy / testidx.npy)
  - normalised input loaded via mmap (same as fw-qatd-rac main)
  - materialise_enc_tgt_tpred into contiguous RAM buffers
  - Hybrid tf.data: from_tensor_slices (GIL-free) for materialised arrays
  - strategy.experimental_distribute_dataset for correct shard distribution
  - bar() progress bar with ETA
  - evaluate_and_save: extract_lifetimes, compute_metrics, hexbin scatters
  - save_loss_curves: multi-panel PNG + CSV
  - Linear LR scaling: lr is auto-scaled by (batch_size / ref_batch_size)
    so large-batch runs match small-batch convergence quality.
    lr_patience and warmup_epochs are also scaled proportionally.

NOT included (fw-qatd-rac specific):
  - Fisher diagonal computation
  - Teacher hidden trajectory cache
  - Float shadow student / RAC loss
  - Projection layer (student -> teacher space)
  - Trajectory distillation loss

Usage example:
  python train_student_vanilla_kd.py \\
      --data-dir /gpfs/.../nmi \\
      --teacher-ckpt /gpfs/.../nmi/teacher_best.weights.h5 \\
      --save-dir /gpfs/.../runs \\
      --bits-kernel 4 --bits-bias 4 --bits-recurrent 4 \\
      --bits-activation 4 --bits-state 4 \\
      --student-units 32 --teacher-units 128 --teacher-layers 2 \\
      --seq-len 135 --n-out 3 --gate-width-ns 0.09 \\
      --batch-size 16384 --epochs 300 --patience 15 \\
      --lr 1e-4 --ref-batch-size 1024 \\
      --lr-factor 0.5 --lr-patience 8 --lr-min 1e-6 \\
      --temperature 4.0 --alpha 0.7 \\
      --log-interval 10 --infer-batch 8192 \\
      --prefetch-batches 32 --pipeline-workers 4 \\
      --split-seed 42

  With --batch-size 16384 and --ref-batch-size 1024 the effective LR becomes
  1e-4 * (16384 / 1024) = 1.6e-3, lr_patience becomes 8 * (16384/1024) = 128,
  and warmup_epochs becomes 5 * (16384/1024) = 80.
  Pass --no-lr-scaling to disable automatic scaling and use --lr as-is.

Outputs (all inside --save-dir / results / job_name /):
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
"""

import argparse
import glob
import json
import os
import sys
import time

# ============================================================
# STEP 1 — Force all 8 GPUs visible to CUDA/TF BEFORE any
# TensorFlow import. If CUDA_VISIBLE_DEVICES is already set
# by SLURM we honour it; otherwise we expose all 8 slots.
# This MUST happen BEFORE `import tensorflow`.
# ============================================================
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(
        "[STEP 1] CUDA_VISIBLE_DEVICES not set — defaulting to 0,1,2,3,4,5,6,7",
        flush=True,
    )
else:
    print(
        f"[STEP 1] CUDA_VISIBLE_DEVICES already set: "
        f"{os.environ['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from fastdtw import fastdtw

from scipy.spatial.distance import euclidean
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Vanilla KD student training for PRISMAI QGRU Seq2Seq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument("--data-dir",  type=str, required=True,
                   help="Directory containing all .npy data files.")
    p.add_argument("--save-dir",  type=str, default=None,
                   help="Root directory for output subfolder. Defaults to --data-dir.")
    p.add_argument("--seq-len",   type=int, default=135,
                   help="Sequence length (number of time bins). SS3 = 135.")
    p.add_argument("--n-out",     type=int, default=3,
                   help="Number of output channels.")
    p.add_argument("--gate-width-ns", type=float, default=0.09,
                   help="Gate width in ns per time bin (SS3 = 0.09 ns / 90 ps).")

    # ── Teacher ───────────────────────────────────────────────────────────────
    p.add_argument("--teacher-ckpt",    type=str, required=True,
                   help="Path to teacher .weights.h5 checkpoint.")
    p.add_argument("--teacher-units",   type=int, default=128,
                   help="Hidden units of each teacher GRU layer.")
    p.add_argument("--teacher-layers",  type=int, default=2,
                   help="Number of stacked GRUCell layers in teacher.")

    # ── KD hyper-parameters ───────────────────────────────────────────────────
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Softening temperature T.")
    p.add_argument("--alpha",       type=float, default=0.7,
                   help="KD loss weight. total = alpha*T^2*KL + (1-alpha)*MSE.")

    # ── Quantisation ──────────────────────────────────────────────────────────
    p.add_argument("--bits-kernel",     type=int, default=4,
                   help="Bits for kernel weights in QGRU and QDense.")
    p.add_argument("--bits-bias",       type=int, default=4,
                   help="Bits for bias terms in QGRU and QDense.")
    p.add_argument("--bits-recurrent",  type=int, default=4,
                   help="Bits for recurrent weights in QGRU.")
    p.add_argument("--bits-activation", type=int, default=4,
                   help="Bits for quantized_tanh activation in QGRU.")
    p.add_argument("--bits-state",      type=int, default=4,
                   help="Bits for GRU hidden state quantization.")

    # ── Architecture ──────────────────────────────────────────────────────────
    p.add_argument("--student-units", type=int, default=32,
                   help="QGRU hidden units in student encoder and decoder.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--batch-size",        type=int,   default=1024,
                   help="Global batch size across all GPUs.")
    p.add_argument("--epochs",            type=int,   default=300)
    p.add_argument("--lr",                type=float, default=1e-4,
                   help=(
                       "Base learning rate at --ref-batch-size. "
                       "Automatically scaled by (batch_size / ref_batch_size) "
                       "unless --no-lr-scaling is set."
                   ))
    p.add_argument("--ref-batch-size",    type=int,   default=1024,
                   help=(
                       "Reference batch size for linear LR scaling. "
                       "The effective LR = lr * (batch_size / ref_batch_size). "
                       "Set equal to --batch-size to disable scaling, "
                       "or use --no-lr-scaling."
                   ))
    p.add_argument("--no-lr-scaling",     action="store_true", default=False,
                   help=(
                       "Disable automatic linear LR / lr_patience / warmup_epochs "
                       "scaling based on batch size ratio. Use --lr exactly as given."
                   ))
    p.add_argument("--lr-factor",         type=float, default=0.5,
                   help="LR reduction factor on plateau.")
    p.add_argument("--lr-patience",       type=int,   default=8,
                   help=(
                       "Epochs without improvement before LR reduction "
                       "(at ref_batch_size=1024). Scaled up proportionally "
                       "when batch_size > ref_batch_size unless --no-lr-scaling."
                   ))
    p.add_argument("--lr-min",            type=float, default=1e-6,
                   help="Minimum learning rate floor.")
    p.add_argument("--patience",          type=int,   default=15,
                   help="EarlyStopping patience (epochs without val_loss improvement).")
    p.add_argument("--min-delta",         type=float, default=1e-5,
                   help="Minimum val loss improvement to reset patience.")
    p.add_argument("--infer-batch",       type=int,   default=8192,
                   help="Batch size for teacher cache inference and test evaluation.")
    p.add_argument("--mixed-precision",   action="store_true", default=False,
                   help="Enable float16 mixed precision training.")
    p.add_argument("--log-interval",      type=int,   default=10,
                   help="Print progress bar every N steps.")
    p.add_argument("--prefetch-batches",  type=int,   default=32,
                   help="Number of batches to prefetch in tf.data pipeline.")
    p.add_argument("--pipeline-workers",  type=int,   default=4,
                   help="num_parallel_calls for tf.data map().")
    p.add_argument("--split-seed",        type=int,   default=42,
                   help="RNG seed passed to tf.keras.utils.set_random_seed.")
    # --- warmup ---
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help=(
                       "Number of linear LR warmup epochs at ref_batch_size=1024. "
                       "Automatically scaled up by (batch_size / ref_batch_size) "
                       "unless --no-lr-scaling is set, so warmup covers the same "
                       "number of gradient updates regardless of batch size. "
                       "Set to 0 to disable warmup entirely."
                   ))
    p.add_argument("--accumulation-steps", type=int, default=1,
                   help=(
                       "Gradient accumulation steps. Effective batch size = "
                       "batch_size * accumulation_steps. Set to 16 when using "
                       "batch_size=16384 to match update frequency of batch_size=1024. "
                       "Each step processes batch_size / accumulation_steps samples "
                       "before calling apply_gradients."
                   ))
    p.add_argument("--resume", action="store_true", default=False,
                   help=(
                       "Resume training from student_best.weights.h5 + "
                       "resume_state.json in the job output directory. "
                       "Restores epoch counter, best_val, patience counter, LR, "
                       "and full loss history so training continues exactly "
                       "where it left off."
                   ))

    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = args.data_dir
    return args


# ==============================================================================
# Job naming / output directory
# ==============================================================================

def make_job_name(args) -> str:
    effective_batch = args.batch_size
    micro_batch     = args.batch_size // args.accumulation_steps
    return (
        f"vanilla_kd"
        f"_T{args.temperature}"
        f"_a{args.alpha}"
        f"_b{args.bits_kernel}k{args.bits_bias}r{args.bits_recurrent}a{args.bits_activation}"
        f"_gru{args.student_units}x1"
        f"_dense{args.n_out}"
        f"_effbs{effective_batch}"
        f"_microbs{micro_batch}"
        f"_lr{args.effective_lr:.0e}"
    )

# ==============================================================================
# GPU / Strategy setup — explicit device list, memory growth, mixed precision.
# mixed_precision policy is set HERE ONLY — not duplicated in main().
# set_memory_growth MUST be called before ANY other TF GPU operation.
# ==============================================================================

def setup_gpus_and_strategy(mixed_precision: bool):
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[STEP 2] No physical GPUs found — running on CPU.", flush=True)
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(
                f"[STEP 2]   WARNING set_memory_growth failed for {gpu.name}: {e}",
                flush=True,
            )

    print(f"[STEP 2] Physical GPUs detected : {len(physical_gpus)}", flush=True)
    for i, g in enumerate(physical_gpus):
        print(f"[STEP 2]   GPU {i}: {g.name}", flush=True)

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[STEP 2] Logical GPUs visible   : {len(logical_gpus)}", flush=True)

    if len(logical_gpus) < len(physical_gpus):
        print(
            f"[STEP 2] WARNING: only {len(logical_gpus)} logical GPUs from "
            f"{len(physical_gpus)} physical. "
            f"Check CUDA_VISIBLE_DEVICES and SLURM --gres=gpu:N",
            flush=True,
        )

    if len(logical_gpus) == 0:
        print("[STEP 2] No logical GPUs available — falling back to CPU.", flush=True)
        return tf.distribute.get_strategy()

    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[STEP 2] Mixed-precision policy: mixed_float16", flush=True)
    else:
        keras.mixed_precision.set_global_policy("float32")

    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    print(
        f"[STEP 2] MirroredStrategy: {strategy.num_replicas_in_sync} replicas  "
        f"devices={gpu_devices}",
        flush=True,
    )

    if strategy.num_replicas_in_sync == 1 and len(logical_gpus) > 1:
        print(
            "[STEP 2] WARNING: MirroredStrategy sees only 1 replica despite "
            f"{len(logical_gpus)} logical GPUs. "
            "Set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 in your SLURM script "
            "BEFORE calling python.",
            flush=True,
        )

    return strategy


# ==============================================================================
# File discovery — glob-based, supports both index filename conventions
# ==============================================================================

def find_data_files(data_dir, seq_len):
    def find_one(pattern_globs, desc):
        for pat in pattern_globs:
            matches = glob.glob(os.path.join(data_dir, pat))
            if matches:
                return sorted(matches)[0]
        raise FileNotFoundError(
            f"Cannot find {desc} in {data_dir}. Tried patterns: {pattern_globs}"
        )

    file_input = find_one(
        [f"tpsf_seq_L{seq_len}_*.npy"],
        "encoder input (tpsf_seq)",
    )
    file_res = find_one(
        [f"res_L{seq_len}_*.npy"],
        "decoder target (res)",
    )
    file_labels = find_one(
        [f"labels_3ch_L{seq_len}_*.npy"],
        "labels (labels_3ch)",
    )

    file_train = None
    for name in ["trainidx.npy", "train_idx.npy"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            file_train = candidate
            break
    if file_train is None:
        raise FileNotFoundError(
            f"Train split index not found in {data_dir}. "
            f"Tried: trainidx.npy, train_idx.npy"
        )

    file_val = None
    for name in ["validx.npy", "val_idx.npy"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            file_val = candidate
            break
    if file_val is None:
        raise FileNotFoundError(
            f"Val split index not found in {data_dir}. "
            f"Tried: validx.npy, val_idx.npy"
        )

    file_test = None
    for name in ["testidx.npy", "test_idx.npy"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            file_test = candidate
            break
    if file_test is None:
        raise FileNotFoundError(
            f"Test split index not found in {data_dir}. "
            f"Tried: testidx.npy, test_idx.npy"
        )

    return file_input, file_res, file_labels, file_train, file_val, file_test


# ==============================================================================
# Teacher model — EXACT replica of train_teacher.py / fw-qatd-rac
# Stacked GRUCell inside keras.layers.RNN
# Layer names: enc_input, dec_input, enc_rnn, dec_rnn, dec_dense
# ==============================================================================

def build_teacher(seq_len, n_out, teacher_units, teacher_layers):
    LAYERS_TEACHER = [teacher_units] * teacher_layers

    encoder_inputs = keras.layers.Input(shape=(None, 1), name="enc_input")
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells,
        return_state=True,
        name="enc_rnn",
    )
    encoder_outputs_and_states = encoder_rnn(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

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
    decoder_outputs_and_states = decoder_rnn(
        decoder_inputs, initial_state=encoder_states
    )
    decoder_hidden_sequence = decoder_outputs_and_states[0]

    decoder_dense = keras.layers.Dense(n_out, activation="linear", name="dec_dense")
    decoder_output = decoder_dense(decoder_hidden_sequence)

    teacher_model = keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_output,
        name="teacher_seq2seq",
    )
    return teacher_model


# ==============================================================================
# Student model — QGRU encoder + QGRU decoder + QDense head
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense
# Same topology as fw-qatd-rac student.
# ==============================================================================

def build_student(
    seq_len,
    n_out,
    student_units,
    bits_kernel,
    bits_recurrent,
    bits_bias,
    bits_activation,
    bits_state,
):
    def qwk():
        return quantized_bits(bits_kernel, 0, 1, alpha=1.0)

    def qwr():
        return quantized_bits(bits_recurrent, 0, 1, alpha=1.0)

    def qwb():
        return quantized_bits(bits_bias, 0, 1, alpha=1.0)

    def qa():
        return quantized_tanh(bits=bits_activation, symmetric=True)

    def qs():
        return quantized_bits(bits_state, 0, 1, alpha=1.0)

    def qd():
        return quantized_bits(bits_kernel, 0)

    enc_inputs = Input(shape=(None, 1), name="senc_input")
    dec_inputs = Input(shape=(None, 1), name="sdec_input")

    s_enc_out, s_enc_state = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(enc_inputs)

    s_dec_hid_seq, _ = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        name="sdecgru",
    )(dec_inputs, initial_state=s_enc_state)

    s_output = QDense(
        n_out,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(s_dec_hid_seq)

    student_model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="student_vanilla_kd",
    )
    return student_model


# ==============================================================================
# Teacher prediction cache — compute once, checkpoint-guarded, mmap reload.
# Only teacher predictions (not hidden states) are needed for vanilla KD.
# ==============================================================================

def cache_teacher_predictions(
    teacher_model,
    normalized_input,
    seq_len,
    n_out,
    n_samples,
    infer_batch,
    data_dir,
    pf,
):
    file_pred = os.path.join(
        data_dir, f"teacherPred_vanillaKD_L{seq_len}{n_samples}.npy"
    )

    if os.path.exists(file_pred):
        pf(f"[CACHE] Teacher prediction cache found — loading mmap:")
        pf(f"  {file_pred}")
        sys.stdout.flush()
        teacher_predictions = np.load(file_pred, mmap_mode="r")
        pf(
            f"[CACHE] teacher_predictions : {teacher_predictions.shape}  "
            f"dtype={teacher_predictions.dtype}"
        )
        sys.stdout.flush()
        return teacher_predictions

    pf("=" * 60)
    pf("[CACHE] Teacher prediction cache NOT found — running full-dataset inference...")
    pf(f"  This runs ONCE and saves results to disk.")
    pf(f"  pred → {file_pred}")
    pf("=" * 60)
    sys.stdout.flush()

    pf(f"[CACHE] Opening mmap for writing: shape=({n_samples}, {seq_len}, {n_out})  dtype=float32")
    sys.stdout.flush()

    teacher_predictions = np.lib.format.open_memmap(
        file_pred, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, n_out),
    )

    pf("[CACHE] mmap opened OK.")
    sys.stdout.flush()

    @tf.function(reduce_retracing=True)
    def teacher_forward(enc_b, dec_b):
        return teacher_model([enc_b, dec_b], training=False)

    pf("[CACHE] Starting teacher_forward warm-up trace...")
    sys.stdout.flush()
    wu_size = min(infer_batch, n_samples)
    enc_wu  = tf.constant(normalized_input[:wu_size], dtype=tf.float32)
    dec_wu  = tf.zeros((wu_size, seq_len, 1), dtype=tf.float32)
    _p      = teacher_forward(enc_wu, dec_wu)
    _       = _p.numpy()
    pf("[CACHE] Warm-up trace compiled and executed.")
    del enc_wu, dec_wu, _p
    sys.stdout.flush()

    n_batches   = int(np.ceil(n_samples / infer_batch))
    print_every = max(1, n_batches // 20)

    pf(
        f"[CACHE] Starting inference loop: {n_batches} batches  "
        f"infer_batch={infer_batch}  n_samples={n_samples:,}"
    )
    sys.stdout.flush()

    t0     = time.time()
    t_last = t0

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_samples)

        enc_b = tf.constant(normalized_input[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)

        pred = teacher_forward(enc_b, dec_b)
        teacher_predictions[s:e] = pred.numpy()
        teacher_predictions.flush()

        del enc_b, dec_b, pred

        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed   = time.time() - t0
            step_time = time.time() - t_last
            pct       = 100.0 * e / n_samples
            eta_s     = (elapsed / max(b + 1, 1)) * (n_batches - b - 1)
            pf(
                f"[CACHE] batch {b + 1:>4d}/{n_batches}  "
                f"samples {e:>9,}/{n_samples:,}  "
                f"({pct:5.1f}%)  "
                f"elapsed={elapsed / 60:.1f}min  "
                f"ETA={eta_s / 60:.1f}min  "
                f"step={step_time:.1f}s"
            )
            t_last = time.time()
            sys.stdout.flush()

    pf("[CACHE] Inference loop complete — flushing mmap...")
    sys.stdout.flush()
    teacher_predictions.flush()
    del teacher_predictions

    elapsed = time.time() - t0
    pf(
        f"[CACHE] Done in {elapsed / 60:.1f} min  "
        f"({n_samples / elapsed:.0f} samples/s)"
    )
    pf(f"[CACHE] Re-opening cache read-only (mmap)...")
    sys.stdout.flush()

    teacher_predictions = np.load(file_pred, mmap_mode="r")
    pf(
        f"[CACHE]   teacher_predictions : {teacher_predictions.shape}  "
        f"dtype={teacher_predictions.dtype}"
    )
    sys.stdout.flush()
    return teacher_predictions


# ==============================================================================
# Materialise split arrays into contiguous float32 RAM buffers.
# enc, tgt, tpred materialised — all three are small enough.
# RAM cost for train split (~80% of 8M):
#   enc   : 6.4M * 135 * 1 * 4B  = ~3.46 GB
#   tgt   : 6.4M * 135 * 3 * 4B  = ~10.37 GB
#   tpred : 6.4M * 135 * 3 * 4B  = ~10.37 GB
#   Total train+val               = ~36 GB
# ==============================================================================

def materialise_enc_tgt_tpred(
    normalized_input,
    res,
    teacher_predictions,
    idx,
    seq_len,
    n_out,
    label,
    pf,
):
    n = len(idx)
    pf(f"  Materialising {label} enc/tgt/tpred ({n:,} samples) into RAM...")
    t0 = time.time()

    enc   = np.empty((n, seq_len, 1),     dtype=np.float32)
    tgt   = np.empty((n, seq_len, n_out), dtype=np.float32)
    tpred = np.empty((n, seq_len, n_out), dtype=np.float32)

    chunk = 65536
    for s in range(0, n, chunk):
        e          = min(s + chunk, n)
        enc[s:e]   = normalized_input[idx[s:e]]
        tgt[s:e]   = res[idx[s:e]]
        tpred[s:e] = teacher_predictions[idx[s:e]]

    pf(
        f"  Done in {time.time() - t0:.1f}s  "
        f"enc={enc.nbytes / 1e9:.2f} GB  "
        f"tgt={tgt.nbytes / 1e9:.2f} GB  "
        f"tpred={tpred.nbytes / 1e9:.2f} GB"
    )
    return enc, tgt, tpred


# ==============================================================================
# tf.data pipeline — pure from_tensor_slices, no py_function, fully GIL-free.
# All three arrays (enc, tgt, tpred) are already materialised in RAM.
# Decoder input = zeros (same shape as enc), built here.
# ==============================================================================
def make_kd_dataset(
    enc_arr,
    tgt_arr,
    tpred_arr,
    batch_size,
    accumulation_steps,
    seq_len,
    n_out,
    shuffle,
    seed,
    prefetch_batches,
    pipeline_workers,
):
    n = len(enc_arr)
    dec_arr = np.zeros_like(enc_arr)

    micro_batch_size = batch_size // accumulation_steps

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

    ds = ds.batch(micro_batch_size, drop_remainder=True)

    def set_shapes(enc_b, dec_b, tpred_b, tgt_b):
        enc_b.set_shape([micro_batch_size, seq_len, 1])
        dec_b.set_shape([micro_batch_size, seq_len, 1])
        tpred_b.set_shape([micro_batch_size, seq_len, n_out])
        tgt_b.set_shape([micro_batch_size, seq_len, n_out])
        batchx = {
            "enc_input": enc_b,
            "dec_input": dec_b,
            "tpred":     tpred_b,
        }
        return batchx, tgt_b

    ds = ds.map(set_shapes, num_parallel_calls=pipeline_workers)
    ds = ds.prefetch(prefetch_batches)
    return ds


def mse_kd_loss(y_teacher, y_student):
    return tf.reduce_mean(tf.square(y_teacher - y_student))

# ==============================================================================
# Per-replica train step — NO @tf.function here.
# @tf.function is placed ONLY on the distributed wrapper below.
# Placing @tf.function here prevents batch sharding in TF 2.10.
# nan_in_grads returned as tf.float32 for explicit cast in distributed wrapper.
# ==============================================================================

def train_step_per_replica(
    batch_x,
    batch_y,
    student_model,
    optimizer,
    temperature,
    alpha,
):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    alpha_f = tf.cast(alpha, tf.float32)

    with tf.GradientTape() as tape:
        student_output = student_model([enc_b, dec_b], training=True)
        hard_loss  = tf.reduce_mean(tf.square(student_output - tgt_b))
        soft_loss  = tf.reduce_mean(tf.square(tpred_b - student_output))
        total_loss = alpha_f * soft_loss + (1.0 - alpha_f) * hard_loss

    grads = tape.gradient(total_loss, student_model.trainable_variables)
    grads = [
        tf.zeros_like(v) if g is None else g
        for g, v in zip(grads, student_model.trainable_variables)
    ]

    nan_in_grads = tf.reduce_any(tf.stack([
        tf.reduce_any(tf.math.is_nan(g)) for g in grads
    ]))

    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, student_model.trainable_variables))

    return (
        total_loss,
        hard_loss,
        soft_loss,
        tf.cast(nan_in_grads, tf.float32),
    )

# ==============================================================================
# Per-replica val step — NO @tf.function here (same reason as train step).
# ==============================================================================

def val_step_per_replica(
    batch_x,
    batch_y,
    student_model,
    temperature,
    alpha,
):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    alpha_f = tf.cast(alpha, tf.float32)

    student_output = student_model([enc_b, dec_b], training=False)

    hard_loss  = tf.reduce_mean(tf.square(student_output - tgt_b))
    soft_loss  = mse_kd_loss(tpred_b, student_output)
    total_loss = alpha_f * soft_loss + (1.0 - alpha_f) * hard_loss
    mae        = tf.reduce_mean(tf.abs(student_output - tgt_b))

    return total_loss, hard_loss, soft_loss, mae

# ==============================================================================
# Distributed wrappers — @tf.function placed HERE ONLY on the function
# that calls strategy.run(). This is the correct TF 2.10 pattern.
# strategy.reduce uses MEAN — no manual local/global loss scaling needed.
# ==============================================================================

def make_distributed_train_step(strategy, student_model, optimizer, temperature, alpha):
    @tf.function
    def distributed_train_step(batch_x, batch_y):
        per_replica = strategy.run(
            train_step_per_replica,
            args=(batch_x, batch_y, student_model, optimizer, temperature, alpha),
        )
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None
        )
        hard_loss  = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None
        )
        soft_loss  = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None
        )
        nan_flag   = strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica[3],
            axis=None,
        )
        return total_loss, hard_loss, soft_loss, nan_flag > 0.0
    return distributed_train_step

def make_distributed_val_step(strategy, student_model, temperature, alpha):
    @tf.function
    def distributed_val_step(batch_x, batch_y):
        per_replica = strategy.run(
            val_step_per_replica,
            args=(batch_x, batch_y, student_model, temperature, alpha),
        )
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None
        )
        hard_loss  = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None
        )
        soft_loss  = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None
        )
        mae        = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None
        )
        return total_loss, hard_loss, soft_loss, mae
    return distributed_val_step


# ==============================================================================
# ReduceLROnPlateau (manual — compatible with MirroredStrategy)
# Handles both plain float and callable LR schedule.
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
            current_lr_val = float(
                optimizer.learning_rate(optimizer.iterations)
            )
        else:
            current_lr_val = float(
                tf.keras.backend.get_value(optimizer.learning_rate)
            )

        self.lr_var = tf.Variable(current_lr_val, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = self.lr_var

    @property
    def current_lr(self):
        return float(tf.keras.backend.get_value(self.lr_var))

    def step(self, val_loss, epoch, pfn):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            old_lr = float(tf.keras.backend.get_value(self.lr_var))
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                self.lr_var.assign(new_lr)
                pfn(f"ReduceLR  {old_lr:.2e} → {new_lr:.2e}  (epoch {epoch + 1})")
            self.wait = 0
            return True
        return False


# ==============================================================================
# Progress bar with ETA
# ==============================================================================

def bar(step, total, metrics: dict, epoch_start_time: float, width=28):
    frac   = step / max(total, 1)
    filled = int(width * frac)
    b      = "█" * filled + "░" * (width - filled)
    stats  = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())

    elapsed = time.time() - epoch_start_time
    if step > 0:
        secs_per_step = elapsed / step
        remaining     = secs_per_step * (total - step)
        if remaining >= 3600:
            eta_str = f"{remaining / 3600:.1f}h"
        elif remaining >= 60:
            eta_str = f"{remaining / 60:.0f}m{int(remaining) % 60:02d}s"
        else:
            eta_str = f"{remaining:.0f}s"
        elapsed_str = (
            f"{elapsed / 60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
        )
        time_str = f"  [{elapsed_str}<{eta_str}]"
    else:
        time_str = ""

    line = f"\r{step:5}/{total}  {b}  {frac * 100:5.1f}%  {stats}{time_str}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ==============================================================================
# Full vanilla KD training loop
# ==============================================================================

def training_loop(
    strategy,
    student_model,
    optimizer,
    lr_scheduler,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    args,
    job_dir,
    pf,
):
    best_ckpt   = os.path.join(job_dir, "student_best.weights.h5")
    resume_path = os.path.join(job_dir, "resume_state.json")

    history = {
        "total":     [],
        "hard":      [],
        "soft":      [],
        "val_total": [],
        "val_hard":  [],
        "val_soft":  [],
        "val_mae":   [],
    }
    best_val    = float("inf")
    patience_ct = 0
    start_epoch = 0
    nan_warn_threshold = max(1, int(train_steps * 0.10))

    # ── Resume: restore weights + full training state ─────────────────────────
    if args.resume:
        if os.path.exists(best_ckpt) and os.path.exists(resume_path):
            pf(f"[RESUME] Restoring weights from: {best_ckpt}")
            sys.stdout.flush()
            student_model.load_weights(best_ckpt)
            pf(f"[RESUME] Weights restored OK.")

            with open(resume_path, "r") as f:
                resume_state = json.load(f)

            start_epoch = int(resume_state["epoch"])
            best_val    = float(resume_state["best_val"])
            patience_ct = int(resume_state["patience_ct"])
            saved_lr    = float(resume_state["lr"])

            lr_scheduler.lr_var.assign(saved_lr)
            lr_scheduler.best = best_val

            if "history" in resume_state:
                saved_hist = resume_state["history"]
                for key in history:
                    if key in saved_hist:
                        history[key] = list(saved_hist[key])

            pf(
                f"[RESUME] Resuming from epoch {start_epoch + 1}  "
                f"best_val={best_val:.6f}  patience={patience_ct}  "
                f"lr={saved_lr:.2e}"
            )
            sys.stdout.flush()
        else:
            missing = []
            if not os.path.exists(best_ckpt):
                missing.append(best_ckpt)
            if not os.path.exists(resume_path):
                missing.append(resume_path)
            pf(
                f"[RESUME] WARNING: --resume set but the following files are "
                f"missing — starting from epoch 1 instead:\n"
                + "\n".join(f"  {p_}" for p_ in missing)
            )
            sys.stdout.flush()

    dist_train_step = make_distributed_train_step(
        strategy, student_model, optimizer, args.temperature, args.alpha
    )
    dist_val_step = make_distributed_val_step(
        strategy, student_model, args.temperature, args.alpha
    )

    # ── Pre-flight timing check — detect silent CPU fallback ──────────────────
    pf("Pre-flight timing check (forward-only, no weight update)...")
    try:
        sample_batch_x, sample_batch_y = next(iter(dist_train_dataset))
        t0 = time.time()
        _ = student_model(
            [sample_batch_x["enc_input"], sample_batch_x["dec_input"]],
            training=False,
        )
        elapsed_preflight = time.time() - t0
        status = (
            "OK ✓" if elapsed_preflight < 1.0 else "SLOW — check GPU visibility"
        )
        pf(f"  Single forward batch: {elapsed_preflight:.3f}s  [{status}]")
        if elapsed_preflight > 1.0:
            pf(
                "  TIP: Ensure CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
                "in SLURM script and --gres=gpu:8 is set."
            )
    except Exception as e:
        pf(f"  Pre-flight check skipped: {e}")

    pf("=" * 60)
    pf("VANILLA KD STUDENT TRAINING")
    micro_batch_size = args.batch_size // args.accumulation_steps
    pf(f"  Accumulation steps  : {args.accumulation_steps}")
    pf(f"  Micro batch size    : {micro_batch_size}  (one optimizer update per this many samples)")
    pf(f"  Logical batch size  : {args.batch_size}  (declared batch_size arg)")
    pf(f"  Optimizer view      : {micro_batch_size} samples / update  (matches BS=1024 run when accum=16)")
    pf(f"  Base LR (at ref BS {args.ref_batch_size}): {args.lr:.2e}")
    pf(f"  Effective LR (scaled): {args.effective_lr:.2e}")
    pf(f"  LR scaling ratio: {args.batch_size / args.ref_batch_size:.4f}  (batch_size / ref_batch_size)")
    pf(f"  LR patience (scaled): {args.effective_lr_patience}")
    pf(f"  Warmup epochs (scaled): {args.effective_warmup_epochs}")
    pf(f"  alpha={args.alpha}  temperature={args.temperature}")
    pf(f"  Student QGRU-{args.student_units}  {args.bits_kernel}-bit kernel")
    pf(f"  SEQ_LEN={args.seq_len}  BATCH={args.batch_size}  EPOCHS={args.epochs}")
    pf(f"  Replicas={strategy.num_replicas_in_sync}")
    pf(
        f"  Global BS: {args.batch_size}  "
        f"Per-GPU BS: {args.batch_size // max(strategy.num_replicas_in_sync, 1)}"
    )
    pf(f"  Train steps/epoch={train_steps}  Val steps/epoch={val_steps}")
    if args.resume and start_epoch > 0:
        pf(f"  Resuming from epoch {start_epoch + 1} / {args.epochs}")
    pf(f"  Checkpoint: {best_ckpt}")
    pf("=" * 60)
    sys.stdout.flush()

    student_model.summary(print_fn=pf)

    csv_path = os.path.join(job_dir, "training_history.csv")

    # Only write the CSV header if we are NOT resuming (append mode when resuming)
    if not args.resume or start_epoch == 0:
        with open(csv_path, "w") as csv_f:
            csv_f.write(
                "epoch,total,hard,soft,val_total,val_hard,val_soft,val_mae,lr\n"
            )

    for epoch in range(start_epoch, args.epochs):
        t_epoch      = time.time()
        t_batch_zero = None

        # ── Training ──────────────────────────────────────────────────────────
        acc_total = acc_hard = acc_soft = 0.0
        acc_steps = nan_count = 0

        pf(
            f"\n[EPOCH {epoch + 1}/{args.epochs}] Starting training  "
            f"lr={lr_scheduler.current_lr:.2e}"
        )
        sys.stdout.flush()

        for step, (bx, by) in enumerate(dist_train_dataset):
            if step == 0:
                t_batch_zero = time.time()

            loss, hard_l, soft_l, nan_flag = dist_train_step(bx, by)

            acc_total += float(loss)
            acc_hard  += float(hard_l)
            acc_soft  += float(soft_l)
            acc_steps += 1

            if bool(nan_flag):
                nan_count += 1

            if (step + 1) % args.log_interval == 0 or (step + 1) == train_steps:
                bar(
                    step + 1,
                    train_steps,
                    {
                        "tot":  acc_total / acc_steps,
                        "hard": acc_hard  / acc_steps,
                        "soft": acc_soft  / acc_steps,
                    },
                    epoch_start_time=t_batch_zero if t_batch_zero is not None
                    else t_epoch,
                )

        train_loss = acc_total / max(acc_steps, 1)
        train_hard = acc_hard  / max(acc_steps, 1)
        train_soft = acc_soft  / max(acc_steps, 1)

        if nan_count > nan_warn_threshold:
            pf(
                f"\n  *** WARNING: {nan_count}/{acc_steps} batches had NaN gradients "
                f"({100.0 * nan_count / max(acc_steps, 1):.1f}%). "
                f"Consider reducing --lr, --temperature, or --alpha. ***"
            )
            sys.stdout.flush()

        # ── Validation ────────────────────────────────────────────────────────
        pf(f"\n[EPOCH {epoch + 1}/{args.epochs}] Starting validation...")
        sys.stdout.flush()

        val_acc_total = val_acc_hard = val_acc_soft = val_acc_mae = 0.0
        val_done = 0

        for bx, by in dist_val_dataset:
            vt_l, vh_l, vs_l, vmae = dist_val_step(bx, by)
            val_acc_total += float(vt_l)
            val_acc_hard  += float(vh_l)
            val_acc_soft  += float(vs_l)
            val_acc_mae   += float(vmae)
            val_done      += 1

        val_loss = val_acc_total / max(val_done, 1)
        val_hard = val_acc_hard  / max(val_done, 1)
        val_soft = val_acc_soft  / max(val_done, 1)
        val_mae  = val_acc_mae   / max(val_done, 1)

        elapsed_epoch = time.time() - t_epoch

        history["total"].append(train_loss)
        history["hard"].append(train_hard)
        history["soft"].append(train_soft)
        history["val_total"].append(val_loss)
        history["val_hard"].append(val_hard)
        history["val_soft"].append(val_soft)
        history["val_mae"].append(val_mae)

        pf(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"hard={train_hard:.6f}  soft={train_soft:.6f}  "
            f"val_mae={val_mae:.6f}  "
            f"lr={lr_scheduler.current_lr:.2e}  "
            f"NaN-batches={nan_count}  "
            f"time={elapsed_epoch:.1f}s"
        )
        sys.stdout.flush()

        with open(csv_path, "a") as csv_f:
            csv_f.write(
                f"{epoch + 1},"
                f"{train_loss:.8f},{train_hard:.8f},{train_soft:.8f},"
                f"{val_loss:.8f},{val_hard:.8f},{val_soft:.8f},"
                f"{val_mae:.8f},{lr_scheduler.current_lr:.2e}\n"
            )

        # ── LR scheduler ──────────────────────────────────────────────────────
        if args.effective_warmup_epochs > 0 and epoch < args.effective_warmup_epochs:
            warmup_lr = float(args.effective_lr) * float(epoch + 1) / float(args.effective_warmup_epochs)
            lr_scheduler.lr_var.assign(warmup_lr)
            pf(
                f"  [WARMUP] epoch {epoch + 1}/{args.effective_warmup_epochs}  "
                f"lr={warmup_lr:.3e}  "
                f"(plateau scheduler suppressed during warmup)"
            )
        else:
            lr_scheduler.step(val_loss, epoch, pf)

        # ── Checkpoint best val_loss ───────────────────────────────────────────
        if val_loss < best_val - args.min_delta:
            best_val    = val_loss
            patience_ct = 0
            student_model.save_weights(best_ckpt)
            pf(f"  ✓ New best val={best_val:.6f}  saved → {best_ckpt}")
            sys.stdout.flush()
        else:
            patience_ct += 1
            pf(f"  patience {patience_ct}/{args.patience}")
            sys.stdout.flush()

        # ── Write resume state after EVERY epoch ──────────────────────────────
        # Written unconditionally so a SLURM preemption at any point leaves
        # a valid resume_state.json pointing at the last completed epoch.
        resume_state_out = {
            "epoch":      epoch + 1,
            "best_val":   float(best_val),
            "patience_ct": patience_ct,
            "lr":         float(lr_scheduler.current_lr),
            "history":    {k: [float(v) for v in vals]
                           for k, vals in history.items()},
        }
        with open(resume_path, "w") as f:
            json.dump(resume_state_out, f, indent=2)

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_ct >= args.patience:
            pf(f"Early stopping at epoch {epoch + 1}")
            sys.stdout.flush()
            break

    if os.path.exists(best_ckpt):
        student_model.load_weights(best_ckpt)
        pf(f"Restored best weights from {best_ckpt}")
        sys.stdout.flush()

    return history, best_val


# ==============================================================================
# Save loss curves — multi-panel PNG + CSV already written incrementally above
# ==============================================================================

def save_loss_curves(history, best_val_loss, args, job_dir, job_name, pf):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(history["total"],     color="tab:blue",   label="train")
    axes[0].plot(history["val_total"], color="tab:orange", linestyle="--", label="val")
    axes[0].set_title("Total KD Loss (train vs val)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["hard"],     color="tab:blue",   label="train hard MSE")
    axes[1].plot(history["val_hard"], color="tab:orange", linestyle="--", label="val hard MSE")
    axes[1].set_title("Hard MSE Loss (GT target)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["soft"],     color="tab:green",  label="train soft MSE")
    axes[2].plot(history["val_soft"], color="tab:red",    linestyle="--", label="val soft MSE")
    axes[2].set_title("Soft MSE Loss (teacher output)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].axis("off")
    axes[3].text(
        0.05, 0.55,
        f"Vanilla KD  SEQLEN={args.seq_len}\n"
        f"α={args.alpha}  (no temperature — pure MSE KD)\n"
        f"Teacher GRU hidden={args.teacher_units} x {args.teacher_layers} layers\n"
        f"Student QGRU hidden={args.student_units}  {args.bits_kernel}-bit kernel\n"
        f"bits: k={args.bits_kernel} r={args.bits_recurrent} "
        f"b={args.bits_bias} a={args.bits_activation} s={args.bits_state}\n"
        f"Batch size={args.batch_size}  ref_batch_size={args.ref_batch_size}\n"
        f"Base LR={args.lr:.2e}  Effective LR={args.effective_lr:.2e}\n"
        f"Best val loss={best_val_loss:.6f}\n"
        f"Epochs run={len(history['total'])}",
        fontsize=9,
        verticalalignment="center",
        transform=axes[3].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    curves_path = os.path.join(job_dir, "training_history.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"Loss curves saved: {curves_path}")
# ==============================================================================
# Post-processing helpers — same as fw-qatd-rac
# ==============================================================================

def extract_lifetimes(preds, t):
    """
    preds: (N, T, 3)  ch0=full, ch1=short, ch2=long
    t:     (T,) time axis in ns
    Returns tau1, tau2, fret each shape (N,).
    """
    ch1 = preds[:, :, 1]
    ch2 = preds[:, :, 2]

    int1 = np.trapz(ch1, t, axis=1)
    int2 = np.trapz(ch2, t, axis=1)

    amp1 = ch1[:, 0]
    amp2 = ch2[:, 0]

    tau1 = np.where(amp1 > 1e-6, int1 / amp1, 0.0).astype(np.float32)
    tau2 = np.where(amp2 > 1e-6, int2 / amp2, 0.0).astype(np.float32)

    denom = amp1 + amp2
    fret  = np.where(denom > 1e-6, amp1 / denom, 0.5).astype(np.float32)
    return tau1, tau2, fret


def compute_metrics(gt, pred, label, pfn):
    rmse = float(np.sqrt(np.mean((gt - pred) ** 2)))
    r, _ = pearsonr(gt.astype(float), pred.astype(float))
    residuals = pred - gt
    sigma = residuals.std()
    cov   = float(np.mean(np.abs(residuals) <= sigma) * 100)
    pfn(f"  {label:12s}  RMSE={rmse:.4f}  r={r:.4f}  1σ-cov={cov:.1f}%")
    return rmse, float(r), cov


def compute_sdf_metrics(gt_seqs, pred_seqs, channel_names, pfn):
    """
    Compute the 4 paper metrics (Table 1/2/3) on raw SDF output sequences.

    gt_seqs   : np.ndarray shape (N, T, C)  — ground truth decoder targets (res)
    pred_seqs : np.ndarray shape (N, T, C)  — student predictions
    channel_names : list of str, length C   — e.g. ["ch0_full","ch1_short","ch2_long"]
    pfn       : print function

    Returns a dict keyed by channel name, each containing:
        rmse, r2, l2_norm, dtw_distance  (all per-sample means)

    RMSE     : sqrt(mean over samples and timesteps of squared error)
    R²       : 1 - SS_res / SS_tot  (computed sample-wise then meaned)
    L2-norm  : mean over samples of sqrt(sum_t (gt_t - pred_t)^2)
    DTW      : mean over samples of FastDTW distance (euclidean path cost)
    """
    N, T, C = gt_seqs.shape
    results = {}

    for c, ch_name in enumerate(channel_names):
        gt_c   = gt_seqs[:, :, c]    # (N, T)
        pred_c = pred_seqs[:, :, c]  # (N, T)

        # ── RMSE (scalar over all samples and timesteps) ─────────────────────
        rmse = float(np.sqrt(np.mean((gt_c - pred_c) ** 2)))

        # ── R² (computed per sample, then averaged) ──────────────────────────
        ss_res = np.sum((gt_c - pred_c) ** 2, axis=1)        # (N,)
        ss_tot = np.sum((gt_c - gt_c.mean(axis=1, keepdims=True)) ** 2, axis=1)  # (N,)
        # Avoid division by zero for flat ground-truth sequences
        r2_per_sample = np.where(
            ss_tot > 1e-12,
            1.0 - ss_res / ss_tot,
            np.where(ss_res < 1e-12, 1.0, 0.0),
        )
        r2 = float(np.mean(r2_per_sample))

        # ── L2-norm (mean over samples of Euclidean distance per sample) ─────
        l2_per_sample = np.sqrt(np.sum((gt_c - pred_c) ** 2, axis=1))  # (N,)
        l2_norm = float(np.mean(l2_per_sample))

        # ── DTW distance (FastDTW, mean over samples) ────────────────────────
        # FastDTW operates on sequences of scalars (1-D arrays).
        # radius=1 matches the paper's DTW implementation (tight band).
        # We chunk the loop and print progress every 10% for large N.
        dtw_total = 0.0
        print_every_dtw = max(1, N // 10)
        t0_dtw = time.time()
        pfn(
            f"  [SDF DTW] channel={ch_name}  N={N:,}  T={T}  "
            f"computing FastDTW (radius=1)..."
        )
        sys.stdout.flush()
        for i in range(N):
            dist, _ = fastdtw(gt_c[i], pred_c[i], radius=1, dist=euclidean)
            dtw_total += dist
            if (i + 1) % print_every_dtw == 0 or (i + 1) == N:
                pct = 100.0 * (i + 1) / N
                elapsed_dtw = time.time() - t0_dtw
                pfn(
                    f"  [SDF DTW]   {i + 1:>8,}/{N:,}  ({pct:5.1f}%)  "
                    f"elapsed={elapsed_dtw / 60:.1f}min"
                )
                sys.stdout.flush()
        dtw_distance = float(dtw_total / N)

        results[ch_name] = {
            "rmse":         rmse,
            "r2_score":     r2,
            "l2_norm":      l2_norm,
            "dtw_distance": dtw_distance,
        }

        pfn(
            f"  SDF {ch_name:12s}  RMSE={rmse:.4f}  R²={r2:.4f}  "
            f"L2={l2_norm:.4f}  DTW={dtw_distance:.4f}"
        )
        sys.stdout.flush()

    return results


def run_inference(model, enc_arr, seq_len, n_out, batch_size, pfn):
    n     = len(enc_arr)
    preds = np.zeros((n, seq_len, n_out), dtype=np.float32)
    for s in tqdm(
        range(0, n, batch_size),
        desc="Student inference",
        unit="batch",
        bar_format="{l_bar}{bar:30}{r_bar}",
    ):
        e     = min(s + batch_size, n)
        enc_b = tf.constant(enc_arr[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        preds[s:e] = model([enc_b, dec_b], training=False).numpy()
    return preds


def evaluate_and_save(
    student_model,
    normalized_input,
    res,
    labels,
    test_idx,
    seq_len,
    n_out,
    gate_width_ns,
    infer_batch,
    job_dir,
    job_name,
    pfn,
):
    pfn("=" * 60)
    pfn("Test set evaluation")
    pfn("=" * 60)

    enc_test = normalized_input[test_idx]
    lab_test = labels[test_idx]
    res_test = res[test_idx]

    student_preds = run_inference(
        student_model, enc_test, seq_len, n_out, infer_batch, pfn
    )
    pfn(f"student_preds shape: {student_preds.shape}")

    t_ns_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(student_preds, t_ns_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    pfn(f"  τ₁ pred range: {tau1_pred.min():.3f} – {tau1_pred.max():.3f} ns")
    pfn(f"  τ₂ pred range: {tau2_pred.min():.3f} – {tau2_pred.max():.3f} ns")
    pfn(f"  FRET pred range: {fret_pred.min():.3f} – {fret_pred.max():.3f}")

    pfn("=" * 55)
    pfn(f"Student Vanilla KD  {job_name}  Test set N={len(test_idx):,}")
    pfn("=" * 55)
    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pfn)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pfn)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pfn)

    test_metrics = {
        "job_name": job_name,
        "n_test":   int(len(test_idx)),
        "tau1":     {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
        "tau2":     {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
        "fret":     {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
    }
    metrics_path = os.path.join(job_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    pfn(f"Test metrics saved: {metrics_path}")

    panels = [
        (tau1_gt, tau1_pred, m1, "τ₁ (ns)", "GT τ₁ (ns)", "Pred τ₁ (ns)",
         (0, 3.0), "Blues",   "test_scatter_tau1.png"),
        (tau2_gt, tau2_pred, m2, "τ₂ (ns)", "GT τ₂ (ns)", "Pred τ₂ (ns)",
         (0, 3.0), "Greens",  "test_scatter_tau2.png"),
        (fret_gt, fret_pred, mf, "FRET (f)", "GT FRET (f)", "Pred FRET (f)",
         (0, 1.0), "Oranges", "test_scatter_fret.png"),
    ]
    for gt, pred, metrics, title, xlabel, ylabel, lims, cmap, fname in panels:
        rmse_v, r_v, cov_v = metrics
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hb = ax.hexbin(
            gt, pred, gridsize=80, bins="log", cmap=cmap,
            extent=(lims[0], lims[1], lims[0], lims[1]), mincnt=1,
        )
        ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}  {job_name}", fontsize=10, fontweight="bold")
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
        fig.colorbar(hb, ax=ax, pad=0.02).set_label("log₁₀(count)", fontsize=9)
        plt.tight_layout()
        scatter_path = os.path.join(job_dir, fname)
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pfn(f"Scatter saved: {scatter_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, gt_v, pred_v, label, color in zip(
        axes,
        [tau1_gt, tau2_gt, fret_gt],
        [tau1_pred, tau2_pred, fret_pred],
        ["τ₁ (ns)", "τ₂ (ns)", "FRET (f)"],
        ["steelblue", "seagreen", "darkorange"],
    ):
        residuals = pred_v - gt_v
        ax.hist(residuals, bins=100, color=color, alpha=0.75, edgecolor="none")
        ax.axvline(0, color="red", linewidth=1.2, linestyle="--")
        ax.set_xlabel(f"Residual {label}", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Residuals {label}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.97, 0.97,
            f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    fig.suptitle(f"Residuals  {job_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    residuals_path = os.path.join(job_dir, "test_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pfn(f"Residuals saved: {residuals_path}")

    # ── SDF-domain metrics (Table 1/2/3 of the paper) ─────────────────────────
    # Computed AFTER scatter PNGs so a DTW timeout/OOM never blocks scatter output.
    # Wrapped in try/except so any failure is logged but does not abort the run.
    pfn("=" * 60)
    pfn("SDF-domain metrics (paper Table 1/2/3): RMSE, R², L2-norm, DTW")
    pfn("=" * 60)
    try:
        sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
        sdf_metrics = compute_sdf_metrics(
            gt_seqs       = res_test.astype(np.float32),
            pred_seqs     = student_preds,
            channel_names = sdf_channel_names,
            pfn           = pfn,
        )
        sdf_metrics["job_name"] = job_name
        sdf_metrics["n_test"]   = int(len(test_idx))
        sdf_metrics_path = os.path.join(job_dir, "test_sdf_metrics.json")
        with open(sdf_metrics_path, "w") as f:
            json.dump(sdf_metrics, f, indent=2)
        pfn(f"SDF metrics saved: {sdf_metrics_path}")
    except Exception as _sdf_exc:
        pfn(f"WARNING: SDF metrics computation failed and was skipped: {_sdf_exc}")
        pfn("Scatter PNGs and test_metrics.json were already saved above.")

    return test_metrics

# ==============================================================================
# main
# ==============================================================================


def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    # ── Linear LR scaling based on batch size ratio ───────────────────────────
    # The linear scaling rule: when batch size increases by factor k, multiply
    # LR by k so each gradient update sees the same expected gradient magnitude.
    # lr_patience and warmup_epochs are scaled by the same ratio so the plateau
    # scheduler and warmup cover the same number of gradient updates regardless
    # of batch size.
    #
    # With --no-lr-scaling: use --lr, --lr-patience, --warmup-epochs exactly.
    # With scaling (default): scale all three by (batch_size / ref_batch_size).
    #
    # Example: --batch-size 16384 --ref-batch-size 1024 --lr 1e-4
    #   scaling ratio = 16384 / 1024 = 16.0
    #   effective_lr  = 1e-4 * 16.0  = 1.6e-3
    #   effective_lr_patience  = 8  * 16 = 128
    #   effective_warmup_epochs = 5 * 16 = 80
    # ─────────────────────────────────────────────────────────────────────────
    if args.no_lr_scaling:
        args.effective_lr             = args.lr
        args.effective_lr_patience    = args.lr_patience
        args.effective_warmup_epochs  = args.warmup_epochs
        pf(
            f"[LR SCALING] Disabled (--no-lr-scaling).  "
            f"lr={args.effective_lr:.2e}  "
            f"lr_patience={args.effective_lr_patience}  "
            f"warmup_epochs={args.effective_warmup_epochs}"
        )
    else:
        scaling_ratio                 = args.batch_size / args.ref_batch_size
        args.effective_lr             = args.lr * scaling_ratio
        args.effective_lr_patience    = max(1, round(args.lr_patience    * scaling_ratio))
        args.effective_warmup_epochs  = max(0, round(args.warmup_epochs  * scaling_ratio))
        pf(
            f"[LR SCALING] batch_size={args.batch_size}  "
            f"ref_batch_size={args.ref_batch_size}  "
            f"ratio={scaling_ratio:.4f}"
        )
        pf(
            f"[LR SCALING] base lr={args.lr:.2e}  "
            f"→  effective lr={args.effective_lr:.2e}"
        )
        pf(
            f"[LR SCALING] base lr_patience={args.lr_patience}  "
            f"→  effective lr_patience={args.effective_lr_patience}"
        )
        pf(
            f"[LR SCALING] base warmup_epochs={args.warmup_epochs}  "
            f"→  effective warmup_epochs={args.effective_warmup_epochs}"
        )

    tf.keras.utils.set_random_seed(args.split_seed)
    pf(f"Global random seed set to {args.split_seed}")

    # ── STEP 2 — GPU strategy + mixed precision (set once, here only) ─────────
    strategy = setup_gpus_and_strategy(args.mixed_precision)

    job_name = make_job_name(args)
    job_dir  = os.path.join(args.save_dir, "results", job_name)
    os.makedirs(job_dir, exist_ok=True)
    pf(f"Job:     {job_name}")
    pf(f"Job dir: {job_dir}")

    with open(os.path.join(job_dir, "student_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── 1. Load data files ────────────────────────────────────────────────────
    pf("Loading data files...")
    (file_input, file_res, file_labels,
     file_train, file_val, file_test) = find_data_files(args.data_dir, args.seq_len)

    pf(f"  encoder input : {file_input}")
    pf(f"  decoder target: {file_res}")
    pf(f"  labels        : {file_labels}")
    pf(f"  train idx     : {file_train}")
    pf(f"  val idx       : {file_val}")
    pf(f"  test idx      : {file_test}")

    normalized_input = np.load(file_input,  mmap_mode="r")
    res              = np.load(file_res,    mmap_mode="r")
    labels           = np.load(file_labels, mmap_mode="r")
    train_idx        = np.load(file_train)
    val_idx          = np.load(file_val)
    test_idx         = np.load(file_test)

    n_samples = normalized_input.shape[0]
    pf(f"  N={n_samples:,}  seq_len={args.seq_len}  n_out={args.n_out}")
    pf(
        f"  Train={len(train_idx):,}  "
        f"Val={len(val_idx):,}  "
        f"Test={len(test_idx):,}"
    )
    assert normalized_input.shape[1] == args.seq_len, (
        f"--seq-len={args.seq_len} but data has seq_len={normalized_input.shape[1]}"
    )

    # ── 2. Build teacher outside strategy scope for cache inference ───────────
    pf("Building teacher model for cache inference (single GPU)...")
    teacher_model = build_teacher(
        args.seq_len, args.n_out,
        args.teacher_units, args.teacher_layers,
    )
    pf(f"Loading teacher weights from: {args.teacher_ckpt}")
    teacher_model.load_weights(args.teacher_ckpt)
    teacher_model.trainable = False
    pf("Teacher weights loaded and frozen.")
    teacher_model.summary(print_fn=pf)

    # ── 3. Cache teacher predictions (checkpoint-guarded, mmap) ──────────────
    pf("=" * 60)
    pf("PHASE 1: Teacher prediction cache")
    pf("=" * 60)
    teacher_predictions = cache_teacher_predictions(
        teacher_model,
        normalized_input,
        args.seq_len,
        args.n_out,
        n_samples,
        args.infer_batch,
        args.data_dir,
        pf,
    )

    # ── 4. Materialise enc / tgt / tpred for train + val into RAM ─────────────
    pf("=" * 60)
    pf("PHASE 2: Materialising enc/tgt/tpred into RAM (one-time cost)")
    pf("=" * 60)
    enc_train, tgt_train, tpred_train = materialise_enc_tgt_tpred(
        normalized_input, res, teacher_predictions,
        train_idx, args.seq_len, args.n_out, "train", pf,
    )
    enc_val, tgt_val, tpred_val = materialise_enc_tgt_tpred(
        normalized_input, res, teacher_predictions,
        val_idx, args.seq_len, args.n_out, "val", pf,
    )
    pf("Materialisation complete.")

    # ── 5. Build student inside strategy.scope() ──────────────────────────────
    pf("=" * 60)
    pf("PHASE 3: Building student model and optimizer")
    pf("=" * 60)
    with strategy.scope():
        student_model = build_student(
            args.seq_len, args.n_out, args.student_units,
            args.bits_kernel, args.bits_recurrent, args.bits_bias,
            args.bits_activation, args.bits_state,
        )
        optimizer    = keras.optimizers.Adam(learning_rate=args.effective_lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor    = args.lr_factor,
            patience  = args.effective_lr_patience,
            min_lr    = args.lr_min,
            min_delta = args.min_delta,
        )

    student_model.summary(print_fn=pf)
    pf(f"Student trainable params: {student_model.count_params():,}")

    # ── 6. tf.data pipelines — micro_batch_size drives dataset, not batch_size ──
    pf("Building tf.data pipelines...")

    micro_batch_size = args.batch_size // args.accumulation_steps
    pf(
        f"  Micro batch size : {micro_batch_size}  "
        f"(batch_size={args.batch_size} / accumulation_steps={args.accumulation_steps})"
    )
    pf(
        f"  Effective batch size (optimizer view): {args.batch_size}  "
        f"  Updates per epoch (train): {len(train_idx) // micro_batch_size}"
    )

    train_dataset = make_kd_dataset(
        enc_train, tgt_train, tpred_train,
        batch_size       = args.batch_size,
        accumulation_steps = args.accumulation_steps,
        seq_len          = args.seq_len,
        n_out            = args.n_out,
        shuffle          = True,
        seed             = args.split_seed,
        prefetch_batches = args.prefetch_batches,
        pipeline_workers = args.pipeline_workers,
    )
    val_dataset = make_kd_dataset(
        enc_val, tgt_val, tpred_val,
        batch_size       = args.batch_size,
        accumulation_steps = args.accumulation_steps,
        seq_len          = args.seq_len,
        n_out            = args.n_out,
        shuffle          = False,
        seed             = args.split_seed,
        prefetch_batches = args.prefetch_batches,
        pipeline_workers = args.pipeline_workers,
    )

    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset   = strategy.experimental_distribute_dataset(val_dataset)

    train_steps = len(train_idx) // micro_batch_size
    val_steps   = len(val_idx)   // micro_batch_size
    pf(f"  Train steps/epoch : {train_steps:,}  (was {len(train_idx) // args.batch_size:,} without accumulation)")
    pf(f"  Val   steps/epoch : {val_steps:,}")    

    # ── 7. Pre-flight diagnostics ─────────────────────────────────────────────
    pf("=== PRE-FLIGHT DIAGNOSTICS ===")
    pf(f"Replicas in sync  : {strategy.num_replicas_in_sync}")
    pf(
        f"Global batch size : {args.batch_size}  "
        f"Per-GPU batch size: "
        f"{args.batch_size // max(strategy.num_replicas_in_sync, 1)}"
    )
    pf(
        f"KD hyper-params   : alpha={args.alpha}  temperature={args.temperature}"
    )
    pf(
        f"Quantization bits : kernel={args.bits_kernel}  "
        f"recurrent={args.bits_recurrent}  bias={args.bits_bias}  "
        f"activation={args.bits_activation}  state={args.bits_state}"
    )
    pf(
        f"LR scaling        : base={args.lr:.2e}  "
        f"effective={args.effective_lr:.2e}  "
        f"ratio={args.batch_size / args.ref_batch_size:.4f}"
    )

    # ── 8. Training loop ──────────────────────────────────────────────────────
    pf("=" * 60)
    pf("VANILLA KD STUDENT TRAINING")
    pf(f"  Job: {job_name}")
    pf(f"  alpha={args.alpha}  temperature={args.temperature}")
    pf(f"  Teacher GRU hidden={args.teacher_units} x {args.teacher_layers} layers")
    pf(f"  Student QGRU-{args.student_units}  {args.bits_kernel}-bit kernel")
    pf(f"  SEQLEN={args.seq_len}  BATCH={args.batch_size}  EPOCHS={args.epochs}")
    pf("=" * 60)

    history, best_val_loss = training_loop(
        strategy,
        student_model,
        optimizer,
        lr_scheduler,
        dist_train_dataset,
        dist_val_dataset,
        train_steps,
        val_steps,
        args,
        job_dir,
        pf,
    )

    # ── 9. Save loss curves ───────────────────────────────────────────────────
    save_loss_curves(history, best_val_loss, args, job_dir, job_name, pf)

    # ── 10. Test-set evaluation ───────────────────────────────────────────────
    evaluate_and_save(
        student_model, normalized_input, res, labels,
        test_idx, args.seq_len, args.n_out, args.gate_width_ns,
        args.infer_batch, job_dir, job_name, pf,
    )

    # ── 11. Save final weights ────────────────────────────────────────────────
    final_weights_path = os.path.join(job_dir, "student_final.weights.h5")
    student_model.save_weights(final_weights_path)
    pf(f"Final weights saved: {final_weights_path}")

    pf("=" * 60)
    pf(f"DONE — best val loss : {best_val_loss:.6f}")
    pf(f"Results in           : {job_dir}")
    pf("=" * 60)


if __name__ == "__main__":
    main()