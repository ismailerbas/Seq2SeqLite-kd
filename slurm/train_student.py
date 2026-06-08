#!/usr/bin/env python3
"""
train_student.py — FW-QATD-RAC Student Training
QKeras quantized GRU seq2seq with Fisher-weighted trajectory distillation
and recurrent accumulator consistency regularization.

Loss: L_total = L_GT + alpha*L_KD + beta*L_traj + gamma*L_RAC
  L_GT   = MSE(y_student, y_true)
  L_KD   = MSE(y_student, y_teacher)
  L_traj = mean( F * ||P*h_quant - h_teacher||^2 )  -- Fisher-weighted
  L_RAC  = mean( t_weights * mean_dim(||h_quant - sg(h_float)||^2) )

Teacher is frozen. Float shadow student synced every --shadow-sync-every epochs.
Projection layer P (student_units -> teacher_units) discarded after training.
MirroredStrategy across all visible GPUs with explicit device list.

IMPROVEMENTS vs original:
  - CUDA_VISIBLE_DEVICES set before TF import if not already in env
  - Explicit MirroredStrategy device list (never auto-guess)
  - Pre-flight GPU timing check
  - @tf.function placed on distributed wrappers only (not per-replica fns)
  - Teacher cache + Fisher diagonal guarded checkpoint (compute once, reuse)
  - Hybrid fast pipeline: enc/tgt/t_pred materialised in RAM (GIL-free),
    teacher_hidden_traj accessed via mmap with single-threaded prefetch
  - Loss scaling bug fixed: MEAN reduce without manual local/global scaling
  - NaN guard: skip apply_gradients, count, warn if >10% of batches
  - shadow-sync-every wired to work per epoch (configurable)
  - Progress bar with ETA (ported from teacher script)
  - 6-panel loss curve PNG + detailed CSV

Naming convention subfolder: student_b{W}k{W}r{W}a{W}_gru{U}x1_dense{NOUT}_bs{B}
e.g. student_b4k4r4a4_gru32x1_dense3_bs1024

Data files expected in --data-dir:
  tpsf_seq_L{SEQ_LEN}_{N}M.npy        -- encoder input  (N, SEQ_LEN, 1)
  res_L{SEQ_LEN}_{N}M.npy             -- decoder target (N, SEQ_LEN, 3)
  labels_3ch_L{SEQ_LEN}_{N}M.npy      -- labels         (N, 3)
  trainidx.npy / validx.npy / testidx.npy (or test_idx.npy)
  teacher_best.weights.h5              -- teacher checkpoint (from train_teacher.py)

Auto-generated cache files (computed once, reused):
  teacherPred_L{SEQ_LEN}{N}.npy        -- teacher predictions (N, SEQ_LEN, 3)
  teacherHidden_L{SEQ_LEN}{N}.npy      -- teacher hidden traj (N, SEQ_LEN, teacher_units)
  fisherDiag_L{SEQ_LEN}{N}.npy         -- Fisher diagonal     (teacher_units,)

Usage example:
  python train_student.py \\
    --data-dir /gpfs/.../nmi \\
    --teacher-ckpt /gpfs/.../nmi/teacher_best.weights.h5 \\
    --bits-kernel 4 --bits-bias 4 --bits-recurrent 4 \\
    --bits-activation 4 --bits-state 4 \\
    --student-units 32 --teacher-units 128 --teacher-layers 2 \\
    --seq-len 135 --n-out 3 --gate-width-ns 0.09 \\
    --batch-size 1024 --epochs 300 --patience 15 \\
    --lr 1e-4 --lr-factor 0.5 --lr-patience 8 --lr-min 1e-6 \\
    --alpha 0.5 --beta 0.05 --gamma 1e-3 \\
    --shadow-sync-every 1 --log-interval 10 \\
    --fisher-batch 4096 --infer-batch 8192 \\
    --prefetch-batches 32 --pipeline-workers 4 \\
    --split-seed 42
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
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GRU, GRUCell, RNN, Input
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="FW-QATD-RAC student training with QKeras + MirroredStrategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- paths ---
    p.add_argument("--data-dir",      type=str, required=True,
                   help="Directory containing all .npy data files")
    p.add_argument("--teacher-ckpt",  type=str, required=True,
                   help="Path to teacher .weights.h5 checkpoint")
    p.add_argument("--save-dir",      type=str, default=None,
                   help="Root directory for output subfolder. Defaults to --data-dir.")
    # --- quantization ---
    p.add_argument("--bits-kernel",     type=int, default=4,
                   help="Bits for kernel (input projection) weights in QGRU and QDense")
    p.add_argument("--bits-recurrent",  type=int, default=4,
                   help="Bits for recurrent weights in QGRU")
    p.add_argument("--bits-bias",       type=int, default=4,
                   help="Bits for bias terms in QGRU and QDense")
    p.add_argument("--bits-activation", type=int, default=4,
                   help="Bits for quantized_tanh activation in QGRU")
    p.add_argument("--bits-state",      type=int, default=4,
                   help="Bits for GRU hidden state quantization")
    # --- architecture ---
    p.add_argument("--student-units",   type=int, default=32,
                   help="QGRU hidden units in student encoder and decoder")
    p.add_argument("--teacher-units",   type=int, default=128,
                   help="Hidden units of each teacher GRU layer")
    p.add_argument("--teacher-layers",  type=int, default=2,
                   help="Number of stacked GRUCell layers in teacher (default: 2)")
    p.add_argument("--seq-len",         type=int, default=135,
                   help="Sequence length (time bins)")
    p.add_argument("--n-out",           type=int, default=3,
                   help="Number of output channels")
    p.add_argument("--gate-width-ns",   type=float, default=0.09,
                   help="Gate width in ns per time bin (SS3 = 0.09 ns / 90 ps)")
    # --- training ---
    p.add_argument("--batch-size",        type=int,   default=1024,
                   help="Global batch size across all GPUs")
    p.add_argument("--epochs",            type=int,   default=300)
    p.add_argument("--patience",          type=int,   default=15,
                   help="Early stopping patience in epochs")
    p.add_argument("--min-delta",         type=float, default=1e-5,
                   help="Minimum val loss improvement to reset patience")
    p.add_argument("--lr",                type=float, default=1e-4,
                   help="Initial Adam learning rate")
    p.add_argument("--lr-factor",         type=float, default=0.5,
                   help="LR reduction factor on plateau")
    p.add_argument("--lr-patience",       type=int,   default=8,
                   help="Epochs without improvement before LR reduction")
    p.add_argument("--lr-min",            type=float, default=1e-6,
                   help="Minimum learning rate floor")
    p.add_argument("--shadow-sync-every", type=int,   default=1,
                   help="Sync float shadow student weights every N epochs")
    p.add_argument("--log-interval",      type=int,   default=10,
                   help="Print progress bar every N steps")
    p.add_argument("--fisher-batch",      type=int,   default=4096,
                   help="Batch size for Fisher diagonal computation")
    p.add_argument("--infer-batch",       type=int,   default=8192,
                   help="Batch size for teacher cache inference and test evaluation")
    p.add_argument("--mixed-precision",   action="store_true",
                   help="Enable float16 mixed precision training")
    p.add_argument("--prefetch-batches",  type=int,   default=32,
                   help="Number of batches to prefetch in tf.data pipeline")
    p.add_argument("--pipeline-workers",  type=int,   default=4,
                   help="num_parallel_calls for tf.data map() — keep low to "
                        "reduce GIL contention from py_function on thid mmap reads")
    p.add_argument("--split-seed",        type=int,   default=42,
                   help="RNG seed passed to tf.keras.utils.set_random_seed")
    # --- loss weights ---
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Weight for L_KD (output knowledge distillation)")
    p.add_argument("--beta",  type=float, default=0.05,
                   help="Weight for L_traj (Fisher-weighted trajectory distillation)")
    p.add_argument("--gamma", type=float, default=1e-3,
                   help="Weight for L_RAC (recurrent accumulator consistency)")
    # --- resume ---
    p.add_argument("--resume", action="store_true",
                   help=(
                       "Resume training from student_best.weights.h5 + "
                       "resume_state.json in the job output directory. "
                       "Restores epoch counter, best_val, patience, LR scheduler "
                       "state, and full loss history so training continues exactly "
                       "where it left off."
                   ))

    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = args.data_dir
    return args

# ---------------------------------------------------------------------------
# Job naming
# ---------------------------------------------------------------------------
def make_job_name(args):
    return (
        f"student_b{args.bits_kernel}k{args.bits_kernel}"
        f"r{args.bits_recurrent}a{args.bits_activation}"
        f"_gru{args.student_units}x1"
        f"_dense{args.n_out}"
        f"_bs{args.batch_size}"
    )


# ---------------------------------------------------------------------------
# STEP 3 — GPU / Strategy setup with explicit device list
# set_memory_growth MUST be called before ANY other TF GPU operation.
# ---------------------------------------------------------------------------
def setup_gpus_and_strategy():
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[STEP 3] No physical GPUs found — running on CPU.", flush=True)
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(
                f"[STEP 3]   WARNING set_memory_growth failed for {gpu.name}: {e}",
                flush=True,
            )

    print(f"[STEP 3] Physical GPUs detected : {len(physical_gpus)}", flush=True)
    for i, g in enumerate(physical_gpus):
        print(f"[STEP 3]   GPU {i}: {g.name}", flush=True)

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[STEP 3] Logical GPUs visible   : {len(logical_gpus)}", flush=True)

    if len(logical_gpus) < len(physical_gpus):
        print(
            f"[STEP 3] WARNING: only {len(logical_gpus)} logical GPUs from "
            f"{len(physical_gpus)} physical. "
            f"Check CUDA_VISIBLE_DEVICES and SLURM --gres=gpu:N",
            flush=True,
        )

    if len(logical_gpus) == 0:
        print("[STEP 3] No logical GPUs available — falling back to CPU.", flush=True)
        return tf.distribute.get_strategy()

    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    print(
        f"[STEP 3] MirroredStrategy: {strategy.num_replicas_in_sync} replicas  "
        f"devices={gpu_devices}",
        flush=True,
    )

    if strategy.num_replicas_in_sync == 1 and len(logical_gpus) > 1:
        print(
            "[STEP 3] WARNING: MirroredStrategy sees only 1 replica despite "
            f"{len(logical_gpus)} logical GPUs. "
            "Set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 in your SLURM script "
            "BEFORE calling python.",
            flush=True,
        )

    return strategy


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
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

    # Try both index filename conventions
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


# ---------------------------------------------------------------------------
# Teacher model — EXACT replica of notebook Cell 5 / train_teacher.py
# Stacked GRUCell inside keras.layers.RNN
# Layer names: enc_input, dec_input, enc_rnn, dec_rnn, dec_dense
# LAYERS_TEACHER = [teacher_units] * teacher_layers  (default [128, 128])
# ---------------------------------------------------------------------------
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


def build_teacher_hidden_model(teacher_model):
    """
    Expose decoder_hidden_sequence (before Dense head).
    Taps the first output of 'dec_rnn' — shape (batch, T, teacher_units).
    """
    decoder_hidden_sequence = teacher_model.get_layer("dec_rnn").output[0]
    hidden_model = keras.models.Model(
        inputs=teacher_model.inputs,
        outputs=decoder_hidden_sequence,
        name="teacher_hidden_model",
    )
    return hidden_model


# ---------------------------------------------------------------------------
# Student model — matches notebook Cell 11 exactly
# QGRU encoder + QGRU decoder + QDense head
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense
# ---------------------------------------------------------------------------
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
        name="student_model",
    )
    return student_model


def build_student_hidden_model(student_model):
    """Expose quantized decoder hidden trajectory (before QDense head)."""
    dec_hid_seq = student_model.get_layer("sdecgru").output[0]
    hidden_model = Model(
        inputs=student_model.inputs,
        outputs=dec_hid_seq,
        name="student_hidden_model",
    )
    return hidden_model


# ---------------------------------------------------------------------------
# Float shadow student — matches notebook Cell 14 exactly
# Identical topology to student but standard float32 GRU (no QKeras).
# Used ONLY for the RAC loss term. Weights synced from quantized student.
# ---------------------------------------------------------------------------
def build_float_shadow_student(seq_len, n_out, student_units):
    fs_enc_inputs = Input(shape=(None, 1), name="fsenc_input")
    fs_dec_inputs = Input(shape=(None, 1), name="fsdec_input")

    _, fs_enc_state = GRU(
        student_units,
        return_state=True,
        name="fsencgru",
    )(fs_enc_inputs)

    fs_dec_hid_seq, _ = GRU(
        student_units,
        return_sequences=True,
        return_state=True,
        name="fsdecgru",
    )(fs_dec_inputs, initial_state=fs_enc_state)

    float_shadow_model = Model(
        inputs=[fs_enc_inputs, fs_dec_inputs],
        outputs=fs_dec_hid_seq,
        name="float_shadow_student",
    )
    return float_shadow_model

def sync_float_student_weights(student_model, float_shadow_model):
    def copy_gru(src_layer, dst_layer):
        src_weights = {v.name.split("/")[-1]: v.numpy()
                       for v in src_layer.weights}
        dst_vars = dst_layer.weights
        for dst_var in dst_vars:
            key = dst_var.name.split("/")[-1]
            key_base = key.replace(":0", "")
            match = None
            for src_key, src_val in src_weights.items():
                if key_base in src_key and src_val.shape == dst_var.shape:
                    match = src_val
                    break
            if match is not None:
                dst_var.assign(match)

    copy_gru(
        student_model.get_layer("sencgru"),
        float_shadow_model.get_layer("fsencgru"),
    )
    copy_gru(
        student_model.get_layer("sdecgru"),
        float_shadow_model.get_layer("fsdecgru"),
    )

# --------------------------------------------------------------------------
# Teacher cache — compute once and save, reload on subsequent runs
# Fused forward pass writes teacherPred and teacherHidden to mmap files.
# ---------------------------------------------------------------------------
def cache_teacher_outputs(
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
    file_pred   = os.path.join(data_dir, f"teacherPred_L{seq_len}{n_samples}.npy")
    file_hidden = os.path.join(data_dir, f"teacherHidden_L{seq_len}{n_samples}.npy")

    if os.path.exists(file_pred) and os.path.exists(file_hidden):
        pf("Teacher cache found — loading from disk (mmap):")
        pf(f"  {file_pred}")
        pf(f"  {file_hidden}")
        teacher_predictions = np.load(file_pred,   mmap_mode="r")
        teacher_hidden_traj = np.load(file_hidden, mmap_mode="r")
        pf(f"  teacher_predictions : {teacher_predictions.shape}  dtype={teacher_predictions.dtype}")
        pf(f"  teacher_hidden_traj : {teacher_hidden_traj.shape}  dtype={teacher_hidden_traj.dtype}")
        return teacher_predictions, teacher_hidden_traj

    pf("=" * 60)
    pf("Teacher cache NOT found — running full-dataset inference...")
    pf(f"  This runs ONCE and saves results to disk.")
    pf(f"  pred  → {file_pred}")
    pf(f"  hidden→ {file_hidden}")
    pf("=" * 60)

    pf(f"[cache] Opening mmap files for writing...")
    pf(f"  teacherPred   shape=({n_samples}, {seq_len}, {n_out})   dtype=float32")
    pf(f"  teacherHidden shape=({n_samples}, {seq_len}, {teacher_units})  dtype=float32")
    sys.stdout.flush()

    teacher_predictions = np.lib.format.open_memmap(
        file_pred, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, n_out),
    )
    teacher_hidden_traj = np.lib.format.open_memmap(
        file_hidden, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, teacher_units),
    )
    pf("[cache] mmap files opened OK.")
    sys.stdout.flush()

    @tf.function(reduce_retracing=True)
    def fused_forward(enc_b, dec_b):
        pred = teacher_model([enc_b, dec_b], training=False)
        hid  = teacher_hidden_model([enc_b, dec_b], training=False)
        return pred, hid

    pf("[cache] Starting fused forward warm-up trace...")
    sys.stdout.flush()
    wu_size = min(infer_batch, n_samples)
    enc_wu  = tf.constant(normalized_input[:wu_size], dtype=tf.float32)
    dec_wu  = tf.zeros((wu_size, seq_len, 1), dtype=tf.float32)
    _p, _h  = fused_forward(enc_wu, dec_wu)
    _ = _p.numpy()
    _ = _h.numpy()
    pf("[cache] Fused forward warm-up done — trace compiled and executed.")
    del enc_wu, dec_wu, _p, _h
    sys.stdout.flush()

    n_batches = int(np.ceil(n_samples / infer_batch))
    pf(f"[cache] Starting inference loop: {n_batches} batches  infer_batch={infer_batch}  n_samples={n_samples}")
    sys.stdout.flush()

    t0        = time.time()
    t_last    = t0
    print_every = max(1, n_batches // 20)   # print ~20 progress lines total

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_samples)

        enc_b = tf.constant(normalized_input[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)

        pred, hid = fused_forward(enc_b, dec_b)

        teacher_predictions[s:e] = pred.numpy()
        teacher_hidden_traj[s:e] = hid.numpy()

        # flush mmap every batch so data is on disk incrementally
        teacher_predictions.flush()
        teacher_hidden_traj.flush()

        del enc_b, dec_b, pred, hid

        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed   = time.time() - t0
            step_time = time.time() - t_last
            samples_done = e
            pct = 100.0 * samples_done / n_samples
            eta_s = (elapsed / max(b + 1, 1)) * (n_batches - b - 1)
            pf(
                f"[cache] batch {b + 1:>4d}/{n_batches}  "
                f"samples {samples_done:>9,}/{n_samples:,}  "
                f"({pct:5.1f}%)  "
                f"elapsed={elapsed / 60:.1f}min  "
                f"ETA={eta_s / 60:.1f}min  "
                f"step={step_time:.1f}s"
            )
            t_last = time.time()
            sys.stdout.flush()

    pf("[cache] Inference loop complete — flushing mmap buffers...")
    sys.stdout.flush()
    teacher_predictions.flush()
    teacher_hidden_traj.flush()
    del teacher_predictions
    del teacher_hidden_traj

    elapsed = time.time() - t0
    pf(
        f"[cache] Teacher cache done in {elapsed / 60:.1f} min  "
        f"({n_samples / elapsed:.0f} samples/s)"
    )
    pf(f"[cache] Re-opening cache files read-only (mmap)...")
    sys.stdout.flush()

    teacher_predictions = np.load(file_pred,   mmap_mode="r")
    teacher_hidden_traj = np.load(file_hidden, mmap_mode="r")
    pf(f"  teacher_predictions : {teacher_predictions.shape}  dtype={teacher_predictions.dtype}")
    pf(f"  teacher_hidden_traj : {teacher_hidden_traj.shape}  dtype={teacher_hidden_traj.dtype}")
    sys.stdout.flush()
    return teacher_predictions, teacher_hidden_traj

# ---------------------------------------------------------------------------
# Fisher diagonal — compute once and save, reload on subsequent runs
# Diagonal Fisher of teacher decoder's dec_dense layer over full dataset.
# ---------------------------------------------------------------------------

def compute_or_load_fisher(
    teacher_model,
    res,
    n_samples,
    seq_len,
    teacher_units,
    fisher_batch,
    data_dir,
    pf,
):
    fisher_path = os.path.join(data_dir, f"fisherDiag_L{seq_len}{n_samples}.npy")

    if os.path.exists(fisher_path):
        pf(f"Fisher diagonal loading from cache: {fisher_path}")
        sys.stdout.flush()
        fisher_raw  = np.load(fisher_path)
        fisher_max  = float(fisher_raw.max())
        if fisher_max > 0:
            fisher_diag = (fisher_raw / fisher_max).astype(np.float32)
        else:
            fisher_diag = np.ones(teacher_units, dtype=np.float32)
        pf(
            f"Fisher diagonal  min={fisher_diag.min():.6f}  "
            f"max={fisher_diag.max():.6f}  mean={fisher_diag.mean():.6f}"
        )
        sys.stdout.flush()
        return tf.constant(fisher_diag, dtype=tf.float32)

    pf("=" * 60)
    pf("Fisher diagonal NOT found — computing from scratch (runs once)...")
    pf(f"  Requires teacherHidden_L{seq_len}{n_samples}.npy to already exist.")
    pf("=" * 60)
    sys.stdout.flush()

    file_hidden = os.path.join(data_dir, f"teacherHidden_L{seq_len}{n_samples}.npy")
    if not os.path.exists(file_hidden):
        raise FileNotFoundError(
            f"Teacher hidden trajectory not found: {file_hidden}\n"
            f"Run teacher cache step first (cache_teacher_outputs must complete "
            f"before Fisher can be computed)."
        )

    pf(f"[fisher] Loading teacherHidden mmap: {file_hidden}")
    sys.stdout.flush()
    teacher_hidden_traj = np.load(file_hidden, mmap_mode="r")
    pf(f"[fisher] teacherHidden loaded: {teacher_hidden_traj.shape}  dtype={teacher_hidden_traj.dtype}")
    sys.stdout.flush()

    dec_dense = teacher_model.get_layer("dec_dense")

    @tf.function(reduce_retracing=True)
    def fisher_step(hb, res_b):
        with tf.GradientTape() as tape:
            tape.watch(hb)
            y_pred = dec_dense(hb)
            logp   = -tf.reduce_mean(tf.square(y_pred - res_b))
        grads = tape.gradient(logp, hb)
        g2    = tf.reduce_mean(tf.square(grads), axis=[0, 1])
        return g2

    pf("[fisher] Starting fisher_step warm-up trace...")
    sys.stdout.flush()
    wu_size = min(fisher_batch, n_samples)
    h_wu    = tf.constant(teacher_hidden_traj[:wu_size], dtype=tf.float32)
    res_wu  = tf.constant(res[:wu_size],                 dtype=tf.float32)
    _g2     = fisher_step(h_wu, res_wu)
    _       = _g2.numpy()
    pf("[fisher] Fisher warm-up done — trace compiled and executed.")
    del h_wu, res_wu, _g2
    sys.stdout.flush()

    fisher_accum  = np.zeros(teacher_units, dtype=np.float64)
    fisher_nseen  = 0
    n_batches     = int(np.ceil(n_samples / fisher_batch))
    print_every   = max(1, n_batches // 20)

    pf(f"[fisher] Starting Fisher loop: {n_batches} batches  fisher_batch={fisher_batch}  n_samples={n_samples}")
    sys.stdout.flush()

    t0     = time.time()
    t_last = t0

    for b in range(n_batches):
        s  = b * fisher_batch
        e  = min(s + fisher_batch, n_samples)
        bs = e - s

        hb    = tf.constant(teacher_hidden_traj[s:e], dtype=tf.float32)
        res_b = tf.constant(res[s:e],                 dtype=tf.float32)
        g2    = fisher_step(hb, res_b).numpy()
        fisher_accum += g2.astype(np.float64) * bs
        fisher_nseen += bs

        del hb, res_b, g2

        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed   = time.time() - t0
            step_time = time.time() - t_last
            pct       = 100.0 * (b + 1) / n_batches
            eta_s     = (elapsed / max(b + 1, 1)) * (n_batches - b - 1)
            pf(
                f"[fisher] batch {b + 1:>4d}/{n_batches}  "
                f"samples {e:>9,}/{n_samples:,}  "
                f"({pct:5.1f}%)  "
                f"elapsed={elapsed / 60:.1f}min  "
                f"ETA={eta_s / 60:.1f}min  "
                f"step={step_time:.1f}s"
            )
            t_last = time.time()
            sys.stdout.flush()

    del teacher_hidden_traj

    fisher_raw  = (fisher_accum / max(fisher_nseen, 1)).astype(np.float32)
    fisher_max  = float(fisher_raw.max())
    if fisher_max > 0:
        fisher_diag = (fisher_raw / fisher_max).astype(np.float32)
    else:
        fisher_diag = np.ones(teacher_units, dtype=np.float32)

    np.save(fisher_path, fisher_diag)
    pf(f"[fisher] Fisher diagonal saved to {fisher_path}")
    pf(
        f"[fisher] Fisher diagonal  min={fisher_diag.min():.6f}  "
        f"max={fisher_diag.max():.6f}  mean={fisher_diag.mean():.6f}"
    )
    sys.stdout.flush()
    return tf.constant(fisher_diag, dtype=tf.float32)


# ---------------------------------------------------------------------------
# Temporal weights — notebook Cell 15
# t_weights[t] = (t+1) / seq_len  for t in 0..seq_len-1
# Encodes that later timesteps have accumulated more quantisation error.
# ---------------------------------------------------------------------------
def build_temporal_weights(seq_len):
    t_weights = tf.cast(tf.range(1, seq_len + 1), dtype=tf.float32) / float(seq_len)
    return t_weights


# ---------------------------------------------------------------------------
# Materialise split arrays into contiguous float32 RAM buffers
#
# We materialise enc, tgt, t_pred for train and val (these are the three
# smaller arrays).  teacher_hidden_traj is NOT materialised because at
# 8M * 135 * 128 * 4 bytes it can exceed RAM.  It stays mmap and is
# accessed via py_function with a single worker thread to minimise GIL
# contention.
#
# RAM cost (train split ~80% of N):
#   enc   : 6.4M * 135 * 1 * 4B   = ~3.46 GB
#   tgt   : 6.4M * 135 * 3 * 4B   = ~10.4 GB
#   t_pred : 6.4M * 135 * 3 * 4B   = ~10.4 GB
#   Total enc+tgt+t_pred train+val  : ~30 GB  — within cluster RAM budget
# ---------------------------------------------------------------------------
def materialise_enc_tgt_t_pred(
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
    pf(f"  Materialising {label} enc/tgt/t_pred ({n:,} samples) into RAM...")
    t0 = time.time()

    enc   = np.empty((n, seq_len, 1),     dtype=np.float32)
    tgt   = np.empty((n, seq_len, n_out), dtype=np.float32)
    t_pred = np.empty((n, seq_len, n_out), dtype=np.float32)

    chunk = 65536
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        enc[s:e]   = normalized_input[idx[s:e]]
        tgt[s:e]   = res[idx[s:e]]
        t_pred[s:e] = teacher_predictions[idx[s:e]]

    pf(
        f"  Done in {time.time() - t0:.1f}s  "
        f"enc={enc.nbytes / 1e9:.2f} GB  "
        f"tgt={tgt.nbytes / 1e9:.2f} GB  "
        f"t_pred={t_pred.nbytes / 1e9:.2f} GB"
    )
    return enc, tgt, t_pred


# ---------------------------------------------------------------------------
# tf.data pipeline — hybrid fast pipeline
#
# enc, tgt, t_pred are materialised in contiguous RAM → fed via
# from_tensor_slices (pure C++, GIL-free).
# teacher_hidden_traj is accessed via a single py_function worker
# (pipeline_workers=1..4 recommended) to limit GIL impact.
#
# Decoder input is always zeros — same shape as enc.
# ---------------------------------------------------------------------------
def make_student_dataset(
    enc_arr,
    tgt_arr,
    t_pred_arr,
    teacher_hidden_traj,
    idx_arr,
    seq_len,
    n_out,
    teacher_units,
    batch_size,
    shuffle,
    pipeline_workers,
    prefetch_batches,
):
    n = len(enc_arr)
    dec_arr = np.zeros_like(enc_arr)

    # Build integer position index for teacher_hidden_traj lookup
    pos_arr = np.arange(n, dtype=np.int64)

    # Materialised arrays → from_tensor_slices (pure TF C++, no GIL)
    ds_main = tf.data.Dataset.from_tensor_slices((
        enc_arr,   # (N, T, 1)
        dec_arr,   # (N, T, 1)
        tgt_arr,   # (N, T, n_out)
        t_pred_arr, # (N, T, n_out)
        pos_arr,   # (N,) integer positions into teacher_hidden_traj mmap
    ))

    if shuffle:
        ds_main = ds_main.shuffle(
            buffer_size=min(n, 200_000),
            reshuffle_each_iteration=True,
        )

    ds_main = ds_main.batch(batch_size, drop_remainder=True)

    # Fetch teacher hidden from mmap via py_function on a single worker
    # to cap GIL contention. set_shape is critical for MirroredStrategy
    # to know tensor shapes before distribution.
    def fetch_hidden(enc_b, dec_b, tgt_b, t_pred_b, pos_b):
        def _fetch_np(pos_np):
            thid = teacher_hidden_traj[pos_np.numpy()].astype(np.float32)
            return thid

        thid = tf.py_function(
            _fetch_np,
            inp=[pos_b],
            Tout=tf.float32,
        )
        B = batch_size
        enc_b.set_shape([B, seq_len, 1])
        dec_b.set_shape([B, seq_len, 1])
        tgt_b.set_shape([B, seq_len, n_out])
        t_pred_b.set_shape([B, seq_len, n_out])
        thid.set_shape([B, seq_len, teacher_units])

        batchx = {
            "enc_input": enc_b,
            "dec_input": dec_b,
            "t_pred":    t_pred_b,
            "thid":     thid,
        }
        return batchx, tgt_b

    ds_main = ds_main.map(
        fetch_hidden,
        num_parallel_calls=pipeline_workers,
    )
    ds_main = ds_main.prefetch(prefetch_batches)
    return ds_main


# ---------------------------------------------------------------------------
# FW-QATD-RAC loss — notebook Cell 16
# NOTE: @tf.function is NOT placed here. It is placed on the distributed
# wrapper functions below. Placing it here would prevent MirroredStrategy
# from correctly sharding the batch across GPUs.
# ---------------------------------------------------------------------------
def fw_qatd_rac_loss(
    y_true,     # (B, T, n_out)
    y_student,  # (B, T, n_out)
    y_teacher,  # (B, T, n_out)
    h_squant,   # (B, T, student_units)  quantized hidden
    h_sfloat,   # (B, T, student_units)  float shadow hidden
    h_teacher,  # (B, T, teacher_units)
    P,          # Dense(teacher_units, use_bias=False) — projection layer
    F_weights,  # (teacher_units,)  Fisher diagonal
    t_weights,  # (T,)              temporal weights
    alpha,
    beta,
    gamma,
):
    # Ground-truth MSE
    L_GT = tf.reduce_mean(tf.square(y_true - y_student))

    # Output knowledge distillation
    L_KD = tf.reduce_mean(tf.square(y_teacher - y_student))

    # Fisher-weighted trajectory distillation
    h_proj  = P(h_squant)                          # (B, T, teacher_units)
    diff_tr = h_proj - h_teacher                   # (B, T, teacher_units)
    L_traj  = tf.reduce_mean(F_weights * tf.square(diff_tr))

    # Recurrent accumulator consistency
    h_float_sg = tf.stop_gradient(h_sfloat)        # no gradient through float path
    diff_rac   = tf.square(h_squant - h_float_sg)  # (B, T, student_units)
    rac_pert   = tf.reduce_mean(diff_rac, axis=[0, 2])  # (T,)
    L_RAC      = tf.reduce_mean(t_weights * rac_pert)

    total = L_GT + alpha * L_KD + beta * L_traj + gamma * L_RAC
    return total, L_GT, L_KD, L_traj, L_RAC


# ---------------------------------------------------------------------------
# Per-replica train step — NO @tf.function here.
# @tf.function is placed ONLY on the distributed wrapper below.
# Placing @tf.function here prevents batch sharding in TF 2.10.
# ---------------------------------------------------------------------------

def train_step_per_replica(
        batch_x, batch_y,
        student_model, student_hidden_model, float_shadow_model,
        projection_layer, optimizer,
        F_weights, t_weights,
        alpha, beta, gamma):

    enc_b  = batch_x["enc_input"]
    dec_b  = batch_x["dec_input"]
    t_pred = batch_x["t_pred"]
    thid   = batch_x["thid"]
    tgt_b  = batch_y

    h_float = float_shadow_model([enc_b, dec_b], training=False)

    trainable_vars = (student_model.trainable_variables
                      + projection_layer.trainable_variables)

    with tf.GradientTape() as tape:
        y_student = student_model([enc_b, dec_b], training=True)
        h_quant   = student_hidden_model([enc_b, dec_b], training=True)
        total_loss, LGT, LKD, Ltraj, LRAC = fw_qatd_rac_loss(
            tgt_b, y_student, t_pred, h_quant, h_float, thid,
            projection_layer, F_weights, t_weights, alpha, beta, gamma)

    gradients = tape.gradient(total_loss, trainable_vars)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

    nan_in_grads = tf.cast(
        tf.reduce_any(tf.stack(
            [tf.reduce_any(tf.math.is_nan(g)) for g in gradients if g is not None]
        )),
        tf.float32,
    )

    optimizer.apply_gradients(
        [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
    )

    return total_loss, LGT, LKD, Ltraj, LRAC, nan_in_grads

# ---------------------------------------------------------------------------
# Per-replica val step — NO @tf.function here (same reason as train step).
# --------------------------------------------------------------------------
def val_step_per_replica(
    batch_x,
    batch_y,
    student_model,
    student_hidden_model,
    float_shadow_model,
    projection_layer,
    F_weights,
    t_weights,
    alpha,
    beta,
    gamma,
):
    enc_b  = batch_x["enc_input"]
    dec_b  = batch_x["dec_input"]
    t_pred = batch_x["t_pred"]
    t_hid  = batch_x["thid"]
    tgt_b  = batch_y

    h_float   = float_shadow_model([enc_b, dec_b], training=False)
    y_student = student_model([enc_b, dec_b], training=False)
    h_quant   = student_hidden_model([enc_b, dec_b], training=False)

    total_loss, L_GT, L_KD, L_traj, L_RAC = fw_qatd_rac_loss(
        tgt_b, y_student, t_pred,
        h_quant, h_float, t_hid,
        projection_layer, F_weights, t_weights,
        alpha, beta, gamma,
    )
    return total_loss, L_GT, L_KD, L_traj, L_RAC


# ---------------------------------------------------------------------------
# Distributed wrappers — @tf.function placed HERE ONLY on the function
# that calls strategy.run(). This is the correct TF 2.10 pattern.
# strategy.reduce uses MEAN — no manual local/global loss scaling needed.
# ---------------------------------------------------------------------------
def make_distributed_train_step(
        strategy, student_model, student_hidden_model, float_shadow_model,
        projection_layer, optimizer, F_weights, t_weights, alpha, beta, gamma):

    def distributed_train_step(batch_x, batch_y):
        per_replica = strategy.run(
            train_step_per_replica,
            args=(batch_x, batch_y,
                  student_model, student_hidden_model, float_shadow_model,
                  projection_layer, optimizer,
                  F_weights, t_weights,
                  alpha, beta, gamma),
        )

        total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None)
        LGT        = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None)
        LKD        = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None)
        Ltraj      = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None)
        LRAC       = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica[4], axis=None)
        nan_flag   = strategy.reduce(tf.distribute.ReduceOp.SUM,  per_replica[5], axis=None)
        nan_flag   = nan_flag > 0.0

        return total_loss, LGT, LKD, Ltraj, LRAC, nan_flag

    return tf.function(distributed_train_step)




def make_distributed_val_step(
    strategy,
    student_model,
    student_hidden_model,
    float_shadow_model,
    projection_layer,
    F_weights,
    t_weights,
    alpha,
    beta,
    gamma,
):
    def distributed_val_step(batch_x, batch_y):
        per_replica = strategy.run(
            val_step_per_replica,
            args=(
                batch_x, batch_y,
                student_model, student_hidden_model,
                float_shadow_model,
                projection_layer,
                F_weights, t_weights,
                alpha, beta, gamma,
            ),
        )
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[0], axis=None
        )
        L_GT   = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[1], axis=None
        )
        L_KD   = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[2], axis=None
        )
        L_traj = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[3], axis=None
        )
        L_RAC  = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica[4], axis=None
        )
        return total_loss, L_GT, L_KD, L_traj, L_RAC

    return tf.function(distributed_val_step)


# ---------------------------------------------------------------------------
# ReduceLROnPlateau (manual — compatible with MirroredStrategy)
# Handles both plain float and callable LR schedule.
# ---------------------------------------------------------------------------
class ReduceLROnPlateau:
    def __init__(self, optimizer, factor, patience, min_lr, min_delta):
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.min_delta = min_delta
        self.best      = float("inf")
        self.wait      = 0

        # Safely extract current LR regardless of whether it is a schedule
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


# ---------------------------------------------------------------------------
# Progress bar with ETA
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def training_loop(
    strategy,
    student_model,
    student_hidden_model,
    float_shadow_model,
    projection_layer,
    optimizer,
    lr_scheduler,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    F_weights,
    t_weights,
    args,
    job_dir,
    pf,
):
    best_ckpt        = os.path.join(job_dir, "student_best.weights.h5")
    resume_state_path = os.path.join(job_dir, "resume_state.json")

    history     = {"total": [], "GT": [], "KD": [], "traj": [], "RAC": [], "val": []}
    best_val    = float("inf")
    patience_ct = 0
    start_epoch = 0

    nan_warn_threshold = max(1, int(train_steps * 0.10))

    # ------------------------------------------------------------------
    # Resume: load checkpoint state if --resume flag is set and the
    # resume_state.json file exists in the job directory.
    # Restores: epoch counter, best_val, patience_ct, full loss history,
    #           LR scheduler internal state (best, wait, lr_var),
    #           and student model weights from student_best.weights.h5.
    # The optimizer slot variables (Adam m/v) are NOT restored because
    # TF 2.x does not create Adam slots until the first apply_gradients
    # call inside strategy.scope(). Restoring LR is sufficient for
    # practical resume — the slots will re-warm over the first few
    # batches of the resumed epoch.
    # ------------------------------------------------------------------
    if args.resume:
        if not os.path.exists(resume_state_path):
            pf(
                f"[RESUME] --resume flag set but no resume_state.json found at:\n"
                f"  {resume_state_path}\n"
                f"[RESUME] Starting from epoch 1 (fresh training)."
            )
        elif not os.path.exists(best_ckpt):
            pf(
                f"[RESUME] --resume flag set but no student_best.weights.h5 found at:\n"
                f"  {best_ckpt}\n"
                f"[RESUME] Starting from epoch 1 (fresh training)."
            )
        else:
            pf(f"[RESUME] Loading resume state from: {resume_state_path}")
            with open(resume_state_path, "r") as _f:
                _state = json.load(_f)

            start_epoch = int(_state["start_epoch"])
            best_val    = float(_state["best_val"])
            patience_ct = int(_state["patience_ct"])

            _hist = _state["history"]
            history["total"] = [float(v) for v in _hist.get("total", [])]
            history["GT"]    = [float(v) for v in _hist.get("GT",    [])]
            history["KD"]    = [float(v) for v in _hist.get("KD",    [])]
            history["traj"]  = [float(v) for v in _hist.get("traj",  [])]
            history["RAC"]   = [float(v) for v in _hist.get("RAC",   [])]
            history["val"]   = [float(v) for v in _hist.get("val",   [])]

            # Restore LR scheduler internal state
            _lr_val      = float(_state["lr_scheduler_lr"])
            _lr_best     = float(_state["lr_scheduler_best"])
            _lr_wait     = int(_state["lr_scheduler_wait"])
            lr_scheduler.lr_var.assign(_lr_val)
            lr_scheduler.best = _lr_best
            lr_scheduler.wait = _lr_wait

            # Restore student model weights from the best checkpoint
            student_model.load_weights(best_ckpt)

            pf(
                f"[RESUME] Restored state:\n"
                f"  start_epoch = {start_epoch}\n"
                f"  best_val    = {best_val:.8f}\n"
                f"  patience_ct = {patience_ct}/{args.patience}\n"
                f"  lr          = {_lr_val:.6e}\n"
                f"  lr_sched_best = {_lr_best:.8f}  lr_sched_wait = {_lr_wait}\n"
                f"  history epochs already recorded = {len(history['total'])}\n"
                f"  weights loaded from: {best_ckpt}"
            )

            if start_epoch >= args.epochs:
                pf(
                    f"[RESUME] start_epoch ({start_epoch}) >= --epochs ({args.epochs}). "
                    f"Nothing left to train. Returning existing history."
                )
                return history, best_val

    dist_train_step = make_distributed_train_step(
        strategy,
        student_model, student_hidden_model, float_shadow_model,
        projection_layer, optimizer,
        F_weights, t_weights,
        args.alpha, args.beta, args.gamma,
    )
    dist_val_step = make_distributed_val_step(
        strategy,
        student_model, student_hidden_model, float_shadow_model,
        projection_layer,
        F_weights, t_weights,
        args.alpha, args.beta, args.gamma,
    )

    # ------------------------------------------------------------------
    # Pre-flight timing check — detect silent CPU fallback
    # DistributedDataset has no .take() — use next(iter(...))
    # ------------------------------------------------------------------
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
    pf(
        f"{'Resuming' if args.resume and start_epoch > 0 else 'Starting'} "
        f"student training — {args.epochs} epochs total  "
        f"start_epoch={start_epoch}  patience={args.patience}"
    )
    pf(f"  Checkpoint: {best_ckpt}")
    pf(f"  Replicas  : {strategy.num_replicas_in_sync}")
    pf(
        f"  Global BS : {args.batch_size}  "
        f"Per-GPU BS: {args.batch_size // max(strategy.num_replicas_in_sync, 1)}"
    )
    pf("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        t_epoch      = time.time()
        t_batch_zero = None

        # Sync float shadow weights every --shadow-sync-every epochs
        if epoch % args.shadow_sync_every == 0:
            try:
                sync_float_student_weights(student_model, float_shadow_model)
                pf(
                    f"Epoch {epoch + 1}/{args.epochs}  "
                    f"float shadow weights synced (every {args.shadow_sync_every} epoch(s))."
                )
            except Exception as e:
                pf(
                    f"Epoch {epoch + 1}/{args.epochs}  "
                    f"shadow sync FAILED: {e} — continuing without sync"
                )

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        acc_total = acc_GT = acc_KD = acc_traj = acc_RAC = 0.0
        acc_steps = nan_count = 0

        for step, (bx, by) in enumerate(dist_train_dataset):
            if step == 0:
                t_batch_zero = time.time()

            loss, lgt, lkd, ltraj, lrac, nan_flag = dist_train_step(bx, by)

            acc_total += float(loss)
            acc_GT    += float(lgt)
            acc_KD    += float(lkd)
            acc_traj  += float(ltraj)
            acc_RAC   += float(lrac)
            acc_steps += 1

            if bool(nan_flag):
                nan_count += 1

            if (step + 1) % args.log_interval == 0 or (step + 1) == train_steps:
                bar(
                    step + 1,
                    train_steps,
                    {
                        "tot": acc_total / acc_steps,
                        "GT":  acc_GT    / acc_steps,
                        "KD":  acc_KD    / acc_steps,
                        "tr":  acc_traj  / acc_steps,
                        "RAC": acc_RAC   / acc_steps,
                    },
                    epoch_start_time=t_batch_zero if t_batch_zero is not None
                    else t_epoch,
                )

        train_loss = acc_total / max(acc_steps, 1)

        if nan_count > nan_warn_threshold:
            pf(
                f"  *** WARNING: {nan_count}/{acc_steps} batches had NaN gradients "
                f"({100.0 * nan_count / max(acc_steps, 1):.1f}%). "
                f"Consider reducing --lr, --beta, or --gamma. ***"
            )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        val_acc  = val_GT = val_KD = val_traj = val_RAC = 0.0
        val_done = 0

        for bx, by in dist_val_dataset:
            vloss, vlgt, vlkd, vltraj, vlrac = dist_val_step(bx, by)
            val_acc   += float(vloss)
            val_GT    += float(vlgt)
            val_KD    += float(vlkd)
            val_traj  += float(vltraj)
            val_RAC   += float(vlrac)
            val_done  += 1

        val_loss = val_acc / max(val_done, 1)

        history["total"].append(train_loss)
        history["GT"].append(acc_GT    / max(acc_steps, 1))
        history["KD"].append(acc_KD    / max(acc_steps, 1))
        history["traj"].append(acc_traj / max(acc_steps, 1))
        history["RAC"].append(acc_RAC  / max(acc_steps, 1))
        history["val"].append(val_loss)

        elapsed = time.time() - t_epoch
        pf(
            f"Epoch {epoch + 1:3d}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr_scheduler.current_lr:.2e}  "
            f"NaN-batches={nan_count}  "
            f"time={elapsed:.1f}s"
        )
        pf(
            f"  Components — GT={acc_GT / max(acc_steps,1):.6f}  "
            f"KD={acc_KD / max(acc_steps,1):.6f}  "
            f"traj={acc_traj / max(acc_steps,1):.6f}  "
            f"RAC={acc_RAC / max(acc_steps,1):.6f}"
        )

        lr_scheduler.step(val_loss, epoch, pf)

        if val_loss < best_val - args.min_delta:
            best_val    = val_loss
            patience_ct = 0
            student_model.save_weights(best_ckpt)
            pf(f"  ✓ New best val={best_val:.6f}  saved → {best_ckpt}")
        else:
            patience_ct += 1
            pf(f"  patience {patience_ct}/{args.patience}")

        # ------------------------------------------------------------------
        # Save resume state every epoch so any job kill can be recovered.
        # Written AFTER the best-checkpoint logic above so that if a new
        # best was just saved, the resume state reflects that updated best.
        # Fields:
        #   start_epoch       — the NEXT epoch to run on resume
        #   best_val          — best validation loss seen so far
        #   patience_ct       — current early-stopping patience counter
        #   lr_scheduler_lr   — current learning rate value
        #   lr_scheduler_best — ReduceLROnPlateau internal best loss tracker
        #   lr_scheduler_wait — ReduceLROnPlateau internal wait counter
        #   history           — full loss history up to and including this epoch
        # ------------------------------------------------------------------
        _resume_state = {
            "start_epoch":        epoch + 1,
            "best_val":           float(best_val),
            "patience_ct":        int(patience_ct),
            "lr_scheduler_lr":    float(lr_scheduler.current_lr),
            "lr_scheduler_best":  float(lr_scheduler.best),
            "lr_scheduler_wait":  int(lr_scheduler.wait),
            "history": {
                "total": [float(v) for v in history["total"]],
                "GT":    [float(v) for v in history["GT"]],
                "KD":    [float(v) for v in history["KD"]],
                "traj":  [float(v) for v in history["traj"]],
                "RAC":   [float(v) for v in history["RAC"]],
                "val":   [float(v) for v in history["val"]],
            },
        }
        _tmp_resume_path = resume_state_path + ".tmp"
        with open(_tmp_resume_path, "w") as _f:
            json.dump(_resume_state, _f, indent=2)
        os.replace(_tmp_resume_path, resume_state_path)

        if patience_ct >= args.patience:
            pf(f"Early stopping at epoch {epoch + 1}")
            break

    if os.path.exists(best_ckpt):
        student_model.load_weights(best_ckpt)
        pf(f"Restored best weights from {best_ckpt}")

    return history, best_val

# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
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
        preds[s:e] = model(
            {"senc_input": enc_b, "sdec_input": dec_b}, training=False
        ).numpy()
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
    pfn(f"Student FW-QATD-RAC  {job_name}  Test set N={len(test_idx):,}")
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

    return test_metrics


def save_loss_curves(history, best_val_loss, args, job_dir, job_name, pfn):
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    axes[0].plot(history["total"], color="tab:blue",   label="train")
    axes[0].plot(history["val"],   color="tab:orange", linestyle="--", label="val")
    axes[0].set_title("Total Loss (train vs val)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["GT"],   color="tab:blue")
    axes[1].set_title("L_GT (Ground-Truth MSE)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["KD"],   color="tab:orange")
    axes[2].set_title("L_KD (Output KD)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(history["traj"], color="tab:green")
    axes[3].set_title("L_traj (Fisher-Weighted Traj KD)")
    axes[3].set_xlabel("Epoch")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(history["RAC"],  color="tab:red")
    axes[4].set_title("L_RAC (Recurrent Acc. Consistency)")
    axes[4].set_xlabel("Epoch")
    axes[4].grid(True, alpha=0.3)

    axes[5].axis("off")
    axes[5].text(
        0.05, 0.55,
        f"FW-QATD-RAC  SEQLEN={args.seq_len}\n"
        f"α={args.alpha}  β={args.beta}  γ={args.gamma}\n"
        f"Teacher GRU hidden={args.teacher_units} x {args.teacher_layers} layers\n"
        f"Student QGRU hidden={args.student_units}  {args.bits_kernel}-bit kernel\n"
        f"bits: k={args.bits_kernel} r={args.bits_recurrent} "
        f"b={args.bits_bias} a={args.bits_activation} s={args.bits_state}\n"
        f"Batch size={args.batch_size}  GPUs={args.batch_size}\n"
        f"Best val loss={best_val_loss:.6f}\n"
        f"Epochs run={len(history['total'])}",
        fontsize=9,
        verticalalignment="center",
        transform=axes[5].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    curves_path = os.path.join(job_dir, "training_history.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pfn(f"Loss curves saved: {curves_path}")

    csv_path = os.path.join(job_dir, "training_history.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,total,GT,KD,traj,RAC,val\n")
        for i in range(len(history["total"])):
            f.write(
                f"{i + 1},"
                f"{history['total'][i]:.8f},"
                f"{history['GT'][i]:.8f},"
                f"{history['KD'][i]:.8f},"
                f"{history['traj'][i]:.8f},"
                f"{history['RAC'][i]:.8f},"
                f"{history['val'][i]:.8f}\n"
            )
    pfn(f"Training CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    tf.keras.utils.set_random_seed(args.split_seed)
    pf(f"Global random seed set to {args.split_seed}")

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        pf("Mixed precision: float16 enabled")

    # STEP 3 — GPU strategy
    strategy = setup_gpus_and_strategy()

    job_name = make_job_name(args)
    job_dir  = os.path.join(args.save_dir, "results", job_name)
    os.makedirs(job_dir, exist_ok=True)
    pf(f"Job:     {job_name}")
    pf(f"Job dir: {job_dir}")

    # Save args for reproducibility
    with open(os.path.join(job_dir, "student_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # 1. Load data files
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Build teacher (stacked GRUCell/RNN)
    # Teacher is built OUTSIDE strategy.scope() for cache inference —
    # it runs on a single GPU (GPU:0) during the cache pass to avoid
    # MirroredStrategy complications with mmap writes.
    # ------------------------------------------------------------------
    pf("Building teacher model for cache inference (single GPU)...")
    teacher_model        = build_teacher(
        args.seq_len, args.n_out,
        args.teacher_units, args.teacher_layers,
    )
    teacher_hidden_model = build_teacher_hidden_model(teacher_model)

    pf(f"Loading teacher weights from: {args.teacher_ckpt}")
    teacher_model.load_weights(args.teacher_ckpt)
    teacher_model.trainable = False
    teacher_hidden_model.trainable = False
    pf("Teacher weights loaded and frozen.")
    teacher_model.summary(print_fn=pf)

    # ------------------------------------------------------------------
    # 3. Cache teacher predictions + hidden states (checkpoint guard)
    # Computed ONCE and saved to disk. Subsequent runs load from mmap.
    # ------------------------------------------------------------------
    pf("=" * 60)
    pf("PHASE 1: Teacher cache (pred + hidden trajectory)")
    pf("=" * 60)
    teacher_predictions, teacher_hidden_traj = cache_teacher_outputs(
        teacher_model,
        teacher_hidden_model,
        normalized_input,
        args.seq_len,
        args.n_out,
        args.teacher_units,
        n_samples,
        args.infer_batch,
        args.data_dir,
        pf,
    )

    # ------------------------------------------------------------------
    # 4. Fisher diagonal (checkpoint guard)
    # Requires teacherHidden to exist — guaranteed by step 3 above.
    # ------------------------------------------------------------------
    pf("=" * 60)
    pf("PHASE 2: Fisher diagonal")
    pf("=" * 60)
    F_weights = compute_or_load_fisher(
        teacher_model,
        res,
        n_samples,
        args.seq_len,
        args.teacher_units,
        args.fisher_batch,
        args.data_dir,
        pf,
    )

    # ------------------------------------------------------------------
    # 5. Temporal weights
    # ------------------------------------------------------------------
    t_weights = build_temporal_weights(args.seq_len)
    pf(
        f"RAC temporal weights: "
        f"first={float(t_weights[0]):.5f}  last={float(t_weights[-1]):.5f}"
    )

    # ------------------------------------------------------------------
    # 6. Materialise enc / tgt / t_pred for train + val into RAM
    # teacher_hidden_traj stays as mmap — too large for RAM
    # ------------------------------------------------------------------
    pf("=" * 60)
    pf("PHASE 3: Materialising enc/tgt/t_pred into RAM (one-time cost)")
    pf("=" * 60)
    enc_train, tgt_train, t_pred_train = materialise_enc_tgt_t_pred(
        normalized_input, res, teacher_predictions,
        train_idx, args.seq_len, args.n_out, "train", pf,
    )
    enc_val, tgt_val, t_pred_val = materialise_enc_tgt_t_pred(
        normalized_input, res, teacher_predictions,
        val_idx, args.seq_len, args.n_out, "val", pf,
    )
    pf("Materialisation complete.")

    # Build positional index arrays so the pipeline knows which rows
    # of teacher_hidden_traj (mmap) to fetch per batch element
    train_pos = np.arange(len(train_idx), dtype=np.int64)
    val_pos   = np.arange(len(val_idx),   dtype=np.int64)

    # We need the actual original indices into teacher_hidden_traj
    # (which is indexed over all N samples, not just the split)
    train_hidden_idx = train_idx.astype(np.int64)
    val_hidden_idx   = val_idx.astype(np.int64)

    # ------------------------------------------------------------------
    # 7. Build all models inside strategy.scope()
    # ------------------------------------------------------------------
    pf("=" * 60)
    pf("PHASE 4: Building student, shadow, projection, optimizer")
    pf("=" * 60)
    with strategy.scope():
        # Rebuild teacher inside scope so MirroredStrategy can distribute
        # its frozen inference during val (needed for thid in batch_x)
        teacher_model_dist        = build_teacher(
            args.seq_len, args.n_out,
            args.teacher_units, args.teacher_layers,
        )
        teacher_hidden_model_dist = build_teacher_hidden_model(teacher_model_dist)
        teacher_model_dist.load_weights(args.teacher_ckpt)
        teacher_model_dist.trainable = False
        teacher_hidden_model_dist.trainable = False

        student_model = build_student(
            args.seq_len, args.n_out, args.student_units,
            args.bits_kernel, args.bits_recurrent, args.bits_bias,
            args.bits_activation, args.bits_state,
        )
        student_hidden_model = build_student_hidden_model(student_model)
        float_shadow_model   = build_float_shadow_student(
            args.seq_len, args.n_out, args.student_units,
        )

        # Projection layer: student_units → teacher_units
        projection_layer = Dense(
            args.teacher_units,
            use_bias=False,
            name="student_to_teacher_projection",
        )
        # Build projection with a dummy call so weights are created
        dummy_proj = tf.zeros((1, args.seq_len, args.student_units))
        projection_layer(dummy_proj)
        pf(
            f"Projection layer built: {args.student_units} → {args.teacher_units}  "
            f"params={args.student_units * args.teacher_units:,}"
        )

        optimizer    = keras.optimizers.Adam(learning_rate=args.lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor    = args.lr_factor,
            patience  = args.lr_patience,
            min_lr    = args.lr_min,
            min_delta = args.min_delta,
        )

    student_model.summary(print_fn=pf)
    pf(f"Student trainable params: {student_model.count_params():,}")

    # ------------------------------------------------------------------
    # 8. tf.data pipelines — hybrid fast pipeline
    # enc/tgt/t_pred from RAM (GIL-free), thid from mmap (single worker)
    # ------------------------------------------------------------------
    pf("Building hybrid tf.data pipelines...")

    # The dataset fetches thid rows using train_hidden_idx/val_hidden_idx
    # which map each split position back to the original sample index in
    # the full teacher_hidden_traj mmap array.
    # We pass teacher_hidden_traj[train_hidden_idx] lazily via py_function.

    # Build a split-local mmap-indexed version of teacher_hidden_traj
    # by closing over the original mmap array and the index array.
    def make_thid_fetcher(hidden_mmap, original_idx):
        def fetch(pos_np):
            rows = pos_np
            global_rows = original_idx[rows]
            return hidden_mmap[global_rows].astype(np.float32)
        return fetch


    train_thid_fetcher = make_thid_fetcher(teacher_hidden_traj, train_hidden_idx)
    val_thid_fetcher   = make_thid_fetcher(teacher_hidden_traj, val_hidden_idx)

    # We rebuild make_student_dataset to accept per-split thid fetchers
    def build_ds(enc_arr, tgt_arr, t_pred_arr, thid_fetcher, n_split, do_shuffle):
        dec_arr  = np.zeros_like(enc_arr)
        pos_arr  = np.arange(n_split, dtype=np.int64)
        slices   = (enc_arr, dec_arr, tgt_arr, t_pred_arr, pos_arr)
        ds = tf.data.Dataset.from_tensor_slices(slices)
        if do_shuffle:
            ds = ds.shuffle(
                buffer_size=min(n_split, 200000),
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(args.batch_size, drop_remainder=True)

        def np_vec(pos_np):
            rows = pos_np
            return thid_fetcher(rows).astype(np.float32)

        def fetch_thid(batch_enc, batch_dec, batch_tgt, batch_t_pred, batch_pos):
            thid = tf.py_function(
                func=np_vec,
                inp=[batch_pos],
                Tout=tf.float32,
            )
            B = args.batch_size
            batch_enc.set_shape([B, args.seq_len, 1])
            batch_dec.set_shape([B, args.seq_len, 1])
            batch_tgt.set_shape([B, args.seq_len, args.n_out])
            batch_t_pred.set_shape([B, args.seq_len, args.n_out])
            thid.set_shape([B, args.seq_len, args.teacher_units])
            batch_x = {
                "enc_input": batch_enc,
                "dec_input": batch_dec,
                "t_pred":     batch_t_pred,
                "thid":      thid,
            }
            batch_y = batch_tgt
            return batch_x, batch_y

        ds = ds.map(fetch_thid, num_parallel_calls=args.pipeline_workers)
        ds = ds.prefetch(args.prefetch_batches)
        return ds








    train_dataset = build_ds(
        enc_train, tgt_train, t_pred_train, train_thid_fetcher,
        len(train_idx), do_shuffle=True,
    )
    val_dataset = build_ds(
        enc_val, tgt_val, t_pred_val, val_thid_fetcher,
        len(val_idx), do_shuffle=False,
    )

    dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset   = strategy.experimental_distribute_dataset(val_dataset)

    train_steps = len(train_idx) // args.batch_size
    val_steps   = len(val_idx)   // args.batch_size
    pf(f"  Train steps/epoch : {train_steps:,}")
    pf(f"  Val   steps/epoch : {val_steps:,}")

    # ------------------------------------------------------------------
    # 9. Pre-flight diagnostics
    # ------------------------------------------------------------------
    pf("=== PRE-FLIGHT DIAGNOSTICS ===")
    pf(f"Replicas in sync  : {strategy.num_replicas_in_sync}")
    pf(
        f"Global batch size : {args.batch_size}  "
        f"Per-GPU batch size: "
        f"{args.batch_size // max(strategy.num_replicas_in_sync, 1)}"
    )
    pf(
        f"Loss weights      : α={args.alpha}  β={args.beta}  γ={args.gamma}"
    )
    pf(
        f"Quantization bits : kernel={args.bits_kernel}  "
        f"recurrent={args.bits_recurrent}  bias={args.bits_bias}  "
        f"activation={args.bits_activation}  state={args.bits_state}"
    )

    # ------------------------------------------------------------------
    # 10. Training loop
    # ------------------------------------------------------------------
    pf("=" * 60)
    pf("FW-QATD-RAC STUDENT TRAINING")
    pf(f"  Job: {job_name}")
    pf(f"  α={args.alpha}  β={args.beta}  γ={args.gamma}")
    pf(f"  Teacher stacked GRU hidden={args.teacher_units} x {args.teacher_layers}")
    pf(f"  Student QGRU-{args.student_units}  {args.bits_kernel}-bit")
    pf(f"  SEQLEN={args.seq_len}  BATCH={args.batch_size}  EPOCHS={args.epochs}")
    pf("=" * 60)

    history, best_val_loss = training_loop(
        strategy,
        student_model, student_hidden_model, float_shadow_model,
        projection_layer, optimizer, lr_scheduler,
        dist_train_dataset, dist_val_dataset,
        train_steps, val_steps,
        F_weights, t_weights,
        args, job_dir, pf,
    )

    # -----------------------------------------------------------------
    # 11. Save loss curves + CSV
    # ------------------------------------------------------------------
    save_loss_curves(history, best_val_loss, args, job_dir, job_name, pf)

    # ------------------------------------------------------------------
    # 12. Test-set evaluation
    # ------------------------------------------------------------------
    evaluate_and_save(
        student_model, normalized_input, res, labels,
        test_idx, args.seq_len, args.n_out, args.gate_width_ns,
        args.infer_batch, job_dir, job_name, pf,
    )

    # ------------------------------------------------------------------
    # 13. Save final weights (in addition to best checkpoint)
    # ------------------------------------------------------------------
    final_weights_path = os.path.join(job_dir, "student_final.weights.h5")
    student_model.save_weights(final_weights_path)
    pf(f"Final weights saved: {final_weights_path}")

    pf("=" * 60)
    pf(f"DONE — best val loss : {best_val_loss:.6f}")
    pf(f"Results in           : {job_dir}")
    pf("=" * 60)


if __name__ == "__main__":
    main()