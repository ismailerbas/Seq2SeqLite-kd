#!/usr/bin/env python3
"""
train_teacher.py — Supercomputer-compatible Teacher Seq2Seq GRU Training
Stacked GRUCell inside keras.layers.RNN — matches supercomputer student script.

Architecture:
  Encoder: RNN([GRUCell(u0), GRUCell(u1), ...], return_state=True)  -> "encrnn"
  Decoder: RNN([GRUCell(u0), GRUCell(u1), ...],
               return_sequences=True, return_state=True)             -> "decrnn"
  Dense:   Dense(n_out, activation='linear')                        -> "decdense"
  Input names: "encinput", "decinput"

--teacher-layers-list defines the hidden units for each stacked GRUCell layer.
Examples:
  --teacher-layers-list 128 128   →  LAYERS_TEACHER = [128, 128]  (default, 299 139 params)
  --teacher-layers-list 64 64     →  LAYERS_TEACHER = [64, 64]
  --teacher-layers-list 64 16     →  LAYERS_TEACHER = [64, 16]
  --teacher-layers-list 45 45     →  LAYERS_TEACHER = [45, 45]
  --teacher-layers-list 32 32     →  LAYERS_TEACHER = [32, 32]
  --teacher-layers-list 16 16     →  LAYERS_TEACHER = [16, 16]
  --teacher-layers-list 128       →  LAYERS_TEACHER = [128]  (single layer)

Backward-compatible: --teacher-units and --teacher-layers still work and
produce [teacher_units] * teacher_layers.  If both --teacher-layers-list
and --teacher-units/--teacher-layers are given, --teacher-layers-list wins.

Data files expected in --data-dir:
  tpsf_seq_L{SEQ_LEN}_{N}M.npy     -- encoder input  (N, SEQ_LEN, 1)
  res_L{SEQ_LEN}_{N}M.npy          -- decoder target (N, SEQ_LEN, 3)
  labels_3ch_L{SEQ_LEN}_{N}M.npy   -- labels         (N, 3)
  trainidx.npy / validx.npy / testidx.npy

IMPORTANT: After retraining teacher, delete old student caches before
running train_student.py:
  rm <data-dir>/teacherPred_L135*.npy
  rm <data-dir>/teacherHidden_L135*.npy
  rm <data-dir>/fisherDiag_L135*.npy

Usage:
  python train_teacher.py \
    --data-dir /gpfs/.../nmi \
    --n-total-m 8 \
    --seq-len 135 --n-out 3 \
    --teacher-layers-list 128 128 \
    --batch-size 2048 --epochs 300 --patience 20 \
    --lr 1e-3 --lr-factor 0.5 --lr-patience 8 --lr-min 1e-6 \
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
# This must happen BEFORE `import tensorflow` or it has no effect.
# ============================================================
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(f"[STEP 1] CUDA_VISIBLE_DEVICES not set — defaulting to 0,1,2,3,4,5,6,7",
          flush=True)
else:
    print(f"[STEP 1] CUDA_VISIBLE_DEVICES already set: "
          f"{os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

# Disable TF_FORCE_GPU_ALLOW_GROWTH env var — we handle this in Python
os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GRUCell, Input, RNN
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Teacher seq2seq GRU training (supercomputer-compatible replica)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- paths ---
    p.add_argument("--data-dir",    type=str, required=True,
                   help="Directory containing all .npy data files")
    p.add_argument("--save-dir",    type=str, default=None,
                   help="Root dir for output. Defaults to --data-dir.")
    # --- data ---
    p.add_argument("--n-total-m",   type=int, default=8,
                   help="Nominal millions in filename (e.g. 8 → *_8M.npy). "
                        "Used for file discovery only; actual N comes from array shape.")
    p.add_argument("--seq-len",     type=int, default=135)
    p.add_argument("--n-out",       type=int, default=3)
    p.add_argument("--gate-width-ns", type=float, default=0.09,
                   help="Gate width in ns per time bin (SS3 = 0.09 ns)")
    # --- architecture ---
    # New: heterogeneous per-layer unit counts.  Takes priority over
    # --teacher-units / --teacher-layers when provided.
    p.add_argument("--teacher-layers-list", type=int, nargs="+", default=None,
                   help="Hidden units for each stacked GRUCell layer in order. "
                        "Overrides --teacher-units and --teacher-layers when given. "
                        "Example: --teacher-layers-list 64 16  →  [64, 16]")
    # Legacy args kept for backward compatibility.
    p.add_argument("--teacher-units",  type=int, default=128,
                   help="Hidden units per GRUCell layer (all layers identical). "
                        "Ignored when --teacher-layers-list is given.")
    p.add_argument("--teacher-layers", type=int, default=2,
                   help="Number of stacked GRUCell layers (all identical size). "
                        "Ignored when --teacher-layers-list is given.")
    # --- training ---
    p.add_argument("--batch-size",  type=int,   default=2048)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early stopping patience (epochs)")
    p.add_argument("--min-delta",   type=float, default=1e-5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lr-factor",   type=float, default=0.5)
    p.add_argument("--lr-patience", type=int,   default=8)
    p.add_argument("--lr-min",      type=float, default=1e-6)
    p.add_argument("--split-seed",  type=int,   default=42,
                   help="RNG seed for train/val/test split AND model init")
    p.add_argument("--val-frac",    type=float, default=0.10)
    p.add_argument("--test-frac",   type=float, default=0.10)
    p.add_argument("--log-interval", type=int,  default=10)
    p.add_argument("--infer-batch", type=int,   default=8192)
    p.add_argument("--mixed-precision", action="store_true")
    # --- pipeline tuning ---
    p.add_argument("--pipeline-workers", type=int, default=16,
                   help="num_parallel_calls for tf.data map()")
    p.add_argument("--prefetch-batches", type=int, default=32,
                   help="Number of batches to prefetch ahead of GPU")

    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = args.data_dir

    # Resolve LAYERS_TEACHER here once so every downstream function
    # reads args.layers_teacher — a plain Python list of ints.
    if args.teacher_layers_list is not None:
        if len(args.teacher_layers_list) < 1:
            p.error("--teacher-layers-list must have at least one value")
        for u in args.teacher_layers_list:
            if u < 1:
                p.error(f"--teacher-layers-list: all unit counts must be >= 1, got {u}")
        args.layers_teacher = args.teacher_layers_list
    else:
        if args.teacher_units < 1:
            p.error("--teacher-units must be >= 1")
        if args.teacher_layers < 1:
            p.error("--teacher-layers must be >= 1")
        args.layers_teacher = [args.teacher_units] * args.teacher_layers

    # Build a short human-readable tag that uniquely identifies the architecture.
    # Used for checkpoint filename and job_dir so parallel ablation runs never
    # overwrite each other.
    # Examples:
    #   [128, 128]  →  "gru128x128"
    #   [64, 16]    →  "gru64x16"
    #   [128]       →  "gru128"
    #   [45, 45]    →  "gru45x45"
    args.ckpt_tag = "gru" + "x".join(str(u) for u in args.layers_teacher)

    return args
# ---------------------------------------------------------------------------
# STEP 3 — GPU / Strategy setup with explicit device list
# set_memory_growth MUST be called before ANY other TF GPU operation.
# Then build MirroredStrategy with an explicit device list so all
# N physical GPUs are used — never rely on TF auto-detection alone.
# ---------------------------------------------------------------------------
def setup_gpus_and_strategy():
    # set_memory_growth first — ordering matters
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[STEP 3] No physical GPUs found — running on CPU.", flush=True)
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[STEP 3]   WARNING set_memory_growth failed for {gpu.name}: {e}",
                  flush=True)

    print(f"[STEP 3] Physical GPUs detected : {len(physical_gpus)}", flush=True)
    for i, g in enumerate(physical_gpus):
        print(f"[STEP 3]   GPU {i}: {g.name}", flush=True)

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[STEP 3] Logical GPUs visible   : {len(logical_gpus)}", flush=True)

    if len(logical_gpus) < len(physical_gpus):
        print(f"[STEP 3] WARNING: only {len(logical_gpus)} logical GPUs from "
              f"{len(physical_gpus)} physical. "
              f"Check CUDA_VISIBLE_DEVICES and SLURM --gres=gpu:N", flush=True)

    if len(logical_gpus) == 0:
        print("[STEP 3] No logical GPUs available — falling back to CPU.", flush=True)
        return tf.distribute.get_strategy()

    # Explicit device list — never let MirroredStrategy guess
    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    print(f"[STEP 3] MirroredStrategy: {strategy.num_replicas_in_sync} replicas  "
          f"devices={gpu_devices}", flush=True)

    if strategy.num_replicas_in_sync == 1 and len(logical_gpus) > 1:
        print("[STEP 3] WARNING: MirroredStrategy sees only 1 replica despite "
              f"{len(logical_gpus)} logical GPUs. "
              "Set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 in your SLURM script "
              "BEFORE calling python.", flush=True)

    return strategy


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def find_file(data_dir, patterns, desc):
    for pat in patterns:
        full = os.path.join(data_dir, pat)
        if "*" in pat or "?" in pat:
            matches = sorted(glob.glob(full))
            if matches:
                return matches[0]
        else:
            if os.path.exists(full):
                return full
    raise FileNotFoundError(
        f"Cannot find {desc} in {data_dir}.\n"
        f"Tried patterns: {patterns}"
    )


def load_data(args, pf):
    sl = args.seq_len
    m  = args.n_total_m

    file_input = find_file(
        args.data_dir,
        [f"tpsf_seq_L{sl}_{m}M.npy", f"tpsf_seq_L{sl}_*.npy"],
        "encoder input (tpsf_seq)"
    )
    file_res = find_file(
        args.data_dir,
        [f"res_L{sl}_{m}M.npy", f"res_L{sl}_*.npy"],
        "decoder target (res)"
    )
    file_labels = find_file(
        args.data_dir,
        [f"labels_3ch_L{sl}_{m}M.npy", f"labels_3ch_L{sl}_*.npy"],
        "labels (labels_3ch)"
    )

    pf(f"encoder input  : {file_input}")
    pf(f"decoder target : {file_res}")
    pf(f"labels         : {file_labels}")

    normalized_input = np.load(file_input,  mmap_mode="r")
    res              = np.load(file_res,    mmap_mode="r")
    labels           = np.load(file_labels, mmap_mode="r")

    n_samples = normalized_input.shape[0]

    expected_from_name = m * 1_000_000
    if n_samples != expected_from_name:
        pf(f"WARNING: filename says {m}M but loaded N={n_samples:,}  "
           f"(difference: {n_samples - expected_from_name:+,}). "
           f"Teacher cache files will use actual N={n_samples}.")

    seq_len_data = normalized_input.shape[1]
    assert seq_len_data == args.seq_len, (
        f"--seq-len={args.seq_len} but data has seq_len={seq_len_data}"
    )

    pf(f"normalized_input : {normalized_input.shape}  dtype={normalized_input.dtype}")
    pf(f"res              : {res.shape}  dtype={res.dtype}")
    pf(f"labels           : {labels.shape}  dtype={labels.dtype}")
    pf(f"N={n_samples:,}  seq_len={args.seq_len}  n_out={args.n_out}")

    return normalized_input, res, labels, n_samples


def load_or_create_split_indices(args, n_samples, pf):
    train_path = os.path.join(args.data_dir, "trainidx.npy")
    val_path   = os.path.join(args.data_dir, "validx.npy")
    test_path  = os.path.join(args.data_dir, "testidx.npy")

    if (os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)):
        pf("Loading existing split indices (trainidx / validx / testidx)...")
        train_idx = np.load(train_path)
        val_idx   = np.load(val_path)
        test_idx  = np.load(test_path)
        pf(f"  Train={len(train_idx):,}  Val={len(val_idx):,}  Test={len(test_idx):,}")
        return train_idx, val_idx, test_idx

    pf(f"Split files not found — creating new split (seed={args.split_seed})...")
    rng = np.random.default_rng(args.split_seed)
    idx = rng.permutation(n_samples)

    n_test  = int(n_samples * args.test_frac)
    n_val   = int(n_samples * args.val_frac)
    n_train = n_samples - n_test - n_val

    train_idx = np.sort(idx[:n_train])
    val_idx   = np.sort(idx[n_train:n_train + n_val])
    test_idx  = np.sort(idx[n_train + n_val:])

    np.save(train_path, train_idx)
    np.save(val_path,   val_idx)
    np.save(test_path,  test_idx)
    pf(f"  Saved: trainidx ({len(train_idx):,})  validx ({len(val_idx):,})  "
       f"testidx ({len(test_idx):,})")
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# FAST tf.data pipeline — no py_function, no GIL bottleneck
#
# Materialise split arrays into contiguous float32 RAM buffers once,
# then hand them directly to from_tensor_slices — pure C++ with zero
# Python overhead per batch. Eliminates the 41m ETA caused by py_function
# acquiring the GIL on every batch across 8 GPU replica threads.
#
# RAM cost:
#   enc: 1.28M * 135 * 1 * 4B = ~0.69 GB
#   tgt: 1.28M * 135 * 3 * 4B = ~2.07 GB
#   Total train+val: ~3.5 GB — well within cluster RAM budget.
# ---------------------------------------------------------------------------
def materialise_split(normalized_input, res, idx, seq_len, n_out, label, pf):
    n = len(idx)
    pf(f"  Materialising {label} ({n:,} samples) into RAM...")
    t0 = time.time()

    enc = np.empty((n, seq_len, 1),     dtype=np.float32)
    tgt = np.empty((n, seq_len, n_out), dtype=np.float32)

    # Copy in chunks to avoid stalling the mmap pager
    chunk = 65536
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        enc[s:e] = normalized_input[idx[s:e]]
        tgt[s:e] = res[idx[s:e]]

    pf(f"  Done in {time.time()-t0:.1f}s  "
       f"enc={enc.nbytes/1e9:.2f} GB  tgt={tgt.nbytes/1e9:.2f} GB")
    return enc, tgt


def make_fast_dataset(enc_arr, tgt_arr, batch_size, shuffle, prefetch_batches):
    """
    Pure-TF pipeline — no py_function, no GIL.
    Decoder input is zeros (same shape as enc), built once here.
    """
    dec_arr = np.zeros_like(enc_arr)

    ds = tf.data.Dataset.from_tensor_slices((
        {"encinput": enc_arr, "decinput": dec_arr},
        tgt_arr,
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(enc_arr), 200_000),
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch_batches)
    return ds


# ---------------------------------------------------------------------------
# Teacher model — stacked GRUCell inside RNN (exact notebook match)
# Layer names: encinput, decinput, encrnn, decrnn, decdense
# ---------------------------------------------------------------------------
def build_teacher(seq_len, n_out, layers_teacher):
    """
    Build a stacked GRU Seq2Seq teacher.

    Parameters
    ----------
    seq_len       : int   — sequence length (used only for shape comment; model
                            accepts variable-length via shape=(None, 1))
    n_out         : int   — number of output channels (e.g. 3 for tau1/tau2/fret)
    layers_teacher: list[int] — hidden units per GRUCell layer in stack order.
                    Examples:
                      [128, 128]  →  two-layer 128-unit teacher  (default)
                      [64, 16]    →  heterogeneous paper ablation config
                      [64]        →  single-layer teacher
    """
    encoder_inputs = Input(shape=(None, 1), name="encinput")
    encoder_cells  = [
        GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    encoder_rnn = RNN(encoder_cells, return_state=True, name="encrnn")
    enc_outputs_and_states = encoder_rnn(encoder_inputs)
    # enc_outputs_and_states[0]  : last output  (B, units[-1])
    # enc_outputs_and_states[1:] : hidden states per cell [h0_T, h1_T, ...]
    encoder_states = enc_outputs_and_states[1:]

    decoder_inputs = Input(shape=(None, 1), name="decinput")
    decoder_cells  = [
        GRUCell(units, reset_after=True, name=f"dec_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    decoder_rnn = RNN(
        decoder_cells,
        return_sequences=True,
        return_state=True,
        name="decrnn",
    )
    dec_outputs_and_states = decoder_rnn(
        decoder_inputs, initial_state=encoder_states
    )
    # dec_outputs_and_states[0] : full hidden sequence (B, T, units[-1])
    decoder_hidden_sequence = dec_outputs_and_states[0]

    decoder_output = Dense(n_out, activation="linear", name="decdense")(
        decoder_hidden_sequence
    )

    teacher_model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_output,
        name="teacher_seq2seq",
    )
    return teacher_model


# ---------------------------------------------------------------------------
# ReduceLROnPlateau (manual — compatible with MirroredStrategy)
# ---------------------------------------------------------------------------

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor, patience, min_lr, min_delta):
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.min_delta = min_delta
        self.best      = float("inf")
        self.wait      = 0

        current_lr_val = float(tf.keras.backend.get_value(optimizer.learning_rate))
        self.lr_var = tf.Variable(current_lr_val, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = self.lr_var

    @property
    def current_lr(self):
        # Use get_value() — works for both tf.Variable and MirroredVariable
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
                pfn(f"ReduceLR  {old_lr:.2e} → {new_lr:.2e}  (epoch {epoch+1})")
            self.wait = 0
            return True
        return False



# --------------------------------------------------------------------------
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
            eta_str = f"{remaining/3600:.1f}h"
        elif remaining >= 60:
            eta_str = f"{remaining/60:.0f}m{int(remaining)%60:02d}s"
        else:
            eta_str = f"{remaining:.0f}s"
        elapsed_str = (f"{elapsed/60:.1f}m" if elapsed >= 60
                       else f"{elapsed:.0f}s")
        time_str = f"  [{elapsed_str}<{eta_str}]"
    else:
        time_str = ""

    line = f"\r{step:5}/{total}  {b}  {frac*100:5.1f}%  {stats}{time_str}"
    sys.stdout.write(line)
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Train / val steps
# ---------------------------------------------------------------------------

# NO @tf.function — strategy.run() handles dispatch per-replica.
# @tf.function on train_step prevents batch sharding in TF 2.10.
def train_step(teacher_model, optimizer, batch_x, batch_y):
    with tf.GradientTape() as tape:
        y_pred = teacher_model(batch_x, training=True)
        loss   = tf.reduce_mean(tf.square(batch_y - y_pred))

    grads = tape.gradient(loss, teacher_model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)

    nan_in_grads = tf.reduce_any(tf.stack([
        tf.reduce_any(tf.math.is_nan(g))
        for g in grads if g is not None
    ]))

    optimizer.apply_gradients(zip(grads, teacher_model.trainable_variables))
    return loss, nan_in_grads


# NO @tf.function on val_step either — same reason.
def val_step(teacher_model, batch_x, batch_y):
    y_pred = teacher_model(batch_x, training=False)
    loss   = tf.reduce_mean(tf.square(batch_y - y_pred))
    return loss


def make_distributed_train_step(strategy, teacher_model, optimizer, global_batch_size):
    # @tf.function goes HERE ONLY — on the function that calls strategy.run().
    # This is the correct TF 2.10 MirroredStrategy pattern.
    @tf.function
    def distributed_train_step(batch_x, batch_y):
        per_replica_loss, per_replica_nan = strategy.run(
            train_step,
            args=(teacher_model, optimizer, batch_x, batch_y),
        )
        total_loss = strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
        )
        nan_flag = strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            tf.cast(per_replica_nan, tf.float32),
            axis=None,
        )
        return total_loss, nan_flag > 0.0
    return distributed_train_step


def make_distributed_val_step(strategy, teacher_model):
    @tf.function
    def distributed_val_step(batch_x, batch_y):
        per_replica_loss = strategy.run(
            val_step,
            args=(teacher_model, batch_x, batch_y),
        )
        return strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None
        )
    return distributed_val_step
# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def extract_lifetimes(preds, t):
    ch1  = preds[:, :, 1]
    ch2  = preds[:, :, 2]
    int1 = np.trapz(ch1, t, axis=1)
    int2 = np.trapz(ch2, t, axis=1)
    amp1 = ch1[:, 0]
    amp2 = ch2[:, 0]
    tau1  = np.where(amp1 > 1e-6, int1 / amp1, 0.0).astype(np.float32)
    tau2  = np.where(amp2 > 1e-6, int2 / amp2, 0.0).astype(np.float32)
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


def run_inference(teacher_model, normalized_input, idx, seq_len, n_out,
                  infer_batch, pf):
    n     = len(idx)
    preds = np.zeros((n, seq_len, n_out), dtype=np.float32)
    for s in tqdm(range(0, n, infer_batch), desc="Teacher inference",
                  unit="batch", bar_format="{l_bar}{bar:30}{r_bar}"):
        e     = min(s + infer_batch, n)
        enc_b = tf.constant(normalized_input[idx[s:e]], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        preds[s:e] = teacher_model(
            {"encinput": enc_b, "decinput": dec_b}, training=False
        ).numpy()
    return preds


def evaluate_and_save(teacher_model, normalized_input, res, labels,
                      test_idx, seq_len, n_out, gate_width_ns,
                      infer_batch, job_dir, pf):
    pf("=" * 60)
    pf("Test set evaluation")
    pf("=" * 60)

    teacher_preds = run_inference(
        teacher_model, normalized_input, test_idx,
        seq_len, n_out, infer_batch, pf
    )
    lab_test = labels[test_idx]

    t_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(teacher_preds, t_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    pf("=" * 55)
    pf(f"Teacher Test set  N={len(test_idx):,}")
    pf("=" * 55)
    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

    test_metrics = {
        "n_test": int(len(test_idx)),
        "tau1":   {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
        "tau2":   {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
        "fret":   {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
    }
    with open(os.path.join(job_dir, "teacher_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    panels = [
        (tau1_gt, tau1_pred, m1, "τ₁ (ns)", "GT τ₁ (ns)", "Pred τ₁ (ns)",
         (0, 3.0), "Blues",   "teacher_scatter_tau1.png"),
        (tau2_gt, tau2_pred, m2, "τ₂ (ns)", "GT τ₂ (ns)", "Pred τ₂ (ns)",
         (0, 3.0), "Greens",  "teacher_scatter_tau2.png"),
        (fret_gt, fret_pred, mf, "FRET (f)", "GT FRET (f)", "Pred FRET (f)",
         (0, 1.0), "Oranges", "teacher_scatter_fret.png"),
    ]
    for gt, pred, metrics, title, xlabel, ylabel, lims, cmap, fname in panels:
        rmse_v, r_v, cov_v = metrics
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hb = ax.hexbin(gt, pred, gridsize=80, bins="log", cmap=cmap,
                       extent=(lims[0], lims[1], lims[0], lims[1]), mincnt=1)
        ax.plot(lims, lims, "r--", linewidth=1.5, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Teacher  {title}", fontsize=10, fontweight="bold")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
        ax.text(0.03, 0.97,
                f"RMSE={rmse_v:.4f}\nr={r_v:.4f}\n1σ-cov={cov_v:.1f}%",
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(loc="lower right", fontsize=9)
        fig.colorbar(hb, ax=ax, pad=0.02).set_label("log₁₀(count)", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(job_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"Scatter saved: {fname}")

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
        ax.set_title(f"Teacher Residuals  {label}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.text(0.97, 0.97,
                f"μ={residuals.mean():.4f}\nσ={residuals.std():.4f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.suptitle("Teacher Test Residuals", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, "teacher_residuals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf("Residuals saved: teacher_residuals.png")

    return test_metrics


def save_loss_curves(history, best_val_loss, args, job_dir, pf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_ran = len(history["train"])

    axes[0].plot(history["train"], color="tab:blue",   label="train")
    axes[0].plot(history["val"],   color="tab:orange", linestyle="--", label="val")
    axes[0].set_title("Teacher Training Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axis("off")
    axes[1].text(
        0.05, 0.55,
        f"Teacher seq2seq GRU\n"
        f"LAYERS = {args.layers_teacher}\n"
        f"SEQLEN={args.seq_len}  n_out={args.n_out}\n"
        f"Batch size={args.batch_size}\n"
        f"Best val loss={best_val_loss:.6f}\n"
        f"Epochs run={epochs_ran}",
        fontsize=10, verticalalignment="center",
        transform=axes[1].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig(os.path.join(job_dir, "teacher_training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(job_dir, "teacher_training_history.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        for i in range(epochs_ran):
            f.write(f"{i+1},{history['train'][i]:.8f},{history['val'][i]:.8f}\n")
    pf(f"Loss curves + CSV saved to {job_dir}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def training_loop(
    strategy,
    teacher_model, optimizer, lr_scheduler,
    dist_train_ds, dist_val_ds,
    train_steps, val_steps,
    args, job_dir, pf,
):
    best_ckpt   = os.path.join(args.data_dir, f"teacher_best_{args.ckpt_tag}.weights.h5")
    history     = {"train": [], "val": []}
    best_val    = float("inf")
    patience_ct = 0

    dist_train_step = make_distributed_train_step(
        strategy, teacher_model, optimizer, args.batch_size
    )
    dist_val_step = make_distributed_val_step(strategy, teacher_model)

    # ------------------------------------------------------------------
    # Pre-flight timing check
    # FIX: DistributedDataset has no .take() — use next(iter(...)) instead
    # ------------------------------------------------------------------
    pf("Timing check (forward-only, no weight update)...")
    try:
        sample_batch_x, sample_batch_y = next(iter(dist_train_ds))
        t0 = time.time()
        _ = teacher_model(sample_batch_x, training=False)
        elapsed = time.time() - t0
        status = "OK ✓" if elapsed < 0.5 else "SLOW — check GPU visibility"
        pf(f"  Single forward batch: {elapsed:.3f}s  [{status}]")
        if elapsed > 0.5:
            pf(f"  TIP: Ensure CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
               f"in SLURM script and --gres=gpu:8 is set.")
    except Exception as e:
        pf(f"  Pre-flight check skipped: {e}")

    pf("=" * 60)
    pf(f"Starting teacher training — {args.epochs} epochs  "
       f"patience={args.patience}")
    pf(f"  Checkpoint: {best_ckpt}")
    pf("=" * 60)

    for epoch in range(args.epochs):
        t_epoch      = time.time()
        t_batch_zero = None

        # --- Train ---
        acc_loss  = 0.0
        acc_steps = 0
        nan_count = 0

        for step, (bx, by) in enumerate(dist_train_ds):
            if step == 0:
                t_batch_zero = time.time()

            loss, nan_flag = dist_train_step(bx, by)
            acc_loss  += float(loss)
            acc_steps += 1
            if nan_flag:
                nan_count += 1

            if (step + 1) % args.log_interval == 0 or (step + 1) == train_steps:
                bar(
                    step + 1, train_steps,
                    {"loss": acc_loss / acc_steps, "nan": float(nan_count)},
                    epoch_start_time=t_batch_zero,
                )

        train_loss = acc_loss / max(acc_steps, 1)

        # --- Val ---
        val_acc        = 0.0
        val_steps_done = 0
        for bx, by in dist_val_ds:
            vloss           = dist_val_step(bx, by)
            val_acc        += float(vloss)
            val_steps_done += 1

        val_loss = val_acc / max(val_steps_done, 1)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        elapsed = time.time() - t_epoch
        pf(f"Epoch {epoch+1:3d}/{args.epochs}  "
           f"train={train_loss:.6f}  val={val_loss:.6f}  "
           f"lr={lr_scheduler.current_lr:.2e}  "
           f"NaN-batches={nan_count}  "
           f"time={elapsed:.1f}s")

        lr_scheduler.step(val_loss, epoch, pf)

        if val_loss < best_val - args.min_delta:
            best_val    = val_loss
            patience_ct = 0
            teacher_model.save_weights(best_ckpt)
            pf(f"  ✓ New best val={best_val:.6f}  saved → {best_ckpt}")
        else:
            patience_ct += 1
            pf(f"  patience {patience_ct}/{args.patience}")
            if patience_ct >= args.patience:
                pf(f"Early stopping at epoch {epoch+1}")
                break

    if os.path.exists(best_ckpt):
        teacher_model.load_weights(best_ckpt)
        pf(f"Restored best weights from {best_ckpt}")

    return history, best_val, best_ckpt
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    tf.keras.utils.set_random_seed(args.split_seed)
    pf(f"Global random seed set to {args.split_seed} "
       f"(numpy + python + tensorflow)")

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        pf("Mixed precision: float16 enabled")

    # STEP 3 — GPU strategy (explicit device list, memory growth first)
    strategy = setup_gpus_and_strategy()

    job_dir = os.path.join(args.save_dir, f"teacher_training_{args.ckpt_tag}")
    os.makedirs(job_dir, exist_ok=True)
    pf(f"Job dir: {job_dir}")

    with open(os.path.join(job_dir, "teacher_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # 1. Load data (mmap — no RAM copy yet)
    # ------------------------------------------------------------------
    pf("Loading data (memory-mapped — no RAM copy yet)...")
    normalized_input, res, labels, n_samples = load_data(args, pf)

    # ------------------------------------------------------------------
    # 2. Split indices
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = load_or_create_split_indices(
        args, n_samples, pf
    )

    # ------------------------------------------------------------------
    # 3. Materialise train + val into RAM (eliminates py_function GIL)
    # ------------------------------------------------------------------
    pf("Materialising train/val splits into RAM (one-time cost)...")
    enc_train, tgt_train = materialise_split(
        normalized_input, res, train_idx,
        args.seq_len, args.n_out, "train", pf
    )
    enc_val, tgt_val = materialise_split(
        normalized_input, res, val_idx,
        args.seq_len, args.n_out, "val", pf
    )
    pf("Materialisation complete — GPU data pipeline is now GIL-free.")

    # ------------------------------------------------------------------
    # 4. Build teacher model inside strategy.scope()
    # ------------------------------------------------------------------
    pf("Building teacher model (stacked GRUCell/RNN)...")
    with strategy.scope():
        teacher_model = build_teacher(
            args.seq_len, args.n_out,
            args.layers_teacher,
        )
        optimizer    = keras.optimizers.Adam(learning_rate=args.lr)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor    = args.lr_factor,
            patience  = args.lr_patience,
            min_lr    = args.lr_min,
            min_delta = args.min_delta,
        )

    teacher_model.summary(print_fn=pf)
    pf(f"Teacher params    : {teacher_model.count_params():,}")
    pf(f"LAYERS_TEACHER    : {args.layers_teacher}")
    pf(f"Encoder layer name: {teacher_model.get_layer('encrnn').name}")
    pf(f"Decoder layer name: {teacher_model.get_layer('decrnn').name}")
    pf(f"Dense head name   : {teacher_model.get_layer('decdense').name}")

    # ------------------------------------------------------------------
    # 5. Fast tf.data pipelines (pure-TF, no py_function)
    # ------------------------------------------------------------------
    pf("Building fast tf.data pipelines (no py_function)...")
    train_ds = make_fast_dataset(
        enc_train, tgt_train,
        args.batch_size, shuffle=True,
        prefetch_batches=args.prefetch_batches,
    )
    val_ds = make_fast_dataset(
        enc_val, tgt_val,
        args.batch_size, shuffle=False,
        prefetch_batches=args.prefetch_batches,
    )

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
    dist_val_ds   = strategy.experimental_distribute_dataset(val_ds)

    train_steps = len(train_idx) // args.batch_size
    val_steps   = len(val_idx)   // args.batch_size
    pf(f"  Train steps/epoch: {train_steps:,}")
    pf(f"  Val   steps/epoch: {val_steps:,}")

    # ------------------------------------------------------------------
    # 6. Pre-flight diagnostics
    # ------------------------------------------------------------------
    pf("=== PRE-FLIGHT SPEED DIAGNOSTICS ===")
    pf(f"Replicas in sync  : {strategy.num_replicas_in_sync}")
    pf(f"Global batch size : {args.batch_size}  "
       f"Per-GPU batch size: {args.batch_size // max(strategy.num_replicas_in_sync, 1)}")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    history, best_val_loss, best_ckpt = training_loop(
        strategy,
        teacher_model, optimizer, lr_scheduler,
        dist_train_ds, dist_val_ds,
        train_steps, val_steps,
        args, job_dir, pf,
    )

    # ------------------------------------------------------------------
    # 8. Loss curves
    # ------------------------------------------------------------------
    save_loss_curves(history, best_val_loss, args, job_dir, pf)

    # ------------------------------------------------------------------
    # 9. Test set evaluation
    # ------------------------------------------------------------------
    evaluate_and_save(
        teacher_model, normalized_input, res, labels,
        test_idx, args.seq_len, args.n_out, args.gate_width_ns,
        args.infer_batch, job_dir, pf,
    )

    # ------------------------------------------------------------------
    # 10. Remind user to delete old student caches
    # ------------------------------------------------------------------
    pf("")
    pf("=" * 60)
    pf("DONE. Teacher training complete.")
    pf(f"Best val loss : {best_val_loss:.6f}")
    pf(f"Best weights  : {best_ckpt}")
    pf("")
    pf("IMPORTANT — before running train_student.py, delete old caches:")
    pf(f"  rm {args.data_dir}/teacherPred_L{args.seq_len}*.npy")
    pf(f"  rm {args.data_dir}/teacherHidden_L{args.seq_len}*.npy")
    pf(f"  rm {args.data_dir}/fisherDiag_L{args.seq_len}*.npy")
    pf("Otherwise the student will silently use stale teacher outputs.")
    pf("=" * 60)


if __name__ == "__main__":
    main()