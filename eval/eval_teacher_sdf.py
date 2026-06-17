#!/usr/bin/env python3
"""
eval/eval_teacher_sdf.py — Post-hoc SDF-domain metric evaluation for teacher ablations.

Scans /scratch/nmi for all subdirectories containing teacher_best.weights.h5,
loads each checkpoint, runs inference on the SAME frozen test split (testidx.npy),
computes RMSE, R², L2-norm, DTW per SDF channel, and saves test_sdf_metrics.json
next to the weights file.  Also (re)saves test_metrics.json for tau1/tau2/FRET
if it is missing.

Usage:
    python eval/eval_teacher_sdf.py \
        --data-dir /scratch/nmi \
        --results-dir /scratch/nmi \
        --seq-len 135 \
        --n-out 3 \
        --gate-width-ns 0.09 \
        --teacher-units 128 \
        --teacher-layers 2 \
        --infer-batch 8192 \
        --overwrite

    --overwrite   : re-compute even if test_sdf_metrics.json already exists.
                    Without this flag the script skips runs that already have
                    test_sdf_metrics.json.

The script discovers teacher run directories by recursively walking --results-dir
and finding every directory that contains teacher_best.weights.h5.
teacher_args.json is loaded when present so per-run --teacher-units /
--teacher-layers override the CLI defaults automatically.
"""

import argparse
import glob
import json
import os
import sys
import time

# ============================================================
# Force CUDA_VISIBLE_DEVICES before TF import.
# ============================================================
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print(
        "[INIT] CUDA_VISIBLE_DEVICES not set — defaulting to 0,1,2,3,4,5,6,7",
        flush=True,
    )
else:
    print(
        f"[INIT] CUDA_VISIBLE_DEVICES already set: "
        f"{os.environ['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate teacher ablations — SDF metrics for Table 1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",      type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, testidx .npy files.")
    p.add_argument("--results-dir",   type=str, required=True,
                   help="Root directory to walk for teacher_best.weights.h5 files.")
    p.add_argument("--seq-len",       type=int, default=135)
    p.add_argument("--n-out",         type=int, default=3)
    p.add_argument("--gate-width-ns", type=float, default=0.09)
    p.add_argument("--teacher-units", type=int, default=128,
                   help="Default teacher GRU hidden units (overridden by teacher_args.json if found).")
    p.add_argument("--teacher-layers", type=int, default=2,
                   help="Default teacher GRU layers (overridden by teacher_args.json if found).")
    p.add_argument("--infer-batch",   type=int, default=8192)
    p.add_argument("--overwrite",     action="store_true", default=False,
                   help="Re-compute and overwrite test_sdf_metrics.json if it already exists.")
    return p.parse_args()


# ==============================================================================
# GPU setup — single-GPU eval, no MirroredStrategy needed.
# ==============================================================================

def setup_gpu():
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[GPU] No physical GPUs — running on CPU.", flush=True)
        return
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[GPU] set_memory_growth failed for {gpu.name}: {e}", flush=True)
    keras.mixed_precision.set_global_policy("float32")
    print(
        f"[GPU] {len(physical_gpus)} physical GPU(s) detected. "
        f"float32 policy set.",
        flush=True,
    )


# ==============================================================================
# File discovery — glob-based, supports both naming conventions
# ==============================================================================

def find_data_files(data_dir, seq_len):
    def find_one(pattern_globs, desc):
        for pat in pattern_globs:
            matches = glob.glob(os.path.join(data_dir, pat))
            if matches:
                return sorted(matches)[0]
        raise FileNotFoundError(
            f"Cannot find {desc} in {data_dir}. Tried: {pattern_globs}"
        )

    file_input  = find_one([f"tpsf_seq_L{seq_len}_*.npy"],   "encoder input (tpsf_seq)")
    file_res    = find_one([f"res_L{seq_len}_*.npy"],         "decoder target (res)")
    file_labels = find_one([f"labels_3ch_L{seq_len}_*.npy"], "labels (labels_3ch)")

    file_test = None
    for name in ["testidx.npy", "test_idx.npy"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            file_test = candidate
            break
    if file_test is None:
        raise FileNotFoundError(
            f"Test split index not found in {data_dir}. Tried: testidx.npy, test_idx.npy"
        )

    return file_input, file_res, file_labels, file_test


# ==============================================================================
# Teacher model — EXACT replica of train_teacher.py
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
# Inference
# ==============================================================================

def run_inference(model, enc_arr, seq_len, n_out, batch_size, pf):
    n     = len(enc_arr)
    preds = np.zeros((n, seq_len, n_out), dtype=np.float32)
    for s in tqdm(
        range(0, n, batch_size),
        desc="Teacher inference",
        unit="batch",
        bar_format="{l_bar}{bar:30}{r_bar}",
    ):
        e     = min(s + batch_size, n)
        enc_b = tf.constant(enc_arr[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        preds[s:e] = model([enc_b, dec_b], training=False).numpy()
    return preds


# ==============================================================================
# Post-processing helpers
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
    pred_seqs : np.ndarray shape (N, T, C)  — model predictions
    channel_names : list of str, length C   — e.g. ["ch0_full","ch1_short","ch2_long"]
    pfn       : print function

    Returns a dict keyed by channel name, each containing:
        rmse, r2_score, l2_norm, dtw_distance  (all per-sample means)

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


# ==============================================================================
# Discover teacher run directories
# ==============================================================================

def find_teacher_run_dirs(results_dir):
    """
    Walk results_dir recursively and return every directory that contains
    teacher_best.weights.h5.
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        if "teacher_best.weights.h5" in files:
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


# ==============================================================================
# Evaluate a single teacher run directory
# ==============================================================================

def evaluate_teacher_run(
    run_dir,
    normalized_input,
    res,
    labels,
    test_idx,
    seq_len,
    n_out,
    gate_width_ns,
    default_teacher_units,
    default_teacher_layers,
    infer_batch,
    overwrite,
    pf,
):
    sdf_metrics_path = os.path.join(run_dir, "test_sdf_metrics.json")
    if os.path.exists(sdf_metrics_path) and not overwrite:
        pf(f"  SKIP (already exists): {sdf_metrics_path}")
        return

    ckpt_path = os.path.join(run_dir, "teacher_best.weights.h5")
    pf(f"  Checkpoint : {ckpt_path}")

    # Try to read per-run hyper-params from teacher_args.json
    teacher_units  = default_teacher_units
    teacher_layers = default_teacher_layers
    args_path = os.path.join(run_dir, "teacher_args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            run_args = json.load(f)
        teacher_units  = int(run_args.get("teacher_units",  default_teacher_units))
        teacher_layers = int(run_args.get("teacher_layers", default_teacher_layers))
        pf(f"  teacher_args.json: units={teacher_units}  layers={teacher_layers}")
    else:
        pf(
            f"  teacher_args.json not found — using CLI defaults: "
            f"units={teacher_units}  layers={teacher_layers}"
        )

    # Build and load teacher
    tf.keras.backend.clear_session()
    teacher_model = build_teacher(seq_len, n_out, teacher_units, teacher_layers)
    teacher_model.load_weights(ckpt_path)
    teacher_model.trainable = False
    pf(f"  Weights loaded OK.")
    sys.stdout.flush()

    enc_test = normalized_input[test_idx]
    res_test = res[test_idx]
    lab_test = labels[test_idx]

    pf(f"  Running inference on {len(test_idx):,} test samples...")
    sys.stdout.flush()
    teacher_preds = run_inference(
        teacher_model, enc_test, seq_len, n_out, infer_batch, pf
    )
    pf(f"  teacher_preds shape: {teacher_preds.shape}")
    sys.stdout.flush()

    t_ns_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns

    # ── Lifetime metrics (test_metrics.json) ──────────────────────────────────
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(metrics_path) or overwrite:
        tau1_pred, tau2_pred, fret_pred = extract_lifetimes(teacher_preds, t_ns_axis)
        tau1_gt  = lab_test[:, 0]
        tau2_gt  = lab_test[:, 1]
        fret_gt  = lab_test[:, 2]

        pf("  Lifetime metrics:")
        m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
        m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
        mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

        job_name = os.path.basename(run_dir)
        test_metrics = {
            "job_name": job_name,
            "n_test":   int(len(test_idx)),
            "tau1":     {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
            "tau2":     {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
            "fret":     {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
        }
        with open(metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        pf(f"  test_metrics.json saved: {metrics_path}")
        sys.stdout.flush()

    # ── SDF metrics (test_sdf_metrics.json) ───────────────────────────────────
    pf("  SDF-domain metrics (RMSE, R², L2-norm, DTW):")
    sys.stdout.flush()
    sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
    sdf_metrics = compute_sdf_metrics(
        gt_seqs       = res_test.astype(np.float32),
        pred_seqs     = teacher_preds,
        channel_names = sdf_channel_names,
        pfn           = pf,
    )
    job_name = os.path.basename(run_dir)
    sdf_metrics["job_name"] = job_name
    sdf_metrics["n_test"]   = int(len(test_idx))
    with open(sdf_metrics_path, "w") as f:
        json.dump(sdf_metrics, f, indent=2)
    pf(f"  test_sdf_metrics.json saved: {sdf_metrics_path}")
    sys.stdout.flush()


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    setup_gpu()

    pf("=" * 70)
    pf("eval_teacher_sdf.py — Table 1: Teacher ablation SDF metrics")
    pf(f"  data-dir    : {args.data_dir}")
    pf(f"  results-dir : {args.results_dir}")
    pf(f"  overwrite   : {args.overwrite}")
    pf("=" * 70)
    sys.stdout.flush()

    # ── Load shared data (mmap — never materialised fully into RAM) ────────────
    pf("Loading data files (mmap)...")
    file_input, file_res, file_labels, file_test = find_data_files(
        args.data_dir, args.seq_len
    )
    pf(f"  encoder input : {file_input}")
    pf(f"  decoder target: {file_res}")
    pf(f"  labels        : {file_labels}")
    pf(f"  test idx      : {file_test}")
    sys.stdout.flush()

    normalized_input = np.load(file_input,  mmap_mode="r")
    res              = np.load(file_res,    mmap_mode="r")
    labels           = np.load(file_labels, mmap_mode="r")
    test_idx         = np.load(file_test)

    pf(
        f"  N={normalized_input.shape[0]:,}  "
        f"seq_len={args.seq_len}  "
        f"n_out={args.n_out}  "
        f"test_N={len(test_idx):,}"
    )
    sys.stdout.flush()

    # ── Discover all teacher run directories ───────────────────────────────────
    pf(f"Discovering teacher run directories under: {args.results_dir}")
    run_dirs = find_teacher_run_dirs(args.results_dir)
    if not run_dirs:
        pf("ERROR: No directories with teacher_best.weights.h5 found.")
        sys.exit(1)
    pf(f"Found {len(run_dirs)} teacher run(s):")
    for d in run_dirs:
        pf(f"  {d}")
    sys.stdout.flush()

    # ── Evaluate each run ──────────────────────────────────────────────────────
    t_total = time.time()
    for i, run_dir in enumerate(run_dirs):
        pf("")
        pf("=" * 70)
        pf(f"[{i + 1}/{len(run_dirs)}] {run_dir}")
        pf("=" * 70)
        sys.stdout.flush()
        try:
            evaluate_teacher_run(
                run_dir           = run_dir,
                normalized_input  = normalized_input,
                res               = res,
                labels            = labels,
                test_idx          = test_idx,
                seq_len           = args.seq_len,
                n_out             = args.n_out,
                gate_width_ns     = args.gate_width_ns,
                default_teacher_units  = args.teacher_units,
                default_teacher_layers = args.teacher_layers,
                infer_batch       = args.infer_batch,
                overwrite         = args.overwrite,
                pf                = pf,
            )
        except Exception as exc:
            pf(f"  ERROR in {run_dir}: {exc}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    pf("")
    pf("=" * 70)
    pf(
        f"All teacher runs processed.  "
        f"Total elapsed: {(time.time() - t_total) / 60:.1f} min"
    )
    pf("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()