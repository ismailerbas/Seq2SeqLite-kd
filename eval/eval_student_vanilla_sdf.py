#!/usr/bin/env python3
"""
eval/eval_student_vanilla_sdf.py — Post-hoc SDF-domain metric evaluation
for vanilla KD student ablations (Table 3).

Scans --results-dir for all subdirectories whose name starts with "vanilla_kd"
and that contain student_best.weights.h5.  For each run, loads student_args.json
to reconstruct the exact architecture, runs inference on the frozen test split
(testidx.npy from --data-dir), computes RMSE, R², L2-norm, DTW per SDF channel,
and saves test_sdf_metrics.json next to the weights file.  Also (re)saves
test_metrics.json, scatter PNGs, and residual PNG if missing or --overwrite.

Usage:
    python eval/eval_student_vanilla_sdf.py \
        --data-dir /scratch/nmi \
        --results-dir /scratch/nmi/results \
        --seq-len 135 \
        --n-out 3 \
        --gate-width-ns 0.09 \
        --infer-batch 8192 \
        --overwrite

    --overwrite   : re-compute even if test_sdf_metrics.json already exists.
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
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate vanilla KD student ablations — SDF metrics for Table 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",      type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, testidx .npy files.")
    p.add_argument("--results-dir",   type=str, required=True,
                   help="Root directory to scan for vanilla_kd* subdirectories.")
    p.add_argument("--seq-len",       type=int, default=135)
    p.add_argument("--n-out",         type=int, default=3)
    p.add_argument("--gate-width-ns", type=float, default=0.09)
    p.add_argument("--infer-batch",   type=int, default=8192)
    p.add_argument("--overwrite",     action="store_true", default=False,
                   help="Re-compute and overwrite existing output files.")
    return p.parse_args()


# ==============================================================================
# GPU setup
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
        f"[GPU] {len(physical_gpus)} physical GPU(s). float32 policy set.",
        flush=True,
    )


# ==============================================================================
# File discovery
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
# Student model — EXACT replica of train_student_vanilla_kd.py
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense
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
# Inference
# ==============================================================================

def run_inference(model, enc_arr, seq_len, n_out, batch_size, pf):
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


# ==============================================================================
# Post-processing helpers (identical to train_student_vanilla_kd.py)
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
        rmse, r2_score, l2_norm, dtw_distance  (all per-sample means)

    RMSE     : sqrt(mean over samples and timesteps of squared error)
    R²       : 1 - SS_res / SS_tot  (computed sample-wise then meaned)
    L2-norm  : mean over samples of sqrt(sum_t (gt_t - pred_t)^2)
    DTW      : mean over samples of FastDTW distance (abs scalar distance per step)

    NOTE: fastdtw is called with dist=lambda a, b: abs(float(a) - float(b))
    because the sequences are 1-D float arrays and scipy.spatial.distance.euclidean
    raises ValueError("Input vector should be 1-D.") when called on numpy scalar
    elements on numpy 1.23.x / scipy 1.9.x (the environment used here: TF 2.10.1).
    The lambda is mathematically identical to euclidean for scalar pairs.
    """
    N, T, C = gt_seqs.shape
    results = {}

    def _scalar_dist(a, b):
        return abs(float(a) - float(b))

    for c, ch_name in enumerate(channel_names):
        gt_c   = gt_seqs[:, :, c].astype(np.float64)    # (N, T)  float64 for fastdtw
        pred_c = pred_seqs[:, :, c].astype(np.float64)  # (N, T)

        # ── RMSE ─────────────────────────────────────────────────────────────
        rmse = float(np.sqrt(np.mean((gt_c - pred_c) ** 2)))

        # ── R² ───────────────────────────────────────────────────────────────
        ss_res = np.sum((gt_c - pred_c) ** 2, axis=1)
        ss_tot = np.sum((gt_c - gt_c.mean(axis=1, keepdims=True)) ** 2, axis=1)
        r2_per_sample = np.where(
            ss_tot > 1e-12,
            1.0 - ss_res / ss_tot,
            np.where(ss_res < 1e-12, 1.0, 0.0),
        )
        r2 = float(np.mean(r2_per_sample))

        # ── L2-norm ──────────────────────────────────────────────────────────
        l2_per_sample = np.sqrt(np.sum((gt_c - pred_c) ** 2, axis=1))
        l2_norm = float(np.mean(l2_per_sample))

        # ── DTW ──────────────────────────────────────────────────────────────
        dtw_total = 0.0
        print_every_dtw = max(1, N // 10)
        t0_dtw = time.time()
        pfn(
            f"  [SDF DTW] channel={ch_name}  N={N:,}  T={T}  "
            f"computing FastDTW (radius=1, scalar dist)..."
        )
        sys.stdout.flush()
        for i in range(N):
            dist, _ = fastdtw(gt_c[i], pred_c[i], radius=1, dist=_scalar_dist)
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
# Scatter and residual plot helpers
# ==============================================================================

def save_scatter_plots(
    tau1_gt, tau1_pred, m1,
    tau2_gt, tau2_pred, m2,
    fret_gt, fret_pred, mf,
    job_name, job_dir, pf,
):
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
        pf(f"  Scatter saved: {scatter_path}")


def save_residual_plots(
    tau1_gt, tau1_pred,
    tau2_gt, tau2_pred,
    fret_gt, fret_pred,
    job_name, job_dir, pf,
):
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
    pf(f"  Residuals saved: {residuals_path}")


# ==============================================================================
# Discover vanilla KD student run directories
# ==============================================================================

def find_student_run_dirs(results_dir):
    """
    Walk results_dir recursively and return every directory that:
    1. Has a name starting with "vanilla_kd"
    2. Contains student_best.weights.h5
    3. Contains student_args.json
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        basename = os.path.basename(root)
        if (
            basename.startswith("vanilla_kd")
            and "student_best.weights.h5" in files
            and "student_args.json" in files
        ):
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


# ==============================================================================
# Evaluate a single student run directory
# ==============================================================================

def evaluate_student_run(
    run_dir,
    normalized_input,
    res,
    labels,
    test_idx,
    seq_len,
    n_out,
    gate_width_ns,
    infer_batch,
    overwrite,
    pf,
):
    sdf_metrics_path = os.path.join(run_dir, "test_sdf_metrics.json")
    if os.path.exists(sdf_metrics_path) and not overwrite:
        pf(f"  SKIP (already exists): {sdf_metrics_path}")
        return

    ckpt_path = os.path.join(run_dir, "student_best.weights.h5")
    args_path = os.path.join(run_dir, "student_args.json")
    pf(f"  Checkpoint   : {ckpt_path}")
    pf(f"  student_args : {args_path}")
    sys.stdout.flush()

    with open(args_path, "r") as f:
        run_args = json.load(f)

    student_units    = int(run_args["student_units"])
    bits_kernel      = int(run_args["bits_kernel"])
    bits_recurrent   = int(run_args["bits_recurrent"])
    bits_bias        = int(run_args["bits_bias"])
    bits_activation  = int(run_args["bits_activation"])
    bits_state       = int(run_args["bits_state"])

    pf(
        f"  Architecture: QGRU-{student_units}  "
        f"k={bits_kernel} r={bits_recurrent} b={bits_bias} "
        f"a={bits_activation} s={bits_state}"
    )
    sys.stdout.flush()

    # Build and load student
    tf.keras.backend.clear_session()
    student_model = build_student(
        seq_len         = seq_len,
        n_out           = n_out,
        student_units   = student_units,
        bits_kernel     = bits_kernel,
        bits_recurrent  = bits_recurrent,
        bits_bias       = bits_bias,
        bits_activation = bits_activation,
        bits_state      = bits_state,
    )
    student_model.load_weights(ckpt_path)
    student_model.trainable = False
    pf(f"  Weights loaded OK.")
    sys.stdout.flush()

    enc_test = normalized_input[test_idx]
    res_test = res[test_idx]
    lab_test = labels[test_idx]

    pf(f"  Running inference on {len(test_idx):,} test samples...")
    sys.stdout.flush()
    student_preds = run_inference(
        student_model, enc_test, seq_len, n_out, infer_batch, pf
    )
    pf(f"  student_preds shape: {student_preds.shape}")
    sys.stdout.flush()

    t_ns_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns
    job_name  = os.path.basename(run_dir)

    # ── Lifetime metrics (test_metrics.json) ──────────────────────────────────
    metrics_path      = os.path.join(run_dir, "test_metrics.json")
    scatter_tau1_path = os.path.join(run_dir, "test_scatter_tau1.png")
    scatter_tau2_path = os.path.join(run_dir, "test_scatter_tau2.png")
    scatter_fret_path = os.path.join(run_dir, "test_scatter_fret.png")
    residuals_path    = os.path.join(run_dir, "test_residuals.png")

    need_lifetime = (
        overwrite
        or not os.path.exists(metrics_path)
        or not os.path.exists(scatter_tau1_path)
        or not os.path.exists(scatter_tau2_path)
        or not os.path.exists(scatter_fret_path)
        or not os.path.exists(residuals_path)
    )

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(student_preds, t_ns_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    pf(f"  τ₁ pred range: {tau1_pred.min():.3f} – {tau1_pred.max():.3f} ns")
    pf(f"  τ₂ pred range: {tau2_pred.min():.3f} – {tau2_pred.max():.3f} ns")
    pf(f"  FRET pred range: {fret_pred.min():.3f} – {fret_pred.max():.3f}")

    pf(f"  Lifetime metrics:")
    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

    if need_lifetime:
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

        save_scatter_plots(
            tau1_gt, tau1_pred, m1,
            tau2_gt, tau2_pred, m2,
            fret_gt, fret_pred, mf,
            job_name, run_dir, pf,
        )
        save_residual_plots(
            tau1_gt, tau1_pred,
            tau2_gt, tau2_pred,
            fret_gt, fret_pred,
            job_name, run_dir, pf,
        )
    sys.stdout.flush()

    # ── SDF metrics (test_sdf_metrics.json) ───────────────────────────────────
    pf("  SDF-domain metrics (RMSE, R², L2-norm, DTW):")
    sys.stdout.flush()
    sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
    sdf_metrics = compute_sdf_metrics(
        gt_seqs       = res_test.astype(np.float32),
        pred_seqs     = student_preds,
        channel_names = sdf_channel_names,
        pfn           = pf,
    )
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
    pf("eval_student_vanilla_sdf.py — Table 3: Vanilla KD student SDF metrics")
    pf(f"  data-dir    : {args.data_dir}")
    pf(f"  results-dir : {args.results_dir}")
    pf(f"  overwrite   : {args.overwrite}")
    pf("=" * 70)
    sys.stdout.flush()

    # ── Load shared data (mmap) ────────────────────────────────────────────────
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

    # ── Discover all vanilla_kd student run directories ────────────────────────
    pf(f"Discovering vanilla_kd student run directories under: {args.results_dir}")
    run_dirs = find_student_run_dirs(args.results_dir)
    if not run_dirs:
        pf("ERROR: No vanilla_kd* directories with student_best.weights.h5 + student_args.json found.")
        sys.exit(1)
    pf(f"Found {len(run_dirs)} vanilla_kd student run(s):")
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
            evaluate_student_run(
                run_dir          = run_dir,
                normalized_input = normalized_input,
                res              = res,
                labels           = labels,
                test_idx         = test_idx,
                seq_len          = args.seq_len,
                n_out            = args.n_out,
                gate_width_ns    = args.gate_width_ns,
                infer_batch      = args.infer_batch,
                overwrite        = args.overwrite,
                pf               = pf,
            )
        except Exception as exc:
            pf(f"  ERROR in {run_dir}: {exc}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    pf("")
    pf("=" * 70)
    pf(
        f"All vanilla_kd student runs processed.  "
        f"Total elapsed: {(time.time() - t_total) / 60:.1f} min"
    )
    pf("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()