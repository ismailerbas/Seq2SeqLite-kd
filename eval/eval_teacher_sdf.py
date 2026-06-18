#!/usr/bin/env python3
"""
eval/eval_teacher_sdf.py — Post-hoc SDF-domain metric evaluation for teacher ablations.

Scans --results-dir for all subdirectories containing teacher_best*.weights.h5
(glob match — train_teacher.py saves teacher_best_{ckpt_tag}.weights.h5),
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
and finding every directory that contains teacher_best*.weights.h5 (glob).
teacher_args.json is loaded when present so per-run architecture is resolved
automatically from the saved args instead of CLI defaults.
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
    p.add_argument("--data-dir",       type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, testidx .npy files.")
    p.add_argument("--results-dir",    type=str, required=True,
                   help="Root directory to walk for teacher_best*.weights.h5 files.")
    p.add_argument("--seq-len",        type=int, default=135)
    p.add_argument("--n-out",          type=int, default=3)
    p.add_argument("--gate-width-ns",  type=float, default=0.09)
    p.add_argument("--teacher-units",  type=int, default=128,
                   help="Default teacher GRU hidden units (overridden by teacher_args.json if found).")
    p.add_argument("--teacher-layers", type=int, default=2,
                   help="Default teacher GRU layers (overridden by teacher_args.json if found).")
    p.add_argument("--infer-batch",    type=int, default=8192)
    p.add_argument("--overwrite",      action="store_true", default=False,
                   help="Re-compute and overwrite test_sdf_metrics.json if it already exists.")
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
        f"[GPU] {len(physical_gpus)} physical GPU(s) detected. float32 policy set.",
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
# Teacher model — EXACT replica of train_teacher.py
# Layer names MUST match train_teacher.py exactly:
#   encinput, decinput, encrnn, decrnn, decdense
#   enc_cell0, enc_cell1, ..., dec_cell0, dec_cell1, ...
# ==============================================================================

def build_teacher(seq_len, n_out, layers_teacher):
    """
    Build a stacked GRU Seq2Seq teacher — EXACT architecture replica of
    train_teacher.py build_teacher().

    Parameters
    ----------
    seq_len        : int       — sequence length
    n_out          : int       — number of output channels
    layers_teacher : list[int] — hidden units per GRUCell layer in stack order.
                     Examples:
                       [128, 128]  ->  two-layer 128-unit teacher
                       [64, 64]    ->  64x64
                       [64, 16]    ->  heterogeneous paper ablation config
                       [45, 45]    ->  45x45
                       [32, 32]    ->  32x32
                       [16, 16]    ->  16x16
    """
    encoder_inputs = keras.layers.Input(shape=(None, 1), name="encinput")
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells,
        return_state=True,
        name="encrnn",
    )
    encoder_outputs_and_states = encoder_rnn(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

    decoder_inputs = keras.layers.Input(shape=(None, 1), name="decinput")
    decoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"dec_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    decoder_rnn = keras.layers.RNN(
        decoder_cells,
        return_sequences=True,
        return_state=True,
        name="decrnn",
    )
    decoder_outputs_and_states = decoder_rnn(
        decoder_inputs, initial_state=encoder_states
    )
    decoder_hidden_sequence = decoder_outputs_and_states[0]

    decoder_dense = keras.layers.Dense(n_out, activation="linear", name="decdense")
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
        preds[s:e] = model({"encinput": enc_b, "decinput": dec_b}, training=False).numpy()
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
    Compute the 4 paper metrics (Table 1) on raw SDF output sequences.

    gt_seqs   : np.ndarray shape (N, T, C)  — ground truth decoder targets (res)
    pred_seqs : np.ndarray shape (N, T, C)  — model predictions
    channel_names : list of str, length C   — ["ch0_full","ch1_short","ch2_long"]
    pfn       : print function

    Returns a dict keyed by channel name, each containing:
        rmse, r2_score, l2_norm, dtw_distance  (all per-sample means)

    RMSE     : sqrt(mean over samples and timesteps of squared error)
    R²       : 1 - SS_res / SS_tot  (computed sample-wise then meaned)
    L2-norm  : mean over samples of sqrt(sum_t (gt_t - pred_t)^2)
    DTW      : mean over samples of FastDTW distance with scalar abs-diff distance.
               Each row gt_c[i] and pred_c[i] is a 1-D sequence of length T
               (scalars per timestep).  scipy.spatial.distance.euclidean expects
               1-D *vectors* not scalars — using it here raises "Input vector
               should be 1-D".  The correct distance function for scalar
               sequences is abs(float(a) - float(b)).
    """
    N, T, C = gt_seqs.shape
    results = {}

    # Scalar absolute difference — correct dist function for 1-D scalar sequences.
    # fastdtw passes individual elements gt_c[i][t] which are numpy scalars (0-D)
    # when gt_c[i] is shape (T,).  scipy euclidean rejects 0-D inputs.
    # abs(float(a) - float(b)) handles numpy scalars, Python floats, and 0-D arrays.
    scalar_dist = lambda a, b: abs(float(a) - float(b))

    for c, ch_name in enumerate(channel_names):
        gt_c   = gt_seqs[:, :, c]    # (N, T)
        pred_c = pred_seqs[:, :, c]  # (N, T)

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
            # gt_c[i] and pred_c[i] are each 1-D arrays of shape (T,).
            # fastdtw will call scalar_dist(gt_c[i][t1], pred_c[i][t2])
            # where each element is a numpy scalar — scalar_dist handles this correctly.
            dist, _ = fastdtw(gt_c[i], pred_c[i], radius=1, dist=scalar_dist)
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
# Scatter plot helpers
# ==============================================================================

def save_scatter_plots(
    tau1_gt, tau1_pred, m1,
    tau2_gt, tau2_pred, m2,
    fret_gt, fret_pred, mf,
    run_name, run_dir, pf,
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
        ax.set_title(f"{title}  {run_name}", fontsize=10, fontweight="bold")
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
        scatter_path = os.path.join(run_dir, fname)
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"  Scatter saved: {scatter_path}")


def save_residual_plots(
    tau1_gt, tau1_pred,
    tau2_gt, tau2_pred,
    fret_gt, fret_pred,
    run_name, run_dir, pf,
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
    fig.suptitle(f"Residuals  {run_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    residuals_path = os.path.join(run_dir, "test_residuals.png")
    plt.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"  Residuals saved: {residuals_path}")


# ==============================================================================
# Discover all teacher run directories.
# Finds every directory containing teacher_best*.weights.h5 (glob pattern).
# train_teacher.py saves: teacher_best_{ckpt_tag}.weights.h5
# e.g. teacher_best_gru128x128.weights.h5, teacher_best_gru64x16.weights.h5
# ==============================================================================

def find_teacher_run_dirs(results_dir):
    """
    Walk results_dir recursively and return every directory that contains
    a file matching teacher_best*.weights.h5.

    train_teacher.py saves checkpoints as:
        teacher_best_{ckpt_tag}.weights.h5
    where ckpt_tag = "gru" + "x".join(str(u) for u in layers_teacher)
    e.g. gru128x128, gru64x64, gru64x16, gru45x45, gru32x32, gru16x16

    The old code used exact match on "teacher_best.weights.h5" which never
    found any runs.  This glob-based version handles all ckpt_tag variants.
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        ckpt_matches = glob.glob(os.path.join(root, "teacher_best*.weights.h5"))
        if ckpt_matches:
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


def find_checkpoint_in_run_dir(run_dir):
    """
    Return the path to teacher_best*.weights.h5 inside run_dir.
    Raises FileNotFoundError if none found.
    If multiple matches exist, returns the first (sorted alphabetically).
    """
    matches = sorted(glob.glob(os.path.join(run_dir, "teacher_best*.weights.h5")))
    if not matches:
        raise FileNotFoundError(
            f"No teacher_best*.weights.h5 found in {run_dir}"
        )
    return matches[0]


# ==============================================================================
# Resolve architecture from teacher_args.json.
# train_teacher.py saves args.layers_teacher as a list under "layers_teacher".
# It also saves "teacher_units" and "teacher_layers" (legacy flat values) but
# those are only correct when all layers have the same unit count.
# We always prefer "layers_teacher" (list) when it exists.
# ==============================================================================

def resolve_layers_teacher(run_dir, default_teacher_units, default_teacher_layers, pf):
    """
    Load per-run architecture from teacher_args.json saved by train_teacher.py.

    train_teacher.py writes args to teacher_args.json via:
        json.dump(vars(args), f, indent=2)

    Keys checked in order of preference:
        "layers_teacher"   : list[int]  — authoritative source. Present when
                             saved by current train_teacher.py.
        "teacher_units"    : int        — legacy flat value. Only correct when
                             all layers are identical. Ignored when layers_teacher
                             list is present.
        "teacher_layers"   : int        — legacy layer count, same caveat.

    Returns layers_teacher as list[int].
    """
    args_path = os.path.join(run_dir, "teacher_args.json")
    if not os.path.exists(args_path):
        pf(
            f"    teacher_args.json not found — using CLI defaults: "
            f"units={default_teacher_units}  layers={default_teacher_layers}"
        )
        return [default_teacher_units] * default_teacher_layers

    with open(args_path, "r") as f:
        run_args = json.load(f)

    # Prefer "layers_teacher" (list) — authoritative for heterogeneous configs
    if "layers_teacher" in run_args and isinstance(run_args["layers_teacher"], list):
        layers_teacher = [int(u) for u in run_args["layers_teacher"]]
        pf(f"    teacher_args.json: layers_teacher={layers_teacher}  (from list)")
        return layers_teacher

    # Fall back to legacy flat teacher_units + teacher_layers
    teacher_units  = int(run_args.get("teacher_units",  default_teacher_units))
    teacher_layers = int(run_args.get("teacher_layers", default_teacher_layers))
    layers_teacher = [teacher_units] * teacher_layers
    pf(
        f"    teacher_args.json: teacher_units={teacher_units}  "
        f"teacher_layers={teacher_layers}  -> layers_teacher={layers_teacher}  "
        f"(from legacy flat keys)"
    )
    return layers_teacher


# ==============================================================================
# Evaluate one teacher run
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

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    ckpt_path = find_checkpoint_in_run_dir(run_dir)
    pf(f"  Checkpoint : {ckpt_path}")

    # ── Resolve per-run architecture from teacher_args.json ──────────────────
    layers_teacher = resolve_layers_teacher(
        run_dir, default_teacher_units, default_teacher_layers, pf
    )
    pf(f"  layers_teacher resolved: {layers_teacher}")

    # ── Build teacher with EXACT architecture from train_teacher.py ──────────
    tf.keras.backend.clear_session()
    teacher_model = build_teacher(seq_len, n_out, layers_teacher)
    pf(f"  Teacher model built: params={teacher_model.count_params():,}")
    sys.stdout.flush()

    # ── Load weights ─────────────────────────────────────────────────────────
    teacher_model.load_weights(ckpt_path)
    teacher_model.trainable = False
    pf(f"  Weights loaded OK.")
    sys.stdout.flush()

    # ── Batched GPU inference ─────────────────────────────────────────────────
    enc_test = normalized_input[test_idx]
    res_test = res[test_idx]
    lab_test = labels[test_idx]

    pf(f"  Running inference on {len(test_idx):,} test samples...")
    sys.stdout.flush()
    t0_infer = time.time()
    teacher_preds = run_inference(
        teacher_model, enc_test, seq_len, n_out, infer_batch, pf
    )
    pf(
        f"  teacher_preds shape: {teacher_preds.shape}  "
        f"({time.time() - t0_infer:.1f}s)"
    )
    sys.stdout.flush()

    t_ns_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns

    # ── Lifetime metrics ──────────────────────────────────────────────────────
    pf("  Lifetime metrics:")
    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(teacher_preds, t_ns_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

    test_metrics = {
        "run_dir":        run_dir,
        "n_test":         int(len(test_idx)),
        "layers_teacher": layers_teacher,
        "tau1":           {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
        "tau2":           {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
        "fret":           {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
    }
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    pf(f"  test_metrics.json saved: {metrics_path}")
    sys.stdout.flush()

    # ── Scatter and residual plots ────────────────────────────────────────────
    run_name = os.path.basename(os.path.normpath(run_dir))
    pf("  Saving scatter and residual plots...")
    save_scatter_plots(
        tau1_gt, tau1_pred, m1,
        tau2_gt, tau2_pred, m2,
        fret_gt, fret_pred, mf,
        run_name, run_dir, pf,
    )
    save_residual_plots(
        tau1_gt, tau1_pred,
        tau2_gt, tau2_pred,
        fret_gt, fret_pred,
        run_name, run_dir, pf,
    )
    sys.stdout.flush()

    # ── SDF metrics ───────────────────────────────────────────────────────────
    pf("  SDF-domain metrics (RMSE, R², L2-norm, DTW):")
    sys.stdout.flush()
    sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
    sdf_metrics = compute_sdf_metrics(
        gt_seqs       = res_test.astype(np.float32),
        pred_seqs     = teacher_preds,
        channel_names = sdf_channel_names,
        pfn           = pf,
    )
    sdf_metrics["run_dir"]        = run_dir
    sdf_metrics["n_test"]         = int(len(test_idx))
    sdf_metrics["layers_teacher"] = layers_teacher
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

    # ── Discover all teacher run directories ───────────────────────────────────
    pf(f"Discovering teacher run directories under: {args.results_dir}")
    pf(f"  (looking for teacher_best*.weights.h5 in all subdirectories)")
    run_dirs = find_teacher_run_dirs(args.results_dir)
    if not run_dirs:
        pf("ERROR: No directories with teacher_best*.weights.h5 found.")
        pf(
            "  Make sure train_teacher.py has been run and saved "
            "teacher_best_{ckpt_tag}.weights.h5 under --results-dir."
        )
        sys.exit(1)
    pf(f"Found {len(run_dirs)} teacher run(s):")
    for d in run_dirs:
        ckpt = find_checkpoint_in_run_dir(d)
        pf(f"  {d}")
        pf(f"    checkpoint: {os.path.basename(ckpt)}")
    sys.stdout.flush()

    # ── Main loop ─────────────────────────────────────────────────────────────
    t_total = time.time()
    for idx, run_dir in enumerate(run_dirs, 1):
        pf("")
        pf("=" * 70)
        pf(f"[{idx}/{len(run_dirs)}]  {run_dir}")
        pf("=" * 70)
        sys.stdout.flush()
        try:
            evaluate_teacher_run(
                run_dir               = run_dir,
                normalized_input      = normalized_input,
                res                   = res,
                labels                = labels,
                test_idx              = test_idx,
                seq_len               = args.seq_len,
                n_out                 = args.n_out,
                gate_width_ns         = args.gate_width_ns,
                default_teacher_units  = args.teacher_units,
                default_teacher_layers = args.teacher_layers,
                infer_batch           = args.infer_batch,
                overwrite             = args.overwrite,
                pf                    = pf,
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