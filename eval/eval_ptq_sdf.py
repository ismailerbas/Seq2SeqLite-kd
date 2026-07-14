#!/usr/bin/env python3
"""
eval/eval_ptq_sdf.py — Post-training weight-only quantization (PTQ) evaluation
for Table 2.

Auto-discovers all teacher runs under --results-dir (same walk logic as
eval_teacher_sdf.py), applies symmetric per-tensor weight-only quantization
at --bits (8, 16, or "both"), runs batched GPU inference on the frozen test
split (testidx.npy), computes RMSE, R², L2-norm, DTW per SDF channel, and
saves results into:

    <results-dir>/ptq/ptq_{bits}bit_{run_name}/
        test_sdf_metrics.json
        test_metrics.json
        test_scatter_tau1.png
        test_scatter_tau2.png
        test_scatter_fret.png
        test_residuals.png
        ptq_args.json

Weight-only quantization:
    Symmetric per-tensor min-max quantization is applied to every weight
    tensor (kernel, recurrent_kernel, bias) in every layer of the teacher
    model.  The quantized float32 values are written back via layer.set_weights()
    and inference runs at full float32 speed on GPU using the normal Keras path.
    This measures the accuracy degradation from storing weights at reduced
    precision, which is the standard definition of PTQ weight-only quantization
    used in the paper.

    8-bit:  scale = max(|w|) / 127   (INT8 symmetric, 127 levels)
    16-bit: scale = max(|w|) / 32767 (INT16 symmetric, 32767 levels)

    Weights with max(|w|) < 1e-8 are left unchanged (zero/near-zero tensors).
    After quantization the weights are dequantized back to float32 before being
    set on the layer, so inference runs in float32 throughout — exactly as
    weight-only PTQ works on hardware that dequantizes at runtime.

Usage:
    # Both 8-bit and 16-bit in one job (default):
    python eval/eval_ptq_sdf.py \
        --data-dir /scratch/nmi \
        --results-dir /scratch/nmi \
        --seq-len 135 \
        --n-out 3 \
        --gate-width-ns 0.09 \
        --teacher-units 128 \
        --teacher-layers 2 \
        --infer-batch 8192 \
        --bits both \
        --overwrite

    # Only 8-bit:
    python eval/eval_ptq_sdf.py \
        --data-dir /scratch/nmi \
        --results-dir /scratch/nmi \
        --bits 8 \
        --seq-len 135 --n-out 3 --gate-width-ns 0.09 \
        --teacher-units 128 --teacher-layers 2 \
        --infer-batch 8192

    # Only 16-bit:
    python eval/eval_ptq_sdf.py \
        --data-dir /scratch/nmi \
        --results-dir /scratch/nmi \
        --bits 16 \
        --seq-len 135 --n-out 3 --gate-width-ns 0.09 \
        --teacher-units 128 --teacher-layers 2 \
        --infer-batch 8192

    --overwrite : re-compute even if test_sdf_metrics.json already exists.

The script discovers teacher run directories by recursively walking --results-dir
and finding every directory that contains teacher_best.weights.h5.
teacher_args.json is loaded when present so per-run --teacher-units /
--teacher-layers override the CLI defaults automatically.
"""

import argparse
import copy
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
from scipy.stats import pearsonr
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras as keras


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PTQ weight-only evaluation — Table 2 (8-bit and 16-bit).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",       type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, testidx .npy files.")
    p.add_argument("--results-dir",    type=str, required=True,
                   help="Root directory to walk for teacher_best.weights.h5 files.")
    p.add_argument("--bits",           type=str, default="both",
                   choices=["8", "16", "both"],
                   help="Bit-width to evaluate: 8, 16, or both.")
    p.add_argument("--seq-len",        type=int, default=135)
    p.add_argument("--n-out",          type=int, default=3)
    p.add_argument("--gate-width-ns",  type=float, default=0.09)
    p.add_argument("--teacher-units",  type=int, default=128,
                   help="Default teacher GRU hidden units (overridden by teacher_args.json).")
    p.add_argument("--teacher-layers", type=int, default=2,
                   help="Default teacher GRU layers (overridden by teacher_args.json).")
    p.add_argument("--infer-batch",    type=int, default=8192)
    p.add_argument("--overwrite",      action="store_true", default=False,
                   help="Re-compute and overwrite existing test_sdf_metrics.json.")
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
# Stacked GRUCell inside keras.layers.RNN
# Layer names: enc_input, dec_input, enc_rnn, dec_rnn, dec_dense
# ==============================================================================

def build_teacher(seq_len, n_out, layers_teacher, use_old_names=False):
    """
    Build the teacher Seq2Seq GRU model.

    Parameters
    ----------
    seq_len        : int        — sequence length (informational; model uses None).
    n_out          : int        — number of output channels.
    layers_teacher : list[int]  — hidden units per GRUCell layer in stack order.
                                  Examples: [128, 128], [64, 16], [64, 32], [128].
    use_old_names  : bool       — when True uses legacy layer names (encrnn/decrnn/
                                  decdense/encinput/decinput) matching old checkpoint
                                  files.  When False uses new names (enc_rnn/dec_rnn/
                                  dec_dense/enc_input/dec_input).
    """
    if use_old_names:
        enc_input_name = "encinput"
        dec_input_name = "decinput"
        enc_rnn_name   = "encrnn"
        dec_rnn_name   = "decrnn"
        dec_dense_name = "decdense"
    else:
        enc_input_name = "enc_input"
        dec_input_name = "dec_input"
        enc_rnn_name   = "enc_rnn"
        dec_rnn_name   = "dec_rnn"
        dec_dense_name = "dec_dense"

    encoder_inputs = keras.layers.Input(shape=(None, 1), name=enc_input_name)
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells,
        return_state=True,
        name=enc_rnn_name,
    )
    encoder_outputs_and_states = encoder_rnn(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]

    decoder_inputs = keras.layers.Input(shape=(None, 1), name=dec_input_name)
    decoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"dec_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    decoder_rnn = keras.layers.RNN(
        decoder_cells,
        return_sequences=True,
        return_state=True,
        name=dec_rnn_name,
    )
    decoder_outputs_and_states = decoder_rnn(
        decoder_inputs, initial_state=encoder_states
    )
    decoder_hidden_sequence = decoder_outputs_and_states[0]

    decoder_dense = keras.layers.Dense(n_out, activation="linear", name=dec_dense_name)
    decoder_output = decoder_dense(decoder_hidden_sequence)

    teacher_model = keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_output,
        name="teacher_seq2seq",
    )
    return teacher_model

# ==============================================================================
# Weight-only symmetric per-tensor quantization
# ==============================================================================

def quantize_weights_symmetric(w, bits):
    """
    Symmetric per-tensor min-max quantization.

    Maps the float32 weight tensor w to a quantized grid of (2^(bits-1) - 1)
    levels, then dequantizes back to float32.  The result has the same dtype
    as the input (float32) but its values are constrained to the quantized grid
    defined by scale = max(|w|) / n_levels.

    Args:
        w    : np.ndarray of any shape, dtype float32.
        bits : int — 8 for INT8-equivalent, 16 for INT16-equivalent.

    Returns:
        np.ndarray of same shape and dtype as w with quantized values.
    """
    n_levels = (2 ** (bits - 1)) - 1        # INT8: 127   INT16: 32767
    w_max = float(np.max(np.abs(w)))
    if w_max < 1e-8:
        return w.copy()                      # zero/near-zero — leave unchanged
    scale = w_max / n_levels
    w_q = np.clip(np.round(w / scale), -n_levels, n_levels)
    return (w_q * scale).astype(np.float32)  # dequantize back to float32


def apply_weight_quantization(model, bits, pf):
    """
    Apply symmetric per-tensor weight-only quantization to every weight tensor
    in every layer of model in-place via layer.set_weights().

    Only layers that have weights are touched.  The model is modified in-place.
    A summary of layers modified and their weight shapes is printed.

    Args:
        model : tf.keras.Model — the loaded teacher model.
        bits  : int — 8 or 16.
        pf    : callable print function.

    Returns:
        n_tensors_quantized : int — total number of weight tensors quantized.
        n_params_quantized  : int — total number of scalar parameters quantized.
    """
    n_tensors_quantized = 0
    n_params_quantized  = 0

    pf(f"  Applying {bits}-bit weight-only quantization to all layers...")
    sys.stdout.flush()

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        quantized_weights = []
        layer_tensors = 0
        layer_params  = 0

        for w in weights:
            w_arr = np.array(w, dtype=np.float32)
            w_q   = quantize_weights_symmetric(w_arr, bits)
            quantized_weights.append(w_q)
            layer_tensors += 1
            layer_params  += w_arr.size

        layer.set_weights(quantized_weights)
        n_tensors_quantized += layer_tensors
        n_params_quantized  += layer_params

        pf(
            f"    layer={layer.name:30s}  "
            f"tensors={layer_tensors}  "
            f"params={layer_params:>9,}  "
            f"shapes={[list(w.shape) for w in weights]}"
        )
        sys.stdout.flush()

    pf(
        f"  Quantization complete: "
        f"{n_tensors_quantized} tensors  "
        f"{n_params_quantized:,} parameters quantized at {bits}-bit."
    )
    sys.stdout.flush()
    return n_tensors_quantized, n_params_quantized


# ==============================================================================
# Batched GPU inference — identical to eval_teacher_sdf.py
# ==============================================================================

def run_inference(model, enc_arr, seq_len, n_out, batch_size, pf):
    n     = len(enc_arr)
    preds = np.zeros((n, seq_len, n_out), dtype=np.float32)
    # Detect input names from the model itself
    enc_input_name = model.input_names[0]
    dec_input_name = model.input_names[1]
    for s in tqdm(
        range(0, n, batch_size),
        desc="PTQ inference",
        unit="batch",
        bar_format="{l_bar}{bar:30}{r_bar}",
    ):
        e     = min(s + batch_size, n)
        enc_b = tf.constant(enc_arr[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        preds[s:e] = model(
            {enc_input_name: enc_b, dec_input_name: dec_b}, training=False
        ).numpy()
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
    N, T, C = gt_seqs.shape
    results = {}

    for c, ch_name in enumerate(channel_names):
        gt_c   = gt_seqs[:, :, c]
        pred_c = pred_seqs[:, :, c]

        rmse = float(np.sqrt(np.mean((gt_c - pred_c) ** 2)))

        ss_res = np.sum((gt_c - pred_c) ** 2, axis=1)
        ss_tot = np.sum((gt_c - gt_c.mean(axis=1, keepdims=True)) ** 2, axis=1)
        r2_per_sample = np.where(
            ss_tot > 1e-12,
            1.0 - ss_res / ss_tot,
            np.where(ss_res < 1e-12, 1.0, 0.0),
        )
        r2 = float(np.mean(r2_per_sample))

        l2_per_sample = np.sqrt(np.sum((gt_c - pred_c) ** 2, axis=1))
        l2_norm = float(np.mean(l2_per_sample))

        results[ch_name] = {
            "rmse":     rmse,
            "r2_score": r2,
            "l2_norm":  l2_norm,
        }

        pfn(
            f"  SDF {ch_name:12s}  RMSE={rmse:.4f}  R²={r2:.4f}  "
            f"L2={l2_norm:.4f}"
        )
        sys.stdout.flush()

    return results

# ==============================================================================
# Scatter plot helper
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
# Discover all teacher run directories — identical walk to eval_teacher_sdf.py
# ==============================================================================

def find_teacher_run_dirs(results_dir):
    """
    Walk results_dir recursively and return every directory that contains
    teacher_best.weights.h5 AND whose basename starts with 'teacher_training_'.

    Directories such as the root results_dir itself, 'm740bp',
    'teacher_gru128x128 12vials', 'teklayer_gru128', etc. are intentionally
    excluded.  Only teacher_training_* runs are valid PTQ targets.
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        basename = os.path.basename(os.path.normpath(root))
        if not basename.startswith("teacher_training_"):
            continue
        if any(f.startswith("teacher_best") and f.endswith(".weights.h5") for f in files):
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


# ==============================================================================
# Evaluate one teacher run at one bit-width
# ==============================================================================

def evaluate_one_run_at_bits(
    run_dir,
    bits,
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
    save_root,
    overwrite,
    pf,
):
    """
    Load teacher from run_dir, quantize weights to `bits`, run batched GPU
    inference on test_idx, compute and save all metrics.

    Results are saved into:
        save_root/ptq_{bits}bit_{run_basename}/
            test_sdf_metrics.json
            test_metrics.json
            test_scatter_tau1.png
            test_scatter_tau2.png
            test_scatter_fret.png
            test_residuals.png
            ptq_args.json

    Args:
        run_dir               : str — full path to teacher run directory.
        bits                  : int — 8 or 16.
        normalized_input      : np.ndarray mmap (N, T, 1)
        res                   : np.ndarray mmap (N, T, 3)
        labels                : np.ndarray mmap (N, 3)
        test_idx              : np.ndarray (n_test,)
        seq_len               : int
        n_out                 : int
        gate_width_ns         : float
        default_teacher_units : int
        default_teacher_layers: int
        infer_batch           : int
        save_root             : str — root dir under which ptq subdirs are created.
        overwrite             : bool
        pf                    : callable print function.
    """
    run_basename = os.path.basename(os.path.normpath(run_dir))
    job_name     = f"ptq_{bits}bit_{run_basename}"
    job_dir      = os.path.join(save_root, job_name)
    os.makedirs(job_dir, exist_ok=True)

    sdf_metrics_path = os.path.join(job_dir, "test_sdf_metrics.json")
    if os.path.exists(sdf_metrics_path) and not overwrite:
        pf(f"    SKIP (already exists): {sdf_metrics_path}")
        return


    ckpt_matches = sorted(
        f for f in os.listdir(run_dir)
        if f.startswith("teacher_best") and f.endswith(".weights.h5")
    )
    if not ckpt_matches:
        raise FileNotFoundError(f"No teacher_best*.weights.h5 found in {run_dir}")
    ckpt_path = os.path.join(run_dir, ckpt_matches[0])

    pf(f"    Checkpoint : {ckpt_path}")
    pf(f"    Output dir : {job_dir}")

    # ── Resolve per-run hyper-params ─────────────────────────────────────────
    # Priority 1: parse units from the checkpoint filename itself.
    # teacher_best_gru128x128.weights.h5  → units=128
    # teacher_best_gru64x16.weights.h5    → units=64  (encoder units; asymmetric runs)
    # teacher_best_gru16x16.weights.h5    → units=16
    # teacher_best_gru45x45.weights.h5    → units=45
    # teacher_best_gru128.weights.h5      → units=128 (single number, symmetric)
    # teacher_best.weights.h5             → no number → fall through to JSON / CLI default
    import re as _re
    import h5py

    # ── Resolve layers_teacher list ──────────────────────────────────────────
    # Priority 1: teacher_args.json contains the authoritative layers_teacher
    # list written by train_teacher.py at training time.  This is the only
    # source that correctly encodes heterogeneous architectures like [64, 16]
    # or [64, 32].  Always read it first.
    layers_teacher = None
    args_path = os.path.join(run_dir, "teacher_args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            run_args = json.load(f)
        _lt = run_args.get("layers_teacher", None)
        if _lt is not None and isinstance(_lt, list) and len(_lt) >= 1:
            layers_teacher = [int(u) for u in _lt]
            pf(
                f"    teacher_args.json layers_teacher: {layers_teacher} "
                f"(authoritative source)"
            )
        else:
            # teacher_args.json exists but has no layers_teacher key —
            # fall back to legacy teacher_units / teacher_layers fields.
            _units  = int(run_args.get("teacher_units",  default_teacher_units))
            _layers = int(run_args.get("teacher_layers", default_teacher_layers))
            layers_teacher = [_units] * _layers
            pf(
                f"    teacher_args.json (legacy keys): units={_units}  "
                f"layers={_layers}  -> layers_teacher={layers_teacher}"
            )
    else:
        pf(
            f"    teacher_args.json NOT FOUND in {run_dir}"
        )

    # Priority 2: if teacher_args.json was absent, parse the checkpoint
    # filename for gruAxB or gruA patterns as a best-effort fallback.
    # NOTE: gruAxB only encodes the first-layer units and total layer count
    # from the filename alone — this is ambiguous for heterogeneous models.
    # teacher_args.json should always be present for correct results.
    if layers_teacher is None:
        ckpt_basename = os.path.basename(ckpt_path)
        _m = _re.search(r"gru(\d+)(?:x(\d+))?", ckpt_basename)
        if _m:
            _units = int(_m.group(1))
            # gruAxB: B is either a second layer size or a layer count.
            # We cannot distinguish here — assume symmetric [A]*N where N
            # is derived from B if B >= units (layer count interpretation)
            # or [A, B] if B < A (heterogeneous interpretation).
            # This is inherently ambiguous without teacher_args.json.
            if _m.group(2) is not None:
                _second = int(_m.group(2))
                if _second < _units:
                    # e.g. gru64x16 — heterogeneous [64, 16]
                    layers_teacher = [_units, _second]
                else:
                    # e.g. gru128x128 or gru64x64 — symmetric, interpret as
                    # [units, units] (2 layers of same size)
                    layers_teacher = [_units, _second]
            else:
                # e.g. gru128 — single number means one layer
                layers_teacher = [_units]
            pf(
                f"    Checkpoint filename fallback: '{os.path.basename(ckpt_path)}' "
                f"-> layers_teacher={layers_teacher}  "
                f"(WARNING: teacher_args.json missing, result may be wrong)"
            )
        else:
            # Last resort: use CLI defaults
            layers_teacher = [default_teacher_units] * default_teacher_layers
            pf(
                f"    No gru pattern in checkpoint filename and no teacher_args.json "
                f"— using CLI default layers_teacher={layers_teacher}  "
                f"(WARNING: may be wrong)"
            )

    pf(f"    Final layers_teacher: {layers_teacher}")
    sys.stdout.flush()

    # ── Save ptq_args.json ───────────────────────────────────────────────────
    ptq_args = {
        "run_dir":         run_dir,
        "run_basename":    run_basename,
        "bits":            bits,
        "layers_teacher":  layers_teacher,
        "seq_len":         seq_len,
        "n_out":           n_out,
        "gate_width_ns":   gate_width_ns,
        "infer_batch":     infer_batch,
        "n_test":          int(len(test_idx)),
    }
    with open(os.path.join(job_dir, "ptq_args.json"), "w") as f:
        json.dump(ptq_args, f, indent=2)
    pf(f"    ptq_args.json saved.")
    sys.stdout.flush()

    # ── Detect checkpoint layer naming convention ────────────────────────────
    # Old train_teacher.py: encrnn / decrnn / decdense / encinput / decinput
    # Current train_teacher.py: same old names — train_teacher.py has always
    # used old-style names.  The HDF5 detection is kept for safety.
    def _detect_layer_name_convention(ckpt_path):
        with h5py.File(ckpt_path, "r") as f:
            top_keys = list(f.keys())
            if any("encrnn" in k or "decrnn" in k or "decdense" in k for k in top_keys):
                return "old"
            return "new"

    convention = _detect_layer_name_convention(ckpt_path)
    pf(f"    Checkpoint layer naming convention: {convention}")

    # ── Build teacher with the correct heterogeneous layers_teacher list ─────
    tf.keras.backend.clear_session()
    teacher_model = build_teacher(
        seq_len,
        n_out,
        layers_teacher,
        use_old_names=(convention == "old"),
    )
    teacher_model.load_weights(ckpt_path)
    teacher_model.trainable = False
    pf(f"    Float32 weights loaded OK.  layers_teacher={layers_teacher}")
    sys.stdout.flush()

    # ── Apply weight-only quantization in-place ──────────────────────────────
    pf(f"    Applying {bits}-bit weight-only quantization...")
    sys.stdout.flush()
    t0_quant = time.time()
    n_tensors, n_params = apply_weight_quantization(teacher_model, bits, pf)
    pf(
        f"    Quantization done in {time.time() - t0_quant:.2f}s  "
        f"({n_tensors} tensors  {n_params:,} params)"
    )
    sys.stdout.flush()

    # ── Batched GPU inference ────────────────────────────────────────────────
    enc_test = normalized_input[test_idx]
    res_test = res[test_idx]
    lab_test = labels[test_idx]

    pf(f"    Running batched GPU inference on {len(test_idx):,} test samples...")
    sys.stdout.flush()
    t0_infer = time.time()
    ptq_preds = run_inference(
        teacher_model, enc_test, seq_len, n_out, infer_batch, pf
    )
    pf(
        f"    Inference done in {time.time() - t0_infer:.1f}s  "
        f"shape={ptq_preds.shape}"
    )
    sys.stdout.flush()

    t_ns_axis = np.arange(seq_len, dtype=np.float32) * gate_width_ns

    # ── Lifetime metrics ─────────────────────────────────────────────────────
    pf(f"    Lifetime metrics (tau1, tau2, FRET):")
    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(ptq_preds, t_ns_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    pf(f"      τ₁ pred range : {tau1_pred.min():.3f} – {tau1_pred.max():.3f} ns")
    pf(f"      τ₂ pred range : {tau2_pred.min():.3f} – {tau2_pred.max():.3f} ns")
    pf(f"      FRET pred range: {fret_pred.min():.3f} – {fret_pred.max():.3f}")

    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

    test_metrics = {
        "job_name": job_name,
        "n_test":   int(len(test_idx)),
        "bits":     bits,
        "tau1":     {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
        "tau2":     {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
        "fret":     {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
    }
    metrics_path = os.path.join(job_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    pf(f"    test_metrics.json saved: {metrics_path}")
    sys.stdout.flush()

    # ── Scatter and residual plots ────────────────────────────────────────────
    pf(f"    Saving scatter and residual plots...")
    save_scatter_plots(
        tau1_gt, tau1_pred, m1,
        tau2_gt, tau2_pred, m2,
        fret_gt, fret_pred, mf,
        job_name, job_dir, pf,
    )
    save_residual_plots(
        tau1_gt, tau1_pred,
        tau2_gt, tau2_pred,
        fret_gt, fret_pred,
        job_name, job_dir, pf,
    )
    sys.stdout.flush()

    # ── SDF metrics ───────────────────────────────────────────────────────────
    pf(f"    SDF-domain metrics (RMSE, R², L2-norm):")

    sys.stdout.flush()
    sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
    sdf_metrics = compute_sdf_metrics(
        gt_seqs       = res_test.astype(np.float32),
        pred_seqs     = ptq_preds,
        channel_names = sdf_channel_names,
        pfn           = pf,
    )
    sdf_metrics["job_name"] = job_name
    sdf_metrics["n_test"]   = int(len(test_idx))
    sdf_metrics["bits"]     = bits
    with open(sdf_metrics_path, "w") as f:
        json.dump(sdf_metrics, f, indent=2)
    pf(f"    test_sdf_metrics.json saved: {sdf_metrics_path}")
    sys.stdout.flush()


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    setup_gpu()

    # Determine which bit-widths to run
    if args.bits == "both":
        bits_list = [8, 16]
    else:
        bits_list = [int(args.bits)]

    pf("=" * 70)
    pf("eval_ptq_sdf.py — Table 2: Weight-only PTQ evaluation")
    pf(f"  data-dir    : {args.data_dir}")
    pf(f"  results-dir : {args.results_dir}")
    pf(f"  bits        : {bits_list}")
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
    run_dirs = find_teacher_run_dirs(args.results_dir)
    if not run_dirs:
        pf("ERROR: No directories with teacher_best.weights.h5 found.")
        sys.exit(1)
    pf(f"Found {len(run_dirs)} teacher run(s):")
    for d in run_dirs:
        pf(f"  {d}")
    sys.stdout.flush()

    # ── PTQ save root ─────────────────────────────────────────────────────────
    save_root = os.path.join(args.results_dir, "ptq")
    os.makedirs(save_root, exist_ok=True)
    pf(f"PTQ results root: {save_root}")
    sys.stdout.flush()

    # ── Main loop: for each run, for each bit-width ───────────────────────────
    t_total = time.time()
    total_jobs = len(run_dirs) * len(bits_list)
    job_idx    = 0

    for run_dir in run_dirs:
        for bits in bits_list:
            job_idx += 1
            pf("")
            pf("=" * 70)
            pf(
                f"[{job_idx}/{total_jobs}]  {bits}-bit PTQ  "
                f"run={os.path.basename(run_dir)}"
            )
            pf(f"  run_dir: {run_dir}")
            pf("=" * 70)
            sys.stdout.flush()
            try:
                evaluate_one_run_at_bits(
                    run_dir               = run_dir,
                    bits                  = bits,
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
                    save_root             = save_root,
                    overwrite             = args.overwrite,
                    pf                    = pf,
                )
            except Exception as exc:
                pf(f"  ERROR in {run_dir} at {bits}-bit: {exc}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

    pf("")
    pf("=" * 70)
    pf(
        f"All PTQ jobs processed.  "
        f"Total elapsed: {(time.time() - t_total) / 60:.1f} min"
    )
    pf(f"Results saved under: {save_root}")
    pf("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()