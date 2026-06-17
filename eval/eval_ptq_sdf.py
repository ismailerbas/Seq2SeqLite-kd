#!/usr/bin/env python3
"""
eval/eval_ptq_sdf.py — Post-training quantization (PTQ) evaluation for Table 2.

Loads teacher_best.weights.h5 from --teacher-run-dir, applies TensorFlow Lite
representative-dataset-based PTQ at --bits (16 or 8), runs inference on the
frozen test split (testidx.npy), computes RMSE, R², L2-norm, DTW per SDF channel,
and saves:
    test_sdf_metrics.json
    test_metrics.json
    test_scatter_tau1.png
    test_scatter_tau2.png
    test_scatter_fret.png
    test_residuals.png
    ptq_args.json
into --save-dir/<job_name>/.

PTQ approach:
    We use TFLite full-integer quantization with float32 I/O for 8-bit and
    float16 quantization for 16-bit, both via a representative dataset drawn
    from a random 2048-sample subset of the TRAIN split.
    INT8 PTQ: converter.optimizations = [OPTIMIZE_FOR_SIZE] +
              representative_dataset + target_spec = [INT8]
    FP16 PTQ: converter.optimizations = [OPTIMIZE_FOR_SIZE] +
              target_spec = [FP16]
    The TFLite model is saved as teacher_ptq_{bits}bit.tflite next to the
    weights file, then reloaded for inference.
    Inference is done sample-by-sample (TFLite interpreter is sequential).
    For large test sets this is slow — use --max-test-samples to cap.

Usage:
    # 8-bit PTQ
    python eval/eval_ptq_sdf.py \
        --data-dir /scratch/nmi \
        --teacher-run-dir /scratch/nmi/your_teacher_run \
        --save-dir /scratch/nmi/results/ptq \
        --bits 8 \
        --seq-len 135 --n-out 3 --gate-width-ns 0.09 \
        --teacher-units 128 --teacher-layers 2 \
        --rep-samples 2048 \
        --max-test-samples 50000

    # 16-bit PTQ
    python eval/eval_ptq_sdf.py \
        --data-dir /scratch/nmi \
        --teacher-run-dir /scratch/nmi/your_teacher_run \
        --save-dir /scratch/nmi/results/ptq \
        --bits 16 \
        --seq-len 135 --n-out 3 --gate-width-ns 0.09 \
        --teacher-units 128 --teacher-layers 2 \
        --rep-samples 2048 \
        --max-test-samples 50000

    --max-test-samples : cap the test set size for speed.  0 = use all.
    --rep-samples      : number of representative samples for PTQ calibration.
    --overwrite        : re-run even if test_sdf_metrics.json already exists.
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
        description="PTQ evaluation — Table 2 (16-bit and 8-bit quantization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",         type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, split .npy files.")
    p.add_argument("--teacher-run-dir",  type=str, required=True,
                   help="Directory containing teacher_best.weights.h5.")
    p.add_argument("--save-dir",         type=str, required=True,
                   help="Root directory for PTQ results subfolders.")
    p.add_argument("--bits",             type=int, required=True, choices=[8, 16],
                   help="PTQ bit-width: 8 (INT8 full-integer) or 16 (FP16).")
    p.add_argument("--seq-len",          type=int, default=135)
    p.add_argument("--n-out",            type=int, default=3)
    p.add_argument("--gate-width-ns",    type=float, default=0.09)
    p.add_argument("--teacher-units",    type=int, default=128,
                   help="Teacher GRU hidden units (overridden by teacher_args.json if found).")
    p.add_argument("--teacher-layers",   type=int, default=2,
                   help="Teacher GRU layers (overridden by teacher_args.json if found).")
    p.add_argument("--rep-samples",      type=int, default=2048,
                   help="Number of representative calibration samples for INT8 PTQ.")
    p.add_argument("--max-test-samples", type=int, default=0,
                   help="Cap test set at this many samples for speed (0 = use all).")
    p.add_argument("--overwrite",        action="store_true", default=False,
                   help="Re-run even if test_sdf_metrics.json already exists.")
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

    file_train = None
    for name in ["trainidx.npy", "train_idx.npy"]:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            file_train = candidate
            break
    if file_train is None:
        raise FileNotFoundError(
            f"Train split index not found in {data_dir}. Tried: trainidx.npy, train_idx.npy"
        )

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

    return file_input, file_res, file_labels, file_train, file_test


# ==============================================================================
# Teacher model — EXACT replica of train_teacher.py
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
# PTQ: convert Keras model to TFLite with INT8 or FP16 quantization
# ==============================================================================

def convert_to_tflite_ptq(
    teacher_model,
    normalized_input,
    train_idx,
    seq_len,
    bits,
    rep_samples,
    tflite_save_path,
    pf,
):
    """
    Convert teacher_model to TFLite with PTQ at `bits` (8 or 16).

    For INT8 (bits=8):
        Full-integer quantization with float32 I/O.
        Uses a representative dataset of `rep_samples` samples from train_idx.
    For FP16 (bits=16):
        Float16 quantization.
        No representative dataset needed.

    Saves the .tflite flatbuffer to tflite_save_path and returns the path.
    """
    pf(f"  Building TFLite converter for {bits}-bit PTQ...")
    sys.stdout.flush()

    # Wrap the Keras model in a concrete function so the converter can trace it.
    # The model takes two inputs: [enc_input (N,T,1), dec_input (N,T,1)].
    # TFLite requires a single-input signature.  We fuse both inputs into one
    # call by creating a wrapper tf.Module.
    class TeacherWrapper(tf.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, None, 1], dtype=tf.float32, name="enc_input"),
            tf.TensorSpec(shape=[1, None, 1], dtype=tf.float32, name="dec_input"),
        ])
        def serve(self, enc_input, dec_input):
            return self.model([enc_input, dec_input], training=False)

    wrapper = TeacherWrapper(teacher_model)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [wrapper.serve.get_concrete_function()],
        wrapper,
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if bits == 8:
        # Representative dataset for INT8 calibration
        rng = np.random.default_rng(42)
        rep_idx = rng.choice(train_idx, size=min(rep_samples, len(train_idx)), replace=False)
        rep_enc = normalized_input[rep_idx].astype(np.float32)  # (rep_samples, T, 1)

        def representative_dataset_gen():
            for i in range(len(rep_idx)):
                enc_b = rep_enc[i : i + 1]                        # (1, T, 1)
                dec_b = np.zeros((1, seq_len, 1), dtype=np.float32)
                yield [enc_b, dec_b]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.float32
        converter.inference_output_type = tf.float32
        pf(f"  INT8 PTQ: representative dataset = {len(rep_idx)} samples from train split")

    elif bits == 16:
        converter.target_spec.supported_types = [tf.float16]
        pf(f"  FP16 PTQ: no representative dataset needed")

    sys.stdout.flush()

    pf(f"  Running converter.convert()  (this may take a few minutes)...")
    sys.stdout.flush()
    t0 = time.time()
    tflite_model = converter.convert()
    pf(f"  Conversion done in {time.time() - t0:.1f}s  size={len(tflite_model)/1e6:.2f} MB")
    sys.stdout.flush()

    with open(tflite_save_path, "wb") as f:
        f.write(tflite_model)
    pf(f"  TFLite model saved: {tflite_save_path}")
    sys.stdout.flush()

    return tflite_save_path


# ==============================================================================
# TFLite inference — sequential, sample by sample or small batches
# ==============================================================================

def run_tflite_inference(tflite_path, enc_arr, seq_len, n_out, pf):
    """
    Run inference with the TFLite model on enc_arr.
    TFLite interpreters are sequential (no batching across samples at the
    interpreter level when batch dim = 1).  We iterate sample by sample.
    This is slow for large test sets — use --max-test-samples to cap.

    enc_arr : (N, T, 1) float32
    Returns preds : (N, T, n_out) float32
    """
    pf(f"  Loading TFLite interpreter from: {tflite_path}")
    sys.stdout.flush()

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pf(f"  TFLite inputs  : {[(d['name'], d['shape'], d['dtype']) for d in input_details]}")
    pf(f"  TFLite outputs : {[(d['name'], d['shape'], d['dtype']) for d in output_details]}")
    sys.stdout.flush()

    # Identify enc and dec input tensor indices by name
    enc_idx = None
    dec_idx = None
    for d in input_details:
        if "enc_input" in d["name"]:
            enc_idx = d["index"]
        elif "dec_input" in d["name"]:
            dec_idx = d["index"]

    if enc_idx is None or dec_idx is None:
        # Fallback: assign by order (enc=0, dec=1)
        pf("  WARNING: could not identify enc/dec inputs by name — using index order 0/1")
        enc_idx = input_details[0]["index"]
        dec_idx = input_details[1]["index"]

    out_idx = output_details[0]["index"]

    N = len(enc_arr)
    preds = np.zeros((N, seq_len, n_out), dtype=np.float32)
    dec_zero = np.zeros((1, seq_len, 1), dtype=np.float32)

    t0 = time.time()
    print_every = max(1, N // 20)

    for i in tqdm(
        range(N),
        desc="TFLite inference",
        unit="sample",
        bar_format="{l_bar}{bar:30}{r_bar}",
    ):
        enc_b = enc_arr[i : i + 1].astype(np.float32)
        interpreter.set_tensor(enc_idx, enc_b)
        interpreter.set_tensor(dec_idx, dec_zero)
        interpreter.invoke()
        preds[i] = interpreter.get_tensor(out_idx)[0]

        if (i + 1) % print_every == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            pct     = 100.0 * (i + 1) / N
            eta_s   = (elapsed / max(i + 1, 1)) * (N - i - 1)
            pf(
                f"  TFLite [{i + 1:>8,}/{N:,}  {pct:5.1f}%]  "
                f"elapsed={elapsed / 60:.1f}min  ETA={eta_s / 60:.1f}min"
            )
            sys.stdout.flush()

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
# Scatter plot helpers
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
# Main
# ==============================================================================

def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    setup_gpu()

    # ── Resolve per-run teacher hyper-params from teacher_args.json ───────────
    teacher_units  = args.teacher_units
    teacher_layers = args.teacher_layers
    args_path = os.path.join(args.teacher_run_dir, "teacher_args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            run_args = json.load(f)
        teacher_units  = int(run_args.get("teacher_units",  teacher_units))
        teacher_layers = int(run_args.get("teacher_layers", teacher_layers))
        pf(f"[INFO] teacher_args.json: units={teacher_units}  layers={teacher_layers}")
    else:
        pf(f"[INFO] teacher_args.json not found — using CLI: units={teacher_units}  layers={teacher_layers}")

    # ── Job name and output directory ─────────────────────────────────────────
    teacher_run_name = os.path.basename(os.path.normpath(args.teacher_run_dir))
    job_name = f"ptq_{args.bits}bit_{teacher_run_name}"
    job_dir  = os.path.join(args.save_dir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    pf("=" * 70)
    pf(f"eval_ptq_sdf.py — Table 2: PTQ {args.bits}-bit evaluation")
    pf(f"  teacher-run-dir : {args.teacher_run_dir}")
    pf(f"  job_name        : {job_name}")
    pf(f"  job_dir         : {job_dir}")
    pf(f"  bits            : {args.bits}")
    pf(f"  overwrite       : {args.overwrite}")
    pf("=" * 70)
    sys.stdout.flush()

    sdf_metrics_path = os.path.join(job_dir, "test_sdf_metrics.json")
    if os.path.exists(sdf_metrics_path) and not args.overwrite:
        pf(f"SKIP — test_sdf_metrics.json already exists: {sdf_metrics_path}")
        pf("Pass --overwrite to re-run.")
        sys.exit(0)

    # ── Save args ─────────────────────────────────────────────────────────────
    with open(os.path.join(job_dir, "ptq_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    pf(f"ptq_args.json saved.")
    sys.stdout.flush()

    # ── Load data ─────────────────────────────────────────────────────────────
    pf("Loading data files (mmap)...")
    file_input, file_res, file_labels, file_train, file_test = find_data_files(
        args.data_dir, args.seq_len
    )
    pf(f"  encoder input : {file_input}")
    pf(f"  decoder target: {file_res}")
    pf(f"  labels        : {file_labels}")
    pf(f"  train idx     : {file_train}")
    pf(f"  test idx      : {file_test}")
    sys.stdout.flush()

    normalized_input = np.load(file_input,  mmap_mode="r")
    res              = np.load(file_res,    mmap_mode="r")
    labels           = np.load(file_labels, mmap_mode="r")
    train_idx        = np.load(file_train)
    test_idx         = np.load(file_test)

    if args.max_test_samples > 0 and len(test_idx) > args.max_test_samples:
        rng      = np.random.default_rng(42)
        test_idx = rng.choice(test_idx, size=args.max_test_samples, replace=False)
        pf(f"  Capped test set to {len(test_idx):,} samples (--max-test-samples {args.max_test_samples})")

    pf(
        f"  N={normalized_input.shape[0]:,}  "
        f"test_N={len(test_idx):,}  "
        f"train_N={len(train_idx):,}"
    )
    sys.stdout.flush()

    enc_test = normalized_input[test_idx].astype(np.float32)
    res_test = res[test_idx].astype(np.float32)
    lab_test = labels[test_idx]

    # ── Build and load teacher (float32) ─────────────────────────────────────
    pf("Building teacher model...")
    tf.keras.backend.clear_session()
    teacher_model = build_teacher(args.seq_len, args.n_out, teacher_units, teacher_layers)
    ckpt_path = os.path.join(args.teacher_run_dir, "teacher_best.weights.h5")
    pf(f"Loading weights: {ckpt_path}")
    teacher_model.load_weights(ckpt_path)
    teacher_model.trainable = False
    pf(f"Weights loaded OK.")
    teacher_model.summary(print_fn=pf)
    sys.stdout.flush()

    # ── Convert to TFLite PTQ ─────────────────────────────────────────────────
    tflite_filename = f"teacher_ptq_{args.bits}bit.tflite"
    tflite_save_path = os.path.join(job_dir, tflite_filename)

    if os.path.exists(tflite_save_path) and not args.overwrite:
        pf(f"TFLite model already exists — reusing: {tflite_save_path}")
    else:
        pf("Converting to TFLite PTQ...")
        convert_to_tflite_ptq(
            teacher_model    = teacher_model,
            normalized_input = normalized_input,
            train_idx        = train_idx,
            seq_len          = args.seq_len,
            bits             = args.bits,
            rep_samples      = args.rep_samples,
            tflite_save_path = tflite_save_path,
            pf               = pf,
        )
    sys.stdout.flush()

    # ── TFLite inference on test set ─────────────────────────────────────────
    pf(f"Running TFLite inference on {len(test_idx):,} test samples...")
    sys.stdout.flush()
    ptq_preds = run_tflite_inference(tflite_save_path, enc_test, args.seq_len, args.n_out, pf)
    pf(f"ptq_preds shape: {ptq_preds.shape}")
    sys.stdout.flush()

    t_ns_axis = np.arange(args.seq_len, dtype=np.float32) * args.gate_width_ns

    # ── Lifetime metrics ──────────────────────────────────────────────────────
    pf("=" * 60)
    pf("Lifetime metrics (tau1, tau2, FRET)")
    pf("=" * 60)
    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(ptq_preds, t_ns_axis)
    tau1_gt  = lab_test[:, 0]
    tau2_gt  = lab_test[:, 1]
    fret_gt  = lab_test[:, 2]

    pf(f"  τ₁ pred range: {tau1_pred.min():.3f} – {tau1_pred.max():.3f} ns")
    pf(f"  τ₂ pred range: {tau2_pred.min():.3f} – {tau2_pred.max():.3f} ns")
    pf(f"  FRET pred range: {fret_pred.min():.3f} – {fret_pred.max():.3f}")

    m1 = compute_metrics(tau1_gt, tau1_pred, "τ₁ (ns)",  pf)
    m2 = compute_metrics(tau2_gt, tau2_pred, "τ₂ (ns)",  pf)
    mf = compute_metrics(fret_gt, fret_pred, "FRET (f)", pf)

    test_metrics = {
        "job_name": job_name,
        "n_test":   int(len(test_idx)),
        "bits":     args.bits,
        "tau1":     {"rmse": m1[0], "r": m1[1], "cov1sigma": m1[2]},
        "tau2":     {"rmse": m2[0], "r": m2[1], "cov1sigma": m2[2]},
        "fret":     {"rmse": mf[0], "r": mf[1], "cov1sigma": mf[2]},
    }
    metrics_path = os.path.join(job_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    pf(f"test_metrics.json saved: {metrics_path}")
    sys.stdout.flush()

    # ── Scatter and residual plots ────────────────────────────────────────────
    pf("Saving scatter and residual plots...")
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
    pf("=" * 60)
    pf("SDF-domain metrics (paper Table 2): RMSE, R², L2-norm, DTW")
    pf("=" * 60)
    sdf_channel_names = ["ch0_full", "ch1_short", "ch2_long"]
    sdf_metrics = compute_sdf_metrics(
        gt_seqs       = res_test,
        pred_seqs     = ptq_preds,
        channel_names = sdf_channel_names,
        pfn           = pf,
    )
    sdf_metrics["job_name"] = job_name
    sdf_metrics["n_test"]   = int(len(test_idx))
    sdf_metrics["bits"]     = args.bits
    with open(sdf_metrics_path, "w") as f:
        json.dump(sdf_metrics, f, indent=2)
    pf(f"test_sdf_metrics.json saved: {sdf_metrics_path}")
    sys.stdout.flush()

    pf("=" * 70)
    pf(f"DONE — {job_name}")
    pf(f"Results in: {job_dir}")
    pf("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()