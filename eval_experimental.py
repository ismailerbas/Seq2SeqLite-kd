#!/usr/bin/env python3
"""
eval_experimental.py — Evaluate trained teacher / student / ablation models
on experimental TCSPC data loaded from a .mat file.

The .mat file contains a variable 'tp4d' with shape (484, 250, 135) where:
  axis 0 : 484  pixel rows
  axis 1 : 250  pixel columns
  axis 2 : 135  time bins (gate width = 0.09 ns each)

Pipeline:
  1. Load .mat file, extract tp4d  → (484, 250, 135)
  2. Baseline correction           → subtract per-pixel pre-gate background
  3. Normalize per pixel           → divide by max, clamp negatives to 0
  4. Reshape to encoder input      → (N_pixels, 135, 1)  N_pixels = 484*250
  5. Discover ablation weight files automatically under --ablation-root
  6. For each model found, run inference
  7. Extract tau1, tau2, fret per pixel via extract_lifetimes
  8. Save pixel maps (484×250 images), histograms, statistics JSON,
     and scatter plots (pred vs pred self-consistency check)
     all inside the same folder as the weight file

Usage:
  python eval_experimental.py \\
    --mat-file  /path/to/af700\\ [Accumulated].mat \\
    --ablation-root  /path/to/results \\
    --gate-width-ns  0.09 \\
    --n-out  3 \\
    --seq-len  135 \\
    --infer-batch  4096 \\
    --n-rows  484 \\
    --n-cols  250 \\
    --baseline-bins  10

  To evaluate teacher only (no student scan):
    add --teacher-ckpt /path/to/teacher_training_gru128x128/teacher_best_gru128x128.weights.h5 \\
        --teacher-layers-list 128 128

  To evaluate a single student ablation folder directly:
    add --ablation-root  /path/to/results/student_b4k4r4a4_gru32x1_dense3_bs1024
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GRUCell, Input, RNN
from tensorflow.keras.models import Model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate teacher/student/ablation models on experimental .mat data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mat-file", type=str, required=True,
                   help="Path to the .mat file (e.g. 'af700 [Accumulated].mat'). "
                        "Must contain a variable named 'tp4d' with shape "
                        "(n_rows, n_cols, seq_len).")
    p.add_argument("--ablation-root", type=str, default=None,
                   help="Root directory to scan recursively for weight files "
                        "(.weights.h5). Each weight file found will be evaluated "
                        "and results saved next to it. Can point to a single "
                        "ablation subfolder or the parent 'results' directory.")
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="Path to a specific teacher weight file "
                        "(.weights.h5). Evaluated with --teacher-layers-list. "
                        "Can be combined with --ablation-root.")
    p.add_argument("--teacher-layers-list", type=int, nargs="+", default=[128, 128],
                   help="Teacher GRUCell layer sizes in order. "
                        "Default: 128 128 (matches teacher_best_gru128x128).")
    p.add_argument("--seq-len", type=int, default=135,
                   help="Number of time bins (sequence length).")
    p.add_argument("--n-out", type=int, default=3,
                   help="Number of decoder output channels.")
    p.add_argument("--n-rows", type=int, default=484,
                   help="Number of pixel rows in the spatial image.")
    p.add_argument("--n-cols", type=int, default=250,
                   help="Number of pixel columns in the spatial image.")
    p.add_argument("--gate-width-ns", type=float, default=0.09,
                   help="Gate width per time bin in nanoseconds.")
    p.add_argument("--baseline-bins", type=int, default=10,
                   help="Number of leading time bins to average for "
                        "per-pixel baseline (background) subtraction.")
    p.add_argument("--mask-file", type=str, default=None,
                   help="Optional path to a binary mask file. Accepted formats: "
                        ".npy (bool or uint8 array of shape (n_rows, n_cols)), "
                        ".mat (must contain a variable named 'mask' of shape "
                        "(n_rows, n_cols)), or any image file readable by "
                        "matplotlib (PNG/TIFF/BMP — non-zero pixels = valid). "
                        "True / non-zero = valid pixel included in statistics "
                        "and visualisations. Pixels where mask is False / 0 are "
                        "set to NaN in pixel maps and excluded from histograms "
                        "and scatter plots. This mask is applied ON TOP of the "
                        "intensity-based pixel_mask (max-after-baseline > 0), "
                        "so a pixel must satisfy BOTH conditions to be included.")
    p.add_argument("--infer-batch", type=int, default=4096,
                   help="Batch size for model inference.")
    p.add_argument("--student-units-default", type=int, default=32,
                   help="Fallback student GRU hidden units when the folder name "
                        "cannot be parsed. Only used for vanilla-KD students.")
    p.add_argument("--teacher-units-default", type=int, default=128,
                   help="Fallback teacher hidden units when the folder name "
                        "cannot be parsed. Only used for vanilla-KD students.")
    p.add_argument("--teacher-layers-default", type=int, default=2,
                   help="Fallback teacher layer count when the folder name "
                        "cannot be parsed. Only used for vanilla-KD students.")
    p.add_argument("--bits-default", type=int, default=4,
                   help="Fallback quantisation bits when the folder name "
                        "cannot be parsed (kernel/recurrent/bias/activation/state).")

    args = p.parse_args()

    if args.ablation_root is None and args.teacher_ckpt is None:
        p.error("Provide at least one of --ablation-root or --teacher-ckpt.")

    return args

# ---------------------------------------------------------------------------
# GPU setup (CPU-friendly — no MirroredStrategy needed for inference)
# ---------------------------------------------------------------------------
def setup_gpu():
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[GPU] No GPU found — running on CPU.", flush=True)
        return
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[GPU] set_memory_growth failed: {e}", flush=True)
    print(f"[GPU] {len(physical_gpus)} GPU(s) available. Using GPU:0 for inference.",
          flush=True)


# ---------------------------------------------------------------------------
# .mat loading and preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess_mat(mat_path, n_rows, n_cols, seq_len,
                             baseline_bins, pf):
    """
    Load tp4d from .mat file and produce encoder_input ready for the model.

    Steps:
      1. Load mat file, extract tp4d (n_rows, n_cols, seq_len)
      2. Cast to float32
      3. Per-pixel baseline correction:
           baseline[r, c] = mean(tp4d[r, c, :baseline_bins])
           tp4d[r, c, :] -= baseline[r, c]
      4. Clamp negatives to 0
      5. Per-pixel max normalization:
           mx[r, c] = max(tp4d[r, c, :])
           if mx > 0: tp4d[r, c, :] /= mx
      6. Reshape to (N_pixels, seq_len, 1)  where N_pixels = n_rows * n_cols

    Returns
    -------
    encoder_input : np.ndarray  shape (N_pixels, seq_len, 1)  float32
    pixel_mask    : np.ndarray  shape (n_rows, n_cols)         bool
                    True where the pixel had a non-zero max after baseline
                    subtraction (i.e. valid pixels for interpretation).
    """
    pf(f"[MAT] Loading: {mat_path}")
    mat_data = loadmat(mat_path)

    if "tp4d" not in mat_data:
        available = [k for k in mat_data.keys() if not k.startswith("_")]
        raise KeyError(
            f"Variable 'tp4d' not found in {mat_path}. "
            f"Available variables: {available}"
        )

    tp4d = mat_data["tp4d"].astype(np.float32)
    pf(f"[MAT] tp4d loaded: shape={tp4d.shape}  dtype={tp4d.dtype}")

    actual_rows, actual_cols, actual_time = tp4d.shape
    if actual_rows != n_rows or actual_cols != n_cols:
        pf(f"[MAT] WARNING: expected ({n_rows}, {n_cols}, {seq_len}) "
           f"but got ({actual_rows}, {actual_cols}, {actual_time}). "
           f"Using actual dimensions.")
        n_rows = actual_rows
        n_cols = actual_cols

    if actual_time != seq_len:
        raise ValueError(
            f"[MAT] seq_len mismatch: --seq-len={seq_len} but "
            f"tp4d has {actual_time} time bins. "
            f"Pass --seq-len {actual_time} to fix."
        )

    pf(f"[MAT] Applying baseline correction using first {baseline_bins} bins...")
    baseline = np.mean(tp4d[:, :, :baseline_bins], axis=2, keepdims=True)
    tp4d = tp4d - baseline
    pf(f"[MAT] Baseline range: min={baseline.min():.4f}  max={baseline.max():.4f}")

    pf("[MAT] Clamping negatives to 0...")
    np.clip(tp4d, 0.0, None, out=tp4d)

    pf("[MAT] Per-pixel max normalization...")
    pixel_max = tp4d.max(axis=2)
    pixel_mask = pixel_max > 0.0
    n_valid = pixel_mask.sum()
    n_total = n_rows * n_cols
    pf(f"[MAT] Valid pixels (max > 0): {n_valid:,} / {n_total:,} "
       f"({100.0 * n_valid / n_total:.1f}%)")

    safe_max = np.where(pixel_max > 0.0, pixel_max, 1.0)
    tp4d = tp4d / safe_max[:, :, np.newaxis]

    encoder_input = tp4d.reshape(n_rows * n_cols, seq_len, 1).astype(np.float32)
    pf(f"[MAT] encoder_input shape: {encoder_input.shape}  dtype={encoder_input.dtype}")
    pf(f"[MAT] encoder_input value range: [{encoder_input.min():.4f}, "
       f"{encoder_input.max():.4f}]")

    return encoder_input, pixel_mask, n_rows, n_cols


# ---------------------------------------------------------------------------
# Load external binary mask and combine with intensity-based pixel_mask
# ---------------------------------------------------------------------------
def load_binary_mask(mask_path, n_rows, n_cols, pf):
    """
    Load a binary mask from disk and return it as a bool array (n_rows, n_cols).

    Supported formats
    -----------------
    .npy  — numpy array of shape (n_rows, n_cols), dtype bool or uint8/int/float.
            Non-zero = valid.
    .mat  — MATLAB file containing a variable named 'mask' of shape
            (n_rows, n_cols). Non-zero = valid.
    image — any format readable by matplotlib (PNG, TIFF, BMP, JPG).
            Loaded as greyscale (mean across channels if RGB/RGBA).
            Non-zero = valid.

    The loaded mask is resized to (n_rows, n_cols) using nearest-neighbour
    interpolation only if its spatial dimensions differ from (n_rows, n_cols).
    A warning is printed in that case.

    Returns
    -------
    ext_mask : np.ndarray  shape (n_rows, n_cols)  dtype bool
               True = valid pixel (included in all statistics and plots).
    """
    pf(f"[MASK] Loading external binary mask: {mask_path}")

    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"[MASK] --mask-file not found on disk: {mask_path}"
        )

    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".npy":
        raw = np.load(mask_path)
        pf(f"[MASK] .npy loaded: shape={raw.shape}  dtype={raw.dtype}")

    elif ext == ".mat":
        mat_data = loadmat(mask_path)
        available = [k for k in mat_data.keys() if not k.startswith("_")]
        if "mask" not in mat_data:
            raise KeyError(
                f"[MASK] Variable 'mask' not found in {mask_path}. "
                f"Available variables: {available}"
            )
        raw = mat_data["mask"]
        pf(f"[MASK] .mat 'mask' loaded: shape={raw.shape}  dtype={raw.dtype}")

    else:
        # Treat as image (PNG, TIFF, BMP, JPG, etc.)
        import matplotlib.image as mpimg
        img = mpimg.imread(mask_path)
        pf(f"[MASK] Image loaded: shape={img.shape}  dtype={img.dtype}")
        if img.ndim == 3:
            raw = img.mean(axis=2)
        else:
            raw = img
        pf(f"[MASK] After channel collapse: shape={raw.shape}")

    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(
            f"[MASK] Mask must be 2-D after loading but got shape {raw.shape}."
        )

    if raw.shape[0] != n_rows or raw.shape[1] != n_cols:
        pf(
            f"[MASK] WARNING: mask shape {raw.shape} != expected ({n_rows}, {n_cols}). "
            f"Resizing with nearest-neighbour interpolation."
        )
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(raw.astype(np.float32))
        pil_img = pil_img.resize((n_cols, n_rows), PILImage.NEAREST)
        raw = np.array(pil_img)
        pf(f"[MASK] After resize: shape={raw.shape}")

    ext_mask = raw.astype(bool)

    n_valid = int(ext_mask.sum())
    n_total = n_rows * n_cols
    pf(
        f"[MASK] External mask: {n_valid:,} / {n_total:,} valid pixels "
        f"({100.0 * n_valid / n_total:.1f}%)"
    )

    return ext_mask

# ---------------------------------------------------------------------------
# Post-processing: extract tau1, tau2, fret from model output sequences
# Identical to the function used in train_teacher.py and train_student.py
# ---------------------------------------------------------------------------
def extract_lifetimes(preds, t):
    """
    preds : (N, T, 3)  channel 0=full decay, channel 1=short, channel 2=long
    t     : (T,) time axis in ns

    Returns tau1, tau2, fret each shape (N,) float32.
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


# ---------------------------------------------------------------------------
# Compute statistics for a 1-D array of values
# ---------------------------------------------------------------------------
def compute_stats(arr, label, pf):
    finite = arr[np.isfinite(arr)]
    stats = {
        "label":  label,
        "n":      int(len(finite)),
        "mean":   float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std":    float(np.std(finite)),
        "min":    float(np.min(finite)),
        "max":    float(np.max(finite)),
        "p5":     float(np.percentile(finite, 5)),
        "p25":    float(np.percentile(finite, 25)),
        "p75":    float(np.percentile(finite, 75)),
        "p95":    float(np.percentile(finite, 95)),
    }
    pf(f"  {label:12s}  mean={stats['mean']:.4f}  "
       f"median={stats['median']:.4f}  std={stats['std']:.4f}  "
       f"[{stats['min']:.4f}, {stats['max']:.4f}]")
    return stats


# ---------------------------------------------------------------------------
# Build teacher model — exact replica of train_teacher.py
# Layer names: encinput, decinput, encrnn, decrnn, decdense
# ---------------------------------------------------------------------------
def build_teacher(seq_len, n_out, layers_teacher):
    encoder_inputs = Input(shape=(None, 1), name="encinput")
    encoder_cells = [
        GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(layers_teacher)
    ]
    encoder_rnn = RNN(encoder_cells, return_state=True, name="encrnn")
    enc_outputs_and_states = encoder_rnn(encoder_inputs)
    encoder_states = enc_outputs_and_states[1:]

    decoder_inputs = Input(shape=(None, 1), name="decinput")
    decoder_cells = [
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
# Build teacher model — alternate layer names used in train_student.py
# Layer names: enc_input, dec_input, enc_rnn, dec_rnn, dec_dense
# ---------------------------------------------------------------------------
def build_teacher_student_names(seq_len, n_out, teacher_units, teacher_layers):
    """
    Replica of build_teacher() in train_student.py.
    Used when loading a teacher checkpoint that was saved by train_student.py
    (which uses enc_input / dec_input naming).
    """
    LAYERS_TEACHER = [teacher_units] * teacher_layers

    encoder_inputs = keras.layers.Input(shape=(None, 1), name="enc_input")
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells, return_state=True, name="enc_rnn"
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


# ---------------------------------------------------------------------------
# Build vanilla-KD student model — exact replica of train_student_vanilla_kd.py
# We build a plain float32 GRU student (no QKeras) for the vanilla-KD ablations.
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense
# ---------------------------------------------------------------------------
def build_vanilla_student(student_units, seq_len, n_out):
    enc_input = tf.keras.Input(shape=(seq_len, 1), name="s_enc_input")
    dec_input = tf.keras.Input(shape=(seq_len, 1), name="s_dec_input")

    enc_gru = tf.keras.layers.GRU(
        student_units,
        return_state=True,
        return_sequences=False,
        reset_after=True,
        name="s_enc_gru",
    )
    _, enc_state = enc_gru(enc_input)

    dec_gru = tf.keras.layers.GRU(
        student_units,
        return_sequences=True,
        return_state=False,
        reset_after=True,
        name="s_dec_gru",
    )
    dec_out = dec_gru(dec_input, initial_state=enc_state)

    out = tf.keras.layers.Dense(n_out, name="s_out_dense")(dec_out)

    model = tf.keras.Model(
        inputs=[enc_input, dec_input],
        outputs=out,
        name="vanilla_student",
    )
    return model

# ---------------------------------------------------------------------------
# Build QKeras student model — exact replica of train_student.py
# Requires QKeras to be installed.
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense
# ---------------------------------------------------------------------------
def build_qkeras_student(seq_len, n_out, student_units,
                          bits_kernel, bits_recurrent, bits_bias,
                          bits_activation, bits_state):
    from qkeras import QDense, QGRU, quantized_bits, quantized_tanh

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
        name="qkeras_student",
    )
    return student_model


# ---------------------------------------------------------------------------
# Parse model config from folder / file name
# Returns a dict with keys: model_type, layers_teacher (list), student_units,
# teacher_units, teacher_layers, bits_kernel, bits_recurrent, bits_bias,
# bits_activation, bits_state
# ---------------------------------------------------------------------------
def parse_config_from_path(weight_path, args):
    """
    Infer model architecture from the weight file path and parent folder name.

    Conventions (from the training scripts):
      Teacher:
        teacher_best_gru{U}x{U}.weights.h5
        teacher_best_gru{U}.weights.h5

      FW-QATD-RAC student (train_student.py):
        results/student_b{K}k{K}r{R}a{A}_gru{U}x1_dense{D}_bs{B}/
            student_best.weights.h5
            student_final.weights.h5

      Vanilla-KD student (train_student_vanilla_kd.py):
        Any folder not matching the above patterns, containing a file
        named student_best.weights.h5 or student_final.weights.h5.

    Returns
    -------
    dict with keys: model_type (str), layers_teacher (list[int]),
                    student_units (int), teacher_units (int),
                    teacher_layers (int), bits_kernel (int),
                    bits_recurrent (int), bits_bias (int),
                    bits_activation (int), bits_state (int)
    """
    import re

    fname = os.path.basename(weight_path)
    folder = os.path.basename(os.path.dirname(weight_path))

    config = {
        "model_type":     "unknown",
        "layers_teacher": list(args.teacher_layers_list),
        "student_units":  args.student_units_default,
        "teacher_units":  args.teacher_units_default,
        "teacher_layers": args.teacher_layers_default,
        "bits_kernel":    args.bits_default,
        "bits_recurrent": args.bits_default,
        "bits_bias":      args.bits_default,
        "bits_activation":args.bits_default,
        "bits_state":     args.bits_default,
    }

    # --- Teacher: teacher_best_gru128x128.weights.h5 ---
    m = re.search(r"teacher_best_(gru[\dx]+)\.weights\.h5", fname)
    if m:
        gru_tag = m.group(1)
        units_parts = [int(u) for u in gru_tag.replace("gru", "").split("x") if u]
        config["model_type"]     = "teacher"
        config["layers_teacher"] = units_parts
        return config

    # --- FW-QATD-RAC student folder: student_b4k4r4a4_gru32x1_dense3_bs1024 ---
    m_folder = re.match(
        r"student_b(\d+)k(\d+)r(\d+)a(\d+)_gru(\d+)x1_dense(\d+)_bs(\d+)$",
        folder,
    )
    if m_folder:
        bk = int(m_folder.group(1))
        kk = int(m_folder.group(2))
        rr = int(m_folder.group(3))
        aa = int(m_folder.group(4))
        gu = int(m_folder.group(5))
        config["model_type"]      = "student_qkeras"
        config["student_units"]   = gu
        config["bits_kernel"]     = bk
        config["bits_recurrent"]  = rr
        config["bits_bias"]       = bk
        config["bits_activation"] = aa
        config["bits_state"]      = bk
        return config

    # --- Vanilla-KD student: any other folder with student*.weights.h5 ---
    if "student" in fname.lower():
        config["model_type"] = "student_vanilla"
        # Try to parse gru units from folder name e.g. gru32 or gru32x1
        m_gru = re.search(r"gru(\d+)", folder)
        if m_gru:
            config["student_units"] = int(m_gru.group(1))
        return config

    # Fallback
    config["model_type"] = "unknown"
    return config


# ---------------------------------------------------------------------------
# Discover all weight files under a root directory
# ---------------------------------------------------------------------------
def discover_weight_files(root_dir, pf):
    """
    Walk root_dir recursively and collect all .weights.h5 files.
    Returns a sorted list of absolute paths.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".weights.h5"):
                found.append(os.path.join(dirpath, fname))
    found = sorted(found)
    pf(f"[DISCOVER] Found {len(found)} weight file(s) under {root_dir}:")
    for f in found:
        pf(f"  {f}")
    return found


# ---------------------------------------------------------------------------
# Run inference with a Keras model on encoder_input
# Handles both teacher (encinput/decinput) and student (senc_input/sdec_input)
# input name conventions automatically.
# ---------------------------------------------------------------------------
def run_inference(model, encoder_input, seq_len, n_out, infer_batch, pf="INFER"):
    N = encoder_input.shape[0]
    dec_input_zeros = np.zeros((N, seq_len, 1), dtype=np.float32)

    input_names = [inp.name for inp in model.inputs]
    enc_key = None
    dec_key = None
    for name in input_names:
        stripped = name.split(":")[0]
        if "enc" in stripped:
            enc_key = stripped
        elif "dec" in stripped:
            dec_key = stripped
    if enc_key is None or dec_key is None:
        raise ValueError(
            f"[{pf}] Could not auto-detect enc/dec input names from model inputs: {input_names}"
        )

    print(f"[{pf}] N={N:,} batch={infer_batch} enc_key={enc_key} dec_key={dec_key}")
    all_preds = []
    for start in range(0, N, infer_batch):
        enc_b = encoder_input[start:start + infer_batch]
        dec_b = dec_input_zeros[start:start + infer_batch]
        out = model({enc_key: enc_b, dec_key: dec_b}, training=False)
        all_preds.append(out.numpy() if hasattr(out, "numpy") else out)
    return np.concatenate(all_preds, axis=0)

# ---------------------------------------------------------------------------
# Save pixel maps (spatial images)
# ---------------------------------------------------------------------------
def save_pixel_maps(tau1_map, tau2_map, fret_map, pixel_mask,
                    n_rows, n_cols, out_dir, model_tag, pf):
    """
    tau1_map, tau2_map, fret_map : (n_rows, n_cols) float32
    pixel_mask                   : (n_rows, n_cols) bool — valid pixels
    Saves three PNG pixel map images.
    """
    masked_tau1 = np.where(pixel_mask, tau1_map, np.nan)
    masked_tau2 = np.where(pixel_mask, tau2_map, np.nan)
    masked_fret = np.where(pixel_mask, fret_map, np.nan)

    panels = [
        (masked_tau1, "τ₁ (ns)",  (0.0, 3.0), "inferno",  "pixmap_tau1.png"),
        (masked_tau2, "τ₂ (ns)",  (0.0, 3.0), "viridis",  "pixmap_tau2.png"),
        (masked_fret, "FRET (f)", (0.0, 1.0), "plasma",   "pixmap_fret.png"),
    ]

    for data, label, clim, cmap, fname in panels:
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        im = ax.imshow(
            data,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(f"{model_tag}\n{label}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Column", fontsize=9)
        ax.set_ylabel("Row", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, pad=0.02).set_label(label, fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"  Pixel map saved: {out_path}")


# ---------------------------------------------------------------------------
# Save histograms of tau1, tau2, fret distributions
# ---------------------------------------------------------------------------
def save_histograms(tau1_vals, tau2_vals, fret_vals,
                    pixel_mask_flat, out_dir, model_tag, pf):
    """
    tau1_vals, tau2_vals, fret_vals : (N_pixels,) float32
    pixel_mask_flat                 : (N_pixels,) bool
    Only valid (non-masked) pixels are histogrammed.
    """
    tau1_valid = tau1_vals[pixel_mask_flat]
    tau2_valid = tau2_vals[pixel_mask_flat]
    fret_valid = fret_vals[pixel_mask_flat]

    panels = [
        (tau1_valid, "τ₁ (ns)",  (0.0, 3.0),  "steelblue",   "hist_tau1.png"),
        (tau2_valid, "τ₂ (ns)",  (0.0, 3.0),  "seagreen",    "hist_tau2.png"),
        (fret_valid, "FRET (f)", (0.0, 1.0),  "darkorange",  "hist_fret.png"),
    ]

    for data, label, xlim, color, fname in panels:
        finite_data = data[np.isfinite(data)]
        clipped     = np.clip(finite_data, xlim[0], xlim[1])

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.hist(clipped, bins=100, color=color, alpha=0.75, edgecolor="none",
                range=xlim)
        ax.axvline(float(np.median(clipped)), color="red",
                   linewidth=1.5, linestyle="--", label="median")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Pixel count", fontsize=11)
        ax.set_title(f"{model_tag}\n{label} distribution", fontsize=10,
                     fontweight="bold")
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)
        ax.text(
            0.97, 0.97,
            f"n={len(clipped):,}\n"
            f"μ={float(np.mean(clipped)):.4f}\n"
            f"σ={float(np.std(clipped)):.4f}\n"
            f"med={float(np.median(clipped)):.4f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"  Histogram saved: {out_path}")


# ---------------------------------------------------------------------------
# Save scatter plots — tau1 vs tau2, tau1 vs fret, tau2 vs fret
# (self-consistency / correlation plots for experimental data)
# ---------------------------------------------------------------------------
def save_scatter_plots(tau1_vals, tau2_vals, fret_vals,
                       pixel_mask_flat, out_dir, model_tag, pf):
    """
    For experimental data there are no GT labels, so we plot pairwise
    correlations between the three predicted quantities as a self-consistency
    and sanity-check visualisation.
    """
    tau1_valid = tau1_vals[pixel_mask_flat]
    tau2_valid = tau2_vals[pixel_mask_flat]
    fret_valid = fret_vals[pixel_mask_flat]

    # Finite-only
    valid_idx = (
        np.isfinite(tau1_valid) &
        np.isfinite(tau2_valid) &
        np.isfinite(fret_valid)
    )
    tau1_v = tau1_valid[valid_idx]
    tau2_v = tau2_valid[valid_idx]
    fret_v = fret_valid[valid_idx]

    panels = [
        (tau1_v, tau2_v, "τ₁ (ns)", "τ₂ (ns)",
         (0, 3.0), (0, 3.0), "Blues",   "scatter_tau1_vs_tau2.png"),
        (tau1_v, fret_v, "τ₁ (ns)", "FRET (f)",
         (0, 3.0), (0, 1.0), "Greens",  "scatter_tau1_vs_fret.png"),
        (tau2_v, fret_v, "τ₂ (ns)", "FRET (f)",
         (0, 3.0), (0, 1.0), "Oranges", "scatter_tau2_vs_fret.png"),
    ]

    for xdata, ydata, xlabel, ylabel, xlim, ylim, cmap, fname in panels:
        if len(xdata) == 0:
            pf(f"  [SCATTER] No valid data for {fname} — skipping.")
            continue

        r_val = float("nan")
        if len(xdata) > 1:
            try:
                r_val, _ = pearsonr(xdata.astype(float), ydata.astype(float))
            except Exception:
                pass

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        hb = ax.hexbin(
            xdata, ydata,
            gridsize=80,
            bins="log",
            cmap=cmap,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            mincnt=1,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{model_tag}\n{xlabel} vs {ylabel}", fontsize=10,
                     fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.03, 0.97,
            f"n={len(xdata):,}\nr={r_val:.4f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        fig.colorbar(hb, ax=ax, pad=0.02).set_label("log₁₀(count)", fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"  Scatter saved: {out_path}")


# ---------------------------------------------------------------------------
# Combined overview figure: pixel maps + histograms in one panel
# ---------------------------------------------------------------------------
def save_overview_figure(tau1_map, tau2_map, fret_map,
                          tau1_vals, tau2_vals, fret_vals,
                          pixel_mask, pixel_mask_flat,
                          n_rows, n_cols, out_dir, model_tag, pf):
    masked_tau1 = np.where(pixel_mask, tau1_map, np.nan)
    masked_tau2 = np.where(pixel_mask, tau2_map, np.nan)
    masked_fret = np.where(pixel_mask, fret_map, np.nan)

    tau1_valid = np.clip(tau1_vals[pixel_mask_flat &
                                    np.isfinite(tau1_vals)], 0, 3.0)
    tau2_valid = np.clip(tau2_vals[pixel_mask_flat &
                                    np.isfinite(tau2_vals)], 0, 3.0)
    fret_valid = np.clip(fret_vals[pixel_mask_flat &
                                    np.isfinite(fret_vals)], 0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im0 = axes[0, 0].imshow(masked_tau1, cmap="inferno",
                              vmin=0, vmax=3.0, origin="upper",
                              interpolation="nearest", aspect="auto")
    axes[0, 0].set_title("τ₁ (ns) — pixel map", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Column", fontsize=9)
    axes[0, 0].set_ylabel("Row", fontsize=9)
    fig.colorbar(im0, ax=axes[0, 0], pad=0.02).set_label("ns", fontsize=9)

    im1 = axes[0, 1].imshow(masked_tau2, cmap="viridis",
                              vmin=0, vmax=3.0, origin="upper",
                              interpolation="nearest", aspect="auto")
    axes[0, 1].set_title("τ₂ (ns) — pixel map", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Column", fontsize=9)
    axes[0, 1].set_ylabel("Row", fontsize=9)
    fig.colorbar(im1, ax=axes[0, 1], pad=0.02).set_label("ns", fontsize=9)

    im2 = axes[0, 2].imshow(masked_fret, cmap="plasma",
                              vmin=0, vmax=1.0, origin="upper",
                              interpolation="nearest", aspect="auto")
    axes[0, 2].set_title("FRET (f) — pixel map", fontsize=10, fontweight="bold")
    axes[0, 2].set_xlabel("Column", fontsize=9)
    axes[0, 2].set_ylabel("Row", fontsize=9)
    fig.colorbar(im2, ax=axes[0, 2], pad=0.02).set_label("a.u.", fontsize=9)

    axes[1, 0].hist(tau1_valid, bins=100, color="steelblue",
                    alpha=0.75, edgecolor="none", range=(0, 3.0))
    axes[1, 0].axvline(float(np.median(tau1_valid)), color="red",
                       linewidth=1.5, linestyle="--")
    axes[1, 0].set_xlabel("τ₁ (ns)", fontsize=10)
    axes[1, 0].set_ylabel("Count", fontsize=10)
    axes[1, 0].set_title("τ₁ distribution", fontsize=10, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.2)

    axes[1, 1].hist(tau2_valid, bins=100, color="seagreen",
                    alpha=0.75, edgecolor="none", range=(0, 3.0))
    axes[1, 1].axvline(float(np.median(tau2_valid)), color="red",
                       linewidth=1.5, linestyle="--")
    axes[1, 1].set_xlabel("τ₂ (ns)", fontsize=10)
    axes[1, 1].set_ylabel("Count", fontsize=10)
    axes[1, 1].set_title("τ₂ distribution", fontsize=10, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.2)

    axes[1, 2].hist(fret_valid, bins=100, color="darkorange",
                    alpha=0.75, edgecolor="none", range=(0, 1.0))
    axes[1, 2].axvline(float(np.median(fret_valid)), color="red",
                       linewidth=1.5, linestyle="--")
    axes[1, 2].set_xlabel("FRET (f)", fontsize=10)
    axes[1, 2].set_ylabel("Count", fontsize=10)
    axes[1, 2].set_title("FRET distribution", fontsize=10, fontweight="bold")
    axes[1, 2].grid(True, alpha=0.2)

    fig.suptitle(
        f"Experimental data evaluation\n{model_tag}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"  Overview figure saved: {out_path}")

# ---------------------------------------------------------------------------
# Save per-pixel results and raw predictions as .mat files for MATLAB import
# ---------------------------------------------------------------------------
def save_mat_outputs(tau1_map, tau2_map, fret_map, pixel_mask,
                     preds, out_dir, model_tag, pf):
    """
    Save all per-pixel output arrays and raw model predictions as MATLAB .mat
    files into out_dir so they can be loaded directly in MATLAB via load().

    Files written
    -------------
    experimental_results.mat
        tau1_map    : (n_rows, n_cols)  float32  — τ₁ lifetime map in ns
        tau2_map    : (n_rows, n_cols)  float32  — τ₂ lifetime map in ns
        fret_map    : (n_rows, n_cols)  float32  — FRET fraction map
        pixel_mask  : (n_rows, n_cols)  uint8    — 1=valid, 0=masked pixel
        model_tag   : char array        str      — identifier string

    experimental_preds.mat
        preds       : (N_pixels, seq_len, 3)  float32
                      channel 0 = reconstructed full decay
                      channel 1 = short-lifetime component
                      channel 2 = long-lifetime component
        Note: N_pixels = n_rows * n_cols (row-major flattening)
    """
    from scipy.io import savemat

    results_path = os.path.join(out_dir, "experimental_results.mat")
    savemat(
        results_path,
        {
            "tau1_map":   tau1_map.astype(np.float64),
            "tau2_map":   tau2_map.astype(np.float64),
            "fret_map":   fret_map.astype(np.float64),
            "pixel_mask": pixel_mask.astype(np.uint8),
            "model_tag":  model_tag,
        },
        do_compression=True,
    )
    pf(f"  .mat results saved: {results_path}")

    preds_path = os.path.join(out_dir, "experimental_preds.mat")
    savemat(
        preds_path,
        {
            "preds": preds.astype(np.float64),
        },
        do_compression=True,
    )
    pf(f"  .mat preds saved  : {preds_path}")
# ---------------------------------------------------------------------------
# Evaluate a single weight file and save all outputs next to it
# ---------------------------------------------------------------------------
def evaluate_weight_file(weight_path, encoder_input, pixel_mask,
                          n_rows, n_cols, seq_len, n_out, gate_width_ns,
                          infer_batch, args, pf):
    """
    Load the model described by weight_path, run inference on encoder_input,
    extract tau1/tau2/fret, and save all outputs into the same directory as
    weight_path.
    """
    out_dir    = os.path.dirname(weight_path)
    model_tag  = os.path.basename(out_dir) + "/" + os.path.basename(weight_path)
    pf("=" * 70)
    pf(f"[EVAL] Weight file : {weight_path}")
    pf(f"[EVAL] Output dir  : {out_dir}")
    pf(f"[EVAL] Model tag   : {model_tag}")

    config = parse_config_from_path(weight_path, args)
    pf(f"[EVAL] Parsed config: {config}")

    # Build model
    model_type = config["model_type"]

    if model_type == "teacher":
        pf(f"[EVAL] Building teacher: layers={config['layers_teacher']}")
        model = build_teacher(seq_len, n_out, config["layers_teacher"])
    elif model_type == "student_qkeras":
        pf(f"[EVAL] Building QKeras student: "
           f"units={config['student_units']}  bits_k={config['bits_kernel']}  "
           f"bits_r={config['bits_recurrent']}  bits_a={config['bits_activation']}")
        try:
            model = build_qkeras_student(
                seq_len, n_out,
                config["student_units"],
                config["bits_kernel"],
                config["bits_recurrent"],
                config["bits_bias"],
                config["bits_activation"],
                config["bits_state"],
            )
        except ImportError:
            pf("[EVAL] WARNING: QKeras not available — "
               "falling back to vanilla float student for weight loading.")
            model = build_vanilla_student(
                seq_len, n_out, config["student_units"]
            )
    elif model_type == "student_vanilla":
        pf(f"[EVAL] Building vanilla student: units={config['student_units']}")
        model = build_vanilla_student(seq_len, n_out, config["student_units"])
    else:
        pf(f"[EVAL] WARNING: model_type='{model_type}' — "
           f"attempting teacher build with layers={config['layers_teacher']}")
        model = build_teacher(seq_len, n_out, config["layers_teacher"])

    # Load weights
    pf(f"[EVAL] Loading weights from: {weight_path}")
    try:
        model.load_weights(weight_path)
    except Exception as e:
        pf(f"[EVAL] ERROR loading weights: {e}")
        pf(f"[EVAL] Trying alternate teacher naming (enc_input/dec_input)...")
        try:
            model = build_teacher_student_names(
                seq_len, n_out,
                config["teacher_units"],
                config["teacher_layers"],
            )
            model.load_weights(weight_path)
            pf("[EVAL] Alternate teacher naming succeeded.")
        except Exception as e2:
            pf(f"[EVAL] ERROR: could not load weights with either naming. "
               f"Skipping.\n  {e2}")
            return None

    model.trainable = False
    pf(f"[EVAL] Model params: {model.count_params():,}")

    # Run inference
    pf(f"[EVAL] Running inference on {len(encoder_input):,} pixels...")
    t_axis   = np.arange(seq_len, dtype=np.float32) * gate_width_ns
    preds    = run_inference(model, encoder_input, seq_len, n_out, infer_batch, pf)
    pf(f"[EVAL] preds shape: {preds.shape}")

    # Extract lifetimes
    tau1_vals, tau2_vals, fret_vals = extract_lifetimes(preds, t_axis)
    pf(f"[EVAL] tau1 range: [{tau1_vals.min():.4f}, {tau1_vals.max():.4f}]")
    pf(f"[EVAL] tau2 range: [{tau2_vals.min():.4f}, {tau2_vals.max():.4f}]")
    pf(f"[EVAL] fret range: [{fret_vals.min():.4f}, {fret_vals.max():.4f}]")

    # Reshape to pixel maps
    tau1_map = tau1_vals.reshape(n_rows, n_cols)
    tau2_map = tau2_vals.reshape(n_rows, n_cols)
    fret_map = fret_vals.reshape(n_rows, n_cols)

    pixel_mask_flat = pixel_mask.flatten()

    # Statistics — valid pixels only
    pf(f"[EVAL] Computing statistics (valid pixels only)...")
    tau1_valid_flat = tau1_vals[pixel_mask_flat]
    tau2_valid_flat = tau2_vals[pixel_mask_flat]
    fret_valid_flat = fret_vals[pixel_mask_flat]

    pf("  [stats] τ₁:")
    s1 = compute_stats(tau1_valid_flat, "tau1_ns",  pf)
    pf("  [stats] τ₂:")
    s2 = compute_stats(tau2_valid_flat, "tau2_ns",  pf)
    pf("  [stats] FRET:")
    sf = compute_stats(fret_valid_flat, "fret",     pf)

    results = {
        "weight_file":  weight_path,
        "model_tag":    model_tag,
        "model_config": config,
        "n_pixels_total": int(n_rows * n_cols),
        "n_pixels_valid": int(pixel_mask_flat.sum()),
        "tau1": s1,
        "tau2": s2,
        "fret": sf,
    }

    stats_path = os.path.join(out_dir, "experimental_stats.json")
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    pf(f"[EVAL] Stats saved: {stats_path}")

    # Save raw per-pixel arrays as .npy for downstream use
    np.save(os.path.join(out_dir, "experimental_tau1.npy"), tau1_map)
    np.save(os.path.join(out_dir, "experimental_tau2.npy"), tau2_map)
    np.save(os.path.join(out_dir, "experimental_fret.npy"), fret_map)
    pf(f"[EVAL] Per-pixel .npy arrays saved to {out_dir}")

    # Save .mat files for MATLAB import
    save_mat_outputs(tau1_map, tau2_map, fret_map, pixel_mask,
                     preds, out_dir, model_tag, pf)

    # Pixel maps
    save_pixel_maps(tau1_map, tau2_map, fret_map, pixel_mask,
                    n_rows, n_cols, out_dir, model_tag, pf)

    # Histograms
    save_histograms(tau1_vals, tau2_vals, fret_vals, pixel_mask_flat,
                    out_dir, model_tag, pf)

    # Scatter plots (pairwise correlations)
    save_scatter_plots(tau1_vals, tau2_vals, fret_vals, pixel_mask_flat,
                       out_dir, model_tag, pf)

    # Combined overview
    save_overview_figure(
        tau1_map, tau2_map, fret_map,
        tau1_vals, tau2_vals, fret_vals,
        pixel_mask, pixel_mask_flat,
        n_rows, n_cols, out_dir, model_tag, pf,
    )

    pf(f"[EVAL] Done: {model_tag}")
    pf("=" * 70)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    pf   = lambda s: print(s, flush=True)

    setup_gpu()

    # ------------------------------------------------------------------
    # 1. Load and preprocess experimental .mat file
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 1. Load and preprocess experimental .mat file
    # ------------------------------------------------------------------
    pf("=" * 70)
    pf("PHASE 1: Load and preprocess experimental .mat file")
    pf("=" * 70)
    encoder_input, pixel_mask, n_rows, n_cols = load_and_preprocess_mat(
        args.mat_file,
        args.n_rows,
        args.n_cols,
        args.seq_len,
        args.baseline_bins,
        pf,
    )

    pf(f"[MAIN] encoder_input shape : {encoder_input.shape}")
    pf(f"[MAIN] pixel_mask shape    : {pixel_mask.shape}  "
       f"valid={pixel_mask.sum():,}")

    # ------------------------------------------------------------------
    # 1b. Apply external binary mask (if provided)
    # ------------------------------------------------------------------
    if args.mask_file is not None:
        pf("=" * 70)
        pf("PHASE 1b: Applying external binary mask")
        pf("=" * 70)
        ext_mask = load_binary_mask(args.mask_file, n_rows, n_cols, pf)
        n_before = int(pixel_mask.sum())
        pixel_mask = pixel_mask & ext_mask
        n_after  = int(pixel_mask.sum())
        pf(
            f"[MAIN] pixel_mask after combining with external mask: "
            f"{n_after:,} valid pixels "
            f"(was {n_before:,} from intensity mask alone, "
            f"removed {n_before - n_after:,})"
        )
    else:
        pf("[MAIN] No --mask-file provided — using intensity-based pixel_mask only.")

    # ------------------------------------------------------------------
    # 2. Collect all weight files to evaluate
    # ------------------------------------------------------------------
    pf("=" * 70)
    pf("PHASE 2: Discover weight files")
    pf("=" * 70)

    weight_files = []

    if args.teacher_ckpt is not None:
        if os.path.isfile(args.teacher_ckpt):
            weight_files.append(args.teacher_ckpt)
            pf(f"[MAIN] Added explicit teacher ckpt: {args.teacher_ckpt}")
        else:
            pf(f"[MAIN] WARNING: --teacher-ckpt not found: {args.teacher_ckpt}")

    if args.ablation_root is not None:
        if not os.path.isdir(args.ablation_root):
            pf(f"[MAIN] ERROR: --ablation-root is not a directory: "
               f"{args.ablation_root}")
            sys.exit(1)
        discovered = discover_weight_files(args.ablation_root, pf)
        for wf in discovered:
            if wf not in weight_files:
                weight_files.append(wf)

    if not weight_files:
        pf("[MAIN] ERROR: No weight files found. "
           "Check --ablation-root and --teacher-ckpt.")
        sys.exit(1)

    pf(f"[MAIN] Total weight files to evaluate: {len(weight_files)}")

    # ------------------------------------------------------------------
    # 3. Evaluate each weight file
    # ------------------------------------------------------------------
    pf("=" * 70)
    pf("PHASE 3: Evaluate all weight files")
    pf("=" * 70)

    all_results = []
    failed      = []

    for i, weight_path in enumerate(weight_files):
        pf(f"\n[MAIN] [{i + 1}/{len(weight_files)}] Evaluating: {weight_path}")
        try:
            result = evaluate_weight_file(
                weight_path,
                encoder_input,
                pixel_mask,
                n_rows,
                n_cols,
                args.seq_len,
                args.n_out,
                args.gate_width_ns,
                args.infer_batch,
                args,
                pf,
            )
            if result is not None:
                all_results.append(result)
            else:
                failed.append(weight_path)
        except Exception as exc:
            import traceback
            pf(f"[MAIN] ERROR evaluating {weight_path}:\n{traceback.format_exc()}")
            failed.append(weight_path)

    # ------------------------------------------------------------------
    # 4. Write summary JSON with all results in one place
    # ------------------------------------------------------------------
    pf("=" * 70)
    pf("PHASE 4: Writing summary")
    pf("=" * 70)

    summary = {
        "mat_file":        args.mat_file,
        "n_rows":          n_rows,
        "n_cols":          n_cols,
        "seq_len":         args.seq_len,
        "gate_width_ns":   args.gate_width_ns,
        "baseline_bins":   args.baseline_bins,
        "n_pixels_total":  int(n_rows * n_cols),
        "n_pixels_valid":  int(pixel_mask.sum()),
        "n_models_ok":     len(all_results),
        "n_models_failed": len(failed),
        "failed_files":    failed,
        "results":         all_results,
    }

    # Save summary next to the mat file if possible, else next to ablation root
    if args.ablation_root is not None:
        summary_dir = args.ablation_root
    else:
        summary_dir = os.path.dirname(os.path.abspath(args.teacher_ckpt))

    summary_path = os.path.join(summary_dir, "experimental_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    pf(f"[MAIN] Summary saved: {summary_path}")

    pf("=" * 70)
    pf(f"[MAIN] DONE.")
    pf(f"[MAIN] Models evaluated OK  : {len(all_results)}")
    pf(f"[MAIN] Models failed        : {len(failed)}")
    if failed:
        pf("[MAIN] Failed files:")
        for f_path in failed:
            pf(f"  {f_path}")
    pf("=" * 70)


if __name__ == "__main__":
    main()