#!/usr/bin/env python3
"""
predict_5_pixels_real_model.py -- Run the ACTUAL trained QKeras model
(real TensorFlow + real QKeras) on all 5 experimental pixels, compute
both the raw decay predictions and extracted lifetimes (tau1, tau2, fret)
for each, and compare against the real ground truth files.

Requires: tensorflow (2.10.x), qkeras, numpy.

EDIT THE PATHS BELOW to match your actual files before running.
"""

import numpy as np
import os
import sys
import tensorflow as tf
from qkeras import QDense, QGRU, quantized_bits, quantized_tanh

# ============================================================================
# EDIT THESE PATHS
# ============================================================================
WEIGHTS_H5_PATH = r"D:\ismail\IBM\s2slitekd\hls_seq2seqlite\student_final.weights.h5"
CSV_DIR = r"E:\ismail\IBM\nminewmodel\vitis"
NUM_PIXELS = 5
# ============================================================================

UNITS = 32
BITS = 8
BASELINE_BINS = 10
SEQ_LEN = 135
GATE_WIDTH_NS = 0.09


def qwk(): return quantized_bits(BITS, 0, 1, alpha=1.0)
def qwr(): return quantized_bits(BITS, 0, 1, alpha=1.0)
def qwb(): return quantized_bits(BITS, 0, 1, alpha=1.0)
def qa(): return quantized_tanh(bits=BITS, symmetric=True)
def qs(): return quantized_bits(BITS, 0, 1, alpha=1.0)
def qd(): return quantized_bits(BITS, 0)


def build_model():
    enc_in = tf.keras.Input(shape=(None, 1), name="senc_input")
    dec_in = tf.keras.Input(shape=(None, 1), name="sdec_input")
    senc_out, senc_state = QGRU(
        units=UNITS, activation=qa(), kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(), bias_quantizer=qwb(), state_quantizer=qs(),
        return_state=True, name="sencgru",
    )(enc_in)
    sdec_hid, _ = QGRU(
        units=UNITS, activation=qa(), kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(), bias_quantizer=qwb(), state_quantizer=qs(),
        return_sequences=True, return_state=True, name="sdecgru",
    )(dec_in, initial_state=senc_state)
    out = QDense(
        3, kernel_quantizer=qd(), bias_quantizer=qd(), activation="linear",
        name="sdec_dense",
    )(sdec_hid)
    return tf.keras.Model(inputs=[enc_in, dec_in], outputs=out)


def preprocess_pixel(raw):
    baseline = raw[:BASELINE_BINS].mean()
    corrected = np.clip(raw - baseline, 0.0, None)
    max_val = corrected.max()
    norm = corrected / max_val if max_val > 0 else corrected
    return norm.astype(np.float32), baseline, max_val


def extract_lifetimes(preds):
    h = GATE_WIDTH_NS
    ch1 = preds[:, 1].astype(np.float64)
    ch2 = preds[:, 2].astype(np.float64)

    int1 = h * (0.5 * ch1[0] + ch1[1:-1].sum() + 0.5 * ch1[-1])
    int2 = h * (0.5 * ch2[0] + ch2[1:-1].sum() + 0.5 * ch2[-1])

    amp1 = ch1[0]
    amp2 = ch2[0]
    tau1 = (int1 / amp1) if amp1 > 1e-6 else 0.0
    tau2 = (int2 / amp2) if amp2 > 1e-6 else 0.0
    denom = amp1 + amp2
    fret = (amp1 / denom) if denom > 1e-6 else 0.5
    return tau1, tau2, fret


def run_one_pixel(idx, model):
    raw_path = os.path.join(CSV_DIR, f"raw_pixel_{idx}.csv")
    expected_path = os.path.join(CSV_DIR, f"expected_out_{idx}.csv")

    if not os.path.isfile(raw_path):
        print(f"[pixel_{idx}] ERROR: {raw_path} not found. Skipping.")
        return None
    if not os.path.isfile(expected_path):
        print(f"[pixel_{idx}] ERROR: {expected_path} not found. Skipping.")
        return None

    raw = np.loadtxt(raw_path, delimiter=",")
    if raw.shape[0] != SEQ_LEN:
        print(f"[pixel_{idx}] ERROR: {raw_path} has {raw.shape[0]} values, "
              f"expected {SEQ_LEN}. Skipping.")
        return None

    norm, baseline, max_val = preprocess_pixel(raw)
    enc_input = norm.reshape(1, -1, 1)
    dec_input = np.zeros_like(enc_input)

    pred = model({"senc_input": enc_input, "sdec_input": dec_input}, training=False)
    pred = pred.numpy()[0]

    expected = np.loadtxt(expected_path, delimiter=",")
    if expected.shape[0] != SEQ_LEN:
        print(f"[pixel_{idx}] ERROR: {expected_path} has {expected.shape[0]} rows, "
              f"expected {SEQ_LEN}. Skipping.")
        return None

    decay_diff = np.abs(pred - expected)
    max_decay_err = decay_diff.max()
    max_decay_err_loc = np.unravel_index(np.argmax(decay_diff), decay_diff.shape)

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(pred)
    tau1_true, tau2_true, fret_true = extract_lifetimes(expected)

    print(f"\n========== pixel_{idx} (REAL TF/QKeras model) ==========")
    print(f"baseline={baseline:.6f}  max_val={max_val:.6f}")
    print(f"\nDecay predictions vs expected, t=0..5:")
    for t in range(6):
        print(f"  t={t}  pred={pred[t]}  expected={expected[t]}")
    print(f"\nMax abs decay error: {max_decay_err:.6f} at t={max_decay_err_loc[0]}, "
          f"channel={max_decay_err_loc[1]}")
    print(f"\nLifetimes:")
    print(f"  tau1: predicted={tau1_pred:.6f} ns   ground truth={tau1_true:.6f} ns   "
          f"diff={abs(tau1_pred - tau1_true):.6f} ns")
    print(f"  tau2: predicted={tau2_pred:.6f} ns   ground truth={tau2_true:.6f} ns   "
          f"diff={abs(tau2_pred - tau2_true):.6f} ns")
    print(f"  fret: predicted={fret_pred:.6f}      ground truth={fret_true:.6f}      "
          f"diff={abs(fret_pred - fret_true):.6f}")

    return {
        "idx": idx,
        "max_decay_err": max_decay_err,
        "tau1_pred": tau1_pred, "tau1_true": tau1_true,
        "tau2_pred": tau2_pred, "tau2_true": tau2_true,
        "fret_pred": fret_pred, "fret_true": fret_true,
    }


def main():
    if not os.path.isfile(WEIGHTS_H5_PATH):
        print(f"ERROR: WEIGHTS_H5_PATH not found: {WEIGHTS_H5_PATH}")
        sys.exit(1)

    print("Building model architecture...")
    model = build_model()
    print(f"Loading weights from {WEIGHTS_H5_PATH}")
    model.load_weights(WEIGHTS_H5_PATH)
    print("Weights loaded OK.")

    results = []
    for idx in range(1, NUM_PIXELS + 1):
        r = run_one_pixel(idx, model)
        if r is not None:
            results.append(r)

    print("\n\n========== SUMMARY (REAL TF/QKeras model) ==========")
    print(f"{'pixel':<8}{'max_decay_err':<16}{'tau1_diff_ns':<15}"
          f"{'tau2_diff_ns':<15}{'fret_diff':<12}")
    for r in results:
        tau1_diff = abs(r["tau1_pred"] - r["tau1_true"])
        tau2_diff = abs(r["tau2_pred"] - r["tau2_true"])
        fret_diff = abs(r["fret_pred"] - r["fret_true"])
        print(f"{r['idx']:<8}{r['max_decay_err']:<16.6f}{tau1_diff:<15.6f}"
              f"{tau2_diff:<15.6f}{fret_diff:<12.6f}")


if __name__ == "__main__":
    main()