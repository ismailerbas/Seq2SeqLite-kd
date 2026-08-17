#!/usr/bin/env python3
"""
predict_5_pixels_local.py -- Pure NumPy replica of the student GRU seq2seq
model, using the FAKE-QUANTIZED float32 weights and the CONFIRMED-CORRECT
0.5-slope QKeras hard_sigmoid (verified via inspect.getsource on the real
installed QGRUCell), run on all 5 experimental pixels. Computes both the
raw decay predictions and the extracted lifetimes (tau1, tau2, fret) for
each pixel, and compares both against the real ground truth files.

No TensorFlow, no QKeras required -- only numpy.

EDIT THE PATHS BELOW to match your actual files before running.
"""

import numpy as np
import os
import sys

# ============================================================================
# EDIT THESE PATHS
# ============================================================================
NPZ_PATH = r"D:\ismail\IBM\s2slitekd\hls_seq2seqlite\student_weights_fp32.npz"
CSV_DIR = r"E:\ismail\IBM\nminewmodel\vitis"
NUM_PIXELS = 5
# ============================================================================

UNITS = 32
BASELINE_BINS = 10
SEQ_LEN = 135
GATE_WIDTH_NS = 0.09


def hard_sigmoid(x):
    # CONFIRMED via inspect.getsource(cell.recurrent_activation) on a real
    # instantiated QGRUCell: QKeras's own hard_sigmoid, slope 0.5, NOT the
    # standard tf.keras.backend hard_sigmoid (slope 0.2).
    return np.clip(0.5 * x + 0.5, 0.0, 1.0)


def gru_forward(x_seq, kernel, recurrent_kernel, bias, units, init_state):
    T = x_seq.shape[0]
    h = init_state.copy()
    outputs = np.zeros((T, units), dtype=np.float32)

    Wz = kernel[:, 0 * units:1 * units]
    Wr = kernel[:, 1 * units:2 * units]
    Wh = kernel[:, 2 * units:3 * units]
    Uz = recurrent_kernel[:, 0 * units:1 * units]
    Ur = recurrent_kernel[:, 1 * units:2 * units]
    Uh = recurrent_kernel[:, 2 * units:3 * units]
    bz = bias[0 * units:1 * units]
    br = bias[1 * units:2 * units]
    bh = bias[2 * units:3 * units]

    for t in range(T):
        x = x_seq[t]
        x_z = x @ Wz + bz
        x_r = x @ Wr + br
        x_h = x @ Wh + bh
        rec_z = h @ Uz
        rec_r = h @ Ur
        z = hard_sigmoid(x_z + rec_z)
        r = hard_sigmoid(x_r + rec_r)
        rec_h = (r * h) @ Uh
        # quantized_tanh reduces algebraically to clip(x,-1,1) when
        # use_real_tanh=False (default, never overridden) -- confirmed
        # earlier this session via inspect.getsource(quantized_tanh.__call__).
        hh = np.clip(x_h + rec_h, -1.0, 1.0)
        h = z * h + (1 - z) * hh
        outputs[t] = h

    return outputs, h


def preprocess_pixel(raw):
    baseline = raw[:BASELINE_BINS].mean()
    corrected = np.clip(raw - baseline, 0.0, None)
    max_val = corrected.max()
    norm = corrected / max_val if max_val > 0 else corrected
    return norm.astype(np.float32), baseline, max_val


def extract_lifetimes(preds):
    # preds: (SEQ_LEN, 3). channel 0 = full decay, 1 = short, 2 = long.
    # Replicates extract_lifetimes() from eval_experimental.py exactly.
    t = np.arange(SEQ_LEN, dtype=np.float64) * GATE_WIDTH_NS
    ch1 = preds[:, 1].astype(np.float64)
    ch2 = preds[:, 2].astype(np.float64)
    int1 = np.trapz(ch1, t)
    int2 = np.trapz(ch2, t)
    amp1 = ch1[0]
    amp2 = ch2[0]
    tau1 = (int1 / amp1) if amp1 > 1e-6 else 0.0
    tau2 = (int2 / amp2) if amp2 > 1e-6 else 0.0
    denom = amp1 + amp2
    fret = (amp1 / denom) if denom > 1e-6 else 0.5
    return tau1, tau2, fret


def run_one_pixel(idx, data):
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
    enc_input = norm.reshape(-1, 1).astype(np.float32)

    enc_kernel = data["sencgru_kernel_fq"]
    enc_recurrent = data["sencgru_recurrent_kernel_fq"]
    enc_bias = data["sencgru_bias_fq"]
    _, h_enc_final = gru_forward(
        enc_input, enc_kernel, enc_recurrent, enc_bias, UNITS,
        np.zeros(UNITS, dtype=np.float32)
    )

    dec_kernel = data["sdecgru_kernel_fq"]
    dec_recurrent = data["sdecgru_recurrent_kernel_fq"]
    dec_bias = data["sdecgru_bias_fq"]
    dec_input = np.zeros_like(enc_input)
    dec_hidden_seq, _ = gru_forward(
        dec_input, dec_kernel, dec_recurrent, dec_bias, UNITS, h_enc_final
    )

    dense_kernel = data["sdec_dense_kernel_fq"]
    dense_bias = data["sdec_dense_bias_fq"]
    preds = dec_hidden_seq @ dense_kernel + dense_bias

    expected = np.loadtxt(expected_path, delimiter=",")
    if expected.shape[0] != SEQ_LEN:
        print(f"[pixel_{idx}] ERROR: {expected_path} has {expected.shape[0]} rows, "
              f"expected {SEQ_LEN}. Skipping.")
        return None

    decay_diff = np.abs(preds - expected)
    max_decay_err = decay_diff.max()
    max_decay_err_loc = np.unravel_index(np.argmax(decay_diff), decay_diff.shape)

    tau1_pred, tau2_pred, fret_pred = extract_lifetimes(preds)
    tau1_true, tau2_true, fret_true = extract_lifetimes(expected)

    print(f"\n========== pixel_{idx} ==========")
    print(f"baseline={baseline:.6f}  max_val={max_val:.6f}")
    print(f"tpsf_in[0..9]: {norm[:10]}")
    print(f"\nDecay predictions vs expected, t=0..5:")
    for t in range(6):
        print(f"  t={t}  pred={preds[t]}  expected={expected[t]}")
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
    if not os.path.isfile(NPZ_PATH):
        print(f"ERROR: NPZ_PATH not found: {NPZ_PATH}")
        sys.exit(1)

    print(f"Loading weights from {NPZ_PATH}")
    data = np.load(NPZ_PATH)

    required_keys = [
        "sencgru_kernel_fq", "sencgru_recurrent_kernel_fq", "sencgru_bias_fq",
        "sdecgru_kernel_fq", "sdecgru_recurrent_kernel_fq", "sdecgru_bias_fq",
        "sdec_dense_kernel_fq", "sdec_dense_bias_fq",
    ]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        print(f"ERROR: missing expected keys: {missing}")
        print(f"Available keys: {list(data.files)}")
        sys.exit(1)

    results = []
    for idx in range(1, NUM_PIXELS + 1):
        r = run_one_pixel(idx, data)
        if r is not None:
            results.append(r)

    print("\n\n========== SUMMARY ==========")
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