#!/usr/bin/env python3
"""
verify_student_model_local_fq.py -- Pure NumPy replica of the student GRU
seq2seq model, using the FAKE-QUANTIZED (post-quantizer, pre-int8-packing)
float32 weights extracted by extract_student_weights.py -- i.e. the exact
same weight values QGRUCell.call() actually uses during every forward pass
(quantized_kernel = self.kernel_quantizer_internal(self.kernel)), NOT the
raw unquantized weights.

No TensorFlow, no QKeras required -- only numpy.

WINDOWS VERSION: run this on your local Windows machine where the CSVs
and the .npz file actually live.

EDIT THE THREE PATHS BELOW to match your actual files before running.
Use raw strings (r"...") or forward slashes so backslashes don't need
escaping.
"""

import numpy as np
import os
import sys

# ============================================================================
# EDIT THESE THREE PATHS
# ============================================================================
NPZ_PATH = r"D:\ismail\IBM\s2slitekd\hls_seq2seqlite\student_weights_fp32.npz"
RAW_PIXEL_CSV = r"E:\ismail\IBM\nminewmodel\vitis\raw_pixel_1.csv"
EXPECTED_OUT_CSV = r"E:\ismail\IBM\nminewmodel\vitis\expected_out_1.csv"
# ============================================================================

UNITS = 32
BASELINE_BINS = 10


def hard_sigmoid(x):
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
        hh = np.clip(x_h + rec_h, -1.0, 1.0)
        h = z * h + (1 - z) * hh
        outputs[t] = h

    return outputs, h


def main():
    for path, label in [(NPZ_PATH, "NPZ_PATH"), (RAW_PIXEL_CSV, "RAW_PIXEL_CSV")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} does not exist on disk: {path}")
            print("Edit the path constants at the top of this script and re-run.")
            sys.exit(1)

    print(f"Loading weights from {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    print("Available keys in npz file:")
    for k in data.files:
        print(f"  {k}  shape={data[k].shape}")

    required_keys = [
        "sencgru_kernel_fq", "sencgru_recurrent_kernel_fq", "sencgru_bias_fq",
        "sdecgru_kernel_fq", "sdecgru_recurrent_kernel_fq", "sdecgru_bias_fq",
        "sdec_dense_kernel_fq", "sdec_dense_bias_fq",
    ]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        print(f"\nERROR: missing expected keys: {missing}")
        print("Check the key list printed above and tell me the actual names.")
        sys.exit(1)

    print(f"\nLoading raw pixel trace from {RAW_PIXEL_CSV}")
    raw = np.loadtxt(RAW_PIXEL_CSV, delimiter=",")
    print(f"raw shape: {raw.shape}, first 10 values: {raw[:10]}")

    baseline = raw[:BASELINE_BINS].mean()
    corrected = np.clip(raw - baseline, 0.0, None)
    max_val = corrected.max()
    norm = corrected / max_val if max_val > 0 else corrected
    print(f"baseline={baseline:.6f}  max_val={max_val:.6f}")
    print(f"normalized tpsf_in[0..9]: {norm[:10]}")

    enc_input = norm.reshape(-1, 1).astype(np.float32)

    # ---- FAKE-QUANTIZED weights (post-quantizer, what QGRUCell.call()
    #      actually multiplies by on every forward pass) ----
    enc_kernel = data["sencgru_kernel_fq"]
    enc_recurrent = data["sencgru_recurrent_kernel_fq"]
    enc_bias = data["sencgru_bias_fq"]
    _, h_enc_final = gru_forward(
        enc_input, enc_kernel, enc_recurrent, enc_bias, UNITS,
        np.zeros(UNITS, dtype=np.float32)
    )
    print(f"\nh_enc_final[0..5]: {h_enc_final[:6]}")

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

    print("\nNumPy replica predictions (fake-quantized weights), t=0..5:")
    for t in range(6):
        print(f"  t={t}  {preds[t]}")

    if os.path.isfile(EXPECTED_OUT_CSV):
        expected = np.loadtxt(EXPECTED_OUT_CSV, delimiter=",")
        print("\nexpected_out_1.csv, t=0..5:")
        for t in range(6):
            print(f"  t={t}  {expected[t]}")
        diff = np.abs(preds[:expected.shape[0]] - expected)
        print(f"\nMax abs diff (NumPy replica, fake-quantized weights, "
              f"vs expected_out_1.csv): {diff.max():.6f}")
    else:
        print(f"\nEXPECTED_OUT_CSV not found at {EXPECTED_OUT_CSV} -- "
              f"edit the path and re-run to get the comparison.")


if __name__ == "__main__":
    main()
