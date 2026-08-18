#!/usr/bin/env python3
"""
verify_reshape.py -- Cross-check the manual pixel-index formula against the
SAME reshape convention eval_experimental.py actually uses for tau1map /
tau2map / fretmap (evaluate_weight_file: tau1map = tau1vals.reshape(nrows, ncols)).

If preds.reshape(nrows, ncols, seq_len, n_out) indexed at [row-1, col-1, :, :]
matches expected_out_k.csv exactly, pixel pairing is triple-confirmed correct.
If it does NOT match, the manual (row-1)*ncols+col formula used in MATLAB was
wrong and THIS is the bug.

Requires only numpy and scipy (no TensorFlow / QKeras).

EDIT THE PATHS BELOW.
"""

import numpy as np
from scipy.io import loadmat
import os
import sys

# ============================================================================
# EDIT THESE PATHS
# ============================================================================
PREDS_MAT_PATH = r"E:\ismail\IBM\nminewmodel\expsliced\1mm328bkd06\experimental_preds.mat"
EXPECTED_OUT_DIR = r"E:\ismail\IBM\nminewmodel\vitis"
OUTPUT_DIR = r"E:\ismail\IBM\nminewmodel\vitis"

# From your MATLAB ginput extraction (row, col are 1-based, MATLAB convention)
PIXELS = [
    {"idx": 1, "row": 110, "col": 279},
    {"idx": 2, "row": 74,  "col": 285},
    {"idx": 3, "row": 57,  "col": 291},
    {"idx": 4, "row": 47,  "col": 319},
    {"idx": 5, "row": 144, "col": 225},
]

N_ROWS = 250
N_COLS = 484
SEQ_LEN = 135
N_OUT = 3
# ============================================================================


def main():
    if not os.path.isfile(PREDS_MAT_PATH):
        print(f"ERROR: PREDS_MAT_PATH not found: {PREDS_MAT_PATH}")
        sys.exit(1)

    print(f"Loading {PREDS_MAT_PATH}")
    mat = loadmat(PREDS_MAT_PATH)
    if "preds" not in mat:
        print(f"ERROR: 'preds' variable not found. Available keys: "
              f"{[k for k in mat.keys() if not k.startswith('__')]}")
        sys.exit(1)

    preds = mat["preds"]
    print(f"preds raw shape: {preds.shape}")

    n_pixels_total = N_ROWS * N_COLS
    if preds.shape[0] != n_pixels_total:
        print(f"ERROR: preds.shape[0]={preds.shape[0]} != "
              f"N_ROWS*N_COLS={n_pixels_total}. Check N_ROWS/N_COLS.")
        sys.exit(1)

    # Same reshape convention as evaluate_weight_file's tau1map = tau1vals.reshape(nrows, ncols):
    # numpy default reshape is C-order (row-major), so this exactly matches
    # how the pixel maps (which visually look like real images) were built.
    preds_4d = preds.reshape(N_ROWS, N_COLS, SEQ_LEN, N_OUT)
    print(f"preds reshaped to: {preds_4d.shape}")

    for p in PIXELS:
        idx, row, col = p["idx"], p["row"], p["col"]
        # MATLAB row/col are 1-based; numpy indices are 0-based.
        extracted = preds_4d[row - 1, col - 1, :, :]

        manual_idx0 = (row - 1) * N_COLS + (col - 1)  # 0-based manual formula
        manual_extracted = preds[manual_idx0, :, :]

        match = np.allclose(extracted, manual_extracted)
        print(f"\npixel {idx}: row={row} col={col}")
        print(f"  reshape-based   preds_4d[{row-1},{col-1},0,:] = {extracted[0]}")
        print(f"  manual-formula  preds[{manual_idx0},0,:]      = {manual_extracted[0]}")
        print(f"  MATCH: {match}")

        expected_path = os.path.join(EXPECTED_OUT_DIR, f"expected_out_{idx}.csv")
        if os.path.isfile(expected_path):
            expected_csv = np.loadtxt(expected_path, delimiter=",")
            diff_reshape = np.abs(extracted - expected_csv).max()
            diff_manual = np.abs(manual_extracted - expected_csv).max()
            print(f"  max abs diff vs expected_out_{idx}.csv "
                  f"(reshape-based): {diff_reshape:.6f}")
            print(f"  max abs diff vs expected_out_{idx}.csv "
                  f"(manual-formula): {diff_manual:.6f}")
        else:
            print(f"  (expected_out_{idx}.csv not found at {expected_path})")

        out_path = os.path.join(OUTPUT_DIR, f"expected_out_reshape_verified_{idx}.csv")
        np.savetxt(out_path, extracted, delimiter=",")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()