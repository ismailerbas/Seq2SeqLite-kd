#!/usr/bin/env python3
"""
extract_student_weights.py — Extract QKeras Seq2SeqLite student weights into
int8 arrays and a C header for Vitis HLS.

Loads a single student_best.weights.h5 checkpoint (rebuilt via the exact
build_student architecture from train_student_vanilla_kd.py), splits each
QGRU kernel/recurrent_kernel/bias into [z, r, h] gates, quantizes to int8
using quantized_bits(bits, 0, 1, alpha=1.0), and writes into --out-dir:
  student_weights_int8.h      (Vitis HLS header, static const int8 arrays)
  student_weights_fp32.npz    (pre-quantization float32 tensors)
  student_weights_int8.npz    (post-quantization int8 tensors)
  extraction_report.json      (per-tensor shape/bits/min/max/saturation)

Confirmed architecture (from train_student_vanilla_kd.py / eval_experimental.py
build_student() / build_vanilla_student(), cross-checked against the real
QKeras source at qkeras/qrecurrent.py):

  senc_input (None,1) --> QGRU(units, name="sencgru", return_state=True)
  sdec_input (None,1) --> QGRU(units, name="sdecgru",
                                return_sequences=True, return_state=True,
                                initial_state=s_enc_state)
                       --> QDense(n_out, name="sdec_dense", activation="linear")

QGRUCell defaults confirmed from qkeras/qrecurrent.py source:
  implementation=1, reset_after=False  -> single bias vector (3*units,)
  recurrent_activation default = 'hard_sigmoid' (never overridden, so gates
  z, r use hard_sigmoid, UNQUANTIZED, applied to the quantized matmul sums).
  activation is overridden with quantized_tanh(bits=bits_activation,
  symmetric=True) -> used ONLY for the candidate hidden state h_tilde.

Gate column order confirmed from QGRUCell.call() matmul slicing:
  columns [0 : units]         -> z (update gate)
  columns [units : 2*units]   -> r (reset gate)
  columns [2*units : 3*units] -> h (candidate hidden state)
Bias (3*units,) uses the identical column split.

GRU update equations (reset_after=False, matches paper Appendix A exactly):
  z  = hard_sigmoid( x . Wz + h_prev . Uz + bz )
  r  = hard_sigmoid( x . Wr + h_prev . Ur + br )
  hh = quantized_tanh( x . Wh + (r * h_prev) . Uh + bh )
  h_new = z * h_prev + (1 - z) * hh

Quantizer for kernel / recurrent_kernel / bias / state:
  quantized_bits(bits, 0, 1, alpha=1.0)   integer=0, symmetric=1
  -> zero integer bits -> weight values live in [-1, 1) at step 2^-(bits-1)

Quantizer for QDense kernel / bias:
  quantized_bits(bits, 0)                 integer=0, symmetric=0 (default)
  -> same [-1, 1) range, same step, but -128 is a legal (asymmetric) code.

Usage:
  python extract_student_weights.py \
      --weights-h5    /path/to/student_best.weights.h5 \
      --student-args  /path/to/student_args.json \
      --out-dir       /path/to/hls_weights \
      --seq-len       135 \
      --n-out         3
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


# ==============================================================================
# Exact replica of build_student() / build_vanilla_student() from
# train_student_vanilla_kd.py and eval_experimental.py. Layer names MUST match
# exactly: senc_input, sdec_input, sencgru, sdecgru, sdec_dense.
# ==============================================================================
def build_student_for_extraction(
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
        name="student_extraction",
    )
    return student_model


# ==============================================================================
# Quantize a raw float weight array using the REAL qkeras quantizer object
# (not a reimplementation), then pack into int8 codes at the given step size.
# ==============================================================================
def quantize_and_pack_int8(raw_array, quantizer, bits, tensor_name, report):
    tf_tensor = tf.constant(raw_array, dtype=tf.float32)
    fake_quant_f = quantizer(tf_tensor).numpy().astype(np.float32)

    step = 2.0 ** (-(bits - 1))
    codes_float = fake_quant_f / step
    codes_rounded = np.round(codes_float)

    int8_min, int8_max = -128, 127
    n_clipped = int(np.sum((codes_rounded < int8_min) | (codes_rounded > int8_max)))
    codes_clipped = np.clip(codes_rounded, int8_min, int8_max)
    int8_codes = codes_clipped.astype(np.int8)

    report[tensor_name] = {
        "shape": list(raw_array.shape),
        "bits": bits,
        "step": step,
        "raw_min": float(raw_array.min()),
        "raw_max": float(raw_array.max()),
        "fake_quant_min": float(fake_quant_f.min()),
        "fake_quant_max": float(fake_quant_f.max()),
        "int8_min": int(int8_codes.min()),
        "int8_max": int(int8_codes.max()),
        "n_saturated": n_clipped,
    }
    if n_clipped > 0:
        print(
            f"  [WARNING] {tensor_name}: {n_clipped} value(s) fell outside "
            f"int8 range before clipping — check bit width / quantizer alpha.",
            file=sys.stderr,
        )

    return int8_codes, fake_quant_f


def split_gates(packed_array, units):
    z = packed_array[..., 0 * units : 1 * units]
    r = packed_array[..., 1 * units : 2 * units]
    h = packed_array[..., 2 * units : 3 * units]
    return z, r, h


def emit_gru_layer_header(f, layer_tag, units, input_dim,
                           kernel_i8, recurrent_i8, bias_i8, step):
    kz, kr, kh = split_gates(kernel_i8, units)
    uz, ur, uh = split_gates(recurrent_i8, units)
    bz, br, bh = split_gates(bias_i8, units)

    def emit_array_2d(name, arr):
        rows, cols = arr.shape
        f.write(f"static const int8_t {name}[{rows}][{cols}] = {{\n")
        for r_idx in range(rows):
            row_vals = ", ".join(str(int(v)) for v in arr[r_idx])
            f.write(f"  {{ {row_vals} }},\n")
        f.write("};\n\n")

    def emit_array_1d(name, arr):
        vals = ", ".join(str(int(v)) for v in arr)
        f.write(f"static const int8_t {name}[{arr.shape[0]}] = {{ {vals} }};\n\n")

    f.write(f"// ---- {layer_tag} : input_dim={input_dim} units={units} ----\n")
    f.write(f"// quantizer step = 2^-{int(-np.log2(step))} = {step:.10f}\n")
    f.write(f"#define {layer_tag.upper()}_UNITS {units}\n")
    f.write(f"#define {layer_tag.upper()}_INPUT_DIM {input_dim}\n\n")

    emit_array_2d(f"{layer_tag}_kernel_z", kz)
    emit_array_2d(f"{layer_tag}_kernel_r", kr)
    emit_array_2d(f"{layer_tag}_kernel_h", kh)
    emit_array_2d(f"{layer_tag}_recurrent_kernel_z", uz)
    emit_array_2d(f"{layer_tag}_recurrent_kernel_r", ur)
    emit_array_2d(f"{layer_tag}_recurrent_kernel_h", uh)
    emit_array_1d(f"{layer_tag}_bias_z", bz)
    emit_array_1d(f"{layer_tag}_bias_r", br)
    emit_array_1d(f"{layer_tag}_bias_h", bh)


def emit_dense_layer_header(f, layer_tag, in_dim, out_dim,
                             kernel_i8, bias_i8, step):
    f.write(f"// ---- {layer_tag} : in_dim={in_dim} out_dim={out_dim} ----\n")
    f.write(f"// quantizer step = 2^-{int(-np.log2(step))} = {step:.10f}\n")
    f.write(f"#define {layer_tag.upper()}_IN_DIM {in_dim}\n")
    f.write(f"#define {layer_tag.upper()}_OUT_DIM {out_dim}\n\n")

    f.write(f"static const int8_t {layer_tag}_kernel[{in_dim}][{out_dim}] = {{\n")
    for r_idx in range(in_dim):
        row_vals = ", ".join(str(int(v)) for v in kernel_i8[r_idx])
        f.write(f"  {{ {row_vals} }},\n")
    f.write("};\n\n")

    bias_vals = ", ".join(str(int(v)) for v in bias_i8)
    f.write(f"static const int8_t {layer_tag}_bias[{out_dim}] = {{ {bias_vals} }};\n\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract QKeras Seq2SeqLite student weights into int8 "
                     "arrays for Vitis HLS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights-h5", type=str, required=True,
                   help="Path to the trained student .weights.h5 checkpoint.")
    p.add_argument("--student-args", type=str, required=True,
                   help="Path to student_args.json saved alongside the "
                        "checkpoint by train_student_vanilla_kd.py. Must "
                        "contain keys: student_units, bits_kernel, "
                        "bits_recurrent, bits_bias, bits_activation, "
                        "bits_state.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for all extracted files.")
    p.add_argument("--seq-len", type=int, default=135,
                   help="Sequence length used during training.")
    p.add_argument("--n-out", type=int, default=3,
                   help="Number of decoder output channels.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[ARGS] Loading student_args.json: {args.student_args}")
    with open(args.student_args, "r") as f:
        student_args = json.load(f)

    required_keys = [
        "student_units", "bits_kernel", "bits_recurrent",
        "bits_bias", "bits_activation", "bits_state",
    ]
    missing = [k for k in required_keys if k not in student_args]
    if missing:
        raise KeyError(
            f"[ARGS] student_args.json is missing required key(s): {missing}. "
            f"Available keys: {sorted(student_args.keys())}"
        )

    student_units = int(student_args["student_units"])
    bits_kernel = int(student_args["bits_kernel"])
    bits_recurrent = int(student_args["bits_recurrent"])
    bits_bias = int(student_args["bits_bias"])
    bits_activation = int(student_args["bits_activation"])
    bits_state = int(student_args["bits_state"])

    print(f"[ARGS] student_units={student_units}  bits_kernel={bits_kernel}  "
          f"bits_recurrent={bits_recurrent}  bits_bias={bits_bias}  "
          f"bits_activation={bits_activation}  bits_state={bits_state}")

    print(f"[LOAD] Building student model...")
    model = build_student_for_extraction(
        seq_len=args.seq_len,
        n_out=args.n_out,
        student_units=student_units,
        bits_kernel=bits_kernel,
        bits_recurrent=bits_recurrent,
        bits_bias=bits_bias,
        bits_activation=bits_activation,
        bits_state=bits_state,
    )

    print(f"[LOAD] Loading weights from: {args.weights_h5}")
    model.load_weights(args.weights_h5)
    print("[LOAD] Weights loaded OK.")
    model.summary()

    qwk = quantized_bits(bits_kernel, 0, 1, alpha=1.0)
    qwr = quantized_bits(bits_recurrent, 0, 1, alpha=1.0)
    qwb = quantized_bits(bits_bias, 0, 1, alpha=1.0)
    qd = quantized_bits(bits_kernel, 0)

    step_kernel = 2.0 ** (-(bits_kernel - 1))
    step_dense = 2.0 ** (-(bits_kernel - 1))

    npz_fp32 = {}
    npz_i8 = {}
    report = {}

    def process_gru_layer(layer_name, layer_tag, units, input_dim):
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()
        if len(weights) != 3:
            raise RuntimeError(
                f"[EXTRACT] Layer '{layer_name}' returned {len(weights)} "
                f"weight arrays, expected exactly 3 (kernel, recurrent_kernel, "
                f"bias) for reset_after=False. Check QKeras version / "
                f"architecture mismatch."
            )
        raw_kernel, raw_recurrent, raw_bias = weights

        expected_kernel_shape = (input_dim, 3 * units)
        expected_recurrent_shape = (units, 3 * units)
        expected_bias_shape = (3 * units,)
        if raw_kernel.shape != expected_kernel_shape:
            raise RuntimeError(
                f"[EXTRACT] '{layer_name}' kernel shape {raw_kernel.shape} "
                f"!= expected {expected_kernel_shape}."
            )
        if raw_recurrent.shape != expected_recurrent_shape:
            raise RuntimeError(
                f"[EXTRACT] '{layer_name}' recurrent_kernel shape "
                f"{raw_recurrent.shape} != expected {expected_recurrent_shape}."
            )
        if raw_bias.shape != expected_bias_shape:
            raise RuntimeError(
                f"[EXTRACT] '{layer_name}' bias shape {raw_bias.shape} "
                f"!= expected {expected_bias_shape}."
            )

        print(f"[EXTRACT] {layer_name}: kernel={raw_kernel.shape}  "
              f"recurrent_kernel={raw_recurrent.shape}  bias={raw_bias.shape}")

        kernel_i8, kernel_fq = quantize_and_pack_int8(
            raw_kernel, qwk, bits_kernel, f"{layer_tag}_kernel", report
        )
        recurrent_i8, recurrent_fq = quantize_and_pack_int8(
            raw_recurrent, qwr, bits_recurrent, f"{layer_tag}_recurrent_kernel", report
        )
        bias_i8, bias_fq = quantize_and_pack_int8(
            raw_bias, qwb, bits_bias, f"{layer_tag}_bias", report
        )

        npz_fp32[f"{layer_tag}_kernel"] = raw_kernel
        npz_fp32[f"{layer_tag}_recurrent_kernel"] = raw_recurrent
        npz_fp32[f"{layer_tag}_bias"] = raw_bias
        npz_fp32[f"{layer_tag}_kernel_fq"] = kernel_fq
        npz_fp32[f"{layer_tag}_recurrent_kernel_fq"] = recurrent_fq
        npz_fp32[f"{layer_tag}_bias_fq"] = bias_fq

        npz_i8[f"{layer_tag}_kernel_i8"] = kernel_i8
        npz_i8[f"{layer_tag}_recurrent_kernel_i8"] = recurrent_i8
        npz_i8[f"{layer_tag}_bias_i8"] = bias_i8

        return kernel_i8, recurrent_i8, bias_i8

    sencgru_k_i8, sencgru_u_i8, sencgru_b_i8 = process_gru_layer(
        "sencgru", "sencgru", student_units, input_dim=1
    )
    sdecgru_k_i8, sdecgru_u_i8, sdecgru_b_i8 = process_gru_layer(
        "sdecgru", "sdecgru", student_units, input_dim=1
    )

    dense_layer = model.get_layer("sdec_dense")
    dense_weights = dense_layer.get_weights()
    if len(dense_weights) != 2:
        raise RuntimeError(
            f"[EXTRACT] 'sdec_dense' returned {len(dense_weights)} weight "
            f"arrays, expected exactly 2 (kernel, bias)."
        )
    raw_dense_kernel, raw_dense_bias = dense_weights
    expected_dense_kernel_shape = (student_units, args.n_out)
    expected_dense_bias_shape = (args.n_out,)
    if raw_dense_kernel.shape != expected_dense_kernel_shape:
        raise RuntimeError(
            f"[EXTRACT] 'sdec_dense' kernel shape {raw_dense_kernel.shape} "
            f"!= expected {expected_dense_kernel_shape}."
        )
    if raw_dense_bias.shape != expected_dense_bias_shape:
        raise RuntimeError(
            f"[EXTRACT] 'sdec_dense' bias shape {raw_dense_bias.shape} "
            f"!= expected {expected_dense_bias_shape}."
        )

    print(f"[EXTRACT] sdec_dense: kernel={raw_dense_kernel.shape}  "
          f"bias={raw_dense_bias.shape}")

    dense_kernel_i8, dense_kernel_fq = quantize_and_pack_int8(
        raw_dense_kernel, qd, bits_kernel, "sdec_dense_kernel", report
    )
    dense_bias_i8, dense_bias_fq = quantize_and_pack_int8(
        raw_dense_bias, qd, bits_kernel, "sdec_dense_bias", report
    )

    npz_fp32["sdec_dense_kernel"] = raw_dense_kernel
    npz_fp32["sdec_dense_bias"] = raw_dense_bias
    npz_fp32["sdec_dense_kernel_fq"] = dense_kernel_fq
    npz_fp32["sdec_dense_bias_fq"] = dense_bias_fq
    npz_i8["sdec_dense_kernel_i8"] = dense_kernel_i8
    npz_i8["sdec_dense_bias_i8"] = dense_bias_i8

    fp32_path = os.path.join(args.out_dir, "student_weights_fp32.npz")
    i8_path = os.path.join(args.out_dir, "student_weights_int8.npz")
    header_path = os.path.join(args.out_dir, "student_weights_int8.h")
    report_path = os.path.join(args.out_dir, "extraction_report.json")

    np.savez(fp32_path, **npz_fp32)
    print(f"[SAVE] Wrote {fp32_path}")

    np.savez(i8_path, **npz_i8)
    print(f"[SAVE] Wrote {i8_path}")

    with open(header_path, "w") as f:
        f.write("// Auto-generated by extract_student_weights.py\n")
        f.write("// DO NOT EDIT BY HAND.\n")
        f.write("#ifndef STUDENT_WEIGHTS_INT8_H\n")
        f.write("#define STUDENT_WEIGHTS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")

        emit_gru_layer_header(
            f, "sencgru", student_units, input_dim=1,
            kernel_i8=sencgru_k_i8, recurrent_i8=sencgru_u_i8,
            bias_i8=sencgru_b_i8, step=step_kernel,
        )
        emit_gru_layer_header(
            f, "sdecgru", student_units, input_dim=1,
            kernel_i8=sdecgru_k_i8, recurrent_i8=sdecgru_u_i8,
            bias_i8=sdecgru_b_i8, step=step_kernel,
        )
        emit_dense_layer_header(
            f, "sdec_dense", student_units, args.n_out,
            kernel_i8=dense_kernel_i8, bias_i8=dense_bias_i8, step=step_dense,
        )

        f.write("#endif // STUDENT_WEIGHTS_INT8_H\n")
    print(f"[SAVE] Wrote {header_path}")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE] Wrote {report_path}")

    total_saturated = sum(v["n_saturated"] for v in report.values())
    if total_saturated > 0:
        print(f"[SUMMARY] WARNING: {total_saturated} total saturated value(s) "
              f"across all tensors — review extraction_report.json.")
    else:
        print("[SUMMARY] No saturation detected. Extraction is clean.")

    print("[DONE] Extraction complete.")


if __name__ == "__main__":
    main()