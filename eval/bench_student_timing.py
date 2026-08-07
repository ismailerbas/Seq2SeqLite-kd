#!/usr/bin/env python3
"""
eval/bench_student_timing.py — CPU vs GPU inference timing benchmark for
vanilla KD student ablation checkpoints.

Scans --results-dir for all subdirectories whose name starts with "vanilla_kd"
and that contain student_best.weights.h5 + student_args.json. For each run,
loads student_args.json to reconstruct the EXACT architecture used by
eval/eval_student_vanilla_sdf.py, and times forward-pass inference latency
and throughput across a list of batch sizes on either CPU or GPU. Results
are saved to timing_benchmark.json next to the weights file, merged across
separate --device cpu / --device gpu runs so a single JSON ends up holding
both device results.

--device MUST be passed on the command line and controls CUDA_VISIBLE_DEVICES
BEFORE TensorFlow is imported, so run this script once with --device cpu and
once with --device gpu (two separate Slurm submissions) to get both timings.

Usage:
python eval/bench_student_timing.py \
    --data-dir /scratch/nmi \
    --results-dir /scratch/nmi/results \
    --seq-len 135 \
    --n-out 3 \
    --batch-sizes 1,8,32,128,1024,8192 \
    --n-warmup 10 \
    --n-repeat 50 \
    --device gpu \
    --overwrite

--overwrite : re-benchmark and overwrite the entry for --device even if
              timing_benchmark.json already has an entry for that device.
"""

import os
import sys


def _get_cli_device_arg():
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--device" and i + 1 < len(argv):
            return argv[i + 1].strip().lower()
        if tok.startswith("--device="):
            return tok.split("=", 1)[1].strip().lower()
    return None


_CLI_DEVICE = _get_cli_device_arg()
if _CLI_DEVICE not in ("cpu", "gpu"):
    print(
        f"[INIT] --device must be 'cpu' or 'gpu', got: {_CLI_DEVICE!r}. "
        f"Pass --device cpu or --device gpu explicitly.",
        flush=True,
    )
    sys.exit(1)

if _CLI_DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print(
        "[INIT] --device=cpu — forcing CUDA_VISIBLE_DEVICES=-1 (GPU hidden from TensorFlow).",
        flush=True,
    )
else:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        print(
            "[INIT] --device=gpu — CUDA_VISIBLE_DEVICES not set — defaulting to 0,1,2,3,4,5,6,7",
            flush=True,
        )
    else:
        print(
            f"[INIT] --device=gpu — CUDA_VISIBLE_DEVICES already set: "
            f"{os.environ['CUDA_VISIBLE_DEVICES']}",
            flush=True,
        )

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import argparse
import glob
import json
import time

import numpy as np

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
        description="Benchmark vanilla KD student model CPU/GPU inference timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing tpsf_seq, res, labels, testidx .npy files.")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Root directory to scan for vanilla_kd* subdirectories.")
    p.add_argument("--seq-len", type=int, default=135)
    p.add_argument("--n-out", type=int, default=3)
    p.add_argument("--batch-sizes", type=str, default="1,8,32,128,1024,8192",
                   help="Comma-separated list of batch sizes to benchmark.")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Number of warmup iterations before timing (per batch size).")
    p.add_argument("--n-repeat", type=int, default=50,
                   help="Number of timed iterations to average (per batch size).")
    p.add_argument("--device", type=str, choices=["cpu", "gpu"], required=True,
                   help="Device to benchmark. Must match the --device flag used to set "
                        "CUDA_VISIBLE_DEVICES at process start.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Re-benchmark and overwrite the entry for --device if it already "
                        "exists in timing_benchmark.json.")
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

    file_input = find_one([f"tpsf_seq_L{seq_len}_*.npy"], "encoder input (tpsf_seq)")
    file_res = find_one([f"res_L{seq_len}_*.npy"], "decoder target (res)")
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
# Student model — EXACT replica of train_student_vanilla_kd.py /
# eval_student_vanilla_sdf.py
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
# Timing benchmark core
# ==============================================================================

def benchmark_model_inference(
    model,
    enc_arr,
    seq_len,
    n_out,
    batch_size,
    n_warmup,
    n_repeat,
    device_str,
    input_mode,
    pf,
):
    """
    Time forward-pass inference latency and throughput for `model` on the
    device given by `device_str` (e.g. "/CPU:0" or "/GPU:0").

    input_mode : "list" for student_model([enc_b, dec_b])

    Uses a single fixed batch of real data (the first `batch_size` rows of
    enc_arr) for every warmup and timed iteration so that timing measures
    pure model forward-pass latency, not data-loading overhead.
    """
    n_available = len(enc_arr)
    actual_batch = min(batch_size, n_available)
    if actual_batch < batch_size:
        pf(
            f"  WARNING: requested batch_size={batch_size} exceeds available "
            f"samples={n_available}. Using batch_size={actual_batch}."
        )

    enc_batch_np = np.ascontiguousarray(enc_arr[:actual_batch], dtype=np.float32)
    dec_batch_np = np.zeros((actual_batch, seq_len, 1), dtype=np.float32)

    with tf.device(device_str):
        enc_b = tf.constant(enc_batch_np, dtype=tf.float32)
        dec_b = tf.constant(dec_batch_np, dtype=tf.float32)

        def _forward():
            if input_mode == "dict":
                out = model({"encinput": enc_b, "decinput": dec_b}, training=False)
            else:
                out = model([enc_b, dec_b], training=False)
            return out

        pf(f"  Warming up on {device_str} for {n_warmup} iteration(s)...")
        for _ in range(n_warmup):
            out = _forward()
            _ = out.numpy()

        pf(f"  Timing on {device_str} for {n_repeat} iteration(s), batch_size={actual_batch}...")
        elapsed_list = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            out = _forward()
            _ = out.numpy()
            t1 = time.perf_counter()
            elapsed_list.append(t1 - t0)

    elapsed_arr = np.array(elapsed_list, dtype=np.float64)
    mean_s = float(np.mean(elapsed_arr))
    std_s = float(np.std(elapsed_arr))
    median_s = float(np.median(elapsed_arr))
    p95_s = float(np.percentile(elapsed_arr, 95))
    min_s = float(np.min(elapsed_arr))
    max_s = float(np.max(elapsed_arr))
    throughput = float(actual_batch / mean_s) if mean_s > 0 else 0.0

    result = {
        "device_str": device_str,
        "batch_size": int(actual_batch),
        "n_warmup": int(n_warmup),
        "n_repeat": int(n_repeat),
        "mean_ms": mean_s * 1000.0,
        "std_ms": std_s * 1000.0,
        "median_ms": median_s * 1000.0,
        "p95_ms": p95_s * 1000.0,
        "min_ms": min_s * 1000.0,
        "max_ms": max_s * 1000.0,
        "throughput_samples_per_sec": throughput,
    }
    pf(
        f"  {device_str}: mean={result['mean_ms']:.3f}ms "
        f"std={result['std_ms']:.3f}ms median={result['median_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms throughput={throughput:.1f} samples/s"
    )
    return result

# ==============================================================================
# Benchmark one vanilla_kd student run directory
# ==============================================================================

def benchmark_student_run(
    run_dir,
    normalized_input,
    seq_len,
    n_out,
    batch_sizes,
    n_warmup,
    n_repeat,
    device,
    device_str,
    overwrite,
    pf,
):
    timing_path = os.path.join(run_dir, "timing_benchmark.json")

    if os.path.exists(timing_path):
        with open(timing_path, "r") as f:
            timing_data = json.load(f)
    else:
        timing_data = {}

    if "results" not in timing_data:
        timing_data["results"] = {}

    if device in timing_data["results"] and not overwrite:
        pf(f"  SKIP (device='{device}' already benchmarked): {timing_path}")
        return

    ckpt_path = os.path.join(run_dir, "student_best.weights.h5")
    args_path = os.path.join(run_dir, "student_args.json")
    pf(f"  Checkpoint    : {ckpt_path}")
    pf(f"  student_args  : {args_path}")
    sys.stdout.flush()

    with open(args_path, "r") as f:
        run_args = json.load(f)

    student_units = int(run_args["student_units"])
    bits_kernel = int(run_args["bits_kernel"])
    bits_recurrent = int(run_args["bits_recurrent"])
    bits_bias = int(run_args["bits_bias"])
    bits_activation = int(run_args["bits_activation"])
    bits_state = int(run_args["bits_state"])

    pf(
        f"  Architecture: QGRU-{student_units} "
        f"k={bits_kernel} r={bits_recurrent} b={bits_bias} "
        f"a={bits_activation} s={bits_state}"
    )
    sys.stdout.flush()

    tf.keras.backend.clear_session()
    student_model = build_student(
        seq_len=seq_len,
        n_out=n_out,
        student_units=student_units,
        bits_kernel=bits_kernel,
        bits_recurrent=bits_recurrent,
        bits_bias=bits_bias,
        bits_activation=bits_activation,
        bits_state=bits_state,
    )
    student_model.load_weights(ckpt_path)
    student_model.trainable = False
    pf(f"  Weights loaded OK.")
    sys.stdout.flush()

    device_results = []
    for batch_size in batch_sizes:
        pf(f"  --- batch_size={batch_size} ---")
        result = benchmark_model_inference(
            model=student_model,
            enc_arr=normalized_input,
            seq_len=seq_len,
            n_out=n_out,
            batch_size=batch_size,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            device_str=device_str,
            input_mode="list",
            pf=pf,
        )
        device_results.append(result)
        sys.stdout.flush()

    timing_data["run_dir"] = run_dir
    timing_data["student_units"] = student_units
    timing_data["bits_kernel"] = bits_kernel
    timing_data["bits_recurrent"] = bits_recurrent
    timing_data["bits_bias"] = bits_bias
    timing_data["bits_activation"] = bits_activation
    timing_data["bits_state"] = bits_state
    timing_data["results"][device] = device_results

    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    pf(f"  timing_benchmark.json saved: {timing_path}")
    sys.stdout.flush()

# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    pf = lambda s: print(s, flush=True)

    setup_gpu()

    pf("=" * 70)
    pf("bench_student_timing.py — CPU/GPU inference timing for vanilla KD student models")
    pf(f"  data-dir     : {args.data_dir}")
    pf(f"  results-dir  : {args.results_dir}")
    pf(f"  device       : {args.device}")
    pf(f"  batch-sizes  : {args.batch_sizes}")
    pf(f"  n-warmup     : {args.n_warmup}")
    pf(f"  n-repeat     : {args.n_repeat}")
    pf(f"  overwrite    : {args.overwrite}")
    pf("=" * 70)
    sys.stdout.flush()

    if args.device == "gpu":
        physical_gpus = tf.config.list_physical_devices("GPU")
        if not physical_gpus:
            pf("FATAL: --device=gpu was requested but no physical GPU is visible to TensorFlow.")
            pf("       Check CUDA_VISIBLE_DEVICES and that this job was submitted with --gres=gpu:1.")
            sys.exit(1)
        device_str = "/GPU:0"
    else:
        device_str = "/CPU:0"

    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    if not batch_sizes:
        pf("FATAL: --batch-sizes produced an empty list.")
        sys.exit(1)

    pf("Loading data files (mmap)...")
    file_input, file_res, file_labels, file_test = find_data_files(
        args.data_dir, args.seq_len
    )
    pf(f"  encoder input : {file_input}")
    sys.stdout.flush()

    normalized_input = np.load(file_input, mmap_mode="r")
    pf(f"  N={normalized_input.shape[0]:,} seq_len={args.seq_len} n_out={args.n_out}")
    sys.stdout.flush()

    pf(f"Discovering vanilla_kd student run directories under: {args.results_dir}")
    run_dirs = find_student_run_dirs(args.results_dir)
    if not run_dirs:
        pf("ERROR: No vanilla_kd* directories with student_best.weights.h5 + student_args.json found.")
        sys.exit(1)
    pf(f"Found {len(run_dirs)} vanilla_kd student run(s):")
    for d in run_dirs:
        pf(f"  {d}")
    sys.stdout.flush()

    t_total = time.time()
    for idx, run_dir in enumerate(run_dirs, 1):
        pf("")
        pf("=" * 70)
        pf(f"[{idx}/{len(run_dirs)}] {run_dir}")
        pf("=" * 70)
        sys.stdout.flush()
        try:
            benchmark_student_run(
                run_dir=run_dir,
                normalized_input=normalized_input,
                seq_len=args.seq_len,
                n_out=args.n_out,
                batch_sizes=batch_sizes,
                n_warmup=args.n_warmup,
                n_repeat=args.n_repeat,
                device=args.device,
                device_str=device_str,
                overwrite=args.overwrite,
                pf=pf,
            )
        except Exception as exc:
            pf(f"  ERROR in {run_dir}: {exc}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    pf("")
    pf("=" * 70)
    pf(
        f"All vanilla_kd student timing benchmarks processed. "
        f"Total elapsed: {(time.time() - t_total) / 60:.1f} min"
    )
    pf("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()