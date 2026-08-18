#!/usr/bin/env python3
"""
eval/analyze_memoq_deadzone.py — Post-hoc hidden-state resolution analysis for
MemoQ P2D/P2E/P2F checkpoints from the paper run.

This script is deliberately inference-only. It does not train, fine-tune, or
modify any checkpoint. It reconstructs the Phase-2 split-gate model using the
current repository's MemoQGRUCell/MemoQRNNUnroll implementation, then forces
the hard inference semantics used by the pre-2026-07-04 MemoQ stage design:

  P2D: 4-bit gate kernels/recurrent kernels/biases; float activation; float state
  P2E: P2D + 4-bit candidate activation; float state
  P2F: P2E + 4-bit recurrent state, hard state_blend_beta=1

The July-4 commit 9ec85899b478b3269767af7a7d3ad3f201ee1aed changed P2E/P2F
training to activation soft blending and joint activation/state annealing. The
paper checkpoints analyzed here predate that change. Training-time state noise
or dither is never enabled here because the original per-phase evaluation uses
training=False. The analysis therefore measures the deterministic deployment
trajectory represented by each saved checkpoint.

For every decoder recurrence transition, including the transition from the
encoder final state into decoder step 0, the script computes:

  normalized_update = abs(h_t - h_{t-1}) / Delta_s
  dead_zone          = normalized_update < 0.5
  same_bin           = Q_s(h_t) == Q_s(h_{t-1})
  high_z             = sigmoid(z_logit_t) > high_z_threshold

For the paper's 4-bit state quantizer:
  Delta_s = 2^(-(bits_state - 1)) = 0.125
  half LSB = Delta_s / 2 = 0.0625

P2D/P2E same_bin is a projected collision rate on the target 4-bit state grid.
P2F same_bin is the actual target-grid collision rate of the state values used
by the next recurrent step.

The script also recomputes test sequence MAE for every phase and compares it to
test_metrics_<PHASE>.json. It exits non-zero if the reconstructed inference
path does not reproduce the saved MAE within --mae-tolerance. This guard is
intentional: dead-zone numbers are not written as valid paper results unless
the checkpoint reconstruction is verified against the historical evaluation.

Outputs under --out-dir:
  deadzone_summary.json
  deadzone_summary.csv
  deadzone_per_unit.csv
  deadzone_per_sequence.npz
  deadzone_update_histogram.csv
  deadzone_update_cdf_full.png
  deadzone_update_cdf_zoom.png
  deadzone_dead_fraction_per_unit.png
  deadzone_same_bin_fraction_per_unit.png
  deadzone_summary_percent.png
  deadzone_manifest.json
  deadzone_complete.flag
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Keep TensorFlow runtime behavior aligned with the repository's SLURM jobs.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("[INIT] CUDA_VISIBLE_DEVICES not set -- defaulting to 0", flush=True)
else:
    print(
        f"[INIT] CUDA_VISIBLE_DEVICES already set: "
        f"{os.environ['CUDA_VISIBLE_DEVICES']}",
        flush=True,
    )

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, quantized_bits, quantized_tanh


THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
REPO_ROOT = EVAL_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_student_memoq import (  # noqa: E402
    MemoQGRUCell,
    MemoQRNNUnroll,
    find_data_files,
)


EXPECTED_RUN_BASENAME = (
    "memoq_b4k4r4a4s4_gru32_dense3_effbs1024_microbs1024_lr1e-04_"
    "p1-40_2a30_2b30_2c10_p3-170p2fcalismadipapericinkullan"
)

EXPECTED_CONFIG = {
    "seq_len": 135,
    "n_out": 3,
    "student_units": 32,
    "bits_kernel": 4,
    "bits_bias": 4,
    "bits_recurrent": 4,
    "bits_activation": 4,
    "bits_state": 4,
    "batch_size": 1024,
    "split_seed": 42,
    "memoq_stage2d_epochs": 10,
    "memoq_stage2e_epochs": 10,
    "memoq_stage2f_epochs": 30,
    "memoq_state_anneal_epochs": 15,
}

EXPECTED_PHASE_METRICS = {
    "P2D": {
        "mae_seq": 0.020808501169085503,
        "rmse_tau1": 0.28251229628385693,
        "rmse_tau2": 0.3149366733049176,
        "r_tau1": 0.8246662569167262,
        "r_tau2": 0.863912189690561,
    },
    "P2E": {
        "mae_seq": 0.032993823289871216,
        "rmse_tau1": 0.48153971769683857,
        "rmse_tau2": 0.2891835239302995,
        "r_tau1": 0.8045819780795951,
        "r_tau2": 0.8690163318996221,
    },
    "P2F": {
        "mae_seq": 0.28905847668647766,
        "rmse_tau1": 4.600428895132946,
        "rmse_tau2": 4.725837115807589,
        "r_tau1": 0.5462473067299856,
        "r_tau2": 0.5257674412871891,
    },
}

CHECKPOINT_NAMES = {
    "P2D": "stage2d_best.weights.h5",
    "P2E": "stage2e_best.weights.h5",
    "P2F": "stage2f_best.weights.h5",
}

METRIC_NAMES = {
    "P2D": "test_metrics_P2D.json",
    "P2E": "test_metrics_P2E.json",
    "P2F": "test_metrics_P2F.json",
}

PRE_JULY4_STAGE_REFERENCE_COMMIT = (
    "446b819654404d4e7832cd02c947e96f0083ff19"
)

JULY4_BEHAVIOR_CHANGE_COMMIT = (
    "9ec85899b478b3269767af7a7d3ad3f201ee1aed"
)

EXPECTED_CURRENT_TRAIN_MEMOQ_GIT_BLOB = (
    "71eee1eb022919558068891258250ec7658c6921"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc MemoQ hidden-state dead-zone and target-grid collision "
            "analysis for the identified paper run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help=(
            "Exact MemoQ paper run directory containing student_args.json "
            "and phase checkpoints."
        ),
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help=(
            "Dataset directory containing tpsf/res/labels and "
            "train/val/test index files."
        ),
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Directory where dead-zone analysis outputs will be written.",
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("P2D", "P2E", "P2F"),
        default=["P2D", "P2E", "P2F"],
        help="Phase checkpoints to analyze in this order.",
    )

    parser.add_argument(
        "--infer-batch",
        type=int,
        default=4096,
        help="Inference batch size for hidden-state extraction.",
    )

    parser.add_argument(
        "--high-z-threshold",
        type=float,
        default=0.90,
        help=(
            "Retention-gate threshold used for conditional "
            "dead-zone occupancy."
        ),
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
        help=(
            "Nonparametric sequence-level bootstrap replicates "
            "for 95%% confidence intervals."
        ),
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed for sequence-level bootstrap resampling.",
    )

    parser.add_argument(
        "--bootstrap-batch-reps",
        type=int,
        default=32,
        help=(
            "Number of bootstrap replicates vectorized together "
            "to bound RAM use."
        ),
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=4096,
        help=(
            "Number of bins for the streaming normalized-update "
            "histogram/CDF."
        ),
    )

    parser.add_argument(
        "--mae-tolerance",
        type=float,
        default=5e-5,
        help=(
            "Maximum allowed absolute difference between recomputed "
            "and saved sequence MAE."
        ),
    )

    return parser.parse_args()


def pf(message: str = "") -> None:
    print(message, flush=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    os.replace(tmp_path, path)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def git_command(args: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return completed.stdout.strip()

    except Exception as exc:
        return f"unavailable: {exc}"


def validate_repository_source() -> None:
    train_path = REPO_ROOT / "train_student_memoq.py"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Missing repository source file: {train_path}"
        )

    actual_blob = git_command(
        ["hash-object", str(train_path)]
    )

    if actual_blob != EXPECTED_CURRENT_TRAIN_MEMOQ_GIT_BLOB:
        raise RuntimeError(
            "train_student_memoq.py does not match the GitHub main "
            "version validated for this analysis.\n"
            f"Expected git blob: "
            f"{EXPECTED_CURRENT_TRAIN_MEMOQ_GIT_BLOB}\n"
            f"Actual git blob:   {actual_blob}"
        )

    pf(
        "[SOURCE] train_student_memoq.py git blob verified: "
        f"{actual_blob}"
    )


def configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(42)

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError(
            "TensorFlow found no GPU. This analysis SLURM job "
            "is expected to run on one GPU."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )

        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to set TensorFlow memory growth for "
                f"{gpu}: {exc}"
            ) from exc

    pf(
        f"[GPU] TensorFlow physical GPUs: {len(gpus)}"
    )

    for index, gpu in enumerate(gpus):
        pf(
            f"[GPU]   {index}: {gpu}"
        )


def validate_cli(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    if args.infer_batch <= 0:
        raise ValueError(
            "--infer-batch must be > 0"
        )

    if not (
        0.0
        < args.high_z_threshold
        < 1.0
    ):
        raise ValueError(
            "--high-z-threshold must be strictly between 0 and 1"
        )

    if args.bootstrap_reps <= 0:
        raise ValueError(
            "--bootstrap-reps must be > 0"
        )

    if args.bootstrap_batch_reps <= 0:
        raise ValueError(
            "--bootstrap-batch-reps must be > 0"
        )

    if args.hist_bins < 128:
        raise ValueError(
            "--hist-bins must be >= 128"
        )

    if args.mae_tolerance <= 0.0:
        raise ValueError(
            "--mae-tolerance must be > 0"
        )

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    if run_dir.name != EXPECTED_RUN_BASENAME:
        raise RuntimeError(
            "Refusing to analyze a different run directory.\n"
            f"Expected basename: {EXPECTED_RUN_BASENAME}\n"
            f"Received basename: {run_dir.name}"
        )


def validate_paper_run(
    run_dir: Path,
    cfg: Dict,
    phases: List[str],
) -> Dict[str, Dict]:
    missing_cfg = [
        key
        for key in EXPECTED_CONFIG
        if key not in cfg
    ]

    if missing_cfg:
        raise KeyError(
            "student_args.json is missing required "
            f"paper-run keys: {missing_cfg}"
        )

    mismatches = []

    for key, expected in EXPECTED_CONFIG.items():
        actual = cfg[key]

        if isinstance(expected, float):
            ok = math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

        else:
            ok = int(actual) == int(expected)

        if not ok:
            mismatches.append(
                (
                    key,
                    actual,
                    expected,
                )
            )

    q_alpha = float(
        cfg.get(
            "q_alpha",
            cfg.get(
                "quantizer_alpha",
                "nan",
            ),
        )
    )

    if not math.isclose(
        q_alpha,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        mismatches.append(
            (
                "q_alpha",
                q_alpha,
                1.0,
            )
        )

    if mismatches:
        formatted = "\n".join(
            (
                f"  {key}: actual={actual!r} "
                f"expected={expected!r}"
            )
            for key, actual, expected in mismatches
        )

        raise RuntimeError(
            "student_args.json does not match the identified "
            "MemoQ paper run:\n"
            + formatted
        )

    saved_metrics = {}

    for phase in phases:
        checkpoint_path = (
            run_dir
            / CHECKPOINT_NAMES[phase]
        )

        metric_path = (
            run_dir
            / METRIC_NAMES[phase]
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Missing required checkpoint: "
                f"{checkpoint_path}"
            )

        if not metric_path.is_file():
            raise FileNotFoundError(
                "Missing required metrics file: "
                f"{metric_path}"
            )

        metrics = load_json(
            metric_path
        )

        expected_metrics = (
            EXPECTED_PHASE_METRICS[phase]
        )

        metric_mismatches = []

        for key, expected_value in expected_metrics.items():
            if key not in metrics:
                metric_mismatches.append(
                    f"{key}: missing"
                )
                continue

            actual_value = float(
                metrics[key]
            )

            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=0.0,
                abs_tol=5e-12,
            ):
                metric_mismatches.append(
                    (
                        f"{key}: "
                        f"actual={actual_value:.15g} "
                        f"expected={expected_value:.15g}"
                    )
                )

        if metric_mismatches:
            raise RuntimeError(
                f"{metric_path} does not match "
                "the paper-run signature:\n  "
                + "\n  ".join(
                    metric_mismatches
                )
            )

        saved_metrics[phase] = metrics

    return saved_metrics


def build_phase2_diagnostic_model(
    seq_len: int,
    n_out: int,
    student_units: int,
    bits_kernel: int,
    q_alpha: float,
    input_dim: int = 1,
) -> Tuple[
    keras.Model,
    MemoQGRUCell,
    MemoQGRUCell,
]:
    """
    Exact Phase-2 topology from build_phase2_model(), with one additional
    read-only output: encoder final hidden state.

    No trainable layer or weight shape is added or removed, so the historical
    stage2*.weights.h5 files load directly.

    The extra output lets the analysis include the encoder-final ->
    decoder-step-0 recurrent transition instead of dropping that transition.
    """
    enc_cell = MemoQGRUCell(
        units=student_units,
        input_dim=input_dim,
        name="memoq_enc_cell",
    )

    dec_cell = MemoQGRUCell(
        units=student_units,
        input_dim=input_dim,
        name="memoq_dec_cell",
    )

    enc_unroll = MemoQRNNUnroll(
        enc_cell,
        name="sencgru_unroll",
    )

    dec_unroll = MemoQRNNUnroll(
        dec_cell,
        name="sdecgru_unroll",
    )

    enc_inputs = keras.layers.Input(
        shape=(
            seq_len,
            input_dim,
        ),
        name="senc_input",
    )

    dec_inputs = keras.layers.Input(
        shape=(
            seq_len,
            input_dim,
        ),
        name="sdec_input",
    )

    _, _, enc_final_h = enc_unroll(
        enc_inputs
    )

    enc_initial_z = keras.layers.Lambda(
        lambda x: tf.zeros_like(x),
        name="enc_initial_z_zero",
    )(
        enc_final_h
    )

    (
        dec_hidden_seq,
        dec_z_logit_seq,
        _,
    ) = dec_unroll(
        dec_inputs,
        initial_state=[
            enc_final_h,
            enc_initial_z,
        ],
    )

    if q_alpha == 1.0:
        dense_kernel_quantizer = quantized_bits(
            bits_kernel,
            0,
        )

        dense_bias_quantizer = quantized_bits(
            bits_kernel,
            0,
        )

    else:
        dense_kernel_quantizer = quantized_bits(
            bits_kernel,
            0,
            1,
            alpha=q_alpha,
        )

        dense_bias_quantizer = quantized_bits(
            bits_kernel,
            0,
            1,
            alpha=q_alpha,
        )

    seq_output = QDense(
        n_out,
        kernel_quantizer=dense_kernel_quantizer,
        bias_quantizer=dense_bias_quantizer,
        activation="linear",
        name="sdec_dense",
    )(
        dec_hidden_seq
    )

    model = keras.models.Model(
        inputs=[
            enc_inputs,
            dec_inputs,
        ],
        outputs=[
            seq_output,
            dec_hidden_seq,
            dec_z_logit_seq,
            enc_final_h,
        ],
        name=(
            "memoq_phase2_"
            "deadzone_diagnostic_model"
        ),
    )

    return (
        model,
        enc_cell,
        dec_cell,
    )


def configure_paper_phase_quantizers(
    cfg: Dict,
    enc_cell: MemoQGRUCell,
    dec_cell: MemoQGRUCell,
    phase: str,
) -> None:
    """
    Configure deterministic hard inference for the pre-July-4 stage semantics.

    This intentionally does NOT call current-main set_phase2_quantizers(),
    because current main resets activation/state blend betas for the newer
    July-4 joint-anneal training path.

    The paper P2D/P2E/P2F checkpoints were produced before that behavior
    change.

    Training-time Gaussian LSB noise/dither is zero during inference. All
    blend betas are therefore forced to 1.0 and all dither/noise controls
    that can affect current-main MemoQGRUCell are forced to zero.
    """
    bits_kernel = int(
        cfg["bits_kernel"]
    )

    bits_recurrent = int(
        cfg["bits_recurrent"]
    )

    bits_bias = int(
        cfg["bits_bias"]
    )

    bits_activation = int(
        cfg["bits_activation"]
    )

    bits_state = int(
        cfg["bits_state"]
    )

    q_alpha = float(
        cfg.get(
            "q_alpha",
            cfg["quantizer_alpha"],
        )
    )

    q4k = quantized_bits(
        bits_kernel,
        0,
        1,
        alpha=q_alpha,
    )

    q4r = quantized_bits(
        bits_recurrent,
        0,
        1,
        alpha=q_alpha,
    )

    q4b = quantized_bits(
        bits_bias,
        0,
        1,
        alpha=q_alpha,
    )

    q4a = quantized_tanh(
        bits=bits_activation,
        symmetric=True,
    )

    q4s = quantized_bits(
        bits_state,
        0,
        1,
        alpha=1.0,
    )

    if phase == "P2D":
        q_activation = None
        q_state = None

    elif phase == "P2E":
        q_activation = q4a
        q_state = None

    elif phase == "P2F":
        q_activation = q4a
        q_state = q4s

    else:
        raise ValueError(
            "Unsupported phase for "
            f"dead-zone analysis: {phase}"
        )

    for cell in (
        enc_cell,
        dec_cell,
    ):
        cell.quantizer_h = q4k
        cell.quantizer_r = q4k
        cell.quantizer_z = q4k

        cell.quantizer_recurrent_h = q4r
        cell.quantizer_recurrent_r = q4r
        cell.quantizer_recurrent_z = q4r

        cell.quantizer_bias = q4b
        cell.quantizer_activation = q_activation
        cell.quantizer_state = q_state

        # Historical P2F evaluation forces the state blend to hard beta=1.
        cell.state_blend_beta = 1.0

        # Current-main compatibility controls. These did not exist in the
        # original paper run and must be disabled to recover the historical
        # deterministic inference path.
        if hasattr(
            cell,
            "activation_blend_beta",
        ):
            cell.activation_blend_beta = 1.0

        if hasattr(
            cell,
            "act_dither_delta",
        ):
            cell.act_dither_delta = 0.0

        if hasattr(
            cell,
            "state_dither_delta",
        ):
            cell.state_dither_delta = 0.0

        if hasattr(
            cell,
            "state_lsb_noise_std",
        ):
            cell.state_lsb_noise_std = 0.0


def compute_training_normalization(
    raw_input: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[
    float,
    float,
]:
    pf(
        "[DATA] Recomputing encoder normalization "
        "exactly as train_student_memoq.py..."
    )

    train_raw = np.asarray(
        raw_input[train_idx]
    )

    inp_mean = float(
        np.mean(
            train_raw
        )
    )

    inp_std = float(
        np.std(
            train_raw
        )
    )

    del train_raw

    inp_std = max(
        inp_std,
        1e-6,
    )

    pf(
        f"[DATA] input mean={inp_mean:.12g}  "
        f"std={inp_std:.12g}"
    )

    return (
        inp_mean,
        inp_std,
    )


def validate_dataset_shapes(
    raw_input: np.ndarray,
    res_data: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seq_len: int,
    n_out: int,
) -> None:
    if raw_input.ndim != 3:
        raise RuntimeError(
            "Expected raw encoder input rank 3, "
            f"got shape {raw_input.shape}"
        )

    if raw_input.shape[1:] != (
        seq_len,
        1,
    ):
        raise RuntimeError(
            "Expected raw encoder input shape "
            f"(N,{seq_len},1), got {raw_input.shape}"
        )

    if res_data.ndim != 3:
        raise RuntimeError(
            "Expected sequence target rank 3, "
            f"got shape {res_data.shape}"
        )

    if res_data.shape[1:] != (
        seq_len,
        n_out,
    ):
        raise RuntimeError(
            "Expected target shape "
            f"(N,{seq_len},{n_out}), got {res_data.shape}"
        )

    if (
        raw_input.shape[0]
        != res_data.shape[0]
    ):
        raise RuntimeError(
            "Input/target sample count mismatch: "
            f"{raw_input.shape[0]} vs "
            f"{res_data.shape[0]}"
        )

    if (
        train_idx.ndim != 1
        or test_idx.ndim != 1
    ):
        raise RuntimeError(
            "Expected 1-D train/test indices, got "
            f"{train_idx.shape}, {test_idx.shape}"
        )

    if len(test_idx) != 160000:
        raise RuntimeError(
            "Paper run expects 160000 test samples, "
            f"found {len(test_idx)}"
        )

    n_total = raw_input.shape[0]

    for name, idx in (
        (
            "train",
            train_idx,
        ),
        (
            "test",
            test_idx,
        ),
    ):
        if len(idx) == 0:
            raise RuntimeError(
                f"{name} index is empty"
            )

        if (
            np.min(idx) < 0
            or np.max(idx) >= n_total
        ):
            raise RuntimeError(
                f"{name} index contains values "
                f"outside [0,{n_total - 1}]"
            )


def histogram_quantile(
    counts: np.ndarray,
    edges: np.ndarray,
    quantile: float,
) -> float:
    total = int(
        np.sum(
            counts
        )
    )

    if total <= 0:
        return float(
            "nan"
        )

    if not (
        0.0
        <= quantile
        <= 1.0
    ):
        raise ValueError(
            "quantile must be in [0,1]"
        )

    target = quantile * total

    cumulative = np.cumsum(
        counts,
        dtype=np.int64,
    )

    index = int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )

    index = min(
        max(
            index,
            0,
        ),
        len(counts) - 1,
    )

    before = (
        int(
            cumulative[index - 1]
        )
        if index > 0
        else 0
    )

    in_bin = int(
        counts[index]
    )

    left = float(
        edges[index]
    )

    right = float(
        edges[index + 1]
    )

    if in_bin <= 0:
        return 0.5 * (
            left
            + right
        )

    fraction = (
        target
        - before
    ) / in_bin

    fraction = min(
        max(
            float(fraction),
            0.0,
        ),
        1.0,
    )

    return (
        left
        + fraction
        * (
            right
            - left
        )
    )


def bootstrap_sequence_level(
    dead_counts: np.ndarray,
    same_bin_counts: np.ndarray,
    high_z_dead_counts: np.ndarray,
    high_z_counts: np.ndarray,
    transitions_per_sequence: int,
    reps: int,
    seed: int,
    batch_reps: int,
) -> Dict[
    str,
    Dict[
        str,
        float,
    ],
]:
    n_sequences = int(
        dead_counts.shape[0]
    )

    if not (
        same_bin_counts.shape[0]
        == n_sequences
        and high_z_dead_counts.shape[0]
        == n_sequences
        and high_z_counts.shape[0]
        == n_sequences
    ):
        raise RuntimeError(
            "Bootstrap sequence arrays "
            "have inconsistent lengths"
        )

    rng = np.random.default_rng(
        seed
    )

    dead_samples = np.empty(
        reps,
        dtype=np.float64,
    )

    same_bin_samples = np.empty(
        reps,
        dtype=np.float64,
    )

    high_z_dead_samples = np.empty(
        reps,
        dtype=np.float64,
    )

    denom = float(
        n_sequences
        * transitions_per_sequence
    )

    written = 0

    while written < reps:
        current = min(
            batch_reps,
            reps - written,
        )

        sample_idx = rng.integers(
            0,
            n_sequences,
            size=(
                current,
                n_sequences,
            ),
            dtype=np.int32,
        )

        dead_sum = np.take(
            dead_counts,
            sample_idx,
        ).sum(
            axis=1,
            dtype=np.int64,
        )

        same_sum = np.take(
            same_bin_counts,
            sample_idx,
        ).sum(
            axis=1,
            dtype=np.int64,
        )

        high_num = np.take(
            high_z_dead_counts,
            sample_idx,
        ).sum(
            axis=1,
            dtype=np.int64,
        )

        high_den = np.take(
            high_z_counts,
            sample_idx,
        ).sum(
            axis=1,
            dtype=np.int64,
        )

        dead_samples[
            written:
            written + current
        ] = (
            dead_sum
            / denom
        )

        same_bin_samples[
            written:
            written + current
        ] = (
            same_sum
            / denom
        )

        high_z_dead_samples[
            written:
            written + current
        ] = np.divide(
            high_num,
            high_den,
            out=np.full(
                current,
                np.nan,
                dtype=np.float64,
            ),
            where=high_den > 0,
        )

        written += current

    def interval(
        values: np.ndarray,
    ) -> Dict[
        str,
        float,
    ]:
        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.size == 0:
            return {
                "lower_95": None,
                "upper_95": None,
            }

        low, high = np.percentile(
            finite,
            [
                2.5,
                97.5,
            ],
        )

        return {
            "lower_95": float(
                low
            ),
            "upper_95": float(
                high
            ),
        }

    return {
        "dead_zone_fraction": interval(
            dead_samples
        ),
        "same_bin_fraction": interval(
            same_bin_samples
        ),
        "dead_zone_given_high_z": interval(
            high_z_dead_samples
        ),
    }


def analyze_phase(
    phase: str,
    run_dir: Path,
    cfg: Dict,
    saved_metrics: Dict,
    raw_input: np.ndarray,
    res_data: np.ndarray,
    test_idx: np.ndarray,
    inp_mean: float,
    inp_std: float,
    infer_batch: int,
    high_z_threshold: float,
    hist_bins: int,
    bootstrap_reps: int,
    bootstrap_seed: int,
    bootstrap_batch_reps: int,
    mae_tolerance: float,
) -> Dict:
    seq_len = int(
        cfg["seq_len"]
    )

    n_out = int(
        cfg["n_out"]
    )

    student_units = int(
        cfg["student_units"]
    )

    bits_kernel = int(
        cfg["bits_kernel"]
    )

    bits_state = int(
        cfg["bits_state"]
    )

    q_alpha = float(
        cfg.get(
            "q_alpha",
            cfg["quantizer_alpha"],
        )
    )

    delta_s = (
        2.0
        ** (
            -(
                bits_state
                - 1
            )
        )
    )

    half_lsb = (
        delta_s
        * 0.5
    )

    transitions_per_sequence = (
        seq_len
        * student_units
    )

    n_sequences = int(
        len(
            test_idx
        )
    )

    # h_t is a convex combination of values in [-1,1], so |Delta h| <= 2.
    # The tiny epsilon keeps an exact endpoint in the last histogram bin.
    hist_max = (
        2.0
        / delta_s
    ) + 1e-6

    hist_edges = np.linspace(
        0.0,
        hist_max,
        hist_bins + 1,
        dtype=np.float64,
    )

    hist_counts = np.zeros(
        hist_bins,
        dtype=np.int64,
    )

    sequence_dead_counts = np.zeros(
        n_sequences,
        dtype=np.int32,
    )

    sequence_same_bin_counts = np.zeros(
        n_sequences,
        dtype=np.int32,
    )

    sequence_high_z_dead_counts = np.zeros(
        n_sequences,
        dtype=np.int32,
    )

    sequence_high_z_counts = np.zeros(
        n_sequences,
        dtype=np.int32,
    )

    sequence_update_median = np.zeros(
        n_sequences,
        dtype=np.float32,
    )

    unit_dead_counts = np.zeros(
        student_units,
        dtype=np.int64,
    )

    unit_same_bin_counts = np.zeros(
        student_units,
        dtype=np.int64,
    )

    unit_high_z_dead_counts = np.zeros(
        student_units,
        dtype=np.int64,
    )

    unit_high_z_counts = np.zeros(
        student_units,
        dtype=np.int64,
    )

    total_dead = 0
    total_same_bin = 0
    total_high_z_dead = 0
    total_high_z = 0
    total_transitions = 0

    seq_abs_error_sum = 0.0
    seq_value_count = 0

    observed_norm_max = 0.0
    observed_hidden_min = float("inf")
    observed_hidden_max = float("-inf")

    nonfinite_hidden = 0
    nonfinite_z = 0

    keras.backend.clear_session()

    (
        model,
        enc_cell,
        dec_cell,
    ) = build_phase2_diagnostic_model(
        seq_len=seq_len,
        n_out=n_out,
        student_units=student_units,
        bits_kernel=bits_kernel,
        q_alpha=q_alpha,
        input_dim=1,
    )

    configure_paper_phase_quantizers(
        cfg,
        enc_cell,
        dec_cell,
        phase,
    )

    checkpoint_path = (
        run_dir
        / CHECKPOINT_NAMES[phase]
    )

    pf(
        f"[{phase}] Loading checkpoint: "
        f"{checkpoint_path}"
    )

    model.load_weights(
        str(
            checkpoint_path
        )
    )

    pf(
        f"[{phase}] Checkpoint loaded successfully"
    )

    q_state = quantized_bits(
        bits_state,
        0,
        1,
        alpha=1.0,
    )

    @tf.function(
        reduce_retracing=True
    )
    def inference_forward(
        enc_tensor,
        dec_tensor,
    ):
        return model(
            [
                enc_tensor,
                dec_tensor,
            ],
            training=False,
        )

    n_batches = math.ceil(
        n_sequences
        / infer_batch
    )

    phase_start = time.time()

    for (
        batch_number,
        start,
    ) in enumerate(
        range(
            0,
            n_sequences,
            infer_batch,
        ),
        start=1,
    ):
        end = min(
            start
            + infer_batch,
            n_sequences,
        )

        row_idx = test_idx[
            start:end
        ]

        raw_batch = np.asarray(
            raw_input[
                row_idx
            ],
            dtype=np.float32,
        )

        enc_batch = np.asarray(
            (
                raw_batch
                - inp_mean
            )
            / inp_std,
            dtype=np.float32,
        )

        del raw_batch

        tgt_batch = np.asarray(
            res_data[
                row_idx
            ],
            dtype=np.float32,
        )

        dec_batch = np.zeros(
            (
                end - start,
                seq_len,
                1,
            ),
            dtype=np.float32,
        )

        (
            seq_output,
            dec_hidden,
            dec_z_logits,
            enc_final_h,
        ) = inference_forward(
            tf.convert_to_tensor(
                enc_batch,
                dtype=tf.float32,
            ),
            tf.convert_to_tensor(
                dec_batch,
                dtype=tf.float32,
            ),
        )

        # Full decoder transition set:
        # encoder final state -> decoder step 0,
        # then decoder step t-1 -> decoder step t for t=1..T-1.
        prev_hidden = tf.concat(
            [
                enc_final_h[
                    :,
                    tf.newaxis,
                    :,
                ],
                dec_hidden[
                    :,
                    :-1,
                    :,
                ],
            ],
            axis=1,
        )

        abs_delta = tf.abs(
            dec_hidden
            - prev_hidden
        )

        norm_delta = (
            abs_delta
            / tf.cast(
                delta_s,
                tf.float32,
            )
        )

        dead_mask = (
            norm_delta
            < tf.cast(
                0.5,
                tf.float32,
            )
        )

        q_prev = q_state(
            prev_hidden
        )

        q_curr = q_state(
            dec_hidden
        )

        same_bin_mask = tf.equal(
            q_prev,
            q_curr,
        )

        z_value = tf.sigmoid(
            dec_z_logits
        )

        high_z_mask = (
            z_value
            > tf.cast(
                high_z_threshold,
                tf.float32,
            )
        )

        high_z_dead_mask = tf.logical_and(
            dead_mask,
            high_z_mask,
        )

        finite_hidden = tf.math.is_finite(
            dec_hidden
        )

        finite_z = tf.math.is_finite(
            z_value
        )

        nonfinite_hidden += int(
            tf.size(
                finite_hidden
            ).numpy()
            - tf.math.count_nonzero(
                finite_hidden
            ).numpy()
        )

        nonfinite_z += int(
            tf.size(
                finite_z
            ).numpy()
            - tf.math.count_nonzero(
                finite_z
            ).numpy()
        )

        seq_output_np = np.asarray(
            seq_output.numpy(),
            dtype=np.float32,
        )

        seq_abs_error_sum += float(
            np.sum(
                np.abs(
                    seq_output_np
                    - tgt_batch
                ),
                dtype=np.float64,
            )
        )

        seq_value_count += int(
            seq_output_np.size
        )

        hidden_np = np.asarray(
            dec_hidden.numpy(),
            dtype=np.float32,
        )

        observed_hidden_min = min(
            observed_hidden_min,
            float(
                np.min(
                    hidden_np
                )
            ),
        )

        observed_hidden_max = max(
            observed_hidden_max,
            float(
                np.max(
                    hidden_np
                )
            ),
        )

        norm_np = np.asarray(
            norm_delta.numpy(),
            dtype=np.float32,
        )

        dead_np = np.asarray(
            dead_mask.numpy(),
            dtype=np.bool_,
        )

        same_bin_np = np.asarray(
            same_bin_mask.numpy(),
            dtype=np.bool_,
        )

        high_z_np = np.asarray(
            high_z_mask.numpy(),
            dtype=np.bool_,
        )

        high_z_dead_np = np.asarray(
            high_z_dead_mask.numpy(),
            dtype=np.bool_,
        )

        observed_norm_max = max(
            observed_norm_max,
            float(
                np.max(
                    norm_np
                )
            ),
        )

        if np.any(
            norm_np < 0.0
        ):
            raise RuntimeError(
                f"[{phase}] Negative normalized "
                "update encountered"
            )

        if np.any(
            norm_np > hist_max
        ):
            raise RuntimeError(
                f"[{phase}] normalized update "
                "exceeds histogram bound: "
                f"max={float(np.max(norm_np))} "
                f"bound={hist_max}"
            )

        (
            batch_hist,
            _,
        ) = np.histogram(
            norm_np,
            bins=hist_edges,
        )

        hist_counts += batch_hist.astype(
            np.int64
        )

        batch_dead_counts = dead_np.sum(
            axis=(
                1,
                2,
            ),
            dtype=np.int32,
        )

        batch_same_bin_counts = same_bin_np.sum(
            axis=(
                1,
                2,
            ),
            dtype=np.int32,
        )

        batch_high_z_dead_counts = high_z_dead_np.sum(
            axis=(
                1,
                2,
            ),
            dtype=np.int32,
        )

        batch_high_z_counts = high_z_np.sum(
            axis=(
                1,
                2,
            ),
            dtype=np.int32,
        )

        sequence_dead_counts[
            start:end
        ] = batch_dead_counts

        sequence_same_bin_counts[
            start:end
        ] = batch_same_bin_counts

        sequence_high_z_dead_counts[
            start:end
        ] = batch_high_z_dead_counts

        sequence_high_z_counts[
            start:end
        ] = batch_high_z_counts

        sequence_update_median[
            start:end
        ] = np.median(
            norm_np,
            axis=(
                1,
                2,
            ),
        ).astype(
            np.float32
        )

        unit_dead_counts += dead_np.sum(
            axis=(
                0,
                1,
            ),
            dtype=np.int64,
        )

        unit_same_bin_counts += same_bin_np.sum(
            axis=(
                0,
                1,
            ),
            dtype=np.int64,
        )

        unit_high_z_dead_counts += high_z_dead_np.sum(
            axis=(
                0,
                1,
            ),
            dtype=np.int64,
        )

        unit_high_z_counts += high_z_np.sum(
            axis=(
                0,
                1,
            ),
            dtype=np.int64,
        )

        total_dead += int(
            batch_dead_counts.sum(
                dtype=np.int64
            )
        )

        total_same_bin += int(
            batch_same_bin_counts.sum(
                dtype=np.int64
            )
        )

        total_high_z_dead += int(
            batch_high_z_dead_counts.sum(
                dtype=np.int64
            )
        )

        total_high_z += int(
            batch_high_z_counts.sum(
                dtype=np.int64
            )
        )

        total_transitions += int(
            (
                end
                - start
            )
            * transitions_per_sequence
        )

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or end == n_sequences
        ):
            elapsed = (
                time.time()
                - phase_start
            )

            pf(
                f"[{phase}] batch "
                f"{batch_number}/{n_batches}  "
                f"samples {end}/{n_sequences}  "
                f"elapsed={elapsed / 60.0:.1f} min"
            )

        del (
            enc_batch,
            tgt_batch,
            dec_batch,
            seq_output,
            dec_hidden,
            dec_z_logits,
            enc_final_h,
            prev_hidden,
            abs_delta,
            norm_delta,
            dead_mask,
            q_prev,
            q_curr,
            same_bin_mask,
            z_value,
            high_z_mask,
            high_z_dead_mask,
            seq_output_np,
            hidden_np,
            norm_np,
            dead_np,
            same_bin_np,
            high_z_np,
            high_z_dead_np,
        )

    if (
        nonfinite_hidden != 0
        or nonfinite_z != 0
    ):
        raise RuntimeError(
            f"[{phase}] Non-finite values detected: "
            f"hidden={nonfinite_hidden}, "
            f"z={nonfinite_z}"
        )

    expected_total_transitions = (
        n_sequences
        * transitions_per_sequence
    )

    if (
        total_transitions
        != expected_total_transitions
    ):
        raise RuntimeError(
            f"[{phase}] Transition count mismatch: "
            f"actual={total_transitions} "
            f"expected={expected_total_transitions}"
        )

    if (
        int(
            hist_counts.sum()
        )
        != expected_total_transitions
    ):
        raise RuntimeError(
            f"[{phase}] Histogram count mismatch: "
            f"actual={int(hist_counts.sum())} "
            f"expected={expected_total_transitions}"
        )

    sequence_mae = (
        seq_abs_error_sum
        / float(
            seq_value_count
        )
    )

    saved_mae = float(
        saved_metrics["mae_seq"]
    )

    mae_abs_diff = abs(
        sequence_mae
        - saved_mae
    )

    if (
        mae_abs_diff
        > mae_tolerance
    ):
        raise RuntimeError(
            f"[{phase}] Reconstruction fidelity check FAILED. "
            f"Recomputed MAE={sequence_mae:.12g}, "
            f"saved MAE={saved_mae:.12g}, "
            f"abs diff={mae_abs_diff:.12g}, "
            f"tolerance={mae_tolerance:.12g}. "
            "Dead-zone results from this reconstruction "
            "are not accepted."
        )

    dead_fraction = (
        total_dead
        / float(
            total_transitions
        )
    )

    same_bin_fraction = (
        total_same_bin
        / float(
            total_transitions
        )
    )

    high_z_fraction = (
        total_high_z
        / float(
            total_transitions
        )
    )

    dead_given_high_z = (
        total_high_z_dead
        / float(
            total_high_z
        )
        if total_high_z > 0
        else float(
            "nan"
        )
    )

    median_norm_update = histogram_quantile(
        hist_counts,
        hist_edges,
        0.50,
    )

    q25_norm_update = histogram_quantile(
        hist_counts,
        hist_edges,
        0.25,
    )

    q75_norm_update = histogram_quantile(
        hist_counts,
        hist_edges,
        0.75,
    )

    q90_norm_update = histogram_quantile(
        hist_counts,
        hist_edges,
        0.90,
    )

    pf(
        f"[{phase}] Bootstrapping "
        f"{bootstrap_reps} sequence-level replicates..."
    )

    bootstrap_ci = bootstrap_sequence_level(
        dead_counts=sequence_dead_counts,
        same_bin_counts=sequence_same_bin_counts,
        high_z_dead_counts=sequence_high_z_dead_counts,
        high_z_counts=sequence_high_z_counts,
        transitions_per_sequence=transitions_per_sequence,
        reps=bootstrap_reps,
        seed=bootstrap_seed,
        batch_reps=bootstrap_batch_reps,
    )

    unit_denominator = (
        n_sequences
        * seq_len
    )

    per_unit = []

    for unit in range(
        student_units
    ):
        high_z_unit = int(
            unit_high_z_counts[
                unit
            ]
        )

        per_unit.append(
            {
                "unit": unit,
                "dead_zone_fraction": float(
                    unit_dead_counts[
                        unit
                    ]
                    / unit_denominator
                ),
                "same_bin_fraction": float(
                    unit_same_bin_counts[
                        unit
                    ]
                    / unit_denominator
                ),
                "high_z_fraction": float(
                    high_z_unit
                    / unit_denominator
                ),
                "dead_zone_given_high_z": (
                    float(
                        unit_high_z_dead_counts[
                            unit
                        ]
                        / high_z_unit
                    )
                    if high_z_unit > 0
                    else None
                ),
            }
        )

    same_bin_interpretation = (
        "actual_target_4bit_state_grid_used_by_recurrence"
        if phase == "P2F"
        else "projected_onto_target_4bit_state_grid"
    )

    elapsed_seconds = (
        time.time()
        - phase_start
    )

    result = {
        "phase": phase,
        "checkpoint": str(
            checkpoint_path
        ),
        "n_sequences": n_sequences,
        "seq_len": seq_len,
        "student_units": student_units,
        "transitions_per_sequence": (
            transitions_per_sequence
        ),
        "n_transitions": (
            total_transitions
        ),
        "bits_state": (
            bits_state
        ),
        "delta_s": (
            delta_s
        ),
        "half_lsb": (
            half_lsb
        ),
        "dead_zone_definition": (
            "abs(h_t-h_prev)/delta_s < 0.5"
        ),
        "same_bin_definition": (
            "Q_s(h_t) == Q_s(h_prev)"
        ),
        "same_bin_interpretation": (
            same_bin_interpretation
        ),
        "high_z_threshold": (
            high_z_threshold
        ),
        "dead_zone_fraction": float(
            dead_fraction
        ),
        "dead_zone_percent": float(
            dead_fraction
            * 100.0
        ),
        "dead_zone_ci95": (
            bootstrap_ci[
                "dead_zone_fraction"
            ]
        ),
        "same_bin_fraction": float(
            same_bin_fraction
        ),
        "same_bin_percent": float(
            same_bin_fraction
            * 100.0
        ),
        "same_bin_ci95": (
            bootstrap_ci[
                "same_bin_fraction"
            ]
        ),
        "high_z_fraction": float(
            high_z_fraction
        ),
        "high_z_percent": float(
            high_z_fraction
            * 100.0
        ),
        "dead_zone_given_high_z": (
            float(
                dead_given_high_z
            )
            if np.isfinite(
                dead_given_high_z
            )
            else None
        ),
        "dead_zone_given_high_z_percent": (
            float(
                dead_given_high_z
                * 100.0
            )
            if np.isfinite(
                dead_given_high_z
            )
            else None
        ),
        "dead_zone_given_high_z_ci95": (
            bootstrap_ci[
                "dead_zone_given_high_z"
            ]
        ),
        "normalized_update_q25_hist": float(
            q25_norm_update
        ),
        "normalized_update_median_hist": float(
            median_norm_update
        ),
        "normalized_update_q75_hist": float(
            q75_norm_update
        ),
        "normalized_update_q90_hist": float(
            q90_norm_update
        ),
        "normalized_update_observed_max": float(
            observed_norm_max
        ),
        "sequence_update_median_mean": float(
            np.mean(
                sequence_update_median
            )
        ),
        "sequence_update_median_std": float(
            np.std(
                sequence_update_median
            )
        ),
        "hidden_observed_min": float(
            observed_hidden_min
        ),
        "hidden_observed_max": float(
            observed_hidden_max
        ),
        "recomputed_mae_seq": float(
            sequence_mae
        ),
        "saved_mae_seq": float(
            saved_mae
        ),
        "mae_abs_diff": float(
            mae_abs_diff
        ),
        "mae_tolerance": float(
            mae_tolerance
        ),
        "fidelity_check_passed": True,
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "per_unit": per_unit,
        "hist_counts": (
            hist_counts
        ),
        "hist_edges": (
            hist_edges
        ),
        "sequence_dead_counts": (
            sequence_dead_counts
        ),
        "sequence_same_bin_counts": (
            sequence_same_bin_counts
        ),
        "sequence_high_z_dead_counts": (
            sequence_high_z_dead_counts
        ),
        "sequence_high_z_counts": (
            sequence_high_z_counts
        ),
        "sequence_update_median": (
            sequence_update_median
        ),
    }

    high_z_text = (
        f"{dead_given_high_z * 100.0:.3f}%"
        if np.isfinite(
            dead_given_high_z
        )
        else "n/a"
    )

    pf(
        f"[{phase}] PASS  "
        f"MAE={sequence_mae:.8f}  "
        f"dead={dead_fraction * 100.0:.3f}%  "
        f"same_bin={same_bin_fraction * 100.0:.3f}%  "
        f"high_z_dead={high_z_text}  "
        "median(|dh|/Delta_s)="
        f"{median_norm_update:.4f}"
    )

    return result


def write_summary_csv(
    path: Path,
    phase_results: List[Dict],
) -> None:
    fieldnames = [
        "phase",
        "n_sequences",
        "n_transitions",
        "delta_s",
        "half_lsb",
        "dead_zone_fraction",
        "dead_zone_percent",
        "dead_zone_ci95_lower",
        "dead_zone_ci95_upper",
        "same_bin_fraction",
        "same_bin_percent",
        "same_bin_ci95_lower",
        "same_bin_ci95_upper",
        "high_z_threshold",
        "high_z_fraction",
        "high_z_percent",
        "dead_zone_given_high_z",
        "dead_zone_given_high_z_percent",
        "dead_zone_given_high_z_ci95_lower",
        "dead_zone_given_high_z_ci95_upper",
        "normalized_update_q25_hist",
        "normalized_update_median_hist",
        "normalized_update_q75_hist",
        "normalized_update_q90_hist",
        "recomputed_mae_seq",
        "saved_mae_seq",
        "mae_abs_diff",
        "same_bin_interpretation",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in phase_results:
            writer.writerow(
                {
                    "phase": (
                        result["phase"]
                    ),
                    "n_sequences": (
                        result["n_sequences"]
                    ),
                    "n_transitions": (
                        result["n_transitions"]
                    ),
                    "delta_s": (
                        result["delta_s"]
                    ),
                    "half_lsb": (
                        result["half_lsb"]
                    ),
                    "dead_zone_fraction": (
                        result[
                            "dead_zone_fraction"
                        ]
                    ),
                    "dead_zone_percent": (
                        result[
                            "dead_zone_percent"
                        ]
                    ),
                    "dead_zone_ci95_lower": (
                        result[
                            "dead_zone_ci95"
                        ][
                            "lower_95"
                        ]
                    ),
                    "dead_zone_ci95_upper": (
                        result[
                            "dead_zone_ci95"
                        ][
                            "upper_95"
                        ]
                    ),
                    "same_bin_fraction": (
                        result[
                            "same_bin_fraction"
                        ]
                    ),
                    "same_bin_percent": (
                        result[
                            "same_bin_percent"
                        ]
                    ),
                    "same_bin_ci95_lower": (
                        result[
                            "same_bin_ci95"
                        ][
                            "lower_95"
                        ]
                    ),
                    "same_bin_ci95_upper": (
                        result[
                            "same_bin_ci95"
                        ][
                            "upper_95"
                        ]
                    ),
                    "high_z_threshold": (
                        result[
                            "high_z_threshold"
                        ]
                    ),
                    "high_z_fraction": (
                        result[
                            "high_z_fraction"
                        ]
                    ),
                    "high_z_percent": (
                        result[
                            "high_z_percent"
                        ]
                    ),
                    "dead_zone_given_high_z": (
                        result[
                            "dead_zone_given_high_z"
                        ]
                    ),
                    "dead_zone_given_high_z_percent": (
                        result[
                            "dead_zone_given_high_z_percent"
                        ]
                    ),
                    "dead_zone_given_high_z_ci95_lower": (
                        result[
                            "dead_zone_given_high_z_ci95"
                        ][
                            "lower_95"
                        ]
                    ),
                    "dead_zone_given_high_z_ci95_upper": (
                        result[
                            "dead_zone_given_high_z_ci95"
                        ][
                            "upper_95"
                        ]
                    ),
                    "normalized_update_q25_hist": (
                        result[
                            "normalized_update_q25_hist"
                        ]
                    ),
                    "normalized_update_median_hist": (
                        result[
                            "normalized_update_median_hist"
                        ]
                    ),
                    "normalized_update_q75_hist": (
                        result[
                            "normalized_update_q75_hist"
                        ]
                    ),
                    "normalized_update_q90_hist": (
                        result[
                            "normalized_update_q90_hist"
                        ]
                    ),
                    "recomputed_mae_seq": (
                        result[
                            "recomputed_mae_seq"
                        ]
                    ),
                    "saved_mae_seq": (
                        result[
                            "saved_mae_seq"
                        ]
                    ),
                    "mae_abs_diff": (
                        result[
                            "mae_abs_diff"
                        ]
                    ),
                    "same_bin_interpretation": (
                        result[
                            "same_bin_interpretation"
                        ]
                    ),
                }
            )


def write_per_unit_csv(
    path: Path,
    phase_results: List[Dict],
) -> None:
    fieldnames = [
        "phase",
        "unit",
        "dead_zone_fraction",
        "dead_zone_percent",
        "same_bin_fraction",
        "same_bin_percent",
        "high_z_fraction",
        "high_z_percent",
        "dead_zone_given_high_z",
        "dead_zone_given_high_z_percent",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in phase_results:
            for row in result[
                "per_unit"
            ]:
                dz_high = row[
                    "dead_zone_given_high_z"
                ]

                writer.writerow(
                    {
                        "phase": (
                            result["phase"]
                        ),
                        "unit": (
                            row["unit"]
                        ),
                        "dead_zone_fraction": (
                            row[
                                "dead_zone_fraction"
                            ]
                        ),
                        "dead_zone_percent": (
                            row[
                                "dead_zone_fraction"
                            ]
                            * 100.0
                        ),
                        "same_bin_fraction": (
                            row[
                                "same_bin_fraction"
                            ]
                        ),
                        "same_bin_percent": (
                            row[
                                "same_bin_fraction"
                            ]
                            * 100.0
                        ),
                        "high_z_fraction": (
                            row[
                                "high_z_fraction"
                            ]
                        ),
                        "high_z_percent": (
                            row[
                                "high_z_fraction"
                            ]
                            * 100.0
                        ),
                        "dead_zone_given_high_z": (
                            dz_high
                        ),
                        "dead_zone_given_high_z_percent": (
                            dz_high
                            * 100.0
                            if dz_high
                            is not None
                            else None
                        ),
                    }
                )


def write_histogram_csv(
    path: Path,
    phase_results: List[Dict],
) -> None:
    reference_edges = phase_results[
        0
    ][
        "hist_edges"
    ]

    for result in phase_results[
        1:
    ]:
        if not np.array_equal(
            reference_edges,
            result[
                "hist_edges"
            ],
        ):
            raise RuntimeError(
                "Histogram edges differ "
                "across phases"
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "bin_left",
            "bin_right",
        ]

        for result in phase_results:
            fieldnames.extend(
                [
                    (
                        f"{result['phase']}"
                        "_count"
                    ),
                    (
                        f"{result['phase']}"
                        "_cdf"
                    ),
                ]
            )

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        cdfs = {}

        for result in phase_results:
            counts = result[
                "hist_counts"
            ]

            cdfs[
                result["phase"]
            ] = (
                np.cumsum(
                    counts
                )
                / float(
                    np.sum(
                        counts
                    )
                )
            )

        for index in range(
            len(
                reference_edges
            )
            - 1
        ):
            row = {
                "bin_left": float(
                    reference_edges[
                        index
                    ]
                ),
                "bin_right": float(
                    reference_edges[
                        index + 1
                    ]
                ),
            }

            for result in phase_results:
                phase = result[
                    "phase"
                ]

                row[
                    f"{phase}_count"
                ] = int(
                    result[
                        "hist_counts"
                    ][
                        index
                    ]
                )

                row[
                    f"{phase}_cdf"
                ] = float(
                    cdfs[
                        phase
                    ][
                        index
                    ]
                )

            writer.writerow(
                row
            )


def write_per_sequence_npz(
    path: Path,
    phase_results: List[Dict],
) -> None:
    payload = {}

    for result in phase_results:
        phase = result[
            "phase"
        ]

        payload[
            f"{phase}_dead_counts"
        ] = result[
            "sequence_dead_counts"
        ]

        payload[
            f"{phase}_same_bin_counts"
        ] = result[
            "sequence_same_bin_counts"
        ]

        payload[
            f"{phase}_high_z_dead_counts"
        ] = result[
            "sequence_high_z_dead_counts"
        ]

        payload[
            f"{phase}_high_z_counts"
        ] = result[
            "sequence_high_z_counts"
        ]

        payload[
            f"{phase}_update_median_over_delta_s"
        ] = result[
            "sequence_update_median"
        ]

    np.savez_compressed(
        path,
        **payload,
    )


def plot_update_cdf(
    path: Path,
    phase_results: List[Dict],
    x_max: float,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            7.2,
            5.2,
        )
    )

    for result in phase_results:
        edges = result[
            "hist_edges"
        ]

        counts = result[
            "hist_counts"
        ]

        centers = 0.5 * (
            edges[:-1]
            + edges[1:]
        )

        cdf = (
            np.cumsum(
                counts
            )
            / float(
                np.sum(
                    counts
                )
            )
        )

        ax.plot(
            centers,
            cdf,
            linewidth=2.0,
            label=result[
                "phase"
            ],
        )

    ax.axvline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="half-LSB threshold",
    )

    ax.set_xlim(
        0.0,
        x_max,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        r"Normalized hidden-state update "
        r"$|h_t-h_{t-1}|/\Delta_s$"
    )

    ax.set_ylabel(
        "Cumulative fraction"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_per_unit(
    path: Path,
    phase_results: List[Dict],
    metric_key: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.0,
        )
    )

    for result in phase_results:
        units = np.array(
            [
                row[
                    "unit"
                ]
                for row in result[
                    "per_unit"
                ]
            ],
            dtype=np.int32,
        )

        values = np.array(
            [
                row[
                    metric_key
                ]
                * 100.0
                for row in result[
                    "per_unit"
                ]
            ],
            dtype=np.float64,
        )

        ax.plot(
            units,
            values,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=result[
                "phase"
            ],
        )

    ax.set_xlabel(
        "Decoder hidden unit"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xlim(
        -0.5,
        phase_results[
            0
        ][
            "student_units"
        ]
        - 0.5,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_summary_percent(
    path: Path,
    phase_results: List[Dict],
) -> None:
    phases = [
        result[
            "phase"
        ]
        for result
        in phase_results
    ]

    x = np.arange(
        len(
            phases
        ),
        dtype=np.float64,
    )

    width = 0.24

    dead = np.array(
        [
            result[
                "dead_zone_percent"
            ]
            for result
            in phase_results
        ],
        dtype=np.float64,
    )

    same = np.array(
        [
            result[
                "same_bin_percent"
            ]
            for result
            in phase_results
        ],
        dtype=np.float64,
    )

    high = np.array(
        [
            (
                result[
                    "dead_zone_given_high_z_percent"
                ]
                if result[
                    "dead_zone_given_high_z_percent"
                ]
                is not None
                else np.nan
            )
            for result
            in phase_results
        ],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(
        figsize=(
            7.2,
            5.2,
        )
    )

    ax.bar(
        x - width,
        dead,
        width,
        label="Sub-half-LSB updates",
    )

    ax.bar(
        x,
        same,
        width,
        label="Same-bin transitions",
    )

    threshold = phase_results[
        0
    ][
        "high_z_threshold"
    ]

    ax.bar(
        x + width,
        high,
        width,
        label=(
            "Sub-half-LSB given "
            f"z > {threshold:.2f}"
        ),
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        phases
    )

    ax.set_ylabel(
        "Fraction of transitions (%)"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def serializable_phase_result(
    result: Dict,
) -> Dict:
    excluded = {
        "hist_counts",
        "hist_edges",
        "sequence_dead_counts",
        "sequence_same_bin_counts",
        "sequence_high_z_dead_counts",
        "sequence_high_z_counts",
        "sequence_update_median",
    }

    return {
        key: value
        for key, value
        in result.items()
        if key not in excluded
    }


def write_outputs(
    out_dir: Path,
    phase_results: List[Dict],
    manifest: Dict,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_payload = {
        "analysis": (
            "MemoQ hidden-state dead-zone "
            "and target-grid collision"
        ),
        "phases": [
            serializable_phase_result(
                result
            )
            for result
            in phase_results
        ],
    }

    atomic_write_json(
        out_dir
        / "deadzone_summary.json",
        summary_payload,
    )

    write_summary_csv(
        out_dir
        / "deadzone_summary.csv",
        phase_results,
    )

    write_per_unit_csv(
        out_dir
        / "deadzone_per_unit.csv",
        phase_results,
    )

    write_histogram_csv(
        out_dir
        / "deadzone_update_histogram.csv",
        phase_results,
    )

    write_per_sequence_npz(
        out_dir
        / "deadzone_per_sequence.npz",
        phase_results,
    )

    full_x_max = float(
        max(
            result[
                "hist_edges"
            ][
                -1
            ]
            for result
            in phase_results
        )
    )

    plot_update_cdf(
        out_dir
        / "deadzone_update_cdf_full.png",
        phase_results,
        x_max=full_x_max,
    )

    plot_update_cdf(
        out_dir
        / "deadzone_update_cdf_zoom.png",
        phase_results,
        x_max=2.0,
    )

    plot_per_unit(
        out_dir
        / "deadzone_dead_fraction_per_unit.png",
        phase_results,
        metric_key="dead_zone_fraction",
        ylabel="Sub-half-LSB transitions (%)",
    )

    plot_per_unit(
        out_dir
        / "deadzone_same_bin_fraction_per_unit.png",
        phase_results,
        metric_key="same_bin_fraction",
        ylabel="Same-bin transitions (%)",
    )

    plot_summary_percent(
        out_dir
        / "deadzone_summary_percent.png",
        phase_results,
    )

    atomic_write_json(
        out_dir
        / "deadzone_manifest.json",
        manifest,
    )

    complete_path = (
        out_dir
        / "deadzone_complete.flag"
    )

    tmp_complete = (
        out_dir
        / "deadzone_complete.flag.tmp"
    )

    tmp_complete.write_text(
        "done\n",
        encoding="utf-8",
    )

    os.replace(
        tmp_complete,
        complete_path,
    )


def main() -> None:
    args = parse_args()

    validate_cli(
        args
    )

    validate_repository_source()

    configure_tensorflow()

    run_dir = Path(
        args.run_dir
    ).resolve()

    data_dir = Path(
        args.data_dir
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never leave a stale success marker from a previous run. The completion
    # flag is recreated only after all phases pass fidelity checks and every
    # output is written successfully.
    complete_flag = (
        out_dir
        / "deadzone_complete.flag"
    )

    if complete_flag.exists():
        complete_flag.unlink()

    pf(
        "="
        * 88
    )

    pf(
        "MEMOQ DEAD-ZONE ANALYSIS — PAPER RUN"
    )

    pf(
        "="
        * 88
    )

    pf(
        f"[PATH] repo_root : {REPO_ROOT}"
    )

    pf(
        f"[PATH] run_dir   : {run_dir}"
    )

    pf(
        f"[PATH] data_dir  : {data_dir}"
    )

    pf(
        f"[PATH] out_dir   : {out_dir}"
    )

    pf(
        f"[ARGS] phases    : {args.phases}"
    )

    pf(
        f"[ARGS] infer_batch={args.infer_batch}"
    )

    pf(
        "[ARGS] high_z_threshold="
        f"{args.high_z_threshold}"
    )

    pf(
        "[ARGS] bootstrap_reps="
        f"{args.bootstrap_reps} "
        f"seed={args.bootstrap_seed}"
    )

    student_args_path = (
        run_dir
        / "student_args.json"
    )

    if not student_args_path.is_file():
        raise FileNotFoundError(
            "Missing student_args.json: "
            f"{student_args_path}"
        )

    cfg = load_json(
        student_args_path
    )

    saved_metrics = validate_paper_run(
        run_dir,
        cfg,
        args.phases,
    )

    pf(
        "[GUARD] Paper-run configuration and "
        "saved metric signatures verified"
    )

    seq_len = int(
        cfg["seq_len"]
    )

    n_out = int(
        cfg["n_out"]
    )

    bits_state = int(
        cfg["bits_state"]
    )

    delta_s = (
        2.0
        ** (
            -(
                bits_state
                - 1
            )
        )
    )

    pf(
        f"[GRID] bits_state={bits_state}  "
        f"Delta_s={delta_s}  "
        f"half_LSB={delta_s / 2.0}"
    )

    (
        file_input,
        file_res,
        file_labels,
        file_train,
        file_val,
        file_test,
    ) = find_data_files(
        str(
            data_dir
        ),
        seq_len,
    )

    pf(
        f"[DATA] encoder input : {file_input}"
    )

    pf(
        f"[DATA] target        : {file_res}"
    )

    pf(
        f"[DATA] labels        : {file_labels}"
    )

    pf(
        f"[DATA] train index   : {file_train}"
    )

    pf(
        f"[DATA] val index     : {file_val}"
    )

    pf(
        f"[DATA] test index    : {file_test}"
    )

    raw_input = np.load(
        file_input,
        mmap_mode="r",
    )

    res_data = np.load(
        file_res,
        mmap_mode="r",
    )

    train_idx = np.load(
        file_train
    )

    test_idx = np.load(
        file_test
    )

    validate_dataset_shapes(
        raw_input=raw_input,
        res_data=res_data,
        train_idx=train_idx,
        test_idx=test_idx,
        seq_len=seq_len,
        n_out=n_out,
    )

    pf(
        f"[DATA] n_total={raw_input.shape[0]}  "
        f"train={len(train_idx)}  "
        f"test={len(test_idx)}"
    )

    (
        inp_mean,
        inp_std,
    ) = compute_training_normalization(
        raw_input,
        train_idx,
    )

    manifest = {
        "analysis_script": str(
            THIS_FILE
        ),
        "analysis_script_sha256": (
            sha256_file(
                THIS_FILE
            )
        ),
        "repo_root": str(
            REPO_ROOT
        ),
        "git_head": git_command(
            [
                "rev-parse",
                "HEAD",
            ]
        ),
        "git_branch": git_command(
            [
                "branch",
                "--show-current",
            ]
        ),
        "git_status_porcelain": git_command(
            [
                "status",
                "--porcelain",
            ]
        ),
        "train_student_memoq_git_blob": git_command(
            [
                "hash-object",
                str(
                    REPO_ROOT
                    / "train_student_memoq.py"
                ),
            ]
        ),
        "expected_train_student_memoq_git_blob": (
            EXPECTED_CURRENT_TRAIN_MEMOQ_GIT_BLOB
        ),
        "paper_run_dir": str(
            run_dir
        ),
        "student_args_json": str(
            student_args_path
        ),
        "student_args_sha256": sha256_file(
            student_args_path
        ),
        "data_dir": str(
            data_dir
        ),
        "data_files": {
            "input": (
                file_input
            ),
            "target": (
                file_res
            ),
            "labels": (
                file_labels
            ),
            "train_idx": (
                file_train
            ),
            "val_idx": (
                file_val
            ),
            "test_idx": (
                file_test
            ),
        },
        "normalization": {
            "mean": (
                inp_mean
            ),
            "std": (
                inp_std
            ),
            "computed_from": (
                "raw_input[train_idx] exactly as "
                "train_student_memoq.py"
            ),
        },
        "pre_july4_stage_reference_commit": (
            PRE_JULY4_STAGE_REFERENCE_COMMIT
        ),
        "july4_behavior_change_commit": (
            JULY4_BEHAVIOR_CHANGE_COMMIT
        ),
        "historical_inference_note": (
            "P2D/P2E/P2F are reconstructed with pre-July-4 hard "
            "inference semantics; all training-only noise/dither is "
            "disabled and all blend betas are 1.0."
        ),
        "tensorflow_version": (
            tf.__version__
        ),
        "numpy_version": (
            np.__version__
        ),
        "qkeras_version": getattr(
            __import__(
                "qkeras"
            ),
            "__version__",
            "n/a",
        ),
        "phases": (
            args.phases
        ),
        "infer_batch": (
            args.infer_batch
        ),
        "high_z_threshold": (
            args.high_z_threshold
        ),
        "bootstrap_reps": (
            args.bootstrap_reps
        ),
        "bootstrap_seed": (
            args.bootstrap_seed
        ),
        "bootstrap_batch_reps": (
            args.bootstrap_batch_reps
        ),
        "hist_bins": (
            args.hist_bins
        ),
        "mae_tolerance": (
            args.mae_tolerance
        ),
        "checkpoints": {},
    }

    for phase in args.phases:
        checkpoint_path = (
            run_dir
            / CHECKPOINT_NAMES[
                phase
            ]
        )

        metric_path = (
            run_dir
            / METRIC_NAMES[
                phase
            ]
        )

        manifest[
            "checkpoints"
        ][
            phase
        ] = {
            "path": str(
                checkpoint_path
            ),
            "sha256": sha256_file(
                checkpoint_path
            ),
            "size_bytes": (
                checkpoint_path.stat().st_size
            ),
            "mtime_epoch": (
                checkpoint_path.stat().st_mtime
            ),
            "saved_metrics_path": str(
                metric_path
            ),
            "saved_metrics_sha256": sha256_file(
                metric_path
            ),
        }

    phase_results = []

    for (
        phase_index,
        phase,
    ) in enumerate(
        args.phases
    ):
        pf(
            "="
            * 88
        )

        pf(
            f"ANALYZING {phase}"
        )

        pf(
            "="
            * 88
        )

        result = analyze_phase(
            phase=phase,
            run_dir=run_dir,
            cfg=cfg,
            saved_metrics=saved_metrics[
                phase
            ],
            raw_input=raw_input,
            res_data=res_data,
            test_idx=test_idx,
            inp_mean=inp_mean,
            inp_std=inp_std,
            infer_batch=args.infer_batch,
            high_z_threshold=args.high_z_threshold,
            hist_bins=args.hist_bins,
            bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=(
                args.bootstrap_seed
                + phase_index
            ),
            bootstrap_batch_reps=(
                args.bootstrap_batch_reps
            ),
            mae_tolerance=args.mae_tolerance,
        )

        phase_results.append(
            result
        )

    pf(
        "="
        * 88
    )

    pf(
        "WRITING OUTPUTS"
    )

    pf(
        "="
        * 88
    )

    write_outputs(
        out_dir,
        phase_results,
        manifest,
    )

    pf()

    pf(
        "="
        * 88
    )

    pf(
        "FINAL DEAD-ZONE SUMMARY"
    )

    pf(
        "="
        * 88
    )

    pf(
        "Phase   Dead-zone %   Same-bin %   "
        "High-z dead %   Median |dh|/Delta_s   "
        "MAE fidelity"
    )

    for result in phase_results:
        high_text = (
            (
                f"{result['dead_zone_given_high_z_percent']:.3f}"
            )
            if result[
                "dead_zone_given_high_z_percent"
            ]
            is not None
            else "n/a"
        )

        pf(
            f"{result['phase']:5s}   "
            f"{result['dead_zone_percent']:11.3f}   "
            f"{result['same_bin_percent']:10.3f}   "
            f"{high_text:13s}   "
            f"{result['normalized_update_median_hist']:19.4f}   "
            f"PASS ({result['mae_abs_diff']:.3e})"
        )

    pf(
        "="
        * 88
    )

    pf(
        f"[DONE] Outputs written to: {out_dir}"
    )

    pf(
        "[DONE] Completion flag: "
        f"{out_dir / 'deadzone_complete.flag'}"
    )


if __name__ == "__main__":
    main()
