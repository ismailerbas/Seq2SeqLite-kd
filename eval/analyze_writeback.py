#!/usr/bin/env python3
"""
eval/analyze_writeback.py

Fail-closed TensorFlow/QKeras recurrent-writeback diagnostics for the MemoQ
P2D/P2E/P2F/P3 checkpoints and vanilla QKeras student checkpoints.

The analysis reconstructs the exact GRU recurrence used by the repository,
validates that reconstruction directly against the original Keras/QKeras graph
on held-out samples, and then runs full-test inference under explicitly chosen
encoder and decoder writeback operators.

Live recurrent quantities use the causal chain

    q_recv_t -> h_t -> q_recv_{t+1}

for transitions whose newly produced hidden state is actually reused by a
subsequent recurrent step. The terminal decoder state is intentionally excluded
from W/N_write statistics because it is read by the output head but is not fed
back into another decoder recurrence.

For each aligned recurrent transition:

    I_t = |h_t - q_recv_t| / Delta
    W_t = |q_recv_{t+1} - q_recv_t| / Delta

and N_write(t) counts hidden units whose recurrence-visible state level changes.
W/N_write are only defined for live discrete writeback. Continuous-state
conditions still report I_t and projected grid occupancy, but never relabel a
counterfactual projection as a live writeback event.

The encoder and decoder writeback operators are independent. This permits the
causal location controls required by the paper:

    encoder-only state quantization
    decoder-only state quantization
    both encoder and decoder state quantization
    neither

Error-feedback residuals are local to a recurrent layer and are reset at the
encoder-to-decoder boundary. The hidden state crosses that boundary; the
artificial quantizer residual does not.

Two execution modes are supported:

  native_fidelity
      Builds the original repository model, loads the requested checkpoint,
      checks diagnostic-vs-original output/decoder-hidden/encoder-final tensors
      on held-out samples, runs the full native test set, compares aggregate
      metrics with the saved evaluation JSON, writes full native diagnostics,
      and creates native_fidelity.json.

  condition
      Requires a previously successful native_fidelity.json for the same
      checkpoint, student_args.json, and analysis source. It then runs only the
      requested writeback intervention, avoiding redundant full native passes.

No training, fine-tuning, checkpoint mutation, or weight overwrite occurs.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import h5py
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from qkeras import QDense, QGRU, quantized_bits, quantized_tanh

THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
REPO_ROOT = EVAL_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_student_memoq import (  # noqa: E402
    build_final_qkeras_student,
    build_phase2_model,
    find_data_files,
    qkeras_hard_sigmoid,
)


PHASE_CHOICES = (
    "P2D",
    "P2E",
    "P2F",
    "P3",
    "VANILLA",
)

MODE_CHOICES = (
    "native_fidelity",
    "condition",
)

WRITEBACK_CHOICES = (
    "native",
    "none",
    "deterministic",
    "stochastic",
    "error_feedback",
)

CHECKPOINT_NAMES = {
    "P2D": "stage2d_best.weights.h5",
    "P2E": "stage2e_best.weights.h5",
    "P2F": "stage2f_best.weights.h5",
    "P3": "student_best.weights.h5",
    "VANILLA": "student_best.weights.h5",
}

METRIC_FILE_CANDIDATES = {
    "P2D": (
        "test_metrics_P2D.json",
    ),
    "P2E": (
        "test_metrics_P2E.json",
    ),
    "P2F": (
        "test_metrics_P2F.json",
    ),
    "P3": (
        "test_metrics_P3.json",
        "test_metrics.json",
    ),
    "VANILLA": (
        "test_metrics.json",
    ),
}

NATIVE_QUANTIZED_STATE = {
    "P2D": False,
    "P2E": False,
    "P2F": True,
    "P3": True,
    "VANILLA": True,
}

SURVIVAL_EDGES = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    float("inf"),
)

AMP_FLOOR = 1e-6
DEFAULT_GATE_WIDTH_NS = 0.09
GRID_EPS = 1e-6

np_trapz = (
    np.trapz
    if hasattr(
        np,
        "trapz",
    )
    else np.trapezoid
)


def pf(
    message: str = "",
) -> None:
    print(
        message,
        flush=True,
    )


def load_json(
    path: Path,
) -> Dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def atomic_write_json(
    path: Path,
    payload: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def json_safe(
    value,
):
    if isinstance(
        value,
        (
            np.floating,
            np.integer,
        ),
    ):
        value = value.item()

    if isinstance(
        value,
        float,
    ) and (
        math.isnan(
            value
        )
        or math.isinf(
            value
        )
    ):
        return None

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal recurrent writeback diagnostics "
            "for MemoQ/QKeras students."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=MODE_CHOICES,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--phase",
        required=True,
        choices=PHASE_CHOICES,
    )

    parser.add_argument(
        "--checkpoint-file",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--encoder-writeback",
        choices=WRITEBACK_CHOICES,
        default="native",
    )

    parser.add_argument(
        "--decoder-writeback",
        choices=WRITEBACK_CHOICES,
        default="native",
    )

    parser.add_argument(
        "--encoder-state-bits",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--decoder-state-bits",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--fidelity-json",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--equivalence-samples",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--tensor-tolerance",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--tensor-mean-tolerance",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--tensor-mismatch-fraction",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--tensor-tie-fraction",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--mae-tolerance",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--rmse-tolerance",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--infer-batch",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--gate-hist-bins",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--bootstrap-batch-reps",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--sr-seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sr-seed-base",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--gate-width-ns",
        type=float,
        default=DEFAULT_GATE_WIDTH_NS,
    )

    return parser.parse_args()


def validate_cli(
    args: argparse.Namespace,
) -> None:
    run_dir = Path(
        args.run_dir
    ).resolve()

    data_dir = Path(
        args.data_dir
    ).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: "
            f"{run_dir}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{data_dir}"
        )

    if not (
        run_dir
        / "student_args.json"
    ).is_file():
        raise FileNotFoundError(
            f"Missing student_args.json in "
            f"{run_dir}"
        )

    if args.infer_batch <= 0:
        raise ValueError(
            "--infer-batch must be > 0"
        )

    if args.equivalence_samples <= 0:
        raise ValueError(
            "--equivalence-samples must be > 0"
        )

    if args.tensor_tolerance <= 0.0:
        raise ValueError(
            "--tensor-tolerance must be > 0"
        )

    if args.tensor_mean_tolerance <= 0.0:
        raise ValueError(
            "--tensor-mean-tolerance must be > 0"
        )

    if (
        args.tensor_mismatch_fraction < 0.0
        or args.tensor_mismatch_fraction >= 1.0
    ):
        raise ValueError(
            "--tensor-mismatch-fraction must be in [0, 1)"
        )

    if (
        args.tensor_tie_fraction < 0.0
        or args.tensor_tie_fraction >= 1.0
    ):
        raise ValueError(
            "--tensor-tie-fraction must be in [0, 1)"
        )

    if args.mae_tolerance <= 0.0:
        raise ValueError(
            "--mae-tolerance must be > 0"
        )

    if args.rmse_tolerance <= 0.0:
        raise ValueError(
            "--rmse-tolerance must be > 0"
        )

    if args.hist_bins < 128:
        raise ValueError(
            "--hist-bins must be >= 128"
        )

    if args.gate_hist_bins < 32:
        raise ValueError(
            "--gate-hist-bins must be >= 32"
        )

    if args.bootstrap_reps <= 0:
        raise ValueError(
            "--bootstrap-reps must be > 0"
        )

    if args.bootstrap_batch_reps <= 0:
        raise ValueError(
            "--bootstrap-batch-reps must be > 0"
        )

    if args.sr_seeds <= 0:
        raise ValueError(
            "--sr-seeds must be > 0"
        )

    if args.gate_width_ns <= 0.0:
        raise ValueError(
            "--gate-width-ns must be > 0"
        )

    for value, name in (
        (
            args.encoder_state_bits,
            "--encoder-state-bits",
        ),
        (
            args.decoder_state_bits,
            "--decoder-state-bits",
        ),
    ):
        if (
            value != -1
            and not (
                2
                <= value
                <= 16
            )
        ):
            raise ValueError(
                f"{name} must be -1 or "
                f"an integer in [2,16]"
            )

    if (
        args.mode
        == "condition"
        and not args.fidelity_json
    ):
        raise ValueError(
            "--mode condition requires "
            "--fidelity-json"
        )


def configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy(
        "float32"
    )

    tf.keras.utils.set_random_seed(
        42
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if not gpus:
        raise RuntimeError(
            "TensorFlow found no GPU; "
            "this paper analysis is GPU-only."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )

        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to enable memory growth "
                f"on {gpu}: {exc}"
            ) from exc

    pf(
        f"[GPU] TensorFlow physical GPUs: "
        f"{len(gpus)}"
    )

    for i, gpu in enumerate(
        gpus
    ):
        pf(
            f"[GPU]   {i}: {gpu}"
        )


def load_run_config(
    run_dir: Path,
) -> Dict:
    raw = load_json(
        run_dir
        / "student_args.json"
    )

    required = (
        "seq_len",
        "n_out",
        "student_units",
        "bits_kernel",
        "bits_recurrent",
        "bits_bias",
        "bits_activation",
        "bits_state",
    )

    missing = [
        key
        for key in required
        if key not in raw
    ]

    if missing:
        raise RuntimeError(
            "student_args.json missing "
            f"required keys: {missing}"
        )

    cfg = {
        "seq_len": int(
            raw["seq_len"]
        ),
        "n_out": int(
            raw["n_out"]
        ),
        "student_units": int(
            raw["student_units"]
        ),
        "bits_kernel": int(
            raw["bits_kernel"]
        ),
        "bits_recurrent": int(
            raw["bits_recurrent"]
        ),
        "bits_bias": int(
            raw["bits_bias"]
        ),
        "bits_activation": int(
            raw["bits_activation"]
        ),
        "bits_state": int(
            raw["bits_state"]
        ),
        "q_alpha": float(
            raw.get(
                "q_alpha",
                raw.get(
                    "quantizer_alpha",
                    1.0,
                ),
            )
        ),
    }

    if cfg["seq_len"] <= 1:
        raise RuntimeError(
            f"Invalid seq_len="
            f"{cfg['seq_len']}"
        )

    if cfg["student_units"] <= 0:
        raise RuntimeError(
            f"Invalid student_units="
            f"{cfg['student_units']}"
        )

    if cfg["n_out"] < 3:
        raise RuntimeError(
            "Lifetime evaluation requires "
            "n_out >= 3"
        )

    pf(
        "[CFG] seq_len={seq_len} "
        "n_out={n_out} "
        "units={student_units} "
        "bits k/r/b/a/s="
        "{bits_kernel}/"
        "{bits_recurrent}/"
        "{bits_bias}/"
        "{bits_activation}/"
        "{bits_state} "
        "q_alpha={q_alpha}".format(
            **cfg
        )
    )

    return cfg


def checkpoint_path_for(
    args: argparse.Namespace,
    run_dir: Path,
) -> Path:
    path = (
        run_dir
        / (
            args.checkpoint_file
            or CHECKPOINT_NAMES[
                args.phase
            ]
        )
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: "
            f"{path}"
        )

    return path


def load_dataset(
    data_dir: Path,
    cfg: Dict,
):
    (
        file_input,
        file_res,
        file_labels,
        _file_train,
        _file_val,
        file_test,
    ) = find_data_files(
        str(
            data_dir
        ),
        cfg["seq_len"],
    )

    pf(
        f"[DATA] encoder input : "
        f"{file_input}"
    )

    pf(
        f"[DATA] decoder target: "
        f"{file_res}"
    )

    pf(
        f"[DATA] labels        : "
        f"{file_labels}"
    )

    pf(
        f"[DATA] test index    : "
        f"{file_test}"
    )

    normalized_input = np.load(
        file_input,
        mmap_mode="r",
    )

    res_data = np.load(
        file_res,
        mmap_mode="r",
    )

    labels = np.load(
        file_labels,
        mmap_mode="r",
    )

    test_idx = np.asarray(
        np.load(
            file_test
        ),
        dtype=np.int64,
    )

    expected_in = (
        cfg["seq_len"],
        1,
    )

    expected_out = (
        cfg["seq_len"],
        cfg["n_out"],
    )

    if (
        normalized_input.ndim != 3
        or tuple(
            normalized_input.shape[
                1:
            ]
        )
        != expected_in
    ):
        raise RuntimeError(
            "Expected normalized input "
            f"shape (N,{expected_in[0]},1), "
            f"got {normalized_input.shape}"
        )

    if (
        res_data.ndim != 3
        or tuple(
            res_data.shape[
                1:
            ]
        )
        != expected_out
    ):
        raise RuntimeError(
            "Expected target shape "
            f"(N,{expected_out[0]},"
            f"{expected_out[1]}), "
            f"got {res_data.shape}"
        )

    if (
        labels.ndim != 2
        or labels.shape[1] < 3
    ):
        raise RuntimeError(
            "Expected labels shape "
            f"(N,>=3), got {labels.shape}"
        )

    n_total = (
        normalized_input.shape[0]
    )

    if (
        res_data.shape[0]
        != n_total
        or labels.shape[0]
        != n_total
    ):
        raise RuntimeError(
            "Input/target/label sample "
            "counts do not match"
        )

    if (
        test_idx.ndim != 1
        or len(
            test_idx
        )
        == 0
    ):
        raise RuntimeError(
            "Test index must be a "
            "non-empty 1-D array"
        )

    if (
        int(
            test_idx.min()
        )
        < 0
        or int(
            test_idx.max()
        )
        >= n_total
    ):
        raise RuntimeError(
            "Test index contains "
            "out-of-range entries"
        )

    pf(
        f"[DATA] N_total={n_total} "
        f"N_test={len(test_idx)}"
    )

    return (
        normalized_input,
        res_data,
        labels,
        test_idx,
    )


def build_vanilla_reference(
    cfg: Dict,
) -> Model:
    def qwk():
        return quantized_bits(
            cfg["bits_kernel"],
            0,
            1,
            alpha=1.0,
        )

    def qwr():
        return quantized_bits(
            cfg["bits_recurrent"],
            0,
            1,
            alpha=1.0,
        )

    def qwb():
        return quantized_bits(
            cfg["bits_bias"],
            0,
            1,
            alpha=1.0,
        )

    def qa():
        return quantized_tanh(
            bits=cfg[
                "bits_activation"
            ],
            symmetric=True,
        )

    def qs():
        return quantized_bits(
            cfg["bits_state"],
            0,
            1,
            alpha=1.0,
        )

    def qd():
        return quantized_bits(
            cfg["bits_kernel"],
            0,
        )

    enc_inputs = Input(
        shape=(
            None,
            1,
        ),
        name="senc_input",
    )

    dec_inputs = Input(
        shape=(
            None,
            1,
        ),
        name="sdec_input",
    )

    (
        _enc_out,
        enc_state,
    ) = QGRU(
        units=cfg[
            "student_units"
        ],
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(
        enc_inputs
    )

    (
        dec_hid,
        _dec_state,
    ) = QGRU(
        units=cfg[
            "student_units"
        ],
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        name="sdecgru",
    )(
        dec_inputs,
        initial_state=enc_state,
    )

    output = QDense(
        cfg["n_out"],
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(
        dec_hid
    )

    return Model(
        [
            enc_inputs,
            dec_inputs,
        ],
        output,
        name=(
            "student_vanilla_kd_"
            "reference"
        ),
    )


def configure_phase2_hard_semantics(
    cfg: Dict,
    phase: str,
    enc_cell,
    dec_cell,
) -> None:
    if phase not in (
        "P2D",
        "P2E",
        "P2F",
    ):
        raise ValueError(
            "Invalid Phase-2 phase: "
            f"{phase}"
        )

    qk = quantized_bits(
        cfg["bits_kernel"],
        0,
        1,
        alpha=cfg[
            "q_alpha"
        ],
    )

    qr = quantized_bits(
        cfg["bits_recurrent"],
        0,
        1,
        alpha=cfg[
            "q_alpha"
        ],
    )

    qb = quantized_bits(
        cfg["bits_bias"],
        0,
        1,
        alpha=cfg[
            "q_alpha"
        ],
    )

    qa = quantized_tanh(
        bits=cfg[
            "bits_activation"
        ],
        symmetric=True,
    )

    qs = quantized_bits(
        cfg["bits_state"],
        0,
        1,
        alpha=1.0,
    )

    for cell in (
        enc_cell,
        dec_cell,
    ):
        cell.quantizer_h = qk
        cell.quantizer_r = qk
        cell.quantizer_z = qk

        cell.quantizer_recurrent_h = qr
        cell.quantizer_recurrent_r = qr
        cell.quantizer_recurrent_z = qr

        cell.quantizer_bias = qb

        cell.quantizer_activation = (
            qa
            if phase
            in (
                "P2E",
                "P2F",
            )
            else None
        )

        cell.quantizer_state = (
            qs
            if phase == "P2F"
            else None
        )

        if hasattr(
            cell,
            "activation_blend_beta",
        ):
            cell.activation_blend_beta = (
                1.0
            )

        if hasattr(
            cell,
            "state_blend_beta",
        ):
            cell.state_blend_beta = (
                1.0
            )

        if hasattr(
            cell,
            "act_dither_delta",
        ):
            cell.act_dither_delta = (
                0.0
            )

        if hasattr(
            cell,
            "state_dither_delta",
        ):
            cell.state_dither_delta = (
                0.0
            )

        if hasattr(
            cell,
            "state_lsb_noise_std",
        ):
            cell.state_lsb_noise_std = (
                0.0
            )


def load_checkpoint_weights_by_name(
    model,
    checkpoint_path: Path,
) -> None:
    """
    Load an HDF5 weights checkpoint into ``model`` by layer NAME.

    Keras's positional ``load_weights`` fails with "Layer count mismatch"
    whenever the checkpoint was saved from a training graph that carried
    extra weight-bearing layers.  MemoQ checkpoints carry exactly one such
    extra layer, ``teacher_hidden_seq2seq``, because the trainer assigns the
    teacher model onto the student model as an attribute and Keras
    auto-tracks it as a sublayer.

    This loader:

      1. inventories every layer saved in the checkpoint WITHOUT resolving
         leaf names, so that nested sub-models (whose weight paths legally
         repeat leaves such as ``kernel`` across their internal layers)
         never trigger a false ambiguity error;
      2. resolves leaf names only for the layers the model actually needs;
      3. matches weights inside each needed layer by leaf variable name
         (e.g. ``kernel``, ``recurrent_kernel``, ``bias``, ``W_z``),
         verifying shapes exactly;
      4. raises RuntimeError if any model layer that owns weights has no
         saved counterpart, if a needed leaf is missing or ambiguous, or if
         any shape differs, so silent partial loads are impossible;
      5. reports saved layers the model does not use.
    """
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    with h5py.File(
        str(
            checkpoint_path
        ),
        "r",
    ) as handle:
        if "layer_names" in handle.attrs:
            group = handle
        elif (
            "model_weights" in handle
            and "layer_names"
            in handle["model_weights"].attrs
        ):
            group = handle[
                "model_weights"
            ]
        else:
            raise RuntimeError(
                "Unrecognized HDF5 weights "
                "layout (no layer_names "
                "attribute) in "
                f"{checkpoint_path}"
            )

        saved_layer_names = [
            name.decode("utf8")
            if isinstance(
                name,
                bytes,
            )
            else str(
                name
            )
            for name
            in group.attrs[
                "layer_names"
            ]
        ]

        saved = {}

        for layer_name in saved_layer_names:
            layer_group = group[
                layer_name
            ]

            weight_names = [
                name.decode("utf8")
                if isinstance(
                    name,
                    bytes,
                )
                else str(
                    name
                )
                for name
                in layer_group.attrs.get(
                    "weight_names",
                    [],
                )
            ]

            entries = []

            for weight_name in weight_names:
                entries.append(
                    (
                        weight_name,
                        np.asarray(
                            layer_group[
                                weight_name
                            ][
                                ...
                            ],
                            dtype=np.float32,
                        ),
                    )
                )

            saved[
                layer_name
            ] = entries

            pf(
                f"[CKPT] saved layer "
                f"{layer_name!r}: "
                f"{len(entries)} weights"
            )

    consumed = set()

    for layer in model.layers:
        if not layer.weights:
            continue

        if layer.name not in saved:
            raise RuntimeError(
                f"Model layer "
                f"{layer.name!r} has no "
                "saved counterpart in "
                f"{checkpoint_path}. "
                "Saved layers: "
                f"{sorted(saved)}"
            )

        entries = saved[
            layer.name
        ]

        by_leaf = {}

        for weight_name, array in entries:
            leaf = (
                weight_name.split(
                    "/"
                )[-1]
                .split(
                    ":"
                )[0]
            )

            if leaf in by_leaf:
                raise RuntimeError(
                    "Ambiguous saved weight "
                    f"leaf {leaf!r} in "
                    f"checkpoint layer "
                    f"{layer.name!r} of "
                    f"{checkpoint_path}; "
                    "saved weight paths: "
                    f"{[name for name, _ in entries]}"
                )

            by_leaf[
                leaf
            ] = array

        new_values = []

        used_leaves = set()

        for variable in layer.weights:
            leaf = (
                variable.name.split(
                    "/"
                )[-1]
                .split(
                    ":"
                )[0]
            )

            if leaf not in by_leaf:
                raise RuntimeError(
                    f"Weight {leaf!r} of "
                    f"model layer "
                    f"{layer.name!r} is "
                    "missing from the "
                    "checkpoint. Saved "
                    "weights for this "
                    "layer: "
                    f"{sorted(by_leaf)}"
                )

            if leaf in used_leaves:
                raise RuntimeError(
                    "Ambiguous model weight "
                    f"leaf {leaf!r} in layer "
                    f"{layer.name!r}"
                )

            used_leaves.add(
                leaf
            )

            array = by_leaf[
                leaf
            ]

            expected_shape = tuple(
                int(
                    dim
                )
                for dim
                in variable.shape
            )

            if (
                tuple(
                    array.shape
                )
                != expected_shape
            ):
                raise RuntimeError(
                    f"Shape mismatch for "
                    f"{layer.name}/{leaf}: "
                    f"checkpoint "
                    f"{tuple(array.shape)} "
                    f"vs model "
                    f"{expected_shape}"
                )

            new_values.append(
                array
            )

        unused_leaves = sorted(
            set(
                by_leaf
            )
            - used_leaves
        )

        if unused_leaves:
            raise RuntimeError(
                f"Checkpoint layer "
                f"{layer.name!r} carries "
                "weights the model does "
                "not own: "
                f"{unused_leaves}"
            )

        layer.set_weights(
            new_values
        )

        consumed.add(
            layer.name
        )

        pf(
            f"[CKPT] loaded layer "
            f"{layer.name!r} "
            f"({len(new_values)} "
            "weights) by name"
        )

    for name in saved:
        if (
            name not in consumed
            and saved[
                name
            ]
        ):
            pf(
                f"[CKPT] ignoring saved "
                f"layer {name!r} "
                "(not part of the "
                "reference model): "
                f"{len(saved[name])} "
                "weights"
            )


def build_original_reference(
    cfg: Dict,
    phase: str,
    checkpoint_path: Path,
):
    if phase in (
        "P2D",
        "P2E",
        "P2F",
    ):
        (
            model,
            enc_cell,
            dec_cell,
        ) = build_phase2_model(
            seq_len=cfg[
                "seq_len"
            ],
            n_out=cfg[
                "n_out"
            ],
            student_units=cfg[
                "student_units"
            ],
            input_dim=1,
            q_alpha=cfg[
                "q_alpha"
            ],
            bits_kernel=cfg[
                "bits_kernel"
            ],
        )

        configure_phase2_hard_semantics(
            cfg,
            phase,
            enc_cell,
            dec_cell,
        )

        load_checkpoint_weights_by_name(
            model,
            checkpoint_path,
        )

        configure_phase2_hard_semantics(
            cfg,
            phase,
            enc_cell,
            dec_cell,
        )

        enc_unroll_out = (
            model.get_layer(
                "sencgru_unroll"
            ).output
        )

        enc_final = (
            enc_unroll_out[2]
        )

        reference = keras.Model(
            inputs=model.inputs,
            outputs=[
                model.outputs[0],
                model.outputs[1],
                enc_final,
            ],
            name=(
                f"{phase.lower()}_"
                "reference_outputs"
            ),
        )

        return (
            model,
            reference,
            enc_cell,
            dec_cell,
        )

    if phase == "P3":
        model = (
            build_final_qkeras_student(
                seq_len=cfg[
                    "seq_len"
                ],
                n_out=cfg[
                    "n_out"
                ],
                student_units=cfg[
                    "student_units"
                ],
                bits_kernel=cfg[
                    "bits_kernel"
                ],
                bits_recurrent=cfg[
                    "bits_recurrent"
                ],
                bits_bias=cfg[
                    "bits_bias"
                ],
                bits_activation=cfg[
                    "bits_activation"
                ],
                bits_state=cfg[
                    "bits_state"
                ],
                q_alpha=cfg[
                    "q_alpha"
                ],
            )
        )

    elif phase == "VANILLA":
        model = (
            build_vanilla_reference(
                cfg
            )
        )

    else:
        raise ValueError(
            f"Unsupported phase: "
            f"{phase}"
        )

    load_checkpoint_weights_by_name(
        model,
        checkpoint_path,
    )

    enc_out = (
        model.get_layer(
            "sencgru"
        ).output
    )

    dec_out = (
        model.get_layer(
            "sdecgru"
        ).output
    )

    enc_final = (
        enc_out[1]
        if isinstance(
            enc_out,
            (
                list,
                tuple,
            ),
        )
        else enc_out
    )

    dec_hidden = (
        dec_out[0]
        if isinstance(
            dec_out,
            (
                list,
                tuple,
            ),
        )
        else dec_out
    )

    reference = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.output,
            dec_hidden,
            enc_final,
        ],
        name=(
            f"{phase.lower()}_"
            "reference_outputs"
        ),
    )

    return (
        model,
        reference,
        None,
        None,
    )


def _layer_weight(
    layer,
    leaf: str,
) -> np.ndarray:
    matches = []

    for variable in layer.weights:
        name = (
            variable.name.split(
                "/"
            )[-1]
            .split(
                ":"
            )[0]
        )

        if name == leaf:
            matches.append(
                variable
            )

    if len(
        matches
    ) != 1:
        names = [
            variable.name
            for variable
            in layer.weights
        ]

        raise RuntimeError(
            f"Expected exactly one "
            f"{leaf!r} weight in "
            f"layer {layer.name}; "
            f"found {len(matches)}. "
            f"Available weights: "
            f"{names}"
        )

    return np.asarray(
        matches[0].numpy(),
        dtype=np.float32,
    )


def extract_raw_weights(
    model,
    cfg: Dict,
    phase: str,
    enc_cell=None,
    dec_cell=None,
) -> Dict[str, np.ndarray]:
    H = cfg[
        "student_units"
    ]

    if phase in (
        "P2D",
        "P2E",
        "P2F",
    ):
        if (
            enc_cell is None
            or dec_cell is None
        ):
            raise RuntimeError(
                "Phase-2 extraction "
                "requires encoder/decoder "
                "MemoQ cells"
            )

        weights = {}

        for prefix, cell in (
            (
                "enc",
                enc_cell,
            ),
            (
                "dec",
                dec_cell,
            ),
        ):
            weights[
                f"{prefix}_kernel"
            ] = np.concatenate(
                [
                    cell.W_z.numpy(),
                    cell.W_r.numpy(),
                    cell.W_h.numpy(),
                ],
                axis=1,
            ).astype(
                np.float32
            )

            weights[
                f"{prefix}_recurrent"
            ] = np.concatenate(
                [
                    cell.U_z.numpy(),
                    cell.U_r.numpy(),
                    cell.U_h.numpy(),
                ],
                axis=1,
            ).astype(
                np.float32
            )

            weights[
                f"{prefix}_bias"
            ] = np.concatenate(
                [
                    cell.b_z_inp.numpy(),
                    cell.b_r_inp.numpy(),
                    cell.b_h_inp.numpy(),
                ],
                axis=0,
            ).astype(
                np.float32
            )

    else:
        weights = {}

        for (
            prefix,
            layer_name,
        ) in (
            (
                "enc",
                "sencgru",
            ),
            (
                "dec",
                "sdecgru",
            ),
        ):
            layer = (
                model.get_layer(
                    layer_name
                )
            )

            weights[
                f"{prefix}_kernel"
            ] = _layer_weight(
                layer,
                "kernel",
            )

            weights[
                f"{prefix}_recurrent"
            ] = _layer_weight(
                layer,
                "recurrent_kernel",
            )

            weights[
                f"{prefix}_bias"
            ] = _layer_weight(
                layer,
                "bias",
            )

    dense = model.get_layer(
        "sdec_dense"
    )

    weights[
        "dense_kernel"
    ] = _layer_weight(
        dense,
        "kernel",
    )

    weights[
        "dense_bias"
    ] = _layer_weight(
        dense,
        "bias",
    )

    expected = {
        "enc_kernel": (
            1,
            3 * H,
        ),
        "enc_recurrent": (
            H,
            3 * H,
        ),
        "enc_bias": (
            3 * H,
        ),
        "dec_kernel": (
            1,
            3 * H,
        ),
        "dec_recurrent": (
            H,
            3 * H,
        ),
        "dec_bias": (
            3 * H,
        ),
        "dense_kernel": (
            H,
            cfg["n_out"],
        ),
        "dense_bias": (
            cfg["n_out"],
        ),
    }

    for name, shape in (
        expected.items()
    ):
        if (
            tuple(
                weights[
                    name
                ].shape
            )
            != shape
        ):
            raise RuntimeError(
                f"{name} shape "
                f"{weights[name].shape} "
                f"!= expected "
                f"{shape}"
            )

        if not np.all(
            np.isfinite(
                weights[
                    name
                ]
            )
        ):
            raise RuntimeError(
                f"{name} contains "
                "non-finite values"
            )

        pf(
            f"[WEIGHTS] {name}: "
            f"shape="
            f"{weights[name].shape} "
            f"min="
            f"{weights[name].min():.6g} "
            f"max="
            f"{weights[name].max():.6g}"
        )

    return weights


def _require_quantizer(
    owner: str,
    name: str,
    quantizer,
):
    if quantizer is None:
        raise RuntimeError(
            f"{owner} has no {name} "
            "quantizer; the reference "
            "model is not configured as "
            "this analysis expects"
        )

    return quantizer


def build_parameter_quantizers(
    model,
    cfg: Dict,
    phase: str,
    enc_cell=None,
    dec_cell=None,
) -> Dict:
    """
    Return the ACTUAL quantizer objects the reference model applies.

    These are read off the loaded layers rather than reconstructed from the
    run configuration.  Reconstruction is unsafe: QKeras mutates quantizers
    at layer-construction time.  ``QDense.__init__`` calls
    ``_set_trainable_parameter()`` on its kernel quantizer, and
    ``quantized_bits._set_trainable_parameter`` rewrites ``alpha`` from
    ``None`` to ``"auto_po2"`` and sets ``symmetric = True``.  A separately
    constructed ``quantized_bits(bits, 0)`` therefore quantizes with a plain
    unit scale while the layer itself quantizes with per-channel
    power-of-two scaling, which silently changes the effective dense
    weights.  Reading the layer's own objects makes that class of mismatch
    impossible for every quantizer, now and for any future QKeras change.
    """
    if phase in (
        "P2D",
        "P2E",
        "P2F",
    ):
        if (
            enc_cell is None
            or dec_cell is None
        ):
            raise RuntimeError(
                "Phase-2 quantizer "
                "extraction requires "
                "encoder/decoder MemoQ "
                "cells"
            )

        enc_kernel = _require_quantizer(
            "memoq encoder cell",
            "kernel",
            enc_cell.quantizer_h,
        )

        enc_recurrent = _require_quantizer(
            "memoq encoder cell",
            "recurrent",
            enc_cell.quantizer_recurrent_h,
        )

        enc_bias = _require_quantizer(
            "memoq encoder cell",
            "bias",
            enc_cell.quantizer_bias,
        )

        dec_kernel = _require_quantizer(
            "memoq decoder cell",
            "kernel",
            dec_cell.quantizer_h,
        )

        dec_recurrent = _require_quantizer(
            "memoq decoder cell",
            "recurrent",
            dec_cell.quantizer_recurrent_h,
        )

        dec_bias = _require_quantizer(
            "memoq decoder cell",
            "bias",
            dec_cell.quantizer_bias,
        )

        enc_activation = (
            enc_cell.quantizer_activation
        )

        dec_activation = (
            dec_cell.quantizer_activation
        )

    else:
        enc_layer = model.get_layer(
            "sencgru"
        )

        dec_layer = model.get_layer(
            "sdecgru"
        )

        enc_kernel = _require_quantizer(
            "sencgru",
            "kernel",
            enc_layer.cell.kernel_quantizer_internal,
        )

        enc_recurrent = _require_quantizer(
            "sencgru",
            "recurrent",
            enc_layer.cell.recurrent_quantizer_internal,
        )

        enc_bias = _require_quantizer(
            "sencgru",
            "bias",
            enc_layer.cell.bias_quantizer_internal,
        )

        dec_kernel = _require_quantizer(
            "sdecgru",
            "kernel",
            dec_layer.cell.kernel_quantizer_internal,
        )

        dec_recurrent = _require_quantizer(
            "sdecgru",
            "recurrent",
            dec_layer.cell.recurrent_quantizer_internal,
        )

        dec_bias = _require_quantizer(
            "sdecgru",
            "bias",
            dec_layer.cell.bias_quantizer_internal,
        )

        enc_activation = (
            enc_layer.cell.activation
        )

        dec_activation = (
            dec_layer.cell.activation
        )

    if (
        str(
            enc_activation
        )
        != str(
            dec_activation
        )
    ):
        raise RuntimeError(
            "Encoder and decoder "
            "candidate activations "
            "differ: "
            f"{enc_activation} vs "
            f"{dec_activation}. The "
            "unrolled forward pass "
            "applies a single "
            "activation to both."
        )

    dense_layer = model.get_layer(
        "sdec_dense"
    )

    dense_kernel = _require_quantizer(
        "sdec_dense",
        "kernel",
        dense_layer.kernel_quantizer_internal,
    )

    dense_bias = _require_quantizer(
        "sdec_dense",
        "bias",
        dense_layer.bias_quantizer_internal,
    )

    quantizers = {
        "enc_kernel": enc_kernel,
        "enc_recurrent": enc_recurrent,
        "enc_bias": enc_bias,
        "dec_kernel": dec_kernel,
        "dec_recurrent": dec_recurrent,
        "dec_bias": dec_bias,
        "dense_kernel": dense_kernel,
        "dense_bias": dense_bias,
        "activation": enc_activation,
    }

    for name, quantizer in (
        quantizers.items()
    ):
        pf(
            f"[QUANT] {name}: "
            f"{quantizer}"
        )

    return quantizers


def quantize_effective_weights(
    raw: Dict[str, np.ndarray],
    quantizers: Dict,
) -> Dict[str, np.ndarray]:
    """
    Apply each region's OWN quantizer to that region's raw weights.

    Every tensor is handed to the quantizer in exactly the shape the layer
    itself passes, so scale-bearing quantizers such as ``auto_po2`` compute
    identical scales to the reference model.
    """
    result = {}

    for name in (
        "enc_kernel",
        "enc_recurrent",
        "enc_bias",
        "dec_kernel",
        "dec_recurrent",
        "dec_bias",
        "dense_kernel",
        "dense_bias",
    ):
        result[
            name
        ] = np.asarray(
            quantizers[
                name
            ](
                tf.convert_to_tensor(
                    raw[
                        name
                    ],
                    tf.float32,
                )
            ).numpy(),
            dtype=np.float32,
        )

        pf(
            f"[QUANT] effective {name}: "
            f"shape="
            f"{result[name].shape} "
            f"min="
            f"{result[name].min():.6g} "
            f"max="
            f"{result[name].max():.6g}"
        )

    return result


class WritebackOperator:
    def __init__(
        self,
        kind: str,
        state_bits: int,
    ):
        if kind not in (
            "identity",
            "deterministic",
            "stochastic",
            "error_feedback",
        ):
            raise ValueError(
                f"Unknown writeback "
                f"kind: {kind}"
            )

        self.kind = kind

        self.state_bits = int(
            state_bits
        )

        self.delta = float(
            2.0
            ** (
                -(
                    self.state_bits
                    - 1
                )
            )
        )

        self.qmax = float(
            1.0
            - self.delta
        )

        self.qmin = float(
            -self.qmax
        )

        self._det = quantized_bits(
            self.state_bits,
            0,
            1,
            alpha=1.0,
        )

    @property
    def live_discrete(
        self,
    ) -> bool:
        return (
            self.kind
            != "identity"
        )

    def deterministic_quantize(
        self,
        h: tf.Tensor,
    ) -> tf.Tensor:
        return tf.cast(
            self._det(
                h
            ),
            tf.float32,
        )

    def apply(
        self,
        h: tf.Tensor,
        residual: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
    ]:
        h = tf.cast(
            h,
            tf.float32,
        )

        if (
            self.kind
            == "identity"
        ):
            return (
                h,
                residual,
            )

        if (
            self.kind
            == "deterministic"
        ):
            return (
                self.deterministic_quantize(
                    h
                ),
                residual,
            )

        if (
            self.kind
            == "stochastic"
        ):
            delta = tf.constant(
                self.delta,
                tf.float32,
            )

            clipped = (
                tf.clip_by_value(
                    h,
                    self.qmin,
                    self.qmax,
                )
            )

            low = (
                tf.math.floor(
                    clipped
                    / delta
                )
                * delta
            )

            frac = (
                (
                    clipped
                    - low
                )
                / delta
            )

            uniform = (
                tf.random.stateless_uniform(
                    tf.shape(
                        clipped
                    ),
                    seed=seed_pair,
                    minval=0.0,
                    maxval=1.0,
                    dtype=tf.float32,
                )
            )

            up = tf.cast(
                uniform
                < frac,
                tf.float32,
            )

            q = (
                tf.clip_by_value(
                    low
                    + up
                    * delta,
                    self.qmin,
                    self.qmax,
                )
            )

            return (
                q,
                residual,
            )

        delta = tf.constant(
            self.delta,
            tf.float32,
        )

        compensated = (
            h
            + residual
        )

        q = (
            self.deterministic_quantize(
                compensated
            )
        )

        new_residual = (
            tf.clip_by_value(
                compensated
                - q,
                -delta,
                delta,
            )
        )

        return (
            q,
            new_residual,
        )


def resolve_single_writeback(
    requested: str,
    requested_bits: int,
    phase: str,
    cfg: Dict,
):
    if requested == "native":
        kind = (
            "deterministic"
            if NATIVE_QUANTIZED_STATE[
                phase
            ]
            else "identity"
        )

    elif requested == "none":
        kind = "identity"

    else:
        kind = requested

    bits = (
        cfg[
            "bits_state"
        ]
        if requested_bits
        == -1
        else int(
            requested_bits
        )
    )

    return (
        kind,
        bits,
    )


def native_writeback_spec(
    phase: str,
    cfg: Dict,
):
    kind = (
        "deterministic"
        if NATIVE_QUANTIZED_STATE[
            phase
        ]
        else "identity"
    )

    return (
        kind,
        cfg["bits_state"],
        kind,
        cfg["bits_state"],
    )


def build_forward_fn(
    effective_weights: Dict[
        str,
        np.ndarray,
    ],
    quantizers: Dict,
    enc_operator: WritebackOperator,
    dec_operator: WritebackOperator,
    cfg: Dict,
):
    seq_len = cfg[
        "seq_len"
    ]

    H = cfg[
        "student_units"
    ]

    enc_kernel = tf.constant(
        effective_weights[
            "enc_kernel"
        ],
        tf.float32,
    )

    enc_recurrent = tf.constant(
        effective_weights[
            "enc_recurrent"
        ],
        tf.float32,
    )

    enc_bias = tf.constant(
        effective_weights[
            "enc_bias"
        ],
        tf.float32,
    )

    dec_kernel = tf.constant(
        effective_weights[
            "dec_kernel"
        ],
        tf.float32,
    )

    dec_recurrent = tf.constant(
        effective_weights[
            "dec_recurrent"
        ],
        tf.float32,
    )

    dec_bias = tf.constant(
        effective_weights[
            "dec_bias"
        ],
        tf.float32,
    )

    dense_kernel = tf.constant(
        effective_weights[
            "dense_kernel"
        ],
        tf.float32,
    )

    dense_bias = tf.constant(
        effective_weights[
            "dense_bias"
        ],
        tf.float32,
    )

    q_activation = quantizers[
        "activation"
    ]

    def gru_step(
        x_t,
        q_recv,
        kernel_q,
        recurrent_q,
        bias_q,
    ):
        x_z = (
            tf.matmul(
                x_t,
                kernel_q[
                    :,
                    :H,
                ],
            )
            + bias_q[
                :H
            ]
        )

        x_r = (
            tf.matmul(
                x_t,
                kernel_q[
                    :,
                    H:
                    2 * H,
                ],
            )
            + bias_q[
                H:
                2 * H
            ]
        )

        x_h = (
            tf.matmul(
                x_t,
                kernel_q[
                    :,
                    2 * H:
                ],
            )
            + bias_q[
                2 * H:
            ]
        )

        z = (
            qkeras_hard_sigmoid(
                x_z
                + tf.matmul(
                    q_recv,
                    recurrent_q[
                        :,
                        :H,
                    ],
                )
            )
        )

        r = (
            qkeras_hard_sigmoid(
                x_r
                + tf.matmul(
                    q_recv,
                    recurrent_q[
                        :,
                        H:
                        2 * H,
                    ],
                )
            )
        )

        preact = (
            x_h
            + tf.matmul(
                r
                * q_recv,
                recurrent_q[
                    :,
                    2 * H:
                ],
            )
        )

        if q_activation is None:
            candidate = (
                tf.tanh(
                    preact
                )
            )

        else:
            candidate = (
                tf.cast(
                    q_activation(
                        preact
                    ),
                    tf.float32,
                )
            )

        h = (
            z
            * q_recv
            + (
                1.0
                - z
            )
            * candidate
        )

        return (
            h,
            z,
            r,
            candidate,
        )

    @tf.function(
        reduce_retracing=True
    )
    def forward(
        enc_inputs: tf.Tensor,
        sr_seed: tf.Tensor,
        batch_ordinal: tf.Tensor,
    ):
        enc_inputs = tf.cast(
            enc_inputs,
            tf.float32,
        )

        batch = tf.shape(
            enc_inputs
        )[0]

        h_prev = tf.zeros(
            (
                batch,
                H,
            ),
            tf.float32,
        )

        enc_residual = tf.zeros(
            (
                batch,
                H,
            ),
            tf.float32,
        )

        enc_h_steps = []
        enc_q_steps = []
        enc_z_steps = []
        enc_r_steps = []
        enc_c_steps = []

        base_counter = (
            batch_ordinal
            * tf.constant(
                4
                * seq_len
                + 16,
                tf.int32,
            )
        )

        for t in range(
            seq_len
        ):
            seed_pair = (
                tf.stack(
                    [
                        sr_seed,
                        base_counter
                        + tf.constant(
                            t,
                            tf.int32,
                        ),
                    ]
                )
            )

            (
                q_recv,
                enc_residual,
            ) = enc_operator.apply(
                h_prev,
                enc_residual,
                seed_pair,
            )

            (
                h_prev,
                z,
                r,
                candidate,
            ) = gru_step(
                enc_inputs[
                    :,
                    t,
                    :,
                ],
                q_recv,
                enc_kernel,
                enc_recurrent,
                enc_bias,
            )

            enc_q_steps.append(
                q_recv
            )

            enc_h_steps.append(
                h_prev
            )

            enc_z_steps.append(
                z
            )

            enc_r_steps.append(
                r
            )

            enc_c_steps.append(
                candidate
            )

        enc_final_raw = (
            h_prev
        )

        dec_residual = tf.zeros(
            (
                batch,
                H,
            ),
            tf.float32,
        )

        dec_h_steps = []
        dec_q_steps = []
        dec_z_steps = []
        dec_r_steps = []
        dec_c_steps = []

        dec_x = tf.zeros(
            (
                batch,
                1,
            ),
            tf.float32,
        )

        for t in range(
            seq_len
        ):
            seed_pair = (
                tf.stack(
                    [
                        sr_seed
                        + tf.constant(
                            1000003,
                            tf.int32,
                        ),
                        base_counter
                        + tf.constant(
                            2
                            * seq_len
                            + t,
                            tf.int32,
                        ),
                    ]
                )
            )

            (
                q_recv,
                dec_residual,
            ) = dec_operator.apply(
                h_prev,
                dec_residual,
                seed_pair,
            )

            (
                h_prev,
                z,
                r,
                candidate,
            ) = gru_step(
                dec_x,
                q_recv,
                dec_kernel,
                dec_recurrent,
                dec_bias,
            )

            dec_q_steps.append(
                q_recv
            )

            dec_h_steps.append(
                h_prev
            )

            dec_z_steps.append(
                z
            )

            dec_r_steps.append(
                r
            )

            dec_c_steps.append(
                candidate
            )

        enc_h = tf.stack(
            enc_h_steps,
            axis=1,
        )

        enc_q = tf.stack(
            enc_q_steps,
            axis=1,
        )

        enc_z = tf.stack(
            enc_z_steps,
            axis=1,
        )

        enc_r = tf.stack(
            enc_r_steps,
            axis=1,
        )

        enc_c = tf.stack(
            enc_c_steps,
            axis=1,
        )

        dec_h = tf.stack(
            dec_h_steps,
            axis=1,
        )

        dec_q = tf.stack(
            dec_q_steps,
            axis=1,
        )

        dec_z = tf.stack(
            dec_z_steps,
            axis=1,
        )

        dec_r = tf.stack(
            dec_r_steps,
            axis=1,
        )

        dec_c = tf.stack(
            dec_c_steps,
            axis=1,
        )

        preds = (
            tf.matmul(
                dec_h,
                dense_kernel,
            )
            + dense_bias
        )

        return (
            preds,
            enc_h,
            enc_q,
            enc_z,
            enc_r,
            enc_c,
            dec_h,
            dec_q,
            dec_z,
            dec_r,
            dec_c,
            enc_final_raw,
        )

    return forward


def tensor_error(
    name: str,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    tolerance: float,
) -> Dict:
    reference = np.asarray(
        reference,
        dtype=np.float32,
    )

    reconstructed = np.asarray(
        reconstructed,
        dtype=np.float32,
    )

    if (
        reference.shape
        != reconstructed.shape
    ):
        raise RuntimeError(
            "Equivalence shape mismatch "
            f"for {name}: "
            f"{reference.shape} vs "
            f"{reconstructed.shape}"
        )

    diff = np.abs(
        reference
        - reconstructed
    ).astype(
        np.float64
    )

    n_elements = int(
        diff.size
    )

    n_mismatch = int(
        np.count_nonzero(
            diff
            > float(
                tolerance
            )
        )
    )

    return {
        "name": name,
        "shape": list(
            reference.shape
        ),
        "max_abs": float(
            np.max(
                diff
            )
        ),
        "mean_abs": float(
            np.mean(
                diff
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    diff
                    ** 2
                )
            )
        ),
        "tolerance": float(
            tolerance
        ),
        "n_elements": n_elements,
        "n_mismatch": n_mismatch,
        "mismatch_fraction": float(
            n_mismatch
            / n_elements
        ),
    }


def run_tensor_equivalence(
    reference_model: Model,
    forward,
    normalized_input: np.ndarray,
    test_idx: np.ndarray,
    cfg: Dict,
    effective_weights: Dict[
        str,
        np.ndarray,
    ],
    n_samples: int,
    tolerance: float,
    mean_tolerance: float,
    mismatch_fraction_limit: float,
    tie_fraction_limit: float,
) -> Dict:
    n = min(
        int(
            n_samples
        ),
        len(
            test_idx
        ),
    )

    rows = test_idx[
        :n
    ]

    enc = np.asarray(
        normalized_input[
            rows
        ],
        dtype=np.float32,
    )

    dec = np.zeros(
        (
            n,
            cfg[
                "seq_len"
            ],
            1,
        ),
        dtype=np.float32,
    )

    reference_outputs = (
        reference_model(
            [
                tf.convert_to_tensor(
                    enc,
                    tf.float32,
                ),
                tf.convert_to_tensor(
                    dec,
                    tf.float32,
                ),
            ],
            training=False,
        )
    )

    (
        ref_pred,
        ref_dec_hidden,
        ref_enc_final,
    ) = [
        np.asarray(
            tensor.numpy(),
            dtype=np.float32,
        )
        for tensor
        in reference_outputs
    ]

    outputs = forward(
        tf.convert_to_tensor(
            enc,
            tf.float32,
        ),
        tf.constant(
            12345,
            tf.int32,
        ),
        tf.constant(
            0,
            tf.int32,
        ),
    )

    rec_pred = np.asarray(
        outputs[
            0
        ].numpy(),
        dtype=np.float32,
    )

    rec_dec_hidden = (
        np.asarray(
            outputs[
                6
            ].numpy(),
            dtype=np.float32,
        )
    )

    rec_enc_final = (
        np.asarray(
            outputs[
                11
            ].numpy(),
            dtype=np.float32,
        )
    )

    dense_kernel = np.asarray(
        effective_weights[
            "dense_kernel"
        ],
        dtype=np.float32,
    )

    dense_bias = np.asarray(
        effective_weights[
            "dense_bias"
        ],
        dtype=np.float32,
    )

    dense_probe = (
        np.matmul(
            ref_dec_hidden,
            dense_kernel,
        )
        + dense_bias
    ).astype(
        np.float32
    )

    dense_l1_gain = float(
        np.max(
            np.sum(
                np.abs(
                    dense_kernel
                ),
                axis=0,
            )
        )
    )

    pf(
        f"[EQUIV] dense L1 column "
        f"gain = {dense_l1_gain:.9g} "
        "(output discrepancy is the "
        "certified hidden discrepancy "
        "amplified by up to this "
        "factor)"
    )

    checks = [
        tensor_error(
            "output",
            ref_pred,
            rec_pred,
            tolerance,
        ),
        tensor_error(
            "decoder_hidden",
            ref_dec_hidden,
            rec_dec_hidden,
            tolerance,
        ),
        tensor_error(
            "encoder_final_state",
            ref_enc_final,
            rec_enc_final,
            tolerance,
        ),
        tensor_error(
            "dense_semantics_probe",
            ref_pred,
            dense_probe,
            tolerance,
        ),
    ]

    activation_lsb = float(
        2.0
        ** (
            -(
                int(
                    cfg[
                        "bits_activation"
                    ]
                )
                - 1
            )
        )
    )

    state_lsb = float(
        2.0
        ** (
            -(
                int(
                    cfg[
                        "bits_state"
                    ]
                )
                - 1
            )
        )
    )

    tie_bound = max(
        activation_lsb,
        state_lsb,
    )

    pf(
        f"[EQUIV] single-quantizer-step "
        f"tie bound = {tie_bound:.9g} "
        f"(activation LSB "
        f"{activation_lsb:.9g}, state "
        f"LSB {state_lsb:.9g}); "
        "isolated boundary ties within "
        "this bound are accepted at a "
        "fraction of at most "
        f"{tie_fraction_limit:.9g}"
    )

    failed = []

    for row in checks:
        row[
            "gated"
        ] = True

        row[
            "tie_bound"
        ] = tie_bound

        row[
            "tie_fraction_limit"
        ] = float(
            tie_fraction_limit
        )

        if (
            row[
                "max_abs"
            ]
            <= tolerance
        ):
            row[
                "criterion"
            ] = "max_abs"

            continue

        if (
            row[
                "max_abs"
            ]
            <= tie_bound
            and row[
                "mismatch_fraction"
            ]
            <= tie_fraction_limit
        ):
            row[
                "criterion"
            ] = (
                "isolated_quantizer_ties"
            )

            continue

        row[
            "criterion"
        ] = "failed"

        failed.append(
            row
        )

    for row in checks:
        pf(
            f"[EQUIV] "
            f"{row['name']}: "
            f"max="
            f"{row['max_abs']:.9g} "
            f"mean="
            f"{row['mean_abs']:.9g} "
            f"rmse="
            f"{row['rmse']:.9g} "
            f"mismatch_fraction="
            f"{row['mismatch_fraction']:.9g} "
            f"(n_mismatch="
            f"{row['n_mismatch']}"
            f"/"
            f"{row['n_elements']}) "
            f"criterion="
            f"{row['criterion']}"
        )

    if failed:
        raise RuntimeError(
            "Tensor equivalence FAILED: "
            + "; ".join(
                (
                    f"{row['name']} "
                    f"max_abs="
                    f"{row['max_abs']:.9g} "
                    f"> "
                    f"{tolerance:.9g} "
                    f"and not an isolated "
                    f"quantizer tie "
                    f"(tie bound "
                    f"{tie_bound:.9g}, "
                    f"mismatch_fraction="
                    f"{row['mismatch_fraction']:.9g} "
                    f"> "
                    f"{tie_fraction_limit:.9g}) "
                    f"(mean_abs="
                    f"{row['mean_abs']:.9g})"
                )
                for row in failed
            )
        )

    pf(
        f"[EQUIV] PASS on {n} "
        "held-out sequences with "
        f"maximum absolute tolerance "
        f"{tolerance:.9g} on output, "
        "decoder hidden sequence, "
        "encoder final state and the "
        "dense semantics probe; any "
        "residual differences are "
        "isolated single-quantizer-step "
        "boundary ties bounded by "
        f"{tie_bound:.9g}"
    )

    return {
        "n_samples": int(
            n
        ),
        "tolerance": float(
            tolerance
        ),
        "mean_tolerance": float(
            mean_tolerance
        ),
        "mismatch_fraction_limit": float(
            mismatch_fraction_limit
        ),
        "tie_fraction_limit": float(
            tie_fraction_limit
        ),
        "tie_bound": tie_bound,
        "activation_lsb": activation_lsb,
        "state_lsb": state_lsb,
        "dense_l1_gain": (
            dense_l1_gain
        ),
        "checks": checks,
        "passed": True,
    }


def extract_lifetimes(
    seq_arr: np.ndarray,
    t_axis: np.ndarray,
):
    seq_arr = np.asarray(
        seq_arr,
        dtype=np.float32,
    )

    ch1 = seq_arr[
        :,
        :,
        1,
    ]

    ch2 = seq_arr[
        :,
        :,
        2,
    ]

    int1 = np_trapz(
        ch1,
        t_axis,
        axis=1,
    )

    int2 = np_trapz(
        ch2,
        t_axis,
        axis=1,
    )

    amp1 = ch1[
        :,
        0,
    ]

    amp2 = ch2[
        :,
        0,
    ]

    tau1 = np.where(
        amp1
        > AMP_FLOOR,
        int1
        / amp1,
        0.0,
    ).astype(
        np.float32
    )

    tau2 = np.where(
        amp2
        > AMP_FLOOR,
        int2
        / amp2,
        0.0,
    ).astype(
        np.float32
    )

    denom = (
        amp1
        + amp2
    )

    fret = np.where(
        denom
        > AMP_FLOOR,
        amp1
        / denom,
        0.5,
    ).astype(
        np.float32
    )

    return (
        tau1,
        tau2,
        fret,
    )


def compute_accuracy_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
):
    mask = (
        np.isfinite(
            gt
        )
        & np.isfinite(
            pred
        )
    )

    n = int(
        mask.sum()
    )

    if n < 5:
        return (
            float(
                "nan"
            ),
            float(
                "nan"
            ),
            n,
        )

    g = np.asarray(
        gt[
            mask
        ],
        dtype=np.float64,
    )

    p = np.asarray(
        pred[
            mask
        ],
        dtype=np.float64,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    p
                    - g
                )
                ** 2
            )
        )
    )

    r = float(
        pearsonr(
            g,
            p,
        )[0]
    )

    return (
        rmse,
        r,
        n,
    )


def histogram_quantile(
    counts: np.ndarray,
    edges: np.ndarray,
    q: float,
) -> float:
    total = int(
        counts.sum()
    )

    if total <= 0:
        return float(
            "nan"
        )

    target = (
        q
        * total
    )

    cumulative = (
        np.cumsum(
            counts,
            dtype=np.int64,
        )
    )

    idx = int(
        np.searchsorted(
            cumulative,
            target,
            side="left",
        )
    )

    idx = min(
        max(
            idx,
            0,
        ),
        len(
            counts
        )
        - 1,
    )

    before = (
        int(
            cumulative[
                idx
                - 1
            ]
        )
        if idx > 0
        else 0
    )

    in_bin = int(
        counts[
            idx
        ]
    )

    left = float(
        edges[
            idx
        ]
    )

    right = float(
        edges[
            idx
            + 1
        ]
    )

    if in_bin <= 0:
        return (
            0.5
            * (
                left
                + right
            )
        )

    fraction = (
        (
            target
            - before
        )
        / in_bin
    )

    return (
        left
        + min(
            max(
                float(
                    fraction
                ),
                0.0,
            ),
            1.0,
        )
        * (
            right
            - left
        )
    )


def bootstrap_pooled_ratio(
    per_sequence_counts: np.ndarray,
    denominator_per_sequence: int,
    reps: int,
    seed: int,
    batch_reps: int,
):
    counts = np.asarray(
        per_sequence_counts,
        dtype=np.float64,
    )

    n = len(
        counts
    )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    values = np.empty(
        reps,
        dtype=np.float64,
    )

    done = 0

    while done < reps:
        take = min(
            batch_reps,
            reps
            - done,
        )

        idx = rng.integers(
            0,
            n,
            size=(
                take,
                n,
            ),
        )

        values[
            done:
            done
            + take
        ] = (
            counts[
                idx
            ].mean(
                axis=1
            )
            / float(
                denominator_per_sequence
            )
        )

        done += take

    return (
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


def numpy_project_to_grid(
    values: np.ndarray,
    bits: int,
) -> np.ndarray:
    delta = float(
        2.0
        ** (
            -(
                bits
                - 1
            )
        )
    )

    qmax = (
        1.0
        - delta
    )

    clipped = np.clip(
        values,
        -qmax,
        qmax,
    )

    codes = np.rint(
        clipped
        / delta
    )

    return (
        codes
        * delta
    ).astype(
        np.float32
    )


def grid_codes(
    values: np.ndarray,
    bits: int,
) -> np.ndarray:
    delta = float(
        2.0
        ** (
            -(
                bits
                - 1
            )
        )
    )

    qmax = (
        1.0
        - delta
    )

    qmin = (
        -qmax
    )

    n_levels = (
        2
        ** bits
    ) - 1

    codes = np.rint(
        (
            np.clip(
                values,
                qmin,
                qmax,
            )
            - qmin
        )
        / delta
    ).astype(
        np.int64
    )

    if (
        int(
            codes.min(
                initial=0
            )
        )
        < 0
        or int(
            codes.max(
                initial=0
            )
        )
        >= n_levels
    ):
        raise RuntimeError(
            "Grid code fell outside "
            "representable level range"
        )

    return codes


class RegionAccumulator:
    def __init__(
        self,
        name: str,
        n_sequences: int,
        seq_len: int,
        units: int,
        grid_bits: int,
        live_discrete: bool,
        hist_bins: int,
        gate_hist_bins: int,
    ):
        self.name = name

        self.n_sequences = int(
            n_sequences
        )

        self.seq_len = int(
            seq_len
        )

        self.units = int(
            units
        )

        self.grid_bits = int(
            grid_bits
        )

        self.live_discrete = bool(
            live_discrete
        )

        self.delta = float(
            2.0
            ** (
                -(
                    grid_bits
                    - 1
                )
            )
        )

        self.qmax = (
            1.0
            - self.delta
        )

        self.qmin = (
            -self.qmax
        )

        self.n_levels = (
            2
            ** grid_bits
        ) - 1

        self.aligned_steps = (
            self.seq_len
            - 1
        )

        self.aligned_per_sequence = (
            self.aligned_steps
            * self.units
        )

        self.i_hist_edges = (
            np.linspace(
                0.0,
                (
                    2.0
                    / self.delta
                )
                + GRID_EPS,
                hist_bins
                + 1,
                dtype=np.float64,
            )
        )

        self.i_hist_counts = (
            np.zeros(
                hist_bins,
                dtype=np.int64,
            )
        )

        self.w_levels = int(
            round(
                2.0
                / self.delta
            )
        )

        self.w_hist_counts = (
            np.zeros(
                self.w_levels
                + 1,
                dtype=np.int64,
            )
        )

        self.nwrite_hist_counts = (
            np.zeros(
                self.units
                + 1,
                dtype=np.int64,
            )
        )

        self.survival_total = (
            np.zeros(
                len(
                    SURVIVAL_EDGES
                )
                - 1,
                dtype=np.int64,
            )
        )

        self.survival_w0 = (
            np.zeros(
                len(
                    SURVIVAL_EDGES
                )
                - 1,
                dtype=np.int64,
            )
        )

        self.seq_w0_counts = (
            np.zeros(
                self.n_sequences,
                dtype=np.int64,
            )
        )

        self.seq_nwrite0_counts = (
            np.zeros(
                self.n_sequences,
                dtype=np.int64,
            )
        )

        self.unit_w0_counts = (
            np.zeros(
                self.units,
                dtype=np.int64,
            )
        )

        self.unit_i_subhalf_counts = (
            np.zeros(
                self.units,
                dtype=np.int64,
            )
        )

        self.occupancy_counts = (
            np.zeros(
                (
                    self.units,
                    self.n_levels,
                ),
                dtype=np.int64,
            )
        )

        self.rail_counts = (
            np.zeros(
                self.units,
                dtype=np.int64,
            )
        )

        self.z_hist_edges = (
            np.linspace(
                0.0,
                1.0
                + GRID_EPS,
                gate_hist_bins
                + 1,
            )
        )

        self.r_hist_edges = (
            np.linspace(
                0.0,
                1.0
                + GRID_EPS,
                gate_hist_bins
                + 1,
            )
        )

        self.c_hist_edges = (
            np.linspace(
                -1.0
                - GRID_EPS,
                1.0
                + GRID_EPS,
                gate_hist_bins
                + 1,
            )
        )

        self.z_hist_counts = (
            np.zeros(
                gate_hist_bins,
                dtype=np.int64,
            )
        )

        self.r_hist_counts = (
            np.zeros(
                gate_hist_bins,
                dtype=np.int64,
            )
        )

        self.c_hist_counts = (
            np.zeros(
                gate_hist_bins,
                dtype=np.int64,
            )
        )

        self.unit_z_sum = (
            np.zeros(
                self.units,
                dtype=np.float64,
            )
        )

        self.unit_r_sum = (
            np.zeros(
                self.units,
                dtype=np.float64,
            )
        )

        self.unit_candidate_abs_sum = (
            np.zeros(
                self.units,
                dtype=np.float64,
            )
        )

        self.unit_high_z_counts = (
            np.zeros(
                self.units,
                dtype=np.int64,
            )
        )

        self.total_aligned = 0
        self.total_w0 = 0
        self.total_nwrite0 = 0
        self.total_aligned_steps = 0
        self.total_i_subhalf = 0
        self.total_gate_values = 0

        self.observed_h_min = float(
            "inf"
        )

        self.observed_h_max = float(
            "-inf"
        )

        self.observed_i_max = 0.0

    def update(
        self,
        start,
        end,
        h,
        qrecv,
        z,
        r,
        candidate,
    ):
        batch = (
            end
            - start
        )

        h = np.asarray(
            h,
            dtype=np.float32,
        )

        qrecv = np.asarray(
            qrecv,
            dtype=np.float32,
        )

        z = np.asarray(
            z,
            dtype=np.float32,
        )

        r = np.asarray(
            r,
            dtype=np.float32,
        )

        candidate = np.asarray(
            candidate,
            dtype=np.float32,
        )

        expected = (
            batch,
            self.seq_len,
            self.units,
        )

        for name, arr in (
            (
                "h",
                h,
            ),
            (
                "qrecv",
                qrecv,
            ),
            (
                "z",
                z,
            ),
            (
                "r",
                r,
            ),
            (
                "candidate",
                candidate,
            ),
        ):
            if arr.shape != expected:
                raise RuntimeError(
                    f"{self.name} "
                    f"{name} shape "
                    f"{arr.shape} != "
                    f"{expected}"
                )

            if not np.all(
                np.isfinite(
                    arr
                )
            ):
                raise RuntimeError(
                    "Non-finite values "
                    f"in {self.name} "
                    f"{name}"
                )

        self.observed_h_min = min(
            self.observed_h_min,
            float(
                h.min()
            ),
        )

        self.observed_h_max = max(
            self.observed_h_max,
            float(
                h.max()
            ),
        )

        h_aligned = h[
            :,
            :-1,
            :,
        ]

        q_cur = qrecv[
            :,
            :-1,
            :,
        ]

        q_next = qrecv[
            :,
            1:,
            :,
        ]

        i_values = (
            np.abs(
                h_aligned
                - q_cur
            )
            / self.delta
        )

        self.observed_i_max = max(
            self.observed_i_max,
            float(
                i_values.max()
            ),
        )

        if (
            self.observed_i_max
            > float(
                self.i_hist_edges[
                    -1
                ]
            )
            + 1e-6
        ):
            raise RuntimeError(
                f"{self.name} I_t "
                "exceeded histogram "
                f"bound: "
                f"{self.observed_i_max}"
            )

        hist, _ = np.histogram(
            i_values,
            bins=self.i_hist_edges,
        )

        self.i_hist_counts += (
            hist.astype(
                np.int64
            )
        )

        i_sub = (
            i_values
            < 0.5
        )

        self.total_i_subhalf += int(
            i_sub.sum()
        )

        self.unit_i_subhalf_counts += (
            i_sub.sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.int64,
            )
        )

        self.total_aligned += int(
            batch
            * self.aligned_per_sequence
        )

        occupancy_values = (
            qrecv
            if self.live_discrete
            else numpy_project_to_grid(
                qrecv,
                self.grid_bits,
            )
        )

        codes = grid_codes(
            occupancy_values,
            self.grid_bits,
        )

        for unit in range(
            self.units
        ):
            self.occupancy_counts[
                unit
            ] += np.bincount(
                codes[
                    :,
                    :,
                    unit,
                ].reshape(
                    -1
                ),
                minlength=self.n_levels,
            )

        rail_mask = (
            np.isclose(
                np.abs(
                    occupancy_values
                ),
                self.qmax,
                atol=(
                    self.delta
                    * 1e-4
                ),
            )
        )

        self.rail_counts += (
            rail_mask.sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.int64,
            )
        )

        self.z_hist_counts += (
            np.histogram(
                z,
                bins=self.z_hist_edges,
            )[0].astype(
                np.int64
            )
        )

        self.r_hist_counts += (
            np.histogram(
                r,
                bins=self.r_hist_edges,
            )[0].astype(
                np.int64
            )
        )

        self.c_hist_counts += (
            np.histogram(
                candidate,
                bins=self.c_hist_edges,
            )[0].astype(
                np.int64
            )
        )

        self.unit_z_sum += (
            z.sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.float64,
            )
        )

        self.unit_r_sum += (
            r.sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.float64,
            )
        )

        self.unit_candidate_abs_sum += (
            np.abs(
                candidate
            ).sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.float64,
            )
        )

        self.unit_high_z_counts += (
            (
                z
                > 0.9
            ).sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.int64,
            )
        )

        self.total_gate_values += int(
            batch
            * self.seq_len
            * self.units
        )

        if not self.live_discrete:
            return

        w_values = (
            np.abs(
                q_next
                - q_cur
            )
            / self.delta
        )

        w_round = np.rint(
            w_values
        )

        max_dev = float(
            np.max(
                np.abs(
                    w_values
                    - w_round
                )
            )
        )

        if max_dev > 1e-3:
            raise RuntimeError(
                f"{self.name} W_t "
                "is not integer-valued "
                "on the grid; "
                f"max deviation="
                f"{max_dev}"
            )

        w_int = (
            w_round.astype(
                np.int64
            )
        )

        if (
            int(
                w_int.max(
                    initial=0
                )
            )
            > self.w_levels
        ):
            raise RuntimeError(
                f"{self.name} W "
                "level exceeds "
                "configured grid"
            )

        self.w_hist_counts += (
            np.bincount(
                w_int.reshape(
                    -1
                ),
                minlength=(
                    self.w_levels
                    + 1
                ),
            ).astype(
                np.int64
            )
        )

        w0 = (
            w_int
            == 0
        )

        seq_w0 = (
            w0.sum(
                axis=(
                    1,
                    2,
                ),
                dtype=np.int64,
            )
        )

        self.seq_w0_counts[
            start:
            end
        ] = seq_w0

        self.total_w0 += int(
            seq_w0.sum()
        )

        self.unit_w0_counts += (
            w0.sum(
                axis=(
                    0,
                    1,
                ),
                dtype=np.int64,
            )
        )

        edges = np.asarray(
            SURVIVAL_EDGES,
            dtype=np.float64,
        )

        bin_idx = (
            np.searchsorted(
                edges[
                    1:
                    -1
                ],
                i_values.reshape(
                    -1
                ),
                side="right",
            )
        )

        w0_flat = (
            w0.reshape(
                -1
            )
        )

        self.survival_total += (
            np.bincount(
                bin_idx,
                minlength=(
                    len(
                        SURVIVAL_EDGES
                    )
                    - 1
                ),
            ).astype(
                np.int64
            )
        )

        self.survival_w0 += (
            np.bincount(
                bin_idx[
                    w0_flat
                ],
                minlength=(
                    len(
                        SURVIVAL_EDGES
                    )
                    - 1
                ),
            ).astype(
                np.int64
            )
        )

        nwrite = (
            (
                ~w0
            ).sum(
                axis=2,
                dtype=np.int64,
            )
        )

        self.nwrite_hist_counts += (
            np.bincount(
                nwrite.reshape(
                    -1
                ),
                minlength=(
                    self.units
                    + 1
                ),
            ).astype(
                np.int64
            )
        )

        nwrite0 = (
            nwrite
            == 0
        )

        seq_nwrite0 = (
            nwrite0.sum(
                axis=1,
                dtype=np.int64,
            )
        )

        self.seq_nwrite0_counts[
            start:
            end
        ] = seq_nwrite0

        self.total_nwrite0 += int(
            seq_nwrite0.sum()
        )

        self.total_aligned_steps += int(
            batch
            * self.aligned_steps
        )

    def finalize(
        self,
    ):
        expected = (
            self.n_sequences
            * self.aligned_per_sequence
        )

        if (
            self.total_aligned
            != expected
        ):
            raise RuntimeError(
                f"{self.name} aligned "
                "transition count "
                f"{self.total_aligned} "
                f"!= {expected}"
            )

        if (
            int(
                self.i_hist_counts.sum()
            )
            != expected
        ):
            raise RuntimeError(
                f"{self.name} I "
                "histogram count "
                "mismatch"
            )

        expected_gate = (
            self.n_sequences
            * self.seq_len
            * self.units
        )

        if (
            self.total_gate_values
            != expected_gate
        ):
            raise RuntimeError(
                f"{self.name} gate "
                "count mismatch"
            )

        if (
            int(
                self.occupancy_counts.sum()
            )
            != expected_gate
        ):
            raise RuntimeError(
                f"{self.name} occupancy "
                "count mismatch"
            )

        if self.live_discrete:
            if (
                int(
                    self.w_hist_counts.sum()
                )
                != expected
            ):
                raise RuntimeError(
                    f"{self.name} W "
                    "histogram count "
                    "mismatch"
                )

            expected_steps = (
                self.n_sequences
                * self.aligned_steps
            )

            if (
                self.total_aligned_steps
                != expected_steps
            ):
                raise RuntimeError(
                    f"{self.name} N_write "
                    "step count mismatch"
                )

            if (
                int(
                    self.nwrite_hist_counts.sum()
                )
                != expected_steps
            ):
                raise RuntimeError(
                    f"{self.name} N_write "
                    "histogram count "
                    "mismatch"
                )

    def occupancy_stats(
        self,
    ):
        entropy = np.zeros(
            self.units,
            dtype=np.float64,
        )

        neff = np.zeros(
            self.units,
            dtype=np.float64,
        )

        occupied = np.zeros(
            self.units,
            dtype=np.int64,
        )

        for unit in range(
            self.units
        ):
            counts = (
                self.occupancy_counts[
                    unit
                ].astype(
                    np.float64
                )
            )

            total = (
                counts.sum()
            )

            p = (
                counts[
                    counts
                    > 0
                ]
                / total
            )

            entropy[
                unit
            ] = (
                -np.sum(
                    p
                    * np.log(
                        p
                    )
                )
            )

            neff[
                unit
            ] = math.exp(
                entropy[
                    unit
                ]
            )

            occupied[
                unit
            ] = int(
                np.count_nonzero(
                    counts
                )
            )

        return (
            entropy,
            neff,
            occupied,
        )

    def summary(
        self,
        bootstrap_reps: int,
        bootstrap_seed: int,
        bootstrap_batch_reps: int,
    ) -> Dict:
        (
            entropy,
            neff,
            occupied,
        ) = self.occupancy_stats()

        aligned = float(
            self.total_aligned
        )

        result = {
            "name": self.name,
            "grid_bits": (
                self.grid_bits
            ),
            "delta": (
                self.delta
            ),
            "live_discrete_writeback": (
                self.live_discrete
            ),
            "occupancy_mode": (
                "live_recurrence_visible"
                if self.live_discrete
                else "projected_counterfactual"
            ),
            "n_sequences": (
                self.n_sequences
            ),
            "seq_len": (
                self.seq_len
            ),
            "aligned_reused_steps_per_sequence": (
                self.aligned_steps
            ),
            "aligned_state_element_transitions": (
                self.total_aligned
            ),
            "terminal_state_excluded_from_W": True,
            "observed_h_min": (
                json_safe(
                    self.observed_h_min
                )
            ),
            "observed_h_max": (
                json_safe(
                    self.observed_h_max
                )
            ),
            "observed_i_max": (
                json_safe(
                    self.observed_i_max
                )
            ),
            "i_subhalf_fraction": (
                self.total_i_subhalf
                / aligned
            ),
            "i_median": (
                histogram_quantile(
                    self.i_hist_counts,
                    self.i_hist_edges,
                    0.5,
                )
            ),
            "i_p90": (
                histogram_quantile(
                    self.i_hist_counts,
                    self.i_hist_edges,
                    0.9,
                )
            ),
            "i_p99": (
                histogram_quantile(
                    self.i_hist_counts,
                    self.i_hist_edges,
                    0.99,
                )
            ),
            "z_mean": float(
                self.unit_z_sum.sum()
                / self.total_gate_values
            ),
            "r_mean": float(
                self.unit_r_sum.sum()
                / self.total_gate_values
            ),
            "high_z_fraction": float(
                self.unit_high_z_counts.sum()
                / self.total_gate_values
            ),
            "candidate_abs_mean": float(
                self.unit_candidate_abs_sum.sum()
                / self.total_gate_values
            ),
            "occupancy_entropy_mean": float(
                np.mean(
                    entropy
                )
            ),
            "effective_levels_mean": float(
                np.mean(
                    neff
                )
            ),
            "effective_levels_median": float(
                np.median(
                    neff
                )
            ),
            "occupied_levels_mean": float(
                np.mean(
                    occupied
                )
            ),
            "rail_fraction": float(
                self.rail_counts.sum()
                / self.total_gate_values
            ),
        }

        if not self.live_discrete:
            result.update(
                {
                    "p_w0": None,
                    "p_w0_ci95": None,
                    "p_nwrite0": None,
                    "p_nwrite0_ci95": None,
                    "mean_w": None,
                    "mean_nwrite": None,
                    "median_nwrite": None,
                    "survival": None,
                }
            )

            return result

        p_w0 = (
            self.total_w0
            / aligned
        )

        step_total = float(
            self.n_sequences
            * self.aligned_steps
        )

        p_nwrite0 = (
            self.total_nwrite0
            / step_total
        )

        w_levels = np.arange(
            len(
                self.w_hist_counts
            ),
            dtype=np.float64,
        )

        mean_w = float(
            np.sum(
                w_levels
                * self.w_hist_counts
            )
            / aligned
        )

        nwrite_levels = np.arange(
            len(
                self.nwrite_hist_counts
            ),
            dtype=np.float64,
        )

        mean_nwrite = float(
            np.sum(
                nwrite_levels
                * self.nwrite_hist_counts
            )
            / step_total
        )

        nwrite_edges = (
            np.arange(
                len(
                    self.nwrite_hist_counts
                )
                + 1,
                dtype=np.float64,
            )
            - 0.5
        )

        median_nwrite = (
            histogram_quantile(
                self.nwrite_hist_counts,
                nwrite_edges,
                0.5,
            )
        )

        w0_ci = (
            bootstrap_pooled_ratio(
                self.seq_w0_counts,
                self.aligned_per_sequence,
                bootstrap_reps,
                bootstrap_seed,
                bootstrap_batch_reps,
            )
        )

        nw0_ci = (
            bootstrap_pooled_ratio(
                self.seq_nwrite0_counts,
                self.aligned_steps,
                bootstrap_reps,
                bootstrap_seed
                + 1,
                bootstrap_batch_reps,
            )
        )

        survival = []

        for b in range(
            len(
                SURVIVAL_EDGES
            )
            - 1
        ):
            total = int(
                self.survival_total[
                    b
                ]
            )

            erased = int(
                self.survival_w0[
                    b
                ]
            )

            survival.append(
                {
                    "i_low": (
                        json_safe(
                            SURVIVAL_EDGES[
                                b
                            ]
                        )
                    ),
                    "i_high": (
                        json_safe(
                            SURVIVAL_EDGES[
                                b
                                + 1
                            ]
                        )
                    ),
                    "n_transitions": total,
                    "n_w0": erased,
                    "p_w0_given_i": (
                        json_safe(
                            (
                                erased
                                / total
                            )
                            if total
                            else float(
                                "nan"
                            )
                        )
                    ),
                }
            )

        result.update(
            {
                "p_w0": p_w0,
                "p_w0_ci95": [
                    w0_ci[
                        0
                    ],
                    w0_ci[
                        1
                    ],
                ],
                "p_nwrite0": (
                    p_nwrite0
                ),
                "p_nwrite0_ci95": [
                    nw0_ci[
                        0
                    ],
                    nw0_ci[
                        1
                    ],
                ],
                "mean_w": (
                    mean_w
                ),
                "mean_nwrite": (
                    mean_nwrite
                ),
                "median_nwrite": (
                    median_nwrite
                ),
                "survival": (
                    survival
                ),
            }
        )

        return result


class HandoffAccumulator:
    def __init__(
        self,
        n_sequences: int,
        units: int,
        dec_operator: WritebackOperator,
    ):
        self.n_sequences = int(
            n_sequences
        )

        self.units = int(
            units
        )

        self.live_discrete = (
            dec_operator.live_discrete
        )

        self.bits = (
            dec_operator.state_bits
        )

        self.delta = (
            dec_operator.delta
        )

        self.qmax = (
            dec_operator.qmax
        )

        self.abs_error_sum = 0.0
        self.abs_error_max = 0.0
        self.zero_error_count = 0
        self.rail_count = 0
        self.total = 0

    def update(
        self,
        enc_final_raw: np.ndarray,
        dec_qrecv0: np.ndarray,
    ):
        raw = np.asarray(
            enc_final_raw,
            dtype=np.float32,
        )

        q = np.asarray(
            dec_qrecv0,
            dtype=np.float32,
        )

        err = np.abs(
            q
            - raw
        ).astype(
            np.float64
        )

        self.abs_error_sum += float(
            err.sum()
        )

        self.abs_error_max = max(
            self.abs_error_max,
            float(
                err.max()
            ),
        )

        self.zero_error_count += int(
            np.count_nonzero(
                err
                <= 1e-8
            )
        )

        self.rail_count += int(
            np.count_nonzero(
                np.isclose(
                    np.abs(
                        q
                    ),
                    self.qmax,
                    atol=(
                        self.delta
                        * 1e-4
                    ),
                )
            )
        )

        self.total += int(
            err.size
        )

    def summary(
        self,
    ) -> Dict:
        expected = (
            self.n_sequences
            * self.units
        )

        if self.total != expected:
            raise RuntimeError(
                "Encoder-to-decoder "
                "handoff count mismatch"
            )

        return {
            "decoder_writeback_live_discrete": (
                self.live_discrete
            ),
            "decoder_grid_bits": (
                self.bits
            ),
            "decoder_delta": (
                self.delta
            ),
            "n_state_elements": (
                self.total
            ),
            "mean_abs_handoff_quantization_error": (
                self.abs_error_sum
                / self.total
            ),
            "max_abs_handoff_quantization_error": (
                self.abs_error_max
            ),
            "zero_error_fraction": (
                self.zero_error_count
                / self.total
            ),
            "rail_fraction": (
                self.rail_count
                / self.total
            ),
        }


class PassAccumulator:
    def __init__(
        self,
        n_sequences: int,
        cfg: Dict,
        enc_operator: WritebackOperator,
        dec_operator: WritebackOperator,
        hist_bins: int,
        gate_hist_bins: int,
    ):
        self.n_sequences = int(
            n_sequences
        )

        self.cfg = cfg

        self.encoder = (
            RegionAccumulator(
                "encoder",
                n_sequences,
                cfg[
                    "seq_len"
                ],
                cfg[
                    "student_units"
                ],
                enc_operator.state_bits,
                enc_operator.live_discrete,
                hist_bins,
                gate_hist_bins,
            )
        )

        self.decoder = (
            RegionAccumulator(
                "decoder",
                n_sequences,
                cfg[
                    "seq_len"
                ],
                cfg[
                    "student_units"
                ],
                dec_operator.state_bits,
                dec_operator.live_discrete,
                hist_bins,
                gate_hist_bins,
            )
        )

        self.handoff = (
            HandoffAccumulator(
                n_sequences,
                cfg[
                    "student_units"
                ],
                dec_operator,
            )
        )

        self.tau1_pred = (
            np.zeros(
                n_sequences,
                dtype=np.float32,
            )
        )

        self.tau2_pred = (
            np.zeros(
                n_sequences,
                dtype=np.float32,
            )
        )

        self.fret_pred = (
            np.zeros(
                n_sequences,
                dtype=np.float32,
            )
        )

        self.seq_abs_error_sum = 0.0
        self.seq_value_count = 0

    def update(
        self,
        start,
        end,
        preds,
        target,
        enc_h,
        enc_q,
        enc_z,
        enc_r,
        enc_c,
        dec_h,
        dec_q,
        dec_z,
        dec_r,
        dec_c,
        enc_final_raw,
        t_axis,
    ):
        preds = np.asarray(
            preds,
            dtype=np.float32,
        )

        target = np.asarray(
            target,
            dtype=np.float32,
        )

        self.seq_abs_error_sum += float(
            np.abs(
                preds
                - target
            ).sum(
                dtype=np.float64
            )
        )

        self.seq_value_count += int(
            preds.size
        )

        (
            tau1,
            tau2,
            fret,
        ) = extract_lifetimes(
            preds,
            t_axis,
        )

        self.tau1_pred[
            start:
            end
        ] = tau1

        self.tau2_pred[
            start:
            end
        ] = tau2

        self.fret_pred[
            start:
            end
        ] = fret

        self.encoder.update(
            start,
            end,
            enc_h,
            enc_q,
            enc_z,
            enc_r,
            enc_c,
        )

        self.decoder.update(
            start,
            end,
            dec_h,
            dec_q,
            dec_z,
            dec_r,
            dec_c,
        )

        self.handoff.update(
            enc_final_raw,
            np.asarray(
                dec_q,
                dtype=np.float32,
            )[
                :,
                0,
                :,
            ],
        )

    def finalize(
        self,
    ):
        self.encoder.finalize()
        self.decoder.finalize()

        if (
            self.seq_value_count
            <= 0
        ):
            raise RuntimeError(
                "No prediction values "
                "accumulated"
            )


def run_condition_pass(
    condition_name: str,
    forward,
    enc_operator: WritebackOperator,
    dec_operator: WritebackOperator,
    cfg: Dict,
    normalized_input: np.ndarray,
    res_data: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    infer_batch: int,
    hist_bins: int,
    gate_hist_bins: int,
    gate_width_ns: float,
    sr_seed: int,
):
    n_sequences = len(
        test_idx
    )

    accumulator = PassAccumulator(
        n_sequences,
        cfg,
        enc_operator,
        dec_operator,
        hist_bins,
        gate_hist_bins,
    )

    t_axis = (
        np.arange(
            cfg[
                "seq_len"
            ],
            dtype=np.float32,
        )
        * float(
            gate_width_ns
        )
    )

    n_batches = math.ceil(
        n_sequences
        / infer_batch
    )

    started = time.time()

    pf(
        f"[{condition_name}] start: "
        f"N={n_sequences} "
        f"batches={n_batches} "
        f"enc="
        f"{enc_operator.kind}/"
        f"B{enc_operator.state_bits} "
        f"dec="
        f"{dec_operator.kind}/"
        f"B{dec_operator.state_bits} "
        f"seed={sr_seed}"
    )

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

        rows = test_idx[
            start:
            end
        ]

        enc = np.asarray(
            normalized_input[
                rows
            ],
            dtype=np.float32,
        )

        target = np.asarray(
            res_data[
                rows
            ],
            dtype=np.float32,
        )

        outputs = forward(
            tf.convert_to_tensor(
                enc,
                tf.float32,
            ),
            tf.constant(
                sr_seed,
                tf.int32,
            ),
            tf.constant(
                start
                // infer_batch,
                tf.int32,
            ),
        )

        arrays = [
            np.asarray(
                tensor.numpy(),
                dtype=np.float32,
            )
            for tensor
            in outputs
        ]

        accumulator.update(
            start,
            end,
            arrays[0],
            target,
            arrays[1],
            arrays[2],
            arrays[3],
            arrays[4],
            arrays[5],
            arrays[6],
            arrays[7],
            arrays[8],
            arrays[9],
            arrays[10],
            arrays[11],
            t_axis,
        )

        if (
            batch_number
            == 1
            or batch_number
            % 10
            == 0
            or end
            == n_sequences
        ):
            elapsed = (
                time.time()
                - started
            )

            pf(
                f"[{condition_name}] "
                f"batch "
                f"{batch_number}/"
                f"{n_batches} "
                f"samples "
                f"{end}/"
                f"{n_sequences} "
                f"elapsed="
                f"{elapsed / 60.0:.1f} "
                "min"
            )

    accumulator.finalize()

    labels_test = (
        np.asarray(
            labels[
                test_idx
            ],
            dtype=np.float32,
        )
    )

    (
        rmse1,
        r1,
        n1,
    ) = compute_accuracy_metrics(
        labels_test[
            :,
            0,
        ],
        accumulator.tau1_pred,
    )

    (
        rmse2,
        r2,
        n2,
    ) = compute_accuracy_metrics(
        labels_test[
            :,
            1,
        ],
        accumulator.tau2_pred,
    )

    (
        rmsef,
        rf,
        nf,
    ) = compute_accuracy_metrics(
        labels_test[
            :,
            2,
        ],
        accumulator.fret_pred,
    )

    metrics = {
        "mae_seq": float(
            accumulator.seq_abs_error_sum
            / accumulator.seq_value_count
        ),
        "rmse_tau1": rmse1,
        "rmse_tau2": rmse2,
        "rmse_fret": rmsef,
        "r_tau1": r1,
        "r_tau2": r2,
        "r_fret": rf,
        "n_valid_tau1": n1,
        "n_valid_tau2": n2,
        "n_valid_fret": nf,
        "n_test": int(
            n_sequences
        ),
        "sr_seed": int(
            sr_seed
        ),
    }

    pf(
        f"[{condition_name}] "
        f"mae_seq="
        f"{metrics['mae_seq']:.9g} "
        f"rmse_tau1="
        f"{rmse1:.6g} "
        f"rmse_tau2="
        f"{rmse2:.6g} "
        f"r_tau1="
        f"{r1:.6g} "
        f"r_tau2="
        f"{r2:.6g}"
    )

    return (
        accumulator,
        metrics,
    )


def find_saved_metrics(
    run_dir: Path,
    phase: str,
):
    for name in (
        METRIC_FILE_CANDIDATES[
            phase
        ]
    ):
        path = (
            run_dir
            / name
        )

        if path.is_file():
            return (
                path,
                load_json(
                    path
                ),
            )

    raise FileNotFoundError(
        f"No saved metrics for "
        f"{phase} in {run_dir}; "
        f"tried "
        f"{METRIC_FILE_CANDIDATES[phase]}"
    )


def metric_from_saved(
    saved: Dict,
    key: str,
) -> Optional[float]:
    if (
        key in saved
        and isinstance(
            saved[
                key
            ],
            (
                int,
                float,
            ),
        )
    ):
        return float(
            saved[
                key
            ]
        )

    mapping = {
        "rmse_tau1": (
            "tau1",
            "rmse",
        ),
        "rmse_tau2": (
            "tau2",
            "rmse",
        ),
        "r_tau1": (
            "tau1",
            "r",
        ),
        "r_tau2": (
            "tau2",
            "r",
        ),
        "rmse_fret": (
            "fret",
            "rmse",
        ),
        "r_fret": (
            "fret",
            "r",
        ),
    }

    if key in mapping:
        (
            outer,
            inner,
        ) = mapping[
            key
        ]

        if (
            outer in saved
            and isinstance(
                saved[
                    outer
                ],
                dict,
            )
            and inner
            in saved[
                outer
            ]
        ):
            return float(
                saved[
                    outer
                ][
                    inner
                ]
            )

    return None


def validate_saved_metrics(
    phase: str,
    saved: Dict,
    recomputed: Dict,
    mae_tolerance: float,
    rmse_tolerance: float,
):
    report = {
        "phase": phase,
        "checks": [],
    }

    candidates = (
        (
            "mae_seq",
            mae_tolerance,
        ),
        (
            "rmse_tau1",
            rmse_tolerance,
        ),
        (
            "rmse_tau2",
            rmse_tolerance,
        ),
    )

    for (
        key,
        tolerance,
    ) in candidates:
        saved_value = (
            metric_from_saved(
                saved,
                key,
            )
        )

        if saved_value is None:
            continue

        recomputed_value = float(
            recomputed[
                key
            ]
        )

        diff = abs(
            recomputed_value
            - saved_value
        )

        row = {
            "metric": key,
            "saved": saved_value,
            "recomputed": (
                recomputed_value
            ),
            "abs_diff": diff,
            "tolerance": (
                tolerance
            ),
            "passed": (
                diff
                <= tolerance
            ),
        }

        report[
            "checks"
        ].append(
            row
        )

        pf(
            "[METRIC-FIDELITY] "
            f"{key}: "
            f"saved="
            f"{saved_value:.12g} "
            f"recomputed="
            f"{recomputed_value:.12g} "
            f"diff="
            f"{diff:.3g} "
            f"tol="
            f"{tolerance:.3g}"
        )

    if not report[
        "checks"
    ]:
        raise RuntimeError(
            "Saved metrics JSON "
            "exposes none of "
            "mae_seq/rmse_tau1/"
            "rmse_tau2"
        )

    failed = [
        row
        for row
        in report[
            "checks"
        ]
        if not row[
            "passed"
        ]
    ]

    if failed:
        raise RuntimeError(
            "Saved-metric fidelity "
            "FAILED: "
            + "; ".join(
                (
                    f"{row['metric']} "
                    f"diff="
                    f"{row['abs_diff']:.9g} "
                    f"> "
                    f"{row['tolerance']:.9g}"
                )
                for row
                in failed
            )
        )

    report[
        "passed"
    ] = True

    return report


def summarize_pass(
    accumulator: PassAccumulator,
    metrics: Dict,
    bootstrap_reps: int,
    bootstrap_seed: int,
    bootstrap_batch_reps: int,
):
    return {
        "metrics": {
            key: json_safe(
                value
            )
            for key, value
            in metrics.items()
        },
        "encoder": (
            accumulator.encoder.summary(
                bootstrap_reps,
                bootstrap_seed,
                bootstrap_batch_reps,
            )
        ),
        "decoder": (
            accumulator.decoder.summary(
                bootstrap_reps,
                bootstrap_seed
                + 10,
                bootstrap_batch_reps,
            )
        ),
        "handoff": (
            accumulator.handoff.summary()
        ),
    }


def write_region_outputs(
    out_dir: Path,
    region: RegionAccumulator,
    summary: Dict,
) -> None:
    prefix = region.name

    with (
        out_dir
        / f"{prefix}_I_histogram.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "bin_left",
                "bin_right",
                "count",
            ]
        )

        for i, count in enumerate(
            region.i_hist_counts
        ):
            writer.writerow(
                [
                    region.i_hist_edges[
                        i
                    ],
                    region.i_hist_edges[
                        i
                        + 1
                    ],
                    int(
                        count
                    ),
                ]
            )

    with (
        out_dir
        / f"{prefix}_gate_histograms.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "variable",
                "bin_left",
                "bin_right",
                "count",
            ]
        )

        for (
            name,
            edges,
            counts,
        ) in (
            (
                "z",
                region.z_hist_edges,
                region.z_hist_counts,
            ),
            (
                "r",
                region.r_hist_edges,
                region.r_hist_counts,
            ),
            (
                "candidate",
                region.c_hist_edges,
                region.c_hist_counts,
            ),
        ):
            for i, count in enumerate(
                counts
            ):
                writer.writerow(
                    [
                        name,
                        edges[
                            i
                        ],
                        edges[
                            i
                            + 1
                        ],
                        int(
                            count
                        ),
                    ]
                )

    with (
        out_dir
        / f"{prefix}_W_histogram.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "w_level",
                "count",
            ]
        )

        if region.live_discrete:
            for level, count in enumerate(
                region.w_hist_counts
            ):
                writer.writerow(
                    [
                        level,
                        int(
                            count
                        ),
                    ]
                )

    with (
        out_dir
        / f"{prefix}_nwrite_histogram.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "n_write",
                "count",
            ]
        )

        if region.live_discrete:
            for level, count in enumerate(
                region.nwrite_hist_counts
            ):
                writer.writerow(
                    [
                        level,
                        int(
                            count
                        ),
                    ]
                )

    with (
        out_dir
        / f"{prefix}_survival.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "i_low",
                "i_high",
                "n_transitions",
                "n_w0",
                "p_w0_given_i",
            ]
        )

        if (
            summary[
                "survival"
            ]
            is not None
        ):
            for row in (
                summary[
                    "survival"
                ]
            ):
                writer.writerow(
                    [
                        row[
                            "i_low"
                        ],
                        row[
                            "i_high"
                        ],
                        row[
                            "n_transitions"
                        ],
                        row[
                            "n_w0"
                        ],
                        row[
                            "p_w0_given_i"
                        ],
                    ]
                )

    (
        entropy,
        neff,
        occupied,
    ) = region.occupancy_stats()

    per_unit_den = float(
        region.n_sequences
        * region.aligned_steps
    )

    gate_den = float(
        region.n_sequences
        * region.seq_len
    )

    with (
        out_dir
        / f"{prefix}_per_unit.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "unit",
                "p_w0",
                "p_i_subhalf",
                "z_mean",
                "r_mean",
                "high_z_fraction",
                "candidate_abs_mean",
                "occupancy_entropy",
                "effective_levels",
                "occupied_levels",
                "rail_fraction",
            ]
        )

        for unit in range(
            region.units
        ):
            p_w0 = (
                region.unit_w0_counts[
                    unit
                ]
                / per_unit_den
                if region.live_discrete
                else ""
            )

            writer.writerow(
                [
                    unit,
                    p_w0,
                    (
                        region.unit_i_subhalf_counts[
                            unit
                        ]
                        / per_unit_den
                    ),
                    (
                        region.unit_z_sum[
                            unit
                        ]
                        / gate_den
                    ),
                    (
                        region.unit_r_sum[
                            unit
                        ]
                        / gate_den
                    ),
                    (
                        region.unit_high_z_counts[
                            unit
                        ]
                        / gate_den
                    ),
                    (
                        region.unit_candidate_abs_sum[
                            unit
                        ]
                        / gate_den
                    ),
                    entropy[
                        unit
                    ],
                    neff[
                        unit
                    ],
                    int(
                        occupied[
                            unit
                        ]
                    ),
                    (
                        region.rail_counts[
                            unit
                        ]
                        / gate_den
                    ),
                ]
            )


def write_per_sequence_npz(
    out_dir: Path,
    accumulator: PassAccumulator,
) -> None:
    np.savez_compressed(
        str(
            out_dir
            / "writeback_per_sequence.npz"
        ),
        encoder_seq_w0_counts=(
            accumulator.encoder.seq_w0_counts
        ),
        encoder_seq_nwrite0_counts=(
            accumulator.encoder.seq_nwrite0_counts
        ),
        decoder_seq_w0_counts=(
            accumulator.decoder.seq_w0_counts
        ),
        decoder_seq_nwrite0_counts=(
            accumulator.decoder.seq_nwrite0_counts
        ),
        encoder_aligned_steps=np.asarray(
            accumulator.encoder.aligned_steps,
            dtype=np.int64,
        ),
        decoder_aligned_steps=np.asarray(
            accumulator.decoder.aligned_steps,
            dtype=np.int64,
        ),
        units=np.asarray(
            accumulator.decoder.units,
            dtype=np.int64,
        ),
        tau1_pred=(
            accumulator.tau1_pred
        ),
        tau2_pred=(
            accumulator.tau2_pred
        ),
        fret_pred=(
            accumulator.fret_pred
        ),
    )


def cross_seed_summary(
    realizations: List[Dict],
) -> Optional[Dict]:
    if len(
        realizations
    ) <= 1:
        return None

    result = {}

    paths = (
        (
            "metrics.mae_seq",
            lambda row: (
                row[
                    "metrics"
                ][
                    "mae_seq"
                ]
            ),
        ),
        (
            "metrics.rmse_tau1",
            lambda row: (
                row[
                    "metrics"
                ][
                    "rmse_tau1"
                ]
            ),
        ),
        (
            "metrics.rmse_tau2",
            lambda row: (
                row[
                    "metrics"
                ][
                    "rmse_tau2"
                ]
            ),
        ),
        (
            "decoder.p_w0",
            lambda row: (
                row[
                    "decoder"
                ][
                    "p_w0"
                ]
            ),
        ),
        (
            "decoder.p_nwrite0",
            lambda row: (
                row[
                    "decoder"
                ][
                    "p_nwrite0"
                ]
            ),
        ),
        (
            "encoder.p_w0",
            lambda row: (
                row[
                    "encoder"
                ][
                    "p_w0"
                ]
            ),
        ),
        (
            "encoder.p_nwrite0",
            lambda row: (
                row[
                    "encoder"
                ][
                    "p_nwrite0"
                ]
            ),
        ),
    )

    for (
        name,
        getter,
    ) in paths:
        values = [
            getter(
                row
            )
            for row
            in realizations
        ]

        if any(
            value is None
            for value
            in values
        ):
            result[
                name
            ] = None

            continue

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        result[
            name
        ] = {
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std(
                    ddof=1
                )
            ),
            "values": (
                values.tolist()
            ),
        }

    return result


def validate_fidelity_file(
    fidelity_path: Path,
    checkpoint_path: Path,
    run_dir: Path,
    phase: str,
):
    if not fidelity_path.is_file():
        raise FileNotFoundError(
            "Required fidelity JSON "
            "does not exist: "
            f"{fidelity_path}"
        )

    payload = load_json(
        fidelity_path
    )

    expected = {
        "phase": phase,
        "checkpoint_sha256": (
            sha256_file(
                checkpoint_path
            )
        ),
        "student_args_sha256": (
            sha256_file(
                run_dir
                / "student_args.json"
            )
        ),
        "analysis_script_sha256": (
            sha256_file(
                THIS_FILE
            )
        ),
    }

    for key, value in (
        expected.items()
    ):
        if (
            payload.get(
                key
            )
            != value
        ):
            raise RuntimeError(
                "Fidelity JSON mismatch "
                f"for {key}: "
                f"file="
                f"{payload.get(key)!r} "
                f"current="
                f"{value!r}"
            )

    if (
        payload.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Fidelity JSON does not "
            "record passed=true"
        )

    pf(
        f"[FIDELITY] accepted "
        f"{fidelity_path}"
    )

    return payload


def build_manifest(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    cfg: Dict,
    enc_operator: WritebackOperator,
    dec_operator: WritebackOperator,
    elapsed_seconds: float,
):
    train_source = (
        REPO_ROOT
        / "train_student_memoq.py"
    )

    return {
        "mode": args.mode,
        "phase": args.phase,
        "run_dir": str(
            run_dir
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "checkpoint_sha256": (
            sha256_file(
                checkpoint_path
            )
        ),
        "student_args_sha256": (
            sha256_file(
                run_dir
                / "student_args.json"
            )
        ),
        "analysis_script_sha256": (
            sha256_file(
                THIS_FILE
            )
        ),
        "train_student_memoq_sha256": (
            sha256_file(
                train_source
            )
            if train_source.is_file()
            else None
        ),
        "config": cfg,
        "encoder_writeback": {
            "kind": (
                enc_operator.kind
            ),
            "bits": (
                enc_operator.state_bits
            ),
            "delta": (
                enc_operator.delta
            ),
        },
        "decoder_writeback": {
            "kind": (
                dec_operator.kind
            ),
            "bits": (
                dec_operator.state_bits
            ),
            "delta": (
                dec_operator.delta
            ),
        },
        "cli": {
            key: json_safe(
                value
            )
            for key, value
            in sorted(
                vars(
                    args
                ).items()
            )
        },
        "versions": {
            "python": (
                sys.version.split()[0]
            ),
            "numpy": (
                np.__version__
            ),
            "tensorflow": (
                tf.__version__
            ),
        },
        "error_feedback_boundary_semantics": (
            "decoder residual reset to zero "
            "at encoder-to-decoder boundary"
        ),
        "recurrent_write_definition": (
            "only h_t values consumed by a "
            "subsequent same-layer recurrent "
            "step enter W/N_write"
        ),
        "stochastic_rounding_reproducibility": (
            "fixed sr_seed and infer_batch are "
            "required; both are recorded"
        ),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "timestamp_unix": (
            time.time()
        ),
    }


def write_analysis_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    cfg: Dict,
    enc_operator: WritebackOperator,
    dec_operator: WritebackOperator,
    passes: List[
        Tuple[
            PassAccumulator,
            Dict,
            Dict,
        ]
    ],
    tensor_equivalence: Optional[Dict],
    metric_fidelity: Optional[Dict],
    elapsed_seconds: float,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    realizations = [
        summary
        for (
            _acc,
            _metrics,
            summary,
        )
        in passes
    ]

    payload = {
        "phase": args.phase,
        "mode": args.mode,
        "encoder_writeback": {
            "kind": (
                enc_operator.kind
            ),
            "bits": (
                enc_operator.state_bits
            ),
            "delta": (
                enc_operator.delta
            ),
        },
        "decoder_writeback": {
            "kind": (
                dec_operator.kind
            ),
            "bits": (
                dec_operator.state_bits
            ),
            "delta": (
                dec_operator.delta
            ),
        },
        "n_realizations": (
            len(
                realizations
            )
        ),
        "realizations": (
            realizations
        ),
        "cross_seed": (
            cross_seed_summary(
                realizations
            )
        ),
        "tensor_equivalence": (
            tensor_equivalence
        ),
        "saved_metric_fidelity": (
            metric_fidelity
        ),
        "bootstrap": {
            "reps": (
                args.bootstrap_reps
            ),
            "seed": (
                args.bootstrap_seed
            ),
            "batch_reps": (
                args.bootstrap_batch_reps
            ),
            "level": (
                "sequence"
            ),
            "note": (
                "Confidence intervals are "
                "held-out-set sampling "
                "uncertainty, not "
                "training-seed variability."
            ),
        },
    }

    atomic_write_json(
        out_dir
        / "writeback_summary.json",
        payload,
    )

    (
        first_acc,
        _first_metrics,
        first_summary,
    ) = passes[
        0
    ]

    write_region_outputs(
        out_dir,
        first_acc.encoder,
        first_summary[
            "encoder"
        ],
    )

    write_region_outputs(
        out_dir,
        first_acc.decoder,
        first_summary[
            "decoder"
        ],
    )

    write_per_sequence_npz(
        out_dir,
        first_acc,
    )

    manifest = build_manifest(
        args,
        run_dir,
        checkpoint_path,
        cfg,
        enc_operator,
        dec_operator,
        elapsed_seconds,
    )

    atomic_write_json(
        out_dir
        / "writeback_manifest.json",
        manifest,
    )

    if (
        args.mode
        == "native_fidelity"
    ):
        fidelity_payload = {
            "passed": True,
            "phase": (
                args.phase
            ),
            "run_dir": str(
                run_dir
            ),
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_sha256": (
                sha256_file(
                    checkpoint_path
                )
            ),
            "student_args_sha256": (
                sha256_file(
                    run_dir
                    / "student_args.json"
                )
            ),
            "analysis_script_sha256": (
                sha256_file(
                    THIS_FILE
                )
            ),
            "tensor_equivalence": (
                tensor_equivalence
            ),
            "saved_metric_fidelity": (
                metric_fidelity
            ),
        }

        atomic_write_json(
            out_dir
            / "native_fidelity.json",
            fidelity_payload,
        )

    with (
        out_dir
        / "writeback_complete.flag"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{args.phase} "
            f"{args.mode}\n"
        )

    pf(
        f"[OUT] results written "
        f"under {out_dir}"
    )


def main() -> None:
    started = time.time()

    args = parse_args()

    validate_cli(
        args
    )

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

    cfg = load_run_config(
        run_dir
    )

    checkpoint_path = (
        checkpoint_path_for(
            args,
            run_dir,
        )
    )

    pf(
        f"[CKPT] "
        f"{checkpoint_path}"
    )

    pf(
        f"[CKPT] SHA256 "
        f"{sha256_file(checkpoint_path)}"
    )

    (
        normalized_input,
        res_data,
        labels,
        test_idx,
    ) = load_dataset(
        data_dir,
        cfg,
    )

    (
        original_model,
        reference_model,
        enc_cell,
        dec_cell,
    ) = build_original_reference(
        cfg,
        args.phase,
        checkpoint_path,
    )

    raw_weights = (
        extract_raw_weights(
            original_model,
            cfg,
            args.phase,
            enc_cell,
            dec_cell,
        )
    )

    quantizers = (
        build_parameter_quantizers(
            original_model,
            cfg,
            args.phase,
            enc_cell,
            dec_cell,
        )
    )

    effective_weights = (
        quantize_effective_weights(
            raw_weights,
            quantizers,
        )
    )

    if (
        args.mode
        == "native_fidelity"
    ):
        (
            enc_kind,
            enc_bits,
            dec_kind,
            dec_bits,
        ) = native_writeback_spec(
            args.phase,
            cfg,
        )

    else:
        (
            enc_kind,
            enc_bits,
        ) = resolve_single_writeback(
            args.encoder_writeback,
            args.encoder_state_bits,
            args.phase,
            cfg,
        )

        (
            dec_kind,
            dec_bits,
        ) = resolve_single_writeback(
            args.decoder_writeback,
            args.decoder_state_bits,
            args.phase,
            cfg,
        )

    enc_operator = (
        WritebackOperator(
            enc_kind,
            enc_bits,
        )
    )

    dec_operator = (
        WritebackOperator(
            dec_kind,
            dec_bits,
        )
    )

    forward = build_forward_fn(
        effective_weights,
        quantizers,
        enc_operator,
        dec_operator,
        cfg,
    )

    tensor_equivalence = None
    metric_fidelity = None

    if (
        args.mode
        == "native_fidelity"
    ):
        tensor_equivalence = (
            run_tensor_equivalence(
                reference_model,
                forward,
                normalized_input,
                test_idx,
                cfg,
                effective_weights,
                args.equivalence_samples,
                args.tensor_tolerance,
                args.tensor_mean_tolerance,
                args.tensor_mismatch_fraction,
                args.tensor_tie_fraction,
            )
        )

    else:
        native_fidelity = (
            validate_fidelity_file(
                Path(
                    args.fidelity_json
                ).resolve(),
                checkpoint_path,
                run_dir,
                args.phase,
            )
        )

        pf(
            "[FIDELITY] condition "
            "authorized by native "
            "tensor/metric validation "
            f"from {args.fidelity_json}"
        )

        del native_fidelity

    stochastic = (
        enc_operator.kind
        == "stochastic"
        or dec_operator.kind
        == "stochastic"
    )

    seeds = (
        [
            args.sr_seed_base
            + index
            for index in range(
                args.sr_seeds
            )
        ]
        if stochastic
        else [
            args.sr_seed_base
        ]
    )

    passes = []

    for (
        realization_index,
        seed,
    ) in enumerate(
        seeds
    ):
        (
            accumulator,
            metrics,
        ) = run_condition_pass(
            condition_name=(
                f"{args.phase}_"
                f"{args.mode}_"
                f"r{realization_index}"
            ),
            forward=forward,
            enc_operator=enc_operator,
            dec_operator=dec_operator,
            cfg=cfg,
            normalized_input=normalized_input,
            res_data=res_data,
            labels=labels,
            test_idx=test_idx,
            infer_batch=args.infer_batch,
            hist_bins=args.hist_bins,
            gate_hist_bins=args.gate_hist_bins,
            gate_width_ns=args.gate_width_ns,
            sr_seed=seed,
        )

        summary = summarize_pass(
            accumulator,
            metrics,
            args.bootstrap_reps,
            args.bootstrap_seed,
            args.bootstrap_batch_reps,
        )

        passes.append(
            (
                accumulator,
                metrics,
                summary,
            )
        )

    if (
        args.mode
        == "native_fidelity"
    ):
        (
            metrics_path,
            saved_metrics,
        ) = find_saved_metrics(
            run_dir,
            args.phase,
        )

        pf(
            "[METRIC-FIDELITY] "
            f"source: {metrics_path}"
        )

        metric_fidelity = (
            validate_saved_metrics(
                args.phase,
                saved_metrics,
                passes[
                    0
                ][
                    1
                ],
                args.mae_tolerance,
                args.rmse_tolerance,
            )
        )

    elapsed = (
        time.time()
        - started
    )

    write_analysis_outputs(
        out_dir,
        args,
        run_dir,
        checkpoint_path,
        cfg,
        enc_operator,
        dec_operator,
        passes,
        tensor_equivalence,
        metric_fidelity,
        elapsed,
    )

    pf(
        "[DONE] completed in "
        f"{elapsed / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()