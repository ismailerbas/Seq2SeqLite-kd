#!/usr/bin/env python3
"""
Frozen LSTM recurrent-state location analysis.

Dissects the joint h+c state-writeback intervention of
analyze_lstm_state_writeback.py into per-carrier conditions on the
same trained B8 QLSTM checkpoint, with trained parameters fixed:

    native            h deterministic B_native, c deterministic B_native
    h_only_forced     h deterministic B_forced, c deterministic B_native
    c_only_forced     h deterministic B_native, c deterministic B_forced
    h_only_forced_ef  h error-feedback  B_forced, c deterministic B_native
    c_only_forced_ef  h deterministic B_native, c error-feedback  B_forced

All model reconstruction, operator semantics, statistics accumulation,
performance evaluation, and fail-closed native-equivalence gating are
imported unchanged from analyze_lstm_state_writeback.py. This module
adds only the per-carrier sequence runner, the per-carrier accumulator
grid spacings, and the per-carrier condition bookkeeping.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)
os.environ.setdefault(
    "TF_ENABLE_ONEDNN_OPTS",
    "0",
)

import numpy as np
import tensorflow as tf

_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from analyze_lstm_state_writeback import (
    CarryAccumulator,
    ConditionAccumulator,
    ManualQLSTMCell,
    atomic_json_dump,
    build_reference_model,
    grid_delta,
    performance_summary,
    reference_predict,
    run_driver_predictions,
    sha256_file,
    verify_training_provenance,
)

SUPPORTED_OPERATORS = {
    "identity",
    "deterministic",
    "error_feedback",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Frozen LSTM recurrent-state "
            "location (h/c) writeback analysis."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    p.add_argument(
        "--repo-root",
        type=str,
        required=True,
    )
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    p.add_argument(
        "--student-units",
        type=int,
        required=True,
    )
    p.add_argument(
        "--seq-len",
        type=int,
        required=True,
    )
    p.add_argument(
        "--n-out",
        type=int,
        required=True,
    )
    p.add_argument(
        "--gate-width-ns",
        type=float,
        required=True,
    )
    p.add_argument(
        "--bits-kernel",
        type=int,
        required=True,
    )
    p.add_argument(
        "--bits-bias",
        type=int,
        required=True,
    )
    p.add_argument(
        "--bits-recurrent",
        type=int,
        required=True,
    )
    p.add_argument(
        "--bits-activation",
        type=int,
        required=True,
    )
    p.add_argument(
        "--native-state-bits",
        type=int,
        required=True,
    )
    p.add_argument(
        "--forced-state-bits",
        type=int,
        required=True,
    )
    p.add_argument(
        "--expected-alpha",
        type=float,
        required=True,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=5e-5,
    )
    p.add_argument(
        "--equivalence-samples",
        type=int,
        default=2048,
    )

    return p.parse_args()


def make_location_sequence_runner(
    enc_source_cell,
    dec_source_cell,
    dense_layer,
    seq_len,
    units,
    h_operator,
    h_state_bits,
    c_operator,
    c_state_bits,
):
    if h_operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"unsupported h operator: {h_operator}"
        )

    if c_operator not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"unsupported c operator: {c_operator}"
        )

    enc_cell = ManualQLSTMCell(
        enc_source_cell,
        "encoder",
    )
    dec_cell = ManualQLSTMCell(
        dec_source_cell,
        "decoder",
    )

    from qkeras import quantized_bits

    def build_apply(
        operator,
        state_bits,
    ):
        state_quantizer = quantized_bits(
            state_bits,
            0,
            1,
            alpha=1.0,
        )

        delta = tf.constant(
            grid_delta(state_bits),
            dtype=tf.float32,
        )

        def apply_operator(
            raw,
            previous_read,
            residual,
        ):
            del previous_read

            if operator == "identity":
                return (
                    raw,
                    tf.zeros_like(
                        residual
                    ),
                )

            if operator == "deterministic":
                return (
                    state_quantizer(
                        raw
                    ),
                    tf.zeros_like(
                        residual
                    ),
                )

            compensated = (
                raw + residual
            )

            written = state_quantizer(
                compensated
            )

            residual_next = (
                tf.clip_by_value(
                    compensated - written,
                    -delta,
                    delta,
                )
            )

            return (
                written,
                residual_next,
            )

        return apply_operator

    apply_h = build_apply(
        h_operator,
        h_state_bits,
    )

    apply_c = build_apply(
        c_operator,
        c_state_bits,
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[
                    None,
                    seq_len,
                    1,
                ],
                dtype=tf.float32,
            )
        ],
        reduce_retracing=True,
    )
    def run(
        enc_input,
    ):
        batch = tf.shape(
            enc_input
        )[0]

        zero_state = tf.zeros(
            (
                batch,
                units,
            ),
            dtype=tf.float32,
        )

        h_read = zero_state
        c_read = zero_state
        h_residual = zero_state
        c_residual = zero_state

        enc_h_raw_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        enc_h_before_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        enc_h_after_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        enc_c_raw_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        enc_c_before_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        enc_c_after_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )

        final_h_raw = zero_state
        final_c_raw = zero_state

        for t in tf.range(seq_len):
            h_before = h_read
            c_before = c_read

            (
                h_raw,
                c_raw,
            ) = enc_cell(
                enc_input[:, t, :],
                h_before,
                c_before,
            )

            (
                h_after,
                h_residual,
            ) = apply_h(
                h_raw,
                h_before,
                h_residual,
            )

            (
                c_after,
                c_residual,
            ) = apply_c(
                c_raw,
                c_before,
                c_residual,
            )

            enc_h_raw_ta = (
                enc_h_raw_ta.write(
                    t,
                    h_raw,
                )
            )
            enc_h_before_ta = (
                enc_h_before_ta.write(
                    t,
                    h_before,
                )
            )
            enc_h_after_ta = (
                enc_h_after_ta.write(
                    t,
                    h_after,
                )
            )

            enc_c_raw_ta = (
                enc_c_raw_ta.write(
                    t,
                    c_raw,
                )
            )
            enc_c_before_ta = (
                enc_c_before_ta.write(
                    t,
                    c_before,
                )
            )
            enc_c_after_ta = (
                enc_c_after_ta.write(
                    t,
                    c_after,
                )
            )

            h_read = h_after
            c_read = c_after

            final_h_raw = h_raw
            final_c_raw = c_raw

        zero_residual = tf.zeros_like(
            final_h_raw
        )

        (
            dec_h_read,
            dec_h_residual,
        ) = apply_h(
            final_h_raw,
            final_h_raw,
            zero_residual,
        )

        (
            dec_c_read,
            dec_c_residual,
        ) = apply_c(
            final_c_raw,
            final_c_raw,
            zero_residual,
        )

        handoff_h_read = (
            dec_h_read
        )
        handoff_c_read = (
            dec_c_read
        )

        dec_h_raw_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        dec_h_before_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        dec_h_after_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )

        dec_c_raw_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        dec_c_before_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )
        dec_c_after_ta = (
            tf.TensorArray(
                tf.float32,
                size=seq_len,
            )
        )

        zero_input = tf.zeros(
            (
                batch,
                1,
            ),
            dtype=tf.float32,
        )

        for t in tf.range(seq_len):
            h_before = dec_h_read
            c_before = dec_c_read

            (
                h_raw,
                c_raw,
            ) = dec_cell(
                zero_input,
                h_before,
                c_before,
            )

            (
                h_after,
                dec_h_residual,
            ) = apply_h(
                h_raw,
                h_before,
                dec_h_residual,
            )

            (
                c_after,
                dec_c_residual,
            ) = apply_c(
                c_raw,
                c_before,
                dec_c_residual,
            )

            dec_h_raw_ta = (
                dec_h_raw_ta.write(
                    t,
                    h_raw,
                )
            )
            dec_h_before_ta = (
                dec_h_before_ta.write(
                    t,
                    h_before,
                )
            )
            dec_h_after_ta = (
                dec_h_after_ta.write(
                    t,
                    h_after,
                )
            )

            dec_c_raw_ta = (
                dec_c_raw_ta.write(
                    t,
                    c_raw,
                )
            )
            dec_c_before_ta = (
                dec_c_before_ta.write(
                    t,
                    c_before,
                )
            )
            dec_c_after_ta = (
                dec_c_after_ta.write(
                    t,
                    c_after,
                )
            )

            dec_h_read = h_after
            dec_c_read = c_after

        dec_h_raw = tf.transpose(
            dec_h_raw_ta.stack(),
            [
                1,
                0,
                2,
            ],
        )

        predictions = dense_layer(
            dec_h_raw,
            training=False,
        )

        return (
            predictions,

            tf.transpose(
                enc_h_raw_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                enc_h_before_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                enc_h_after_ta.stack(),
                [1, 0, 2],
            ),

            tf.transpose(
                enc_c_raw_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                enc_c_before_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                enc_c_after_ta.stack(),
                [1, 0, 2],
            ),

            dec_h_raw,

            tf.transpose(
                dec_h_before_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                dec_h_after_ta.stack(),
                [1, 0, 2],
            ),

            tf.transpose(
                dec_c_raw_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                dec_c_before_ta.stack(),
                [1, 0, 2],
            ),
            tf.transpose(
                dec_c_after_ta.stack(),
                [1, 0, 2],
            ),

            final_h_raw,
            handoff_h_read,
            final_c_raw,
            handoff_c_read,
        )

    return run


class LocationConditionAccumulator(
    ConditionAccumulator
):
    """
    Identical statistics pipeline to
    ConditionAccumulator, but the h and c
    carriers use their own state-grid
    spacings so that mixed-precision
    location conditions are measured
    against the correct grid per carrier.
    update(), summary(), and
    per_sequence_arrays() are inherited
    unchanged.
    """

    def __init__(
        self,
        seq_len,
        units,
        delta_h,
        delta_c,
    ):
        self.encoder_h = CarryAccumulator(
            seq_len,
            units,
            delta_h,
            seq_len,
        )

        self.encoder_c = CarryAccumulator(
            seq_len,
            units,
            delta_c,
            seq_len,
        )

        self.decoder_h = CarryAccumulator(
            seq_len,
            units,
            delta_h,
            seq_len - 1,
        )

        self.decoder_c = CarryAccumulator(
            seq_len,
            units,
            delta_c,
            seq_len - 1,
        )

        self.handoff_h_abs_sum = 0.0
        self.handoff_c_abs_sum = 0.0
        self.handoff_h_count = 0
        self.handoff_c_count = 0
        self.handoff_h_changed = 0
        self.handoff_c_changed = 0


def flatten_location_summary(
    name,
    summary,
):
    perf = summary[
        "performance"
    ]

    enc_h = summary[
        "state"
    ]["encoder"]["h"]

    enc_c = summary[
        "state"
    ]["encoder"]["c"]

    dec_h = summary[
        "state"
    ]["decoder"]["h"]

    dec_c = summary[
        "state"
    ]["decoder"]["c"]

    handoff = summary[
        "state"
    ]["handoff"]

    return {
        "condition": name,
        "h_operator": summary[
            "h_operator"
        ],
        "h_state_bits": summary[
            "h_state_bits"
        ],
        "c_operator": summary[
            "c_operator"
        ],
        "c_state_bits": summary[
            "c_state_bits"
        ],

        "tau1_rmse": perf[
            "tau1_rmse"
        ],
        "tau2_rmse": perf[
            "tau2_rmse"
        ],
        "fret_rmse": perf[
            "fret_rmse"
        ],
        "tau1_r": perf[
            "tau1_r"
        ],
        "tau2_r": perf[
            "tau2_r"
        ],
        "fret_r": perf[
            "fret_r"
        ],
        "mae_seq": perf[
            "mae_seq"
        ],

        "encoder_deadband_h": enc_h[
            "deadband_fraction"
        ],
        "encoder_deadband_c": enc_c[
            "deadband_fraction"
        ],
        "encoder_state_change_h": enc_h[
            "state_change_fraction"
        ],
        "encoder_state_change_c": enc_c[
            "state_change_fraction"
        ],
        "encoder_rail_h": enc_h[
            "rail_read_fraction"
        ],
        "encoder_rail_c": enc_c[
            "rail_read_fraction"
        ],
        "encoder_mean_sq_debt_h": enc_h[
            "mean_squared_normalized_unpaid_drift"
        ],
        "encoder_mean_sq_debt_c": enc_c[
            "mean_squared_normalized_unpaid_drift"
        ],

        "decoder_deadband_h": dec_h[
            "deadband_fraction"
        ],
        "decoder_deadband_c": dec_c[
            "deadband_fraction"
        ],
        "decoder_state_change_h": dec_h[
            "state_change_fraction"
        ],
        "decoder_state_change_c": dec_c[
            "state_change_fraction"
        ],
        "decoder_subthreshold_write_h": dec_h[
            "subthreshold_visible_write_fraction"
        ],
        "decoder_subthreshold_write_c": dec_c[
            "subthreshold_visible_write_fraction"
        ],
        "decoder_rail_h": dec_h[
            "rail_read_fraction"
        ],
        "decoder_rail_c": dec_c[
            "rail_read_fraction"
        ],
        "decoder_raw_outside_grid_h": dec_h[
            "raw_outside_grid_fraction"
        ],
        "decoder_raw_outside_grid_c": dec_c[
            "raw_outside_grid_fraction"
        ],
        "decoder_same_sign_h": dec_h[
            "same_sign_fraction"
        ],
        "decoder_same_sign_c": dec_c[
            "same_sign_fraction"
        ],
        "decoder_persistence_odds_h": dec_h[
            "persistence_odds_ratio"
        ],
        "decoder_persistence_odds_c": dec_c[
            "persistence_odds_ratio"
        ],
        "decoder_median_run_h": dec_h[
            "median_completed_same_sign_run"
        ],
        "decoder_median_run_c": dec_c[
            "median_completed_same_sign_run"
        ],
        "decoder_p90_run_h": dec_h[
            "p90_completed_same_sign_run"
        ],
        "decoder_p90_run_c": dec_c[
            "p90_completed_same_sign_run"
        ],
        "decoder_mean_sq_debt_h": dec_h[
            "mean_squared_normalized_unpaid_drift"
        ],
        "decoder_mean_sq_debt_c": dec_c[
            "mean_squared_normalized_unpaid_drift"
        ],

        "handoff_mae_h": handoff[
            "mean_abs_error_h"
        ],
        "handoff_mae_c": handoff[
            "mean_abs_error_c"
        ],
        "handoff_changed_h": handoff[
            "changed_fraction_h"
        ],
        "handoff_changed_c": handoff[
            "changed_fraction_c"
        ],
    }


def main():
    args = parse_args()

    repo_root = Path(
        args.repo_root
    ).resolve()

    data_dir = Path(
        args.data_dir
    ).resolve()

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(
            "repo root does not exist: "
            f"{repo_root}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            "data directory does not exist: "
            f"{data_dir}"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            "checkpoint does not exist: "
            f"{checkpoint}"
        )

    if (
        args.forced_state_bits
        == args.native_state_bits
    ):
        raise ValueError(
            "forced state bits equal native "
            "state bits; the location "
            "conditions would be degenerate: "
            f"{args.forced_state_bits}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sys.path.insert(
        0,
        str(repo_root),
    )

    import train_student_vanilla_kd_lstm as train_lstm

    args_file = (
        checkpoint.parent
        / "student_args.json"
    )

    if not args_file.is_file():
        raise FileNotFoundError(
            "student_args.json not found "
            "beside checkpoint: "
            f"{args_file}"
        )

    with open(
        args_file,
        "r",
    ) as f:
        saved_args = json.load(f)

    verify_training_provenance(
        args,
        saved_args,
    )

    print(
        "[VALIDATION] checkpoint "
        "provenance verified",
        flush=True,
    )

    tf.keras.utils.set_random_seed(
        42
    )

    tf.keras.mixed_precision.set_global_policy(
        "float32"
    )

    (
        file_input,
        file_res,
        file_labels,
        file_train,
        file_val,
        file_test,
    ) = train_lstm.find_data_files(
        str(data_dir),
        args.seq_len,
    )

    del file_train
    del file_val

    normalized_input = np.load(
        file_input,
        mmap_mode="r",
    )

    res = np.load(
        file_res,
        mmap_mode="r",
    )

    labels = np.load(
        file_labels,
        mmap_mode="r",
    )

    test_idx = np.load(
        file_test
    )

    if normalized_input.shape[1:] != (
        args.seq_len,
        1,
    ):
        raise RuntimeError(
            "unexpected input shape "
            f"{normalized_input.shape}; "
            "expected "
            f"(*,{args.seq_len},1)"
        )

    if res.shape[1:] != (
        args.seq_len,
        args.n_out,
    ):
        raise RuntimeError(
            "unexpected target shape "
            f"{res.shape}; expected "
            f"(*,{args.seq_len},{args.n_out})"
        )

    if labels.shape[1] < 3:
        raise RuntimeError(
            "labels must have at least "
            f"3 columns, got {labels.shape}"
        )

    enc_test = np.asarray(
        normalized_input[test_idx],
        dtype=np.float32,
    )

    res_test = np.asarray(
        res[test_idx],
        dtype=np.float32,
    )

    lab_test = np.asarray(
        labels[test_idx],
        dtype=np.float32,
    )

    n_test = len(
        test_idx
    )

    print(
        "[DATA] "
        f"test samples={n_test:,} "
        f"input={enc_test.shape} "
        f"target={res_test.shape}",
        flush=True,
    )

    reference_model = (
        build_reference_model(
            train_lstm,
            args,
        )
    )

    enc_layer = (
        reference_model.get_layer(
            "senclstm"
        )
    )

    dec_layer = (
        reference_model.get_layer(
            "sdeclstm"
        )
    )

    dense_layer = (
        reference_model.get_layer(
            "sdec_dense"
        )
    )

    enc_cell = enc_layer.cell
    dec_cell = dec_layer.cell

    if (
        enc_cell.state_quantizer_internal
        is None
        or dec_cell.state_quantizer_internal
        is None
    ):
        raise RuntimeError(
            "native QLSTM checkpoint "
            "does not contain a state quantizer"
        )

    conditions = {
        "native": (
            "deterministic",
            args.native_state_bits,
            "deterministic",
            args.native_state_bits,
        ),
        "h_only_forced": (
            "deterministic",
            args.forced_state_bits,
            "deterministic",
            args.native_state_bits,
        ),
        "c_only_forced": (
            "deterministic",
            args.native_state_bits,
            "deterministic",
            args.forced_state_bits,
        ),
        "h_only_forced_ef": (
            "error_feedback",
            args.forced_state_bits,
            "deterministic",
            args.native_state_bits,
        ),
        "c_only_forced_ef": (
            "deterministic",
            args.native_state_bits,
            "error_feedback",
            args.forced_state_bits,
        ),
    }

    eq_n = min(
        int(
            args.equivalence_samples
        ),
        n_test,
    )

    if eq_n <= 0:
        raise RuntimeError(
            "equivalence sample count "
            "must be positive"
        )

    print(
        "[VALIDATION] native equivalence "
        f"preflight on {eq_n:,} samples",
        flush=True,
    )

    ref_eq = reference_predict(
        reference_model,
        enc_test[:eq_n],
        args.seq_len,
        args.n_out,
        args.batch_size,
    )

    native_runner = (
        make_location_sequence_runner(
            enc_cell,
            dec_cell,
            dense_layer,
            args.seq_len,
            args.student_units,
            "deterministic",
            args.native_state_bits,
            "deterministic",
            args.native_state_bits,
        )
    )

    drv_eq = run_driver_predictions(
        native_runner,
        enc_test[:eq_n],
        args.seq_len,
        args.n_out,
        args.batch_size,
        accumulator=None,
    )

    eq_abs = np.abs(
        ref_eq - drv_eq
    )

    eq_max = float(
        np.max(eq_abs)
    )

    eq_mean = float(
        np.mean(eq_abs)
    )

    eq_passed = bool(
        eq_max
        <= args.tolerance
    )

    print(
        "[VALIDATION] preflight "
        f"max_abs={eq_max:.6e} "
        f"mean_abs={eq_mean:.6e} "
        f"tolerance={args.tolerance:.6e} "
        f"passed={eq_passed}",
        flush=True,
    )

    if not eq_passed:
        raise RuntimeError(
            "Native external-state LSTM "
            "location driver does not "
            "reproduce the intact QLSTM. "
            "Analysis aborted before "
            "interventions."
        )

    print(
        "[REFERENCE] evaluating intact "
        "native QLSTM on full test set",
        flush=True,
    )

    reference_preds = reference_predict(
        reference_model,
        enc_test,
        args.seq_len,
        args.n_out,
        args.batch_size,
    )

    t_ns_axis = (
        np.arange(
            args.seq_len,
            dtype=np.float32,
        )
        * args.gate_width_ns
    )

    summaries = {}

    validation = {
        "checkpoint_provenance": {
            "passed": True,
        },
        "native_equivalence_preflight": {
            "samples": int(eq_n),
            "max_abs": eq_max,
            "mean_abs": eq_mean,
            "tolerance": float(
                args.tolerance
            ),
            "passed": eq_passed,
        },
    }

    for (
        name,
        (
            h_operator,
            h_state_bits,
            c_operator,
            c_state_bits,
        ),
    ) in conditions.items():
        print(
            "=" * 72,
            flush=True,
        )

        print(
            f"[CONDITION] {name}: "
            f"h_operator={h_operator}, "
            f"h_state_bits={h_state_bits}, "
            f"c_operator={c_operator}, "
            f"c_state_bits={c_state_bits}",
            flush=True,
        )

        runner = (
            make_location_sequence_runner(
                enc_cell,
                dec_cell,
                dense_layer,
                args.seq_len,
                args.student_units,
                h_operator,
                h_state_bits,
                c_operator,
                c_state_bits,
            )
        )

        accumulator = (
            LocationConditionAccumulator(
                args.seq_len,
                args.student_units,
                grid_delta(
                    h_state_bits
                ),
                grid_delta(
                    c_state_bits
                ),
            )
        )

        preds = run_driver_predictions(
            runner,
            enc_test,
            args.seq_len,
            args.n_out,
            args.batch_size,
            accumulator=accumulator,
        )

        if name == "native":
            diff = np.abs(
                preds
                - reference_preds
            )

            max_abs = float(
                np.max(diff)
            )

            mean_abs = float(
                np.mean(diff)
            )

            passed = bool(
                max_abs
                <= args.tolerance
            )

            validation[
                "native_equivalence_full_test"
            ] = {
                "samples": int(
                    n_test
                ),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "tolerance": float(
                    args.tolerance
                ),
                "passed": passed,
            }

            print(
                "[VALIDATION] full-test "
                "native equivalence "
                f"max_abs={max_abs:.6e} "
                f"mean_abs={mean_abs:.6e} "
                f"passed={passed}",
                flush=True,
            )

            if not passed:
                atomic_json_dump(
                    validation,
                    output_dir
                    / "validation.json",
                )

                raise RuntimeError(
                    "Full-test native driver "
                    "equivalence failed. "
                    "Intervention results rejected."
                )

        (
            perf,
            per_seq_perf,
        ) = performance_summary(
            train_lstm,
            preds,
            res_test,
            lab_test,
            t_ns_axis,
        )

        state_summary = (
            accumulator.summary()
        )

        summary = {
            "h_operator": h_operator,
            "h_state_bits": int(
                h_state_bits
            ),
            "c_operator": c_operator,
            "c_state_bits": int(
                c_state_bits
            ),
            "performance": perf,
            "state": state_summary,
        }

        summaries[name] = summary

        per_seq = (
            accumulator.per_sequence_arrays()
        )

        per_seq.update(
            per_seq_perf
        )

        np.savez_compressed(
            output_dir
            / (
                f"{name}_"
                "per_sequence_metrics.npz"
            ),
            **per_seq,
        )

        dec_h = state_summary[
            "decoder"
        ]["h"]

        dec_c = state_summary[
            "decoder"
        ]["c"]

        print(
            f"[{name}] "
            f"tau1_rmse="
            f"{perf['tau1_rmse']:.6f} "
            f"tau2_rmse="
            f"{perf['tau2_rmse']:.6f} "
            f"mae_seq="
            f"{perf['mae_seq']:.6f}",
            flush=True,
        )

        print(
            f"[{name}] decoder h: "
            f"deadband="
            f"{dec_h['deadband_fraction']:.6f} "
            f"state_change="
            f"{dec_h['state_change_fraction']:.6f} "
            f"same_sign="
            f"{dec_h['same_sign_fraction']} "
            f"median_run="
            f"{dec_h['median_completed_same_sign_run']} "
            f"rail="
            f"{dec_h['rail_read_fraction']:.6f} "
            f"mean_sq_debt="
            f"{dec_h['mean_squared_normalized_unpaid_drift']:.6f}",
            flush=True,
        )

        print(
            f"[{name}] decoder c: "
            f"deadband="
            f"{dec_c['deadband_fraction']:.6f} "
            f"state_change="
            f"{dec_c['state_change_fraction']:.6f} "
            f"same_sign="
            f"{dec_c['same_sign_fraction']} "
            f"median_run="
            f"{dec_c['median_completed_same_sign_run']} "
            f"rail="
            f"{dec_c['rail_read_fraction']:.6f} "
            f"mean_sq_debt="
            f"{dec_c['mean_squared_normalized_unpaid_drift']:.6f}",
            flush=True,
        )

    atomic_json_dump(
        validation,
        output_dir
        / "validation.json",
    )

    atomic_json_dump(
        summaries,
        output_dir
        / "lstm_state_location_summary.json",
    )

    rows = [
        flatten_location_summary(
            name,
            summaries[name],
        )
        for name in conditions
    ]

    columns = list(
        rows[0].keys()
    )

    with open(
        output_dir
        / "lstm_state_location_summary.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    manifest = {
        "analysis": (
            "Frozen LSTM recurrent-state "
            "location (h/c) writeback "
            "dissection with targeted "
            "error-feedback rescue"
        ),
        "checkpoint": str(
            checkpoint
        ),
        "checkpoint_sha256": (
            sha256_file(
                checkpoint
            )
        ),
        "student_args_file": str(
            args_file
        ),
        "student_units": int(
            args.student_units
        ),
        "sequence_length": int(
            args.seq_len
        ),
        "encoder_live_write_steps_per_sequence": int(
            args.seq_len
        ),
        "decoder_live_write_steps_per_sequence": int(
            args.seq_len - 1
        ),
        "test_samples": int(
            n_test
        ),
        "bits_kernel": int(
            args.bits_kernel
        ),
        "bits_bias": int(
            args.bits_bias
        ),
        "bits_recurrent": int(
            args.bits_recurrent
        ),
        "bits_activation": int(
            args.bits_activation
        ),
        "native_state_bits": int(
            args.native_state_bits
        ),
        "forced_state_bits": int(
            args.forced_state_bits
        ),
        "expected_alpha": float(
            args.expected_alpha
        ),
        "batch_size": int(
            args.batch_size
        ),
        "equivalence_tolerance": float(
            args.tolerance
        ),
        "equivalence_samples": int(
            eq_n
        ),
        "conditions": {
            name: {
                "h_operator": h_op,
                "h_state_bits": int(
                    h_bits
                ),
                "c_operator": c_op,
                "c_state_bits": int(
                    c_bits
                ),
            }
            for (
                name,
                (
                    h_op,
                    h_bits,
                    c_op,
                    c_bits,
                ),
            ) in conditions.items()
        },
        "tensorflow_version": (
            tf.__version__
        ),
        "numpy_version": (
            np.__version__
        ),
    }

    atomic_json_dump(
        manifest,
        output_dir
        / "analysis_manifest.json",
    )

    print(
        "=" * 72,
        flush=True,
    )

    print(
        "[DONE] LSTM state-location "
        "analysis completed",
        flush=True,
    )

    print(
        f"[DONE] outputs: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()