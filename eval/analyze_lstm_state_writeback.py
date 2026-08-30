#!/usr/bin/env python3

import argparse
import csv
import hashlib
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


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Frozen LSTM recurrent-state "
            "writeback analysis."
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


def sha256_file(
    path,
    chunk_size=8 * 1024 * 1024,
):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_dump(
    obj,
    path,
):
    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(tmp, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    os.replace(
        tmp,
        path,
    )


def grid_delta(bits):
    if bits < 2:
        raise ValueError(
            "state bits must be >= 2, "
            f"got {bits}"
        )

    return float(
        2.0 ** (1 - bits)
    )


def safe_corr(
    x,
    y,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.size < 2 or y.size < 2:
        return None

    sx = float(np.std(x))
    sy = float(np.std(y))

    if sx == 0.0 or sy == 0.0:
        return None

    value = float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )

    if not np.isfinite(value):
        return None

    return value


def verify_training_provenance(
    args,
    saved,
):
    expected = {
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
        "bits_state": int(
            args.native_state_bits
        ),
        "student_units": int(
            args.student_units
        ),
        "seq_len": int(
            args.seq_len
        ),
        "n_out": int(
            args.n_out
        ),
    }

    mismatches = []

    for key, wanted in expected.items():
        got = saved.get(key)

        if got != wanted:
            mismatches.append(
                f"{key}: expected "
                f"{wanted!r}, found {got!r}"
            )

    alpha_got = saved.get(
        "alpha"
    )

    if (
        alpha_got is None
        or not np.isclose(
            float(alpha_got),
            float(args.expected_alpha),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        mismatches.append(
            "alpha: expected "
            f"{args.expected_alpha!r}, "
            f"found {alpha_got!r}"
        )

    if mismatches:
        raise RuntimeError(
            "Checkpoint provenance mismatch:\n  "
            + "\n  ".join(mismatches)
        )


class ManualQLSTMCell:
    """
    External-state QLSTM recurrence using
    the trained QLSTMCell weights,
    quantizers, and activations.
    """

    def __init__(
        self,
        source_cell,
        label,
    ):
        self.label = label
        self.units = int(
            source_cell.units
        )

        if float(
            source_cell.dropout
        ) != 0.0:
            raise RuntimeError(
                f"{label}: dropout must be 0 "
                "for frozen reconstruction"
            )

        if float(
            source_cell.recurrent_dropout
        ) != 0.0:
            raise RuntimeError(
                f"{label}: recurrent_dropout "
                "must be 0 for frozen reconstruction"
            )

        if int(
            source_cell.implementation
        ) != 1:
            raise RuntimeError(
                f"{label}: expected QLSTM "
                "implementation=1, got "
                f"{source_cell.implementation}"
            )

        if not bool(
            source_cell.use_bias
        ):
            raise RuntimeError(
                f"{label}: expected use_bias=True"
            )

        self.kernel = (
            source_cell.kernel
        )
        self.recurrent_kernel = (
            source_cell.recurrent_kernel
        )
        self.bias = (
            source_cell.bias
        )

        self.kernel_quantizer = (
            source_cell.kernel_quantizer_internal
        )
        self.recurrent_quantizer = (
            source_cell.recurrent_quantizer_internal
        )
        self.bias_quantizer = (
            source_cell.bias_quantizer_internal
        )

        self.activation = (
            source_cell.activation
        )
        self.recurrent_activation = (
            source_cell.recurrent_activation
        )

    def __call__(
        self,
        inputs,
        h_read,
        c_read,
    ):
        qkernel = (
            self.kernel_quantizer(
                self.kernel
            )
        )
        qrecurrent = (
            self.recurrent_quantizer(
                self.recurrent_kernel
            )
        )
        qbias = (
            self.bias_quantizer(
                self.bias
            )
        )

        (
            k_i,
            k_f,
            k_c,
            k_o,
        ) = tf.split(
            qkernel,
            4,
            axis=1,
        )

        (
            r_i,
            r_f,
            r_c,
            r_o,
        ) = tf.split(
            qrecurrent,
            4,
            axis=1,
        )

        (
            b_i,
            b_f,
            b_c,
            b_o,
        ) = tf.split(
            qbias,
            4,
            axis=0,
        )

        x_i = tf.nn.bias_add(
            tf.matmul(
                inputs,
                k_i,
            ),
            b_i,
        )

        x_f = tf.nn.bias_add(
            tf.matmul(
                inputs,
                k_f,
            ),
            b_f,
        )

        x_c = tf.nn.bias_add(
            tf.matmul(
                inputs,
                k_c,
            ),
            b_c,
        )

        x_o = tf.nn.bias_add(
            tf.matmul(
                inputs,
                k_o,
            ),
            b_o,
        )

        i = self.recurrent_activation(
            x_i
            + tf.matmul(
                h_read,
                r_i,
            )
        )

        f = self.recurrent_activation(
            x_f
            + tf.matmul(
                h_read,
                r_f,
            )
        )

        c_raw = (
            f * c_read
            + i
            * self.activation(
                x_c
                + tf.matmul(
                    h_read,
                    r_c,
                )
            )
        )

        o = self.recurrent_activation(
            x_o
            + tf.matmul(
                h_read,
                r_o,
            )
        )

        h_raw = (
            o
            * self.activation(
                c_raw
            )
        )

        return (
            h_raw,
            c_raw,
        )


def make_sequence_runner(
    enc_source_cell,
    dec_source_cell,
    dense_layer,
    seq_len,
    units,
    operator,
    state_bits,
):
    if operator not in {
        "identity",
        "deterministic",
        "error_feedback",
    }:
        raise ValueError(
            f"unsupported operator: {operator}"
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
            ) = apply_operator(
                h_raw,
                h_before,
                h_residual,
            )

            (
                c_after,
                c_residual,
            ) = apply_operator(
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
        ) = apply_operator(
            final_h_raw,
            final_h_raw,
            zero_residual,
        )

        (
            dec_c_read,
            dec_c_residual,
        ) = apply_operator(
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
            ) = apply_operator(
                h_raw,
                h_before,
                dec_h_residual,
            )

            (
                c_after,
                dec_c_residual,
            ) = apply_operator(
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


class CarryAccumulator:
    def __init__(
        self,
        seq_len,
        units,
        delta,
        live_steps,
    ):
        self.seq_len = int(
            seq_len
        )
        self.units = int(
            units
        )
        self.delta = float(
            delta
        )
        self.half = (
            self.delta / 2.0
        )
        self.live_steps = int(
            live_steps
        )

        if not (
            1
            <= self.live_steps
            <= self.seq_len
        ):
            raise ValueError(
                "live_steps must be in "
                f"[1,{self.seq_len}], "
                f"got {self.live_steps}"
            )

        self.live_elements = 0
        self.deadband = 0
        self.state_changes = 0
        self.subthreshold_visible_writes = 0
        self.rail_reads = 0
        self.raw_outside_grid = 0
        self.active_votes = 0
        self.normal_events = 0

        self.pp = 0
        self.pn = 0
        self.np_ = 0
        self.nn = 0

        self.debt_sq_sum = 0.0
        self.debt_count = 0

        self.run_hist = np.zeros(
            self.seq_len + 2,
            dtype=np.int64,
        )

        self.per_seq_active_votes = []
        self.per_seq_normal_events = []
        self.per_seq_state_changes = []
        self.per_seq_max_run = []
        self.per_seq_max_debt_grid_steps = []

    def _record_runs(
        self,
        run_lengths,
    ):
        if run_lengths.size == 0:
            return

        counts = np.bincount(
            run_lengths.astype(
                np.int64
            ),
            minlength=self.run_hist.shape[0],
        )

        self.run_hist[
            : counts.shape[0]
        ] += counts

    def update(
        self,
        raw,
        read_before,
        read_after,
    ):
        raw = np.asarray(
            raw[
                :,
                : self.live_steps,
                :,
            ],
            dtype=np.float32,
        )

        before = np.asarray(
            read_before[
                :,
                : self.live_steps,
                :,
            ],
            dtype=np.float32,
        )

        after = np.asarray(
            read_after[
                :,
                : self.live_steps,
                :,
            ],
            dtype=np.float32,
        )

        if (
            raw.shape != before.shape
            or raw.shape != after.shape
        ):
            raise ValueError(
                "carry shapes disagree: "
                f"raw={raw.shape}, "
                f"before={before.shape}, "
                f"after={after.shape}"
            )

        if raw.shape[2] != self.units:
            raise ValueError(
                f"expected {self.units} units, "
                f"got tensor shape {raw.shape}"
            )

        batch = raw.shape[0]

        d = (
            raw - before
        )
        abs_d = np.abs(d)

        inside = (
            abs_d < self.half
        )

        changed = (
            np.abs(
                after - before
            )
            > 1e-7
        )

        self.live_elements += int(
            batch
            * self.live_steps
            * self.units
        )

        self.deadband += int(
            np.sum(inside)
        )

        self.state_changes += int(
            np.sum(changed)
        )

        self.subthreshold_visible_writes += int(
            np.sum(
                inside
                & changed
            )
        )

        rail_hi = (
            1.0 - self.delta
        )

        self.rail_reads += int(
            np.sum(
                (before <= -1.0)
                | (before >= rail_hi)
            )
        )

        self.raw_outside_grid += int(
            np.sum(
                (raw < -1.0)
                | (raw > rail_hi)
            )
        )

        self.normal_events += int(
            np.sum(~inside)
        )

        vote_mask = (
            inside
            & (abs_d > 1e-12)
        )

        signs = np.zeros(
            d.shape,
            dtype=np.int8,
        )

        signs[
            vote_mask
            & (d > 0.0)
        ] = 1

        signs[
            vote_mask
            & (d < 0.0)
        ] = -1

        self.active_votes += int(
            np.sum(vote_mask)
        )

        if self.live_steps > 1:
            pair_mask = (
                (signs[:, :-1, :] != 0)
                & (signs[:, 1:, :] != 0)
                & (~changed[:, :-1, :])
            )

            prev_pos = (
                signs[:, :-1, :]
                > 0
            )
            next_pos = (
                signs[:, 1:, :]
                > 0
            )

            self.pp += int(
                np.sum(
                    pair_mask
                    & prev_pos
                    & next_pos
                )
            )

            self.pn += int(
                np.sum(
                    pair_mask
                    & prev_pos
                    & ~next_pos
                )
            )

            self.np_ += int(
                np.sum(
                    pair_mask
                    & ~prev_pos
                    & next_pos
                )
            )

            self.nn += int(
                np.sum(
                    pair_mask
                    & ~prev_pos
                    & ~next_pos
                )
            )

        run_len = np.zeros(
            (
                batch,
                self.units,
            ),
            dtype=np.int32,
        )

        run_sign = np.zeros(
            (
                batch,
                self.units,
            ),
            dtype=np.int8,
        )

        max_run = np.zeros(
            (
                batch,
                self.units,
            ),
            dtype=np.int32,
        )

        debt = np.zeros(
            (
                batch,
                self.units,
            ),
            dtype=np.float64,
        )

        max_debt = np.zeros(
            (
                batch,
                self.units,
            ),
            dtype=np.float64,
        )

        per_seq_votes = np.zeros(
            batch,
            dtype=np.int64,
        )
        per_seq_normal = np.zeros(
            batch,
            dtype=np.int64,
        )
        per_seq_changes = np.zeros(
            batch,
            dtype=np.int64,
        )

        for t in range(
            self.live_steps
        ):
            s_t = signs[:, t, :]
            changed_t = changed[:, t, :]
            inside_t = inside[:, t, :]

            per_seq_votes += np.sum(
                s_t != 0,
                axis=1,
            )

            per_seq_normal += np.sum(
                ~inside_t,
                axis=1,
            )

            per_seq_changes += np.sum(
                changed_t,
                axis=1,
            )

            eligible = (
                s_t != 0
            )

            continuing = (
                eligible
                & (run_len > 0)
                & (s_t == run_sign)
            )

            starting = (
                eligible
                & ~continuing
            )

            terminate_before = (
                (run_len > 0)
                & ~continuing
                & starting
            )

            terminate_no_vote = (
                (run_len > 0)
                & ~eligible
            )

            self._record_runs(
                run_len[
                    terminate_before
                    | terminate_no_vote
                ]
            )

            run_len = np.where(
                eligible,
                np.where(
                    continuing,
                    run_len + 1,
                    1,
                ),
                0,
            )

            run_sign = np.where(
                eligible,
                s_t,
                0,
            ).astype(
                np.int8
            )

            max_run = np.maximum(
                max_run,
                run_len,
            )

            terminate_write = (
                eligible
                & changed_t
            )

            self._record_runs(
                run_len[
                    terminate_write
                ]
            )

            run_len = np.where(
                terminate_write,
                0,
                run_len,
            )

            run_sign = np.where(
                terminate_write,
                0,
                run_sign,
            ).astype(
                np.int8
            )

            debt += np.where(
                inside_t,
                d[:, t, :],
                0.0,
            ).astype(
                np.float64
            )

            debt = np.where(
                changed_t,
                0.0,
                debt,
            )

            debt_grid = (
                np.abs(debt)
                / self.delta
            )

            max_debt = np.maximum(
                max_debt,
                debt_grid,
            )

            self.debt_sq_sum += float(
                np.sum(
                    debt_grid
                    * debt_grid
                )
            )

            self.debt_count += int(
                debt_grid.size
            )

        self._record_runs(
            run_len[
                run_len > 0
            ]
        )

        self.per_seq_active_votes.append(
            per_seq_votes
        )

        self.per_seq_normal_events.append(
            per_seq_normal
        )

        self.per_seq_state_changes.append(
            per_seq_changes
        )

        self.per_seq_max_run.append(
            np.max(
                max_run,
                axis=1,
            ).astype(
                np.int16
            )
        )

        self.per_seq_max_debt_grid_steps.append(
            np.max(
                max_debt,
                axis=1,
            ).astype(
                np.float32
            )
        )

    def _hist_percentile(
        self,
        fraction,
    ):
        total = int(
            np.sum(
                self.run_hist
            )
        )

        if total == 0:
            return None

        target = int(
            np.ceil(
                float(fraction)
                * total
            )
        )

        target = max(
            target,
            1,
        )

        cumulative = np.cumsum(
            self.run_hist
        )

        return float(
            np.searchsorted(
                cumulative,
                target,
                side="left",
            )
        )

    def summary(
        self,
    ):
        pairs = (
            self.pp
            + self.pn
            + self.np_
            + self.nn
        )

        same = (
            self.pp
            + self.nn
        )

        odds = (
            (self.pp + 0.5)
            * (self.nn + 0.5)
            / (
                (self.pn + 0.5)
                * (self.np_ + 0.5)
            )
        )

        return {
            "live_steps_per_sequence": int(
                self.live_steps
            ),
            "live_elements": int(
                self.live_elements
            ),
            "deadband_fraction": float(
                self.deadband
                / max(
                    self.live_elements,
                    1,
                )
            ),
            "state_change_fraction": float(
                self.state_changes
                / max(
                    self.live_elements,
                    1,
                )
            ),
            "subthreshold_visible_write_fraction": float(
                self.subthreshold_visible_writes
                / max(
                    self.deadband,
                    1,
                )
            ),
            "rail_read_fraction": float(
                self.rail_reads
                / max(
                    self.live_elements,
                    1,
                )
            ),
            "raw_outside_grid_fraction": float(
                self.raw_outside_grid
                / max(
                    self.live_elements,
                    1,
                )
            ),
            "active_votes": int(
                self.active_votes
            ),
            "normal_events": int(
                self.normal_events
            ),
            "adjacent_active_vote_pairs": int(
                pairs
            ),
            "same_sign_pairs": int(
                same
            ),
            "same_sign_fraction": (
                float(
                    same / pairs
                )
                if pairs
                else None
            ),
            "persistence_odds_ratio": float(
                odds
            ),
            "pp": int(
                self.pp
            ),
            "pn": int(
                self.pn
            ),
            "np": int(
                self.np_
            ),
            "nn": int(
                self.nn
            ),
            "median_completed_same_sign_run": (
                self._hist_percentile(
                    0.5
                )
            ),
            "p90_completed_same_sign_run": (
                self._hist_percentile(
                    0.9
                )
            ),
            "mean_squared_normalized_unpaid_drift": float(
                self.debt_sq_sum
                / max(
                    self.debt_count,
                    1,
                )
            ),
        }

    def per_sequence_arrays(
        self,
        prefix,
    ):
        return {
            f"{prefix}_active_votes": np.concatenate(
                self.per_seq_active_votes
            ),
            f"{prefix}_normal_events": np.concatenate(
                self.per_seq_normal_events
            ),
            f"{prefix}_state_changes": np.concatenate(
                self.per_seq_state_changes
            ),
            f"{prefix}_max_same_sign_run": np.concatenate(
                self.per_seq_max_run
            ),
            f"{prefix}_max_unpaid_drift_grid_steps": np.concatenate(
                self.per_seq_max_debt_grid_steps
            ),
        }


class ConditionAccumulator:
    def __init__(
        self,
        seq_len,
        units,
        delta,
    ):
        self.encoder_h = CarryAccumulator(
            seq_len,
            units,
            delta,
            seq_len,
        )

        self.encoder_c = CarryAccumulator(
            seq_len,
            units,
            delta,
            seq_len,
        )

        self.decoder_h = CarryAccumulator(
            seq_len,
            units,
            delta,
            seq_len - 1,
        )

        self.decoder_c = CarryAccumulator(
            seq_len,
            units,
            delta,
            seq_len - 1,
        )

        self.handoff_h_abs_sum = 0.0
        self.handoff_c_abs_sum = 0.0
        self.handoff_h_count = 0
        self.handoff_c_count = 0
        self.handoff_h_changed = 0
        self.handoff_c_changed = 0

    def update(
        self,
        enc_h_raw,
        enc_h_before,
        enc_h_after,
        enc_c_raw,
        enc_c_before,
        enc_c_after,
        dec_h_raw,
        dec_h_before,
        dec_h_after,
        dec_c_raw,
        dec_c_before,
        dec_c_after,
        handoff_h_raw,
        handoff_h_read,
        handoff_c_raw,
        handoff_c_read,
    ):
        self.encoder_h.update(
            enc_h_raw,
            enc_h_before,
            enc_h_after,
        )

        self.encoder_c.update(
            enc_c_raw,
            enc_c_before,
            enc_c_after,
        )

        self.decoder_h.update(
            dec_h_raw,
            dec_h_before,
            dec_h_after,
        )

        self.decoder_c.update(
            dec_c_raw,
            dec_c_before,
            dec_c_after,
        )

        dh = np.abs(
            handoff_h_raw
            - handoff_h_read
        )

        dc = np.abs(
            handoff_c_raw
            - handoff_c_read
        )

        self.handoff_h_abs_sum += float(
            np.sum(dh)
        )

        self.handoff_c_abs_sum += float(
            np.sum(dc)
        )

        self.handoff_h_count += int(
            dh.size
        )

        self.handoff_c_count += int(
            dc.size
        )

        self.handoff_h_changed += int(
            np.sum(
                dh > 1e-7
            )
        )

        self.handoff_c_changed += int(
            np.sum(
                dc > 1e-7
            )
        )

    def summary(
        self,
    ):
        return {
            "encoder": {
                "h": (
                    self.encoder_h.summary()
                ),
                "c": (
                    self.encoder_c.summary()
                ),
            },
            "decoder": {
                "h": (
                    self.decoder_h.summary()
                ),
                "c": (
                    self.decoder_c.summary()
                ),
            },
            "handoff": {
                "mean_abs_error_h": float(
                    self.handoff_h_abs_sum
                    / max(
                        self.handoff_h_count,
                        1,
                    )
                ),
                "mean_abs_error_c": float(
                    self.handoff_c_abs_sum
                    / max(
                        self.handoff_c_count,
                        1,
                    )
                ),
                "changed_fraction_h": float(
                    self.handoff_h_changed
                    / max(
                        self.handoff_h_count,
                        1,
                    )
                ),
                "changed_fraction_c": float(
                    self.handoff_c_changed
                    / max(
                        self.handoff_c_count,
                        1,
                    )
                ),
            },
        }

    def per_sequence_arrays(
        self,
    ):
        out = {}

        out.update(
            self.encoder_h.per_sequence_arrays(
                "encoder_h"
            )
        )

        out.update(
            self.encoder_c.per_sequence_arrays(
                "encoder_c"
            )
        )

        out.update(
            self.decoder_h.per_sequence_arrays(
                "decoder_h"
            )
        )

        out.update(
            self.decoder_c.per_sequence_arrays(
                "decoder_c"
            )
        )

        return out


def build_reference_model(
    train_module,
    args,
):
    model = train_module.build_student(
        args.seq_len,
        args.n_out,
        args.student_units,
        args.bits_kernel,
        args.bits_recurrent,
        args.bits_bias,
        args.bits_activation,
        args.native_state_bits,
    )

    model.load_weights(
        args.checkpoint
    )

    model.trainable = False

    return model


def reference_predict(
    model,
    enc_arr,
    seq_len,
    n_out,
    batch_size,
):
    n = len(enc_arr)

    preds = np.empty(
        (
            n,
            seq_len,
            n_out,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n,
        )

        enc_b = tf.convert_to_tensor(
            enc_arr[start:stop],
            dtype=tf.float32,
        )

        dec_b = tf.zeros(
            (
                stop - start,
                seq_len,
                1,
            ),
            dtype=tf.float32,
        )

        preds[start:stop] = (
            model(
                [
                    enc_b,
                    dec_b,
                ],
                training=False,
            )
            .numpy()
        )

    return preds


def run_driver_predictions(
    runner,
    enc_arr,
    seq_len,
    n_out,
    batch_size,
    accumulator=None,
):
    n = len(enc_arr)

    preds = np.empty(
        (
            n,
            seq_len,
            n_out,
        ),
        dtype=np.float32,
    )

    n_batches = (
        n
        + batch_size
        - 1
    ) // batch_size

    for batch_index, start in enumerate(
        range(
            0,
            n,
            batch_size,
        ),
        start=1,
    ):
        stop = min(
            start + batch_size,
            n,
        )

        enc_b = tf.convert_to_tensor(
            enc_arr[start:stop],
            dtype=tf.float32,
        )

        outputs = runner(
            enc_b
        )

        preds[start:stop] = (
            outputs[0].numpy()
        )

        if accumulator is not None:
            arrays = [
                tensor.numpy()
                for tensor in outputs[1:]
            ]

            accumulator.update(
                *arrays
            )

        if (
            batch_index % 25 == 0
            or batch_index == n_batches
        ):
            print(
                "[DRIVER] "
                f"batch {batch_index}/"
                f"{n_batches}, "
                f"samples {stop:,}/{n:,}",
                flush=True,
            )

    return preds


def performance_summary(
    train_module,
    preds,
    res_test,
    lab_test,
    t_ns_axis,
):
    (
        tau1_pred,
        tau2_pred,
        fret_pred,
    ) = train_module.extract_lifetimes(
        preds,
        t_ns_axis,
    )

    tau1_gt = np.asarray(
        lab_test[:, 0],
        dtype=np.float32,
    )

    tau2_gt = np.asarray(
        lab_test[:, 1],
        dtype=np.float32,
    )

    fret_gt = np.asarray(
        lab_test[:, 2],
        dtype=np.float32,
    )

    summary = {
        "tau1_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        tau1_gt
                        - tau1_pred
                    )
                    ** 2
                )
            )
        ),
        "tau2_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        tau2_gt
                        - tau2_pred
                    )
                    ** 2
                )
            )
        ),
        "fret_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        fret_gt
                        - fret_pred
                    )
                    ** 2
                )
            )
        ),
        "tau1_r": safe_corr(
            tau1_gt,
            tau1_pred,
        ),
        "tau2_r": safe_corr(
            tau2_gt,
            tau2_pred,
        ),
        "fret_r": safe_corr(
            fret_gt,
            fret_pred,
        ),
        "mae_seq": float(
            np.mean(
                np.abs(
                    preds
                    - res_test
                )
            )
        ),
    }

    per_seq = {
        "tau1_abs_error": np.abs(
            tau1_gt
            - tau1_pred
        ).astype(
            np.float32
        ),
        "tau2_abs_error": np.abs(
            tau2_gt
            - tau2_pred
        ).astype(
            np.float32
        ),
        "fret_abs_error": np.abs(
            fret_gt
            - fret_pred
        ).astype(
            np.float32
        ),
    }

    return (
        summary,
        per_seq,
    )


def flatten_summary(
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
        "operator": summary[
            "operator"
        ],
        "state_bits": summary[
            "state_bits"
        ],
        "counterfactual_grid": summary[
            "counterfactual_grid"
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
        ),
        "forced": (
            "deterministic",
            args.forced_state_bits,
        ),
        "forced_ef": (
            "error_feedback",
            args.forced_state_bits,
        ),
        "identity": (
            "identity",
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

    native_runner = make_sequence_runner(
        enc_cell,
        dec_cell,
        dense_layer,
        args.seq_len,
        args.student_units,
        "deterministic",
        args.native_state_bits,
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
            "driver does not reproduce the "
            "intact QLSTM. Analysis aborted "
            "before interventions."
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
            operator,
            state_bits,
        ),
    ) in conditions.items():
        print(
            "=" * 72,
            flush=True,
        )

        print(
            f"[CONDITION] {name}: "
            f"operator={operator}, "
            f"state_bits={state_bits}",
            flush=True,
        )

        runner = make_sequence_runner(
            enc_cell,
            dec_cell,
            dense_layer,
            args.seq_len,
            args.student_units,
            operator,
            state_bits,
        )

        accumulator = (
            ConditionAccumulator(
                args.seq_len,
                args.student_units,
                grid_delta(
                    state_bits
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
            "operator": operator,
            "state_bits": int(
                state_bits
            ),
            "counterfactual_grid": bool(
                operator
                == "identity"
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
        / "lstm_state_writeback_summary.json",
    )

    rows = [
        flatten_summary(
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
        / "lstm_state_writeback_summary.csv",
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
            "writeback cross-architecture "
            "replication"
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
                "operator": operator,
                "state_bits": int(
                    bits
                ),
            }
            for (
                name,
                (
                    operator,
                    bits,
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
        "[DONE] LSTM state-writeback "
        "analysis completed",
        flush=True,
    )

    print(
        f"[DONE] outputs: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()