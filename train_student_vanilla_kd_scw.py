#!/usr/bin/env python3
"""
train_student_vanilla_kd_scw.py

Hard-forward saturating-counter writeback training for the 4-bit vanilla KD
Seq2SeqLite student.

The recurrent forward pass matches the validated repository reconstruction:

    q_recv_t -> h_t -> q_recv_{t+1}

with QKeras-quantized weights, hard-sigmoid GRU gates, quantized-tanh candidate
activation, raw decoder hidden states feeding the QDense output head, and a
4-bit recurrence-visible state.

SCW forward semantics match eval/analyze_recurrent_memory.py exactly:

    delta = h_t - q_recv_t
    normal write if |delta| >= Delta/2
    otherwise cast a signed vote when |delta| > theta
    trigger after T = 2^(k-1) net signed votes
    emit one +/-Delta state step on trigger and reset the counter
    preserve the counter inside the dead-zone
    reset the decoder counter at encoder-to-decoder handoff

Training uses the exact hard SCW state in the forward pass. The gradient through
recurrent state writeback uses an identity straight-through estimator:

    q_ste = h + stop_gradient(q_hard - h)

The counter is a hard, non-trainable recurrent auxiliary state and receives no
gradient. This is a quantization-aware training surrogate, not a relaxation of
the forward operator.

The script is fail-closed on missing data/checkpoints, supports exact
model/optimizer/scheduler resume checkpoints, saves a standard student_args.json
for downstream analysis, evaluates the trained checkpoint under SCW,
deterministic B4, and identity state propagation, and writes per-sequence
predictions for paired analysis.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
from tensorflow import keras
from qkeras import QDense, quantized_bits, quantized_tanh

from train_student_memoq import qkeras_hard_sigmoid
from train_student_vanilla_kd import (
    build_student,
    build_teacher,
    cache_teacher_predictions,
    find_data_files,
    materialise_enc_tgt_tpred,
    make_kd_dataset,
    setup_gpus_and_strategy,
)


STOP_AFTER_EPOCH = False


def pf(message: str = "") -> None:
    print(message, flush=True)


def handle_usr1(signum, frame) -> None:
    del signum, frame
    global STOP_AFTER_EPOCH
    STOP_AFTER_EPOCH = True
    pf(
        "[SIGNAL] SIGUSR1 received. Training will stop after the current "
        "epoch checkpoint is saved."
    )


signal.signal(signal.SIGUSR1, handle_usr1)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_write_json(
    path: Path,
    payload: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
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

        handle.write("\n")
        handle.flush()
        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 4-bit vanilla KD student with hard-forward "
            "SCW recurrent-state writeback."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--save-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--teacher-ckpt",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--teacher-units",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--teacher-layers",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--student-units",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--seq-len",
        default=135,
        type=int,
    )

    parser.add_argument(
        "--n-out",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--gate-width-ns",
        default=0.09,
        type=float,
    )

    parser.add_argument(
        "--temperature",
        default=4.0,
        type=float,
    )

    parser.add_argument(
        "--alpha",
        default=0.6,
        type=float,
    )

    parser.add_argument(
        "--bits-kernel",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--bits-bias",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--bits-recurrent",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--bits-activation",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--bits-state",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--q-alpha",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--counter-bits",
        default=4,
        choices=(
            2,
            3,
            4,
        ),
        type=int,
    )

    parser.add_argument(
        "--scw-deadzone-fraction",
        default=0.125,
        choices=(
            0.0,
            0.125,
        ),
        type=float,
    )

    parser.add_argument(
        "--batch-size",
        default=1024,
        type=int,
    )

    parser.add_argument(
        "--accumulation-steps",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--patience",
        default=15,
        type=int,
    )

    parser.add_argument(
        "--min-delta",
        default=1e-5,
        type=float,
    )

    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--ref-batch-size",
        default=1024,
        type=int,
    )

    parser.add_argument(
        "--no-lr-scaling",
        action="store_true",
    )

    parser.add_argument(
        "--lr-factor",
        default=0.5,
        type=float,
    )

    parser.add_argument(
        "--lr-patience",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--lr-min",
        default=1e-6,
        type=float,
    )

    parser.add_argument(
        "--warmup-epochs",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--infer-batch",
        default=4096,
        type=int,
    )

    parser.add_argument(
        "--log-interval",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--prefetch-batches",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--pipeline-workers",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--split-seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--mixed-precision",
        action="store_true",
    )

    args = parser.parse_args()

    if args.seq_len <= 1:
        raise ValueError(
            "--seq-len must be > 1"
        )

    if args.n_out < 3:
        raise ValueError(
            "--n-out must be >= 3"
        )

    if args.student_units <= 0:
        raise ValueError(
            "--student-units must be > 0"
        )

    if (
        args.teacher_units <= 0
        or args.teacher_layers <= 0
    ):
        raise ValueError(
            "Teacher dimensions must be > 0"
        )

    if args.bits_state != 4:
        raise ValueError(
            "This paper training run is defined for a 4-bit "
            "recurrence-visible state: --bits-state must be 4"
        )

    for name in (
        "bits_kernel",
        "bits_bias",
        "bits_recurrent",
        "bits_activation",
    ):
        if getattr(
            args,
            name,
        ) != 4:
            raise ValueError(
                f"This paper training run requires "
                f"--{name.replace('_', '-')} 4"
            )

    if args.q_alpha != 1.0:
        raise ValueError(
            "This vanilla SCW run requires --q-alpha 1.0 "
            "to match the vanilla QKeras quantizers"
        )

    if not (
        0.0
        <= args.alpha
        <= 1.0
    ):
        raise ValueError(
            "--alpha must be in [0,1]"
        )

    if args.temperature <= 0.0:
        raise ValueError(
            "--temperature must be > 0"
        )

    if (
        args.batch_size <= 0
        or args.accumulation_steps <= 0
    ):
        raise ValueError(
            "Batch size and accumulation steps must be > 0"
        )

    if (
        args.batch_size
        % args.accumulation_steps
        != 0
    ):
        raise ValueError(
            "--batch-size must be exactly divisible by "
            "--accumulation-steps"
        )

    if (
        args.epochs <= 0
        or args.patience <= 0
    ):
        raise ValueError(
            "Epochs and patience must be > 0"
        )

    if (
        args.lr <= 0.0
        or args.ref_batch_size <= 0
    ):
        raise ValueError(
            "Learning rate and reference batch size must be > 0"
        )

    if not (
        0.0
        < args.lr_factor
        < 1.0
    ):
        raise ValueError(
            "--lr-factor must be in (0,1)"
        )

    if (
        args.lr_patience <= 0
        or args.lr_min <= 0.0
    ):
        raise ValueError(
            "LR patience and minimum LR must be > 0"
        )

    if args.warmup_epochs < 0:
        raise ValueError(
            "--warmup-epochs must be >= 0"
        )

    if args.infer_batch <= 0:
        raise ValueError(
            "--infer-batch must be > 0"
        )

    data_dir = Path(
        args.data_dir
    ).resolve()

    save_dir = Path(
        args.save_dir
    ).resolve()

    teacher_ckpt = Path(
        args.teacher_ckpt
    ).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{data_dir}"
        )

    if not save_dir.is_dir():
        raise FileNotFoundError(
            f"Save directory does not exist: "
            f"{save_dir}"
        )

    if not teacher_ckpt.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint does not exist: "
            f"{teacher_ckpt}"
        )

    args.data_dir = str(
        data_dir
    )

    args.save_dir = str(
        save_dir
    )

    args.teacher_ckpt = str(
        teacher_ckpt
    )

    return args


def deadzone_tag(
    value: float,
) -> str:
    text = (
        f"{float(value):.6f}"
        .rstrip("0")
        .rstrip(".")
    )

    if text == "":
        text = "0"

    return (
        text
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
    )


def make_job_name(
    args: argparse.Namespace,
) -> str:
    micro_batch = (
        args.batch_size
        // args.accumulation_steps
    )

    return (
        f"vanilla_scw_kd"
        f"_T{args.temperature}"
        f"_a{args.alpha}"
        f"_b{args.bits_kernel}"
        f"k{args.bits_bias}"
        f"r{args.bits_recurrent}"
        f"a{args.bits_activation}"
        f"s{args.bits_state}"
        f"_scwK{args.counter_bits}"
        f"_th{deadzone_tag(args.scw_deadzone_fraction)}"
        f"_gru{args.student_units}x1"
        f"_dense{args.n_out}"
        f"_effbs{args.batch_size}"
        f"_microbs{micro_batch}"
        f"_lr{args.effective_lr:.0e}"
    )


class CheckpointableReduceLROnPlateau(
    tf.Module
):
    def __init__(
        self,
        optimizer,
        factor,
        patience,
        min_lr,
        min_delta,
    ):
        super().__init__(
            name="reduce_lr_on_plateau"
        )

        self.optimizer = optimizer
        self.factor = float(
            factor
        )
        self.patience = int(
            patience
        )
        self.min_lr = float(
            min_lr
        )
        self.min_delta = float(
            min_delta
        )

        if callable(
            optimizer.learning_rate
        ):
            current_lr = float(
                optimizer.learning_rate(
                    optimizer.iterations
                )
            )

        else:
            current_lr = float(
                tf.keras.backend.get_value(
                    optimizer.learning_rate
                )
            )

        self.best_var = tf.Variable(
            float("inf"),
            trainable=False,
            dtype=tf.float64,
            name="best_validation_loss",
        )

        self.wait_var = tf.Variable(
            0,
            trainable=False,
            dtype=tf.int64,
            name="plateau_wait",
        )

        self.lr_var = tf.Variable(
            current_lr,
            trainable=False,
            dtype=tf.float32,
            name="learning_rate",
        )

        optimizer.learning_rate = (
            self.lr_var
        )

    @property
    def best(
        self,
    ) -> float:
        return float(
            self.best_var.numpy()
        )

    @best.setter
    def best(
        self,
        value: float,
    ) -> None:
        self.best_var.assign(
            float(
                value
            )
        )

    @property
    def wait(
        self,
    ) -> int:
        return int(
            self.wait_var.numpy()
        )

    @wait.setter
    def wait(
        self,
        value: int,
    ) -> None:
        self.wait_var.assign(
            int(
                value
            )
        )

    @property
    def current_lr(
        self,
    ) -> float:
        return float(
            tf.keras.backend.get_value(
                self.lr_var
            )
        )

    def step(
        self,
        val_loss: float,
        epoch: int,
    ) -> bool:
        del epoch

        val_loss = float(
            val_loss
        )

        if (
            val_loss
            < self.best
            - self.min_delta
        ):
            self.best = val_loss
            self.wait = 0
            return False

        self.wait = (
            self.wait
            + 1
        )

        if (
            self.wait
            < self.patience
        ):
            return False

        old_lr = (
            self.current_lr
        )

        new_lr = max(
            old_lr
            * self.factor,
            self.min_lr,
        )

        if new_lr < old_lr:
            self.lr_var.assign(
                float(
                    new_lr
                )
            )

            pf(
                f"[LR] ReduceLROnPlateau "
                f"{old_lr:.3e} -> "
                f"{new_lr:.3e}"
            )

        self.wait = 0
        return True


class QuantizedSCWGRUCore(
    keras.layers.Layer
):
    def __init__(
        self,
        units: int,
        bits_kernel: int,
        bits_recurrent: int,
        bits_bias: int,
        bits_activation: int,
        bits_state: int,
        counter_bits: int,
        deadzone_fraction: float,
        q_alpha: float,
        name: str,
    ) -> None:
        super().__init__(
            name=name
        )

        self.units = int(
            units
        )

        self.bits_kernel = int(
            bits_kernel
        )

        self.bits_recurrent = int(
            bits_recurrent
        )

        self.bits_bias = int(
            bits_bias
        )

        self.bits_activation = int(
            bits_activation
        )

        self.bits_state = int(
            bits_state
        )

        self.counter_bits = int(
            counter_bits
        )

        self.deadzone_fraction = float(
            deadzone_fraction
        )

        self.q_alpha = float(
            q_alpha
        )

        self.delta = float(
            2.0
            ** (
                -(
                    self.bits_state
                    - 1
                )
            )
        )

        self.half_step = float(
            self.delta
            / 2.0
        )

        self.qmax = float(
            1.0
            - self.delta
        )

        self.qmin = float(
            -self.qmax
        )

        self.trigger_votes = int(
            2
            ** (
                self.counter_bits
                - 1
            )
        )

        self.counter_min = int(
            -(
                self.trigger_votes
                - 1
            )
        )

        self.counter_max = int(
            self.trigger_votes
            - 1
        )

        self.deadzone = float(
            self.deadzone_fraction
            * self.delta
        )

        self.kernel_quantizer = (
            quantized_bits(
                self.bits_kernel,
                0,
                1,
                alpha=self.q_alpha,
            )
        )

        self.recurrent_quantizer = (
            quantized_bits(
                self.bits_recurrent,
                0,
                1,
                alpha=self.q_alpha,
            )
        )

        self.bias_quantizer = (
            quantized_bits(
                self.bits_bias,
                0,
                1,
                alpha=self.q_alpha,
            )
        )

        self.activation_quantizer = (
            quantized_tanh(
                bits=self.bits_activation,
                symmetric=True,
            )
        )

        self.state_quantizer = (
            quantized_bits(
                self.bits_state,
                0,
                1,
                alpha=1.0,
            )
        )

    def build(
        self,
        input_shape,
    ) -> None:
        input_dim = int(
            input_shape[-1]
        )

        self.kernel = (
            self.add_weight(
                name="kernel",
                shape=(
                    input_dim,
                    3
                    * self.units,
                ),
                initializer="glorot_uniform",
                trainable=True,
                dtype=tf.float32,
            )
        )

        self.recurrent_kernel = (
            self.add_weight(
                name="recurrent_kernel",
                shape=(
                    self.units,
                    3
                    * self.units,
                ),
                initializer="orthogonal",
                trainable=True,
                dtype=tf.float32,
            )
        )

        self.bias = (
            self.add_weight(
                name="bias",
                shape=(
                    3
                    * self.units,
                ),
                initializer="zeros",
                trainable=True,
                dtype=tf.float32,
            )
        )

        super().build(
            input_shape
        )

    def effective_parameters(
        self,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        kernel_q = tf.cast(
            self.kernel_quantizer(
                self.kernel
            ),
            tf.float32,
        )

        recurrent_q = tf.cast(
            self.recurrent_quantizer(
                self.recurrent_kernel
            ),
            tf.float32,
        )

        bias_q = tf.cast(
            self.bias_quantizer(
                self.bias
            ),
            tf.float32,
        )

        return (
            kernel_q,
            recurrent_q,
            bias_q,
        )

    def deterministic_quantize_state(
        self,
        value: tf.Tensor,
    ) -> tf.Tensor:
        return tf.cast(
            self.state_quantizer(
                tf.cast(
                    value,
                    tf.float32,
                )
            ),
            tf.float32,
        )

    @staticmethod
    def identity_ste(
        raw_state: tf.Tensor,
        hard_state: tf.Tensor,
    ) -> tf.Tensor:
        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        hard_state = tf.cast(
            hard_state,
            tf.float32,
        )

        return (
            raw_state
            + tf.stop_gradient(
                hard_state
                - raw_state
            )
        )

    def gru_step(
        self,
        x_t: tf.Tensor,
        q_recv: tf.Tensor,
        kernel_q: tf.Tensor,
        recurrent_q: tf.Tensor,
        bias_q: tf.Tensor,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        H = self.units

        x_t = tf.cast(
            x_t,
            tf.float32,
        )

        q_recv = tf.cast(
            q_recv,
            tf.float32,
        )

        x_z = (
            tf.matmul(
                x_t,
                kernel_q[
                    :,
                    :H
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
                    2 * H
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
                        :H
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
                        2 * H
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

        candidate = tf.cast(
            self.activation_quantizer(
                preact
            ),
            tf.float32,
        )

        raw_state = (
            z
            * q_recv
            + (
                1.0
                - z
            )
            * candidate
        )

        return (
            raw_state,
            z,
            r,
            candidate,
        )

    def initialize_state(
        self,
        raw_state: tf.Tensor,
        operator_mode: str,
        use_ste: bool,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        counter = tf.zeros_like(
            raw_state,
            dtype=tf.float32,
        )

        if (
            operator_mode
            == "identity"
        ):
            hard_state = (
                raw_state
            )

        elif operator_mode in (
            "deterministic",
            "scw",
        ):
            hard_state = (
                self.deterministic_quantize_state(
                    raw_state
                )
            )

        else:
            raise ValueError(
                f"Unsupported operator_mode="
                f"{operator_mode!r}"
            )

        if use_ste:
            state = (
                self.identity_ste(
                    raw_state,
                    hard_state,
                )
            )

        else:
            state = (
                hard_state
            )

        return (
            state,
            tf.stop_gradient(
                counter
            ),
            tf.stop_gradient(
                hard_state
            ),
        )

    def hard_advance(
        self,
        raw_state: tf.Tensor,
        q_prev_hard: tf.Tensor,
        counter_prev: tf.Tensor,
        operator_mode: str,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
    ]:
        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        q_prev_hard = (
            tf.stop_gradient(
                tf.cast(
                    q_prev_hard,
                    tf.float32,
                )
            )
        )

        counter_prev = (
            tf.stop_gradient(
                tf.cast(
                    counter_prev,
                    tf.float32,
                )
            )
        )

        if (
            operator_mode
            == "identity"
        ):
            return (
                tf.stop_gradient(
                    raw_state
                ),
                tf.zeros_like(
                    counter_prev
                ),
            )

        if (
            operator_mode
            == "deterministic"
        ):
            return (
                tf.stop_gradient(
                    self.deterministic_quantize_state(
                        raw_state
                    )
                ),
                tf.zeros_like(
                    counter_prev
                ),
            )

        if (
            operator_mode
            != "scw"
        ):
            raise ValueError(
                f"Unsupported operator_mode="
                f"{operator_mode!r}"
            )

        counter_rounded = (
            tf.round(
                counter_prev
            )
        )

        tf.debugging.assert_near(
            counter_prev,
            counter_rounded,
            atol=1e-6,
            rtol=0.0,
            message=(
                "SCW counter lost "
                "integer semantics"
            ),
        )

        tf.debugging.assert_less_equal(
            counter_rounded,
            tf.constant(
                float(
                    self.counter_max
                ),
                tf.float32,
            ),
            message=(
                "SCW counter exceeded "
                "positive stored range"
            ),
        )

        tf.debugging.assert_greater_equal(
            counter_rounded,
            tf.constant(
                float(
                    self.counter_min
                ),
                tf.float32,
            ),
            message=(
                "SCW counter exceeded "
                "negative stored range"
            ),
        )

        delta_raw = (
            raw_state
            - q_prev_hard
        )

        abs_delta = tf.abs(
            delta_raw
        )

        normal = (
            abs_delta
            >= tf.constant(
                self.half_step,
                tf.float32,
            )
        )

        subthreshold = (
            ~normal
        )

        active_vote = (
            subthreshold
            & (
                abs_delta
                > tf.constant(
                    self.deadzone,
                    tf.float32,
                )
            )
        )

        vote = tf.sign(
            delta_raw
        )

        counter_candidate = (
            counter_rounded
            + vote
        )

        positive_trigger = (
            active_vote
            & (
                counter_candidate
                >= tf.constant(
                    float(
                        self.trigger_votes
                    ),
                    tf.float32,
                )
            )
        )

        negative_trigger = (
            active_vote
            & (
                counter_candidate
                <= tf.constant(
                    float(
                        -self.trigger_votes
                    ),
                    tf.float32,
                )
            )
        )

        trigger = (
            positive_trigger
            | negative_trigger
        )

        q_normal = (
            self.deterministic_quantize_state(
                raw_state
            )
        )

        forced_direction = (
            tf.cast(
                positive_trigger,
                tf.float32,
            )
            - tf.cast(
                negative_trigger,
                tf.float32,
            )
        )

        q_forced = (
            tf.clip_by_value(
                q_prev_hard
                + forced_direction
                * tf.constant(
                    self.delta,
                    tf.float32,
                ),
                tf.constant(
                    self.qmin,
                    tf.float32,
                ),
                tf.constant(
                    self.qmax,
                    tf.float32,
                ),
            )
        )

        q_next_hard = (
            tf.where(
                normal,
                q_normal,
                tf.where(
                    trigger,
                    q_forced,
                    q_prev_hard,
                ),
            )
        )

        counter_active_hold = (
            tf.clip_by_value(
                counter_candidate,
                tf.constant(
                    float(
                        self.counter_min
                    ),
                    tf.float32,
                ),
                tf.constant(
                    float(
                        self.counter_max
                    ),
                    tf.float32,
                ),
            )
        )

        counter_next = (
            tf.where(
                normal
                | trigger,
                tf.zeros_like(
                    counter_prev
                ),
                tf.where(
                    active_vote,
                    counter_active_hold,
                    counter_rounded,
                ),
            )
        )

        return (
            tf.stop_gradient(
                q_next_hard
            ),
            tf.stop_gradient(
                counter_next
            ),
        )

    def advance_state(
        self,
        raw_state: tf.Tensor,
        q_prev_hard: tf.Tensor,
        counter_prev: tf.Tensor,
        operator_mode: str,
        use_ste: bool,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
    ]:
        (
            q_next_hard,
            counter_next,
        ) = self.hard_advance(
            raw_state=raw_state,
            q_prev_hard=q_prev_hard,
            counter_prev=counter_prev,
            operator_mode=operator_mode,
        )

        if (
            operator_mode
            == "identity"
        ):
            state = (
                raw_state
            )

        elif use_ste:
            state = (
                self.identity_ste(
                    raw_state,
                    q_next_hard,
                )
            )

        else:
            state = (
                q_next_hard
            )

        return (
            state,
            counter_next,
            q_next_hard,
        )

    def metadata(
        self,
    ) -> Dict:
        return {
            "state_bits": (
                self.bits_state
            ),
            "delta": (
                self.delta
            ),
            "half_step": (
                self.half_step
            ),
            "qmin": (
                self.qmin
            ),
            "qmax": (
                self.qmax
            ),
            "counter_bits": (
                self.counter_bits
            ),
            "trigger_votes": (
                self.trigger_votes
            ),
            "stored_counter_min": (
                self.counter_min
            ),
            "stored_counter_max": (
                self.counter_max
            ),
            "stored_active_counter_states": (
                2
                * self.trigger_votes
                - 1
            ),
            "unused_binary_codewords": (
                1
            ),
            "deadzone_fraction_of_delta": (
                self.deadzone_fraction
            ),
            "deadzone_absolute": (
                self.deadzone
            ),
            "forward_operator": (
                "hard SCW"
            ),
            "backward_state_surrogate": (
                "identity straight-through estimator"
            ),
            "counter_gradient": (
                "none"
            ),
        }


class SCWStudentModel(
    keras.Model
):
    def __init__(
        self,
        seq_len: int,
        n_out: int,
        student_units: int,
        bits_kernel: int,
        bits_recurrent: int,
        bits_bias: int,
        bits_activation: int,
        bits_state: int,
        counter_bits: int,
        deadzone_fraction: float,
        q_alpha: float,
    ) -> None:
        super().__init__(
            name="student_vanilla_kd_scw"
        )

        self.seq_len = int(
            seq_len
        )

        self.n_out = int(
            n_out
        )

        self.student_units = int(
            student_units
        )

        self.sencgru = (
            QuantizedSCWGRUCore(
                units=student_units,
                bits_kernel=bits_kernel,
                bits_recurrent=bits_recurrent,
                bits_bias=bits_bias,
                bits_activation=bits_activation,
                bits_state=bits_state,
                counter_bits=counter_bits,
                deadzone_fraction=deadzone_fraction,
                q_alpha=q_alpha,
                name="sencgru",
            )
        )

        self.sdecgru = (
            QuantizedSCWGRUCore(
                units=student_units,
                bits_kernel=bits_kernel,
                bits_recurrent=bits_recurrent,
                bits_bias=bits_bias,
                bits_activation=bits_activation,
                bits_state=bits_state,
                counter_bits=counter_bits,
                deadzone_fraction=deadzone_fraction,
                q_alpha=q_alpha,
                name="sdecgru",
            )
        )

        dense_quantizer = (
            quantized_bits(
                bits_kernel,
                0,
            )
        )

        self.sdec_dense = (
            QDense(
                n_out,
                kernel_quantizer=dense_quantizer,
                bias_quantizer=(
                    quantized_bits(
                        bits_kernel,
                        0,
                    )
                ),
                activation="linear",
                name="sdec_dense",
            )
        )

    def call(
        self,
        inputs,
        training=False,
        operator_mode: str = "scw",
    ) -> tf.Tensor:
        if (
            operator_mode
            not in (
                "scw",
                "deterministic",
                "identity",
            )
        ):
            raise ValueError(
                f"Unsupported operator_mode="
                f"{operator_mode!r}"
            )

        (
            enc_inputs,
            dec_inputs,
        ) = inputs

        enc_inputs = tf.cast(
            enc_inputs,
            tf.float32,
        )

        dec_inputs = tf.cast(
            dec_inputs,
            tf.float32,
        )

        tf.debugging.assert_equal(
            tf.shape(
                enc_inputs
            )[1],
            self.seq_len,
            message=(
                "Encoder sequence length "
                "does not match model seq_len"
            ),
        )

        tf.debugging.assert_equal(
            tf.shape(
                dec_inputs
            )[1],
            self.seq_len,
            message=(
                "Decoder sequence length "
                "does not match model seq_len"
            ),
        )

        use_ste = bool(
            training
        )

        batch = tf.shape(
            enc_inputs
        )[0]

        H = (
            self.student_units
        )

        seq_len_tensor = (
            tf.constant(
                self.seq_len,
                tf.int32,
            )
        )

        final_live_index = (
            tf.constant(
                self.seq_len
                - 1,
                tf.int32,
            )
        )

        (
            enc_kernel_q,
            enc_recurrent_q,
            enc_bias_q,
        ) = (
            self.sencgru
            .effective_parameters()
        )

        (
            dec_kernel_q,
            dec_recurrent_q,
            dec_bias_q,
        ) = (
            self.sdecgru
            .effective_parameters()
        )

        zero_raw = tf.zeros(
            (
                batch,
                H,
            ),
            tf.float32,
        )

        (
            q_enc,
            counter_enc,
            q_enc_hard,
        ) = (
            self.sencgru
            .initialize_state(
                zero_raw,
                operator_mode=operator_mode,
                use_ste=use_ste,
            )
        )

        def encoder_cond(
            i,
            raw_state,
            q_state,
            counter_state,
            q_hard_state,
        ):
            del (
                raw_state,
                q_state,
                counter_state,
                q_hard_state,
            )

            return (
                i
                < seq_len_tensor
            )

        def encoder_body(
            i,
            raw_state,
            q_state,
            counter_state,
            q_hard_state,
        ):
            del raw_state

            (
                raw_next,
                _,
                _,
                _,
            ) = (
                self.sencgru
                .gru_step(
                    enc_inputs[
                        :,
                        i,
                        :
                    ],
                    q_state,
                    enc_kernel_q,
                    enc_recurrent_q,
                    enc_bias_q,
                )
            )

            def advance_live_state():
                return (
                    self.sencgru
                    .advance_state(
                        raw_state=raw_next,
                        q_prev_hard=q_hard_state,
                        counter_prev=counter_state,
                        operator_mode=operator_mode,
                        use_ste=use_ste,
                    )
                )

            def keep_terminal_state():
                return (
                    q_state,
                    counter_state,
                    q_hard_state,
                )

            (
                q_next,
                counter_next,
                q_hard_next,
            ) = tf.cond(
                i
                < final_live_index,
                advance_live_state,
                keep_terminal_state,
            )

            return (
                i + 1,
                raw_next,
                q_next,
                counter_next,
                q_hard_next,
            )

        (
            _,
            raw_enc,
            _,
            _,
            _,
        ) = tf.while_loop(
            encoder_cond,
            encoder_body,
            loop_vars=(
                tf.constant(
                    0,
                    tf.int32,
                ),
                zero_raw,
                q_enc,
                counter_enc,
                q_enc_hard,
            ),
            parallel_iterations=1,
            maximum_iterations=(
                self.seq_len
            ),
        )

        (
            q_dec,
            counter_dec,
            q_dec_hard,
        ) = (
            self.sdecgru
            .initialize_state(
                raw_enc,
                operator_mode=operator_mode,
                use_ste=use_ste,
            )
        )

        dec_hidden_ta = (
            tf.TensorArray(
                dtype=tf.float32,
                size=self.seq_len,
                clear_after_read=False,
                element_shape=(
                    tf.TensorShape(
                        [
                            None,
                            self.student_units,
                        ]
                    )
                ),
            )
        )

        def decoder_cond(
            i,
            raw_state,
            q_state,
            counter_state,
            q_hard_state,
            hidden_ta,
        ):
            del (
                raw_state,
                q_state,
                counter_state,
                q_hard_state,
                hidden_ta,
            )

            return (
                i
                < seq_len_tensor
            )

        def decoder_body(
            i,
            raw_state,
            q_state,
            counter_state,
            q_hard_state,
            hidden_ta,
        ):
            del raw_state

            (
                raw_next,
                _,
                _,
                _,
            ) = (
                self.sdecgru
                .gru_step(
                    dec_inputs[
                        :,
                        i,
                        :
                    ],
                    q_state,
                    dec_kernel_q,
                    dec_recurrent_q,
                    dec_bias_q,
                )
            )

            hidden_ta = (
                hidden_ta.write(
                    i,
                    raw_next,
                )
            )

            def advance_live_state():
                return (
                    self.sdecgru
                    .advance_state(
                        raw_state=raw_next,
                        q_prev_hard=q_hard_state,
                        counter_prev=counter_state,
                        operator_mode=operator_mode,
                        use_ste=use_ste,
                    )
                )

            def keep_terminal_state():
                return (
                    q_state,
                    counter_state,
                    q_hard_state,
                )

            (
                q_next,
                counter_next,
                q_hard_next,
            ) = tf.cond(
                i
                < final_live_index,
                advance_live_state,
                keep_terminal_state,
            )

            return (
                i + 1,
                raw_next,
                q_next,
                counter_next,
                q_hard_next,
                hidden_ta,
            )

        (
            _,
            _,
            _,
            _,
            _,
            dec_hidden_ta,
        ) = tf.while_loop(
            decoder_cond,
            decoder_body,
            loop_vars=(
                tf.constant(
                    0,
                    tf.int32,
                ),
                raw_enc,
                q_dec,
                counter_dec,
                q_dec_hard,
                dec_hidden_ta,
            ),
            parallel_iterations=1,
            maximum_iterations=(
                self.seq_len
            ),
        )

        dec_hidden = tf.transpose(
            dec_hidden_ta.stack(),
            perm=(
                1,
                0,
                2,
            ),
        )

        return (
            self.sdec_dense(
                dec_hidden
            )
        )

    @tf.function(
        reduce_retracing=True
    )
    def diagnostic_forward(
        self,
        enc_inputs: tf.Tensor,
        dec_inputs: tf.Tensor,
        operator_code: tf.Tensor,
    ):
        enc_inputs = tf.cast(
            enc_inputs,
            tf.float32,
        )

        dec_inputs = tf.cast(
            dec_inputs,
            tf.float32,
        )

        operator_code = tf.cast(
            operator_code,
            tf.int32,
        )

        tf.debugging.assert_greater_equal(
            operator_code,
            0,
        )

        tf.debugging.assert_less_equal(
            operator_code,
            2,
        )

        batch = tf.shape(
            enc_inputs
        )[0]

        H = self.student_units

        delta = tf.constant(
            self.sencgru.delta,
            tf.float32,
        )

        half = tf.constant(
            self.sencgru.half_step,
            tf.float32,
        )

        change_eps = (
            delta
            * tf.constant(
                1e-4,
                tf.float32,
            )
        )

        seq_len_tensor = (
            tf.constant(
                self.seq_len,
                tf.int32,
            )
        )

        final_live_index = (
            tf.constant(
                self.seq_len
                - 1,
                tf.int32,
            )
        )

        (
            enc_kernel_q,
            enc_recurrent_q,
            enc_bias_q,
        ) = (
            self.sencgru
            .effective_parameters()
        )

        (
            dec_kernel_q,
            dec_recurrent_q,
            dec_bias_q,
        ) = (
            self.sdecgru
            .effective_parameters()
        )

        def hard_initialize(
            core,
            raw_state,
        ):
            q_det = (
                core
                .deterministic_quantize_state(
                    raw_state
                )
            )

            q = tf.switch_case(
                operator_code,
                branch_fns={
                    0: lambda: q_det,
                    1: lambda: q_det,
                    2: lambda: tf.cast(
                        raw_state,
                        tf.float32,
                    ),
                },
            )

            return (
                tf.stop_gradient(
                    q
                ),
                tf.zeros_like(
                    raw_state
                ),
            )

        def hard_advance_switch(
            core,
            raw_state,
            q_prev_hard,
            counter_prev,
        ):
            return tf.switch_case(
                operator_code,
                branch_fns={
                    0: lambda: (
                        core
                        .hard_advance(
                            raw_state,
                            q_prev_hard,
                            counter_prev,
                            "scw",
                        )
                    ),
                    1: lambda: (
                        core
                        .hard_advance(
                            raw_state,
                            q_prev_hard,
                            counter_prev,
                            "deterministic",
                        )
                    ),
                    2: lambda: (
                        core
                        .hard_advance(
                            raw_state,
                            q_prev_hard,
                            counter_prev,
                            "identity",
                        )
                    ),
                },
            )

        zero_raw = tf.zeros(
            (
                batch,
                H,
            ),
            tf.float32,
        )

        (
            q_enc_hard,
            counter_enc,
        ) = hard_initialize(
            self.sencgru,
            zero_raw,
        )

        def enc_cond(
            i,
            raw_state,
            q_state,
            counter_state,
            deadband_count,
            write_count,
            subthreshold_write_count,
            zero_step_count,
            nwrite_sum,
        ):
            del (
                raw_state,
                q_state,
                counter_state,
                deadband_count,
                write_count,
                subthreshold_write_count,
                zero_step_count,
                nwrite_sum,
            )

            return (
                i
                < seq_len_tensor
            )

        def enc_body(
            i,
            raw_state,
            q_state,
            counter_state,
            deadband_count,
            write_count,
            subthreshold_write_count,
            zero_step_count,
            nwrite_sum,
        ):
            del raw_state

            (
                raw_next,
                _,
                _,
                _,
            ) = (
                self.sencgru
                .gru_step(
                    enc_inputs[
                        :,
                        i,
                        :
                    ],
                    q_state,
                    enc_kernel_q,
                    enc_recurrent_q,
                    enc_bias_q,
                )
            )

            def live_update():
                (
                    q_next,
                    counter_next,
                ) = (
                    hard_advance_switch(
                        self.sencgru,
                        raw_next,
                        q_state,
                        counter_state,
                    )
                )

                deadband = (
                    tf.abs(
                        raw_next
                        - q_state
                    )
                    < half
                )

                changed = (
                    tf.abs(
                        q_next
                        - q_state
                    )
                    > change_eps
                )

                nwrite = (
                    tf.reduce_sum(
                        tf.cast(
                            changed,
                            tf.int64,
                        ),
                        axis=1,
                    )
                )

                return (
                    q_next,
                    counter_next,
                    deadband_count
                    + tf.reduce_sum(
                        tf.cast(
                            deadband,
                            tf.int64,
                        )
                    ),
                    write_count
                    + tf.reduce_sum(
                        tf.cast(
                            changed,
                            tf.int64,
                        )
                    ),
                    subthreshold_write_count
                    + tf.reduce_sum(
                        tf.cast(
                            deadband
                            & changed,
                            tf.int64,
                        )
                    ),
                    zero_step_count
                    + tf.reduce_sum(
                        tf.cast(
                            nwrite
                            == 0,
                            tf.int64,
                        )
                    ),
                    nwrite_sum
                    + tf.reduce_sum(
                        nwrite
                    ),
                )

            def terminal_update():
                return (
                    q_state,
                    counter_state,
                    deadband_count,
                    write_count,
                    subthreshold_write_count,
                    zero_step_count,
                    nwrite_sum,
                )

            (
                q_next,
                counter_next,
                deadband_next,
                write_next,
                subthreshold_next,
                zero_step_next,
                nwrite_sum_next,
            ) = tf.cond(
                i
                < final_live_index,
                live_update,
                terminal_update,
            )

            return (
                i + 1,
                raw_next,
                q_next,
                counter_next,
                deadband_next,
                write_next,
                subthreshold_next,
                zero_step_next,
                nwrite_sum_next,
            )

        (
            _,
            raw_enc,
            _,
            _,
            enc_deadband_count,
            enc_write_count,
            enc_subthreshold_write_count,
            enc_zero_step_count,
            enc_nwrite_sum,
        ) = tf.while_loop(
            enc_cond,
            enc_body,
            loop_vars=(
                tf.constant(
                    0,
                    tf.int32,
                ),
                zero_raw,
                q_enc_hard,
                counter_enc,
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
            ),
            parallel_iterations=1,
            maximum_iterations=(
                self.seq_len
            ),
        )

        (
            q_dec_hard,
            counter_dec,
        ) = hard_initialize(
            self.sdecgru,
            raw_enc,
        )

        handoff_abs_sum = (
            tf.reduce_sum(
                tf.abs(
                    q_dec_hard
                    - raw_enc
                )
            )
        )

        dec_hidden_ta = (
            tf.TensorArray(
                dtype=tf.float32,
                size=self.seq_len,
                clear_after_read=False,
                element_shape=(
                    tf.TensorShape(
                        [
                            None,
                            self.student_units,
                        ]
                    )
                ),
            )
        )

        def dec_cond(
            i,
            raw_state,
            q_state,
            counter_state,
            hidden_ta,
            deadband_count,
            write_count,
            subthreshold_write_count,
            zero_step_count,
            nwrite_sum,
        ):
            del (
                raw_state,
                q_state,
                counter_state,
                hidden_ta,
                deadband_count,
                write_count,
                subthreshold_write_count,
                zero_step_count,
                nwrite_sum,
            )

            return (
                i
                < seq_len_tensor
            )

        def dec_body(
            i,
            raw_state,
            q_state,
            counter_state,
            hidden_ta,
            deadband_count,
            write_count,
            subthreshold_write_count,
            zero_step_count,
            nwrite_sum,
        ):
            del raw_state

            (
                raw_next,
                _,
                _,
                _,
            ) = (
                self.sdecgru
                .gru_step(
                    dec_inputs[
                        :,
                        i,
                        :
                    ],
                    q_state,
                    dec_kernel_q,
                    dec_recurrent_q,
                    dec_bias_q,
                )
            )

            hidden_ta = (
                hidden_ta.write(
                    i,
                    raw_next,
                )
            )

            def live_update():
                (
                    q_next,
                    counter_next,
                ) = (
                    hard_advance_switch(
                        self.sdecgru,
                        raw_next,
                        q_state,
                        counter_state,
                    )
                )

                deadband = (
                    tf.abs(
                        raw_next
                        - q_state
                    )
                    < half
                )

                changed = (
                    tf.abs(
                        q_next
                        - q_state
                    )
                    > change_eps
                )

                nwrite = (
                    tf.reduce_sum(
                        tf.cast(
                            changed,
                            tf.int64,
                        ),
                        axis=1,
                    )
                )

                return (
                    q_next,
                    counter_next,
                    deadband_count
                    + tf.reduce_sum(
                        tf.cast(
                            deadband,
                            tf.int64,
                        )
                    ),
                    write_count
                    + tf.reduce_sum(
                        tf.cast(
                            changed,
                            tf.int64,
                        )
                    ),
                    subthreshold_write_count
                    + tf.reduce_sum(
                        tf.cast(
                            deadband
                            & changed,
                            tf.int64,
                        )
                    ),
                    zero_step_count
                    + tf.reduce_sum(
                        tf.cast(
                            nwrite
                            == 0,
                            tf.int64,
                        )
                    ),
                    nwrite_sum
                    + tf.reduce_sum(
                        nwrite
                    ),
                )

            def terminal_update():
                return (
                    q_state,
                    counter_state,
                    deadband_count,
                    write_count,
                    subthreshold_write_count,
                    zero_step_count,
                    nwrite_sum,
                )

            (
                q_next,
                counter_next,
                deadband_next,
                write_next,
                subthreshold_next,
                zero_step_next,
                nwrite_sum_next,
            ) = tf.cond(
                i
                < final_live_index,
                live_update,
                terminal_update,
            )

            return (
                i + 1,
                raw_next,
                q_next,
                counter_next,
                hidden_ta,
                deadband_next,
                write_next,
                subthreshold_next,
                zero_step_next,
                nwrite_sum_next,
            )

        (
            _,
            _,
            _,
            _,
            dec_hidden_ta,
            dec_deadband_count,
            dec_write_count,
            dec_subthreshold_write_count,
            dec_zero_step_count,
            dec_nwrite_sum,
        ) = tf.while_loop(
            dec_cond,
            dec_body,
            loop_vars=(
                tf.constant(
                    0,
                    tf.int32,
                ),
                raw_enc,
                q_dec_hard,
                counter_dec,
                dec_hidden_ta,
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
                tf.constant(
                    0,
                    tf.int64,
                ),
            ),
            parallel_iterations=1,
            maximum_iterations=(
                self.seq_len
            ),
        )

        dec_hidden = tf.transpose(
            dec_hidden_ta.stack(),
            perm=(
                1,
                0,
                2,
            ),
        )

        preds = (
            self.sdec_dense(
                dec_hidden
            )
        )

        return (
            preds,
            enc_deadband_count,
            enc_write_count,
            enc_subthreshold_write_count,
            enc_zero_step_count,
            enc_nwrite_sum,
            dec_deadband_count,
            dec_write_count,
            dec_subthreshold_write_count,
            dec_zero_step_count,
            dec_nwrite_sum,
            tf.cast(
                handoff_abs_sum,
                tf.float64,
            ),
        )

    def export_raw_weights(
        self,
    ) -> Dict[
        str,
        np.ndarray,
    ]:
        return {
            "enc_kernel": np.asarray(
                self.sencgru.kernel.numpy(),
                dtype=np.float32,
            ),
            "enc_recurrent": np.asarray(
                self.sencgru.recurrent_kernel.numpy(),
                dtype=np.float32,
            ),
            "enc_bias": np.asarray(
                self.sencgru.bias.numpy(),
                dtype=np.float32,
            ),
            "dec_kernel": np.asarray(
                self.sdecgru.kernel.numpy(),
                dtype=np.float32,
            ),
            "dec_recurrent": np.asarray(
                self.sdecgru.recurrent_kernel.numpy(),
                dtype=np.float32,
            ),
            "dec_bias": np.asarray(
                self.sdecgru.bias.numpy(),
                dtype=np.float32,
            ),
            "dense_kernel": np.asarray(
                self.sdec_dense.kernel.numpy(),
                dtype=np.float32,
            ),
            "dense_bias": np.asarray(
                self.sdec_dense.bias.numpy(),
                dtype=np.float32,
            ),
        }


def initialize_from_standard_qgru_and_validate(
    reference_model,
    scw_model: SCWStudentModel,
    normalized_input,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    job_dir: Path,
) -> Dict:
    for (
        layer_name,
        target_layer,
    ) in (
        (
            "sencgru",
            scw_model.sencgru,
        ),
        (
            "sdecgru",
            scw_model.sdecgru,
        ),
        (
            "sdec_dense",
            scw_model.sdec_dense,
        ),
    ):
        source_layer = (
            reference_model
            .get_layer(
                layer_name
            )
        )

        source_weights = (
            source_layer
            .get_weights()
        )

        target_weights = (
            target_layer
            .get_weights()
        )

        if (
            len(
                source_weights
            )
            != len(
                target_weights
            )
        ):
            raise RuntimeError(
                f"Initialization transfer "
                f"weight-count mismatch for "
                f"{layer_name}: "
                f"reference="
                f"{len(source_weights)} "
                f"custom="
                f"{len(target_weights)}"
            )

        for (
            index,
            (
                source,
                target,
            ),
        ) in enumerate(
            zip(
                source_weights,
                target_weights,
            )
        ):
            if (
                tuple(
                    source.shape
                )
                != tuple(
                    target.shape
                )
            ):
                raise RuntimeError(
                    f"Initialization transfer "
                    f"shape mismatch for "
                    f"{layer_name} weight "
                    f"{index}: "
                    f"reference="
                    f"{source.shape} "
                    f"custom="
                    f"{target.shape}"
                )

        target_layer.set_weights(
            source_weights
        )

    n = min(
        512,
        len(
            validation_indices
        ),
    )

    if n <= 0:
        raise RuntimeError(
            "Initialization equivalence "
            "requires at least one "
            "validation sample"
        )

    rows = (
        validation_indices[
            :n
        ]
    )

    enc = tf.constant(
        np.asarray(
            normalized_input[
                rows
            ],
            dtype=np.float32,
        ),
        tf.float32,
    )

    dec = tf.zeros(
        (
            n,
            args.seq_len,
            1,
        ),
        tf.float32,
    )

    reference_pred = np.asarray(
        reference_model(
            [
                enc,
                dec,
            ],
            training=False,
        ).numpy(),
        dtype=np.float32,
    )

    custom_pred = np.asarray(
        scw_model(
            [
                enc,
                dec,
            ],
            training=False,
            operator_mode=(
                "deterministic"
            ),
        ).numpy(),
        dtype=np.float32,
    )

    if (
        reference_pred.shape
        != custom_pred.shape
    ):
        raise RuntimeError(
            f"Initialization equivalence "
            f"shape mismatch: "
            f"{reference_pred.shape} "
            f"vs {custom_pred.shape}"
        )

    diff = np.abs(
        reference_pred
        - custom_pred
    ).astype(
        np.float64
    )

    max_abs = float(
        np.max(
            diff
        )
    )

    mean_abs = float(
        np.mean(
            diff
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    diff
                )
            )
        )
    )

    tolerance = 5e-5

    passed = bool(
        max_abs
        <= tolerance
        and mean_abs
        <= tolerance
    )

    payload = {
        "passed": (
            passed
        ),
        "n_samples": (
            int(
                n
            )
        ),
        "max_abs": (
            max_abs
        ),
        "mean_abs": (
            mean_abs
        ),
        "rmse": (
            rmse
        ),
        "tolerance": (
            tolerance
        ),
        "reference": (
            "train_student_vanilla_kd.py::"
            "build_student deterministic "
            "B4 QGRU"
        ),
        "custom": (
            "SCWStudentModel with "
            "operator_mode=deterministic"
        ),
        "initialization_transfer": (
            "exact raw layer weights "
            "copied by layer name"
        ),
    }

    atomic_write_json(
        job_dir
        / "initialization_equivalence.json",
        payload,
    )

    if not passed:
        raise RuntimeError(
            "Custom SCW recurrent implementation "
            "failed deterministic QGRU "
            "equivalence before training: "
            f"max_abs={max_abs:.8g} "
            f"mean_abs={mean_abs:.8g} "
            f"tolerance={tolerance:.8g}"
        )

    pf(
        f"[EQUIV] deterministic QGRU "
        f"initialization equivalence passed: "
        f"N={n} "
        f"max_abs={max_abs:.3e} "
        f"mean_abs={mean_abs:.3e}"
    )

    return payload


def build_scw_student(
    args: argparse.Namespace,
) -> SCWStudentModel:
    model = SCWStudentModel(
        seq_len=args.seq_len,
        n_out=args.n_out,
        student_units=(
            args.student_units
        ),
        bits_kernel=(
            args.bits_kernel
        ),
        bits_recurrent=(
            args.bits_recurrent
        ),
        bits_bias=(
            args.bits_bias
        ),
        bits_activation=(
            args.bits_activation
        ),
        bits_state=(
            args.bits_state
        ),
        counter_bits=(
            args.counter_bits
        ),
        deadzone_fraction=(
            args.scw_deadzone_fraction
        ),
        q_alpha=(
            args.q_alpha
        ),
    )

    dummy_enc = tf.zeros(
        (
            1,
            args.seq_len,
            1,
        ),
        tf.float32,
    )

    dummy_dec = tf.zeros(
        (
            1,
            args.seq_len,
            1,
        ),
        tf.float32,
    )

    _ = model(
        [
            dummy_enc,
            dummy_dec,
        ],
        training=False,
        operator_mode="scw",
    )

    return model


def train_step_per_replica(
    batch_x,
    batch_y,
    student_model,
    optimizer,
    temperature,
    alpha,
):
    enc_b = (
        batch_x[
            "enc_input"
        ]
    )

    dec_b = (
        batch_x[
            "dec_input"
        ]
    )

    tpred_b = (
        batch_x[
            "tpred"
        ]
    )

    tgt_b = (
        batch_y
    )

    alpha_f = tf.cast(
        alpha,
        tf.float32,
    )

    T = tf.cast(
        temperature,
        tf.float32,
    )

    with tf.GradientTape() as tape:
        student_output = (
            student_model(
                [
                    enc_b,
                    dec_b,
                ],
                training=True,
                operator_mode="scw",
            )
        )

        hard_loss = (
            tf.reduce_mean(
                tf.square(
                    student_output
                    - tgt_b
                )
            )
        )

        soft_loss = (
            T
            * T
            * tf.reduce_mean(
                tf.square(
                    tpred_b
                    / T
                    - student_output
                    / T
                )
            )
        )

        total_loss = (
            alpha_f
            * soft_loss
            + (
                1.0
                - alpha_f
            )
            * hard_loss
        )

    grads = tape.gradient(
        total_loss,
        student_model.trainable_variables,
    )

    grads = [
        (
            tf.zeros_like(
                variable
            )
            if grad is None
            else grad
        )
        for (
            grad,
            variable,
        ) in zip(
            grads,
            student_model.trainable_variables,
        )
    ]

    bad_grad = (
        tf.reduce_any(
            tf.stack(
                [
                    tf.reduce_any(
                        ~tf.math.is_finite(
                            grad
                        )
                    )
                    for grad in grads
                ]
            )
        )
    )

    safe_grads = [
        tf.where(
            bad_grad,
            tf.zeros_like(
                grad
            ),
            grad,
        )
        for grad in grads
    ]

    (
        safe_grads,
        grad_norm,
    ) = tf.clip_by_global_norm(
        safe_grads,
        clip_norm=1.0,
    )

    optimizer.apply_gradients(
        zip(
            safe_grads,
            student_model.trainable_variables,
        )
    )

    return (
        total_loss,
        hard_loss,
        soft_loss,
        tf.cast(
            bad_grad,
            tf.float32,
        ),
        tf.cast(
            grad_norm,
            tf.float32,
        ),
    )


def val_step_per_replica(
    batch_x,
    batch_y,
    student_model,
    temperature,
    alpha,
):
    enc_b = (
        batch_x[
            "enc_input"
        ]
    )

    dec_b = (
        batch_x[
            "dec_input"
        ]
    )

    tpred_b = (
        batch_x[
            "tpred"
        ]
    )

    tgt_b = (
        batch_y
    )

    alpha_f = tf.cast(
        alpha,
        tf.float32,
    )

    T = tf.cast(
        temperature,
        tf.float32,
    )

    student_output = (
        student_model(
            [
                enc_b,
                dec_b,
            ],
            training=False,
            operator_mode="scw",
        )
    )

    hard_loss = (
        tf.reduce_mean(
            tf.square(
                student_output
                - tgt_b
            )
        )
    )

    soft_loss = (
        T
        * T
        * tf.reduce_mean(
            tf.square(
                tpred_b
                / T
                - student_output
                / T
            )
        )
    )

    total_loss = (
        alpha_f
        * soft_loss
        + (
            1.0
            - alpha_f
        )
        * hard_loss
    )

    mae = (
        tf.reduce_mean(
            tf.abs(
                student_output
                - tgt_b
            )
        )
    )

    return (
        total_loss,
        hard_loss,
        soft_loss,
        mae,
    )


def make_distributed_train_step(
    strategy,
    student_model,
    optimizer,
    temperature,
    alpha,
):
    @tf.function
    def distributed_train_step(
        batch_x,
        batch_y,
    ):
        per_replica = (
            strategy.run(
                train_step_per_replica,
                args=(
                    batch_x,
                    batch_y,
                    student_model,
                    optimizer,
                    temperature,
                    alpha,
                ),
            )
        )

        return (
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[0],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[1],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[2],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                per_replica[3],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[4],
                axis=None,
            ),
        )

    return (
        distributed_train_step
    )


def make_distributed_val_step(
    strategy,
    student_model,
    temperature,
    alpha,
):
    @tf.function
    def distributed_val_step(
        batch_x,
        batch_y,
    ):
        per_replica = (
            strategy.run(
                val_step_per_replica,
                args=(
                    batch_x,
                    batch_y,
                    student_model,
                    temperature,
                    alpha,
                ),
            )
        )

        return (
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[0],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[1],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[2],
                axis=None,
            ),
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                per_replica[3],
                axis=None,
            ),
        )

    return (
        distributed_val_step
    )


def save_exact_resume_checkpoint(
    strategy,
    model,
    optimizer,
    scheduler,
    job_dir: Path,
    completed_epochs: int,
) -> str:
    checkpoint_dir = (
        job_dir
        / "resume_checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with strategy.scope():
        checkpoint = (
            tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        )

        manager = (
            tf.train.CheckpointManager(
                checkpoint,
                str(
                    checkpoint_dir
                ),
                max_to_keep=2,
                checkpoint_name="ckpt",
            )
        )

        saved = manager.save(
            checkpoint_number=(
                int(
                    completed_epochs
                )
            )
        )

    expected = (
        checkpoint_dir
        / f"ckpt-{int(completed_epochs)}"
    )

    if (
        Path(
            saved
        ).resolve()
        != expected.resolve()
    ):
        raise RuntimeError(
            f"Unexpected resume checkpoint "
            f"path: expected {expected}, "
            f"got {saved}"
        )

    if not Path(
        str(
            expected
        )
        + ".index"
    ).is_file():
        raise RuntimeError(
            f"Resume checkpoint index was "
            f"not created: {expected}.index"
        )

    return str(
        expected
    )


def restore_exact_resume_checkpoint(
    strategy,
    model,
    optimizer,
    scheduler,
    job_dir: Path,
    completed_epochs: int,
) -> None:
    checkpoint_prefix = (
        job_dir
        / "resume_checkpoints"
        / f"ckpt-{int(completed_epochs)}"
    )

    index_path = Path(
        str(
            checkpoint_prefix
        )
        + ".index"
    )

    if not index_path.is_file():
        raise RuntimeError(
            f"resume_state.json requests "
            f"epoch {completed_epochs}, but "
            f"exact checkpoint is missing: "
            f"{index_path}"
        )

    with strategy.scope():
        create_slots = getattr(
            optimizer,
            "_create_all_weights",
            None,
        )

        if not callable(
            create_slots
        ):
            raise RuntimeError(
                "TensorFlow optimizer does not "
                "expose _create_all_weights; "
                "exact TF 2.10 Adam resume "
                "cannot be guaranteed"
            )

        create_slots(
            model.trainable_variables
        )

        checkpoint = (
            tf.train.Checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        )

        status = checkpoint.restore(
            str(
                checkpoint_prefix
            )
        )

        status.assert_existing_objects_matched()
        status.expect_partial()

    pf(
        f"[RESUME] Restored exact epoch "
        f"{completed_epochs}: "
        f"optimizer_iterations="
        f"{int(optimizer.iterations.numpy())} "
        f"lr={scheduler.current_lr:.3e} "
        f"scheduler_best="
        f"{scheduler.best:.8g} "
        f"scheduler_wait="
        f"{scheduler.wait}"
    )


def training_loop(
    strategy,
    student_model,
    optimizer,
    scheduler,
    dist_train_dataset,
    dist_val_dataset,
    train_steps: int,
    val_steps: int,
    args: argparse.Namespace,
    job_dir: Path,
) -> Tuple[
    Dict[
        str,
        List[
            float
        ],
    ],
    float,
    bool,
]:
    del val_steps

    best_ckpt = (
        job_dir
        / "student_best.weights.h5"
    )

    resume_path = (
        job_dir
        / "resume_state.json"
    )

    completion_path = (
        job_dir
        / "training_complete.flag"
    )

    history_path = (
        job_dir
        / "training_history.csv"
    )

    history = {
        "total": [],
        "hard": [],
        "soft": [],
        "val_total": [],
        "val_hard": [],
        "val_soft": [],
        "val_mae": [],
        "grad_norm": [],
    }

    best_val = float(
        "inf"
    )

    patience_count = 0
    start_epoch = 0

    if args.resume:
        if completion_path.is_file():
            if not best_ckpt.is_file():
                raise RuntimeError(
                    f"Training completion flag "
                    f"exists but best checkpoint "
                    f"is missing: {best_ckpt}"
                )

            student_model.load_weights(
                str(
                    best_ckpt
                )
            )

            pf(
                f"[RESUME] Training already "
                f"complete. Loaded "
                f"{best_ckpt}"
            )

            return (
                history,
                best_val,
                True,
            )

        if not resume_path.is_file():
            raise RuntimeError(
                f"--resume requires "
                f"resume_state.json, but it "
                f"does not exist: "
                f"{resume_path}"
            )

        if not best_ckpt.is_file():
            raise RuntimeError(
                f"--resume requires "
                f"student_best.weights.h5, "
                f"but it does not exist: "
                f"{best_ckpt}"
            )

        with resume_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            state = json.load(
                handle
            )

        start_epoch = int(
            state[
                "epoch"
            ]
        )

        best_val = float(
            state[
                "best_val"
            ]
        )

        patience_count = int(
            state[
                "patience_count"
            ]
        )

        saved_history = (
            state.get(
                "history",
                {},
            )
        )

        for key in history:
            if key in saved_history:
                history[
                    key
                ] = [
                    float(
                        value
                    )
                    for value
                    in saved_history[
                        key
                    ]
                ]

        if start_epoch <= 0:
            raise RuntimeError(
                "Resume epoch must be > 0"
            )

        if (
            start_epoch
            > args.epochs
        ):
            raise RuntimeError(
                f"Resume epoch "
                f"{start_epoch} exceeds "
                f"requested total epochs "
                f"{args.epochs}"
            )

        restore_exact_resume_checkpoint(
            strategy=strategy,
            model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            job_dir=job_dir,
            completed_epochs=(
                start_epoch
            ),
        )

        pf(
            f"[RESUME] Continuing at "
            f"epoch {start_epoch + 1}/"
            f"{args.epochs}; "
            f"best_val={best_val:.8g} "
            f"patience={patience_count}/"
            f"{args.patience}"
        )

    if not args.resume:
        with history_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "epoch,total,hard,soft,"
                "val_total,val_hard,"
                "val_soft,val_mae,"
                "grad_norm,lr,"
                "bad_grad_batches\n"
            )

    train_step = (
        make_distributed_train_step(
            strategy,
            student_model,
            optimizer,
            args.temperature,
            args.alpha,
        )
    )

    val_step = (
        make_distributed_val_step(
            strategy,
            student_model,
            args.temperature,
            args.alpha,
        )
    )

    interrupted = False

    for epoch in range(
        start_epoch,
        args.epochs,
    ):
        epoch_start = (
            time.time()
        )

        train_total = 0.0
        train_hard = 0.0
        train_soft = 0.0
        grad_norm_total = 0.0
        train_count = 0
        bad_grad_batches = 0

        pf(
            f"[EPOCH {epoch + 1}/"
            f"{args.epochs}] train start "
            f"lr="
            f"{scheduler.current_lr:.3e}"
        )

        for (
            step_index,
            (
                batch_x,
                batch_y,
            ),
        ) in enumerate(
            dist_train_dataset
        ):
            (
                total,
                hard,
                soft,
                bad_flag,
                grad_norm,
            ) = train_step(
                batch_x,
                batch_y,
            )

            train_total += float(
                total
            )

            train_hard += float(
                hard
            )

            train_soft += float(
                soft
            )

            grad_norm_total += float(
                grad_norm
            )

            train_count += 1

            if (
                float(
                    bad_flag
                )
                > 0.0
            ):
                bad_grad_batches += 1

            if (
                (
                    step_index
                    + 1
                )
                % args.log_interval
                == 0
                or (
                    step_index
                    + 1
                )
                == train_steps
            ):
                elapsed = (
                    time.time()
                    - epoch_start
                )

                rate = (
                    elapsed
                    / max(
                        step_index
                        + 1,
                        1,
                    )
                )

                remaining = (
                    rate
                    * max(
                        train_steps
                        - step_index
                        - 1,
                        0,
                    )
                )

                pf(
                    f"  step "
                    f"{step_index + 1:5d}/"
                    f"{train_steps} "
                    f"total="
                    f"{train_total / train_count:.6f} "
                    f"hard="
                    f"{train_hard / train_count:.6f} "
                    f"soft="
                    f"{train_soft / train_count:.6f} "
                    f"grad_norm="
                    f"{grad_norm_total / train_count:.5f} "
                    f"bad_grad_batches="
                    f"{bad_grad_batches} "
                    f"eta="
                    f"{remaining / 60.0:.1f}m"
                )

        if train_count == 0:
            raise RuntimeError(
                "Training dataset "
                "yielded zero batches"
            )

        val_total = 0.0
        val_hard = 0.0
        val_soft = 0.0
        val_mae = 0.0
        val_count = 0

        for (
            batch_x,
            batch_y,
        ) in dist_val_dataset:
            (
                total,
                hard,
                soft,
                mae,
            ) = val_step(
                batch_x,
                batch_y,
            )

            val_total += float(
                total
            )

            val_hard += float(
                hard
            )

            val_soft += float(
                soft
            )

            val_mae += float(
                mae
            )

            val_count += 1

        if val_count == 0:
            raise RuntimeError(
                "Validation dataset "
                "yielded zero batches"
            )

        train_total /= (
            train_count
        )

        train_hard /= (
            train_count
        )

        train_soft /= (
            train_count
        )

        grad_norm_mean = (
            grad_norm_total
            / train_count
        )

        val_total /= (
            val_count
        )

        val_hard /= (
            val_count
        )

        val_soft /= (
            val_count
        )

        val_mae /= (
            val_count
        )

        history[
            "total"
        ].append(
            train_total
        )

        history[
            "hard"
        ].append(
            train_hard
        )

        history[
            "soft"
        ].append(
            train_soft
        )

        history[
            "val_total"
        ].append(
            val_total
        )

        history[
            "val_hard"
        ].append(
            val_hard
        )

        history[
            "val_soft"
        ].append(
            val_soft
        )

        history[
            "val_mae"
        ].append(
            val_mae
        )

        history[
            "grad_norm"
        ].append(
            grad_norm_mean
        )

        if (
            args.effective_warmup_epochs
            > 0
            and epoch
            < args.effective_warmup_epochs
        ):
            warmup_lr = (
                float(
                    args.effective_lr
                )
                * float(
                    epoch
                    + 1
                )
                / float(
                    args.effective_warmup_epochs
                )
            )

            scheduler.lr_var.assign(
                warmup_lr
            )

            pf(
                f"[LR] Preserving vanilla "
                f"warmup schedule: "
                f"epoch={epoch + 1}/"
                f"{args.effective_warmup_epochs} "
                f"lr={warmup_lr:.3e}"
            )

        else:
            scheduler.step(
                val_total,
                epoch,
            )

        if (
            val_total
            < best_val
            - args.min_delta
        ):
            best_val = (
                val_total
            )

            patience_count = 0

            student_model.save_weights(
                str(
                    best_ckpt
                )
            )

            pf(
                f"[CHECKPOINT] New best "
                f"val={best_val:.8g}: "
                f"{best_ckpt}"
            )

        else:
            patience_count += 1

        completed_epochs = (
            epoch
            + 1
        )

        exact_checkpoint = (
            save_exact_resume_checkpoint(
                strategy=strategy,
                model=student_model,
                optimizer=optimizer,
                scheduler=scheduler,
                job_dir=job_dir,
                completed_epochs=(
                    completed_epochs
                ),
            )
        )

        resume_payload = {
            "epoch": (
                completed_epochs
            ),
            "best_val": (
                float(
                    best_val
                )
            ),
            "patience_count": (
                int(
                    patience_count
                )
            ),
            "lr": (
                float(
                    scheduler.current_lr
                )
            ),
            "exact_checkpoint": (
                exact_checkpoint
            ),
            "history": {
                key: [
                    float(
                        value
                    )
                    for value in values
                ]
                for (
                    key,
                    values,
                ) in history.items()
            },
        }

        atomic_write_json(
            resume_path,
            resume_payload,
        )

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"{completed_epochs},"
                f"{train_total:.10g},"
                f"{train_hard:.10g},"
                f"{train_soft:.10g},"
                f"{val_total:.10g},"
                f"{val_hard:.10g},"
                f"{val_soft:.10g},"
                f"{val_mae:.10g},"
                f"{grad_norm_mean:.10g},"
                f"{scheduler.current_lr:.10g},"
                f"{bad_grad_batches}\n"
            )

        pf(
            f"[EPOCH "
            f"{completed_epochs}/"
            f"{args.epochs}] "
            f"train={train_total:.6f} "
            f"val={val_total:.6f} "
            f"val_mae={val_mae:.6f} "
            f"best={best_val:.6f} "
            f"patience="
            f"{patience_count}/"
            f"{args.patience} "
            f"bad_grad_batches="
            f"{bad_grad_batches} "
            f"time="
            f"{(time.time() - epoch_start) / 60.0:.1f}m"
        )

        if STOP_AFTER_EPOCH:
            interrupted = True

            pf(
                "[SIGNAL] Epoch checkpoint "
                "is complete. Exiting "
                "cleanly for SLURM "
                "resubmission."
            )

            break

        if (
            patience_count
            >= args.patience
        ):
            pf(
                f"[EARLY STOP] patience "
                f"reached at epoch "
                f"{completed_epochs}"
            )

            break

    if interrupted:
        return (
            history,
            best_val,
            False,
        )

    if not best_ckpt.is_file():
        raise RuntimeError(
            f"Training finished without "
            f"a best checkpoint: "
            f"{best_ckpt}"
        )

    student_model.load_weights(
        str(
            best_ckpt
        )
    )

    completion_path.write_text(
        "passed\n",
        encoding="utf-8",
    )

    pf(
        f"[TRAINING] Loaded selected "
        f"best checkpoint: "
        f"{best_ckpt}"
    )

    pf(
        f"[TRAINING] Completion flag: "
        f"{completion_path}"
    )

    return (
        history,
        best_val,
        True,
    )


def save_training_plot(
    history: Dict[
        str,
        List[
            float
        ],
    ],
    job_dir: Path,
) -> None:
    if not history[
        "total"
    ]:
        return

    epochs = np.arange(
        1,
        len(
            history[
                "total"
            ]
        )
        + 1,
    )

    fig = plt.figure(
        figsize=(
            9,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.plot(
        epochs,
        history[
            "total"
        ],
        label="train total",
    )

    ax.plot(
        epochs,
        history[
            "val_total"
        ],
        label="validation total",
    )

    ax.plot(
        epochs,
        history[
            "val_mae"
        ],
        label="validation MAE",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss / MAE"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        job_dir
        / "training_history.png",
        dpi=180,
    )

    plt.close(
        fig
    )


def lifetime_from_prediction(
    prediction: np.ndarray,
    t_axis: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ch1 = (
        prediction[
            :,
            :,
            1
        ]
    )

    ch2 = (
        prediction[
            :,
            :,
            2
        ]
    )

    int1 = np.trapz(
        ch1,
        t_axis,
        axis=1,
    )

    int2 = np.trapz(
        ch2,
        t_axis,
        axis=1,
    )

    amp1 = (
        ch1[
            :,
            0
        ]
    )

    amp2 = (
        ch2[
            :,
            0
        ]
    )

    tau1 = np.where(
        amp1 > 1e-6,
        int1
        / amp1,
        0.0,
    ).astype(
        np.float32
    )

    tau2 = np.where(
        amp2 > 1e-6,
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
        denom > 1e-6,
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


def metric_summary(
    gt: np.ndarray,
    pred: np.ndarray,
) -> Dict:
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    residual = (
        pred
        - gt
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    residual
                )
            )
        )
    )

    (
        r,
        _,
    ) = pearsonr(
        gt,
        pred,
    )

    sigma = float(
        np.std(
            residual
        )
    )

    coverage = float(
        np.mean(
            np.abs(
                residual
            )
            <= sigma
        )
        * 100.0
    )

    return {
        "rmse": (
            rmse
        ),
        "r": (
            float(
                r
            )
        ),
        "cov1sigma": (
            coverage
        ),
    }


def operator_code(
    mode: str,
) -> int:
    mapping = {
        "scw": 0,
        "deterministic": 1,
        "identity": 2,
    }

    return mapping[
        mode
    ]


def evaluate_operator_comparison(
    model: SCWStudentModel,
    normalized_input,
    res,
    labels,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    job_dir: Path,
) -> Dict:
    modes = (
        "scw",
        "deterministic",
        "identity",
    )

    n_test = len(
        test_idx
    )

    t_axis = (
        np.arange(
            args.seq_len,
            dtype=np.float32,
        )
        * float(
            args.gate_width_ns
        )
    )

    gt_tau1 = np.asarray(
        labels[
            test_idx,
            0
        ],
        dtype=np.float32,
    )

    gt_tau2 = np.asarray(
        labels[
            test_idx,
            1
        ],
        dtype=np.float32,
    )

    gt_fret = np.asarray(
        labels[
            test_idx,
            2
        ],
        dtype=np.float32,
    )

    arrays = {
        "test_idx": np.asarray(
            test_idx,
            dtype=np.int64,
        ),
        "gt_tau1": (
            gt_tau1
        ),
        "gt_tau2": (
            gt_tau2
        ),
        "gt_fret": (
            gt_fret
        ),
    }

    summary = {
        "n_test": (
            int(
                n_test
            )
        ),
        "operator_training_mode": (
            "scw"
        ),
        "operators": {},
    }

    for mode in modes:
        pf(
            f"[EVAL] Operator="
            f"{mode} "
            f"N={n_test:,}"
        )

        tau1_pred = np.empty(
            n_test,
            dtype=np.float32,
        )

        tau2_pred = np.empty(
            n_test,
            dtype=np.float32,
        )

        fret_pred = np.empty(
            n_test,
            dtype=np.float32,
        )

        seq_mae = np.empty(
            n_test,
            dtype=np.float32,
        )

        counts = {
            "enc_deadband": 0,
            "enc_write": 0,
            "enc_subthreshold_write": 0,
            "enc_zero_step": 0,
            "enc_nwrite_sum": 0,
            "dec_deadband": 0,
            "dec_write": 0,
            "dec_subthreshold_write": 0,
            "dec_zero_step": 0,
            "dec_nwrite_sum": 0,
        }

        handoff_abs_sum = 0.0

        for start in range(
            0,
            n_test,
            args.infer_batch,
        ):
            end = min(
                start
                + args.infer_batch,
                n_test,
            )

            rows = (
                test_idx[
                    start:end
                ]
            )

            enc_b_np = np.asarray(
                normalized_input[
                    rows
                ],
                dtype=np.float32,
            )

            tgt_b_np = np.asarray(
                res[
                    rows
                ],
                dtype=np.float32,
            )

            enc_b = tf.constant(
                enc_b_np,
                tf.float32,
            )

            dec_b = tf.zeros(
                (
                    end
                    - start,
                    args.seq_len,
                    1,
                ),
                tf.float32,
            )

            result = (
                model
                .diagnostic_forward(
                    enc_b,
                    dec_b,
                    tf.constant(
                        operator_code(
                            mode
                        ),
                        tf.int32,
                    ),
                )
            )

            pred_np = np.asarray(
                result[
                    0
                ].numpy(),
                dtype=np.float32,
            )

            (
                batch_tau1,
                batch_tau2,
                batch_fret,
            ) = (
                lifetime_from_prediction(
                    pred_np,
                    t_axis,
                )
            )

            tau1_pred[
                start:end
            ] = batch_tau1

            tau2_pred[
                start:end
            ] = batch_tau2

            fret_pred[
                start:end
            ] = batch_fret

            seq_mae[
                start:end
            ] = np.mean(
                np.abs(
                    pred_np
                    - tgt_b_np
                ),
                axis=(
                    1,
                    2,
                ),
            ).astype(
                np.float32
            )

            keys = (
                "enc_deadband",
                "enc_write",
                "enc_subthreshold_write",
                "enc_zero_step",
                "enc_nwrite_sum",
                "dec_deadband",
                "dec_write",
                "dec_subthreshold_write",
                "dec_zero_step",
                "dec_nwrite_sum",
            )

            for (
                key,
                tensor,
            ) in zip(
                keys,
                result[
                    1:11
                ],
            ):
                counts[
                    key
                ] += int(
                    tensor.numpy()
                )

            handoff_abs_sum += float(
                result[
                    11
                ].numpy()
            )

            if (
                end
                == n_test
                or (
                    start
                    // args.infer_batch
                )
                % 10
                == 0
            ):
                pf(
                    f"[EVAL] {mode}: "
                    f"{end:,}/"
                    f"{n_test:,}"
                )

        arrays[
            f"{mode}_tau1_pred"
        ] = tau1_pred

        arrays[
            f"{mode}_tau2_pred"
        ] = tau2_pred

        arrays[
            f"{mode}_fret_pred"
        ] = fret_pred

        arrays[
            f"{mode}_seq_mae_per_sequence"
        ] = seq_mae

        element_denominator = (
            n_test
            * (
                args.seq_len
                - 1
            )
            * args.student_units
        )

        step_denominator = (
            n_test
            * (
                args.seq_len
                - 1
            )
        )

        handoff_denominator = (
            n_test
            * args.student_units
        )

        enc_deadband_fraction = (
            counts[
                "enc_deadband"
            ]
            / element_denominator
        )

        dec_deadband_fraction = (
            counts[
                "dec_deadband"
            ]
            / element_denominator
        )

        operator_payload = {
            "mae_seq": (
                float(
                    np.mean(
                        seq_mae
                    )
                )
            ),
            "tau1": (
                metric_summary(
                    gt_tau1,
                    tau1_pred,
                )
            ),
            "tau2": (
                metric_summary(
                    gt_tau2,
                    tau2_pred,
                )
            ),
            "fret": (
                metric_summary(
                    gt_fret,
                    fret_pred,
                )
            ),
            "encoder": {
                "deadband_fraction": (
                    float(
                        enc_deadband_fraction
                    )
                ),
                "deadband_is_counterfactual": (
                    bool(
                        mode
                        == "identity"
                    )
                ),
                "state_change_fraction": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "enc_write"
                        ]
                        / element_denominator
                    )
                ),
                "p_nwrite0": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "enc_zero_step"
                        ]
                        / step_denominator
                    )
                ),
                "mean_nwrite": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "enc_nwrite_sum"
                        ]
                        / step_denominator
                    )
                ),
                "subthreshold_write_fraction_given_deadband": (
                    None
                    if (
                        mode
                        == "identity"
                        or counts[
                            "enc_deadband"
                        ]
                        == 0
                    )
                    else float(
                        counts[
                            "enc_subthreshold_write"
                        ]
                        / counts[
                            "enc_deadband"
                        ]
                    )
                ),
            },
            "decoder": {
                "deadband_fraction": (
                    float(
                        dec_deadband_fraction
                    )
                ),
                "deadband_is_counterfactual": (
                    bool(
                        mode
                        == "identity"
                    )
                ),
                "state_change_fraction": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "dec_write"
                        ]
                        / element_denominator
                    )
                ),
                "p_nwrite0": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "dec_zero_step"
                        ]
                        / step_denominator
                    )
                ),
                "mean_nwrite": (
                    None
                    if mode
                    == "identity"
                    else float(
                        counts[
                            "dec_nwrite_sum"
                        ]
                        / step_denominator
                    )
                ),
                "subthreshold_write_fraction_given_deadband": (
                    None
                    if (
                        mode
                        == "identity"
                        or counts[
                            "dec_deadband"
                        ]
                        == 0
                    )
                    else float(
                        counts[
                            "dec_subthreshold_write"
                        ]
                        / counts[
                            "dec_deadband"
                        ]
                    )
                ),
            },
            "handoff_mean_abs_quantization_error": (
                float(
                    handoff_abs_sum
                    / handoff_denominator
                )
            ),
        }

        summary[
            "operators"
        ][
            mode
        ] = operator_payload

        pf(
            f"[EVAL] {mode}: "
            f"MAE="
            f"{operator_payload['mae_seq']:.6f} "
            f"tau1="
            f"{operator_payload['tau1']['rmse']:.6f} "
            f"tau2="
            f"{operator_payload['tau2']['rmse']:.6f} "
            f"decoder_deadband="
            f"{operator_payload['decoder']['deadband_fraction']:.6f} "
            f"decoder_state_change="
            f"{operator_payload['decoder']['state_change_fraction']}"
        )

    np.savez_compressed(
        job_dir
        / "scw_training_operator_comparison_per_sequence.npz",
        **arrays,
    )

    atomic_write_json(
        job_dir
        / "scw_training_operator_comparison.json",
        summary,
    )

    scw = (
        summary[
            "operators"
        ][
            "scw"
        ]
    )

    standard_metrics = {
        "job_name": (
            job_dir.name
        ),
        "n_test": (
            int(
                n_test
            )
        ),
        "tau1": (
            scw[
                "tau1"
            ]
        ),
        "tau2": (
            scw[
                "tau2"
            ]
        ),
        "fret": (
            scw[
                "fret"
            ]
        ),
        "mae_seq": (
            scw[
                "mae_seq"
            ]
        ),
        "state_writeback": (
            "SCW"
        ),
        "counter_bits": (
            int(
                args.counter_bits
            )
        ),
        "scw_deadzone_fraction": (
            float(
                args.scw_deadzone_fraction
            )
        ),
    }

    atomic_write_json(
        job_dir
        / "test_metrics.json",
        standard_metrics,
    )

    return summary


def save_raw_weights(
    model: SCWStudentModel,
    job_dir: Path,
) -> Path:
    path = (
        job_dir
        / "scw_trained_raw_weights.npz"
    )

    np.savez_compressed(
        path,
        **model.export_raw_weights(),
    )

    return path


def main() -> None:
    args = parse_args()

    if args.no_lr_scaling:
        args.effective_lr = float(
            args.lr
        )

        args.effective_lr_patience = int(
            args.lr_patience
        )

        args.effective_warmup_epochs = int(
            args.warmup_epochs
        )

    else:
        scaling_ratio = (
            float(
                args.batch_size
            )
            / float(
                args.ref_batch_size
            )
        )

        args.effective_lr = (
            float(
                args.lr
            )
            * scaling_ratio
        )

        args.effective_lr_patience = max(
            1,
            round(
                args.lr_patience
                * scaling_ratio
            ),
        )

        args.effective_warmup_epochs = max(
            0,
            round(
                args.warmup_epochs
                * scaling_ratio
            ),
        )

    tf.keras.utils.set_random_seed(
        args.split_seed
    )

    strategy = (
        setup_gpus_and_strategy(
            args.mixed_precision
        )
    )

    job_name = (
        make_job_name(
            args
        )
    )

    job_dir = (
        Path(
            args.save_dir
        ).resolve()
        / "results"
        / job_name
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args_payload = (
        vars(
            args
        ).copy()
    )

    args_payload.update(
        {
            "training_state_operator": (
                "scw"
            ),
            "scw_forward_semantics": (
                "hard"
            ),
            "scw_backward_state_surrogate": (
                "identity_ste"
            ),
            "scw_decoder_counter_reset": (
                True
            ),
        }
    )

    atomic_write_json(
        job_dir
        / "student_args.json",
        args_payload,
    )

    pf(
        f"[RUN] job_name="
        f"{job_name}"
    )

    pf(
        f"[RUN] job_dir="
        f"{job_dir}"
    )

    pf(
        f"[RUN] B4 + SCW "
        f"K{args.counter_bits}, "
        f"theta="
        f"{args.scw_deadzone_fraction}"
        f"*Delta"
    )

    pf(
        f"[RUN] LR base="
        f"{args.lr:.3e} "
        f"effective="
        f"{args.effective_lr:.3e} "
        f"batch="
        f"{args.batch_size} "
        f"micro="
        f"{args.batch_size // args.accumulation_steps}"
    )

    (
        file_input,
        file_res,
        file_labels,
        file_train,
        file_val,
        file_test,
    ) = find_data_files(
        args.data_dir,
        args.seq_len,
    )

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

    train_idx = np.asarray(
        np.load(
            file_train
        ),
        dtype=np.int64,
    )

    val_idx = np.asarray(
        np.load(
            file_val
        ),
        dtype=np.int64,
    )

    test_idx = np.asarray(
        np.load(
            file_test
        ),
        dtype=np.int64,
    )

    if (
        normalized_input.ndim
        != 3
        or tuple(
            normalized_input.shape[
                1:
            ]
        )
        != (
            args.seq_len,
            1,
        )
    ):
        raise RuntimeError(
            f"Unexpected input shape: "
            f"{normalized_input.shape}"
        )

    if (
        res.ndim
        != 3
        or tuple(
            res.shape[
                1:
            ]
        )
        != (
            args.seq_len,
            args.n_out,
        )
    ):
        raise RuntimeError(
            f"Unexpected target shape: "
            f"{res.shape}"
        )

    if (
        labels.ndim
        != 2
        or labels.shape[
            1
        ]
        < 3
    ):
        raise RuntimeError(
            f"Unexpected label shape: "
            f"{labels.shape}"
        )

    if (
        len(
            train_idx
        )
        == 0
        or len(
            val_idx
        )
        == 0
        or len(
            test_idx
        )
        == 0
    ):
        raise RuntimeError(
            "Train, validation, and test "
            "indices must be non-empty"
        )

    n_samples = int(
        normalized_input.shape[
            0
        ]
    )

    if (
        res.shape[
            0
        ]
        != n_samples
        or labels.shape[
            0
        ]
        != n_samples
    ):
        raise RuntimeError(
            "Input, target, and label "
            "sample counts do not match"
        )

    pf(
        f"[DATA] N="
        f"{n_samples:,} "
        f"train="
        f"{len(train_idx):,} "
        f"val="
        f"{len(val_idx):,} "
        f"test="
        f"{len(test_idx):,}"
    )

    teacher_model = (
        build_teacher(
            args.seq_len,
            args.n_out,
            args.teacher_units,
            args.teacher_layers,
        )
    )

    teacher_model.load_weights(
        args.teacher_ckpt
    )

    teacher_model.trainable = (
        False
    )

    pf(
        f"[TEACHER] loaded "
        f"{args.teacher_ckpt}"
    )

    teacher_predictions = (
        cache_teacher_predictions(
            teacher_model,
            normalized_input,
            args.seq_len,
            args.n_out,
            n_samples,
            args.infer_batch,
            args.data_dir,
            pf,
        )
    )

    (
        enc_train,
        tgt_train,
        tpred_train,
    ) = materialise_enc_tgt_tpred(
        normalized_input,
        res,
        teacher_predictions,
        train_idx,
        args.seq_len,
        args.n_out,
        "train",
        pf,
    )

    (
        enc_val,
        tgt_val,
        tpred_val,
    ) = materialise_enc_tgt_tpred(
        normalized_input,
        res,
        teacher_predictions,
        val_idx,
        args.seq_len,
        args.n_out,
        "val",
        pf,
    )

    with strategy.scope():
        reference_model = (
            build_student(
                seq_len=args.seq_len,
                n_out=args.n_out,
                student_units=(
                    args.student_units
                ),
                bits_kernel=(
                    args.bits_kernel
                ),
                bits_recurrent=(
                    args.bits_recurrent
                ),
                bits_bias=(
                    args.bits_bias
                ),
                bits_activation=(
                    args.bits_activation
                ),
                bits_state=(
                    args.bits_state
                ),
            )
        )

        student_model = (
            build_scw_student(
                args
            )
        )

        initialization_equivalence = (
            initialize_from_standard_qgru_and_validate(
                reference_model=reference_model,
                scw_model=student_model,
                normalized_input=normalized_input,
                validation_indices=val_idx,
                args=args,
                job_dir=job_dir,
            )
        )

        optimizer = (
            keras.optimizers.Adam(
                learning_rate=(
                    args.effective_lr
                )
            )
        )

        scheduler = (
            CheckpointableReduceLROnPlateau(
                optimizer=optimizer,
                factor=args.lr_factor,
                patience=(
                    args.effective_lr_patience
                ),
                min_lr=args.lr_min,
                min_delta=args.min_delta,
            )
        )

    del reference_model

    pf(
        f"[MODEL] trainable params="
        f"{student_model.count_params():,}"
    )

    for variable in (
        student_model
        .trainable_variables
    ):
        pf(
            f"[MODEL] "
            f"{variable.name} "
            f"shape="
            f"{tuple(variable.shape)}"
        )

    micro_batch_size = (
        args.batch_size
        // args.accumulation_steps
    )

    train_dataset = (
        make_kd_dataset(
            enc_train,
            tgt_train,
            tpred_train,
            batch_size=(
                args.batch_size
            ),
            accumulation_steps=(
                args.accumulation_steps
            ),
            seq_len=(
                args.seq_len
            ),
            n_out=(
                args.n_out
            ),
            shuffle=True,
            seed=(
                args.split_seed
            ),
            prefetch_batches=(
                args.prefetch_batches
            ),
            pipeline_workers=(
                args.pipeline_workers
            ),
        )
    )

    val_dataset = (
        make_kd_dataset(
            enc_val,
            tgt_val,
            tpred_val,
            batch_size=(
                args.batch_size
            ),
            accumulation_steps=(
                args.accumulation_steps
            ),
            seq_len=(
                args.seq_len
            ),
            n_out=(
                args.n_out
            ),
            shuffle=False,
            seed=(
                args.split_seed
            ),
            prefetch_batches=(
                args.prefetch_batches
            ),
            pipeline_workers=(
                args.pipeline_workers
            ),
        )
    )

    dist_train_dataset = (
        strategy
        .experimental_distribute_dataset(
            train_dataset
        )
    )

    dist_val_dataset = (
        strategy
        .experimental_distribute_dataset(
            val_dataset
        )
    )

    train_steps = (
        len(
            train_idx
        )
        // micro_batch_size
    )

    val_steps = (
        len(
            val_idx
        )
        // micro_batch_size
    )

    if (
        train_steps <= 0
        or val_steps <= 0
    ):
        raise RuntimeError(
            "Batch configuration "
            "produces zero train or "
            "validation steps"
        )

    (
        history,
        best_val,
        complete,
    ) = training_loop(
        strategy=strategy,
        student_model=student_model,
        optimizer=optimizer,
        scheduler=scheduler,
        dist_train_dataset=(
            dist_train_dataset
        ),
        dist_val_dataset=(
            dist_val_dataset
        ),
        train_steps=train_steps,
        val_steps=val_steps,
        args=args,
        job_dir=job_dir,
    )

    save_training_plot(
        history,
        job_dir,
    )

    if not complete:
        pf(
            "[RUN] Training paused after "
            "an epoch for SLURM "
            "resubmission. Evaluation "
            "is deferred."
        )

        return

    best_ckpt = (
        job_dir
        / "student_best.weights.h5"
    )

    if not best_ckpt.is_file():
        raise RuntimeError(
            f"Selected best checkpoint "
            f"is missing: {best_ckpt}"
        )

    student_model.load_weights(
        str(
            best_ckpt
        )
    )

    raw_weights_path = (
        save_raw_weights(
            student_model,
            job_dir,
        )
    )

    comparison = (
        evaluate_operator_comparison(
            model=student_model,
            normalized_input=(
                normalized_input
            ),
            res=res,
            labels=labels,
            test_idx=test_idx,
            args=args,
            job_dir=job_dir,
        )
    )

    final_weights = (
        job_dir
        / "student_final.weights.h5"
    )

    student_model.save_weights(
        str(
            final_weights
        )
    )

    source_path = Path(
        __file__
    ).resolve()

    baseline_source = (
        source_path.parent
        / "train_student_vanilla_kd.py"
    )

    recurrent_analysis_source = (
        source_path.parent
        / "eval"
        / "analyze_recurrent_memory.py"
    )

    manifest = {
        "job_name": (
            job_name
        ),
        "job_dir": (
            str(
                job_dir
            )
        ),
        "training_complete": (
            True
        ),
        "best_validation_loss": (
            float(
                best_val
            )
        ),
        "selected_checkpoint": (
            str(
                best_ckpt
            )
        ),
        "selected_checkpoint_sha256": (
            sha256_file(
                best_ckpt
            )
        ),
        "final_weights": (
            str(
                final_weights
            )
        ),
        "final_weights_sha256": (
            sha256_file(
                final_weights
            )
        ),
        "raw_weights_npz": (
            str(
                raw_weights_path
            )
        ),
        "raw_weights_npz_sha256": (
            sha256_file(
                raw_weights_path
            )
        ),
        "training_script": (
            str(
                source_path
            )
        ),
        "training_script_sha256": (
            sha256_file(
                source_path
            )
        ),
        "baseline_training_source": (
            str(
                baseline_source
            )
        ),
        "baseline_training_source_sha256": (
            sha256_file(
                baseline_source
            )
            if baseline_source.is_file()
            else None
        ),
        "recurrent_analysis_source": (
            str(
                recurrent_analysis_source
            )
        ),
        "recurrent_analysis_source_sha256": (
            sha256_file(
                recurrent_analysis_source
            )
            if recurrent_analysis_source.is_file()
            else None
        ),
        "teacher_checkpoint": (
            str(
                Path(
                    args.teacher_ckpt
                ).resolve()
            )
        ),
        "teacher_checkpoint_sha256": (
            sha256_file(
                Path(
                    args.teacher_ckpt
                ).resolve()
            )
        ),
        "test_index_file": (
            str(
                Path(
                    file_test
                ).resolve()
            )
        ),
        "test_index_sha256": (
            sha256_file(
                Path(
                    file_test
                ).resolve()
            )
        ),
        "random_seed": (
            int(
                args.split_seed
            )
        ),
        "training_operator": (
            student_model
            .sencgru
            .metadata()
        ),
        "initialization_equivalence": (
            initialization_equivalence
        ),
        "operator_comparison": (
            comparison
        ),
    }

    atomic_write_json(
        job_dir
        / "scw_training_manifest.json",
        manifest,
    )

    (
        job_dir
        / "scw_training_complete.flag"
    ).write_text(
        "passed\n",
        encoding="utf-8",
    )

    pf(
        "="
        * 72
    )

    pf(
        "SCW TRAINING COMPLETE"
    )

    pf(
        f"job_dir="
        f"{job_dir}"
    )

    pf(
        f"best_validation_loss="
        f"{best_val:.8g}"
    )

    for mode in (
        "scw",
        "deterministic",
        "identity",
    ):
        payload = (
            comparison[
                "operators"
            ][
                mode
            ]
        )

        pf(
            f"{mode}: "
            f"mae="
            f"{payload['mae_seq']:.6f} "
            f"tau1="
            f"{payload['tau1']['rmse']:.6f} "
            f"tau2="
            f"{payload['tau2']['rmse']:.6f}"
        )

    pf(
        "="
        * 72
    )


if __name__ == "__main__":
    main()