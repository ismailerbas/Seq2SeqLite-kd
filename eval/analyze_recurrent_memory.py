#!/usr/bin/env python3
"""
eval/analyze_recurrent_memory.py

Frozen-checkpoint recurrent-memory analysis for the Seq2SeqLite-kd repository.

This script is intentionally built on top of the already validated
``eval/analyze_writeback.py`` reconstruction. It does not train, fine-tune,
modify, or overwrite model weights. A previously passed native_fidelity.json
from ``eval/analyze_writeback.py`` is mandatory for every run.

The script adds two capabilities without changing the established analysis:

1. Mechanistic recurrence-margin decomposition

       delta_t = h_t - q_recv_t
               = (1 - z_t) * (candidate_t - q_recv_t)

   for recurrence-visible transitions that are consumed by a subsequent step.
   It records the normalized write margin

       M_t = |delta_t| / (Delta / 2)

   together with the gate factor (1-z_t), candidate displacement, rail
   occupancy, and observed write events. For deterministic round-to-nearest
   writeback, the half-step criterion is checked on interior non-tie events.

2. Recurrent writeback family

   * identity
   * deterministic round-to-nearest
   * stochastic rounding
   * existing full error feedback, preserving the repository's +/-Delta clip
   * fixed-point quantized residual memory with r in {2,3,4}
   * float residual clipped to +/-Delta/2
   * Saturating-Counter Writeback (SCW) with k in {2,3,4}

SCW definition
--------------
For k counter bits, let T = 2^(k-1). The stored signed counter uses the
2*T-1 active integer states

    -(T-1), ..., -1, 0, 1, ..., T-1

and therefore fits in k bits with one binary codeword unused. For an interior
sub-half-step innovation, the counter receives a signed vote. When the
combinational candidate counter reaches +T or -T, one B-bit state level is
emitted in that direction and the stored counter resets to zero. The trigger
value itself never needs to be stored.

A dead-zone theta can suppress tiny innovations. This study locks theta to
0 or Delta/8. Values inside the dead-zone cast no vote and leave the stored
counter unchanged. A normal half-step-or-larger update uses the ordinary B-bit
round-to-nearest state write and resets the counter.

The encoder and decoder auxiliary writeback memories are independent. At the
encoder-to-decoder boundary, the recurrent hidden value crosses the boundary,
while residual/counter memory is reset. Quantized/full residual modes then
capture the decoder handoff quantization residual exactly as their own
writeback rule dictates; SCW starts with a zero counter after handoff.

Every run performs an adapter-equivalence test showing that this file's custom
unroll reproduces the established ``build_forward_fn`` for an unchanged
control operator before any new operator is evaluated.
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

import numpy as np
import tensorflow as tf

THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
REPO_ROOT = EVAL_DIR.parent

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyze_writeback import (  # noqa: E402
    CHECKPOINT_NAMES,
    PassAccumulator,
    WritebackOperator,
    atomic_write_json,
    build_forward_fn,
    build_original_reference,
    build_parameter_quantizers,
    configure_tensorflow,
    cross_seed_summary,
    extract_raw_weights,
    histogram_quantile,
    json_safe,
    load_dataset,
    load_run_config,
    pf,
    qkeras_hard_sigmoid,
    quantize_effective_weights,
    run_tensor_equivalence,
    sha256_file,
    summarize_pass,
    tensor_error,
    validate_fidelity_file,
    write_per_sequence_npz,
    write_region_outputs,
)


SUPPORTED_PHASES = (
    "P2E",
    "P2F",
    "P3",
    "VANILLA",
)

SUPPORTED_METHODS = (
    "identity",
    "deterministic",
    "stochastic",
    "error_feedback",
    "quantized_residual",
    "full_halfstep_residual",
    "scw",
)

RESIDUAL_BITS_ALLOWED = (
    2,
    3,
    4,
)

SCW_COUNTER_BITS_ALLOWED = (
    2,
    3,
    4,
)

SCW_DEADZONE_FRACTIONS_ALLOWED = (
    0.0,
    0.125,
)

DEFAULT_GATE_WIDTH_NS = 0.09
DEFAULT_ADAPTER_TOLERANCE = 1e-6
DEFAULT_DECOMPOSITION_TOLERANCE = 5e-6
DEFAULT_MARGIN_TIE_TOLERANCE = 1e-4
DEFAULT_CRITERION_MISMATCH_FRACTION = 1e-6
GRID_EPS = 1e-6

EVENT_HOLD = 0
EVENT_NORMAL_WRITE = 1
EVENT_NORMAL_NO_WRITE = 2
EVENT_SCW_VOTE_HOLD = 3
EVENT_SCW_FORCED_WRITE = 4
EVENT_SCW_DEADZONE_HOLD = 5
EVENT_SCW_FORCED_RAIL_BLOCK = 6

EVENT_NAMES = {
    EVENT_HOLD: "hold",
    EVENT_NORMAL_WRITE: "normal_write",
    EVENT_NORMAL_NO_WRITE: "normal_no_write",
    EVENT_SCW_VOTE_HOLD: "scw_vote_hold",
    EVENT_SCW_FORCED_WRITE: "scw_forced_write",
    EVENT_SCW_DEADZONE_HOLD: "scw_deadzone_hold",
    EVENT_SCW_FORCED_RAIL_BLOCK: "scw_forced_rail_block",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen recurrent-margin, residual-memory, and SCW analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        choices=SUPPORTED_PHASES,
    )

    parser.add_argument(
        "--checkpoint-file",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--condition-name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--method",
        required=True,
        choices=SUPPORTED_METHODS,
    )

    parser.add_argument(
        "--state-bits",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--residual-bits",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--counter-bits",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--scw-deadzone-fraction",
        default=0.0,
        type=float,
        choices=SCW_DEADZONE_FRACTIONS_ALLOWED,
    )

    parser.add_argument(
        "--fidelity-json",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--infer-batch",
        default=2048,
        type=int,
    )

    parser.add_argument(
        "--hist-bins",
        default=4096,
        type=int,
    )

    parser.add_argument(
        "--gate-hist-bins",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--mechanism-hist-bins",
        default=2048,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-reps",
        default=2000,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-batch-reps",
        default=64,
        type=int,
    )

    parser.add_argument(
        "--sr-seeds",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--sr-seed-base",
        default=1000,
        type=int,
    )

    parser.add_argument(
        "--gate-width-ns",
        default=DEFAULT_GATE_WIDTH_NS,
        type=float,
    )

    parser.add_argument(
        "--equivalence-samples",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--adapter-tolerance",
        default=DEFAULT_ADAPTER_TOLERANCE,
        type=float,
    )

    parser.add_argument(
        "--decomposition-tolerance",
        default=DEFAULT_DECOMPOSITION_TOLERANCE,
        type=float,
    )

    parser.add_argument(
        "--margin-tie-tolerance",
        default=DEFAULT_MARGIN_TIE_TOLERANCE,
        type=float,
    )

    parser.add_argument(
        "--criterion-mismatch-fraction",
        default=DEFAULT_CRITERION_MISMATCH_FRACTION,
        type=float,
    )

    parser.add_argument(
        "--native-tensor-tolerance",
        default=5e-5,
        type=float,
    )

    parser.add_argument(
        "--native-tensor-mean-tolerance",
        default=5e-5,
        type=float,
    )

    parser.add_argument(
        "--native-tensor-mismatch-fraction",
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--native-tensor-tie-fraction",
        default=1e-3,
        type=float,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    fidelity_json = Path(args.fidelity_json).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    if not (run_dir / "student_args.json").is_file():
        raise FileNotFoundError(
            f"Missing student_args.json in {run_dir}"
        )

    if not fidelity_json.is_file():
        raise FileNotFoundError(
            f"Required native fidelity JSON does not exist: {fidelity_json}"
        )

    if not (2 <= args.state_bits <= 16):
        raise ValueError(
            "--state-bits must be an integer in [2,16]"
        )

    if args.infer_batch <= 0:
        raise ValueError(
            "--infer-batch must be > 0"
        )

    if args.hist_bins < 128:
        raise ValueError(
            "--hist-bins must be >= 128"
        )

    if args.gate_hist_bins < 32:
        raise ValueError(
            "--gate-hist-bins must be >= 32"
        )

    if args.mechanism_hist_bins < 128:
        raise ValueError(
            "--mechanism-hist-bins must be >= 128"
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

    if args.equivalence_samples <= 0:
        raise ValueError(
            "--equivalence-samples must be > 0"
        )

    if args.adapter_tolerance <= 0.0:
        raise ValueError(
            "--adapter-tolerance must be > 0"
        )

    if args.decomposition_tolerance <= 0.0:
        raise ValueError(
            "--decomposition-tolerance must be > 0"
        )

    if args.margin_tie_tolerance <= 0.0:
        raise ValueError(
            "--margin-tie-tolerance must be > 0"
        )

    if not (
        0.0
        <= args.criterion_mismatch_fraction
        < 1.0
    ):
        raise ValueError(
            "--criterion-mismatch-fraction must be in [0,1)"
        )

    if args.gate_width_ns <= 0.0:
        raise ValueError(
            "--gate-width-ns must be > 0"
        )

    if args.method == "quantized_residual":
        if args.residual_bits not in RESIDUAL_BITS_ALLOWED:
            raise ValueError(
                "quantized_residual requires --residual-bits in {2,3,4}"
            )

        if args.counter_bits != -1:
            raise ValueError(
                "--counter-bits is not valid for quantized_residual"
            )

        if args.scw_deadzone_fraction != 0.0:
            raise ValueError(
                "--scw-deadzone-fraction is only valid for SCW"
            )

    elif args.method == "scw":
        if args.counter_bits not in SCW_COUNTER_BITS_ALLOWED:
            raise ValueError(
                "scw requires --counter-bits in {2,3,4}"
            )

        if args.residual_bits != -1:
            raise ValueError(
                "--residual-bits is not valid for SCW"
            )

    else:
        if args.residual_bits != -1:
            raise ValueError(
                f"--residual-bits is not valid for method {args.method}"
            )

        if args.counter_bits != -1:
            raise ValueError(
                f"--counter-bits is not valid for method {args.method}"
            )

        if args.scw_deadzone_fraction != 0.0:
            raise ValueError(
                "--scw-deadzone-fraction is only valid for SCW"
            )

    if (
        args.phase == "P2E"
        and args.method != "identity"
    ):
        raise ValueError(
            "P2E is included only as its native continuous-state identity control"
        )

    if (
        args.method == "identity"
        and args.phase != "P2E"
    ):
        raise ValueError(
            "identity is reserved for the P2E continuous-state control"
        )


def checkpoint_path_for(args: argparse.Namespace, run_dir: Path) -> Path:
    name = args.checkpoint_file or CHECKPOINT_NAMES[args.phase]
    path = (run_dir / name).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    return path


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class StatefulWritebackBase:
    def __init__(self, kind: str, state_bits: int) -> None:
        self.kind = str(kind)
        self.state_bits = int(state_bits)
        self.delta = float(
            2.0 ** (-(self.state_bits - 1))
        )
        self.qmax = float(1.0 - self.delta)
        self.qmin = float(-self.qmax)
        self.live_discrete = self.kind != "identity"

    def initialize(
        self,
        raw_state: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def advance(
        self,
        raw_state: tf.Tensor,
        q_prev: tf.Tensor,
        aux_prev: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def metadata(self) -> Dict:
        return {
            "kind": self.kind,
            "state_bits": self.state_bits,
            "delta": self.delta,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "live_discrete": self.live_discrete,
        }


class ExistingWritebackAdapter(StatefulWritebackBase):
    def __init__(self, kind: str, state_bits: int) -> None:
        if kind not in (
            "identity",
            "deterministic",
            "stochastic",
            "error_feedback",
        ):
            raise ValueError(
                f"Unsupported existing writeback kind: {kind}"
            )

        super().__init__(kind, state_bits)
        self.base = WritebackOperator(kind, state_bits)

    def initialize(
        self,
        raw_state: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        aux = tf.zeros_like(
            tf.cast(raw_state, tf.float32)
        )

        return self.base.apply(
            raw_state,
            aux,
            seed_pair,
        )

    def advance(
        self,
        raw_state: tf.Tensor,
        q_prev: tf.Tensor,
        aux_prev: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        q_next, aux_next = self.base.apply(
            raw_state,
            aux_prev,
            seed_pair,
        )

        if self.kind == "identity":
            event = tf.zeros(
                tf.shape(q_next),
                tf.int32,
            )

        else:
            changed = tf.abs(
                q_next - q_prev
            ) > tf.constant(
                self.delta * 1e-4,
                tf.float32,
            )

            event = tf.where(
                changed,
                tf.fill(
                    tf.shape(q_next),
                    tf.constant(
                        EVENT_NORMAL_WRITE,
                        tf.int32,
                    ),
                ),
                tf.fill(
                    tf.shape(q_next),
                    tf.constant(
                        EVENT_HOLD,
                        tf.int32,
                    ),
                ),
            )

        return (
            tf.cast(q_next, tf.float32),
            tf.cast(aux_next, tf.float32),
            event,
        )

    def metadata(self) -> Dict:
        result = super().metadata()
        result.update(
            {
                "source": "eval/analyze_writeback.py::WritebackOperator",
                "auxiliary_memory": (
                    "float32 residual"
                    if self.kind == "error_feedback"
                    else "none"
                ),
                "residual_clip": (
                    self.delta
                    if self.kind == "error_feedback"
                    else None
                ),
                "rng_required": self.kind == "stochastic",
                "deterministic": self.kind != "stochastic",
            }
        )
        return result


class QuantizedResidualWriteback(StatefulWritebackBase):
    def __init__(self, state_bits: int, residual_bits: int) -> None:
        if residual_bits not in RESIDUAL_BITS_ALLOWED:
            raise ValueError(
                f"residual_bits must be in {RESIDUAL_BITS_ALLOWED}"
            )

        super().__init__(
            f"quantized_residual_r{residual_bits}",
            state_bits,
        )

        self.residual_bits = int(residual_bits)
        self.levels = 2 ** self.residual_bits
        self.half_step = self.delta / 2.0
        self.residual_step = self.delta / self.levels
        self.residual_min = -self.half_step
        self.residual_max = (
            self.half_step
            - self.residual_step
        )
        self.state_quantizer = WritebackOperator(
            "deterministic",
            state_bits,
        )

    def quantize_residual(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, tf.float32)

        rmin = tf.constant(
            self.residual_min,
            tf.float32,
        )
        rmax = tf.constant(
            self.residual_max,
            tf.float32,
        )
        step = tf.constant(
            self.residual_step,
            tf.float32,
        )
        levels_minus_one = tf.constant(
            self.levels - 1,
            tf.float32,
        )

        clipped = tf.clip_by_value(
            value,
            rmin,
            rmax,
        )

        code = tf.round(
            (clipped - rmin) / step
        )

        code = tf.clip_by_value(
            code,
            0.0,
            levels_minus_one,
        )

        quantized = rmin + code * step

        return tf.cast(
            quantized,
            tf.float32,
        )

    def initialize(
        self,
        raw_state: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        del seed_pair

        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        q = self.state_quantizer.deterministic_quantize(
            raw_state
        )

        residual = self.quantize_residual(
            raw_state - q
        )

        return (
            q,
            residual,
        )

    def advance(
        self,
        raw_state: tf.Tensor,
        q_prev: tf.Tensor,
        aux_prev: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        del seed_pair

        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        aux_prev = tf.cast(
            aux_prev,
            tf.float32,
        )

        compensated = raw_state + aux_prev

        q_next = self.state_quantizer.deterministic_quantize(
            compensated
        )

        raw_residual = compensated - q_next

        aux_next = self.quantize_residual(
            raw_residual
        )

        changed = tf.abs(
            q_next - q_prev
        ) > tf.constant(
            self.delta * 1e-4,
            tf.float32,
        )

        event = tf.where(
            changed,
            tf.fill(
                tf.shape(q_next),
                tf.constant(
                    EVENT_NORMAL_WRITE,
                    tf.int32,
                ),
            ),
            tf.fill(
                tf.shape(q_next),
                tf.constant(
                    EVENT_HOLD,
                    tf.int32,
                ),
            ),
        )

        return (
            q_next,
            aux_next,
            event,
        )

    def metadata(self) -> Dict:
        result = super().metadata()
        result.update(
            {
                "residual_bits": self.residual_bits,
                "residual_levels": self.levels,
                "residual_step": self.residual_step,
                "residual_min": self.residual_min,
                "residual_max": self.residual_max,
                "residual_codebook": (
                    "two's-complement-like fixed-point levels spanning "
                    "[-Delta/2, Delta/2) with spacing Delta/2^r"
                ),
                "auxiliary_memory": f"{self.residual_bits}-bit residual",
                "rng_required": False,
                "deterministic": True,
                "recurrence_visible_state_bits": self.state_bits,
                "total_stored_bits_per_unit": (
                    self.state_bits + self.residual_bits
                ),
            }
        )
        return result


class FullHalfstepResidualWriteback(StatefulWritebackBase):
    def __init__(self, state_bits: int) -> None:
        super().__init__(
            "full_halfstep_residual",
            state_bits,
        )
        self.half_step = self.delta / 2.0
        self.state_quantizer = WritebackOperator(
            "deterministic",
            state_bits,
        )

    def clip_residual(self, value: tf.Tensor) -> tf.Tensor:
        half = tf.constant(
            self.half_step,
            tf.float32,
        )
        return tf.clip_by_value(
            tf.cast(value, tf.float32),
            -half,
            half,
        )

    def initialize(
        self,
        raw_state: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        del seed_pair

        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        q = self.state_quantizer.deterministic_quantize(
            raw_state
        )

        residual = self.clip_residual(
            raw_state - q
        )

        return (
            q,
            residual,
        )

    def advance(
        self,
        raw_state: tf.Tensor,
        q_prev: tf.Tensor,
        aux_prev: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        del seed_pair

        compensated = (
            tf.cast(raw_state, tf.float32)
            + tf.cast(aux_prev, tf.float32)
        )

        q_next = self.state_quantizer.deterministic_quantize(
            compensated
        )

        aux_next = self.clip_residual(
            compensated - q_next
        )

        changed = tf.abs(
            q_next - q_prev
        ) > tf.constant(
            self.delta * 1e-4,
            tf.float32,
        )

        event = tf.where(
            changed,
            tf.fill(
                tf.shape(q_next),
                tf.constant(
                    EVENT_NORMAL_WRITE,
                    tf.int32,
                ),
            ),
            tf.fill(
                tf.shape(q_next),
                tf.constant(
                    EVENT_HOLD,
                    tf.int32,
                ),
            ),
        )

        return (
            q_next,
            aux_next,
            event,
        )

    def metadata(self) -> Dict:
        result = super().metadata()
        result.update(
            {
                "residual_bits": None,
                "residual_clip": self.half_step,
                "auxiliary_memory": "float32 residual clipped to +/-Delta/2",
                "rng_required": False,
                "deterministic": True,
                "recurrence_visible_state_bits": self.state_bits,
                "total_stored_bits_per_unit": None,
            }
        )
        return result


class SaturatingCounterWriteback(StatefulWritebackBase):
    def __init__(
        self,
        state_bits: int,
        counter_bits: int,
        deadzone_fraction: float,
    ) -> None:
        if counter_bits not in SCW_COUNTER_BITS_ALLOWED:
            raise ValueError(
                f"counter_bits must be in {SCW_COUNTER_BITS_ALLOWED}"
            )

        if deadzone_fraction not in SCW_DEADZONE_FRACTIONS_ALLOWED:
            raise ValueError(
                "deadzone_fraction must be 0 or 0.125"
            )

        super().__init__(
            f"scw_k{counter_bits}",
            state_bits,
        )

        self.counter_bits = int(counter_bits)
        self.trigger_votes = 2 ** (self.counter_bits - 1)
        self.counter_min = -(self.trigger_votes - 1)
        self.counter_max = self.trigger_votes - 1
        self.deadzone_fraction = float(deadzone_fraction)
        self.deadzone = self.deadzone_fraction * self.delta
        self.half_step = self.delta / 2.0
        self.state_quantizer = WritebackOperator(
            "deterministic",
            state_bits,
        )

    def initialize(
        self,
        raw_state: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        del seed_pair

        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        q = self.state_quantizer.deterministic_quantize(
            raw_state
        )

        counter = tf.zeros_like(
            q,
            dtype=tf.float32,
        )

        return (
            q,
            counter,
        )

    def advance(
        self,
        raw_state: tf.Tensor,
        q_prev: tf.Tensor,
        aux_prev: tf.Tensor,
        seed_pair: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        del seed_pair

        raw_state = tf.cast(
            raw_state,
            tf.float32,
        )

        q_prev = tf.cast(
            q_prev,
            tf.float32,
        )

        counter_prev = tf.cast(
            aux_prev,
            tf.float32,
        )

        counter_rounded = tf.round(
            counter_prev
        )

        tf.debugging.assert_near(
            counter_prev,
            counter_rounded,
            atol=1e-6,
            rtol=0.0,
            message="SCW counter lost integer semantics",
        )

        tf.debugging.assert_less_equal(
            counter_rounded,
            tf.constant(
                float(self.counter_max),
                tf.float32,
            ),
            message="SCW counter exceeded positive stored range",
        )

        tf.debugging.assert_greater_equal(
            counter_rounded,
            tf.constant(
                float(self.counter_min),
                tf.float32,
            ),
            message="SCW counter exceeded negative stored range",
        )

        delta_raw = raw_state - q_prev
        abs_delta = tf.abs(delta_raw)

        half = tf.constant(
            self.half_step,
            tf.float32,
        )

        theta = tf.constant(
            self.deadzone,
            tf.float32,
        )

        normal = abs_delta >= half
        subthreshold = ~normal
        active_vote = (
            subthreshold
            & (abs_delta > theta)
        )
        deadzone_hold = (
            subthreshold
            & (~active_vote)
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
                    float(self.trigger_votes),
                    tf.float32,
                )
            )
        )

        negative_trigger = (
            active_vote
            & (
                counter_candidate
                <= tf.constant(
                    float(-self.trigger_votes),
                    tf.float32,
                )
            )
        )

        trigger = positive_trigger | negative_trigger

        q_normal = self.state_quantizer.deterministic_quantize(
            raw_state
        )

        forced_direction = (
            tf.cast(positive_trigger, tf.float32)
            - tf.cast(negative_trigger, tf.float32)
        )

        q_forced = tf.clip_by_value(
            q_prev
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

        q_next = tf.where(
            normal,
            q_normal,
            tf.where(
                trigger,
                q_forced,
                q_prev,
            ),
        )

        counter_active_hold = tf.clip_by_value(
            counter_candidate,
            tf.constant(
                float(self.counter_min),
                tf.float32,
            ),
            tf.constant(
                float(self.counter_max),
                tf.float32,
            ),
        )

        counter_next = tf.where(
            normal | trigger,
            tf.zeros_like(counter_prev),
            tf.where(
                active_vote,
                counter_active_hold,
                counter_rounded,
            ),
        )

        normal_changed = (
            normal
            & (
                tf.abs(q_normal - q_prev)
                > tf.constant(
                    self.delta * 1e-4,
                    tf.float32,
                )
            )
        )

        normal_no_write = (
            normal
            & (~normal_changed)
        )

        forced_changed = (
            trigger
            & (
                tf.abs(q_forced - q_prev)
                > tf.constant(
                    self.delta * 1e-4,
                    tf.float32,
                )
            )
        )

        forced_rail_block = (
            trigger
            & (~forced_changed)
        )

        vote_hold = (
            active_vote
            & (~trigger)
        )

        event = tf.fill(
            tf.shape(q_next),
            tf.constant(
                EVENT_HOLD,
                tf.int32,
            ),
        )

        event = tf.where(
            deadzone_hold,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_SCW_DEADZONE_HOLD,
                    tf.int32,
                ),
            ),
            event,
        )

        event = tf.where(
            vote_hold,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_SCW_VOTE_HOLD,
                    tf.int32,
                ),
            ),
            event,
        )

        event = tf.where(
            normal_no_write,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_NORMAL_NO_WRITE,
                    tf.int32,
                ),
            ),
            event,
        )

        event = tf.where(
            normal_changed,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_NORMAL_WRITE,
                    tf.int32,
                ),
            ),
            event,
        )

        event = tf.where(
            forced_changed,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_SCW_FORCED_WRITE,
                    tf.int32,
                ),
            ),
            event,
        )

        event = tf.where(
            forced_rail_block,
            tf.fill(
                tf.shape(event),
                tf.constant(
                    EVENT_SCW_FORCED_RAIL_BLOCK,
                    tf.int32,
                ),
            ),
            event,
        )

        return (
            tf.cast(q_next, tf.float32),
            tf.cast(counter_next, tf.float32),
            event,
        )

    def metadata(self) -> Dict:
        result = super().metadata()
        result.update(
            {
                "counter_bits": self.counter_bits,
                "trigger_votes": self.trigger_votes,
                "stored_counter_min": self.counter_min,
                "stored_counter_max": self.counter_max,
                "stored_active_counter_states": (
                    2 * self.trigger_votes - 1
                ),
                "unused_binary_codewords": 1,
                "deadzone_fraction_of_delta": self.deadzone_fraction,
                "deadzone_absolute": self.deadzone,
                "half_step": self.half_step,
                "deadzone_semantics": (
                    "inside dead-zone casts no vote and preserves current counter"
                ),
                "normal_write_semantics": (
                    "abs(h-q)>=Delta/2 uses ordinary deterministic state rounding "
                    "and resets the counter"
                ),
                "forced_write_semantics": (
                    "counter candidate reaching +/-2^(k-1) emits one +/-Delta state "
                    "step and resets the stored counter"
                ),
                "rail_semantics": (
                    "forced state step saturates to the B-bit state rail and counter resets"
                ),
                "auxiliary_memory": f"{self.counter_bits}-bit signed vote counter",
                "rng_required": False,
                "deterministic": True,
                "recurrence_visible_state_bits": self.state_bits,
                "total_stored_bits_per_unit": (
                    self.state_bits + self.counter_bits
                ),
            }
        )
        return result


def make_operator(args: argparse.Namespace) -> StatefulWritebackBase:
    if args.method in (
        "identity",
        "deterministic",
        "stochastic",
        "error_feedback",
    ):
        return ExistingWritebackAdapter(
            args.method,
            args.state_bits,
        )

    if args.method == "quantized_residual":
        return QuantizedResidualWriteback(
            args.state_bits,
            args.residual_bits,
        )

    if args.method == "full_halfstep_residual":
        return FullHalfstepResidualWriteback(
            args.state_bits
        )

    if args.method == "scw":
        return SaturatingCounterWriteback(
            args.state_bits,
            args.counter_bits,
            args.scw_deadzone_fraction,
        )

    raise ValueError(
        f"Unsupported method: {args.method}"
    )


def build_stateful_forward_fn(
    effective_weights: Dict[str, np.ndarray],
    quantizers: Dict,
    enc_operator: StatefulWritebackBase,
    dec_operator: StatefulWritebackBase,
    cfg: Dict,
):
    seq_len = int(cfg["seq_len"])
    H = int(cfg["student_units"])

    enc_kernel = tf.constant(
        effective_weights["enc_kernel"],
        tf.float32,
    )
    enc_recurrent = tf.constant(
        effective_weights["enc_recurrent"],
        tf.float32,
    )
    enc_bias = tf.constant(
        effective_weights["enc_bias"],
        tf.float32,
    )

    dec_kernel = tf.constant(
        effective_weights["dec_kernel"],
        tf.float32,
    )
    dec_recurrent = tf.constant(
        effective_weights["dec_recurrent"],
        tf.float32,
    )
    dec_bias = tf.constant(
        effective_weights["dec_bias"],
        tf.float32,
    )

    dense_kernel = tf.constant(
        effective_weights["dense_kernel"],
        tf.float32,
    )
    dense_bias = tf.constant(
        effective_weights["dense_bias"],
        tf.float32,
    )

    q_activation = quantizers["activation"]

    def gru_step(
        x_t: tf.Tensor,
        q_recv: tf.Tensor,
        kernel_q: tf.Tensor,
        recurrent_q: tf.Tensor,
        bias_q: tf.Tensor,
    ):
        x_z = (
            tf.matmul(
                x_t,
                kernel_q[:, :H],
            )
            + bias_q[:H]
        )

        x_r = (
            tf.matmul(
                x_t,
                kernel_q[:, H : 2 * H],
            )
            + bias_q[H : 2 * H]
        )

        x_h = (
            tf.matmul(
                x_t,
                kernel_q[:, 2 * H :],
            )
            + bias_q[2 * H :]
        )

        z = qkeras_hard_sigmoid(
            x_z
            + tf.matmul(
                q_recv,
                recurrent_q[:, :H],
            )
        )

        r = qkeras_hard_sigmoid(
            x_r
            + tf.matmul(
                q_recv,
                recurrent_q[:, H : 2 * H],
            )
        )

        preact = (
            x_h
            + tf.matmul(
                r * q_recv,
                recurrent_q[:, 2 * H :],
            )
        )

        if q_activation is None:
            candidate = tf.tanh(
                preact
            )
        else:
            candidate = tf.cast(
                q_activation(
                    preact
                ),
                tf.float32,
            )

        h = (
            z * q_recv
            + (1.0 - z) * candidate
        )

        return (
            h,
            z,
            r,
            candidate,
        )

    @tf.function(reduce_retracing=True)
    def forward(
        enc_inputs: tf.Tensor,
        sr_seed: tf.Tensor,
        batch_ordinal: tf.Tensor,
    ):
        enc_inputs = tf.cast(
            enc_inputs,
            tf.float32,
        )

        batch = tf.shape(enc_inputs)[0]

        base_counter = (
            batch_ordinal
            * tf.constant(
                4 * seq_len + 16,
                tf.int32,
            )
        )

        zero_raw = tf.zeros(
            (batch, H),
            tf.float32,
        )

        enc_seed0 = tf.stack(
            [
                sr_seed,
                base_counter,
            ]
        )

        q_recv, enc_aux = enc_operator.initialize(
            zero_raw,
            enc_seed0,
        )

        enc_h_steps = []
        enc_q_steps = []
        enc_z_steps = []
        enc_r_steps = []
        enc_c_steps = []
        enc_aux_steps = []
        enc_event_steps = []

        h_prev = zero_raw

        for t in range(seq_len):
            enc_q_steps.append(q_recv)
            enc_aux_steps.append(enc_aux)

            h_prev, z, r, candidate = gru_step(
                enc_inputs[:, t, :],
                q_recv,
                enc_kernel,
                enc_recurrent,
                enc_bias,
            )

            enc_h_steps.append(h_prev)
            enc_z_steps.append(z)
            enc_r_steps.append(r)
            enc_c_steps.append(candidate)

            if t < seq_len - 1:
                seed_pair = tf.stack(
                    [
                        sr_seed,
                        base_counter
                        + tf.constant(
                            t + 1,
                            tf.int32,
                        ),
                    ]
                )

                q_recv, enc_aux, event = enc_operator.advance(
                    h_prev,
                    q_recv,
                    enc_aux,
                    seed_pair,
                )
            else:
                event = tf.zeros(
                    tf.shape(h_prev),
                    tf.int32,
                )

            enc_event_steps.append(event)

        enc_final_raw = h_prev

        dec_seed0 = tf.stack(
            [
                sr_seed
                + tf.constant(
                    1000003,
                    tf.int32,
                ),
                base_counter
                + tf.constant(
                    2 * seq_len,
                    tf.int32,
                ),
            ]
        )

        q_recv, dec_aux = dec_operator.initialize(
            enc_final_raw,
            dec_seed0,
        )

        dec_h_steps = []
        dec_q_steps = []
        dec_z_steps = []
        dec_r_steps = []
        dec_c_steps = []
        dec_aux_steps = []
        dec_event_steps = []

        dec_x = tf.zeros(
            (batch, 1),
            tf.float32,
        )

        for t in range(seq_len):
            dec_q_steps.append(q_recv)
            dec_aux_steps.append(dec_aux)

            h_prev, z, r, candidate = gru_step(
                dec_x,
                q_recv,
                dec_kernel,
                dec_recurrent,
                dec_bias,
            )

            dec_h_steps.append(h_prev)
            dec_z_steps.append(z)
            dec_r_steps.append(r)
            dec_c_steps.append(candidate)

            if t < seq_len - 1:
                seed_pair = tf.stack(
                    [
                        sr_seed
                        + tf.constant(
                            1000003,
                            tf.int32,
                        ),
                        base_counter
                        + tf.constant(
                            2 * seq_len + t + 1,
                            tf.int32,
                        ),
                    ]
                )

                q_recv, dec_aux, event = dec_operator.advance(
                    h_prev,
                    q_recv,
                    dec_aux,
                    seed_pair,
                )
            else:
                event = tf.zeros(
                    tf.shape(h_prev),
                    tf.int32,
                )

            dec_event_steps.append(event)

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

        enc_aux_tensor = tf.stack(
            enc_aux_steps,
            axis=1,
        )
        dec_aux_tensor = tf.stack(
            dec_aux_steps,
            axis=1,
        )
        enc_event_tensor = tf.stack(
            enc_event_steps,
            axis=1,
        )
        dec_event_tensor = tf.stack(
            dec_event_steps,
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
            enc_aux_tensor,
            dec_aux_tensor,
            enc_event_tensor,
            dec_event_tensor,
        )

    return forward


def run_adapter_equivalence(
    base_forward,
    custom_forward,
    normalized_input: np.ndarray,
    test_idx: np.ndarray,
    n_samples: int,
    tolerance: float,
    sr_seed: int,
) -> Dict:
    n = min(
        int(n_samples),
        len(test_idx),
    )

    rows = test_idx[:n]

    enc = np.asarray(
        normalized_input[rows],
        dtype=np.float32,
    )

    enc_tf = tf.convert_to_tensor(
        enc,
        tf.float32,
    )

    base_outputs = base_forward(
        enc_tf,
        tf.constant(
            sr_seed,
            tf.int32,
        ),
        tf.constant(
            0,
            tf.int32,
        ),
    )

    custom_outputs = custom_forward(
        enc_tf,
        tf.constant(
            sr_seed,
            tf.int32,
        ),
        tf.constant(
            0,
            tf.int32,
        ),
    )

    names = (
        "preds",
        "enc_h",
        "enc_q",
        "enc_z",
        "enc_r",
        "enc_candidate",
        "dec_h",
        "dec_q",
        "dec_z",
        "dec_r",
        "dec_candidate",
        "enc_final_raw",
    )

    checks = []
    failed = []

    for index, name in enumerate(names):
        reference = np.asarray(
            base_outputs[index].numpy(),
            dtype=np.float32,
        )

        reconstructed = np.asarray(
            custom_outputs[index].numpy(),
            dtype=np.float32,
        )

        row = tensor_error(
            name,
            reference,
            reconstructed,
            tolerance,
        )

        checks.append(row)

        if row["max_abs"] > tolerance:
            failed.append(row)

        pf(
            f"[ADAPTER] {name}: "
            f"max_abs={row['max_abs']:.9g} "
            f"mean_abs={row['mean_abs']:.9g}"
        )

    if failed:
        raise RuntimeError(
            "Custom recurrent-memory unroll failed adapter equivalence: "
            + "; ".join(
                f"{row['name']} max_abs={row['max_abs']:.9g}"
                for row in failed
            )
        )

    pf(
        f"[ADAPTER] PASS on {n} held-out sequences with tolerance {tolerance:.9g}"
    )

    return {
        "passed": True,
        "n_samples": int(n),
        "tolerance": float(tolerance),
        "sr_seed": int(sr_seed),
        "checks": checks,
    }


def make_margin_edges(
    state_bits: int,
    bins: int,
) -> np.ndarray:
    delta = float(
        2.0 ** (-(state_bits - 1))
    )
    half_step = delta / 2.0
    max_margin = 2.0 / half_step

    positive = np.geomspace(
        1e-6,
        max_margin * (1.0 + 1e-6),
        bins,
    )

    edges = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 1.0]),
                positive,
            )
        )
    )

    edges.sort()

    if len(edges) < 32:
        raise RuntimeError(
            "Failed to construct recurrence-margin histogram edges"
        )

    return edges.astype(
        np.float64
    )


class MechanismAccumulator:
    def __init__(
        self,
        name: str,
        n_sequences: int,
        seq_len: int,
        units: int,
        operator: StatefulWritebackBase,
        hist_bins: int,
        margin_tie_tolerance: float,
    ) -> None:
        self.name = str(name)
        self.n_sequences = int(n_sequences)
        self.seq_len = int(seq_len)
        self.units = int(units)
        self.operator = operator
        self.state_bits = int(operator.state_bits)
        self.delta = float(operator.delta)
        self.half_step = self.delta / 2.0
        self.qmax = float(operator.qmax)
        self.margin_tie_tolerance = float(
            margin_tie_tolerance
        )

        self.aligned_steps = self.seq_len - 1
        self.aligned_per_sequence = (
            self.aligned_steps * self.units
        )

        self.margin_edges = make_margin_edges(
            self.state_bits,
            hist_bins,
        )
        self.margin_counts = np.zeros(
            len(self.margin_edges) - 1,
            dtype=np.int64,
        )

        self.gate_edges = np.linspace(
            0.0,
            1.0 + GRID_EPS,
            hist_bins + 1,
            dtype=np.float64,
        )
        self.gate_counts = np.zeros(
            hist_bins,
            dtype=np.int64,
        )

        self.candidate_delta_edges = np.linspace(
            0.0,
            2.0 + GRID_EPS,
            hist_bins + 1,
            dtype=np.float64,
        )
        self.candidate_delta_counts = np.zeros(
            hist_bins,
            dtype=np.int64,
        )

        self.unit_margin_counts = np.zeros(
            (
                self.units,
                len(self.margin_edges) - 1,
            ),
            dtype=np.int64,
        )

        self.unit_deadband_counts = np.zeros(
            self.units,
            dtype=np.int64,
        )
        self.unit_write_counts = np.zeros(
            self.units,
            dtype=np.int64,
        )
        self.unit_gate_factor_sum = np.zeros(
            self.units,
            dtype=np.float64,
        )
        self.unit_candidate_delta_sum = np.zeros(
            self.units,
            dtype=np.float64,
        )
        self.unit_aux_abs_sum = np.zeros(
            self.units,
            dtype=np.float64,
        )

        self.seq_deadband_counts = np.zeros(
            self.n_sequences,
            dtype=np.int64,
        )
        self.seq_write_counts = np.zeros(
            self.n_sequences,
            dtype=np.int64,
        )
        self.seq_gate_factor_sum = np.zeros(
            self.n_sequences,
            dtype=np.float64,
        )
        self.seq_candidate_delta_sum = np.zeros(
            self.n_sequences,
            dtype=np.float64,
        )

        self.event_counts = np.zeros(
            max(EVENT_NAMES) + 1,
            dtype=np.int64,
        )

        self.total = 0
        self.deadband_count = 0
        self.write_count = 0
        self.rail_count = 0
        self.tie_count = 0
        self.criterion_eligible_count = 0
        self.criterion_mismatch_count = 0

        self.gate_factor_sum = 0.0
        self.candidate_delta_sum = 0.0
        self.innovation_abs_sum = 0.0
        self.aux_abs_sum = 0.0
        self.aux_nonzero_count = 0

        self.deadband_gate_factor_sum = 0.0
        self.deadband_candidate_delta_sum = 0.0

        self.decomposition_max_abs_error = 0.0
        self.decomposition_sum_abs_error = 0.0
        self.decomposition_count = 0

    def update(
        self,
        start: int,
        end: int,
        h: np.ndarray,
        q: np.ndarray,
        z: np.ndarray,
        candidate: np.ndarray,
        aux: np.ndarray,
        event: np.ndarray,
    ) -> None:
        batch = end - start
        expected = (
            batch,
            self.seq_len,
            self.units,
        )

        arrays = {
            "h": np.asarray(h, dtype=np.float32),
            "q": np.asarray(q, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
            "candidate": np.asarray(candidate, dtype=np.float32),
            "aux": np.asarray(aux, dtype=np.float32),
        }

        for key, array in arrays.items():
            if array.shape != expected:
                raise RuntimeError(
                    f"{self.name} {key} shape {array.shape} != {expected}"
                )

            if not np.all(np.isfinite(array)):
                raise RuntimeError(
                    f"{self.name} {key} contains non-finite values"
                )

        event = np.asarray(
            event,
            dtype=np.int32,
        )

        if event.shape != expected:
            raise RuntimeError(
                f"{self.name} event shape {event.shape} != {expected}"
            )

        h_live = arrays["h"][:, :-1, :]
        q_cur = arrays["q"][:, :-1, :]
        q_next = arrays["q"][:, 1:, :]
        z_live = arrays["z"][:, :-1, :]
        candidate_live = arrays["candidate"][:, :-1, :]
        aux_live = arrays["aux"][:, :-1, :]
        event_live = event[:, :-1, :]

        innovation_signed = (
            h_live - q_cur
        ).astype(np.float64)

        innovation_abs = np.abs(
            innovation_signed
        )

        gate_factor = (
            1.0 - z_live
        ).astype(np.float64)

        candidate_signed = (
            candidate_live - q_cur
        ).astype(np.float64)

        candidate_delta = np.abs(
            candidate_signed
        )

        reconstructed = (
            gate_factor * candidate_signed
        )

        decomposition_error = np.abs(
            innovation_signed - reconstructed
        )

        self.decomposition_max_abs_error = max(
            self.decomposition_max_abs_error,
            float(
                decomposition_error.max(initial=0.0)
            ),
        )

        self.decomposition_sum_abs_error += float(
            decomposition_error.sum(dtype=np.float64)
        )
        self.decomposition_count += int(
            decomposition_error.size
        )

        margin = innovation_abs / self.half_step
        deadband = margin < 1.0

        write = np.abs(
            q_next - q_cur
        ) > (
            self.delta * 1e-4
        )

        rail = np.isclose(
            np.abs(q_cur),
            self.qmax,
            atol=self.delta * 1e-4,
            rtol=0.0,
        )

        tie = np.abs(
            margin - 1.0
        ) <= self.margin_tie_tolerance

        self.margin_counts += np.histogram(
            margin,
            bins=self.margin_edges,
        )[0].astype(np.int64)

        self.gate_counts += np.histogram(
            gate_factor,
            bins=self.gate_edges,
        )[0].astype(np.int64)

        self.candidate_delta_counts += np.histogram(
            candidate_delta,
            bins=self.candidate_delta_edges,
        )[0].astype(np.int64)

        for unit in range(self.units):
            self.unit_margin_counts[unit] += np.histogram(
                margin[:, :, unit],
                bins=self.margin_edges,
            )[0].astype(np.int64)

        self.unit_deadband_counts += deadband.sum(
            axis=(0, 1),
            dtype=np.int64,
        )

        self.unit_write_counts += write.sum(
            axis=(0, 1),
            dtype=np.int64,
        )

        self.unit_gate_factor_sum += gate_factor.sum(
            axis=(0, 1),
            dtype=np.float64,
        )

        self.unit_candidate_delta_sum += candidate_delta.sum(
            axis=(0, 1),
            dtype=np.float64,
        )

        self.unit_aux_abs_sum += np.abs(
            aux_live
        ).sum(
            axis=(0, 1),
            dtype=np.float64,
        )

        self.seq_deadband_counts[start:end] = deadband.sum(
            axis=(1, 2),
            dtype=np.int64,
        )

        self.seq_write_counts[start:end] = write.sum(
            axis=(1, 2),
            dtype=np.int64,
        )

        self.seq_gate_factor_sum[start:end] = gate_factor.sum(
            axis=(1, 2),
            dtype=np.float64,
        )

        self.seq_candidate_delta_sum[start:end] = candidate_delta.sum(
            axis=(1, 2),
            dtype=np.float64,
        )

        flat_events = event_live.reshape(-1)

        if (
            flat_events.size > 0
            and (
                int(flat_events.min()) < 0
                or int(flat_events.max()) > max(EVENT_NAMES)
            )
        ):
            raise RuntimeError(
                f"{self.name} observed invalid event code range "
                f"[{flat_events.min()}, {flat_events.max()}]"
            )

        self.event_counts += np.bincount(
            flat_events,
            minlength=len(self.event_counts),
        ).astype(np.int64)

        n_values = int(margin.size)

        self.total += n_values
        self.deadband_count += int(deadband.sum())
        self.write_count += int(write.sum())
        self.rail_count += int(rail.sum())
        self.tie_count += int(tie.sum())

        self.gate_factor_sum += float(
            gate_factor.sum(dtype=np.float64)
        )
        self.candidate_delta_sum += float(
            candidate_delta.sum(dtype=np.float64)
        )
        self.innovation_abs_sum += float(
            innovation_abs.sum(dtype=np.float64)
        )
        self.aux_abs_sum += float(
            np.abs(aux_live).sum(dtype=np.float64)
        )
        self.aux_nonzero_count += int(
            np.count_nonzero(
                np.abs(aux_live) > 1e-12
            )
        )

        if np.any(deadband):
            self.deadband_gate_factor_sum += float(
                gate_factor[deadband].sum(dtype=np.float64)
            )
            self.deadband_candidate_delta_sum += float(
                candidate_delta[deadband].sum(dtype=np.float64)
            )

        if self.operator.kind == "deterministic":
            eligible = (~rail) & (~tie)
            predicted_write = margin > 1.0
            mismatch = eligible & (
                predicted_write != write
            )

            self.criterion_eligible_count += int(
                eligible.sum()
            )
            self.criterion_mismatch_count += int(
                mismatch.sum()
            )

    def finalize(
        self,
        decomposition_tolerance: float,
        criterion_mismatch_fraction_limit: float,
    ) -> None:
        expected = (
            self.n_sequences
            * self.aligned_per_sequence
        )

        if self.total != expected:
            raise RuntimeError(
                f"{self.name} mechanism count {self.total} != {expected}"
            )

        if int(self.margin_counts.sum()) != expected:
            raise RuntimeError(
                f"{self.name} margin histogram count mismatch"
            )

        if self.decomposition_count != expected:
            raise RuntimeError(
                f"{self.name} decomposition count mismatch"
            )

        if self.decomposition_max_abs_error > decomposition_tolerance:
            raise RuntimeError(
                f"{self.name} GRU decomposition failed: "
                f"max_abs={self.decomposition_max_abs_error:.9g} > "
                f"{decomposition_tolerance:.9g}"
            )

        if self.operator.kind == "deterministic":
            if self.criterion_eligible_count <= 0:
                raise RuntimeError(
                    f"{self.name} has no interior non-tie deterministic transitions"
                )

            mismatch_fraction = (
                self.criterion_mismatch_count
                / self.criterion_eligible_count
            )

            if mismatch_fraction > criterion_mismatch_fraction_limit:
                raise RuntimeError(
                    f"{self.name} deterministic half-step criterion failed: "
                    f"mismatch_fraction={mismatch_fraction:.9g} > "
                    f"{criterion_mismatch_fraction_limit:.9g}"
                )

    def summary(self) -> Dict:
        total = float(self.total)
        deadband = float(self.deadband_count)
        per_sequence_den = float(
            self.aligned_per_sequence
        )

        event_summary = {
            EVENT_NAMES[index]: int(count)
            for index, count in enumerate(self.event_counts)
        }

        criterion_fraction = None
        if self.criterion_eligible_count > 0:
            criterion_fraction = (
                self.criterion_mismatch_count
                / self.criterion_eligible_count
            )

        return {
            "name": self.name,
            "state_bits": self.state_bits,
            "delta": self.delta,
            "half_step": self.half_step,
            "n_sequences": self.n_sequences,
            "seq_len": self.seq_len,
            "units": self.units,
            "aligned_reused_steps_per_sequence": self.aligned_steps,
            "aligned_state_element_transitions": self.total,
            "margin_definition": "abs(h_t-q_recv_t)/(Delta/2)",
            "innovation_identity": (
                "h_t-q_recv_t = (1-z_t)*(candidate_t-q_recv_t)"
            ),
            "deadband_fraction": (
                self.deadband_count / total
            ),
            "write_fraction": (
                self.write_count / total
            ),
            "deadband_and_write_fraction": (
                int(
                    self.event_counts[
                        EVENT_SCW_FORCED_WRITE
                    ]
                )
                / total
            ),
            "rail_fraction": (
                self.rail_count / total
            ),
            "tie_fraction": (
                self.tie_count / total
            ),
            "margin_median": histogram_quantile(
                self.margin_counts,
                self.margin_edges,
                0.50,
            ),
            "margin_p90": histogram_quantile(
                self.margin_counts,
                self.margin_edges,
                0.90,
            ),
            "margin_p99": histogram_quantile(
                self.margin_counts,
                self.margin_edges,
                0.99,
            ),
            "gate_factor_mean": (
                self.gate_factor_sum / total
            ),
            "candidate_displacement_mean": (
                self.candidate_delta_sum / total
            ),
            "innovation_abs_mean": (
                self.innovation_abs_sum / total
            ),
            "deadband_gate_factor_mean": (
                self.deadband_gate_factor_sum / deadband
                if self.deadband_count > 0
                else None
            ),
            "deadband_candidate_displacement_mean": (
                self.deadband_candidate_delta_sum / deadband
                if self.deadband_count > 0
                else None
            ),
            "aux_abs_mean": (
                self.aux_abs_sum / total
            ),
            "aux_nonzero_fraction": (
                self.aux_nonzero_count / total
            ),
            "event_counts": event_summary,
            "decomposition_max_abs_error": (
                self.decomposition_max_abs_error
            ),
            "decomposition_mean_abs_error": (
                self.decomposition_sum_abs_error
                / self.decomposition_count
            ),
            "deterministic_criterion_eligible_count": (
                self.criterion_eligible_count
            ),
            "deterministic_criterion_mismatch_count": (
                self.criterion_mismatch_count
            ),
            "deterministic_criterion_mismatch_fraction": (
                criterion_fraction
            ),
            "per_sequence_deadband_denominator": (
                per_sequence_den
            ),
        }

    def write_outputs(self, out_dir: Path) -> None:
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with (
            out_dir
            / f"{self.name}_margin_histogram.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "bin_left",
                    "bin_right",
                    "count",
                ]
            )

            for index, count in enumerate(
                self.margin_counts
            ):
                writer.writerow(
                    [
                        self.margin_edges[index],
                        self.margin_edges[index + 1],
                        int(count),
                    ]
                )

        with (
            out_dir
            / f"{self.name}_mechanism_histograms.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "variable",
                    "bin_left",
                    "bin_right",
                    "count",
                ]
            )

            for variable, edges, counts in (
                (
                    "gate_factor_1_minus_z",
                    self.gate_edges,
                    self.gate_counts,
                ),
                (
                    "candidate_displacement_abs_candidate_minus_q",
                    self.candidate_delta_edges,
                    self.candidate_delta_counts,
                ),
            ):
                for index, count in enumerate(counts):
                    writer.writerow(
                        [
                            variable,
                            edges[index],
                            edges[index + 1],
                            int(count),
                        ]
                    )

        per_unit_den = float(
            self.n_sequences
            * self.aligned_steps
        )

        with (
            out_dir
            / f"{self.name}_mechanism_per_unit.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "unit",
                    "deadband_fraction",
                    "write_fraction",
                    "margin_median",
                    "margin_p90",
                    "gate_factor_mean",
                    "candidate_displacement_mean",
                    "aux_abs_mean",
                ]
            )

            for unit in range(self.units):
                writer.writerow(
                    [
                        unit,
                        self.unit_deadband_counts[unit] / per_unit_den,
                        self.unit_write_counts[unit] / per_unit_den,
                        histogram_quantile(
                            self.unit_margin_counts[unit],
                            self.margin_edges,
                            0.50,
                        ),
                        histogram_quantile(
                            self.unit_margin_counts[unit],
                            self.margin_edges,
                            0.90,
                        ),
                        self.unit_gate_factor_sum[unit] / per_unit_den,
                        self.unit_candidate_delta_sum[unit] / per_unit_den,
                        self.unit_aux_abs_sum[unit] / per_unit_den,
                    ]
                )

        atomic_write_json(
            out_dir
            / f"{self.name}_mechanism_summary.json",
            self.summary(),
        )


class MechanismPass:
    def __init__(
        self,
        n_sequences: int,
        cfg: Dict,
        enc_operator: StatefulWritebackBase,
        dec_operator: StatefulWritebackBase,
        hist_bins: int,
        margin_tie_tolerance: float,
    ) -> None:
        self.encoder = MechanismAccumulator(
            "encoder",
            n_sequences,
            cfg["seq_len"],
            cfg["student_units"],
            enc_operator,
            hist_bins,
            margin_tie_tolerance,
        )

        self.decoder = MechanismAccumulator(
            "decoder",
            n_sequences,
            cfg["seq_len"],
            cfg["student_units"],
            dec_operator,
            hist_bins,
            margin_tie_tolerance,
        )


def run_recurrent_memory_pass(
    condition_name: str,
    forward,
    enc_operator: StatefulWritebackBase,
    dec_operator: StatefulWritebackBase,
    cfg: Dict,
    normalized_input: np.ndarray,
    res_data: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    infer_batch: int,
    hist_bins: int,
    gate_hist_bins: int,
    mechanism_hist_bins: int,
    margin_tie_tolerance: float,
    decomposition_tolerance: float,
    criterion_mismatch_fraction: float,
    gate_width_ns: float,
    sr_seed: int,
):
    n_sequences = len(test_idx)

    accumulator = PassAccumulator(
        n_sequences,
        cfg,
        enc_operator,
        dec_operator,
        hist_bins,
        gate_hist_bins,
    )

    mechanism = MechanismPass(
        n_sequences,
        cfg,
        enc_operator,
        dec_operator,
        mechanism_hist_bins,
        margin_tie_tolerance,
    )

    t_axis = (
        np.arange(
            cfg["seq_len"],
            dtype=np.float32,
        )
        * float(gate_width_ns)
    )

    n_batches = math.ceil(
        n_sequences / infer_batch
    )

    started = time.time()

    pf(
        f"[{condition_name}] start N={n_sequences} batches={n_batches} "
        f"method={enc_operator.kind} B{enc_operator.state_bits} seed={sr_seed}"
    )

    for batch_number, start in enumerate(
        range(
            0,
            n_sequences,
            infer_batch,
        ),
        start=1,
    ):
        end = min(
            start + infer_batch,
            n_sequences,
        )

        rows = test_idx[start:end]

        enc = np.asarray(
            normalized_input[rows],
            dtype=np.float32,
        )

        target = np.asarray(
            res_data[rows],
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
                start // infer_batch,
                tf.int32,
            ),
        )

        arrays = [
            np.asarray(
                tensor.numpy(),
                dtype=np.float32,
            )
            for tensor in outputs[:14]
        ]

        enc_event = np.asarray(
            outputs[14].numpy(),
            dtype=np.int32,
        )
        dec_event = np.asarray(
            outputs[15].numpy(),
            dtype=np.int32,
        )

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

        mechanism.encoder.update(
            start,
            end,
            arrays[1],
            arrays[2],
            arrays[3],
            arrays[5],
            arrays[12],
            enc_event,
        )

        mechanism.decoder.update(
            start,
            end,
            arrays[6],
            arrays[7],
            arrays[8],
            arrays[10],
            arrays[13],
            dec_event,
        )

        if (
            batch_number == 1
            or batch_number % 10 == 0
            or end == n_sequences
        ):
            elapsed = time.time() - started
            pf(
                f"[{condition_name}] batch {batch_number}/{n_batches} "
                f"samples {end}/{n_sequences} elapsed={elapsed / 60.0:.1f} min"
            )

    accumulator.finalize()

    mechanism.encoder.finalize(
        decomposition_tolerance,
        criterion_mismatch_fraction,
    )

    mechanism.decoder.finalize(
        decomposition_tolerance,
        criterion_mismatch_fraction,
    )

    labels_test = np.asarray(
        labels[test_idx],
        dtype=np.float32,
    )

    from analyze_writeback import compute_accuracy_metrics

    rmse1, r1, n1 = compute_accuracy_metrics(
        labels_test[:, 0],
        accumulator.tau1_pred,
    )

    rmse2, r2, n2 = compute_accuracy_metrics(
        labels_test[:, 1],
        accumulator.tau2_pred,
    )

    rmsef, rf, nf = compute_accuracy_metrics(
        labels_test[:, 2],
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
        "n_test": int(n_sequences),
        "sr_seed": int(sr_seed),
    }

    pf(
        f"[{condition_name}] mae_seq={metrics['mae_seq']:.9g} "
        f"rmse_tau1={rmse1:.6g} rmse_tau2={rmse2:.6g} "
        f"r_tau1={r1:.6g} r_tau2={r2:.6g}"
    )

    return (
        accumulator,
        mechanism,
        metrics,
        labels_test,
    )


def write_extended_per_sequence_npz(
    out_dir: Path,
    accumulator: PassAccumulator,
    mechanism: MechanismPass,
    labels_test: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    enc_den = float(
        mechanism.encoder.aligned_per_sequence
    )
    dec_den = float(
        mechanism.decoder.aligned_per_sequence
    )

    enc_gate_den = float(
        mechanism.encoder.aligned_per_sequence
    )
    dec_gate_den = float(
        mechanism.decoder.aligned_per_sequence
    )

    np.savez_compressed(
        str(
            out_dir
            / "recurrent_memory_per_sequence.npz"
        ),
        test_idx=np.asarray(
            test_idx,
            dtype=np.int64,
        ),
        gt_tau1=np.asarray(
            labels_test[:, 0],
            dtype=np.float32,
        ),
        gt_tau2=np.asarray(
            labels_test[:, 1],
            dtype=np.float32,
        ),
        gt_fret=np.asarray(
            labels_test[:, 2],
            dtype=np.float32,
        ),
        tau1_pred=np.asarray(
            accumulator.tau1_pred,
            dtype=np.float32,
        ),
        tau2_pred=np.asarray(
            accumulator.tau2_pred,
            dtype=np.float32,
        ),
        fret_pred=np.asarray(
            accumulator.fret_pred,
            dtype=np.float32,
        ),
        encoder_deadband_fraction_per_sequence=(
            mechanism.encoder.seq_deadband_counts.astype(np.float64)
            / enc_den
        ),
        decoder_deadband_fraction_per_sequence=(
            mechanism.decoder.seq_deadband_counts.astype(np.float64)
            / dec_den
        ),
        encoder_write_fraction_per_sequence=(
            mechanism.encoder.seq_write_counts.astype(np.float64)
            / enc_den
        ),
        decoder_write_fraction_per_sequence=(
            mechanism.decoder.seq_write_counts.astype(np.float64)
            / dec_den
        ),
        encoder_gate_factor_mean_per_sequence=(
            mechanism.encoder.seq_gate_factor_sum
            / enc_gate_den
        ),
        decoder_gate_factor_mean_per_sequence=(
            mechanism.decoder.seq_gate_factor_sum
            / dec_gate_den
        ),
        encoder_candidate_displacement_mean_per_sequence=(
            mechanism.encoder.seq_candidate_delta_sum
            / enc_gate_den
        ),
        decoder_candidate_displacement_mean_per_sequence=(
            mechanism.decoder.seq_candidate_delta_sum
            / dec_gate_den
        ),
    )


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    cfg: Dict,
    operator: StatefulWritebackBase,
    passes: List,
    adapter_equivalence: Dict,
    native_tensor_equivalence: Optional[Dict],
    test_idx: np.ndarray,
    elapsed_seconds: float,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = [
        row[3]
        for row in passes
    ]

    mechanism_realizations = [
        {
            "encoder": row[1].encoder.summary(),
            "decoder": row[1].decoder.summary(),
        }
        for row in passes
    ]

    payload = {
        "condition_name": args.condition_name,
        "phase": args.phase,
        "mode": "recurrent_memory_condition",
        "method": args.method,
        "operator": operator.metadata(),
        "n_realizations": len(summaries),
        "realizations": summaries,
        "cross_seed": cross_seed_summary(summaries),
        "mechanism_realizations": mechanism_realizations,
        "adapter_equivalence": adapter_equivalence,
        "native_tensor_equivalence": native_tensor_equivalence,
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "batch_reps": args.bootstrap_batch_reps,
            "level": "sequence",
            "note": (
                "Confidence intervals are held-out-set sampling uncertainty, "
                "not training-seed variability."
            ),
        },
    }

    atomic_write_json(
        out_dir
        / "recurrent_memory_summary.json",
        payload,
    )

    writeback_payload = {
        "phase": args.phase,
        "mode": "recurrent_memory_condition",
        "encoder_writeback": operator.metadata(),
        "decoder_writeback": operator.metadata(),
        "n_realizations": len(summaries),
        "realizations": summaries,
        "cross_seed": cross_seed_summary(summaries),
        "bootstrap": payload["bootstrap"],
    }

    atomic_write_json(
        out_dir
        / "writeback_summary.json",
        writeback_payload,
    )

    (
        first_acc,
        first_mechanism,
        _first_metrics,
        first_summary,
        labels_test,
    ) = passes[0]

    write_region_outputs(
        out_dir,
        first_acc.encoder,
        first_summary["encoder"],
    )

    write_region_outputs(
        out_dir,
        first_acc.decoder,
        first_summary["decoder"],
    )

    write_per_sequence_npz(
        out_dir,
        first_acc,
    )

    first_mechanism.encoder.write_outputs(
        out_dir
    )
    first_mechanism.decoder.write_outputs(
        out_dir
    )

    write_extended_per_sequence_npz(
        out_dir,
        first_acc,
        first_mechanism,
        labels_test,
        test_idx,
    )

    fidelity_path = Path(
        args.fidelity_json
    ).resolve()

    manifest = {
        "condition_name": args.condition_name,
        "phase": args.phase,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "student_args_sha256": sha256_file(
            run_dir / "student_args.json"
        ),
        "test_idx_sha256": sha256_array(
            np.asarray(test_idx, dtype=np.int64)
        ),
        "analysis_script": str(THIS_FILE),
        "analysis_script_sha256": sha256_file(THIS_FILE),
        "base_analysis_script": str(
            EVAL_DIR / "analyze_writeback.py"
        ),
        "base_analysis_script_sha256": sha256_file(
            EVAL_DIR / "analyze_writeback.py"
        ),
        "fidelity_json": str(fidelity_path),
        "fidelity_json_sha256": sha256_file(fidelity_path),
        "config": cfg,
        "operator": operator.metadata(),
        "adapter_equivalence": adapter_equivalence,
        "native_tensor_equivalence": native_tensor_equivalence,
        "cli": {
            key: json_safe(value)
            for key, value in sorted(vars(args).items())
        },
        "event_codes": {
            str(code): name
            for code, name in EVENT_NAMES.items()
        },
        "recurrent_write_definition": (
            "only h_t values consumed by a subsequent same-layer recurrent step "
            "enter write-margin and W/N_write statistics"
        ),
        "auxiliary_boundary_semantics": (
            "encoder and decoder auxiliary writeback memories are independent; "
            "the decoder auxiliary state is initialized from zero at handoff"
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "timestamp_unix": time.time(),
    }

    atomic_write_json(
        out_dir
        / "recurrent_memory_manifest.json",
        manifest,
    )

    with (
        out_dir
        / "recurrent_memory_complete.flag"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{args.condition_name}\n"
        )

    pf(
        f"[OUT] recurrent-memory results written under {out_dir}"
    )


def main() -> None:
    started = time.time()

    args = parse_args()
    validate_args(args)
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

    checkpoint_path = checkpoint_path_for(
        args,
        run_dir,
    )

    pf(
        f"[CKPT] {checkpoint_path}"
    )
    pf(
        f"[CKPT] SHA256 {sha256_file(checkpoint_path)}"
    )

    validate_fidelity_file(
        Path(args.fidelity_json).resolve(),
        checkpoint_path,
        run_dir,
        args.phase,
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

    raw_weights = extract_raw_weights(
        original_model,
        cfg,
        args.phase,
        enc_cell,
        dec_cell,
    )

    quantizers = build_parameter_quantizers(
        original_model,
        cfg,
        args.phase,
        enc_cell,
        dec_cell,
    )

    effective_weights = quantize_effective_weights(
        raw_weights,
        quantizers,
    )

    operator = make_operator(
        args
    )

    pf(
        "[OPERATOR] "
        + json.dumps(
            operator.metadata(),
            sort_keys=True,
        )
    )

    forward = build_stateful_forward_fn(
        effective_weights,
        quantizers,
        operator,
        operator,
        cfg,
    )

    control_kind = (
        args.method
        if args.method in (
            "identity",
            "deterministic",
            "stochastic",
            "error_feedback",
        )
        else "deterministic"
    )

    base_control_operator = WritebackOperator(
        control_kind,
        args.state_bits,
    )

    base_control_forward = build_forward_fn(
        effective_weights,
        quantizers,
        base_control_operator,
        base_control_operator,
        cfg,
    )

    custom_control_operator = ExistingWritebackAdapter(
        control_kind,
        args.state_bits,
    )

    custom_control_forward = build_stateful_forward_fn(
        effective_weights,
        quantizers,
        custom_control_operator,
        custom_control_operator,
        cfg,
    )

    adapter_equivalence = run_adapter_equivalence(
        base_control_forward,
        custom_control_forward,
        normalized_input,
        test_idx,
        args.equivalence_samples,
        args.adapter_tolerance,
        args.sr_seed_base,
    )

    native_tensor_equivalence = None

    native_method_matches = (
        (
            args.phase == "P2E"
            and args.method == "identity"
        )
        or (
            args.phase in (
                "P2F",
                "P3",
                "VANILLA",
            )
            and args.method == "deterministic"
            and args.state_bits == int(cfg["bits_state"])
        )
    )

    if native_method_matches:
        native_tensor_equivalence = run_tensor_equivalence(
            reference_model,
            forward,
            normalized_input,
            test_idx,
            cfg,
            effective_weights,
            args.equivalence_samples,
            args.native_tensor_tolerance,
            args.native_tensor_mean_tolerance,
            args.native_tensor_mismatch_fraction,
            args.native_tensor_tie_fraction,
        )

    seeds = (
        [
            args.sr_seed_base + index
            for index in range(args.sr_seeds)
        ]
        if args.method == "stochastic"
        else [args.sr_seed_base]
    )

    passes = []

    for realization_index, seed in enumerate(seeds):
        (
            accumulator,
            mechanism,
            metrics,
            labels_test,
        ) = run_recurrent_memory_pass(
            condition_name=(
                f"{args.condition_name}_r{realization_index}"
            ),
            forward=forward,
            enc_operator=operator,
            dec_operator=operator,
            cfg=cfg,
            normalized_input=normalized_input,
            res_data=res_data,
            labels=labels,
            test_idx=test_idx,
            infer_batch=args.infer_batch,
            hist_bins=args.hist_bins,
            gate_hist_bins=args.gate_hist_bins,
            mechanism_hist_bins=args.mechanism_hist_bins,
            margin_tie_tolerance=args.margin_tie_tolerance,
            decomposition_tolerance=args.decomposition_tolerance,
            criterion_mismatch_fraction=args.criterion_mismatch_fraction,
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
                mechanism,
                metrics,
                summary,
                labels_test,
            )
        )

    elapsed = time.time() - started

    write_outputs(
        out_dir,
        args,
        run_dir,
        checkpoint_path,
        cfg,
        operator,
        passes,
        adapter_equivalence,
        native_tensor_equivalence,
        test_idx,
        elapsed,
    )

    pf(
        f"[DONE] completed {args.condition_name} in {elapsed / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()