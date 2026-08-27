#!/usr/bin/env python3
"""Matched recurrent-memory training campaign for the vanilla KD Seq2SeqLite student.

Locked conditions:
  b4      : B4 recurrence-visible state, no auxiliary memory, 4 stored bits/unit.
  b6      : B6 recurrence-visible state, no auxiliary memory, 6 stored bits/unit.
  r2      : B4 state plus 2-bit quantized residual, 6 stored bits/unit.
  scw_k2  : B4 state plus 2-bit SCW counter, T=2, theta=Delta/8, 6 stored bits/unit.

The four conditions share the teacher, fixed train/validation/test split, fixed
training-order seed, loss, optimizer, LR schedule, batch configuration, stopping
rule and test evaluation. --init-seed changes model initialization only.

The implemented KD objective is regression MSE:
    alpha*MSE(student, teacher) + (1-alpha)*MSE(student, target)
The retained temperature argument is provenance-only. In the historical formula,
T^2 and the two 1/T factors cancel exactly for MSE.

For SCW and R2 the forward recurrent state is hard discrete. The visible state
uses an identity straight-through surrogate. Auxiliary counter/residual memory
is stop-gradient recurrent state. SCW has no direct gradient through counter
accumulation or trigger decisions. R2 retains a local amplitude-dependent
surrogate through the compensated visible-state write.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from qkeras import QDense, quantized_bits
from tensorflow import keras

import train_student_vanilla_kd_scw as common

CONDITIONS = ("b4", "b6", "r2", "scw_k2")
TRAINING_ALPHA = 0.6
TRAINING_TEMPERATURE = 4.0
SPLIT_SEED = 42
SCW_COUNTER_BITS = 2
SCW_DEADZONE_FRACTION = 0.125
R2_BITS = 2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one locked recurrent-memory training condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--init-seed", required=True, type=int)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--teacher-units", default=128, type=int)
    p.add_argument("--teacher-layers", default=2, type=int)
    p.add_argument("--student-units", default=32, type=int)
    p.add_argument("--seq-len", default=135, type=int)
    p.add_argument("--n-out", default=3, type=int)
    p.add_argument("--gate-width-ns", default=0.09, type=float)
    p.add_argument(
        "--temperature",
        default=TRAINING_TEMPERATURE,
        type=float,
        help="Provenance-only compatibility parameter; regression KD is direct MSE.",
    )
    p.add_argument(
        "--alpha",
        default=TRAINING_ALPHA,
        type=float,
        help="Teacher MSE weight in alpha*KD_MSE + (1-alpha)*task_MSE.",
    )
    p.add_argument("--batch-size", default=1024, type=int)
    p.add_argument("--accumulation-steps", default=1, type=int)
    p.add_argument("--epochs", default=300, type=int)
    p.add_argument("--patience", default=15, type=int)
    p.add_argument("--min-delta", default=1e-5, type=float)
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--ref-batch-size", default=1024, type=int)
    p.add_argument("--no-lr-scaling", action="store_true")
    p.add_argument("--lr-factor", default=0.5, type=float)
    p.add_argument("--lr-patience", default=8, type=int)
    p.add_argument("--lr-min", default=1e-6, type=float)
    p.add_argument("--warmup-epochs", default=5, type=int)
    p.add_argument("--infer-batch", default=4096, type=int)
    p.add_argument("--log-interval", default=10, type=int)
    p.add_argument("--prefetch-batches", default=32, type=int)
    p.add_argument("--pipeline-workers", default=4, type=int)
    p.add_argument("--split-seed", default=SPLIT_SEED, type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mixed-precision", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    teacher_ckpt = Path(args.teacher_ckpt).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Save directory does not exist: {save_dir}")
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")
    if args.seq_len <= 1 or args.n_out < 3:
        raise ValueError("Invalid sequence or output dimensions")
    if min(args.student_units, args.teacher_units, args.teacher_layers) <= 0:
        raise ValueError("Model dimensions must be positive")
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("Batch size and accumulation steps must be positive")
    if args.batch_size % args.accumulation_steps != 0:
        raise ValueError("--batch-size must be divisible by --accumulation-steps")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("Epochs and patience must be positive")
    if args.lr <= 0.0 or args.ref_batch_size <= 0:
        raise ValueError("Learning rate and reference batch size must be positive")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr-factor must be in (0,1)")
    if args.lr_patience <= 0 or args.lr_min <= 0.0 or args.warmup_epochs < 0:
        raise ValueError("Invalid LR scheduler configuration")
    if args.infer_batch <= 0:
        raise ValueError("--infer-batch must be positive")
    if args.split_seed != SPLIT_SEED:
        raise ValueError(f"Campaign fixes --split-seed at {SPLIT_SEED}")
    if args.alpha != TRAINING_ALPHA:
        raise ValueError(f"Campaign fixes --alpha at {TRAINING_ALPHA}")
    if args.temperature != TRAINING_TEMPERATURE:
        raise ValueError(f"Campaign fixes --temperature at {TRAINING_TEMPERATURE}")

    args.data_dir = str(data_dir)
    args.save_dir = str(save_dir)
    args.teacher_ckpt = str(teacher_ckpt)
    args.bits_kernel = 4
    args.bits_bias = 4
    args.bits_recurrent = 4
    args.bits_activation = 4
    args.bits_state = 6 if args.condition == "b6" else 4
    args.q_alpha = 1.0
    args.counter_bits = SCW_COUNTER_BITS if args.condition == "scw_k2" else -1
    args.scw_deadzone_fraction = (
        SCW_DEADZONE_FRACTION if args.condition == "scw_k2" else 0.0
    )
    args.residual_bits = R2_BITS if args.condition == "r2" else -1
    args.recurrence_visible_state_bits = args.bits_state
    args.auxiliary_memory_bits = 2 if args.condition in ("r2", "scw_k2") else 0
    args.total_stored_bits_per_unit = 4 if args.condition == "b4" else 6
    return args


def job_name_for(args: argparse.Namespace) -> str:
    return f"vanilla_memory_{args.condition}_seed{args.init_seed}"


class QuantizedR2GRUCore(common.QuantizedSCWGRUCore):
    def __init__(self, args: argparse.Namespace, name: str) -> None:
        super().__init__(
            units=args.student_units,
            bits_kernel=args.bits_kernel,
            bits_recurrent=args.bits_recurrent,
            bits_bias=args.bits_bias,
            bits_activation=args.bits_activation,
            bits_state=args.bits_state,
            counter_bits=2,
            deadzone_fraction=0.0,
            q_alpha=args.q_alpha,
            name=name,
        )
        self.residual_bits = int(args.residual_bits)
        if self.residual_bits != R2_BITS:
            raise ValueError(f"R2 requires residual_bits={R2_BITS}")
        self.residual_levels = 2 ** self.residual_bits
        self.residual_step = self.delta / self.residual_levels
        self.residual_min = -self.half_step
        self.residual_max = self.half_step - self.residual_step

    def quantize_residual(self, value: tf.Tensor) -> tf.Tensor:
        value = tf.cast(value, tf.float32)
        rmin = tf.constant(self.residual_min, tf.float32)
        rmax = tf.constant(self.residual_max, tf.float32)
        step = tf.constant(self.residual_step, tf.float32)
        clipped = tf.clip_by_value(value, rmin, rmax)
        code = tf.round((clipped - rmin) / step)
        code = tf.clip_by_value(code, 0.0, float(self.residual_levels - 1))
        return tf.cast(rmin + code * step, tf.float32)

    def initialize_state(
        self, raw_state: tf.Tensor, operator_mode: str, use_ste: bool
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        raw_state = tf.cast(raw_state, tf.float32)
        if operator_mode == "identity":
            q_hard = raw_state
            residual_hard = tf.zeros_like(raw_state)
            state = raw_state
        elif operator_mode == "deterministic":
            q_hard = self.deterministic_quantize_state(raw_state)
            residual_hard = tf.zeros_like(raw_state)
            state = self.identity_ste(raw_state, q_hard) if use_ste else q_hard
        elif operator_mode == "r2":
            q_hard = self.deterministic_quantize_state(raw_state)
            residual_hard = self.quantize_residual(raw_state - tf.stop_gradient(q_hard))
            state = self.identity_ste(raw_state, q_hard) if use_ste else q_hard
        else:
            raise ValueError(f"Unsupported operator_mode={operator_mode!r}")
        return state, tf.stop_gradient(residual_hard), tf.stop_gradient(q_hard)

    def hard_advance(
        self,
        raw_state: tf.Tensor,
        q_prev_hard: tf.Tensor,
        residual_prev: tf.Tensor,
        operator_mode: str,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        del q_prev_hard
        raw_state = tf.cast(raw_state, tf.float32)
        residual_prev = tf.stop_gradient(tf.cast(residual_prev, tf.float32))
        if operator_mode == "identity":
            return tf.stop_gradient(raw_state), tf.zeros_like(raw_state)
        if operator_mode == "deterministic":
            q_hard = self.deterministic_quantize_state(raw_state)
            return tf.stop_gradient(q_hard), tf.zeros_like(raw_state)
        if operator_mode != "r2":
            raise ValueError(f"Unsupported operator_mode={operator_mode!r}")
        compensated = raw_state + residual_prev
        q_hard = self.deterministic_quantize_state(compensated)
        residual_hard = self.quantize_residual(compensated - tf.stop_gradient(q_hard))
        return tf.stop_gradient(q_hard), tf.stop_gradient(residual_hard)

    def advance_state(
        self,
        raw_state: tf.Tensor,
        q_prev_hard: tf.Tensor,
        residual_prev: tf.Tensor,
        operator_mode: str,
        use_ste: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        del q_prev_hard
        raw_state = tf.cast(raw_state, tf.float32)
        residual_prev = tf.stop_gradient(tf.cast(residual_prev, tf.float32))
        if operator_mode == "identity":
            q_hard = raw_state
            residual_hard = tf.zeros_like(raw_state)
            state = raw_state
        elif operator_mode == "deterministic":
            q_hard = self.deterministic_quantize_state(raw_state)
            residual_hard = tf.zeros_like(raw_state)
            state = self.identity_ste(raw_state, q_hard) if use_ste else q_hard
        elif operator_mode == "r2":
            compensated = raw_state + residual_prev
            q_hard = self.deterministic_quantize_state(compensated)
            residual_hard = self.quantize_residual(compensated - tf.stop_gradient(q_hard))
            state = self.identity_ste(compensated, q_hard) if use_ste else q_hard
        else:
            raise ValueError(f"Unsupported operator_mode={operator_mode!r}")
        return state, tf.stop_gradient(residual_hard), tf.stop_gradient(q_hard)

    def metadata(self) -> Dict:
        return {
            "kind": "quantized_residual_r2",
            "state_bits": self.bits_state,
            "delta": self.delta,
            "half_step": self.half_step,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "residual_bits": self.residual_bits,
            "residual_levels": self.residual_levels,
            "residual_step": self.residual_step,
            "residual_min": self.residual_min,
            "residual_max": self.residual_max,
            "residual_codebook": [
                float(self.residual_min + i * self.residual_step)
                for i in range(self.residual_levels)
            ],
            "recurrence_visible_state_bits": self.bits_state,
            "auxiliary_memory_bits": self.residual_bits,
            "total_stored_bits_per_unit": self.bits_state + self.residual_bits,
            "forward_operator": "hard B4 state plus hard R2 quantized residual",
            "backward_state_surrogate": "identity STE through compensated state",
            "auxiliary_gradient": "stop_gradient",
        }


class R2StudentModel(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(name="student_vanilla_kd_r2")
        self.seq_len = args.seq_len
        self.n_out = args.n_out
        self.student_units = args.student_units
        self.sencgru = QuantizedR2GRUCore(args, "sencgru")
        self.sdecgru = QuantizedR2GRUCore(args, "sdecgru")
        self.sdec_dense = QDense(
            args.n_out,
            kernel_quantizer=quantized_bits(args.bits_kernel, 0),
            bias_quantizer=quantized_bits(args.bits_kernel, 0),
            activation="linear",
            name="sdec_dense",
        )

    def call(self, inputs, training=False, operator_mode: str = "r2") -> tf.Tensor:
        if operator_mode not in ("r2", "deterministic", "identity"):
            raise ValueError(f"Unsupported operator_mode={operator_mode!r}")
        enc_inputs, dec_inputs = inputs
        enc_inputs = tf.cast(enc_inputs, tf.float32)
        dec_inputs = tf.cast(dec_inputs, tf.float32)
        tf.debugging.assert_equal(tf.shape(enc_inputs)[1], self.seq_len)
        tf.debugging.assert_equal(tf.shape(dec_inputs)[1], self.seq_len)
        use_ste = bool(training)
        batch = tf.shape(enc_inputs)[0]
        seq_len_tensor = tf.constant(self.seq_len, tf.int32)
        final_live_index = tf.constant(self.seq_len - 1, tf.int32)
        enc_k, enc_rk, enc_b = self.sencgru.effective_parameters()
        dec_k, dec_rk, dec_b = self.sdecgru.effective_parameters()
        zero_raw = tf.zeros((batch, self.student_units), tf.float32)
        q_enc, aux_enc, q_enc_hard = self.sencgru.initialize_state(
            zero_raw, operator_mode, use_ste
        )

        def enc_cond(i, raw, q, aux, q_hard):
            del raw, q, aux, q_hard
            return i < seq_len_tensor

        def enc_body(i, raw, q, aux, q_hard):
            del raw
            raw_next, _, _, _ = self.sencgru.gru_step(
                enc_inputs[:, i, :], q, enc_k, enc_rk, enc_b
            )

            def advance():
                return self.sencgru.advance_state(
                    raw_next, q_hard, aux, operator_mode, use_ste
                )

            q_next, aux_next, q_hard_next = tf.cond(
                i < final_live_index, advance, lambda: (q, aux, q_hard)
            )
            return i + 1, raw_next, q_next, aux_next, q_hard_next

        _, raw_enc, _, _, _ = tf.while_loop(
            enc_cond,
            enc_body,
            (tf.constant(0, tf.int32), zero_raw, q_enc, aux_enc, q_enc_hard),
            parallel_iterations=1,
            maximum_iterations=self.seq_len,
        )

        q_dec, aux_dec, q_dec_hard = self.sdecgru.initialize_state(
            raw_enc, operator_mode, use_ste
        )
        ta = tf.TensorArray(
            tf.float32,
            size=self.seq_len,
            clear_after_read=False,
            element_shape=tf.TensorShape([None, self.student_units]),
        )

        def dec_cond(i, raw, q, aux, q_hard, hidden_ta):
            del raw, q, aux, q_hard, hidden_ta
            return i < seq_len_tensor

        def dec_body(i, raw, q, aux, q_hard, hidden_ta):
            del raw
            raw_next, _, _, _ = self.sdecgru.gru_step(
                dec_inputs[:, i, :], q, dec_k, dec_rk, dec_b
            )
            hidden_ta = hidden_ta.write(i, raw_next)

            def advance():
                return self.sdecgru.advance_state(
                    raw_next, q_hard, aux, operator_mode, use_ste
                )

            q_next, aux_next, q_hard_next = tf.cond(
                i < final_live_index, advance, lambda: (q, aux, q_hard)
            )
            return i + 1, raw_next, q_next, aux_next, q_hard_next, hidden_ta

        _, _, _, _, _, ta = tf.while_loop(
            dec_cond,
            dec_body,
            (tf.constant(0, tf.int32), raw_enc, q_dec, aux_dec, q_dec_hard, ta),
            parallel_iterations=1,
            maximum_iterations=self.seq_len,
        )
        dec_hidden = tf.transpose(ta.stack(), (1, 0, 2))
        return self.sdec_dense(dec_hidden)

    def export_raw_weights(self) -> Dict[str, np.ndarray]:
        return {
            "enc_kernel": np.asarray(self.sencgru.kernel.numpy(), np.float32),
            "enc_recurrent": np.asarray(self.sencgru.recurrent_kernel.numpy(), np.float32),
            "enc_bias": np.asarray(self.sencgru.bias.numpy(), np.float32),
            "dec_kernel": np.asarray(self.sdecgru.kernel.numpy(), np.float32),
            "dec_recurrent": np.asarray(self.sdecgru.recurrent_kernel.numpy(), np.float32),
            "dec_bias": np.asarray(self.sdecgru.bias.numpy(), np.float32),
            "dense_kernel": np.asarray(self.sdec_dense.kernel.numpy(), np.float32),
            "dense_bias": np.asarray(self.sdec_dense.bias.numpy(), np.float32),
        }


def build_custom_children(model, args: argparse.Namespace) -> None:
    if not model.sencgru.built:
        model.sencgru.build(tf.TensorShape([None, 1]))
    if not model.sdecgru.built:
        model.sdecgru.build(tf.TensorShape([None, 1]))
    if not model.sdec_dense.built:
        model.sdec_dense.build(tf.TensorShape([None, args.seq_len, args.student_units]))
    for name, layer in (("sencgru", model.sencgru), ("sdecgru", model.sdecgru)):
        for attribute in ("kernel", "recurrent_kernel", "bias"):
            if not hasattr(layer, attribute):
                raise RuntimeError(f"{name} did not create {attribute}")


def export_model_raw_weights(model, condition: str) -> Dict[str, np.ndarray]:
    if condition in ("r2", "scw_k2"):
        return model.export_raw_weights()

    result: Dict[str, np.ndarray] = {}
    for prefix, layer_name in (
        ("enc", "sencgru"),
        ("dec", "sdecgru"),
    ):
        weights = model.get_layer(layer_name).get_weights()
        if len(weights) != 3:
            raise RuntimeError(
                f"Expected three QGRU weight arrays for {layer_name}, got {len(weights)}"
            )
        result[f"{prefix}_kernel"] = np.asarray(weights[0], np.float32)
        result[f"{prefix}_recurrent"] = np.asarray(weights[1], np.float32)
        result[f"{prefix}_bias"] = np.asarray(weights[2], np.float32)

    dense_weights = model.get_layer("sdec_dense").get_weights()
    if len(dense_weights) != 2:
        raise RuntimeError(
            f"Expected two QDense weight arrays, got {len(dense_weights)}"
        )
    result["dense_kernel"] = np.asarray(dense_weights[0], np.float32)
    result["dense_bias"] = np.asarray(dense_weights[1], np.float32)
    return result


def save_or_verify_initial_weights(
    model, condition: str, job_dir: Path, resume: bool
) -> Tuple[Path, str]:
    path = job_dir / "initial_weights.npz"
    current = export_model_raw_weights(model, condition)
    if resume:
        if not path.is_file():
            raise RuntimeError(f"Resume initial-weight audit file is missing: {path}")
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(current):
                raise RuntimeError(
                    f"Initial-weight audit keys differ: stored={stored.files}, "
                    f"current={sorted(current)}"
                )
            for key, value in current.items():
                stored_value = np.asarray(stored[key], np.float32)
                if not np.array_equal(stored_value, value):
                    max_abs = float(np.max(np.abs(stored_value - value)))
                    raise RuntimeError(
                        f"Initial-weight audit mismatch for {key}: max_abs={max_abs:.9g}"
                    )
    else:
        if path.exists():
            raise RuntimeError(f"Fresh run initial-weight file already exists: {path}")
        np.savez_compressed(path, **current)
    return path, sha256_file(path)


def transfer_reference_weights(reference_model, custom_model) -> None:
    for name, target in (
        ("sencgru", custom_model.sencgru),
        ("sdecgru", custom_model.sdecgru),
        ("sdec_dense", custom_model.sdec_dense),
    ):
        source = reference_model.get_layer(name)
        source_weights = source.get_weights()
        target_weights = target.get_weights()
        if len(source_weights) != len(target_weights):
            raise RuntimeError(f"Weight-count mismatch for {name}")
        for index, (src, dst) in enumerate(zip(source_weights, target_weights)):
            if src.shape != dst.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name} weight {index}: {src.shape} vs {dst.shape}"
                )
        target.set_weights(source_weights)


def deterministic_equivalence(
    reference_model,
    custom_model,
    normalized_input,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    job_dir: Path,
) -> Dict:
    n = min(512, len(val_idx))
    if n <= 0:
        raise RuntimeError("No validation samples for deterministic equivalence")
    rows = val_idx[:n]
    enc = tf.constant(np.asarray(normalized_input[rows], np.float32))
    dec = tf.zeros((n, args.seq_len, 1), tf.float32)
    ref = np.asarray(reference_model([enc, dec], training=False).numpy(), np.float32)
    custom = np.asarray(
        custom_model([enc, dec], training=False, operator_mode="deterministic").numpy(),
        np.float32,
    )
    if ref.shape != custom.shape:
        raise RuntimeError(f"Equivalence shape mismatch: {ref.shape} vs {custom.shape}")
    diff = np.abs(ref - custom).astype(np.float64)
    payload = {
        "passed": bool(np.max(diff) <= 5e-5 and np.mean(diff) <= 5e-5),
        "n_samples": n,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "tolerance": 5e-5,
        "reference": "native QKeras deterministic QGRU",
        "custom": type(custom_model).__name__,
    }
    atomic_write_json(job_dir / "initialization_equivalence.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"Deterministic equivalence failed: {payload}")
    common.pf(
        f"[EQUIV] deterministic QGRU passed N={n} "
        f"max_abs={payload['max_abs']:.3e} mean_abs={payload['mean_abs']:.3e}"
    )
    return payload


def model_forward(
    model,
    condition: str,
    enc: tf.Tensor,
    dec: tf.Tensor,
    training: bool,
    operator_override: Optional[str] = None,
) -> tf.Tensor:
    if condition in ("b4", "b6"):
        if operator_override not in (None, "native"):
            raise ValueError(f"No operator override for native {condition}")
        return model([enc, dec], training=training)
    if condition == "scw_k2":
        return model(
            [enc, dec], training=training, operator_mode=operator_override or "scw"
        )
    if condition == "r2":
        return model(
            [enc, dec], training=training, operator_mode=operator_override or "r2"
        )
    raise ValueError(condition)


def make_distributed_train_step(strategy, model, optimizer, args):
    @tf.function
    def distributed_step(batch_x, batch_y):
        def replica_step(local_x, local_y):
            enc = local_x["enc_input"]
            dec = local_x["dec_input"]
            teacher = local_x["tpred"]
            target = local_y
            T = tf.cast(args.temperature, tf.float32)
            alpha = tf.cast(args.alpha, tf.float32)
            with tf.GradientTape() as tape:
                output = model_forward(model, args.condition, enc, dec, True)
                hard = tf.reduce_mean(tf.square(output - target))
                soft = T * T * tf.reduce_mean(tf.square(teacher / T - output / T))
                total = alpha * soft + (1.0 - alpha) * hard
            grads = tape.gradient(total, model.trainable_variables)
            grads = [
                tf.zeros_like(var) if grad is None else grad
                for grad, var in zip(grads, model.trainable_variables)
            ]
            bad = tf.reduce_any(
                tf.stack([tf.reduce_any(~tf.math.is_finite(grad)) for grad in grads])
            )
            grads = [tf.where(bad, tf.zeros_like(grad), grad) for grad in grads]
            grads, norm = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total, hard, soft, tf.cast(bad, tf.float32), norm

        values = strategy.run(replica_step, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(op, values[i], axis=None)
            for i, op in enumerate(
                (
                    tf.distribute.ReduceOp.MEAN,
                    tf.distribute.ReduceOp.MEAN,
                    tf.distribute.ReduceOp.MEAN,
                    tf.distribute.ReduceOp.SUM,
                    tf.distribute.ReduceOp.MEAN,
                )
            )
        )

    return distributed_step


def make_distributed_val_step(strategy, model, args):
    @tf.function
    def distributed_step(batch_x, batch_y):
        def replica_step(local_x, local_y):
            enc = local_x["enc_input"]
            dec = local_x["dec_input"]
            teacher = local_x["tpred"]
            target = local_y
            T = tf.cast(args.temperature, tf.float32)
            alpha = tf.cast(args.alpha, tf.float32)
            output = model_forward(model, args.condition, enc, dec, False)
            hard = tf.reduce_mean(tf.square(output - target))
            soft = T * T * tf.reduce_mean(tf.square(teacher / T - output / T))
            total = alpha * soft + (1.0 - alpha) * hard
            mae = tf.reduce_mean(tf.abs(output - target))
            return total, hard, soft, mae

        values = strategy.run(replica_step, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
            for value in values
        )

    return distributed_step


def make_campaign_kd_dataset(
    enc_arr: np.ndarray,
    tgt_arr: np.ndarray,
    tpred_arr: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
    seed: int,
):
    """Build one deterministic campaign dataset without materializing decoder zeros.

    Training datasets are rebuilt once per epoch with seed=split_seed+epoch and
    reshuffle_each_iteration=False. This preserves the same epoch-specific sample
    order after a SLURM resume. Validation order is fixed and unshuffled.
    """
    micro_batch_size = args.batch_size // args.accumulation_steps
    ds_enc = tf.data.Dataset.from_tensor_slices(enc_arr)
    ds_tpred = tf.data.Dataset.from_tensor_slices(tpred_arr)
    ds_tgt = tf.data.Dataset.from_tensor_slices(tgt_arr)
    ds = tf.data.Dataset.zip((ds_enc, ds_tpred, ds_tgt))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(enc_arr), 200_000),
            seed=int(seed),
            reshuffle_each_iteration=False,
        )
    ds = ds.batch(micro_batch_size, drop_remainder=True)

    def set_shapes(enc_b, tpred_b, tgt_b):
        enc_b.set_shape([micro_batch_size, args.seq_len, 1])
        tpred_b.set_shape([micro_batch_size, args.seq_len, args.n_out])
        tgt_b.set_shape([micro_batch_size, args.seq_len, args.n_out])
        return {
            "enc_input": enc_b,
            "dec_input": tf.zeros_like(enc_b),
            "tpred": tpred_b,
        }, tgt_b

    ds = ds.map(set_shapes, num_parallel_calls=args.pipeline_workers)
    ds = ds.prefetch(args.prefetch_batches)
    return ds


def training_loop(
    strategy,
    model,
    optimizer,
    scheduler,
    enc_train: np.ndarray,
    tgt_train: np.ndarray,
    tpred_train: np.ndarray,
    enc_val: np.ndarray,
    tgt_val: np.ndarray,
    tpred_val: np.ndarray,
    train_steps: int,
    args,
    job_dir: Path,
) -> Tuple[Dict[str, List[float]], float, bool]:
    best_ckpt = job_dir / "student_best.weights.h5"
    resume_path = job_dir / "resume_state.json"
    training_flag = job_dir / "training_complete.flag"
    history_path = job_dir / "training_history.csv"
    history = {
        key: []
        for key in (
            "total",
            "hard",
            "soft",
            "val_total",
            "val_hard",
            "val_soft",
            "val_mae",
            "grad_norm",
        )
    }
    best_val = float("inf")
    patience_count = 0
    start_epoch = 0

    if args.resume:
        if training_flag.is_file():
            if not best_ckpt.is_file():
                raise RuntimeError(f"Missing best checkpoint: {best_ckpt}")
            model.load_weights(str(best_ckpt))
            return history, best_val, True
        if not resume_path.is_file() or not best_ckpt.is_file():
            raise RuntimeError("Resume requires resume_state.json and student_best.weights.h5")
        with resume_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        start_epoch = int(state["epoch"])
        best_val = float(state["best_val"])
        patience_count = int(state["patience_count"])
        for key, values in state.get("history", {}).items():
            if key in history:
                history[key] = [float(value) for value in values]
        if not 0 < start_epoch <= args.epochs:
            raise RuntimeError(f"Invalid resume epoch: {start_epoch}")
        common.restore_exact_resume_checkpoint(
            strategy, model, optimizer, scheduler, job_dir, start_epoch
        )
    else:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "total",
                    "hard",
                    "soft",
                    "val_total",
                    "val_hard",
                    "val_soft",
                    "val_mae",
                    "grad_norm",
                    "lr",
                    "bad_grad_batches",
                ]
            )

    train_step = make_distributed_train_step(strategy, model, optimizer, args)
    val_step = make_distributed_val_step(strategy, model, args)
    val_dataset = make_campaign_kd_dataset(
        enc_val, tgt_val, tpred_val, args, shuffle=False, seed=args.split_seed
    )
    dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    interrupted = False

    for epoch in range(start_epoch, args.epochs):
        epoch_seed = int(args.split_seed + epoch)
        train_dataset = make_campaign_kd_dataset(
            enc_train, tgt_train, tpred_train, args, shuffle=True, seed=epoch_seed
        )
        dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        started = time.time()
        sums = np.zeros(4, dtype=np.float64)
        count = 0
        bad_count = 0
        common.pf(f"[EPOCH {epoch + 1}/{args.epochs}] train lr={scheduler.current_lr:.3e}")
        common.pf(f"[EPOCH {epoch + 1}] deterministic shuffle seed={epoch_seed}")
        for step_index, (bx, by) in enumerate(dist_train_dataset):
            total, hard, soft, bad, norm = train_step(bx, by)
            sums += (float(total), float(hard), float(soft), float(norm))
            count += 1
            bad_count += int(float(bad) > 0.0)
            if (step_index + 1) % args.log_interval == 0 or step_index + 1 == train_steps:
                elapsed = time.time() - started
                remaining = elapsed / max(step_index + 1, 1) * max(train_steps - step_index - 1, 0)
                common.pf(
                    f"  step {step_index + 1}/{train_steps} total={sums[0]/count:.6f} "
                    f"hard={sums[1]/count:.6f} soft={sums[2]/count:.6f} "
                    f"grad_norm={sums[3]/count:.5f} bad={bad_count} eta={remaining/60:.1f}m"
                )
        if count == 0:
            raise RuntimeError("Training dataset yielded zero batches")
        train_total, train_hard, train_soft, grad_norm = sums / count

        val_sums = np.zeros(4, dtype=np.float64)
        val_count = 0
        for bx, by in dist_val_dataset:
            values = val_step(bx, by)
            val_sums += [float(value) for value in values]
            val_count += 1
        if val_count == 0:
            raise RuntimeError("Validation dataset yielded zero batches")
        val_total, val_hard, val_soft, val_mae = val_sums / val_count

        for key, value in (
            ("total", train_total),
            ("hard", train_hard),
            ("soft", train_soft),
            ("val_total", val_total),
            ("val_hard", val_hard),
            ("val_soft", val_soft),
            ("val_mae", val_mae),
            ("grad_norm", grad_norm),
        ):
            history[key].append(float(value))

        if args.effective_warmup_epochs > 0 and epoch < args.effective_warmup_epochs:
            scheduler.lr_var.assign(
                args.effective_lr * (epoch + 1) / args.effective_warmup_epochs
            )
        else:
            scheduler.step(float(val_total), epoch)

        if val_total < best_val - args.min_delta:
            best_val = float(val_total)
            patience_count = 0
            model.save_weights(str(best_ckpt))
        else:
            patience_count += 1

        completed_epoch = epoch + 1
        exact_checkpoint = common.save_exact_resume_checkpoint(
            strategy, model, optimizer, scheduler, job_dir, completed_epoch
        )
        atomic_write_json(
            resume_path,
            {
                "epoch": completed_epoch,
                "best_val": best_val,
                "patience_count": patience_count,
                "lr": scheduler.current_lr,
                "exact_checkpoint": exact_checkpoint,
                "condition": args.condition,
                "init_seed": args.init_seed,
                "epoch_shuffle_seed": epoch_seed,
                "epoch_shuffle_policy": "split_seed_plus_zero_based_epoch",
                "history": history,
            },
        )
        with history_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    completed_epoch,
                    train_total,
                    train_hard,
                    train_soft,
                    val_total,
                    val_hard,
                    val_soft,
                    val_mae,
                    grad_norm,
                    scheduler.current_lr,
                    bad_count,
                ]
            )
        common.pf(
            f"[EPOCH {completed_epoch}] train={train_total:.6f} val={val_total:.6f} "
            f"val_mae={val_mae:.6f} best={best_val:.6f} "
            f"patience={patience_count}/{args.patience} time={(time.time()-started)/60:.1f}m"
        )
        if common.STOP_AFTER_EPOCH:
            interrupted = True
            common.pf("[SIGNAL] Exact epoch checkpoint saved; exiting for gated resume.")
            break
        if patience_count >= args.patience:
            common.pf(f"[EARLY STOP] epoch={completed_epoch}")
            break

    if interrupted:
        return history, best_val, False
    if not best_ckpt.is_file():
        raise RuntimeError(f"No best checkpoint was created: {best_ckpt}")
    model.load_weights(str(best_ckpt))
    training_flag.write_text("passed\n", encoding="utf-8")
    return history, best_val, True


def save_training_plot(history: Dict[str, List[float]], job_dir: Path, args) -> None:
    if not history["total"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["total"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(epochs, history["total"], label="train total")
    ax.plot(epochs, history["val_total"], label="validation total")
    ax.plot(epochs, history["val_mae"], label="validation MAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MAE")
    ax.set_title(f"{args.condition}, init seed {args.init_seed}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(job_dir / "training_history.png", dpi=180)
    plt.close(fig)


def run_inference(model, normalized_input, test_idx, args, operator=None) -> np.ndarray:
    n = len(test_idx)
    result = np.empty((n, args.seq_len, args.n_out), np.float32)
    for start in range(0, n, args.infer_batch):
        end = min(start + args.infer_batch, n)
        rows = test_idx[start:end]
        enc = tf.constant(np.asarray(normalized_input[rows], np.float32))
        dec = tf.zeros((end - start, args.seq_len, 1), tf.float32)
        result[start:end] = model_forward(
            model, args.condition, enc, dec, False, operator
        ).numpy()
        common.pf(f"[EVAL] {operator or 'native'} {end:,}/{n:,}")
    return result


def evaluate_predictions(pred, res, labels, test_idx, args) -> Tuple[Dict, Dict]:
    t = np.arange(args.seq_len, dtype=np.float32) * args.gate_width_ns
    tau1, tau2, fret = common.lifetime_from_prediction(pred, t)
    gt1 = np.asarray(labels[test_idx, 0], np.float32)
    gt2 = np.asarray(labels[test_idx, 1], np.float32)
    gtf = np.asarray(labels[test_idx, 2], np.float32)
    seq_mae = np.mean(np.abs(pred - np.asarray(res[test_idx], np.float32)), axis=(1, 2)).astype(np.float32)
    metrics = {
        "n_test": len(test_idx),
        "mae_seq": float(np.mean(seq_mae)),
        "tau1": common.metric_summary(gt1, tau1),
        "tau2": common.metric_summary(gt2, tau2),
        "fret": common.metric_summary(gtf, fret),
    }
    arrays = {
        "test_idx": np.asarray(test_idx, np.int64),
        "gt_tau1": gt1,
        "gt_tau2": gt2,
        "gt_fret": gtf,
        "tau1_pred": tau1,
        "tau2_pred": tau2,
        "fret_pred": fret,
        "seq_mae_per_sequence": seq_mae,
    }
    return metrics, arrays


def evaluate_condition(model, normalized_input, res, labels, test_idx, args, job_dir):
    if args.condition in ("b4", "b6"):
        modes: Sequence[Tuple[str, Optional[str]]] = (("native", None),)
        native_key = "native"
    elif args.condition == "r2":
        modes = (("r2", "r2"), ("deterministic", "deterministic"), ("identity", "identity"))
        native_key = "r2"
    else:
        modes = (("scw", "scw"), ("deterministic", "deterministic"), ("identity", "identity"))
        native_key = "scw"

    comparison = {}
    native_arrays = None
    for label, operator in modes:
        pred = run_inference(model, normalized_input, test_idx, args, operator)
        metrics, arrays = evaluate_predictions(pred, res, labels, test_idx, args)
        comparison[label] = metrics
        common.pf(
            f"[RESULT] {label} MAE={metrics['mae_seq']:.6f} "
            f"tau1={metrics['tau1']['rmse']:.6f} tau2={metrics['tau2']['rmse']:.6f}"
        )
        if label == native_key:
            native_arrays = arrays
    if native_arrays is None:
        raise RuntimeError("Native arrays were not captured")
    np.savez_compressed(job_dir / "test_per_sequence.npz", **native_arrays)
    native = comparison[native_key]
    test_metrics = {
        "job_name": job_dir.name,
        "condition": args.condition,
        "init_seed": args.init_seed,
        "n_test": native["n_test"],
        "mae_seq": native["mae_seq"],
        "tau1": native["tau1"],
        "tau2": native["tau2"],
        "fret": native["fret"],
        "native_operator": native_key,
        "recurrence_visible_state_bits": args.recurrence_visible_state_bits,
        "auxiliary_memory_bits": args.auxiliary_memory_bits,
        "total_stored_bits_per_unit": args.total_stored_bits_per_unit,
    }
    atomic_write_json(job_dir / "test_metrics.json", test_metrics)
    atomic_write_json(
        job_dir / "operator_comparison.json",
        {"condition": args.condition, "init_seed": args.init_seed, "native_operator": native_key, "operators": comparison},
    )
    return {"native": test_metrics, "operator_comparison": comparison}


def build_model(strategy, normalized_input, val_idx, args, job_dir):
    equivalence = None
    with strategy.scope():
        tf.keras.utils.set_random_seed(args.init_seed)
        if args.condition in ("b4", "b6"):
            model = common.build_student(
                args.seq_len,
                args.n_out,
                args.student_units,
                args.bits_kernel,
                args.bits_recurrent,
                args.bits_bias,
                args.bits_activation,
                args.bits_state,
            )
        else:
            reference = common.build_student(
                args.seq_len,
                args.n_out,
                args.student_units,
                4,
                4,
                4,
                4,
                4,
            )
            if args.condition == "scw_k2":
                model = common.SCWStudentModel(
                    seq_len=args.seq_len,
                    n_out=args.n_out,
                    student_units=args.student_units,
                    bits_kernel=4,
                    bits_recurrent=4,
                    bits_bias=4,
                    bits_activation=4,
                    bits_state=4,
                    counter_bits=SCW_COUNTER_BITS,
                    deadzone_fraction=SCW_DEADZONE_FRACTION,
                    q_alpha=1.0,
                )
            else:
                model = R2StudentModel(args)
            build_custom_children(model, args)
            dummy = tf.zeros((1, args.seq_len, 1), tf.float32)
            operator = "scw" if args.condition == "scw_k2" else "r2"
            model([dummy, dummy], training=False, operator_mode=operator)
            transfer_reference_weights(reference, model)
            equivalence = deterministic_equivalence(
                reference, model, normalized_input, val_idx, args, job_dir
            )
            del reference
    return model, equivalence


def main() -> None:
    args = parse_args()
    if args.no_lr_scaling:
        args.effective_lr = args.lr
        args.effective_lr_patience = args.lr_patience
        args.effective_warmup_epochs = args.warmup_epochs
    else:
        ratio = args.batch_size / args.ref_batch_size
        args.effective_lr = args.lr * ratio
        args.effective_lr_patience = max(1, round(args.lr_patience * ratio))
        args.effective_warmup_epochs = max(0, round(args.warmup_epochs * ratio))

    strategy = common.setup_gpus_and_strategy(args.mixed_precision)
    job_name = job_name_for(args)
    job_dir = Path(args.save_dir) / "results" / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    common.pf("=" * 72)
    common.pf("MATCHED RECURRENT-MEMORY TRAINING CAMPAIGN")
    common.pf(
        f"condition={args.condition} init_seed={args.init_seed} split_seed={args.split_seed}"
    )
    common.pf(
        f"visible={args.recurrence_visible_state_bits} aux={args.auxiliary_memory_bits} "
        f"total={args.total_stored_bits_per_unit} bits/unit"
    )
    common.pf(f"alpha={args.alpha} temperature={args.temperature} job_dir={job_dir}")
    common.pf("=" * 72)

    atomic_write_json(
        job_dir / "student_args.json",
        {
            **vars(args),
            "campaign_locked": True,
            "kd_loss": "alpha*MSE(student,teacher)+(1-alpha)*MSE(student,target)",
            "temperature_effect": "cancels algebraically in regression MSE implementation",
        },
    )
    script_path = Path(__file__).resolve()
    baseline_source = script_path.parent / "train_student_vanilla_kd.py"
    scw_source = script_path.parent / "train_student_vanilla_kd_scw.py"
    teacher_path = Path(args.teacher_ckpt)
    provenance = {
        "condition": args.condition,
        "init_seed": args.init_seed,
        "split_seed": args.split_seed,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "training_script_sha256": sha256_file(script_path),
        "baseline_source_sha256": sha256_file(baseline_source),
        "scw_source_sha256": sha256_file(scw_source),
        "teacher_checkpoint_sha256": sha256_file(teacher_path),
        "epoch_shuffle_policy": "split_seed_plus_zero_based_epoch",
    }
    provenance_path = job_dir / "run_provenance.json"
    if args.resume:
        if not provenance_path.is_file():
            raise RuntimeError(f"Resume provenance missing: {provenance_path}")
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError(f"Resume provenance mismatch: existing={existing}, current={provenance}")
    else:
        if provenance_path.exists():
            raise RuntimeError(f"Fresh run refused because provenance exists: {provenance_path}")
        atomic_write_json(provenance_path, provenance)

    files = common.find_data_files(args.data_dir, args.seq_len)
    file_input, file_res, file_labels, file_train, file_val, file_test = files
    normalized_input = np.load(file_input, mmap_mode="r")
    res = np.load(file_res, mmap_mode="r")
    labels = np.load(file_labels, mmap_mode="r")
    train_idx = np.asarray(np.load(file_train), np.int64)
    val_idx = np.asarray(np.load(file_val), np.int64)
    test_idx = np.asarray(np.load(file_test), np.int64)
    if normalized_input.shape[1:] != (args.seq_len, 1):
        raise RuntimeError(f"Unexpected input shape: {normalized_input.shape}")
    if res.shape[1:] != (args.seq_len, args.n_out):
        raise RuntimeError(f"Unexpected target shape: {res.shape}")
    if labels.ndim != 2 or labels.shape[1] < 3:
        raise RuntimeError(f"Unexpected label shape: {labels.shape}")
    if min(len(train_idx), len(val_idx), len(test_idx)) <= 0:
        raise RuntimeError("Data partitions must be non-empty")
    n = normalized_input.shape[0]
    if res.shape[0] != n or labels.shape[0] != n:
        raise RuntimeError("Input, target and label sample counts differ")
    common.pf(f"[DATA] N={n:,} train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    campaign_spec_path = Path(args.save_dir) / "campaign_spec.json"
    if not campaign_spec_path.is_file():
        raise RuntimeError(f"Locked campaign specification is missing: {campaign_spec_path}")
    with campaign_spec_path.open("r", encoding="utf-8") as handle:
        campaign_spec = json.load(handle)
    teacher_cache_path = (
        Path(args.data_dir)
        / f"teacherPred_vanillaKD_L{args.seq_len}{n}.npy"
    ).resolve()
    if not teacher_cache_path.is_file():
        raise RuntimeError(
            f"Locked teacher prediction cache is missing: {teacher_cache_path}"
        )
    expected_cache_sha = campaign_spec.get("teacher_prediction_cache_sha256")
    actual_cache_sha = sha256_file(teacher_cache_path)
    if expected_cache_sha != actual_cache_sha:
        raise RuntimeError(
            "Teacher prediction cache hash differs from the locked campaign "
            f"specification: expected={expected_cache_sha}, got={actual_cache_sha}"
        )
    if campaign_spec.get("teacher_prediction_cache") != str(teacher_cache_path):
        raise RuntimeError(
            "Teacher prediction cache path differs from the locked campaign "
            f"specification: expected={campaign_spec.get('teacher_prediction_cache')}, "
            f"got={teacher_cache_path}"
        )
    common.pf(
        f"[CACHE] locked teacher prediction cache verified sha256={actual_cache_sha}"
    )

    teacher = common.build_teacher(
        args.seq_len, args.n_out, args.teacher_units, args.teacher_layers
    )
    teacher.load_weights(args.teacher_ckpt)
    teacher.trainable = False
    teacher_pred = common.cache_teacher_predictions(
        teacher,
        normalized_input,
        args.seq_len,
        args.n_out,
        n,
        args.infer_batch,
        args.data_dir,
        common.pf,
    )
    enc_train, tgt_train, tpred_train = common.materialise_enc_tgt_tpred(
        normalized_input, res, teacher_pred, train_idx, args.seq_len, args.n_out, "train", common.pf
    )
    enc_val, tgt_val, tpred_val = common.materialise_enc_tgt_tpred(
        normalized_input, res, teacher_pred, val_idx, args.seq_len, args.n_out, "val", common.pf
    )

    model, equivalence = build_model(strategy, normalized_input, val_idx, args, job_dir)
    initial_weights_path, initial_weights_sha256 = save_or_verify_initial_weights(
        model, args.condition, job_dir, args.resume
    )
    common.pf(
        f"[INIT] seed={args.init_seed} initial_weights_sha256={initial_weights_sha256}"
    )
    with strategy.scope():
        optimizer = keras.optimizers.Adam(learning_rate=args.effective_lr)
        scheduler = common.CheckpointableReduceLROnPlateau(
            optimizer,
            args.lr_factor,
            args.effective_lr_patience,
            args.lr_min,
            args.min_delta,
        )
    common.pf(f"[MODEL] trainable params={model.count_params():,}")

    micro_batch = args.batch_size // args.accumulation_steps
    train_steps = len(train_idx) // micro_batch
    val_steps = len(val_idx) // micro_batch
    if train_steps <= 0 or val_steps <= 0:
        raise RuntimeError("Batch configuration produces zero steps")
    common.pf(
        f"[DATASET] train_steps={train_steps} val_steps={val_steps} "
        "shuffle_policy=split_seed_plus_zero_based_epoch"
    )

    history, best_val, complete = training_loop(
        strategy,
        model,
        optimizer,
        scheduler,
        enc_train,
        tgt_train,
        tpred_train,
        enc_val,
        tgt_val,
        tpred_val,
        train_steps,
        args,
        job_dir,
    )
    save_training_plot(history, job_dir, args)
    if not complete:
        common.pf("[RUN] Paused cleanly; pipeline gate will resume this condition.")
        return

    best_ckpt = job_dir / "student_best.weights.h5"
    model.load_weights(str(best_ckpt))
    evaluation = evaluate_condition(
        model, normalized_input, res, labels, test_idx, args, job_dir
    )
    final_weights = job_dir / "student_final.weights.h5"
    model.save_weights(str(final_weights))
    raw_weights_path = None
    if args.condition in ("r2", "scw_k2"):
        raw_weights_path = job_dir / "trained_raw_weights.npz"
        np.savez_compressed(raw_weights_path, **model.export_raw_weights())

    test_index_path = Path(file_test).resolve()
    manifest = {
        "training_complete": True,
        "condition": args.condition,
        "init_seed": args.init_seed,
        "split_seed": args.split_seed,
        "epoch_shuffle_policy": "split_seed_plus_zero_based_epoch",
        "job_name": job_name,
        "job_dir": str(job_dir),
        "alpha": args.alpha,
        "temperature": args.temperature,
        "bits_kernel": args.bits_kernel,
        "bits_bias": args.bits_bias,
        "bits_recurrent": args.bits_recurrent,
        "bits_activation": args.bits_activation,
        "recurrence_visible_state_bits": args.recurrence_visible_state_bits,
        "auxiliary_memory_bits": args.auxiliary_memory_bits,
        "total_stored_bits_per_unit": args.total_stored_bits_per_unit,
        "best_validation_loss": best_val,
        "initial_weights": str(initial_weights_path),
        "initial_weights_sha256": initial_weights_sha256,
        "selected_checkpoint": str(best_ckpt),
        "selected_checkpoint_sha256": sha256_file(best_ckpt),
        "final_weights": str(final_weights),
        "final_weights_sha256": sha256_file(final_weights),
        "raw_weights_npz": str(raw_weights_path) if raw_weights_path else None,
        "raw_weights_npz_sha256": sha256_file(raw_weights_path) if raw_weights_path else None,
        "training_script_sha256": sha256_file(script_path),
        "baseline_source_sha256": sha256_file(baseline_source),
        "scw_source_sha256": sha256_file(scw_source),
        "teacher_checkpoint": str(teacher_path),
        "teacher_checkpoint_sha256": sha256_file(teacher_path),
        "teacher_prediction_cache": str(teacher_cache_path),
        "teacher_prediction_cache_sha256": actual_cache_sha,
        "test_index_file": str(test_index_path),
        "test_index_sha256": sha256_file(test_index_path),
        "initialization_equivalence": equivalence,
        "evaluation": evaluation,
    }
    if args.condition in ("r2", "scw_k2"):
        manifest["operator"] = model.sencgru.metadata()
    else:
        manifest["operator"] = {
            "kind": "deterministic_state",
            "state_bits": args.bits_state,
            "recurrence_visible_state_bits": args.bits_state,
            "auxiliary_memory_bits": 0,
            "total_stored_bits_per_unit": args.total_stored_bits_per_unit,
        }
    atomic_write_json(job_dir / "training_manifest.json", manifest)
    (job_dir / "campaign_training_complete.flag").write_text("passed\n", encoding="utf-8")
    native = evaluation["native"]
    common.pf("=" * 72)
    common.pf(
        f"COMPLETE condition={args.condition} seed={args.init_seed} "
        f"MAE={native['mae_seq']:.6f} tau1={native['tau1']['rmse']:.6f} "
        f"tau2={native['tau2']['rmse']:.6f}"
    )
    common.pf("=" * 72)


if __name__ == "__main__":
    main()
