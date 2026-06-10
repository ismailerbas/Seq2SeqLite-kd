#!/usr/bin/env python3
"""
train_student_memoq_full.py  —  MemoQ: Memory-Preserving 4-bit Recurrent
Quantization for GRU-based Sequence-to-Sequence Knowledge Distillation.

ROOT CAUSE TARGETED:
  Small 4-bit GRUs fail through two coupled recurrent mechanisms:
  (1) Memory-gate saturation: simultaneous 4-bit quantization of z/r/h gates
      destabilises the update gate. Saturation toward 1 makes the cell
      feedforward (overwrites memory); saturation toward 0 freezes
      quantisation error in the hidden state.
  (2) Recurrent state-error accumulation: per-step quantisation errors
      accumulate through the recurrence, driving the hidden state to the
      4-bit boundary before output KD can recover temporal memory.

METHOD — THREE PHASES:
  Phase 1  (float/8-bit warm-up, 30-50 ep):
    Float student, same topology+layer names as final QKeras student.
    L = (1-alpha)*HuberCN(s, gt) + alpha*HuberCN(s, teacher)  [alpha=0.7]
    Channel scales from teacher predictions over training split.

  Phase 2  (gate-decoupled 4-bit hardening, stages 2A/2B/2C):
    Custom SplitGateQGRUCell with separate W_z, W_r, W_h / U_z, U_r, U_h.
    Quantise gates in causal order: h first, then r, then z (NOT packed order).
    Stage 2A: quantise h-gate (candidate). L += 0.01 L_mem + 0.0005 L_rail
              After 5 epochs activate L_innov (lambda=0.005).
    Stage 2B: quantise r-gate. L weights escalated.
    Stage 2C: quantise z-gate. Activate L_zsat (logit barrier).
    Full MemoQ loss:
      L = L_seq + alpha*L_KD + lm*L_mem + li*L_innov + lz*L_zsat + lr*L_rail

  Phase 3  (hard 4-bit polish):
    All gates/kernels/biases/activations/states hard 4-bit via standard
    QKeras QGRU. Weights transferred by concatenating split-gate variables
    in Keras/QKeras packed order [z|r|h]. Low LR ~5e-6, clipnorm=0.5.

AUXILIARY LOSSES:
  L_mem    — lagged temporal memory kernel distillation (dimension-free)
  L_innov  — temporal innovation-profile matching (log-ratio)
  L_zsat   — update-gate saturation barrier (logit barrier |logit|>3)
  L_rail   — predictive rail-margin regularisation on hidden state

FINAL EXPORT:
  Standard QKeras QGRU encoder (sencgru) + QGRU decoder (sdecgru) +
  QDense head (sdec_dense). No auxiliary parameters at inference.

USAGE:
  python train_student_memoq_full.py \\
      --data-dir /path/to/data \\
      --teacher-ckpt /path/to/teacher_best.weights.h5 \\
      --save-dir /path/to/runs \\
      --bits-kernel 4 --bits-bias 4 --bits-recurrent 4 \\
      --bits-activation 4 --bits-state 4 \\
      --student-units 32 --teacher-units 128 --teacher-layers 2 \\
      --seq-len 135 --n-out 3 --gate-width-ns 0.09 \\
      --batch-size 16384 --lr 1e-4 --ref-batch-size 1024 \\
      --lr-factor 0.5 --lr-patience 8 --lr-min 1e-6 \\
      --temperature 4.0 --alpha 0.7 \\
      --memoq-warmup-epochs 40 \\
      --memoq-stage2a-epochs 30 --memoq-stage2b-epochs 30 \\
      --memoq-stage2c-epochs 30 --memoq-finetune-epochs 170 \\
      --memoq-lambda-mem 0.03 --memoq-lambda-innov 0.005 \\
      --memoq-lambda-zsat 0.002 --memoq-lambda-rail 0.001 \\
      --memoq-huber-delta 0.1 --memoq-rho-z 0.98 \\
      --memoq-rho-rail 0.88 --memoq-mu-rail 0.9 \\
      --patience 30 --log-interval 10 --infer-batch 8192 \\
      --prefetch-batches 32 --pipeline-workers 4 \\
      --split-seed 42

OUTPUTS (all inside --save-dir / results / job_name /):
  phase1_best.weights.h5
  stage2a_best.weights.h5
  stage2b_best.weights.h5
  stage2c_best.weights.h5
  student_best.weights.h5       <- final hard 4-bit QKeras model
  student_final.weights.h5
  student_args.json
  training_history.csv
  training_history.png
  test_metrics.json
  test_scatter_tau1.png  test_scatter_tau2.png  test_scatter_fret.png
  test_residuals.png
"""

import argparse
import glob
import json
import os
import sys
import time
import math

# ── Force GPU visibility BEFORE any TF import ────────────────────────────────
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    print("[GPU] CUDA_VISIBLE_DEVICES defaulting to 0,1,2,3,4,5,6,7", flush=True)
else:
    print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except Exception:
    fastdtw  = None
    euclidean = None
    HAS_FASTDTW = False
from scipy.spatial.distance import euclidean

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GRU, Lambda, Concatenate
from tensorflow.keras.models import Model

from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MemoQ: Memory-Preserving 4-bit GRU KD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data-dir",       type=str, required=True)
    p.add_argument("--save-dir",       type=str, default=None)
    p.add_argument("--seq-len",        type=int, default=135)
    p.add_argument("--n-out",          type=int, default=3)
    p.add_argument("--gate-width-ns",  type=float, default=0.09)

    # ── Teacher ───────────────────────────────────────────────────────────────
    p.add_argument("--teacher-ckpt",   type=str, required=True)
    p.add_argument("--teacher-units",  type=int, default=128)
    p.add_argument("--teacher-layers", type=int, default=2)

    # ── KD ────────────────────────────────────────────────────────────────────
    p.add_argument("--temperature",    type=float, default=4.0)
    p.add_argument("--alpha",          type=float, default=0.7)

    # ── Quantisation ──────────────────────────────────────────────────────────
    p.add_argument("--bits-kernel",     type=int, default=4)
    p.add_argument("--bits-bias",       type=int, default=4)
    p.add_argument("--bits-recurrent",  type=int, default=4)
    p.add_argument("--bits-activation", type=int, default=4)
    p.add_argument("--bits-state",      type=int, default=4)

    # ── Architecture ──────────────────────────────────────────────────────────
    p.add_argument("--student-units",  type=int, default=32)

    # ── MemoQ curriculum hyper-parameters ─────────────────────────────────────
    p.add_argument("--memoq-warmup-epochs",      type=int,   default=40)
    p.add_argument("--memoq-stage2a-epochs",     type=int,   default=30)
    p.add_argument("--memoq-stage2b-epochs",     type=int,   default=30)
    p.add_argument("--memoq-stage2c-epochs",     type=int,   default=30)
    p.add_argument("--memoq-finetune-epochs",    type=int,   default=170,
                   help="Phase 3 hard 4-bit polish epochs.")
    p.add_argument("--memoq-lambda-mem",         type=float, default=0.03)
    p.add_argument("--memoq-lambda-innov",       type=float, default=0.005)
    p.add_argument("--memoq-lambda-zsat",        type=float, default=0.002)
    p.add_argument("--memoq-lambda-rail",        type=float, default=0.001)
    p.add_argument("--memoq-huber-delta",        type=float, default=0.1)
    p.add_argument("--memoq-rho-z",              type=float, default=0.98)
    p.add_argument("--memoq-rho-rail",           type=float, default=0.88)
    p.add_argument("--memoq-mu-rail",            type=float, default=0.9)
    p.add_argument("--memoq-innov-burnin",       type=int,   default=5)
    p.add_argument("--memoq-phase3-lr",          type=float, default=5e-6,
                   help="Fixed LR for Phase 3 hard fine-tune.")
    p.add_argument("--memoq-phase3-lr-factor",   type=float, default=1.0,
                   help="Multiplier on effective_lr to set Phase 3 LR; result capped at --memoq-phase3-lr.")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--batch-size",       type=int,   default=1024)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--ref-batch-size",   type=int,   default=1024)
    p.add_argument("--no-lr-scaling",    action="store_true", default=False)
    p.add_argument("--lr-factor",        type=float, default=0.5)
    p.add_argument("--lr-patience",      type=int,   default=8)
    p.add_argument("--lr-min",           type=float, default=1e-6)
    p.add_argument("--patience",         type=int,   default=30)
    p.add_argument("--min-delta",        type=float, default=1e-5)
    p.add_argument("--warmup-epochs",    type=int,   default=5)
    p.add_argument("--infer-batch",      type=int,   default=8192)
    p.add_argument("--mixed-precision",  action="store_true", default=False)
    p.add_argument("--log-interval",     type=int,   default=10)
    p.add_argument("--prefetch-batches", type=int,   default=32)
    p.add_argument("--pipeline-workers", type=int,   default=4)
    p.add_argument("--split-seed",       type=int,   default=42)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--resume",           action="store_true", default=False)

    args = p.parse_args()
    if args.save_dir is None:
        args.save_dir = args.data_dir

    # memoq_stage3_epochs is an alias for memoq_finetune_epochs for clarity
    args.memoq_stage3_epochs = args.memoq_finetune_epochs

    # ── Derived scalars ───────────────────────────────────────────────────────
    if args.no_lr_scaling:
        args.effective_lr = args.lr
        args.effective_lr_patience = args.lr_patience
        args.effective_warmup_epochs = args.warmup_epochs
    else:
        ratio = args.batch_size / args.ref_batch_size
        args.effective_lr = args.lr * ratio
        args.effective_lr_patience = max(1, int(round(args.lr_patience / ratio)))
        args.effective_warmup_epochs = max(1, int(round(args.warmup_epochs * ratio)))

    return args

# ==============================================================================
# Job naming
# ==============================================================================

def make_job_name(args) -> str:
    return (
        f"memoq"
        f"_a{args.alpha}"
        f"_b{args.bits_kernel}k{args.bits_recurrent}r{args.bits_activation}a"
        f"_gru{args.student_units}"
        f"_bs{args.batch_size}"
        f"_lr{args.effective_lr:.0e}"
        f"_w{args.memoq_warmup_epochs}"
        f"_2a{args.memoq_stage2a_epochs}"
        f"_2b{args.memoq_stage2b_epochs}"
        f"_2c{args.memoq_stage2c_epochs}"
        f"_f{args.memoq_finetune_epochs}"
    )


# ==============================================================================
# GPU / strategy setup
# ==============================================================================

def setup_gpus_and_strategy(mixed_precision: bool):
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        print("[GPU] No GPUs found — CPU mode.", flush=True)
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[GPU] set_memory_growth warning: {e}", flush=True)

    print(f"[GPU] Physical GPUs: {len(physical_gpus)}", flush=True)

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[GPU] Logical GPUs:  {len(logical_gpus)}", flush=True)

    if not logical_gpus:
        print("[GPU] No logical GPUs — CPU fallback.", flush=True)
        return tf.distribute.get_strategy()

    if mixed_precision:
        print("[GPU] WARNING: --mixed-precision disabled (QKeras incompatible). Using float32.", flush=True)

    keras.mixed_precision.set_global_policy("float32")

    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    print(f"[GPU] MirroredStrategy: {strategy.num_replicas_in_sync} replicas  devices={gpu_devices}", flush=True)
    return strategy


# ==============================================================================
# File discovery
# ==============================================================================

def find_data_files(data_dir, seq_len):
    def find_one(patterns, desc):
        for pat in patterns:
            matches = glob.glob(os.path.join(data_dir, pat))
            if matches:
                return sorted(matches)[0]
        raise FileNotFoundError(
            f"Cannot find {desc} in {data_dir}. Tried: {patterns}"
        )

    file_input = find_one(
        [f"tpsf_seq_L{seq_len}_*.npy"],
        "encoder input",
    )
    file_res = find_one(
        [f"res_L{seq_len}_*.npy"],
        "decoder target",
    )
    file_labels = find_one(
        [f"labels_3ch_L{seq_len}_*.npy"],
        "labels_3ch",
    )

    def find_idx(names, desc):
        for name in names:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{desc} not found in {data_dir}. Tried: {names}"
        )

    file_train = find_idx(["trainidx.npy", "train_idx.npy"], "train index")
    file_val   = find_idx(["validx.npy",   "val_idx.npy"],   "validation index")
    file_test  = find_idx(["testidx.npy",  "test_idx.npy"],  "test index")

    return file_input, file_res, file_labels, file_train, file_val, file_test

# ==============================================================================
# Teacher model
# ==============================================================================

def build_teacher(seq_len, n_out, teacher_units, teacher_layers):
    LAYERS = [teacher_units] * teacher_layers
    enc_inputs = Input(shape=(None, 1), name="enc_input")
    enc_cells = [
        keras.layers.GRUCell(u, reset_after=True, name=f"enc_cell{i}")
        for i, u in enumerate(LAYERS)
    ]
    enc_rnn = keras.layers.RNN(enc_cells, return_state=True, name="enc_rnn")
    enc_out_and_states = enc_rnn(enc_inputs)
    enc_states = enc_out_and_states[1:]

    dec_inputs = Input(shape=(None, 1), name="dec_input")
    dec_cells = [
        keras.layers.GRUCell(u, reset_after=True, name=f"dec_cell{i}")
        for i, u in enumerate(LAYERS)
    ]
    dec_rnn = keras.layers.RNN(dec_cells, return_sequences=True, return_state=True, name="dec_rnn")
    dec_out_and_states = dec_rnn(dec_inputs, initial_state=enc_states)
    dec_hidden_seq = dec_out_and_states[0]

    dec_dense = Dense(n_out, activation="linear", name="dec_dense")
    dec_output = dec_dense(dec_hidden_seq)

    model = Model(inputs=[enc_inputs, dec_inputs], outputs=dec_output, name="teacher_seq2seq")
    return model


def build_teacher_hidden_model(teacher_model):
    """
    Returns a model with same inputs as teacher but outputs the decoder hidden
    sequence (shape B, T, teacher_units) from dec_rnn layer index 0.
    """
    dec_rnn_output = teacher_model.get_layer("dec_rnn").output
    dec_hidden_seq = dec_rnn_output[0]
    hidden_model = Model(
        inputs=teacher_model.inputs,
        outputs=dec_hidden_seq,
        name="teacher_hidden_model",
    )
    return hidden_model


# ==============================================================================
# Float student — Phase 1 and base for Phase 2
# Identical topology + layer names to QKeras student.
# ==============================================================================

def build_float_student(seq_len, n_out, student_units):
    enc_inputs = Input(shape=(None, 1), name="senc_input")
    dec_inputs = Input(shape=(None, 1), name="sdec_input")

    enc_out, enc_state = GRU(
        student_units, return_state=True, name="sencgru"
    )(enc_inputs)

    dec_hid_seq, _ = GRU(
        student_units, return_sequences=True, return_state=True, name="sdecgru"
    )(dec_inputs, initial_state=enc_state)

    s_output = Dense(n_out, activation="linear", name="sdec_dense")(dec_hid_seq)

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="float_student",
    )
    return model


def build_float_student_with_hidden(seq_len, n_out, student_units):
    """
    Same as float student but outputs (predictions, dec_hidden_seq) as a list.
    Used during Phases 2A/2B/2C and Phase 3 to compute L_mem, L_innov, L_rail.
    """
    enc_inputs = Input(shape=(None, 1), name="senc_input")
    dec_inputs = Input(shape=(None, 1), name="sdec_input")

    enc_out, enc_state = GRU(
        student_units, return_state=True, name="sencgru"
    )(enc_inputs)

    dec_hid_seq, _ = GRU(
        student_units, return_sequences=True, return_state=True, name="sdecgru"
    )(dec_inputs, initial_state=enc_state)

    s_output = Dense(n_out, activation="linear", name="sdec_dense")(dec_hid_seq)

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[s_output, dec_hid_seq],
        name="float_student_with_hidden",
    )
    return model


# ==============================================================================
# SplitGateGRUCell — training-only custom cell for Phase 2 gate-decoupled
# quantization curriculum.
#
# Implements a standard reset_after=True GRU cell but with SEPARATE variables
# for each gate's kernel and recurrent kernel:
#   W_z (input_dim, units), W_r, W_h
#   U_z (units, units),     U_r, U_h
#   b_z (2, units),         b_r, b_h   (reset_after=True: 2 bias rows per gate)
#
# Each gate has an independently settable quantizer (or None = float).
# During call(), packed matrices are reconstructed as concat([W_z,W_r,W_h], axis=1)
# and passed through the standard GRU equations so gradients flow correctly.
#
# The cell also stores the last update gate logit z_logit as an attribute
# so that L_zsat can access it without a separate model output.
#
# At export, call get_packed_weights() to get weights in Keras/QKeras order.
# ==============================================================================

class SplitGateGRUCell(keras.layers.AbstractRNNCell):
    def __init__(self, units, input_dim, quantizer_z=None, quantizer_r=None,
                 quantizer_h=None, quantizer_state=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim_val = input_dim
        self.quantizer_z = quantizer_z
        self.quantizer_r = quantizer_r
        self.quantizer_h = quantizer_h
        self.quantizer_state = quantizer_state

        self._last_z_logit = None

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def build(self, input_shape):
        d = self.input_dim_val
        h = self.units
        init_glorot = keras.initializers.GlorotUniform()
        init_ortho  = keras.initializers.Orthogonal()
        init_zeros  = keras.initializers.Zeros()

        # Input kernels — shape (d, h) per gate
        self.W_z = self.add_weight(name="W_z", shape=(d, h), initializer=init_glorot)
        self.W_r = self.add_weight(name="W_r", shape=(d, h), initializer=init_glorot)
        self.W_h = self.add_weight(name="W_h", shape=(d, h), initializer=init_glorot)

        # Recurrent kernels — shape (h, h) per gate
        self.U_z = self.add_weight(name="U_z", shape=(h, h), initializer=init_ortho)
        self.U_r = self.add_weight(name="U_r", shape=(h, h), initializer=init_ortho)
        self.U_h = self.add_weight(name="U_h", shape=(h, h), initializer=init_ortho)

        # Biases — reset_after=True: shape (2, h) per gate.
        # Row 0 = input bias, Row 1 = recurrent bias.
        self.b_z = self.add_weight(name="b_z", shape=(2, h), initializer=init_zeros)
        self.b_r = self.add_weight(name="b_r", shape=(2, h), initializer=init_zeros)
        self.b_h = self.add_weight(name="b_h", shape=(2, h), initializer=init_zeros)

        self.built = True

    def _apply_quantizer(self, w, quantizer):
        if quantizer is None:
            return w
        return quantizer(w)

    def call(self, inputs, states, training=None):
        h_prev = states[0]   # (B, units)

        # Quantize the incoming recurrent state so accumulated error is trained against
        if self.quantizer_state is not None:
            h_prev_q = self.quantizer_state(h_prev)
        else:
            h_prev_q = h_prev

        # Per-gate kernel quantizers (W_* and U_* share one quantizer per gate)
        def qw(q, w):
            return q(w) if q is not None else w

        W_z = qw(self.quantizer_z, self.W_z)
        W_r = qw(self.quantizer_r, self.W_r)
        W_h = qw(self.quantizer_h, self.W_h)
        U_z = qw(self.quantizer_z, self.U_z)
        U_r = qw(self.quantizer_r, self.U_r)
        U_h = qw(self.quantizer_h, self.U_h)

        # b_z[0] = input bias, b_z[1] = recurrent bias
        z_logit = (
            tf.matmul(inputs, W_z) + self.b_z[0]
            + tf.matmul(h_prev_q, U_z) + self.b_z[1]
        )
        r_logit = (
            tf.matmul(inputs, W_r) + self.b_r[0]
            + tf.matmul(h_prev_q, U_r) + self.b_r[1]
        )

        z = tf.sigmoid(z_logit)
        r = tf.sigmoid(r_logit)

        h_cand = tf.tanh(
            tf.matmul(inputs, W_h) + self.b_h[0]
            + r * (tf.matmul(h_prev_q, U_h) + self.b_h[1])
        )

        # Keras/QKeras GRU convention: z * h_prev + (1 - z) * h_cand
        h_new = z * h_prev_q + (1.0 - z) * h_cand

        # Quantize the output state so the NEXT timestep receives a quantized state.
        # This is the core of recurrent error accumulation training.
        if self.quantizer_state is not None:
            h_new = self.quantizer_state(h_new)

        # Expose z_logit for L_zsat loss — training only, not used at inference
        self._last_z_logit = z_logit

        return h_new, [h_new]


    def get_packed_weights(self):
        """
        Returns weights in Keras/QKeras QGRU packed order [z, r, h]:
          kernel          shape (input_dim, 3*units) = concat([W_z, W_r, W_h], axis=1)
          recurrent_kernel shape (units, 3*units)     = concat([U_z, U_r, U_h], axis=1)
          bias            shape (2, 3*units)          = concat([b_z, b_r, b_h], axis=1)
        """
        kernel = tf.concat([self.W_z, self.W_r, self.W_h], axis=1).numpy()
        recurrent_kernel = tf.concat([self.U_z, self.U_r, self.U_h], axis=1).numpy()
        bias = tf.concat([self.b_z, self.b_r, self.b_h], axis=1).numpy()
        return kernel, recurrent_kernel, bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "input_dim_val": self.input_dim_val})
        return cfg


# ==============================================================================
# Build Phase 2 split-gate training models.
#
# We build two RNN layers (encoder + decoder) using SplitGateGRUCell.
# The models return (predictions, dec_hidden_seq, enc_z_logits, dec_z_logits)
# so all auxiliary losses can be computed.
#
# Because tf.keras.layers.RNN does not expose per-step internal state (z_logit)
# through its standard API, we implement the recurrence manually via
# tf.while_loop / unrolling for the decoder, which is where z_logit is needed
# for L_zsat. The encoder cell is also SplitGateGRUCell so its weights are
# split, but we only need decoder z_logits for L_zsat.
# ==============================================================================

class SplitGateRNNLayer(keras.layers.Layer):
    """
    A manual-unroll RNN that exposes per-step internal gate logits.
    For seq2seq: call as:
      outputs, final_state, z_logits_seq = layer(inputs, initial_state)
    outputs      : (B, T, units)
    final_state  : (B, units)
    z_logits_seq : (B, T, units) — update gate logits at each timestep
    """
    def __init__(self, cell, return_sequences=True, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell = cell
        self.return_sequences = return_sequences

    def call(self, inputs, initial_state=None, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len_static = inputs.shape[1]
        units = self.cell.units

        if seq_len_static is None:
            raise ValueError(
                "SplitGateRNNLayer requires a fixed sequence length. "
                "Use Input(shape=(seq_len, input_dim), ...) with a concrete integer seq_len."
            )

        if initial_state is None:
            h = tf.zeros((batch_size, units), dtype=inputs.dtype)
        else:
            h = initial_state

        outputs  = []
        z_logits = []

        for t in range(seq_len_static):
            x_t = inputs[:, t, :]
            h, _ = self.cell(x_t, [h], training=training)
            outputs.append(h)
            z_logits.append(self.cell._last_z_logit)

        outputs_seq  = tf.stack(outputs,  axis=1)   # (B, T, units)
        z_logits_seq = tf.stack(z_logits, axis=1)   # (B, T, units)

        return outputs_seq, h, z_logits_seq

def build_phase2_student(seq_len, n_out, student_units,
                         quantizer_z, quantizer_r, quantizer_h, quantizer_state):
    """
    Returns a Keras functional model (built around SplitGateRNNLayer) that
    outputs [predictions, dec_hidden_seq, dec_z_logits].
    The encoder SplitGateGRUCell is named sencgru_cell.
    The decoder SplitGateGRUCell is named sdecgru_cell.

    NOTE: Because SplitGateRNNLayer uses a Python for-loop over seq_len,
    it is compatible with tf.function tracing when seq_len is a known integer.
    We pass seq_len as a static integer for tracing efficiency.
    """
    input_dim = 1  # single-channel input at each timestep

    enc_cell = SplitGateGRUCell(
        units=student_units,
        input_dim=input_dim,
        quantizer_z=quantizer_z,
        quantizer_r=quantizer_r,
        quantizer_h=quantizer_h,
        quantizer_state=quantizer_state,
        name="sencgru_cell",
    )
    dec_cell = SplitGateGRUCell(
        units=student_units,
        input_dim=input_dim,
        quantizer_z=quantizer_z,
        quantizer_r=quantizer_r,
        quantizer_h=quantizer_h,
        quantizer_state=quantizer_state,
        name="sdecgru_cell",
    )

    enc_rnn_layer = SplitGateRNNLayer(enc_cell, return_sequences=True, name="sencgru")
    dec_rnn_layer = SplitGateRNNLayer(dec_cell, return_sequences=True, name="sdecgru")

    enc_inputs = Input(shape=(seq_len, 1), name="senc_input")
    dec_inputs = Input(shape=(seq_len, 1), name="sdec_input")

    enc_hidden_seq, enc_state, enc_z_logits = enc_rnn_layer(enc_inputs)
    dec_hidden_seq, dec_state, dec_z_logits = dec_rnn_layer(dec_inputs, initial_state=enc_state)

    s_output = Dense(n_out, activation="linear", name="sdec_dense")(dec_hidden_seq)

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[s_output, dec_hidden_seq, dec_z_logits],
        name="phase2_split_student",
    )
    return model, enc_cell, dec_cell


# ==============================================================================
# Final QKeras student — Phase 3 and export
# ==============================================================================

def build_qkeras_student(seq_len, n_out, student_units,
                         bits_kernel, bits_recurrent, bits_bias,
                         bits_activation, bits_state):
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

    enc_out, enc_state = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(enc_inputs)

    dec_hid_seq, _ = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        name="sdecgru",
    )(dec_inputs, initial_state=enc_state)

    s_output = QDense(
        n_out,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(dec_hid_seq)

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="qkeras_student_memoq",
    )
    return model


def build_qkeras_student_with_hidden(seq_len, n_out, student_units,
                                     bits_kernel, bits_recurrent, bits_bias,
                                     bits_activation, bits_state):
    """Phase 3 training model: outputs [predictions, dec_hidden_seq]."""
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

    enc_out, enc_state = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(enc_inputs)

    dec_hid_seq, _ = QGRU(
        units=student_units,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        name="sdecgru",
    )(dec_inputs, initial_state=enc_state)

    s_output = QDense(
        n_out,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(dec_hid_seq)

    model = Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[s_output, dec_hid_seq],
        name="qkeras_student_memoq_with_hidden",
    )
    return model


# ==============================================================================
# Weight transfer utilities
# ==============================================================================

def transfer_float_to_phase2(float_model, phase2_model, enc_cell, dec_cell, pf):
    """
    Transfer float student packed GRU weights into the Phase 2
    SplitGateGRUCell split-gate variables.

    Float student has standard Keras GRU layers named sencgru / sdecgru
    with reset_after=True:
        kernel          (input_dim, 3*units)   — packed [z | r | h]
        recurrent_kernel (units,    3*units)   — packed [z | r | h]
        bias             (2,        3*units)   — row 0 = input, row 1 = recurrent

    SplitGateGRUCell has:
        W_z, W_r, W_h   shape (input_dim, units)
        U_z, U_r, U_h   shape (units,     units)
        b_z, b_r, b_h   shape (2,         units)  — row 0 = input, row 1 = recurrent
    """
    pf("[TRANSFER P1->P2] Splitting packed float GRU weights into SplitGateGRUCell...")
    units = enc_cell.units

    for layer_name, cell in [("sencgru", enc_cell), ("sdecgru", dec_cell)]:
        try:
            fl = float_model.get_layer(layer_name)
        except ValueError:
            pf(f"  SKIP {layer_name} — not found in float_model")
            continue

        fw = fl.get_weights()
        if len(fw) < 3:
            pf(f"  SKIP {layer_name} — unexpected weight count {len(fw)}, expected >= 3")
            continue

        kernel_packed    = fw[0]   # (input_dim, 3*units)
        recurrent_packed = fw[1]   # (units,     3*units)
        bias_packed      = fw[2]   # (2,         3*units)

        # Keras/QKeras GRU packed order: [z | r | h]
        W_z = kernel_packed[:, :units]
        W_r = kernel_packed[:, units:2*units]
        W_h = kernel_packed[:, 2*units:]

        U_z = recurrent_packed[:, :units]
        U_r = recurrent_packed[:, units:2*units]
        U_h = recurrent_packed[:, 2*units:]

        # bias_packed[0] = input biases  [b_z_inp | b_r_inp | b_h_inp]
        # bias_packed[1] = recurrent biases [b_z_rec | b_r_rec | b_h_rec]
        b_z_inp = bias_packed[0, :units]
        b_r_inp = bias_packed[0, units:2*units]
        b_h_inp = bias_packed[0, 2*units:]

        b_z_rec = bias_packed[1, :units]
        b_r_rec = bias_packed[1, units:2*units]
        b_h_rec = bias_packed[1, 2*units:]

        cell.W_z.assign(W_z)
        cell.W_r.assign(W_r)
        cell.W_h.assign(W_h)

        cell.U_z.assign(U_z)
        cell.U_r.assign(U_r)
        cell.U_h.assign(U_h)

        # b_z shape is (2, units): row 0 = input bias, row 1 = recurrent bias
        cell.b_z.assign(np.stack([b_z_inp, b_z_rec], axis=0))
        cell.b_r.assign(np.stack([b_r_inp, b_r_rec], axis=0))
        cell.b_h.assign(np.stack([b_h_inp, b_h_rec], axis=0))

        pf(
            f"  OK {layer_name}: kernel={kernel_packed.shape}  "
            f"recurrent={recurrent_packed.shape}  bias={bias_packed.shape}"
        )

    try:
        src_dense = float_model.get_layer("sdec_dense")
        dst_dense = phase2_model.get_layer("sdec_dense")
        src_w = src_dense.get_weights()
        dst_w = dst_dense.get_weights()
        if len(src_w) == len(dst_w) and all(
            s.shape == d.shape for s, d in zip(src_w, dst_w)
        ):
            dst_dense.set_weights(src_w)
            pf(f"  OK sdec_dense: transferred {len(src_w)} tensors")
        else:
            pf(
                f"  SKIP sdec_dense: shape mismatch "
                f"src={[w.shape for w in src_w]} dst={[w.shape for w in dst_w]}"
            )
    except Exception as exc:
        pf(f"  SKIP sdec_dense: {exc}")

    sys.stdout.flush()


def transfer_phase2_to_qkeras(enc_cell, dec_cell, phase2_model, qkeras_model, pf):
    """
    Pack split-gate cell weights into QKeras QGRU packed format [z|r|h]
    and load them into the QKeras student model.
    """
    pf("[TRANSFER P2->P3] Packing split-gate weights into QKeras QGRU...")

    for layer_name, cell in [("sencgru", enc_cell), ("sdecgru", dec_cell)]:
        try:
            qkeras_layer = qkeras_model.get_layer(layer_name)
        except ValueError:
            pf(f"  SKIP {layer_name} — not found in qkeras_model")
            continue

        kernel_packed, recurrent_packed, bias_packed = cell.get_packed_weights()

        qkeras_weights = qkeras_layer.get_weights()
        if len(qkeras_weights) < 3:
            pf(f"  SKIP {layer_name} — QKeras layer has {len(qkeras_weights)} weights, expected >=3")
            continue

        new_weights = list(qkeras_weights)
        new_weights[0] = kernel_packed
        new_weights[1] = recurrent_packed
        new_weights[2] = bias_packed
        qkeras_layer.set_weights(new_weights)
        pf(f"  OK {layer_name}: kernel={kernel_packed.shape} rec={recurrent_packed.shape} bias={bias_packed.shape}")

    try:
        p2_dense = phase2_model.get_layer("sdec_dense")
        q3_dense = qkeras_model.get_layer("sdec_dense")
        q3_dense.set_weights(p2_dense.get_weights())
        pf("  OK sdec_dense")
    except Exception as e:
        pf(f"  SKIP sdec_dense: {e}")

    sys.stdout.flush()


def transfer_float_to_qkeras_by_name(float_model, qkeras_model, pf):
    """
    Fallback: transfer by matching layer names and weight shapes.
    Used after Phase 1 if Phase 2 is skipped (resume logic).
    For GRU layers, packed weight shapes must match.
    """
    pf("[TRANSFER FLOAT->QKERAS] Transferring by name...")
    float_map = {l.name: l for l in float_model.layers}
    for q_layer in qkeras_model.layers:
        if not q_layer.weights:
            continue
        if q_layer.name not in float_map:
            pf(f"  SKIP {q_layer.name!r} — not in float model")
            continue
        f_layer = float_map[q_layer.name]
        qw = q_layer.get_weights()
        fw = f_layer.get_weights()
        if len(qw) != len(fw):
            pf(f"  SKIP {q_layer.name!r} — weight count mismatch q={len(qw)} f={len(fw)}")
            continue
        if not all(a.shape == b.shape for a, b in zip(qw, fw)):
            pf(f"  SKIP {q_layer.name!r} — shape mismatch")
            continue
        q_layer.set_weights(fw)
        pf(f"  OK {q_layer.name!r}")
    sys.stdout.flush()


# ==============================================================================
# Teacher prediction + hidden cache
# ==============================================================================

def cache_teacher_predictions_and_hidden(
    teacher_model,
    teacher_hidden_model,
    normalized_input,
    seq_len,
    n_out,
    teacher_units,
    n_samples,
    infer_batch,
    data_dir,
    pf,
):
    file_pred   = os.path.join(data_dir, f"teacherPred_L{seq_len}{n_samples}.npy")
    file_hidden = os.path.join(data_dir, f"teacherHidden_L{seq_len}{n_samples}.npy")

    need_pred   = not os.path.exists(file_pred)
    need_hidden = not os.path.exists(file_hidden)

    if not need_pred:
        pf(f"[CACHE] Teacher pred cache found: {file_pred}")
    if not need_hidden:
        pf(f"[CACHE] Teacher hidden cache found: {file_hidden}")

    if not need_pred and not need_hidden:
        teacher_predictions = np.load(file_pred,   mmap_mode="r")
        teacher_hidden      = np.load(file_hidden, mmap_mode="r")
        pf(f"[CACHE] Loaded pred {teacher_predictions.shape}  hidden {teacher_hidden.shape}")
        sys.stdout.flush()
        return teacher_predictions, teacher_hidden

    pf("[CACHE] Running teacher inference to build cache(s)...")
    sys.stdout.flush()

    if need_pred:
        tp = np.lib.format.open_memmap(
            file_pred, mode="w+", dtype=np.float32, shape=(n_samples, seq_len, n_out)
        )
    if need_hidden:
        th = np.lib.format.open_memmap(
            file_hidden, mode="w+", dtype=np.float32, shape=(n_samples, seq_len, teacher_units)
        )

    @tf.function(reduce_retracing=True)
    def teacher_forward_pred(enc_b, dec_b):
        return teacher_model([enc_b, dec_b], training=False)

    @tf.function(reduce_retracing=True)
    def teacher_forward_hidden(enc_b, dec_b):
        return teacher_hidden_model([enc_b, dec_b], training=False)

    n_batches = int(np.ceil(n_samples / infer_batch))
    print_every = max(1, n_batches // 20)
    t0 = time.time()

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_samples)
        enc_b = tf.constant(normalized_input[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)

        if need_pred:
            pred = teacher_forward_pred(enc_b, dec_b)
            tp[s:e] = pred.numpy()
            tp.flush()

        if need_hidden:
            hid = teacher_forward_hidden(enc_b, dec_b)
            th[s:e] = hid.numpy()
            th.flush()

        del enc_b, dec_b

        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed = time.time() - t0
            pct = 100.0 * e / n_samples
            eta = (elapsed / max(b + 1, 1)) * (n_batches - b - 1)
            pf(f"[CACHE] {b+1:>4}/{n_batches}  {pct:5.1f}%  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")
            sys.stdout.flush()

    pf("[CACHE] Cache complete. Reopening read-only...")
    sys.stdout.flush()

    teacher_predictions = np.load(file_pred,   mmap_mode="r")
    teacher_hidden      = np.load(file_hidden, mmap_mode="r")
    pf(f"[CACHE] pred={teacher_predictions.shape}  hidden={teacher_hidden.shape}")
    sys.stdout.flush()
    return teacher_predictions, teacher_hidden


# ==============================================================================
# Materialise split buffers into contiguous RAM
# ==============================================================================

def materialise_buffers(normalized_input, res, teacher_predictions,
                        teacher_hidden, idx, seq_len, n_out, label, pf):
    n = len(idx)
    pf(f"  Materialising {label} buffers ({n:,} samples)...")
    t0 = time.time()

    enc   = np.empty((n, seq_len, 1),             dtype=np.float32)
    tgt   = np.empty((n, seq_len, n_out),          dtype=np.float32)
    tpred = np.empty((n, seq_len, n_out),          dtype=np.float32)
    teacher_hidden  = np.empty((n, seq_len, teacher_hidden.shape[2]), dtype=np.float32)

    chunk = 65536
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        enc[s:e]   = normalized_input[idx[s:e]]
        tgt[s:e]   = res[idx[s:e]]
        tpred[s:e] = teacher_predictions[idx[s:e]]
        teacher_hidden[s:e]  = teacher_hidden[idx[s:e]]

    pf(f"  Done in {time.time()-t0:.1f}s  enc={enc.nbytes/1e9:.2f}GB")
    sys.stdout.flush()
    return enc, tgt, tpred, teacher_hidden


# ==============================================================================
# tf.data pipeline — includes teacher hidden in batch
# ==============================================================================

def make_kd_dataset(enc_arr, tgt_arr, tpred_arr, teacher_hidden_arr,
                    batch_size, seq_len, n_out, teacher_hidden_dim,
                    shuffle, seed, prefetch_batches, pipeline_workers):
    n = len(enc_arr)
    dec_arr = np.zeros_like(enc_arr)
    micro_batch_size = batch_size

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(enc_arr),
        tf.data.Dataset.from_tensor_slices(dec_arr),
        tf.data.Dataset.from_tensor_slices(tpred_arr),
        tf.data.Dataset.from_tensor_slices(tgt_arr),
        tf.data.Dataset.from_tensor_slices(teacher_hidden_arr),
    ))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 200_000), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(micro_batch_size, drop_remainder=True)

    def set_shapes(enc_b, dec_b, tpred_b, tgt_b, teacher_hidden_b):
        enc_b.set_shape([micro_batch_size, seq_len, 1])
        dec_b.set_shape([micro_batch_size, seq_len, 1])
        tpred_b.set_shape([micro_batch_size, seq_len, n_out])
        tgt_b.set_shape([micro_batch_size, seq_len, n_out])
        teacher_hidden_b.set_shape([micro_batch_size, seq_len, teacher_hidden_dim])
        batchx = {
            "enc_input":      enc_b,
            "dec_input":      dec_b,
            "tpred":          tpred_b,
            "teacher_hidden": teacher_hidden_b,
        }
        return batchx, tgt_b

    ds = ds.map(set_shapes, num_parallel_calls=pipeline_workers)
    ds = ds.prefetch(prefetch_batches)
    return ds

# ==============================================================================
# Channel-normalised Huber loss
# ==============================================================================

def channel_normalised_huber(y_true, y_pred, channel_scales, huber_delta):
    """
    y_true, y_pred : (B, T, C)
    channel_scales : (C,) tf.constant — per-channel std of teacher preds
    Returns scalar mean loss.
    """
    residual = (y_pred - y_true) / channel_scales
    abs_res  = tf.abs(residual)
    delta    = tf.cast(huber_delta, tf.float32)
    huber    = tf.where(
        abs_res <= delta,
        0.5 * tf.square(residual),
        delta * (abs_res - 0.5 * delta),
    )
    return tf.reduce_mean(huber)


def compute_channel_scales(tpred_train, n_out, pf):
    pf("[SCALES] Per-channel teacher std:")
    scales = []
    for c in range(n_out):
        std_c = float(np.std(tpred_train[:, :, c]))
        std_c = max(std_c, 1e-3)
        scales.append(std_c)
        pf(f"  channel {c}: std={std_c:.6f}")
    sys.stdout.flush()
    return tf.constant(scales, dtype=tf.float32)


# ==============================================================================
# MemoQ auxiliary losses
# ==============================================================================

# ── L_mem: lagged temporal memory kernel distillation ─────────────────────────

MEMORY_LAGS = [1, 2, 4, 8, 16, 32, 64]

def memory_kernel(h_seq, lag):
    """
    h_seq: (B, T, H)
    Returns scalar: mean over batch and time of cosine similarity at lag ell.
    """
    h_a = h_seq[:, lag:, :]     # (B, T-lag, H)
    h_b = h_seq[:, :-lag, :]    # (B, T-lag, H)

    norm_a = tf.norm(h_a, axis=-1, keepdims=True) + 1e-8
    norm_b = tf.norm(h_b, axis=-1, keepdims=True) + 1e-8
    cos_sim = tf.reduce_sum((h_a / norm_a) * (h_b / norm_b), axis=-1)  # (B, T-lag)
    return tf.reduce_mean(cos_sim)


def loss_mem(student_hidden, teacher_hidden, seq_len):
    """
    student_hidden: (B, T, H_s)
    teacher_hidden: (B, T, H_t)   H_s != H_t is OK — kernel is scalar per lag.
    Returns scalar L_mem.
    """
    T = tf.cast(seq_len, tf.float32)
    total = tf.constant(0.0, dtype=tf.float32)
    for lag in MEMORY_LAGS:
        if lag >= seq_len:
            continue
        lag_f = tf.cast(lag, tf.float32)
        weight = (1.0 / tf.sqrt(lag_f)) * ((T - lag_f) / T)
        m_s = memory_kernel(student_hidden, lag)
        m_t = memory_kernel(teacher_hidden, lag)
        total = total + weight * tf.square(m_s - m_t)
    return total


# ── L_innov: temporal innovation profile matching ────────────────────────────

def innovation_profile(h_seq):
    """
    h_seq: (B, T, H)
    Returns v: (T-1,) — mean squared per-step change per timestep.
    """
    delta = h_seq[:, 1:, :] - h_seq[:, :-1, :]   # (B, T-1, H)
    v = tf.reduce_mean(tf.square(delta), axis=[0, 2])  # (T-1,)
    return v


def loss_innov(student_hidden, teacher_hidden, epsilon_innov):
    """
    student_hidden: (B, T, H_s)
    teacher_hidden: (B, T, H_t)
    epsilon_innov : scalar float32
    Returns scalar L_innov.
    """
    v_s = innovation_profile(student_hidden)  # (T-1,)
    v_t = innovation_profile(teacher_hidden)  # (T-1,)

    eps = tf.cast(epsilon_innov, tf.float32)
    log_ratio = tf.math.log((v_s + eps) / (v_t + eps))
    return tf.reduce_mean(tf.square(log_ratio))


# ── L_zsat: update-gate saturation barrier (logit barrier) ──────────────────

def loss_zsat_logit(z_logits, logit_threshold=3.0):
    """
    z_logits: (B, T, H) — update gate pre-sigmoid logits
    Returns scalar: mean ReLU(|logit| - threshold)^2
    """
    threshold = tf.cast(logit_threshold, tf.float32)
    excess = tf.nn.relu(tf.abs(z_logits) - threshold)
    return tf.reduce_mean(tf.square(excess))


def loss_zsat_value(z_values, rho_z=0.98):
    """
    z_values: (B, T, H) — update gate post-sigmoid values in [0,1]
    Returns scalar: mean (ReLU(z-rho)^2 + ReLU((1-z)-rho)^2)
    Penalises gates very close to 0 or 1 only.
    """
    rho = tf.cast(rho_z, tf.float32)
    penalty_hi = tf.nn.relu(z_values       - rho)
    penalty_lo = tf.nn.relu((1.0 - z_values) - rho)
    return tf.reduce_mean(tf.square(penalty_hi) + tf.square(penalty_lo))


# ── L_rail: predictive rail-margin regularisation ────────────────────────────

def loss_rail(student_hidden, rho_rail=0.88, mu_rail=0.9):
    """
    student_hidden: (B, T, H) — quantized or fake-quantized student hidden state
    Returns scalar L_rail.
    Penalises hidden states where |h_t| + mu*|h_t - h_{t-1}| > rho,
    predicting imminent rail collision before it happens.
    """
    h_curr = student_hidden[:, 1:, :]   # (B, T-1, H)
    h_prev = student_hidden[:, :-1, :]  # (B, T-1, H)
    mu     = tf.cast(mu_rail, tf.float32)
    rho    = tf.cast(rho_rail, tf.float32)
    margin = tf.abs(h_curr) + mu * tf.abs(h_curr - h_prev)
    excess = tf.nn.relu(margin - rho)
    return tf.reduce_mean(tf.square(excess))


# ==============================================================================
# Compute epsilon_innov from teacher hidden cache (once)
# ==============================================================================

def compute_epsilon_innov(teacher_hidden_train, pf):
    """
    epsilon_innov = 0.1 * median(v_t(teacher)) over all timesteps.
    teacher_hidden_train: np.ndarray (N, T, H_t)
    """
    pf("[EPS_INNOV] Computing epsilon_innov from teacher hidden cache...")
    # sample up to 50k for speed
    n = min(50000, teacher_hidden_train.shape[0])
    idx = np.random.choice(teacher_hidden_train.shape[0], n, replace=False)
    h = teacher_hidden_train[idx].astype(np.float32)  # (n, T, H)
    delta = h[:, 1:, :] - h[:, :-1, :]               # (n, T-1, H)
    v = np.mean(delta ** 2, axis=(0, 2))              # (T-1,)
    eps = 0.1 * float(np.median(v))
    eps = max(eps, 1e-6)
    pf(f"[EPS_INNOV] epsilon_innov={eps:.2e}")
    sys.stdout.flush()
    return eps


# ==============================================================================
# ReduceLROnPlateau
# ==============================================================================

class ReduceLROnPlateau:
    def __init__(self, optimizer, factor, patience, min_lr, min_delta):
        self.factor    = factor
        self.patience  = patience
        self.min_lr    = min_lr
        self.min_delta = min_delta
        self.best      = float("inf")
        self.wait      = 0

        if callable(optimizer.learning_rate):
            cur = float(optimizer.learning_rate(optimizer.iterations))
        else:
            cur = float(K.get_value(optimizer.learning_rate))

        self.lr_var = tf.Variable(cur, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = self.lr_var

    @property
    def current_lr(self):
        return float(K.get_value(self.lr_var))

    def step(self, val_loss, epoch, pfn):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            old_lr = self.current_lr
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                self.lr_var.assign(new_lr)
                pfn(f"ReduceLR: {old_lr:.2e} -> {new_lr:.2e}")
            self.wait = 0
            return True
        return False

    def reset(self, new_lr):
        self.best = float("inf")
        self.wait = 0
        self.lr_var.assign(float(new_lr))


# ==============================================================================
# Progress bar
# ==============================================================================

def bar(step, total, metrics: dict, epoch_start_time: float, width=28):
    frac   = step / max(total, 1)
    filled = int(width * frac)
    b      = "█" * filled + "░" * (width - filled)
    stats  = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
    elapsed = time.time() - epoch_start_time
    if step > 0:
        rem = (elapsed / step) * (total - step)
        eta = f"{rem/60:.0f}m{int(rem)%60:02d}s" if rem >= 60 else f"{rem:.0f}s"
        el  = f"{elapsed/60:.1f}m" if elapsed >= 60 else f"{elapsed:.0f}s"
        ts  = f"  [{el}<{eta}]"
    else:
        ts = ""
    sys.stdout.write(f"\r{step:5}/{total}  {b}  {frac*100:5.1f}%  {stats}{ts}")
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ==============================================================================
# Phase 1 train/val steps — float student, no auxiliary losses
# ==============================================================================

def train_step_phase1_per_replica(batch_x, batch_y, model, optimizer,
                                  alpha, channel_scales, huber_delta):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    with tf.GradientTape() as tape:
        s_out = model([enc_b, dec_b], training=True)
        l_seq = channel_normalised_huber(tgt_b,   s_out, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber(tpred_b, s_out, channel_scales, huber_delta)
        total = (1.0 - alpha) * l_seq + alpha * l_kd

    grads = tape.gradient(total, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    nan_flag = tf.cast(
        tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(g)) for g in grads])),
        tf.float32
    )
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total, l_seq, l_kd, nan_flag


def val_step_phase1_per_replica(batch_x, batch_y, model, alpha, channel_scales, huber_delta):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    tgt_b   = batch_y

    s_out = model([enc_b, dec_b], training=False)
    l_seq = channel_normalised_huber(tgt_b,   s_out, channel_scales, huber_delta)
    l_kd  = channel_normalised_huber(tpred_b, s_out, channel_scales, huber_delta)
    total = (1.0 - alpha) * l_seq + alpha * l_kd
    mae   = tf.reduce_mean(tf.abs(s_out - tgt_b))
    return total, l_seq, l_kd, mae


def make_dist_phase1_train(strategy, model, optimizer, alpha, channel_scales, huber_delta):
    @tf.function
    def step(bx, by):
        pr = strategy.run(train_step_phase1_per_replica,
                          args=(bx, by, model, optimizer, alpha, channel_scales, huber_delta))
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM,  pr[3], axis=None) > 0.0,
        )
    return step


def make_dist_phase1_val(strategy, model, alpha, channel_scales, huber_delta):
    @tf.function
    def step(bx, by):
        pr = strategy.run(val_step_phase1_per_replica,
                          args=(bx, by, model, alpha, channel_scales, huber_delta))
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[3], axis=None),
        )
    return step


# ==============================================================================
# Phase 2 / Phase 3 train/val steps — full MemoQ loss
#
# model outputs: [predictions (B,T,n_out), dec_hidden (B,T,H_s), dec_z_logits (B,T,H_s)]
#   for Phase 2 (SplitGateRNNLayer model)
#   OR [predictions, dec_hidden] for Phase 3 (QKeras model, no z_logits)
#
# Flags:
#   use_mem    : bool — compute L_mem
#   use_innov  : bool — compute L_innov
#   use_zsat   : bool — compute L_zsat
#   use_rail   : bool — compute L_rail
#   has_z_logit: bool — model outputs 3 tensors (pred, hidden, z_logit)
#                       if False only (pred, hidden) — use L_zsat value form
# ==============================================================================

def train_step_memoq_per_replica(
    batch_x, batch_y, model, optimizer,
    alpha, channel_scales, huber_delta,
    lambda_m, lambda_i, lambda_z, lambda_r,
    epsilon_innov, seq_len_int,
    rho_rail, mu_rail,
    use_mem, use_innov, use_zsat, use_rail, has_z_logit,
    clipnorm,
):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    teacher_hidden_b = batch_x["teacher_hidden"]
    tgt_b   = batch_y

    with tf.GradientTape() as tape:
        model_out = model([enc_b, dec_b], training=True)

        if has_z_logit:
            s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
        else:
            s_pred, s_hid = model_out[0], model_out[1]
            z_logits = None

        l_seq = channel_normalised_huber(tgt_b,   s_pred, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber(tpred_b, s_pred, channel_scales, huber_delta)

        l_mem   = loss_mem(s_hid, teacher_hidden_b, seq_len_int)   if use_mem   else tf.constant(0.0)
        l_innov = loss_innov(s_hid, teacher_hidden_b, epsilon_innov) if use_innov else tf.constant(0.0)
        l_rail  = loss_rail(s_hid, rho_rail, mu_rail)     if use_rail  else tf.constant(0.0)

        if use_zsat:
            if has_z_logit and z_logits is not None:
                l_zsat = loss_zsat_logit(z_logits, logit_threshold=3.0)
            else:
                # Phase 3: no logits available — use sigmoid(hidden) as proxy
                l_zsat = tf.constant(0.0, dtype=tf.float32)
        else:
            l_zsat = tf.constant(0.0)

        total = (
            (1.0 - alpha) * l_seq
            + alpha        * l_kd
            + lambda_m     * l_mem
            + lambda_i     * l_innov
            + lambda_z     * l_zsat
            + lambda_r     * l_rail
        )

    grads = tape.gradient(total, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
    nan_flag = tf.cast(
        tf.reduce_any(tf.stack([tf.reduce_any(tf.math.is_nan(g)) for g in grads])),
        tf.float32
    )
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=clipnorm)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, nan_flag


def val_step_memoq_per_replica(
    batch_x, batch_y, model,
    alpha, channel_scales, huber_delta,
    lambda_m, lambda_i, lambda_z, lambda_r,
    epsilon_innov, seq_len_int,
    rho_rail, mu_rail,
    use_mem, use_innov, use_zsat, use_rail, has_z_logit,
):
    enc_b   = batch_x["enc_input"]
    dec_b   = batch_x["dec_input"]
    tpred_b = batch_x["tpred"]
    teacher_hidden_b  = batch_x["teacher_hidden"]
    tgt_b   = batch_y

    model_out = model([enc_b, dec_b], training=False)

    if has_z_logit:
        s_pred, s_hid, z_logits = model_out[0], model_out[1], model_out[2]
    else:
        s_pred, s_hid = model_out[0], model_out[1]
        z_logits = None

    l_seq = channel_normalised_huber(tgt_b,   s_pred, channel_scales, huber_delta)
    l_kd  = channel_normalised_huber(tpred_b, s_pred, channel_scales, huber_delta)
    l_mem   = loss_mem(s_hid, teacher_hidden_b, seq_len_int)    if use_mem   else tf.constant(0.0)
    l_innov = loss_innov(s_hid, teacher_hidden_b, epsilon_innov) if use_innov else tf.constant(0.0)
    l_rail  = loss_rail(s_hid, rho_rail, mu_rail)      if use_rail  else tf.constant(0.0)

    if use_zsat:
        if has_z_logit and z_logits is not None:
            l_zsat = loss_zsat_logit(z_logits, logit_threshold=3.0)
        else:
            l_zsat = tf.constant(0.0, dtype=tf.float32)
    else:
        l_zsat = tf.constant(0.0)

    total = (
        (1.0 - alpha) * l_seq
        + alpha        * l_kd
        + lambda_m     * l_mem
        + lambda_i     * l_innov
        + lambda_z     * l_zsat
        + lambda_r     * l_rail
    )
    mae = tf.reduce_mean(tf.abs(s_pred - tgt_b))
    return total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae


def make_dist_memoq_train(strategy, model, optimizer,
                          alpha, channel_scales, huber_delta,
                          lambda_m, lambda_i, lambda_z, lambda_r,
                          epsilon_innov, seq_len_int,
                          rho_rail, mu_rail,
                          use_mem, use_innov, use_zsat, use_rail,
                          has_z_logit, clipnorm):
    # Phase 2 cannot use @tf.function because SplitGateRNNLayer uses Python for-loop.
    # Phase 3 QKeras model CAN use @tf.function.
    # We wrap both the same way — without @tf.function here for safety.
    # Callers that know the model is fully traceable can wrap separately.
    def step(bx, by):
        pr = strategy.run(
            train_step_memoq_per_replica,
            args=(
                bx, by, model, optimizer,
                alpha, channel_scales, huber_delta,
                lambda_m, lambda_i, lambda_z, lambda_r,
                epsilon_innov, seq_len_int,
                rho_rail, mu_rail,
                use_mem, use_innov, use_zsat, use_rail, has_z_logit,
                clipnorm,
            ),
        )
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[3], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[4], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[5], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[6], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.SUM,  pr[7], axis=None) > 0.0,
        )
    return step


def make_dist_memoq_val(strategy, model,
                        alpha, channel_scales, huber_delta,
                        lambda_m, lambda_i, lambda_z, lambda_r,
                        epsilon_innov, seq_len_int,
                        rho_rail, mu_rail,
                        use_mem, use_innov, use_zsat, use_rail,
                        has_z_logit):
    @tf.function
    def step(bx, by):
        pr = strategy.run(
            val_step_memoq_per_replica,
            args=(
                bx, by, model,
                alpha, channel_scales, huber_delta,
                lambda_m, lambda_i, lambda_z, lambda_r,
                epsilon_innov, seq_len_int,
                rho_rail, mu_rail,
                use_mem, use_innov, use_zsat, use_rail, has_z_logit,
            ),
        )
        return (
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[0], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[1], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[2], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[3], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[4], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[5], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[6], axis=None),
            strategy.reduce(tf.distribute.ReduceOp.MEAN, pr[7], axis=None),
        )
    return step


def build_final_qkeras_hidden_model(final_qkeras_student):
    """
    Build a side model from the hard QKeras student that exposes
    [seq_output, dec_hidden_seq] so Phase 3 recurrent losses can fire.

    The final QKeras student has layers named sencgru, sdecgru, sdec_dense.
    sdecgru with return_sequences=True returns shape (B, T, units).
    We extract that as the dec_hidden_seq.
    """
    enc_input  = final_qkeras_student.get_layer("senc_input").output
    dec_input  = final_qkeras_student.get_layer("sdec_input").output

    enc_layer  = final_qkeras_student.get_layer("sencgru")
    dec_layer  = final_qkeras_student.get_layer("sdecgru")
    dense_layer = final_qkeras_student.get_layer("sdec_dense")

    enc_out_full = enc_layer(enc_layer.input)
    if isinstance(enc_out_full, (list, tuple)):
        enc_seq  = enc_out_full[0]
        enc_final = enc_out_full[1]
    else:
        enc_seq   = enc_out_full
        enc_final = enc_seq[:, -1, :]

    dec_out_full = dec_layer(dec_input, initial_state=[enc_final])
    if isinstance(dec_out_full, (list, tuple)):
        dec_hidden_seq = dec_out_full[0]
    else:
        dec_hidden_seq = dec_out_full

    seq_output = dense_layer(dec_hidden_seq)

    hidden_model = keras.models.Model(
        inputs=[enc_layer.input, dec_layer.input],
        outputs=[seq_output, dec_hidden_seq],
        name="final_qkeras_hidden_model",
    )
    return hidden_model


def make_dist_memoq_train_final(
    strategy,
    final_qkeras_student,
    optimizer,
    alpha,
    channel_scales,
    huber_delta,
    lambda_mem,
    lambda_innov,
    lambda_zsat,
    lambda_rail,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    clipnorm=0.5,
):
    """
    Distributed train step for Phase 3 hard 4-bit QKeras polish.

    The final QKeras model is called with training=True.
    To extract dec_hidden_seq for recurrent losses, we call the encoder
    and decoder QGRU layers explicitly so we can intercept the hidden sequence.

    Loss:
      L = L_seq + 0.5*L_KD + lambda_mem*L_mem + lambda_innov*L_innov
          + lambda_zsat*L_zsat + lambda_rail*L_railpred
    """
    enc_layer   = final_qkeras_student.get_layer("sencgru")
    dec_layer   = final_qkeras_student.get_layer("sdecgru")
    dense_layer = final_qkeras_student.get_layer("sdec_dense")

    channel_scales_t = tf.constant(channel_scales, dtype=tf.float32)
    lags             = [1, 2, 4, 8, 16, 32, 64]
    T_f              = tf.cast(seq_len, tf.float32)

    def per_replica_step(batch_x, batch_y):
        enc_inp        = batch_x["enc_input"]
        dec_inp        = batch_x["dec_input"]
        t_pred         = batch_x["tpred"]
        t_hidden       = batch_x["teacher_hidden"]
        ground_truth   = batch_y

        with tf.GradientTape() as tape:
            # ── Forward: encoder ──────────────────────────────────────────────
            enc_result = enc_layer(enc_inp, training=True)
            if isinstance(enc_result, (list, tuple)):
                enc_seq   = enc_result[0]
                enc_final = enc_result[1] if len(enc_result) > 1 else enc_seq[:, -1, :]
            else:
                enc_seq   = enc_result
                enc_final = enc_seq[:, -1, :]

            # ── Forward: decoder ──────────────────────────────────────────────
            dec_result = dec_layer(dec_inp, initial_state=[enc_final], training=True)
            if isinstance(dec_result, (list, tuple)):
                dec_hidden_seq = dec_result[0]
                z_logit_seq    = dec_result[1] if len(dec_result) > 1 else None
            else:
                dec_hidden_seq = dec_result
                z_logit_seq    = None

            seq_output = dense_layer(dec_hidden_seq, training=True)

            # ── L_seq: channel-normalised Huber ───────────────────────────────
            diff_seq  = (seq_output - ground_truth) / (channel_scales_t + 1e-8)
            abs_diff  = tf.abs(diff_seq)
            huber_seq = tf.where(
                abs_diff <= huber_delta,
                0.5 * tf.square(diff_seq),
                huber_delta * (abs_diff - 0.5 * huber_delta),
            )
            l_seq = tf.reduce_mean(huber_seq)

            # ── L_KD: channel-normalised Huber vs teacher preds ───────────────
            diff_kd  = (seq_output - t_pred) / (channel_scales_t + 1e-8)
            abs_kd   = tf.abs(diff_kd)
            huber_kd = tf.where(
                abs_kd <= huber_delta,
                0.5 * tf.square(diff_kd),
                huber_delta * (abs_kd - 0.5 * huber_delta),
            )
            l_kd = tf.reduce_mean(huber_kd)

            # ── L_mem: lagged temporal memory kernel distillation ─────────────
            l_mem = tf.constant(0.0, dtype=tf.float32)
            if lambda_mem > 0.0:
                for lag in lags:
                    if lag >= seq_len:
                        continue
                    lag_f  = tf.cast(lag, tf.float32)
                    w_ell  = (1.0 / tf.sqrt(lag_f)) * ((T_f - lag_f) / T_f)
                    s_a    = dec_hidden_seq[:, lag:, :]
                    s_b    = dec_hidden_seq[:, :-lag, :]
                    t_a    = t_hidden[:, lag:, :]
                    t_b    = t_hidden[:, :-lag, :]
                    eps_cs = 1e-8
                    s_dot  = tf.reduce_sum(s_a * s_b, axis=-1)
                    s_na   = tf.norm(s_a, axis=-1)
                    s_nb   = tf.norm(s_b, axis=-1)
                    m_s    = tf.reduce_mean(s_dot / (s_na * s_nb + eps_cs))
                    t_dot  = tf.reduce_sum(t_a * t_b, axis=-1)
                    t_na   = tf.norm(t_a, axis=-1)
                    t_nb   = tf.norm(t_b, axis=-1)
                    m_t    = tf.reduce_mean(t_dot / (t_na * t_nb + eps_cs))
                    l_mem  = l_mem + w_ell * tf.square(m_s - m_t)

            # ── L_innov: temporal innovation-profile matching ─────────────────
            l_innov = tf.constant(0.0, dtype=tf.float32)
            if lambda_innov > 0.0:
                s_diff  = dec_hidden_seq[:, 1:, :] - dec_hidden_seq[:, :-1, :]
                t_diff  = t_hidden[:, 1:, :] - t_hidden[:, :-1, :]
                v_s     = tf.reduce_mean(tf.square(s_diff), axis=-1)
                v_t     = tf.reduce_mean(tf.square(t_diff), axis=-1)
                eps_inv = tf.cast(epsilon_innov, tf.float32)
                log_ratio = tf.math.log(
                    (v_s + eps_inv) / (v_t + eps_inv + 1e-30)
                )
                l_innov = tf.reduce_mean(tf.square(log_ratio))

            # ── L_zsat: update-gate saturation barrier (logit form) ───────────
            l_zsat = tf.constant(0.0, dtype=tf.float32)
            if lambda_zsat > 0.0 and z_logit_seq is not None:
                l_zsat = tf.reduce_mean(
                    tf.square(tf.nn.relu(tf.abs(z_logit_seq) - 3.0))
                )

            # ── L_railpred: recurrent state rail-margin regularisation ─────────
            l_rail = tf.constant(0.0, dtype=tf.float32)
            if lambda_rail > 0.0:
                h_t   = dec_hidden_seq[:, 1:, :]
                h_tm1 = dec_hidden_seq[:, :-1, :]
                rho_t = tf.cast(rho_rail, tf.float32)
                mu_t  = tf.cast(mu_rail, tf.float32)
                l_rail = tf.reduce_mean(
                    tf.square(
                        tf.nn.relu(
                            tf.abs(h_t) + mu_t * tf.abs(h_t - h_tm1) - rho_t
                        )
                    )
                )

            # ── Total loss ────────────────────────────────────────────────────
            l_total = (
                l_seq
                + alpha * l_kd
                + lambda_mem   * l_mem
                + lambda_innov * l_innov
                + lambda_zsat  * l_zsat
                + lambda_rail  * l_rail
            )

        trainable_vars = final_qkeras_student.trainable_variables
        grads          = tape.gradient(l_total, trainable_vars)
        grads_clipped, _ = tf.clip_by_global_norm(grads, clipnorm)
        optimizer.apply_gradients(zip(grads_clipped, trainable_vars))

        mae = tf.reduce_mean(tf.abs(seq_output - ground_truth))
        return l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae

    @tf.function
    def dist_step(batch_x, batch_y):
        per_rep = strategy.run(per_replica_step, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, t, axis=None)
            for t in per_rep
        )

    return dist_step


def make_dist_memoq_val_final(
    strategy,
    final_qkeras_student,
    alpha,
    channel_scales,
    huber_delta,
    lambda_mem,
    lambda_innov,
    lambda_zsat,
    lambda_rail,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
):
    """
    Distributed validation step for Phase 3 hard 4-bit QKeras polish.
    No gradient tape. Same loss decomposition as train step.
    """
    enc_layer   = final_qkeras_student.get_layer("sencgru")
    dec_layer   = final_qkeras_student.get_layer("sdecgru")
    dense_layer = final_qkeras_student.get_layer("sdec_dense")

    channel_scales_t = tf.constant(channel_scales, dtype=tf.float32)
    lags             = [1, 2, 4, 8, 16, 32, 64]
    T_f              = tf.cast(seq_len, tf.float32)

    def per_replica_val(batch_x, batch_y):
        enc_inp      = batch_x["enc_input"]
        dec_inp      = batch_x["dec_input"]
        t_pred       = batch_x["tpred"]
        t_hidden     = batch_x["teacher_hidden"]
        ground_truth = batch_y

        enc_result = enc_layer(enc_inp, training=False)
        if isinstance(enc_result, (list, tuple)):
            enc_seq   = enc_result[0]
            enc_final = enc_result[1] if len(enc_result) > 1 else enc_seq[:, -1, :]
        else:
            enc_seq   = enc_result
            enc_final = enc_seq[:, -1, :]

        dec_result = dec_layer(dec_inp, initial_state=[enc_final], training=False)
        if isinstance(dec_result, (list, tuple)):
            dec_hidden_seq = dec_result[0]
            z_logit_seq    = dec_result[1] if len(dec_result) > 1 else None
        else:
            dec_hidden_seq = dec_result
            z_logit_seq    = None

        seq_output = dense_layer(dec_hidden_seq, training=False)

        diff_seq  = (seq_output - ground_truth) / (channel_scales_t + 1e-8)
        abs_diff  = tf.abs(diff_seq)
        huber_seq = tf.where(
            abs_diff <= huber_delta,
            0.5 * tf.square(diff_seq),
            huber_delta * (abs_diff - 0.5 * huber_delta),
        )
        l_seq = tf.reduce_mean(huber_seq)

        diff_kd  = (seq_output - t_pred) / (channel_scales_t + 1e-8)
        abs_kd   = tf.abs(diff_kd)
        huber_kd = tf.where(
            abs_kd <= huber_delta,
            0.5 * tf.square(diff_kd),
            huber_delta * (abs_kd - 0.5 * huber_delta),
        )
        l_kd = tf.reduce_mean(huber_kd)

        l_mem = tf.constant(0.0, dtype=tf.float32)
        if lambda_mem > 0.0:
            for lag in lags:
                if lag >= seq_len:
                    continue
                lag_f  = tf.cast(lag, tf.float32)
                w_ell  = (1.0 / tf.sqrt(lag_f)) * ((T_f - lag_f) / T_f)
                s_a    = dec_hidden_seq[:, lag:, :]
                s_b    = dec_hidden_seq[:, :-lag, :]
                t_a    = t_hidden[:, lag:, :]
                t_b    = t_hidden[:, :-lag, :]
                eps_cs = 1e-8
                s_dot  = tf.reduce_sum(s_a * s_b, axis=-1)
                s_na   = tf.norm(s_a, axis=-1)
                s_nb   = tf.norm(s_b, axis=-1)
                m_s    = tf.reduce_mean(s_dot / (s_na * s_nb + eps_cs))
                t_dot  = tf.reduce_sum(t_a * t_b, axis=-1)
                t_na   = tf.norm(t_a, axis=-1)
                t_nb   = tf.norm(t_b, axis=-1)
                m_t    = tf.reduce_mean(t_dot / (t_na * t_nb + eps_cs))
                l_mem  = l_mem + w_ell * tf.square(m_s - m_t)

        l_innov = tf.constant(0.0, dtype=tf.float32)
        if lambda_innov > 0.0:
            s_diff  = dec_hidden_seq[:, 1:, :] - dec_hidden_seq[:, :-1, :]
            t_diff  = t_hidden[:, 1:, :] - t_hidden[:, :-1, :]
            v_s     = tf.reduce_mean(tf.square(s_diff), axis=-1)
            v_t     = tf.reduce_mean(tf.square(t_diff), axis=-1)
            eps_inv = tf.cast(epsilon_innov, tf.float32)
            log_ratio = tf.math.log(
                (v_s + eps_inv) / (v_t + eps_inv + 1e-30)
            )
            l_innov = tf.reduce_mean(tf.square(log_ratio))

        l_zsat = tf.constant(0.0, dtype=tf.float32)
        if lambda_zsat > 0.0 and z_logit_seq is not None:
            l_zsat = tf.reduce_mean(
                tf.square(tf.nn.relu(tf.abs(z_logit_seq) - 3.0))
            )

        l_rail = tf.constant(0.0, dtype=tf.float32)
        if lambda_rail > 0.0:
            h_t   = dec_hidden_seq[:, 1:, :]
            h_tm1 = dec_hidden_seq[:, :-1, :]
            rho_t = tf.cast(rho_rail, tf.float32)
            mu_t  = tf.cast(mu_rail, tf.float32)
            l_rail = tf.reduce_mean(
                tf.square(
                    tf.nn.relu(
                        tf.abs(h_t) + mu_t * tf.abs(h_t - h_tm1) - rho_t
                    )
                )
            )

        l_total = (
            l_seq
            + alpha * l_kd
            + lambda_mem   * l_mem
            + lambda_innov * l_innov
            + lambda_zsat  * l_zsat
            + lambda_rail  * l_rail
        )

        mae = tf.reduce_mean(tf.abs(seq_output - ground_truth))
        return l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae

    @tf.function
    def dist_val_step(batch_x, batch_y):
        per_rep = strategy.run(per_replica_val, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, t, axis=None)
            for t in per_rep
        )

    return dist_val_step

def set_phase2_quantizers(args, enc_cell, dec_cell, stage):
    """
    Assign quantizers to SplitGateGRUCell gate variables based on the
    MemoQ curriculum stage.

    Stage P2A : quantize h gate only (W_h, U_h)
    Stage P2B : quantize h + r gates
    Stage P2C : quantize h + r + z gates
    Stage P3  : quantize h + r + z gates + state (all hard 4-bit)

    One shared quantized_bits instance per 4-bit setting is intentional:
    the STE is stateless so sharing is safe and avoids unnecessary objects.
    """
    q4 = quantized_bits(args.bits_kernel, 0, 1, alpha=1.0)

    if stage == "P2A":
        q_h = q4
        q_r = None
        q_z = None
        q_s = None
    elif stage == "P2B":
        q_h = q4
        q_r = q4
        q_z = None
        q_s = None
    elif stage == "P2C":
        q_h = q4
        q_r = q4
        q_z = q4
        q_s = None
    elif stage == "P3":
        q_h = q4
        q_r = q4
        q_z = q4
        q_s = quantized_bits(args.bits_state, 0, 1, alpha=1.0)
    else:
        raise ValueError(f"Unknown stage: {stage!r}. Expected one of P2A, P2B, P2C, P3.")

    for cell in [enc_cell, dec_cell]:
        cell.quantizer_h     = q_h
        cell.quantizer_r     = q_r
        cell.quantizer_z     = q_z
        cell.quantizer_state = q_s

# ==============================================================================
# Main MemoQ training loop
# ==============================================================================

def training_loop_memoq(
    strategy,
    float_student,
    final_qkeras_student,
    enc_cell_p2,
    dec_cell_p2,
    phase2_model,
    args,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    channel_scales,
    epsilon_innov,
    job_dir,
    pf,
):
    p1_ckpt   = os.path.join(job_dir, "phase1_best.weights.h5")
    p2a_ckpt  = os.path.join(job_dir, "stage2a_best.weights.h5")
    p2b_ckpt  = os.path.join(job_dir, "stage2b_best.weights.h5")
    p2c_ckpt  = os.path.join(job_dir, "stage2c_best.weights.h5")
    p3_ckpt   = os.path.join(job_dir, "student_best.weights.h5")
    resume_path = os.path.join(job_dir, "resume_state.json")

    history = {
        "total":     [], "seq":       [], "kd":        [],
        "mem":       [], "innov":     [], "zsat":      [], "rail":      [],
        "val_total": [], "val_seq":   [], "val_kd":    [],
        "val_mem":   [], "val_innov": [], "val_zsat":  [], "val_rail":  [],
        "val_mae":   [], "phase":     [],
    }

    total_planned = (
        args.memoq_warmup_epochs
        + args.memoq_stage2a_epochs
        + args.memoq_stage2b_epochs
        + args.memoq_stage2c_epochs
        + args.memoq_stage3_epochs
    )
    global_epoch = 0
    nan_warn_threshold = max(1, int(train_steps * 0.10))

    resume_stage = "P1"
    resume_epoch_in_stage = 0
    best_vals = {
        "P1": float("inf"), "P2A": float("inf"), "P2B": float("inf"),
        "P2C": float("inf"), "P3": float("inf"),
    }
    patience_cts = {"P1": 0, "P2A": 0, "P2B": 0, "P2C": 0, "P3": 0}

    if args.resume and os.path.exists(resume_path):
        pf(f"[RESUME] {resume_path}")
        with open(resume_path) as f:
            rs = json.load(f)
        resume_stage          = rs.get("stage", "P1")
        resume_epoch_in_stage = int(rs.get("epoch_in_stage", 0))
        best_vals.update({k: float(v) for k, v in rs.get("best_vals", {}).items()})
        patience_cts.update({k: int(v) for k, v in rs.get("patience_cts", {}).items()})
        if "history" in rs:
            for key in history:
                if key in rs["history"]:
                    history[key] = list(rs["history"][key])
        pf(f"[RESUME] stage={resume_stage} epoch_in_stage={resume_epoch_in_stage}")
        stage_order = ["P1", "P2A", "P2B", "P2C", "P3"]
        stage_idx = stage_order.index(resume_stage)
        if stage_idx >= 1 and os.path.exists(p1_ckpt):
            float_student.load_weights(p1_ckpt)
            pf("[RESUME] Loaded P1 weights for float_student")
        if stage_idx >= 2:
            ckpt = p2a_ckpt if stage_idx == 2 else (p2b_ckpt if stage_idx == 3 else p2c_ckpt)
            if os.path.exists(ckpt):
                phase2_model.load_weights(ckpt)
                pf(f"[RESUME] Loaded P2 weights for phase2_model from {ckpt}")
        if stage_idx >= 4 and os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf("[RESUME] Loaded P3 weights for final_qkeras_student")
        sys.stdout.flush()

    csv_path = os.path.join(job_dir, "training_history.csv")
    if not args.resume or (resume_stage == "P1" and resume_epoch_in_stage == 0):
        with open(csv_path, "w") as f:
            f.write(
                "epoch,phase,total,seq,kd,mem,innov,zsat,rail,"
                "val_total,val_seq,val_kd,val_mem,val_innov,val_zsat,val_rail,"
                "val_mae,lr\n"
            )

    stage_order_list = ["P1", "P2A", "P2B", "P2C", "P3"]

    def save_resume(stage_tag, ep_in_stage):
        state = {
            "stage":          stage_tag,
            "epoch_in_stage": ep_in_stage,
            "best_vals":      {k: float(v) for k, v in best_vals.items()},
            "patience_cts":   {k: int(v)   for k, v in patience_cts.items()},
            "history":        {
                k: ([float(x) for x in v] if k != "phase" else list(v))
                for k, v in history.items()
            },
        }
        with open(resume_path, "w") as f:
            json.dump(state, f, indent=2)

    def should_run(stage_tag):
        if not args.resume:
            return True
        return stage_order_list.index(stage_tag) >= stage_order_list.index(resume_stage)

    def start_ep(stage_tag):
        if args.resume and stage_tag == resume_stage:
            return resume_epoch_in_stage
        return 0

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Float warm-up
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P1"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 1 — Float warm-up ({args.memoq_warmup_epochs} epochs)")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p1 = args.effective_lr
        opt_p1 = keras.optimizers.Adam(learning_rate=lr_p1)
        sched_p1 = ReduceLROnPlateau(
            opt_p1, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p1.reset(lr_p1)

        dist_train_p1 = make_dist_phase1_train(
            strategy, float_student, opt_p1, args.alpha, channel_scales, args.memoq_huber_delta
        )
        dist_val_p1 = make_dist_phase1_val(
            strategy, float_student, args.alpha, channel_scales, args.memoq_huber_delta
        )

        for ep_in_phase in range(start_ep("P1"), args.memoq_warmup_epochs):
            history, best_vals["P1"], patience_cts["P1"], early_stop = run_epoch(
                phase_tag="P1",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p1,
                dist_val_step_fn=dist_val_p1,
                lr_scheduler=sched_p1,
                effective_warmup_epochs=args.effective_warmup_epochs,
                effective_lr=lr_p1,
                history=history,
                best_val=best_vals["P1"],
                patience_ct=patience_cts["P1"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p1_ckpt,
                model_to_save=float_student,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P1", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p1_ckpt):
            float_student.load_weights(p1_ckpt)
            pf(f"[P1] Best weights loaded: {p1_ckpt}")
            sys.stdout.flush()

    # ── Transfer Phase 1 float weights into Phase 2 split-gate model ──────────
    if should_run("P2A"):
        pf("[P1->P2] Transferring float weights to split-gate model...")
        transfer_float_to_phase2(float_student, phase2_model, enc_cell_p2, dec_cell_p2, pf)
        sys.stdout.flush()



    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2A — Candidate gate (h) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2A"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2A — Candidate gate h 4-bit ({args.memoq_stage2a_epochs} epochs)")
        pf("  Quantising W_h, U_h only. z and r gates float.")
        if should_run("P2A"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2A")
            pf("=" * 60)
        sys.stdout.flush()

        qbits_h = quantized_bits(args.bits_kernel, 0, 1, alpha=1.0)
        enc_cell_p2.quantizer_h = qbits_h
        dec_cell_p2.quantizer_h = qbits_h
        enc_cell_p2.quantizer_z = None
        dec_cell_p2.quantizer_z = None
        enc_cell_p2.quantizer_r = None
        dec_cell_p2.quantizer_r = None

        lr_p2a = args.effective_lr * 0.5
        opt_p2a = keras.optimizers.Adam(learning_rate=lr_p2a)
        sched_p2a = ReduceLROnPlateau(
            opt_p2a, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2a.reset(lr_p2a)

        innov_active = False
        ep2a_start = start_ep("P2A")

        for ep_in_phase in range(ep2a_start, args.memoq_stage2a_epochs):
            if ep_in_phase >= args.memoq_innov_burnin:
                innov_active = True

            lambda_i_2a = args.memoq_lambda_innov if innov_active else 0.0

            dist_train_p2a = make_dist_memoq_train(
                strategy, phase2_model, opt_p2a,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.01, lambda_i_2a, 0.0, 0.0005,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=innov_active,
                use_zsat=False, use_rail=True,
                has_z_logit=True, clipnorm=1.0,
            )
            dist_val_p2a = make_dist_memoq_val(
                strategy, phase2_model,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.01, lambda_i_2a, 0.0, 0.0005,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=innov_active,
                use_zsat=False, use_rail=True,
                has_z_logit=True,
            )

            history, best_vals["P2A"], patience_cts["P2A"], early_stop = run_epoch(
                phase_tag="P2A",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2a,
                dist_val_step_fn=dist_val_p2a,
                lr_scheduler=sched_p2a,
                effective_warmup_epochs=0,
                effective_lr=lr_p2a,
                history=history,
                best_val=best_vals["P2A"],
                patience_ct=patience_cts["P2A"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2a_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2A", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2a_ckpt):
            phase2_model.load_weights(p2a_ckpt)
            pf(f"[P2A] Loaded best weights from {p2a_ckpt}")
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2B — Reset gate (r) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2B"):
        if should_run("P2B"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2B")
            pf("=" * 60)
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2B — Reset gate r 4-bit ({args.memoq_stage2b_epochs} epochs)")
        pf("  Quantising W_r, U_r. z gate float. h gate already 4-bit.")
        pf("=" * 60)
        sys.stdout.flush()

        qbits_r = quantized_bits(args.bits_recurrent, 0, 1, alpha=1.0)
        enc_cell_p2.quantizer_r = qbits_r
        dec_cell_p2.quantizer_r = qbits_r

        lr_p2b = args.effective_lr * 0.3
        opt_p2b = keras.optimizers.Adam(learning_rate=lr_p2b)
        sched_p2b = ReduceLROnPlateau(
            opt_p2b, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2b.reset(lr_p2b)

        ep2b_start = start_ep("P2B")

        for ep_in_phase in range(ep2b_start, args.memoq_stage2b_epochs):
            dist_train_p2b = make_dist_memoq_train(
                strategy, phase2_model, opt_p2b,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.03, 0.005, 0.0, 0.001,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=True,
                use_zsat=False, use_rail=True,
                has_z_logit=True, clipnorm=1.0,
            )
            dist_val_p2b = make_dist_memoq_val(
                strategy, phase2_model,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.03, 0.005, 0.0, 0.001,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=True,
                use_zsat=False, use_rail=True,
                has_z_logit=True,
            )

            history, best_vals["P2B"], patience_cts["P2B"], early_stop = run_epoch(
                phase_tag="P2B",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2b,
                dist_val_step_fn=dist_val_p2b,
                lr_scheduler=sched_p2b,
                effective_warmup_epochs=0,
                effective_lr=lr_p2b,
                history=history,
                best_val=best_vals["P2B"],
                patience_ct=patience_cts["P2B"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2b_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2B", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2b_ckpt):
            phase2_model.load_weights(p2b_ckpt)
            pf(f"[P2B] Loaded best weights from {p2b_ckpt}")
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2C — Update gate (z) 4-bit quantization
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P2C"):
        if should_run("P2C"):
            set_phase2_quantizers(args, enc_cell_p2, dec_cell_p2, "P2C")
            pf("=" * 60)
        pf("=" * 60)
        pf(f"MEMOQ PHASE 2C — Update gate z 4-bit ({args.memoq_stage2c_epochs} epochs)")
        pf("  All gates now 4-bit. Activating L_zsat.")
        pf("=" * 60)
        sys.stdout.flush()

        qbits_z = quantized_bits(args.bits_kernel, 0, 1, alpha=1.0)
        enc_cell_p2.quantizer_z = qbits_z
        dec_cell_p2.quantizer_z = qbits_z

        lr_p2c = args.effective_lr * 0.2
        opt_p2c = keras.optimizers.Adam(learning_rate=lr_p2c)
        sched_p2c = ReduceLROnPlateau(
            opt_p2c, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p2c.reset(lr_p2c)

        ep2c_start = start_ep("P2C")

        for ep_in_phase in range(ep2c_start, args.memoq_stage2c_epochs):
            dist_train_p2c = make_dist_memoq_train(
                strategy, phase2_model, opt_p2c,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.05, 0.005, 0.002, 0.001,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=True,
                use_zsat=True, use_rail=True,
                has_z_logit=True, clipnorm=1.0,
            )
            dist_val_p2c = make_dist_memoq_val(
                strategy, phase2_model,
                args.alpha, channel_scales, args.memoq_huber_delta,
                0.05, 0.005, 0.002, 0.001,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                use_mem=True, use_innov=True,
                use_zsat=True, use_rail=True,
                has_z_logit=True,
            )

            history, best_vals["P2C"], patience_cts["P2C"], early_stop = run_epoch(
                phase_tag="P2C",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p2c,
                dist_val_step_fn=dist_val_p2c,
                lr_scheduler=sched_p2c,
                effective_warmup_epochs=0,
                effective_lr=lr_p2c,
                history=history,
                best_val=best_vals["P2C"],
                patience_ct=patience_cts["P2C"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p2c_ckpt,
                model_to_save=phase2_model,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P2C", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p2c_ckpt):
            phase2_model.load_weights(p2c_ckpt)
            pf(f"[P2C] Loaded best weights from {p2c_ckpt}")
            sys.stdout.flush()

    # ══════════════════════════════════════════════════════════════════════════
    # Export Phase 2 split-gate weights into final hard QKeras student
    # ══════════════════════════════════════════════════════════════════════════
    pf("=" * 60)
    pf("[EXPORT] Packing split gate weights into standard QKeras QGRU format...")
    pf("=" * 60)
    sys.stdout.flush()

    transfer_splitgate_to_qkeras(enc_cell_p2, dec_cell_p2, phase2_model, final_qkeras_student, pf)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Hard 4-bit QKeras polish
    # ══════════════════════════════════════════════════════════════════════════
    if should_run("P3"):
        pf("=" * 60)
        pf(f"MEMOQ PHASE 3 — Hard 4-bit QKeras polish ({args.memoq_stage3_epochs} epochs)")
        pf("  All QGRU/QDense weights hard 4-bit. LR=5e-6. clipnorm=0.5.")
        pf("  Loss: L_seq+0.5*L_KD+0.03*L_mem+0.002*L_innov+0.001*L_zsat+0.0005*L_rail")
        pf("=" * 60)
        sys.stdout.flush()

        lr_p3 = min(args.effective_lr * args.memoq_phase3_lr_factor, args.memoq_phase3_lr)
        opt_p3 = keras.optimizers.Adam(learning_rate=lr_p3)
        sched_p3 = ReduceLROnPlateau(
            opt_p3, args.lr_factor, args.effective_lr_patience, args.lr_min, args.min_delta
        )
        sched_p3.reset(lr_p3)

        ep3_start = start_ep("P3")

        for ep_in_phase in range(ep3_start, args.memoq_stage3_epochs):
            dist_train_p3 = make_dist_memoq_train_final(
                strategy, final_qkeras_student, opt_p3,
                0.5, channel_scales, args.memoq_huber_delta,
                0.03, 0.002, 0.0, 0.0005,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
                clipnorm=0.5,
            )

            dist_val_p3 = make_dist_memoq_val_final(
                strategy, final_qkeras_student,
                0.5, channel_scales, args.memoq_huber_delta,
                0.03, 0.002, 0.0, 0.0005,
                epsilon_innov, args.seq_len,
                args.memoq_rho_rail, args.memoq_mu_rail,
            )

            history, best_vals["P3"], patience_cts["P3"], early_stop = run_epoch(
                phase_tag="P3",
                epoch=global_epoch,
                total_epochs=total_planned,
                dist_train_dataset=dist_train_dataset,
                dist_val_dataset=dist_val_dataset,
                train_steps=train_steps,
                val_steps=val_steps,
                dist_train_step_fn=dist_train_p3,
                dist_val_step_fn=dist_val_p3,
                lr_scheduler=sched_p3,
                effective_warmup_epochs=0,
                effective_lr=lr_p3,
                history=history,
                best_val=best_vals["P3"],
                patience_ct=patience_cts["P3"],
                patience_max=args.patience,
                min_delta=args.min_delta,
                best_ckpt_path=p3_ckpt,
                model_to_save=final_qkeras_student,
                csv_path=csv_path,
                log_interval=args.log_interval,
                nan_warn_threshold=nan_warn_threshold,
                pf=pf,
                epoch_in_phase=ep_in_phase,
            )
            global_epoch += 1
            save_resume("P3", ep_in_phase + 1)
            if early_stop:
                break

        if os.path.exists(p3_ckpt):
            final_qkeras_student.load_weights(p3_ckpt)
            pf(f"[P3] Loaded best phase3 weights from {p3_ckpt}")
            sys.stdout.flush()

    return history, best_vals.get(
        "P3", best_vals.get(
            "P2C", best_vals.get(
                "P2B", best_vals.get(
                    "P2A", best_vals.get("P1", float("inf"))
                )
            )
        )
    )

# ==============================================================================
# transfer_splitgate_to_qkeras:
# Pack enc_cell_p2 and dec_cell_p2 split gate variables into the
# standard Keras/QKeras GRU packed layout and transfer to final_qkeras_student.
#
# Keras/QKeras GRU packed kernel shape: (input_dim, 3*units)
#   columns [0:units]     -> z gate (update)
#   columns [units:2H]    -> r gate (reset)
#   columns [2H:3H]       -> h gate (candidate)
#
# Keras/QKeras GRU packed recurrent_kernel shape: (units, 3*units)
# same column ordering.
#
# Keras/QKeras GRU packed bias shape: (2, 3*units) for reset_after=True
#   bias[0] = input bias [z|r|h]
#   bias[1] = recurrent bias [z|r|h]
# ==============================================================================

def transfer_splitgate_to_qkeras(enc_cell, dec_cell, phase2_model, final_qkeras_student, pf):
    """
    Pack split-gate cell weights from SplitGateGRUCell into standard
    Keras/QKeras GRU packed format [z | r | h] and load into final_qkeras_student.

    SplitGateGRUCell variables:
        W_z, W_r, W_h   (input_dim, units)
        U_z, U_r, U_h   (units,     units)
        b_z, b_r, b_h   (2,         units)  row 0 = inp, row 1 = rec

    Keras/QKeras reset_after=True GRU expects:
        kernel           (input_dim, 3*units)   concat [W_z | W_r | W_h]
        recurrent_kernel (units,     3*units)   concat [U_z | U_r | U_h]
        bias             (2,         3*units)   concat [b_z | b_r | b_h] along axis=1
    """
    pf("[EXPORT] transfer_splitgate_to_qkeras: packing [W_z|W_r|W_h] / [U_z|U_r|U_h] / [b_z|b_r|b_h]...")

    def pack_and_load(cell, layer_name, target_model):
        W_z = cell.W_z.numpy()
        W_r = cell.W_r.numpy()
        W_h = cell.W_h.numpy()

        U_z = cell.U_z.numpy()
        U_r = cell.U_r.numpy()
        U_h = cell.U_h.numpy()

        # b_z shape (2, units): row 0 = input bias, row 1 = recurrent bias
        b_z = cell.b_z.numpy()   # (2, units)
        b_r = cell.b_r.numpy()   # (2, units)
        b_h = cell.b_h.numpy()   # (2, units)

        packed_kernel    = np.concatenate([W_z, W_r, W_h], axis=1)   # (input_dim, 3*units)
        packed_recurrent = np.concatenate([U_z, U_r, U_h], axis=1)   # (units,     3*units)
        # Concatenate along axis=1 (the units dimension) to get (2, 3*units)
        packed_bias      = np.concatenate([b_z, b_r, b_h], axis=1)   # (2,         3*units)

        try:
            target_layer = target_model.get_layer(layer_name)
        except ValueError:
            pf(f"[EXPORT]   SKIP {layer_name} — not found in target model")
            return

        q_weights = target_layer.get_weights()
        n_w = len(q_weights)

        if n_w == 3:
            target_layer.set_weights([packed_kernel, packed_recurrent, packed_bias])
            pf(
                f"[EXPORT]   {layer_name}: kernel={packed_kernel.shape}  "
                f"recurrent={packed_recurrent.shape}  bias={packed_bias.shape} (3-weight mode)"
            )
        elif n_w == 4:
            # Some QKeras builds store inp and rec biases as two separate (3*units,) vectors
            target_layer.set_weights([
                packed_kernel,
                packed_recurrent,
                packed_bias[0],   # (3*units,) input bias
                packed_bias[1],   # (3*units,) recurrent bias
            ])
            pf(
                f"[EXPORT]   {layer_name}: kernel={packed_kernel.shape}  "
                f"recurrent={packed_recurrent.shape}  "
                f"bias_inp={packed_bias[0].shape}  bias_rec={packed_bias[1].shape} (4-weight mode)"
            )
        else:
            pf(
                f"[EXPORT]   WARNING: {layer_name} has unexpected {n_w} weight tensors, "
                f"attempting 3-weight convention."
            )
            target_layer.set_weights([packed_kernel, packed_recurrent, packed_bias])

    pack_and_load(enc_cell, "sencgru", final_qkeras_student)
    pack_and_load(dec_cell, "sdecgru", final_qkeras_student)

    pf("[EXPORT] Dense head (sdec_dense): transferring from phase2_model...")
    try:
        src_dense = phase2_model.get_layer("sdec_dense")
        dst_dense = final_qkeras_student.get_layer("sdec_dense")
        src_w = src_dense.get_weights()
        dst_w = dst_dense.get_weights()
        if len(src_w) == len(dst_w) and all(
            s.shape == d.shape for s, d in zip(src_w, dst_w)
        ):
            dst_dense.set_weights(src_w)
            pf(f"[EXPORT]   sdec_dense: OK {len(src_w)} tensors transferred")
        else:
            pf(
                f"[EXPORT]   sdec_dense: shape mismatch "
                f"src={[w.shape for w in src_w]} dst={[w.shape for w in dst_w]} — SKIPPED"
            )
    except Exception as exc:
        pf(f"[EXPORT]   sdec_dense: FAILED ({exc}) — skipped")

    sys.stdout.flush()

# ==============================================================================
# MemoQGRUCell — custom training-time GRU cell with split gate variables.
#
# Variables:
#   W_z, W_r, W_h  : input kernels, shape (input_dim, units)
#   U_z, U_r, U_h  : recurrent kernels, shape (units, units)
#   b_z_inp, b_r_inp, b_h_inp : input biases, shape (units,)
#   b_z_rec, b_r_rec, b_h_rec : recurrent biases, shape (units,) [reset_after=True]
#
# Quantizers:
#   quantizer_z, quantizer_r, quantizer_h : applied per gate independently.
#   None means float (no quantization for that gate).
#   quantizer_state : applied to hidden state h_t. None means float.
#
# Outputs during training:
#   (h_t, [h_t, z_logit_t])
#   z_logit_t : pre-sigmoid update gate logit for L_zsat computation.
#   h_t is the hidden state (quantized if quantizer_state is set).
#
# At inference (in final QKeras model): this cell is NOT used.
# ==============================================================================

class MemoQGRUCell(keras.layers.Layer):
    def __init__(
        self,
        units,
        input_dim,
        quantizer_z=None,
        quantizer_r=None,
        quantizer_h=None,
        quantizer_state=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self._quantizer_z = quantizer_z
        self._quantizer_r = quantizer_r
        self._quantizer_h = quantizer_h
        self._quantizer_state = quantizer_state
        # state_size must be a list so Keras RNN allocates two state tensors:
        #   states[0] = h_prev (B, units)
        #   states[1] = z_logit_prev (B, units)  -- carried for extraction only
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        d = self.input_dim
        H = self.units

        init  = keras.initializers.GlorotUniform()
        orth  = keras.initializers.Orthogonal()
        zeros = keras.initializers.Zeros()

        self.W_z     = self.add_weight(name="W_z",     shape=(d, H), initializer=init,  trainable=True)
        self.W_r     = self.add_weight(name="W_r",     shape=(d, H), initializer=init,  trainable=True)
        self.W_h     = self.add_weight(name="W_h",     shape=(d, H), initializer=init,  trainable=True)
        self.U_z     = self.add_weight(name="U_z",     shape=(H, H), initializer=orth,  trainable=True)
        self.U_r     = self.add_weight(name="U_r",     shape=(H, H), initializer=orth,  trainable=True)
        self.U_h     = self.add_weight(name="U_h",     shape=(H, H), initializer=orth,  trainable=True)
        self.b_z_inp = self.add_weight(name="b_z_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_r_inp = self.add_weight(name="b_r_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_h_inp = self.add_weight(name="b_h_inp", shape=(H,),   initializer=zeros, trainable=True)
        self.b_z_rec = self.add_weight(name="b_z_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.b_r_rec = self.add_weight(name="b_r_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.b_h_rec = self.add_weight(name="b_h_rec", shape=(H,),   initializer=zeros, trainable=True)
        self.built = True

    def _apply_quantizer(self, q, w):
        if q is None:
            return w
        return q(w)

    def call(self, inputs, states):
        h_prev     = states[0]   # (B, units)
        # states[1] is z_logit_prev — carried but not used in forward pass

        W_z = self._apply_quantizer(self._quantizer_z, self.W_z)
        W_r = self._apply_quantizer(self._quantizer_r, self.W_r)
        W_h = self._apply_quantizer(self._quantizer_h, self.W_h)
        U_z = self._apply_quantizer(self._quantizer_z, self.U_z)
        U_r = self._apply_quantizer(self._quantizer_r, self.U_r)
        U_h = self._apply_quantizer(self._quantizer_h, self.U_h)

        z_logit = (
            tf.matmul(inputs, W_z) + self.b_z_inp
            + tf.matmul(h_prev, U_z) + self.b_z_rec
        )
        r_logit = (
            tf.matmul(inputs, W_r) + self.b_r_inp
            + tf.matmul(h_prev, U_r) + self.b_r_rec
        )

        z = tf.sigmoid(z_logit)
        r = tf.sigmoid(r_logit)

        h_candidate = tf.tanh(
            tf.matmul(inputs, W_h) + self.b_h_inp
            + r * (tf.matmul(h_prev, U_h) + self.b_h_rec)
        )

        h_prev_q = h_prev
        if self._quantizer_state is not None:
            h_prev_q = self._quantizer_state(h_prev_q)

        h_t = z * h_prev_q + (1.0 - z) * h_candidate

        if self._quantizer_state is not None:
            h_t = self._quantizer_state(h_t)

        # Return h_t as output; carry [h_t, z_logit] as states
        return h_t, [h_t, z_logit]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None and inputs is not None:
            batch_size = tf.shape(inputs)[0]
        if dtype is None:
            dtype = tf.float32
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype),
        ]

    @property
    def quantizer_z(self):
        return self._quantizer_z

    @quantizer_z.setter
    def quantizer_z(self, q):
        self._quantizer_z = q

    @property
    def quantizer_r(self):
        return self._quantizer_r

    @quantizer_r.setter
    def quantizer_r(self, q):
        self._quantizer_r = q

    @property
    def quantizer_h(self):
        return self._quantizer_h

    @quantizer_h.setter
    def quantizer_h(self, q):
        self._quantizer_h = q

    @property
    def quantizer_state(self):
        return self._quantizer_state

    @quantizer_state.setter
    def quantizer_state(self, q):
        self._quantizer_state = q

# ==============================================================================
# build_phase2_model:
# Constructs the training-time phase2 model using MemoQGRUCell instances.
# Returns (model, enc_cell, dec_cell).
# The model outputs (seq_output, dec_h_seq, dec_z_logit_seq).
#   seq_output     : (batch, T, n_out) — student predictions for KD loss
#   dec_h_seq      : (batch, T, units) — decoder hidden trajectory for mem/innov/rail
#   dec_z_logit_seq: (batch, T, units) — decoder z gate logits for L_zsat
# ==============================================================================

class MemoQRNNUnroll(keras.layers.Layer):
    """
    Manually unrolls a MemoQGRUCell over a sequence and exposes
    both the hidden trajectory and the z_logit trajectory.

    Inputs : (batch, T, input_dim)
    Outputs: (hidden_seq (B,T,units), z_logit_seq (B,T,units), final_h (B,units))
    """
    def __init__(self, cell, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.cell = cell

    def call(self, inputs, initial_state=None, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len    = tf.shape(inputs)[1]
        units      = self.cell.units

        if initial_state is None:
            h     = tf.zeros((batch_size, units), dtype=tf.float32)
            z_log = tf.zeros((batch_size, units), dtype=tf.float32)
        else:
            h     = initial_state[0]
            z_log = initial_state[1]

        h_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                               element_shape=(None, units))
        z_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False,
                               element_shape=(None, units))

        for t in tf.range(seq_len):
            x_t = inputs[:, t, :]
            h_out, new_states = self.cell(x_t, [h, z_log], training=training)
            h     = new_states[0]
            z_log = new_states[1]
            h_ta  = h_ta.write(t, h_out)
            z_ta  = z_ta.write(t, z_log)

        hidden_seq   = tf.transpose(h_ta.stack(),  [1, 0, 2])
        z_logit_seq  = tf.transpose(z_ta.stack(),  [1, 0, 2])
        final_h      = hidden_seq[:, -1, :]
        return hidden_seq, z_logit_seq, final_h

    def get_config(self):
        cfg = super().get_config()
        return cfg


def build_phase2_model(seq_len, n_out, student_units, input_dim=1):
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

    enc_unroll = MemoQRNNUnroll(enc_cell, name="sencgru_unroll")
    dec_unroll = MemoQRNNUnroll(dec_cell, name="sdecgru_unroll")

    enc_inputs = keras.layers.Input(shape=(None, input_dim), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(None, input_dim), name="sdec_input")

    enc_hidden_seq, enc_z_logit_seq, enc_final_h = enc_unroll(enc_inputs)

    enc_initial_h   = enc_final_h
    enc_initial_z = keras.layers.Lambda(
        lambda x: tf.zeros_like(x),
        name="enc_initial_z_zero",
    )(enc_initial_h)

    dec_hidden_seq, dec_z_logit_seq, _ = dec_unroll(
        dec_inputs,
        initial_state=[enc_initial_h, enc_initial_z],
    )

    seq_output = keras.layers.Dense(n_out, activation="linear", name="sdec_dense")(dec_hidden_seq)

    model = keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=[seq_output, dec_hidden_seq, dec_z_logit_seq],
        name="memoq_phase2_model",
    )

    return model, enc_cell, dec_cell

# ==============================================================================
# build_final_qkeras_student:
# Standard hard 4-bit QKeras QGRU/QDense student with identical layer names
# to what vanilla_kd uses. Output is (batch, T, n_out) only.
# Layer names: senc_input, sdec_input, sencgru, sdecgru, sdec_dense.
# ==============================================================================

def build_final_qkeras_student(
    seq_len, n_out, student_units,
    bits_kernel, bits_recurrent, bits_bias,
    bits_activation, bits_state,
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

    enc_inputs = keras.layers.Input(shape=(None, 1), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(None, 1), name="sdec_input")

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

    return keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="memoq_final_qkeras_student",
    )


# ==============================================================================
# MemoQ auxiliary losses.
# All losses are tf.function-traceable.
# ==============================================================================

def loss_mem(h_student, h_teacher, seq_len):
    """
    Memory-kernel distillation loss L_mem.

    h_student : (batch, T, Hs) — student decoder hidden sequence
    h_teacher : (batch, T, Ht) — teacher decoder hidden sequence
    seq_len   : T (integer, used for lag weight computation)

    Returns scalar tf.Tensor.
    """
    lags = [1, 2, 4, 8, 16, 32, 64]
    T = tf.cast(seq_len, tf.float32)

    def cosine_mean(h, lag):
        a = h[:, lag:, :]
        b = h[:, :-lag, :]
        dot = tf.reduce_sum(a * b, axis=-1)
        norm_a = tf.norm(a, axis=-1) + 1e-8
        norm_b = tf.norm(b, axis=-1) + 1e-8
        cos = dot / (norm_a * norm_b)
        return tf.reduce_mean(cos)

    total = tf.constant(0.0, dtype=tf.float32)
    weight_sum = tf.constant(0.0, dtype=tf.float32)

    for lag in lags:
        lag_f = tf.cast(lag, tf.float32)
        w = (1.0 / tf.sqrt(lag_f)) * ((T - lag_f) / T)
        M_s = cosine_mean(h_student, lag)
        M_t = cosine_mean(h_teacher, lag)
        total = total + w * tf.square(M_s - M_t)
        weight_sum = weight_sum + w

    return total / (weight_sum + 1e-8)


def compute_innovation_profile(h):
    """
    Per-timestep mean squared temporal innovation of hidden sequence.

    h : (batch, T, H)
    Returns v : (T-1,) tf.Tensor, mean over batch and H dimensions.
    """
    diff = h[:, 1:, :] - h[:, :-1, :]
    v = tf.reduce_mean(tf.square(diff), axis=[0, 2])
    return v


def loss_innov(h_student, h_teacher, epsilon_innov):
    """
    Temporal innovation-profile matching loss L_innov.

    Uses log-ratio form with calibrated epsilon to avoid log-domain spikes.

    h_student     : (batch, T, Hs)
    h_teacher     : (batch, T, Ht)
    epsilon_innov : float scalar, = 0.1 * median(v_t(teacher)), precomputed

    Returns scalar tf.Tensor.
    """
    v_s = compute_innovation_profile(h_student)
    v_t = compute_innovation_profile(h_teacher)

    eps = tf.cast(epsilon_innov, tf.float32)
    log_ratio = tf.math.log((v_s + eps) / (v_t + eps))
    return tf.reduce_mean(tf.square(log_ratio))


def loss_zsat_logit(z_logit_seq, logit_threshold=3.0):
    """
    Update-gate saturation barrier loss using logit form L_zsat.

    z_logit_seq : (batch, T, units) — pre-sigmoid update gate logit
    logit_threshold : float, default 3.0 (sigmoid(3) ≈ 0.95)

    L_zsat = mean(ReLU(|z_logit| - threshold)^2)

    Returns scalar tf.Tensor.
    """
    threshold = tf.cast(logit_threshold, tf.float32)
    excess = tf.nn.relu(tf.abs(z_logit_seq) - threshold)
    return tf.reduce_mean(tf.square(excess))


def loss_railpred(h_student, rho_rail=0.88, mu_rail=0.9):
    """
    Predictive rail-margin regularization L_railpred.

    Penalises hidden states predicted to hit the quantization boundary
    before saturation (forward-looking).

    h_student : (batch, T, H) — student decoder hidden sequence
    rho_rail  : float, boundary threshold (default 0.88 for tanh in [-1,1])
    mu_rail   : float, predictive step weight (default 0.9)

    L_railpred = mean(ReLU(|h_t| + mu * |h_t - h_{t-1}| - rho)^2)

    Returns scalar tf.Tensor.
    """
    rho = tf.cast(rho_rail, tf.float32)
    mu  = tf.cast(mu_rail,  tf.float32)

    h_curr = h_student[:, 1:, :]
    h_prev = h_student[:, :-1, :]

    predicted_mag = tf.abs(h_curr) + mu * tf.abs(h_curr - h_prev)
    excess = tf.nn.relu(predicted_mag - rho)
    return tf.reduce_mean(tf.square(excess))


def channel_normalised_huber_memoq(y_true, y_pred, channel_scales, huber_delta):
    """
    Channel-normalised Huber loss.

    y_true, y_pred  : (batch, T, C)
    channel_scales  : (C,) tf.constant
    huber_delta     : float

    Returns scalar tf.Tensor.
    """
    residual = (y_pred - y_true) / channel_scales
    abs_res = tf.abs(residual)
    delta = tf.cast(huber_delta, tf.float32)
    huber = tf.where(
        abs_res <= delta,
        0.5 * tf.square(residual),
        delta * (abs_res - 0.5 * delta),
    )
    return tf.reduce_mean(huber)


# ==============================================================================
# make_dist_memoq_train:
# Factory that returns a distributed training step function for
# phase2 model. The phase2 model outputs (seq_out, dec_h_seq, dec_z_logit_seq).
# Teacher hidden cache is passed through the dataset as batch_x["teacher_hidden"].
# ==============================================================================

def make_dist_memoq_train(
    strategy,
    phase2_model,
    optimizer,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    use_mem,
    use_innov,
    use_zsat,
    use_rail,
    has_z_logit,
    clipnorm,
):
    alpha_f      = tf.cast(alpha,        tf.float32)
    lambda_m_f   = tf.cast(lambda_m,     tf.float32)
    lambda_i_f   = tf.cast(lambda_i,     tf.float32)
    lambda_z_f   = tf.cast(lambda_z,     tf.float32)
    lambda_r_f   = tf.cast(lambda_r,     tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    clipnorm_f   = tf.cast(clipnorm,     tf.float32)

    def train_step_per_replica(batch_x, batch_y):
        enc_b         = batch_x["enc_input"]
        dec_b         = batch_x["dec_input"]
        tpred_b       = batch_x["tpred"]
        teacher_hid_b = batch_x["teacher_hidden"]
        tgt_b         = batch_y

        with tf.GradientTape() as tape:
            model_out = phase2_model([enc_b, dec_b], training=True)
            seq_out        = model_out[0]
            dec_h_seq      = model_out[1]
            dec_z_logit_seq = model_out[2]

            l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
            l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)

            total = alpha_f * l_kd + (1.0 - alpha_f) * l_seq

            if use_mem:
                l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len)
                total = total + lambda_m_f * l_m
            else:
                l_m = tf.constant(0.0)

            if use_innov:
                l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
                total = total + lambda_i_f * l_i
            else:
                l_i = tf.constant(0.0)

            if use_zsat and has_z_logit:
                l_z = loss_zsat_logit(dec_z_logit_seq)
                total = total + lambda_z_f * l_z
            else:
                l_z = tf.constant(0.0)

            if use_rail:
                l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
                total = total + lambda_r_f * l_r
            else:
                l_r = tf.constant(0.0)

        grads = tape.gradient(total, phase2_model.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, phase2_model.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) for g in grads
        ]))
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, phase2_model.trainable_variables))

        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, tf.cast(nan_in_grads, tf.float32)

    @tf.function
    def dist_step(batch_x, batch_y):
        per = strategy.run(train_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_step


def make_dist_memoq_val(
    strategy,
    phase2_model,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    use_mem,
    use_innov,
    use_zsat,
    use_rail,
    has_z_logit,
):
    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)

    def val_step_per_replica(batch_x, batch_y):
        enc_b         = batch_x["enc_input"]
        dec_b         = batch_x["dec_input"]
        tpred_b       = batch_x["tpred"]
        teacher_hid_b = batch_x["teacher_hidden"]
        tgt_b         = batch_y

        model_out = phase2_model([enc_b, dec_b], training=False)
        seq_out         = model_out[0]
        dec_h_seq       = model_out[1]
        dec_z_logit_seq = model_out[2]

        l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)
        total = alpha_f * l_kd + (1.0 - alpha_f) * l_seq

        if use_mem:
            l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len)
            total = total + lambda_m_f * l_m
        else:
            l_m = tf.constant(0.0)

        if use_innov:
            l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
            total = total + lambda_i_f * l_i
        else:
            l_i = tf.constant(0.0)

        if use_zsat and has_z_logit:
            l_z = loss_zsat_logit(dec_z_logit_seq)
            total = total + lambda_z_f * l_z
        else:
            l_z = tf.constant(0.0)

        if use_rail:
            l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r
        else:
            l_r = tf.constant(0.0)

        mae = tf.reduce_mean(tf.abs(seq_out - tgt_b))
        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, mae

    @tf.function
    def dist_val_step(batch_x, batch_y):
        per = strategy.run(val_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_val_step


# ==============================================================================
# make_dist_memoq_train_final / make_dist_memoq_val_final:
# Phase 3 — hard QKeras final student.
# The final_qkeras_student outputs only seq_out (batch, T, n_out).
# For hidden trajectory we build a side hidden model once and call it
# inside the gradient tape. The side model shares weights with
# final_qkeras_student.
# ==============================================================================

def build_final_hidden_model(final_qkeras_student):
    dec_out = final_qkeras_student.get_layer("sdecgru").output

    if isinstance(dec_out, (list, tuple)):
        dec_hid_seq = dec_out[0]
    else:
        dec_hid_seq = dec_out

    return keras.models.Model(
        inputs=final_qkeras_student.inputs,
        outputs=[final_qkeras_student.output, dec_hid_seq],
        name="memoq_final_hidden_model",
    )

def make_dist_memoq_train_final(
    strategy,
    final_qkeras_student,
    optimizer,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
    clipnorm,
):
    final_hidden_model = build_final_hidden_model(final_qkeras_student)

    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)
    clipnorm_f   = tf.cast(clipnorm,      tf.float32)

    def train_step_per_replica(batch_x, batch_y):
        enc_b         = batch_x["enc_input"]
        dec_b         = batch_x["dec_input"]
        tpred_b       = batch_x["tpred"]
        teacher_hid_b = batch_x["teacher_hidden"]
        tgt_b         = batch_y

        with tf.GradientTape() as tape:
            seq_out, dec_h_seq = final_hidden_model([enc_b, dec_b], training=True)

            l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
            l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)
            total = alpha_f * l_kd + (1.0 - alpha_f) * l_seq

            l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len)
            total = total + lambda_m_f * l_m

            l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
            total = total + lambda_i_f * l_i

            l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
            total = total + lambda_r_f * l_r

            # Phase 3 z_sat: use dec_h_seq as proxy (no z_logit in hard QGRU).
            # We skip L_zsat in P3 unless lambda_z > 0 AND we want to fire it.
            # With lambda_z = 0.001 we apply a soft version on the hidden state
            # magnitude directly (clamp very close to 1.0).
            if lambda_z > 0.0:
                excess_z = tf.nn.relu(tf.abs(dec_h_seq) - 0.95)
                l_z = tf.reduce_mean(tf.square(excess_z))
                total = total + lambda_z_f * l_z
            else:
                l_z = tf.constant(0.0)

        grads = tape.gradient(total, final_qkeras_student.trainable_variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, final_qkeras_student.trainable_variables)
        ]
        nan_in_grads = tf.reduce_any(tf.stack([
            tf.reduce_any(tf.math.is_nan(g)) for g in grads
        ]))
        grads, _ = tf.clip_by_global_norm(grads, clipnorm_f)
        optimizer.apply_gradients(zip(grads, final_qkeras_student.trainable_variables))

        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, tf.cast(nan_in_grads, tf.float32)

    @tf.function
    def dist_step(batch_x, batch_y):
        per = strategy.run(train_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_step


def make_dist_memoq_val_final(
    strategy,
    final_qkeras_student,
    alpha,
    channel_scales,
    huber_delta,
    lambda_m,
    lambda_i,
    lambda_z,
    lambda_r,
    epsilon_innov,
    seq_len,
    rho_rail,
    mu_rail,
):
    final_hidden_model_val = build_final_hidden_model(final_qkeras_student)

    alpha_f      = tf.cast(alpha,         tf.float32)
    lambda_m_f   = tf.cast(lambda_m,      tf.float32)
    lambda_i_f   = tf.cast(lambda_i,      tf.float32)
    lambda_z_f   = tf.cast(lambda_z,      tf.float32)
    lambda_r_f   = tf.cast(lambda_r,      tf.float32)
    eps_innov_f  = tf.cast(epsilon_innov, tf.float32)

    def val_step_per_replica(batch_x, batch_y):
        enc_b         = batch_x["enc_input"]
        dec_b         = batch_x["dec_input"]
        tpred_b       = batch_x["tpred"]
        teacher_hid_b = batch_x["teacher_hidden"]
        tgt_b         = batch_y

        seq_out, dec_h_seq = final_hidden_model_val([enc_b, dec_b], training=False)

        l_seq = channel_normalised_huber_memoq(tgt_b,   seq_out, channel_scales, huber_delta)
        l_kd  = channel_normalised_huber_memoq(tpred_b, seq_out, channel_scales, huber_delta)
        total = alpha_f * l_kd + (1.0 - alpha_f) * l_seq

        l_m = loss_mem(dec_h_seq, teacher_hid_b, seq_len)
        total = total + lambda_m_f * l_m

        l_i = loss_innov(dec_h_seq, teacher_hid_b, eps_innov_f)
        total = total + lambda_i_f * l_i

        l_r = loss_railpred(dec_h_seq, rho_rail, mu_rail)
        total = total + lambda_r_f * l_r

        if lambda_z > 0.0:
            excess_z = tf.nn.relu(tf.abs(dec_h_seq) - 0.95)
            l_z = tf.reduce_mean(tf.square(excess_z))
            total = total + lambda_z_f * l_z
        else:
            l_z = tf.constant(0.0)

        mae = tf.reduce_mean(tf.abs(seq_out - tgt_b))
        return total, l_seq, l_kd, l_m, l_i, l_z, l_r, mae

    @tf.function
    def dist_val_step(batch_x, batch_y):
        per = strategy.run(val_step_per_replica, args=(batch_x, batch_y))
        return tuple(
            strategy.reduce(tf.distribute.ReduceOp.MEAN, p, axis=None)
            for p in per
        )

    return dist_val_step


# ==============================================================================
# run_epoch for MemoQ — extended to handle 8-output step functions.
# Identical calling convention to run_epoch in train_student_sqkd.py but
# handles (total, l_seq, l_kd, l_m, l_i, l_z, l_r, nan_flag) tuples.
# ==============================================================================
def run_epoch(
    phase_tag,
    epoch,
    total_epochs,
    dist_train_dataset,
    dist_val_dataset,
    train_steps,
    val_steps,
    dist_train_step_fn,
    dist_val_step_fn,
    lr_scheduler,
    effective_warmup_epochs,
    effective_lr,
    history,
    best_val,
    patience_ct,
    patience_max,
    min_delta,
    best_ckpt_path,
    model_to_save,
    csv_path,
    log_interval,
    nan_warn_threshold,
    pf,
    epoch_in_phase=0,
):
    t_epoch       = time.time()
    t_batch_zero  = None

    train_acc = {
        "total": 0.0, "seq": 0.0, "kd":    0.0,
        "mem":   0.0, "innov": 0.0, "zsat": 0.0, "rail": 0.0, "mae": 0.0,
    }
    train_steps_done = 0
    nan_count_train  = 0

    pf(
        f"\n[{phase_tag} EPOCH {epoch + 1}/{total_epochs}]"
        f"  Training  lr={lr_scheduler.current_lr:.2e}"
    )
    sys.stdout.flush()

    for step, (batch_x, batch_y) in enumerate(dist_train_dataset):
        if train_steps_done >= train_steps:
            break

        t0 = time.time()
        result = dist_train_step_fn(batch_x, batch_y)

        if t_batch_zero is None:
            t_batch_zero = time.time() - t0

        if len(result) == 8:
            l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae = result
        elif len(result) == 4:
            l_total, l_seq, l_kd, mae = result
            l_mem = l_innov = l_zsat = l_rail = tf.constant(0.0)
        else:
            l_total = result[0]
            l_seq = result[1] if len(result) > 1 else tf.constant(0.0)
            l_kd  = result[2] if len(result) > 2 else tf.constant(0.0)
            l_mem = l_innov = l_zsat = l_rail = tf.constant(0.0)
            mae   = result[-1] if len(result) > 3 else tf.constant(0.0)

        for v, k in zip(
            [l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae],
            ["total", "seq", "kd", "mem", "innov", "zsat", "rail", "mae"],
        ):
            val_py = float(v)
            if not math.isfinite(val_py):
                nan_count_train += 1
                val_py = 0.0
            train_acc[k] += val_py

        train_steps_done += 1

        if (step + 1) % log_interval == 0:
            pf(
                f"  step {train_steps_done}/{train_steps}"
                f"  loss={train_acc['total'] / train_steps_done:.5f}"
                f"  seq={train_acc['seq'] / train_steps_done:.5f}"
                f"  kd={train_acc['kd'] / train_steps_done:.5f}"
                f"  mem={train_acc['mem'] / train_steps_done:.5f}"
                f"  innov={train_acc['innov'] / train_steps_done:.5f}"
                f"  zsat={train_acc['zsat'] / train_steps_done:.5f}"
                f"  rail={train_acc['rail'] / train_steps_done:.5f}"
            )
            sys.stdout.flush()

    if nan_count_train >= nan_warn_threshold:
        pf(
            f"  [WARNING] {nan_count_train}/{train_steps_done} NaN/Inf steps in training."
        )
        sys.stdout.flush()

    n_tr = max(train_steps_done, 1)
    train_metrics = {k: v / n_tr for k, v in train_acc.items()}

    # ── Validation loop ───────────────────────────────────────────────────────
    pf(f"  Validation...")
    sys.stdout.flush()

    val_acc = {
        "total": 0.0, "seq": 0.0, "kd":    0.0,
        "mem":   0.0, "innov": 0.0, "zsat": 0.0, "rail": 0.0, "mae": 0.0,
    }
    val_steps_done = 0
    nan_count_val  = 0

    for step, (batch_x, batch_y) in enumerate(dist_val_dataset):
        if val_steps_done >= val_steps:
            break

        result = dist_val_step_fn(batch_x, batch_y)

        if len(result) == 8:
            l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae = result
        elif len(result) == 4:
            l_total, l_seq, l_kd, mae = result
            l_mem = l_innov = l_zsat = l_rail = tf.constant(0.0)
        else:
            l_total = result[0]
            l_seq = result[1] if len(result) > 1 else tf.constant(0.0)
            l_kd  = result[2] if len(result) > 2 else tf.constant(0.0)
            l_mem = l_innov = l_zsat = l_rail = tf.constant(0.0)
            mae   = result[-1] if len(result) > 3 else tf.constant(0.0)

        for v, k in zip(
            [l_total, l_seq, l_kd, l_mem, l_innov, l_zsat, l_rail, mae],
            ["total", "seq", "kd", "mem", "innov", "zsat", "rail", "mae"],
        ):
            val_py = float(v)
            if not math.isfinite(val_py):
                nan_count_val += 1
                val_py = 0.0
            val_acc[k] += val_py

        val_steps_done += 1

    if nan_count_val >= max(1, int(val_steps * 0.10)):
        pf(
            f"  [WARNING] {nan_count_val}/{val_steps_done} NaN/Inf steps in validation."
        )
        sys.stdout.flush()

    n_vl = max(val_steps_done, 1)
    val_metrics = {k: v / n_vl for k, v in val_acc.items()}
    val_loss     = val_metrics["total"]

    t_elapsed = time.time() - t_epoch
    pf(
        f"  [{phase_tag} EPOCH {epoch + 1}]"
        f"  train_loss={train_metrics['total']:.6f}"
        f"  val_loss={val_loss:.6f}"
        f"  val_mae={val_metrics['mae']:.6f}"
        f"  train_seq={train_metrics['seq']:.6f}"
        f"  train_kd={train_metrics['kd']:.6f}"
        f"  train_mem={train_metrics['mem']:.6f}"
        f"  train_innov={train_metrics['innov']:.6f}"
        f"  train_zsat={train_metrics['zsat']:.6f}"
        f"  train_rail={train_metrics['rail']:.6f}"
        f"  lr={lr_scheduler.current_lr:.2e}"
        f"  t={t_elapsed:.1f}s"
        f"  (first_batch={t_batch_zero:.2f}s)" if t_batch_zero is not None else ""
    )
    sys.stdout.flush()

    # ── History accumulation ──────────────────────────────────────────────────
    history.setdefault("total",     []).append(train_metrics["total"])
    history.setdefault("seq",       []).append(train_metrics["seq"])
    history.setdefault("kd",        []).append(train_metrics["kd"])
    history.setdefault("mem",       []).append(train_metrics["mem"])
    history.setdefault("innov",     []).append(train_metrics["innov"])
    history.setdefault("zsat",      []).append(train_metrics["zsat"])
    history.setdefault("rail",      []).append(train_metrics["rail"])
    history.setdefault("val_total", []).append(val_metrics["total"])
    history.setdefault("val_seq",   []).append(val_metrics["seq"])
    history.setdefault("val_kd",    []).append(val_metrics["kd"])
    history.setdefault("val_mem",   []).append(val_metrics["mem"])
    history.setdefault("val_innov", []).append(val_metrics["innov"])
    history.setdefault("val_zsat",  []).append(val_metrics["zsat"])
    history.setdefault("val_rail",  []).append(val_metrics["rail"])
    history.setdefault("val_mae",   []).append(val_metrics["mae"])
    history.setdefault("phase",     []).append(phase_tag)

    # ── CSV logging ───────────────────────────────────────────────────────────
    with open(csv_path, "a") as csv_f:
        csv_f.write(
            f"{epoch + 1},{phase_tag},"
            f"{train_metrics['total']:.8f},{train_metrics['seq']:.8f},"
            f"{train_metrics['kd']:.8f},{train_metrics['mem']:.8f},"
            f"{train_metrics['innov']:.8f},{train_metrics['zsat']:.8f},"
            f"{train_metrics['rail']:.8f},"
            f"{val_metrics['total']:.8f},{val_metrics['seq']:.8f},"
            f"{val_metrics['kd']:.8f},{val_metrics['mem']:.8f},"
            f"{val_metrics['innov']:.8f},{val_metrics['zsat']:.8f},"
            f"{val_metrics['rail']:.8f},"
            f"{val_metrics['mae']:.8f},{lr_scheduler.current_lr:.2e}\n"
        )

    # ── LR warmup / plateau ───────────────────────────────────────────────────
    if effective_warmup_epochs > 0 and epoch_in_phase < effective_warmup_epochs:
        warmup_lr = float(effective_lr) * (epoch_in_phase + 1) / float(effective_warmup_epochs)
        lr_scheduler.lr_var.assign(warmup_lr)
        pf(
            f"  [WARMUP] ep_in_phase={epoch_in_phase + 1}/{effective_warmup_epochs}"
            f"  lr={warmup_lr:.3e}"
        )
    else:
        lr_scheduler.step(val_loss, epoch, pf)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    early_stop = False
    if val_loss < best_val - min_delta:
        best_val    = val_loss
        patience_ct = 0
        model_to_save.save_weights(best_ckpt_path)
        pf(
            f"  ✓ [{phase_tag}] New best val={best_val:.6f}"
            f"  -> {best_ckpt_path}"
        )
        sys.stdout.flush()
    else:
        patience_ct += 1
        pf(f"  patience {patience_ct}/{patience_max}")
        sys.stdout.flush()
        if patience_ct >= patience_max:
            pf(f"[{phase_tag}] Early stop at epoch {epoch + 1}")
            sys.stdout.flush()
            early_stop = True

    return history, best_val, patience_ct, early_stop

# ==============================================================================
# cache_teacher_hidden:
# Caches the decoder hidden sequence of the teacher model to a mmap file.
# teacherHidden_L{seq_len}{N}.npy — shape (N, T, teacher_units)
# ==============================================================================

def cache_teacher_hidden(
    teacher_hidden_model,
    normalized_input,
    seq_len,
    teacher_units,
    n_samples,
    infer_batch,
    data_dir,
    pf,
):
    file_hidden = os.path.join(
        data_dir, f"teacherHidden_L{seq_len}{n_samples}.npy"
    )

    if os.path.exists(file_hidden):
        pf(f"[CACHE-HID] Teacher hidden cache found — loading mmap:")
        pf(f"  {file_hidden}")
        sys.stdout.flush()
        teacher_hidden = np.load(file_hidden, mmap_mode="r")
        pf(
            f"[CACHE-HID] teacher_hidden : {teacher_hidden.shape}  "
            f"dtype={teacher_hidden.dtype}"
        )
        sys.stdout.flush()
        return teacher_hidden

    pf("=" * 60)
    pf("[CACHE-HID] Teacher hidden cache NOT found — running full-dataset inference...")
    pf(f"  hidden → {file_hidden}")
    pf("=" * 60)
    sys.stdout.flush()

    teacher_hidden = np.lib.format.open_memmap(
        file_hidden, mode="w+", dtype=np.float32,
        shape=(n_samples, seq_len, teacher_units),
    )

    @tf.function(reduce_retracing=True)
    def teacher_hidden_forward(enc_b, dec_b):
        return teacher_hidden_model([enc_b, dec_b], training=False)

    wu_size = min(infer_batch, n_samples)
    enc_wu = tf.constant(normalized_input[:wu_size], dtype=tf.float32)
    dec_wu = tf.zeros((wu_size, seq_len, 1), dtype=tf.float32)
    _h = teacher_hidden_forward(enc_wu, dec_wu)
    _ = _h.numpy()
    del enc_wu, dec_wu, _h

    n_batches = int(np.ceil(n_samples / infer_batch))
    print_every = max(1, n_batches // 20)
    t0 = time.time()

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_samples)
        enc_b = tf.constant(normalized_input[s:e], dtype=tf.float32)
        dec_b = tf.zeros((e - s, seq_len, 1), dtype=tf.float32)
        hid = teacher_hidden_forward(enc_b, dec_b)
        teacher_hidden[s:e] = hid.numpy()
        teacher_hidden.flush()
        del enc_b, dec_b, hid
        if (b % print_every == 0) or (b == n_batches - 1):
            elapsed = time.time() - t0
            pct = 100.0 * e / n_samples
            pf(f"[CACHE-HID] batch {b + 1:>4d}/{n_batches}  {pct:5.1f}%  elapsed={elapsed / 60:.1f}min")
            sys.stdout.flush()

    teacher_hidden.flush()
    del teacher_hidden
    teacher_hidden = np.load(file_hidden, mmap_mode="r")
    pf(
        f"[CACHE-HID] Done  shape={teacher_hidden.shape}  "
        f"dtype={teacher_hidden.dtype}"
    )
    sys.stdout.flush()
    return teacher_hidden


# ==============================================================================
# compute_epsilon_innov:
# Compute epsilon_innov = 0.1 * median(v_t(teacher)) over cached teacher hidden.
# Runs once before training.
# ==============================================================================

def compute_epsilon_innov(teacher_hidden, seq_len, sample_cap=50000, pf=print):
    pf("[INNOV] Computing epsilon_innov from teacher hidden cache...")
    N = teacher_hidden.shape[0]
    n = min(N, sample_cap)
    idx = np.random.choice(N, n, replace=False)
    h = teacher_hidden[idx].astype(np.float32)
    diff = h[:, 1:, :] - h[:, :-1, :]
    v_t = np.mean(diff ** 2, axis=(0, 2))
    eps = 0.1 * float(np.median(v_t))
    eps = max(eps, 1e-6)
    pf(
        f"[INNOV] epsilon_innov={eps:.6f}  "
        f"(median v_t={float(np.median(v_t)):.6f}  "
        f"from {n:,} samples)"
    )
    sys.stdout.flush()
    return eps


# ==============================================================================
# make_memoq_kd_dataset:
# Extends make_kd_dataset to include teacher_hidden in batch_x.
# batch_x keys: enc_input, dec_input, tpred, teacher_hidden
# batch_y: ground truth labels
# ==============================================================================

def make_memoq_kd_dataset(
    enc_arr,
    tgt_arr,
    tpred_arr,
    teacher_hidden_arr,
    batch_size,
    accumulation_steps,
    seq_len,
    n_out,
    teacher_units,
    shuffle,
    seed,
    prefetch_batches,
    pipeline_workers,
):
    n = len(enc_arr)
    dec_arr = np.zeros_like(enc_arr)
    micro_batch_size = batch_size // accumulation_steps

    ds_enc    = tf.data.Dataset.from_tensor_slices(enc_arr)
    ds_dec    = tf.data.Dataset.from_tensor_slices(dec_arr)
    ds_tpred  = tf.data.Dataset.from_tensor_slices(tpred_arr)
    ds_teacher_hidden   = tf.data.Dataset.from_tensor_slices(teacher_hidden_arr)
    ds_tgt    = tf.data.Dataset.from_tensor_slices(tgt_arr)

    ds = tf.data.Dataset.zip((ds_enc, ds_dec, ds_tpred, ds_teacher_hidden, ds_tgt))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(n, 200_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(micro_batch_size, drop_remainder=True)

    def set_shapes(enc_b, dec_b, tpred_b, teacher_hidden_b, tgt_b):
        enc_b.set_shape([micro_batch_size, seq_len, 1])
        dec_b.set_shape([micro_batch_size, seq_len, 1])
        tpred_b.set_shape([micro_batch_size, seq_len, n_out])
        teacher_hidden_b.set_shape([micro_batch_size, seq_len, teacher_units])
        tgt_b.set_shape([micro_batch_size, seq_len, n_out])
        batchx = {
            "enc_input":       enc_b,
            "dec_input":       dec_b,
            "tpred":           tpred_b,
            "teacher_hidden":  teacher_hidden_b,
        }
        return batchx, tgt_b

    ds = ds.map(set_shapes, num_parallel_calls=pipeline_workers)
    ds = ds.prefetch(prefetch_batches)
    return ds


# ==============================================================================
# materialise_memoq_buffers:
# Like materialise_enc_tgt_tpred but also materialises teacher_hidden.
# ==============================================================================

def materialise_memoq_buffers(
    normalized_input,
    res,
    teacher_predictions,
    teacher_hidden,
    idx,
    seq_len,
    n_out,
    teacher_units,
    label,
    pf,
):
    n = len(idx)
    pf(f"  Materialising MemoQ {label} buffers ({n:,} samples)...")
    t0 = time.time()

    enc      = np.empty((n, seq_len, 1),            dtype=np.float32)
    tgt      = np.empty((n, seq_len, n_out),        dtype=np.float32)
    tpred    = np.empty((n, seq_len, n_out),        dtype=np.float32)
    teacher_hidden     = np.empty((n, seq_len, teacher_units), dtype=np.float32)

    chunk = 65536
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        enc[s:e]   = normalized_input[idx[s:e]]
        tgt[s:e]   = res[idx[s:e]]
        tpred[s:e] = teacher_predictions[idx[s:e]]
        teacher_hidden[s:e]  = teacher_hidden[idx[s:e]]

    pf(
        f"  Done in {time.time() - t0:.1f}s  "
        f"enc={enc.nbytes/1e9:.2f}GB  "
        f"tgt={tgt.nbytes/1e9:.2f}GB  "
        f"tpred={tpred.nbytes/1e9:.2f}GB  "
        f"teacher_hidden={teacher_hidden.nbytes/1e9:.2f}GB"
    )
    sys.stdout.flush()
    return enc, tgt, tpred, teacher_hidden


# ==============================================================================
# transfer_float_to_phase2:
# Transfer weights from phase1 float student (standard Keras GRU) into the
# MemoQGRUCell split-gate variables of phase2_model.
# The float GRU stores packed kernel (input_dim, 3*H) and
# recurrent_kernel (H, 3*H) in columns [z, r, h] order.
# ==============================================================================

def transfer_float_to_phase2(float_student, phase2_model, enc_cell_p2, dec_cell_p2, pf):
    pf("[P1->P2 TRANSFER] Unpacking float GRU weights into split-gate cells...")

    def unpack_and_set(gru_layer_name, cell):
        layer   = float_student.get_layer(gru_layer_name)
        weights = layer.get_weights()
        H = cell.units

        if len(weights) != 3:
            raise ValueError(
                f"{gru_layer_name} expected 3 Keras GRU tensors "
                f"[kernel, recurrent_kernel, bias], got {len(weights)}"
            )

        packed_kernel, packed_recurrent, packed_bias = weights

        if packed_bias.ndim == 2:
            packed_bias_inp = packed_bias[0]
            packed_bias_rec = packed_bias[1]
        else:
            packed_bias_inp = packed_bias
            packed_bias_rec = np.zeros_like(packed_bias)

        cell.W_z.assign(packed_kernel[:, 0:H])
        cell.W_r.assign(packed_kernel[:, H:2 * H])
        cell.W_h.assign(packed_kernel[:, 2 * H:3 * H])

        cell.U_z.assign(packed_recurrent[:, 0:H])
        cell.U_r.assign(packed_recurrent[:, H:2 * H])
        cell.U_h.assign(packed_recurrent[:, 2 * H:3 * H])

        cell.b_z_inp.assign(packed_bias_inp[0:H])
        cell.b_r_inp.assign(packed_bias_inp[H:2 * H])
        cell.b_h_inp.assign(packed_bias_inp[2 * H:3 * H])

        cell.b_z_rec.assign(packed_bias_rec[0:H])
        cell.b_r_rec.assign(packed_bias_rec[H:2 * H])
        cell.b_h_rec.assign(packed_bias_rec[2 * H:3 * H])

        pf(
            f"[P1->P2 TRANSFER] {gru_layer_name}: "
            f"W={packed_kernel.shape}, U={packed_recurrent.shape}, "
            f"bias_inp={packed_bias_inp.shape}, bias_rec={packed_bias_rec.shape}"
        )

    unpack_and_set("sencgru", enc_cell_p2)
    unpack_and_set("sdecgru", dec_cell_p2)

    src_dense = float_student.get_layer("sdec_dense")
    dst_dense = phase2_model.get_layer("sdec_dense")
    dst_dense.set_weights(src_dense.get_weights())
    pf("[P1->P2 TRANSFER] sdec_dense transferred.")

    sys.stdout.flush()

# ==============================================================================
# save_loss_curves_memoq:
# Multi-panel PNG with all 7 loss components across all phases.
# ==============================================================================

def save_loss_curves_memoq(history, best_val_loss, args, job_dir, pf):
    phases_arr = history.get("phase", [])
    n_ep = len(history.get("total", []))
    epochs_arr = list(range(1, n_ep + 1))

    phase_colors = {
        "P1":  ("tab:blue",   "tab:cyan"),
        "P2A": ("tab:orange", "tab:red"),
        "P2B": ("tab:purple", "tab:pink"),
        "P2C": ("tab:brown",  "tab:olive"),
        "P3":  ("tab:green",  "tab:lime"),
    }

    fig, axes = plt.subplots(2, 5, figsize=(36, 8))
    axes = axes.flatten()

    loss_pairs = [
        ("total",   "val_total",  "Total Loss"),
        ("seq",     "val_seq",    "L_seq (HuberCN GT)"),
        ("kd",      "val_kd",     "L_KD (HuberCN Teacher)"),
        ("mem",     "val_mem",    "L_mem (Memory Kernel)"),
        ("innov",   "val_innov",  "L_innov (Innovation)"),
        ("zsat",    "val_zsat",   "L_zsat (Gate Sat)"),
        ("rail",    "val_rail",   "L_railpred"),
        ("val_mae", None,         "Val MAE"),
    ]

    for ax_idx, (train_key, val_key, title) in enumerate(loss_pairs):
        if ax_idx >= len(axes) - 2:
            break
        ax = axes[ax_idx]
        for phase_tag, (ctr, cva) in phase_colors.items():
            mask = [i for i, ph in enumerate(phases_arr) if ph == phase_tag]
            if not mask:
                continue
            ep_ph = [epochs_arr[i] for i in mask]
            tr_vals = [history.get(train_key, [0]*n_ep)[i] for i in mask]
            ax.plot(ep_ph, tr_vals, color=ctr, label=f"{phase_tag} train", linewidth=1.2)
            if val_key and val_key in history:
                va_vals = [history[val_key][i] for i in mask]
                ax.plot(ep_ph, va_vals, color=cva, linestyle="--", label=f"{phase_tag} val", linewidth=1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    axes[8].axis("off")
    axes[8].text(
        0.05, 0.5,
        f"MemoQ  SEQLEN={args.seq_len}\n"
        f"α={args.alpha}  huber_delta={args.memoq_huber_delta}\n"
        f"bits: k={args.bits_kernel} r={args.bits_recurrent} "
        f"b={args.bits_bias} a={args.bits_activation} s={args.bits_state}\n"
        f"Student QGRU hidden={args.student_units}\n"
        f"Teacher GRU hidden={args.teacher_units} x {args.teacher_layers}\n"
        f"P1={args.memoq_stage1_epochs}  2A={args.memoq_stage2a_epochs}  "
        f"2B={args.memoq_stage2b_epochs}  2C={args.memoq_stage2c_epochs}  "
        f"P3={args.memoq_stage3_epochs}\n"
        f"Batch={args.batch_size}  EffLR={args.effective_lr:.2e}\n"
        f"rho_rail={args.memoq_rho_rail}  mu_rail={args.memoq_mu_rail}\n"
        f"innov_burnin={args.memoq_innov_burnin}\n"
        f"Best val (P3)={best_val_loss:.6f}\n"
        f"Total epochs run={n_ep}",
        fontsize=8,
        verticalalignment="center",
        transform=axes[8].transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    axes[9].axis("off")

    plt.tight_layout()
    curves_path = os.path.join(job_dir, "training_history.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"Loss curves saved: {curves_path}")
    sys.stdout.flush()


# ==============================================================================
# evaluate_and_save — identical to vanilla_kd but uses final_qkeras_student.
# ==============================================================================

def evaluate_and_save(
    final_qkeras_student,
    enc_test,
    tgt_test,
    labels_test,
    gate_width_ns,
    n_out,
    seq_len,
    infer_batch,
    job_dir,
    args,
    pf,
):
    pf("[EVAL] Running test set evaluation...")
    sys.stdout.flush()

    n_test = len(enc_test)
    dec_test = np.zeros((n_test, seq_len, 1), dtype=np.float32)

    @tf.function(reduce_retracing=True)
    def predict_step(enc_b, dec_b):
        return final_qkeras_student([enc_b, dec_b], training=False)

    preds = np.empty((n_test, seq_len, n_out), dtype=np.float32)
    n_batches = int(np.ceil(n_test / infer_batch))

    for b in range(n_batches):
        s = b * infer_batch
        e = min(s + infer_batch, n_test)
        enc_b = tf.constant(enc_test[s:e], dtype=tf.float32)
        dec_b = tf.constant(dec_test[s:e], dtype=tf.float32)
        preds[s:e] = predict_step(enc_b, dec_b).numpy()

    pf(f"[EVAL] Predictions complete: shape={preds.shape}")
    sys.stdout.flush()

    mse_per_sample  = np.mean((preds - tgt_test) ** 2, axis=(1, 2))
    mae_per_sample  = np.mean(np.abs(preds - tgt_test), axis=(1, 2))
    overall_mse     = float(np.mean(mse_per_sample))
    overall_mae     = float(np.mean(mae_per_sample))

    pf(f"[EVAL] Test MSE={overall_mse:.6f}  Test MAE={overall_mae:.6f}")
    sys.stdout.flush()

    channel_names = ["tau1", "tau2", "fret"] if n_out >= 3 else [f"ch{c}" for c in range(n_out)]

    per_channel_mse = {}
    per_channel_mae = {}
    per_channel_pearson = {}
    for c in range(n_out):
        ch = channel_names[c] if c < len(channel_names) else f"ch{c}"
        pred_flat = preds[:, :, c].flatten()
        tgt_flat  = tgt_test[:, :, c].flatten()
        per_channel_mse[ch]     = float(np.mean((pred_flat - tgt_flat) ** 2))
        per_channel_mae[ch]     = float(np.mean(np.abs(pred_flat - tgt_flat)))
        r, _ = pearsonr(pred_flat, tgt_flat)
        per_channel_pearson[ch] = float(r)
        pf(
            f"[EVAL]   {ch}  MSE={per_channel_mse[ch]:.6f}  "
            f"MAE={per_channel_mae[ch]:.6f}  r={per_channel_pearson[ch]:.4f}"
        )

    dtw_scores = []
    n_dtw = min(200, n_test)
    dtw_idx = np.random.choice(n_test, n_dtw, replace=False)
    for i in dtw_idx:
        if HAS_FASTDTW:
            d, _ = fastdtw(preds[i], tgt_test[i], dist=euclidean)
        else:
            d = float("nan")
        dtw_scores.append(d)
    mean_dtw = float(np.mean(dtw_scores))
    pf(f"[EVAL] Mean DTW (n={n_dtw}): {mean_dtw:.4f}")
    sys.stdout.flush()

    metrics = {
        "overall_mse":       overall_mse,
        "overall_mae":       overall_mae,
        "mean_dtw":          mean_dtw,
        "per_channel_mse":   per_channel_mse,
        "per_channel_mae":   per_channel_mae,
        "per_channel_pearson": per_channel_pearson,
    }
    with open(os.path.join(job_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pf(f"[EVAL] test_metrics.json saved.")
    sys.stdout.flush()

    time_axis = np.arange(seq_len) * gate_width_ns

    for c, ch in enumerate(channel_names[:n_out]):
        fig, ax = plt.subplots(figsize=(6, 5))
        pred_mean = preds[:, :, c].mean(axis=1)
        tgt_mean  = tgt_test[:, :, c].mean(axis=1)
        ax.scatter(tgt_mean, pred_mean, alpha=0.3, s=6, rasterized=True)
        mn = min(tgt_mean.min(), pred_mean.min())
        mx = max(tgt_mean.max(), pred_mean.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel(f"Ground truth {ch}")
        ax.set_ylabel(f"Student predicted {ch}")
        ax.set_title(
            f"MemoQ Student — {ch}\n"
            f"r={per_channel_pearson[ch]:.3f}  "
            f"MSE={per_channel_mse[ch]:.4f}  "
            f"MAE={per_channel_mae[ch]:.4f}"
        )
        ax.grid(True, alpha=0.3)
        scatter_path = os.path.join(job_dir, f"test_scatter_{ch}.png")
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        pf(f"[EVAL]   scatter saved: {scatter_path}")

    n_res = min(8, n_test)
    res_idx = np.random.choice(n_test, n_res, replace=False)
    fig, axes_res = plt.subplots(n_res, n_out, figsize=(4 * n_out, 2 * n_res))
    if n_res == 1:
        axes_res = axes_res[np.newaxis, :]
    for row, i in enumerate(res_idx):
        for c, ch in enumerate(channel_names[:n_out]):
            ax = axes_res[row, c]
            ax.plot(time_axis, tgt_test[i, :, c], "k-",  linewidth=1.0, label="GT")
            ax.plot(time_axis, preds[i, :, c],    "r--", linewidth=1.0, label="Student")
            ax.set_title(f"Sample {i}  {ch}", fontsize=7)
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.legend(fontsize=6)
    plt.suptitle("MemoQ Student — Test Residual Curves", fontsize=10)
    plt.tight_layout()
    res_path = os.path.join(job_dir, "test_residuals.png")
    plt.savefig(res_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pf(f"[EVAL]   residuals saved: {res_path}")
    sys.stdout.flush()


# ==============================================================================
# parse_args — all MemoQ-specific hyperparameters
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MemoQ: memory-preserving 4-bit recurrent quantization KD for QGRU Seq2Seq.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data-dir",      type=str, required=True)
    p.add_argument("--save-dir",      type=str, default=None)
    p.add_argument("--seq-len",       type=int, default=135)
    p.add_argument("--n-out",         type=int, default=3)
    p.add_argument("--gate-width-ns", type=float, default=0.09)

    p.add_argument("--teacher-ckpt",   type=str, required=True)
    p.add_argument("--teacher-units",  type=int, default=128)
    p.add_argument("--teacher-layers", type=int, default=2)

    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha",       type=float, default=0.7)

    p.add_argument("--bits-kernel",     type=int, default=4)
    p.add_argument("--bits-bias",       type=int, default=4)
    p.add_argument("--bits-recurrent",  type=int, default=4)
    p.add_argument("--bits-activation", type=int, default=4)
    p.add_argument("--bits-state",      type=int, default=4)

    p.add_argument("--student-units", type=int, default=32)

    p.add_argument(
        "--memoq-warmup-epochs", "--memoq-stage1-epochs",
        dest="memoq_warmup_epochs", type=int, default=40,
    )
    p.add_argument("--memoq-stage2a-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2b-epochs", type=int, default=30)
    p.add_argument("--memoq-stage2c-epochs", type=int, default=30)
    p.add_argument(
        "--memoq-finetune-epochs", "--memoq-stage3-epochs",
        dest="memoq_stage3_epochs", type=int, default=170,
    )

    p.add_argument("--memoq-lambda-mem",   type=float, default=0.03)
    p.add_argument("--memoq-lambda-innov", type=float, default=0.005)
    p.add_argument("--memoq-lambda-zsat",  type=float, default=0.002)
    p.add_argument("--memoq-lambda-rail",  type=float, default=0.001)

    p.add_argument("--memoq-huber-delta",    type=float, default=0.1)
    p.add_argument("--memoq-rho-z",          type=float, default=0.98)
    p.add_argument("--memoq-rho-rail",       type=float, default=0.88)
    p.add_argument("--memoq-mu-rail",        type=float, default=0.9)
    p.add_argument("--memoq-innov-burnin",   type=int,   default=5)

    p.add_argument("--memoq-phase3-lr",        type=float, default=5e-6)
    p.add_argument("--memoq-phase3-lr-factor", type=float, default=1.0)

    p.add_argument("--batch-size",        type=int,   default=1024)
    p.add_argument("--epochs",            type=int,   default=330)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--ref-batch-size",    type=int,   default=1024)
    p.add_argument("--no-lr-scaling",     action="store_true", default=False)
    p.add_argument("--lr-factor",         type=float, default=0.5)
    p.add_argument("--lr-patience",       type=int,   default=8)
    p.add_argument("--lr-min",            type=float, default=1e-6)
    p.add_argument("--patience",          type=int,   default=30)
    p.add_argument("--min-delta",         type=float, default=1e-5)
    p.add_argument("--infer-batch",       type=int,   default=8192)
    p.add_argument("--mixed-precision",   action="store_true", default=False)
    p.add_argument("--log-interval",      type=int,   default=10)
    p.add_argument("--prefetch-batches",  type=int,   default=32)
    p.add_argument("--pipeline-workers",  type=int,   default=4)
    p.add_argument("--split-seed",        type=int,   default=42)
    p.add_argument("--warmup-epochs",     type=int,   default=5)
    p.add_argument("--accumulation-steps", type=int,  default=1)
    p.add_argument("--resume",            action="store_true", default=False)

    args = p.parse_args()

    if args.save_dir is None:
        args.save_dir = args.data_dir

    args.memoq_stage1_epochs   = args.memoq_warmup_epochs
    args.memoq_finetune_epochs = args.memoq_stage3_epochs

    scale = float(args.batch_size) / float(args.ref_batch_size)
    if args.no_lr_scaling:
        args.effective_lr             = args.lr
        args.effective_lr_patience    = args.lr_patience
        args.effective_warmup_epochs  = args.warmup_epochs
    else:
        args.effective_lr             = args.lr * scale
        args.effective_lr_patience    = max(1, int(round(args.lr_patience / scale)))
        args.effective_warmup_epochs  = max(1, int(round(args.warmup_epochs * scale)))

    return args

# ==============================================================================
# make_job_name
# ==============================================================================

def make_job_name(args):
    effective_batch = args.batch_size
    micro_batch = args.batch_size // args.accumulation_steps
    return (
        f"memoq"
        f"_b{args.bits_kernel}k{args.bits_bias}r{args.bits_recurrent}"
        f"a{args.bits_activation}s{args.bits_state}"
        f"_gru{args.student_units}"
        f"_dense{args.n_out}"
        f"_effbs{effective_batch}"
        f"_microbs{micro_batch}"
        f"_lr{args.effective_lr:.0e}"
        f"_p1-{args.memoq_stage1_epochs}"
        f"_2a{args.memoq_stage2a_epochs}"
        f"_2b{args.memoq_stage2b_epochs}"
        f"_2c{args.memoq_stage2c_epochs}"
        f"_p3-{args.memoq_stage3_epochs}"
    )


# ==============================================================================
# setup_gpus_and_strategy — identical to vanilla_kd
# ==============================================================================

def setup_gpus_and_strategy(mixed_precision_flag, pf=print):
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        pf("[GPU] No GPUs found — running on CPU.")
        return tf.distribute.get_strategy()

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            pf(f"[GPU]   WARNING set_memory_growth: {e}")

    logical_gpus = tf.config.list_logical_devices("GPU")
    pf(f"[GPU] Physical={len(physical_gpus)}  Logical={len(logical_gpus)}")

    keras.mixed_precision.set_global_policy("float32")
    pf("[GPU] Policy: float32 (QKeras requirement)")

    gpu_devices = [f"GPU:{i}" for i in range(len(logical_gpus))]
    strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
    pf(f"[GPU] MirroredStrategy replicas={strategy.num_replicas_in_sync}")
    sys.stdout.flush()
    return strategy


# ==============================================================================
# build_teacher — exactly as in vanilla_kd
# ==============================================================================

def build_teacher(seq_len, n_out, teacher_units, teacher_layers):
    LAYERS_TEACHER = [teacher_units] * teacher_layers

    encoder_inputs = keras.layers.Input(shape=(None, 1), name="enc_input")
    encoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"enc_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    encoder_rnn = keras.layers.RNN(
        encoder_cells, return_state=True, name="enc_rnn",
    )
    encoder_out_states = encoder_rnn(encoder_inputs)
    encoder_states = encoder_out_states[1:]

    decoder_inputs = keras.layers.Input(shape=(None, 1), name="dec_input")
    decoder_cells = [
        keras.layers.GRUCell(units, reset_after=True, name=f"dec_cell{i}")
        for i, units in enumerate(LAYERS_TEACHER)
    ]
    decoder_rnn = keras.layers.RNN(
        decoder_cells,
        return_sequences=True,
        return_state=True,
        name="dec_rnn",
    )
    decoder_out_states = decoder_rnn(decoder_inputs, initial_state=encoder_states)
    decoder_hidden_sequence = decoder_out_states[0]

    decoder_dense = keras.layers.Dense(n_out, activation="linear", name="dec_dense")
    decoder_output = decoder_dense(decoder_hidden_sequence)

    teacher_model = keras.models.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_output,
        name="teacher_seq2seq",
    )
    return teacher_model


def build_teacher_hidden_model(teacher_model):
    """
    Returns a model that exposes the decoder hidden sequence from teacher.
    Shares all weights with teacher_model (same layer objects).
    """
    dec_rnn_layer = teacher_model.get_layer("dec_rnn")
    dec_rnn_out = dec_rnn_layer.output
    if isinstance(dec_rnn_out, (list, tuple)):
        dec_hidden_seq = dec_rnn_out[0]
    else:
        dec_hidden_seq = dec_rnn_out

    teacher_hidden_model = keras.models.Model(
        inputs=teacher_model.inputs,
        outputs=dec_hidden_seq,
        name="teacher_hidden_seq2seq",
    )
    return teacher_hidden_model


# ==============================================================================
# build_float_student — phase 1 standard Keras GRU student.
# Same topology and same layer names as final QKeras student so that
# transfer_float_to_phase2 can unpack by name.
# ==============================================================================

def build_float_student(seq_len, n_out, student_units):
    enc_inputs = keras.layers.Input(shape=(None, 1), name="senc_input")
    dec_inputs = keras.layers.Input(shape=(None, 1), name="sdec_input")

    enc_out, enc_state = keras.layers.GRU(
        units=student_units,
        return_state=True,
        reset_after=True,
        name="sencgru",
    )(enc_inputs)

    dec_hid_seq, _ = keras.layers.GRU(
        units=student_units,
        return_sequences=True,
        return_state=True,
        reset_after=True,
        name="sdecgru",
    )(dec_inputs, initial_state=enc_state)

    s_output = keras.layers.Dense(
        n_out, activation="linear", name="sdec_dense"
    )(dec_hid_seq)

    return keras.models.Model(
        inputs=[enc_inputs, dec_inputs],
        outputs=s_output,
        name="float_student_memoq",
    )


# ==============================================================================
# find_data_files — identical to vanilla_kd
# ==============================================================================




# ==============================================================================
# main
# ==============================================================================

def main():
    args = parse_args()
    pf = print

    strategy = setup_gpus_and_strategy(args.mixed_precision)

    # ── Job directory ──────────────────────────────────────────────────────────
    job_name = make_job_name(args)
    job_dir  = os.path.join(args.save_dir, "results", job_name)
    os.makedirs(job_dir, exist_ok=True)
    pf(f"[JOB] {job_name}")
    pf(f"[JOB] Output dir: {job_dir}")
    sys.stdout.flush()

    with open(os.path.join(job_dir, "student_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Data loading ──────────────────────────────────────────────────────────
    file_input, file_res, file_labels, file_train, file_val, file_test = find_data_files(
        args.data_dir, args.seq_len
    )
    pf(f"[DATA] enc_input : {file_input}")
    pf(f"[DATA] dec_target: {file_res}")
    pf(f"[DATA] labels    : {file_labels}")
    sys.stdout.flush()

    raw_input = np.load(file_input,  mmap_mode="r")
    raw_res   = np.load(file_res,    mmap_mode="r")
    raw_labels = np.load(file_labels, mmap_mode="r")
    train_idx  = np.load(file_train)
    val_idx    = np.load(file_val)
    test_idx   = np.load(file_test)

    N = raw_input.shape[0]
    pf(f"[DATA] total={N:,}  train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")
    sys.stdout.flush()

    # ── Normalise encoder input ────────────────────────────────────────────────
    pf("[DATA] Normalising encoder input...")
    train_enc = raw_input[train_idx]
    mu_enc    = float(np.mean(train_enc))
    std_enc   = float(np.std(train_enc))
    std_enc   = max(std_enc, 1e-6)
    pf(f"[DATA] enc mu={mu_enc:.6f}  std={std_enc:.6f}")
    sys.stdout.flush()

    # Expand dims: (N, seq_len) -> (N, seq_len, 1)
    if raw_input.ndim == 2:
        normalized_input = ((raw_input.astype(np.float32) - mu_enc) / std_enc)[:, :, np.newaxis]
    else:
        normalized_input = (raw_input.astype(np.float32) - mu_enc) / std_enc

    # res shape: (N, seq_len, n_out)
    res = raw_res.astype(np.float32)

    # ── Build teacher + cache ──────────────────────────────────────────────────
    pf("[TEACHER] Building teacher model...")
    sys.stdout.flush()

    with strategy.scope():
        teacher_model = build_teacher(
            args.seq_len, args.n_out, args.teacher_units, args.teacher_layers
        )

    teacher_model.load_weights(args.teacher_ckpt)
    pf(f"[TEACHER] Loaded weights from {args.teacher_ckpt}")
    sys.stdout.flush()

    teacher_hidden_model = build_teacher_hidden_model(teacher_model)

    teacher_predictions, teacher_hidden = cache_teacher_predictions_and_hidden(
        teacher_model=teacher_model,
        teacher_hidden_model=teacher_hidden_model,
        normalized_input=normalized_input,
        seq_len=args.seq_len,
        n_out=args.n_out,
        teacher_units=args.teacher_units,
        n_samples=N,
        infer_batch=args.infer_batch,
        data_dir=args.data_dir,
        pf=pf,
    )

    # ── Channel scales ─────────────────────────────────────────────────────────
    channel_scales = compute_channel_scales(teacher_predictions[train_idx], args.n_out, pf)

    # ── epsilon_innov ──────────────────────────────────────────────────────────
    epsilon_innov = compute_epsilon_innov(
    teacher_hidden[train_idx],
    args.seq_len,
    pf=pf,
)
    # ── Materialise buffers ────────────────────────────────────────────────────
    pf("[DATA] Materialising split buffers...")
    sys.stdout.flush()

    enc_train, tgt_train, tpred_train, teacher_hidden_train = materialise_buffers(
        normalized_input, res, teacher_predictions, teacher_hidden,
        train_idx, args.seq_len, args.n_out, "train", pf
    )
    enc_val, tgt_val, tpred_val, teacher_hidden_val = materialise_buffers(
        normalized_input, res, teacher_predictions, teacher_hidden,
        val_idx, args.seq_len, args.n_out, "val", pf
    )

    teacher_hidden_dim = teacher_hidden.shape[2]

    # ── tf.data pipelines ─────────────────────────────────────────────────────
    pf("[DATA] Building tf.data pipelines...")
    sys.stdout.flush()

    train_ds = make_kd_dataset(
        enc_train, tgt_train, tpred_train, teacher_hidden_train,
        args.batch_size, args.seq_len, args.n_out, teacher_hidden_dim,
        shuffle=True, seed=args.split_seed,
        prefetch_batches=args.prefetch_batches,
        pipeline_workers=args.pipeline_workers,
    )
    val_ds = make_kd_dataset(
        enc_val, tgt_val, tpred_val, teacher_hidden_val,
        args.batch_size, args.seq_len, args.n_out, teacher_hidden_dim,
        shuffle=False, seed=args.split_seed,
        prefetch_batches=args.prefetch_batches,
        pipeline_workers=args.pipeline_workers,
    )

    train_steps = len(enc_train) // args.batch_size
    val_steps   = len(enc_val)   // args.batch_size

    dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
    dist_val_ds   = strategy.experimental_distribute_dataset(val_ds)

    pf(f"[DATA] train_steps={train_steps}  val_steps={val_steps}")
    sys.stdout.flush()

    # ── Build models ──────────────────────────────────────────────────────────
    pf("[MODEL] Building float student (Phase 1)...")
    sys.stdout.flush()

    with strategy.scope():
        float_student = build_float_student(args.seq_len, args.n_out, args.student_units)

    pf("[MODEL] Building Phase 2 split-gate model...")
    sys.stdout.flush()

    with strategy.scope():
        phase2_model, enc_cell_p2, dec_cell_p2 = build_phase2_model(
            args.seq_len, args.n_out, args.student_units, input_dim=1
        )

    pf("[MODEL] Building final hard 4-bit QKeras student (Phase 3 / export)...")
    sys.stdout.flush()

    with strategy.scope():
        final_qkeras_student = build_final_qkeras_student(
            args.seq_len, args.n_out, args.student_units,
            args.bits_kernel, args.bits_recurrent, args.bits_bias,
            args.bits_activation, args.bits_state,
        )

    # ── Build models (print summaries) ────────────────────────────────────────
    pf("\n[MODEL] Float student summary:")
    float_student.summary(print_fn=pf)
    pf("\n[MODEL] Phase2 model summary:")
    phase2_model.summary(print_fn=pf)
    pf("\n[MODEL] Final QKeras student summary:")
    final_qkeras_student.summary(print_fn=pf)
    sys.stdout.flush()

    # ── Training ──────────────────────────────────────────────────────────────
    history, best_val = training_loop_memoq(
        strategy=strategy,
        float_student=float_student,
        final_qkeras_student=final_qkeras_student,
        enc_cell_p2=enc_cell_p2,
        dec_cell_p2=dec_cell_p2,
        phase2_model=phase2_model,
        args=args,
        dist_train_dataset=dist_train_ds,
        dist_val_dataset=dist_val_ds,
        train_steps=train_steps,
        val_steps=val_steps,
        channel_scales=channel_scales,
        epsilon_innov=epsilon_innov,
        job_dir=job_dir,
        pf=pf,
    )

    pf(f"[DONE] Best val loss: {best_val:.6f}")
    sys.stdout.flush()

    # ── Save final weights ─────────────────────────────────────────────────────
    final_save = os.path.join(job_dir, "student_final.weights.h5")
    final_qkeras_student.save_weights(final_save)
    pf(f"[SAVE] Final weights: {final_save}")
    sys.stdout.flush()

    # ── Plot training history ──────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        axes = axes.flatten()
        keys_to_plot = ["total", "seq", "kd", "mem", "innov", "zsat", "rail", "val_mae"]
        for i, key in enumerate(keys_to_plot):
            if key in history and history[key]:
                axes[i].plot(history[key], label=key)
            val_key = f"val_{key}" if not key.startswith("val_") else key
            if val_key in history and history[val_key]:
                axes[i].plot(history[val_key], label=val_key, linestyle="--")
            axes[i].set_title(key)
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        hist_png = os.path.join(job_dir, "training_history.png")
        plt.savefig(hist_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        pf(f"[PLOT] Training history: {hist_png}")
    except Exception as e:
        pf(f"[PLOT] Warning: could not save history plot: {e}")
    sys.stdout.flush()

    # ── Test evaluation ────────────────────────────────────────────────────────
    pf("[TEST] Evaluating on test set...")
    sys.stdout.flush()

    enc_test = normalized_input[test_idx]
    dec_test = np.zeros((len(test_idx), args.seq_len, 1), dtype=np.float32)
    tgt_test = res[test_idx]

    n_test   = len(test_idx)
    preds    = np.empty((n_test, args.seq_len, args.n_out), dtype=np.float32)
    n_batches = int(np.ceil(n_test / args.infer_batch))

    for b in range(n_batches):
        s = b * args.infer_batch
        e = min(s + args.infer_batch, n_test)
        enc_b = tf.constant(enc_test[s:e], dtype=tf.float32)
        dec_b = tf.constant(dec_test[s:e], dtype=tf.float32)
        out   = final_qkeras_student([enc_b, dec_b], training=False)
        preds[s:e] = out.numpy()

    mae_test  = float(np.mean(np.abs(preds - tgt_test)))
    rmse_test = float(np.sqrt(np.mean((preds - tgt_test) ** 2)))
    pf(f"[TEST] MAE={mae_test:.6f}  RMSE={rmse_test:.6f}")

    test_metrics = {"mae": mae_test, "rmse": rmse_test}
    with open(os.path.join(job_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    pf(f"[TEST] Metrics saved to {os.path.join(job_dir, 'test_metrics.json')}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
