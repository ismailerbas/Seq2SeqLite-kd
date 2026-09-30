#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


SEQ_LEN = 135
N_OUT = 3


def build_teacher():
    enc_input = keras.layers.Input(shape=(None, 1), name="encinput")

    enc_cells = [
        keras.layers.GRUCell(128, reset_after=True, name="enc_cell0"),
        keras.layers.GRUCell(128, reset_after=True, name="enc_cell1"),
    ]

    enc_rnn = keras.layers.RNN(
        enc_cells,
        return_state=True,
        name="encrnn",
    )

    enc_out = enc_rnn(enc_input)
    enc_states = enc_out[1:]

    dec_input = keras.layers.Input(shape=(None, 1), name="decinput")

    dec_cells = [
        keras.layers.GRUCell(128, reset_after=True, name="dec_cell0"),
        keras.layers.GRUCell(128, reset_after=True, name="dec_cell1"),
    ]

    dec_rnn = keras.layers.RNN(
        dec_cells,
        return_sequences=True,
        return_state=True,
        name="decrnn",
    )

    dec_out = dec_rnn(
        dec_input,
        initial_state=enc_states,
    )[0]

    output = keras.layers.Dense(
        N_OUT,
        activation="linear",
        name="decdense",
    )(dec_out)

    return keras.Model(
        [enc_input, dec_input],
        output,
        name="teacher_seq2seq",
    )


def build_student():

    def qwk():
        return quantized_bits(8, 0, 1, alpha=1.0)

    def qwr():
        return quantized_bits(8, 0, 1, alpha=1.0)

    def qwb():
        return quantized_bits(8, 0, 1, alpha=1.0)

    def qa():
        return quantized_tanh(bits=8, symmetric=True)

    def qs():
        return quantized_bits(8, 0, 1, alpha=1.0)

    def qd():
        return quantized_bits(8, 0)

    enc_input = keras.layers.Input(
        shape=(None, 1),
        name="senc_input",
    )

    dec_input = keras.layers.Input(
        shape=(None, 1),
        name="sdec_input",
    )

    _, enc_state = QGRU(
        units=32,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(enc_input)

    dec_seq, _ = QGRU(
        units=32,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_sequences=True,
        return_state=True,
        name="sdecgru",
    )(dec_input, initial_state=enc_state)

    output = QDense(
        N_OUT,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(dec_seq)

    return keras.Model(
        [enc_input, dec_input],
        output,
        name="student_vanilla_kd",
    )


def run_inference(model, x, batch_size, teacher=False):
    pred = np.empty(
        (len(x), SEQ_LEN, N_OUT),
        dtype=np.float32,
    )

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))

        xb = tf.convert_to_tensor(
            x[start:end],
            dtype=tf.float32,
        )

        db = tf.zeros(
            (end - start, SEQ_LEN, 1),
            dtype=tf.float32,
        )

        if teacher:
            out = model(
                {"encinput": xb, "decinput": db},
                training=False,
            )
        else:
            out = model(
                [xb, db],
                training=False,
            )

        pred[start:end] = out.numpy()

        print(f"  {end:,}/{len(x):,}", flush=True)

    return pred


def compute_metrics(gt_all, pred):
    names = [
        "ch0_full",
        "ch1_short",
        "ch2_long",
    ]

    results = {}

    for c, name in enumerate(names):
        gt = np.asarray(
            gt_all[:, :, c],
            dtype=np.float32,
        )

        pr = pred[:, :, c]

        residual = gt - pr

        rmse = float(
            np.sqrt(np.mean(residual ** 2))
        )

        ss_res = np.sum(
            residual ** 2,
            axis=1,
        )

        centered = (
            gt -
            gt.mean(axis=1, keepdims=True)
        )

        ss_tot = np.sum(
            centered ** 2,
            axis=1,
        )

        r2_each = np.empty(
            len(gt),
            dtype=np.float32,
        )

        valid = ss_tot > 1e-12

        r2_each[valid] = (
            1.0 -
            ss_res[valid] / ss_tot[valid]
        )

        r2_each[~valid] = np.where(
            ss_res[~valid] < 1e-12,
            1.0,
            0.0,
        )

        r2 = float(np.mean(r2_each))

        l2 = float(
            np.mean(
                np.sqrt(
                    np.sum(
                        residual ** 2,
                        axis=1,
                    )
                )
            )
        )

        results[name] = {
            "rmse": rmse,
            "r2": r2,
            "l2": l2,
        }

    return results


def print_metrics(name, metrics):
    print(f"\n{name}")

    for channel, values in metrics.items():
        print(
            f"{channel:10s}: "
            f"RMSE={values['rmse']:.6f}  "
            f"R2={values['r2']:.6f}  "
            f"L2={values['l2']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce synthetic held-out test metrics for "
            "the baseline teacher and selected Seq2SeqLite student."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing "
            "tpsf_seq_L135_test.npy and res_L135_test.npy."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    package_dir = script_dir.parent

    teacher_ckpt = (
        package_dir
        / "checkpoints"
        / "teacher_best_gru128x128.weights.h5"
    )

    student_ckpt = (
        package_dir
        / "checkpoints"
        / "student_32u_8bit_kd.weights.h5"
    )

    data_dir = Path(args.data_dir)

    input_file = (
        data_dir
        / "tpsf_seq_L135_test.npy"
    )

    target_file = (
        data_dir
        / "res_L135_test.npy"
    )

    for path in [
        input_file,
        target_file,
        teacher_ckpt,
        student_ckpt,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading held-out test arrays...")

    x = np.load(
        input_file,
        mmap_mode="r",
    )

    y = np.load(
        target_file,
        mmap_mode="r",
    )

    print("Input :", x.shape)
    print("Target:", y.shape)

    if x.shape != (160000, 135, 1):
        raise ValueError(
            f"Unexpected input shape: {x.shape}"
        )

    if y.shape != (160000, 135, 3):
        raise ValueError(
            f"Unexpected target shape: {y.shape}"
        )

    print("\nBuilding baseline teacher...")

    teacher = build_teacher()

    print(
        "Teacher parameters:",
        f"{teacher.count_params():,}",
    )

    if teacher.count_params() != 299139:
        raise RuntimeError(
            "Teacher parameter count mismatch."
        )

    teacher.load_weights(
        str(teacher_ckpt)
    )

    print("Teacher weights loaded.")

    print(
        "\nRunning teacher inference..."
    )

    teacher_pred = run_inference(
        teacher,
        x,
        args.batch_size,
        teacher=True,
    )

    teacher_metrics = compute_metrics(
        y,
        teacher_pred,
    )

    print_metrics(
        "BASELINE TEACHER",
        teacher_metrics,
    )

    del teacher_pred
    del teacher

    tf.keras.backend.clear_session()

    print(
        "\nBuilding selected 32-unit "
        "8-bit KD student..."
    )

    student = build_student()

    print(
        "Student parameters:",
        f"{student.count_params():,}",
    )

    if student.count_params() != 6627:
        raise RuntimeError(
            "Student parameter count mismatch."
        )

    student.load_weights(
        str(student_ckpt)
    )

    print("Student weights loaded.")

    print(
        "\nRunning student inference..."
    )

    student_pred = run_inference(
        student,
        x,
        args.batch_size,
        teacher=False,
    )

    student_metrics = compute_metrics(
        y,
        student_pred,
    )

    print_metrics(
        "SELECTED STUDENT",
        student_metrics,
    )

    paper_values = {
        "teacher_ch0_full": {
            "rmse": 0.014,
            "r2": 0.992,
            "l2": 0.113,
        },
        "student_ch0_full": {
            "rmse": 0.016,
            "r2": 0.989,
            "l2": 0.143,
        },
    }

    reproduced = {
        "teacher": teacher_metrics,
        "student": student_metrics,
        "paper_reported_rounded_values": paper_values,
        "n_test": int(len(x)),
    }

    output_path = (
        Path(args.output)
        if args.output
        else package_dir
        / "results"
        / "selected_model_metrics.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            reproduced,
            f,
            indent=2,
        )

    print(
        "\nPaper comparison using ch0_full:"
    )

    print(
        "Teacher reproduced rounded:",
        round(
            teacher_metrics[
                "ch0_full"
            ]["rmse"],
            3,
        ),
        round(
            teacher_metrics[
                "ch0_full"
            ]["r2"],
            3,
        ),
        round(
            teacher_metrics[
                "ch0_full"
            ]["l2"],
            3,
        ),
    )

    print(
        "Teacher paper reported:    ",
        "0.014 0.992 0.113",
    )

    print(
        "Student reproduced rounded:",
        round(
            student_metrics[
                "ch0_full"
            ]["rmse"],
            3,
        ),
        round(
            student_metrics[
                "ch0_full"
            ]["r2"],
            3,
        ),
        round(
            student_metrics[
                "ch0_full"
            ]["l2"],
            3,
        ),
    )

    print(
        "Student paper reported:    ",
        "0.016 0.989 0.143",
    )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
