import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from qkeras import QDense, QGRU, quantized_bits, quantized_tanh


DATA_DIR = "/gpfs/u/scratch/HBNN/HBNNrbss/nmi"

CKPT = (
    "/gpfs/u/scratch/HBNN/HBNNrbss/nmi/exptests/"
    "vanilla_kd_T4.0_a0.6_b8k8r8a8_gru32x1_dense3_"
    "effbs1024_microbs1024_lr1e-04bu/"
    "student_final.weights.h5"
)

SEQ_LEN = 135
N_OUT = 3
BATCH = 8192


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

    enc_inputs = Input(shape=(None, 1), name="senc_input")
    dec_inputs = Input(shape=(None, 1), name="sdec_input")

    _, enc_state = QGRU(
        units=32,
        activation=qa(),
        kernel_quantizer=qwk(),
        recurrent_quantizer=qwr(),
        bias_quantizer=qwb(),
        state_quantizer=qs(),
        return_state=True,
        name="sencgru",
    )(enc_inputs)

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
    )(dec_inputs, initial_state=enc_state)

    output = QDense(
        N_OUT,
        kernel_quantizer=qd(),
        bias_quantizer=qd(),
        activation="linear",
        name="sdec_dense",
    )(dec_seq)

    return Model(
        [enc_inputs, dec_inputs],
        output,
        name="student_vanilla_kd",
    )


print("Loading test-only arrays...")

x = np.load(
    f"{DATA_DIR}/tpsf_seq_L135_test.npy",
    mmap_mode="r",
)

y = np.load(
    f"{DATA_DIR}/res_L135_test.npy",
    mmap_mode="r",
)

print("Input :", x.shape)
print("Target:", y.shape)


print("\nBuilding student...")
model = build_student()

print("Parameters:", f"{model.count_params():,}")
assert model.count_params() == 6627

print("Loading:", CKPT)
model.load_weights(CKPT)
model.trainable = False
print("Weights loaded successfully.")


print("\nRunning inference...")

pred = np.empty(
    (len(x), SEQ_LEN, N_OUT),
    dtype=np.float32
)

for start in range(0, len(x), BATCH):

    end = min(start + BATCH, len(x))

    xb = tf.convert_to_tensor(
        x[start:end],
        dtype=tf.float32
    )

    db = tf.zeros(
        (end - start, SEQ_LEN, 1),
        dtype=tf.float32
    )

    pred[start:end] = model(
        [xb, db],
        training=False
    ).numpy()

    print(f"{end:,}/{len(x):,}")


print("\nSDF TEST METRICS")

names = [
    "ch0_full",
    "ch1_short",
    "ch2_long"
]

results = {}

for c, name in enumerate(names):

    gt = np.asarray(
        y[:, :, c],
        dtype=np.float32
    )

    pr = pred[:, :, c]

    rmse = float(
        np.sqrt(
            np.mean((gt - pr) ** 2)
        )
    )

    ss_res = np.sum(
        (gt - pr) ** 2,
        axis=1
    )

    ss_tot = np.sum(
        (gt - gt.mean(axis=1, keepdims=True)) ** 2,
        axis=1
    )

    r2_each = np.empty(
        len(gt),
        dtype=np.float32
    )

    valid = ss_tot > 1e-12

    r2_each[valid] = (
        1.0 -
        ss_res[valid] / ss_tot[valid]
    )

    r2_each[~valid] = np.where(
        ss_res[~valid] < 1e-12,
        1.0,
        0.0
    )

    r2 = float(np.mean(r2_each))

    l2 = float(
        np.mean(
            np.sqrt(
                np.sum(
                    (gt - pr) ** 2,
                    axis=1
                )
            )
        )
    )

    results[name] = {
        "rmse": rmse,
        "r2": r2,
        "l2": l2,
    }

    print(
        f"{name:10s}: "
        f"RMSE={rmse:.6f}  "
        f"R2={r2:.6f}  "
        f"L2={l2:.6f}"
    )


with open(
    f"{DATA_DIR}/student_test_sdf_metrics.json",
    "w"
) as f:
    json.dump(results, f, indent=2)

print("\nSaved student_test_sdf_metrics.json")