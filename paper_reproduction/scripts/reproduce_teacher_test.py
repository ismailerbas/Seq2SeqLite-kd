import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/gpfs/u/scratch/HBNN/HBNNrbss/nmi"
CKPT = (
    "/gpfs/u/scratch/HBNN/HBNNrbss/nmi/"
    "teacher_training_gru128x128/"
    "teacher_best_gru128x128.weights.h5"
)

SEQ_LEN = 135
N_OUT = 3
BATCH = 8192


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

    dec_out = dec_rnn(dec_input, initial_state=enc_states)[0]

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

assert x.shape == (160000, 135, 1)
assert y.shape == (160000, 135, 3)


print("\nBuilding teacher...")
model = build_teacher()

print("Parameters:", f"{model.count_params():,}")
assert model.count_params() == 299139

print("Loading:", CKPT)
model.load_weights(CKPT)
model.trainable = False
print("Weights loaded successfully.")


print("\nRunning inference...")

pred = np.empty((len(x), SEQ_LEN, N_OUT), dtype=np.float32)

for start in range(0, len(x), BATCH):
    end = min(start + BATCH, len(x))

    xb = tf.convert_to_tensor(x[start:end], dtype=tf.float32)
    db = tf.zeros((end - start, SEQ_LEN, 1), dtype=tf.float32)

    pred[start:end] = model(
        {"encinput": xb, "decinput": db},
        training=False,
    ).numpy()

    print(f"{end:,}/{len(x):,}")


print("\nSDF TEST METRICS")

names = ["ch0_full", "ch1_short", "ch2_long"]
results = {}

for c, name in enumerate(names):

    gt = np.asarray(y[:, :, c], dtype=np.float32)
    pr = pred[:, :, c]

    rmse = float(np.sqrt(np.mean((gt - pr) ** 2)))

    ss_res = np.sum((gt - pr) ** 2, axis=1)
    ss_tot = np.sum(
        (gt - gt.mean(axis=1, keepdims=True)) ** 2,
        axis=1,
    )

    r2_each = np.where(
        ss_tot > 1e-12,
        1.0 - ss_res / ss_tot,
        np.where(ss_res < 1e-12, 1.0, 0.0),
    )

    r2 = float(np.mean(r2_each))

    l2_each = np.sqrt(
        np.sum((gt - pr) ** 2, axis=1)
    )
    l2 = float(np.mean(l2_each))

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
    f"{DATA_DIR}/teacher_test_sdf_metrics.json",
    "w",
) as f:
    json.dump(results, f, indent=2)

print("\nSaved teacher_test_sdf_metrics.json")