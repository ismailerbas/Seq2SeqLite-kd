# Model Architecture

## Teacher — `build_teacher()`

Stacked two-layer GRU seq2seq. Standard float32, no quantisation.

```
Encoder:
  Input  "encinput"   (B, T, 1)
  RNN    "encrnn"     stacked [GRUCell(128), GRUCell(128)], return_state=True
  → encoder_states = [h1_T, h2_T]   each (B, 128)

Decoder:
  Input  "decinput"   (B, T, 1)   ← zeros at inference
  RNN    "decrnn"     stacked [GRUCell(128), GRUCell(128)],
                      return_sequences=True, return_state=True,
                      initial_state=encoder_states
  → decoder_hidden_sequence  (B, T, 128)
  Dense  "decdense"   Dense(3, activation="linear")
  → output  (B, T, 3)
```

**Layer naming convention** (matches `train_teacher.py` weight file):
`encinput`, `decinput`, `encrnn`, `decrnn`, `decdense` — no underscores.

Both student scripts must use these exact names when rebuilding the teacher for
weight loading, or `load_weights()` will silently skip all weights.

---

## Student — `build_student()`

Single-layer QGRU seq2seq. 4-bit quantised weights, activations, and hidden state.

```
Encoder:
  Input  "sencinput"  (B, T, 1)
  QGRU   "sencgru"    units=32, return_state=True
  → s_enc_state  (B, 32)

Decoder:
  Input  "sdecinput"  (B, T, 1)  ← zeros at inference
  QGRU   "sdecgru"    units=32, return_sequences=True, return_state=True
                      initial_state=s_enc_state
  → s_dec_hid_seq  (B, T, 32)
  QDense "sdecdense"  Dense(3, activation="linear")
  → output  (B, T, 3)
```

Quantiser defaults (all configurable via CLI):
- kernel / bias / recurrent: `quantized_bits(4, 0, 1, alpha=1.0)`
- activation: `quantized_tanh(bits=4, symmetric=True)`
- state: `quantized_bits(4, 0, 1, alpha=1.0)`

---

## FW-QATD-RAC student extras

Only in `train_student.py`:

- **Float shadow student** (`build_float_shadow_student`): identical topology to student but
  standard `GRU` (no QKeras). Used only to compute L_RAC. Weights synced from quantised
  student every `--shadow-sync-every` epochs.

- **Projection layer** (`Dense(teacher_units, use_bias=False)`): maps student hidden
  trajectory (B, T, 32) → (B, T, 128) for comparison against teacher hidden trajectory in
  L_traj. Discarded after training; not part of the deployed model.

- **Teacher hidden model** (`build_teacher_hidden_model`): taps `decrnn.output[0]` to
  expose the (B, T, 128) hidden trajectory for L_traj computation and Fisher diagonal.

---

## Loss functions

### Teacher
L = MSE(y_pred, y_true)

### FW-QATD-RAC student
```
L_total = L_GT + α·L_KD + β·L_traj + γ·L_RAC

L_GT   = MSE(y_student, y_true)
L_KD   = MSE(y_student, y_teacher)
L_traj = mean_t( F ⊙ ||P·h_quant - h_teacher||² )   Fisher-weighted
L_RAC  = mean_t( t_weights · mean_d(||h_quant - sg(h_float)||²) )
```

Defaults: α=0.5, β=0.05, γ=1e-3

### Vanilla KD student (**BUG — see README**)
Currently implemented as KL divergence with softmax (incorrect for regression).
Should be:
```
L_total = α·T²·MSE(y_student, y_teacher) + (1-α)·MSE(y_student, y_true)
```
