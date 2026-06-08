# Data Format

All data files are NumPy `.npy` arrays stored in a single flat directory (`--data-dir`).

## Required files

### Encoder input — `tpsf_seq_L{SEQ_LEN}_{N}M.npy`

Shape: `(N, SEQ_LEN, 1)` — dtype `float32`

Normalised TCSPC photon-count decay curve per sample, one time-bin per step.
`SEQ_LEN = 135` for the SS3 scanner (90 ps gate, 12.15 ns window).
Values are normalised to peak = 1 per sample before saving.

### Decoder target — `res_L{SEQ_LEN}_{N}M.npy`

Shape: `(N, SEQ_LEN, 3)` — dtype `float32`

Three-channel reconstruction target for the decoder:
- channel 0 — total (sum) decay
- channel 1 — short-lifetime component (τ₁)
- channel 2 — long-lifetime component (τ₂)

### Labels — `labels_3ch_L{SEQ_LEN}_{N}M.npy`

Shape: `(N, 3)` — dtype `float32`

Ground-truth scalar parameters per sample:
- column 0 — τ₁ (ns)
- column 1 — τ₂ (ns)
- column 2 — FRET fraction f ∈ [0, 1]

### Split index files

| File | Fallback name | Contents |
|---|---|---|
| `trainidx.npy` | `train_idx.npy` | Integer indices into the full dataset for training |
| `validx.npy` | `val_idx.npy` | Validation indices |
| `testidx.npy` | `test_idx.npy` | Test indices |

Split files are created automatically on first run of `train_teacher.py` if they do not
exist, using an 80/10/10 train/val/test split with `--split-seed 42`.

## Auto-generated cache files (do not commit)

These are written by `train_student.py` on first run and reused on subsequent runs.
Delete them whenever the teacher checkpoint changes.

| File | Shape | Created by |
|---|---|---|
| `teacherPred_L{SEQ_LEN}{N}.npy` | `(N, SEQ_LEN, 3)` | `cache_teacher_outputs` |
| `teacherHidden_L{SEQ_LEN}{N}.npy` | `(N, SEQ_LEN, 128)` | `cache_teacher_outputs` |
| `fisherDiag_L{SEQ_LEN}{N}.npy` | `(128,)` | `compute_or_load_fisher` |
| `teacherPredvanillaKD_L{SEQ_LEN}{N}.npy` | `(N, SEQ_LEN, 3)` | `cache_teacher_predictions` (vanilla KD) |

## File naming convention

The `{N}M` suffix in filenames (e.g. `_8M`) is the nominal sample count in millions, used
only for file discovery. The actual `N` is read from `array.shape[0]` at load time. If your
actual N differs from the filename suffix, a warning is printed but training proceeds.
