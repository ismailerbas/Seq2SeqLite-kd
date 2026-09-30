# Reproduction package for "Resource-Aware Hardware–Software Co-Design for Biomedical Imaging"

This directory contains the paper-specific artifacts required to reproduce the
synthetic held-out test-set results for the baseline Seq2Seq teacher and the
selected Seq2SeqLite student reported in the manuscript.

## Reproduced models

### Baseline teacher

- Architecture: Seq2Seq
- Encoder: 2 GRU layers, 128 hidden units each
- Decoder: 2 GRU layers, 128 hidden units each
- Output channels: 3
- Parameters: 299,139
- Numerical form: 32-bit

### Selected student

- Architecture: Seq2SeqLite
- Encoder: 1 QGRU layer
- Decoder: 1 QGRU layer
- Hidden units: 32
- Output channels: 3
- Parameters: 6,627
- Kernel precision: 8-bit
- Recurrent precision: 8-bit
- Bias precision: 8-bit
- Activation precision: 8-bit
- State precision: 8-bit
- Knowledge distillation: yes
- KD alpha: 0.6
- Temperature: 4.0

The exact student training configuration is stored in:

    configs/student_32u_8bit_kd.json

## Frozen dataset split

The original synthetic dataset contains 1,600,000 samples.

The exact frozen split indices used for model development and evaluation are
provided in:

    splits/trainidx.npy
    splits/validx.npy
    splits/testidx.npy

Split sizes:

- training: 1,280,000 samples
- validation: 160,000 samples
- test: 160,000 samples

The three splits are mutually exclusive and together contain all 1,600,000
samples.

The held-out test split was not used for model training or checkpoint
selection.

## Checkpoints

The exact checkpoints used for the paper-specific reproduction are provided in:

    checkpoints/teacher_best_gru128x128.weights.h5
    checkpoints/student_32u_8bit_kd.weights.h5

The student checkpoint corresponds to the selected 32-unit, 8-bit,
knowledge-distilled Seq2SeqLite configuration.

## Test-set data required for reproduction

The default reproduction command uses test-only arrays derived from the frozen
test indices:

    tpsf_seq_L135_test.npy
    res_L135_test.npy

Expected shapes:

    tpsf_seq_L135_test.npy : (160000, 135, 1)
    res_L135_test.npy      : (160000, 135, 3)

These test-only arrays are intentionally not stored in the Git repository
because of their size. They are distributed with the paper reproduction data
package / Code Ocean capsule.

The test-only arrays were generated from the full synthetic arrays using the
provided `splits/testidx.npy`.

## Software environment

The paper training and evaluation environment used:

- Python 3
- TensorFlow 2.10.1
- NumPy 1.23.5
- QKeras
- CUDA 11.2
- cuDNN 8

The selected-model reproduction script can also run on CPU, although GPU
execution is faster.

Install the repository dependencies using:

    pip install -r requirements.txt

## One-command selected-model reproduction

From the repository root, run:

    python paper_reproduction/scripts/reproduce_selected_models.py \
        --data-dir /path/to/test_data

The directory supplied through `--data-dir` must contain:

    tpsf_seq_L135_test.npy
    res_L135_test.npy

The script automatically loads the paper-specific teacher and student
checkpoints from `paper_reproduction/checkpoints/`.

No checkpoint path needs to be edited manually.

## Expected output

For the full SFD output channel (`ch0_full`), the reproduction should report
approximately:

### Baseline Seq2Seq teacher

    RMSE = 0.013910
    R2   = 0.992053
    L2   = 0.112537

Rounded to the precision reported in the manuscript:

    RMSE = 0.014
    R2   = 0.992
    L2   = 0.113

### Selected Seq2SeqLite student

    RMSE = 0.016240
    R2   = 0.988987
    L2   = 0.142927

Rounded to the precision reported in the manuscript:

    RMSE = 0.016
    R2   = 0.989
    L2   = 0.143

The script also reports metrics for the short- and long-component SFD
channels.

## Metric definitions

Metrics are evaluated independently for each SFD output channel.

RMSE is pooled over all evaluated samples and temporal gates.

R2 is calculated for each sample across its temporal sequence and then averaged
over samples.

L2 is the Euclidean distance between predicted and reference temporal
sequences, calculated per sample and then averaged.

The manuscript comparison uses the full SFD output channel (`ch0_full`).

## Output file

The combined reproduction script writes:

    results/selected_model_metrics.json

Reference outputs from the verified reproduction run are also included in:

    results/teacher_test_sdf_metrics.json
    results/student_test_sdf_metrics.json
    results/selected_model_metrics.json

## Directory structure

    paper_reproduction/
    ├── README.md
    ├── checkpoints/
    │   ├── teacher_best_gru128x128.weights.h5
    │   └── student_32u_8bit_kd.weights.h5
    ├── configs/
    │   └── student_32u_8bit_kd.json
    ├── results/
    │   ├── teacher_test_sdf_metrics.json
    │   ├── student_test_sdf_metrics.json
    │   └── selected_model_metrics.json
    ├── scripts/
    │   ├── reproduce_selected_models.py
    │   ├── reproduce_teacher_test.py
    │   └── reproduce_student_test.py
    └── splits/
        ├── trainidx.npy
        ├── validx.npy
        └── testidx.npy

`reproduce_selected_models.py` is the recommended reviewer-facing entry point.

The two separate teacher/student scripts are retained as provenance from the
independent verification runs.

## Scope

This directory reproduces the selected neural-network checkpoint evaluation on
the frozen synthetic test set.

FPGA/Vitis HLS synthesis artifacts and STOMP scheduling experiments are
maintained separately because those workflows require different software and
hardware environments.
