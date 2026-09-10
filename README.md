# Seq2SeqLite: Resource-Aware and Low-Precision Recurrent Inference

This repository contains the Seq2Seq and Seq2SeqLite training, quantization, evaluation, and recurrent-state analysis code used for fluorescence lifetime inference.

The codebase supports two complementary studies:

1. **When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference**  
   arXiv: https://arxiv.org/abs/2609.04490

2. **Resource-Aware Co-Design for Real-Time Biomedical Inference on Constrained Hardware**

Both studies use the Seq2SeqLite model family and the same fluorescence lifetime inference framework. The recurrent-state study focuses on low-precision state storage and recurrent-state write-back. The resource-aware co-design study uses the Seq2SeqLite model family for compact biomedical inference under constrained hardware resources.

## Repository structure

```text
configs/
docs/
eval/
slurm/
tables/

train_teacher.py
train_student_vanilla_kd.py
train_student_memoq.py
train_student_vanilla_kd_lstm.py
train_student_vanilla_kd_memory_campaign.py
train_student_vanilla_kd_scw.py

eval_experimental.py
predict_5_pixels_local.py
extract_student_weights.py
requirements.txt
```

Important recurrent-state analysis scripts in `eval/` include:

```text
analyze_lifetime_binned_error.py
analyze_lstm_state_writeback.py
analyze_memoq_deadzone.py
analyze_recurrent_memory.py
analyze_scw_sign_persistence.py
analyze_writeback.py
bench_student_timing.py
bench_teacher_timing.py
build_recurrent_memory_results.py
build_recurrent_training_campaign.py
build_v8_allocation_results.py
recurrent_memory_stats.py
validate_recurrent_memory_smoke.py
```

## Model family

The baseline Seq2Seq model is a stacked GRU encoder-decoder used for time-resolved fluorescence reconstruction and lifetime estimation.

Seq2SeqLite is the compact student model used in the low-precision studies. The configuration analyzed in the recurrent-state paper is a single-layer 32-unit GRU encoder-decoder with a linear dense readout and 6,627 trainable parameters.

Knowledge distillation transfers the output behavior of the larger Seq2Seq teacher to the compact Seq2SeqLite student.

## Fluorescence lifetime data

The recurrent-state study uses a synthetic fluorescence lifetime dataset generated with the open-source PyFLI framework:

https://github.com/vkp217/pyfli-pkg

The dataset contains 1,600,000 simulated time-resolved fluorescence signals, each represented by 135 temporal bins.

The fixed partition contains:

| Partition | Samples |
| --- | ---: |
| Training | 1,280,000 |
| Validation | 160,000 |
| Held-out test | 160,000 |

The partitions are defined by mutually exclusive index arrays. The same partition is used across model training, QMem, frozen write-back interventions, recurrent-state precision analyses, matched recurrent-memory training, and held-out evaluation.

Expected split-index filenames are:

```text
trainidx.npy or train_idx.npy
validx.npy or val_idx.npy
testidx.npy or test_idx.npy
```

### Simulation parameterization

Synthetic signals include mono-exponential and bi-exponential fluorescence decays.

For the bi-exponential simulations:

```text
0.05 ns <= tau1 <= 3 ns
0.05 ns <= tau2 <= 3 ns
tau2 >= tau1
0 <= a <= 1
```

where `a` is the fractional contribution of the short-lifetime component.

The simulations include photon statistics, detector-dependent noise, and measured pixel-wise instrument response functions. The IRFs were acquired from the imaging system using reflected 700 nm excitation from a white diffuse target with neutral-density attenuation to avoid detector saturation.

The full generated array does not need to be stored in the repository because the data are synthetic and can be regenerated with PyFLI using the study-specific parameterization described in the manuscript and Supplementary Information.

## QMem

`train_student_memoq.py` implements the staged QMem training trajectory used in the recurrent-state study.

QMem progressively introduces 4-bit quantization into the recurrent model:

```text
P2A    candidate kernels
P2B    reset-gate kernels
P2C    update-gate kernels
P2D    biases
P2E    candidate activation
P2F    recurrent-state write-back
P3     continued optimization on the fully quantized inference graph
```

Checkpoints around the P2E to P2F transition are used to isolate the effect of recurrent-state write-back while holding trained model parameters fixed.

## Recurrent-state write-back analysis

The recurrent-state study evaluates how the value computed at recurrent step `t` is stored and returned to step `t+1`.

The analyzed write-back conditions include:

- deterministic low-precision write-back
- continuous state propagation
- stochastic rounding
- error feedback
- quantized residual memory
- direction memory
- recurrent-state precision sweeps
- matched recurrent-memory training

### Recurrent write margin

For proposed state change `delta` and recurrent-state grid spacing `Delta_B`,

```text
M = 2 |delta| / Delta_B
```

For an interior grid point away from a rounding tie, `M < 1` means that the proposed change lies inside the deterministic half-step write boundary and does not change the recurrence-visible stored state.

### Recurrent-memory accounting

For the 32-unit recurrent state used in the GRU experiments, ideal bit-packed storage is:

| Recurrent-memory allocation | Bits per unit | Total bits | Total bytes |
| --- | ---: | ---: | ---: |
| 4-bit state | 4 | 128 | 16 |
| 6-bit state | 6 | 192 | 24 |
| 4-bit state plus 2 auxiliary bits | 6 | 192 | 24 |
| 8-bit state | 8 | 256 | 32 |
| 4-bit state plus 4 auxiliary bits | 8 | 256 | 32 |

These values describe recurrent-state storage only. Equal stored-bit counts do not imply equal arithmetic, logic, routing, latency, energy, or total model-memory cost.

## Reference models and cross-architecture analysis

The recurrent-state study uses independently trained 4-bit-state and 8-bit-state Seq2SeqLite GRUs as reference solutions.

An independently trained 32-unit LSTM is used for cross-architecture analysis. The LSTM evaluation changes cell-state and hidden-state write-back together and separately while the trained parameters remain fixed.

## Matched recurrent-memory training

`train_student_vanilla_kd_memory_campaign.py` compares recurrent-memory interfaces from matched initial trainable parameters.

The compared configurations are:

- 4-bit recurrence-visible state
- 6-bit recurrence-visible state
- 4-bit recurrence-visible state with 2-bit residual memory
- 4-bit recurrence-visible state with 2-bit direction memory

The campaign tests how a recurrent solution adapts when a particular state-storage interface is present throughout optimization.

## Evaluation

The principal recurrent-state evaluations use the complete 160,000-sample held-out test partition.

Reported quantities include:

- sequence mean absolute error
- short-lifetime RMSE
- long-lifetime RMSE
- Pearson correlation where applicable
- recurrent write margin
- write-deadband fraction
- recurrence-visible state-change fraction
- recurrent-state occupancy
- conditional sub-threshold write probability
- same-direction update persistence
- completed same-direction run length
- paired bootstrap uncertainty

## Training and benchmark environment

The Seq2Seq and Seq2SeqLite studies were run on a Slurm-managed compute node with:

- Intel Xeon Gold 6248 CPU at 2.50 GHz
- 755 GB host system memory
- NVIDIA Tesla V100-SXM2 GPU
- 32 GB HBM2 GPU memory
- compute capability 7.0
- NVIDIA driver 535.104.05

Training jobs used one V100 GPU with no data-parallel replication.

The reported software environment used:

```text
TensorFlow 2.10.1
NumPy 1.23.5
CUDA 11.2
cuDNN 8
cudatoolkit 11.3.1
cudnn 8.2.1
OpenBLAS
```

CPU benchmark jobs used 16 allocated CPU cores. CPU and GPU inference benchmarks used TensorFlow float32 execution, with QKeras reproducing the quantized student behavior within the TensorFlow execution environment.

## Installation

Clone the repository:

```bash
git clone https://github.com/ismailerbas/Seq2SeqLite-kd.git
cd Seq2SeqLite-kd
```

Create an isolated Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyFLI is available separately at:

https://github.com/vkp217/pyfli-pkg

## Reproducibility notes

Post-training recurrent-state intervention analyses keep the trained checkpoint fixed while changing only the recurrent-state write-back rule under study.

Before a post-training intervention is analyzed, the reconstructed native checkpoint is checked against the corresponding original implementation at the tensor and task-metric levels.

The QMem hardening sequence represents one training trajectory. The native 4-bit and native 8-bit GRU models are independently trained reference solutions. Matched recurrent-memory experiments use identical initial trainable parameters within each matched run.

## Citation

### When Quantization Breaks Memory

```bibtex
@misc{erbas2026quantizationbreaksmemoryrecurrentstate,
      title={When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference}, 
      author={Ismail Erbas and Xavier Intes and Vikas Pandey},
      year={2026},
      eprint={2609.04490},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.04490}, 
}
```

### Resource-Aware Co-Design for Real-Time Biomedical Inference on Constrained Hardware

The citation for this study will be added when its public record is available.

## Contact

Ismail Erbas  
Department of Biomedical Engineering  
Rensselaer Polytechnic Institute  
Troy, New York, USA  
erbasi@rpi.edu
