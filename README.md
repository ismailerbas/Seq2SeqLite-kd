# Seq2SeqLite: Resource-Aware and Low-Precision Recurrent Inference

This repository contains Seq2Seq and Seq2SeqLite software for fluorescence lifetime inference, including model training, quantization, recurrent-state analysis, evaluation, and paper-specific reproducibility material.

## Publications and reviewer reproduction

This repository supports two complementary studies that use the same Seq2SeqLite model family and fluorescence lifetime inference framework.

### 1. When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference

**Ismail Erbas, Xavier Intes, and Vikas Pandey**

arXiv:2609.04490

This study examines how low-precision recurrent-state storage affects temporal inference in GRU and LSTM models, and evaluates recurrent-state write-back methods that preserve information lost through coarse state quantization.

**Reviewer reproduction:** [`docs/QUANTIZATION_PAPER_REPRODUCTION.md`](docs/QUANTIZATION_PAPER_REPRODUCTION.md)

### 2. Resource-Aware Hardware–Software Co-Design for Biomedical Imaging

This study evaluates Seq2SeqLite as a compact fluorescence lifetime inference model under hardware resource constraints, including knowledge distillation, quantization, experimental validation, FPGA synthesis, and resource-aware scheduling.

**Reviewer reproduction:** [`paper_reproduction/README.md`](paper_reproduction/README.md)

The software-only repository contains the model definitions, evaluation and verification scripts, configuration files, table-generation utilities, and expected results required to trace the reported results. Large evaluation datasets and model checkpoints are distributed separately through the manuscript's reviewer reproducibility package.

Hardware synthesis and resource-aware scheduling are maintained separately:

- **Vitis HLS reproduction:** [Seq2SeqLite-HLS](https://github.com/ismailerbas/Seq2SeqLite-HLS)
- **STOMP resource-aware scheduling:** [STOMP](https://github.com/ismailerbas/stomp/tree/meta)

## Synthetic fluorescence lifetime data

Synthetic fluorescence lifetime data used in this research were generated using the PyFLI fluorescence lifetime imaging framework.

PyFLI provides simulation and processing tools for fluorescence lifetime imaging and can be used to generate additional or custom synthetic FLI datasets:

[PyFLI repository](https://github.com/vkp217/pyfli-pkg)

The paper-specific reproduction workflows use frozen evaluation datasets so that reported numerical results can be compared directly. PyFLI is provided as the route for generating new simulated datasets rather than as a replacement for the frozen evaluation data used to reproduce the reported manuscript values.

## Repository structure

```text
configs/             Model and experiment configurations
docs/                Documentation and paper-specific guidance
eval/                Evaluation and recurrent-state analysis
slurm/               Cluster job templates
tables/              Table-generation utilities

paper_reproduction/  Resource-aware paper reproduction package

train_teacher.py
train_student_vanilla_kd.py
train_student_memoq.py
train_student_vanilla_kd_lstm.py
train_student_vanilla_kd_memory_campaign.py
train_student_vanilla_kd_scw.py
```

## General installation

General development dependencies are provided in `requirements.txt`.

Paper-specific reviewer environments and commands are documented separately in the corresponding reproduction guide:

- [`docs/QUANTIZATION_PAPER_REPRODUCTION.md`](docs/QUANTIZATION_PAPER_REPRODUCTION.md)
- [`paper_reproduction/README.md`](paper_reproduction/README.md)

Using the paper-specific instructions is recommended when reproducing reported manuscript results.

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
      url={https://arxiv.org/abs/2609.04490}
}
```

### Resource-Aware Hardware–Software Co-Design for Biomedical Imaging

Citation information will be added when a public manuscript record is available.
