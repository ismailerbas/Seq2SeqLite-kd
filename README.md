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
