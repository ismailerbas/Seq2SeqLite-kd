#!/usr/bin/env python3
"""
eval/submit_recurrent_memory_matrix.py

SLURM submission controller for the recurrent-memory study.

Stages
------
conditions
    Submit every frozen-checkpoint GPU condition required for the mechanism,
    residual-family, SCW, and equal-storage analyses. Completed conditions are
    skipped safely.

lifetime
    Submit the paired P2E-vs-P2F lifetime-binned excess-error analysis after
    both prerequisite conditions have completed.

aggregate
    Submit the final fail-closed aggregation after all condition and lifetime
    outputs exist.

This file never trains a model and never changes a checkpoint.
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


PROJECT_DIR = Path(
    "/gpfs/u/home/HBNN/HBNNrbss/scratch/nmi"
)

DATA_DIR = Path(
    "/gpfs/u/scratch/HBNN/HBNNrbss/nmi"
)

MEMOQ_RUN_DIR = PROJECT_DIR / (
    "results/"
    "paper_main_qkeras_matched_memoq_b4k4r4a4s4_gru32_dense3_"
    "effbs1024_microbs1024_lr1e-04_p1-40_2a30_2b30_2c10_2d10_"
    "2e30_2f30_p3-170_mse_q1_curr_nodither_aux0"
)

VANILLA4_RUN_DIR = PROJECT_DIR / (
    "results/"
    "vanilla_kd_T4.0_a0.6_b4k4r4a4s4_gru32x1_dense3_"
    "effbs1024_microbs1024_lr1e-04"
)

VANILLA8_RUN_DIR = PROJECT_DIR / (
    "results/"
    "vanilla_kd_T4.0_a0.6_b8k8r8a8_gru32x1_dense3_"
    "effbs1024_microbs1024_lr1e-04"
)

GPU_WORKER = (
    PROJECT_DIR
    / "slurm/analyze_recurrent_memory"
)

CPU_WORKER = (
    PROJECT_DIR
    / "slurm/analyze_recurrent_memory_cpu"
)

MEMOQ_ROOT = (
    MEMOQ_RUN_DIR
    / "recurrent_memory_analysis"
)

V4_ROOT = (
    VANILLA4_RUN_DIR
    / "recurrent_memory_analysis"
)

V8_ROOT = (
    VANILLA8_RUN_DIR
    / "recurrent_memory_analysis"
)

P2E_FIDELITY = (
    MEMOQ_RUN_DIR
    / "writeback_analysis/P2E_native/native_fidelity.json"
)

P2F_FIDELITY = (
    MEMOQ_RUN_DIR
    / "writeback_analysis/P2F_native/native_fidelity.json"
)

P3_FIDELITY = (
    MEMOQ_RUN_DIR
    / "writeback_analysis/P3_native/native_fidelity.json"
)

V4_FIDELITY = (
    VANILLA4_RUN_DIR
    / "writeback_analysis/VANILLA4S4_native/native_fidelity.json"
)

V8_FIDELITY = (
    VANILLA8_RUN_DIR
    / "writeback_analysis/VANILLA8_native/native_fidelity.json"
)

LIFETIME_DIR = (
    MEMOQ_ROOT
    / "lifetime_binned_P2F_vs_P2E"
)

STUDY_OUT_DIR = (
    MEMOQ_RUN_DIR
    / "recurrent_memory_study"
)


@dataclass(frozen=True)
class Condition:
    name: str
    run_dir: Path
    phase: str
    method: str
    state_bits: int
    fidelity_json: Path
    out_root: Path
    residual_bits: int = -1
    counter_bits: int = -1
    deadzone_fraction: float = 0.0

    @property
    def out_dir(self) -> Path:
        return (
            self.out_root
            / self.name
        )

    @property
    def complete_flag(self) -> Path:
        return (
            self.out_dir
            / "recurrent_memory_complete.flag"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the frozen recurrent-memory experiment matrix."
        )
    )

    parser.add_argument(
        "stage",
        choices=(
            "conditions",
            "lifetime",
            "aggregate",
        ),
    )

    return parser.parse_args()


def run_checked(command: List[str]) -> str:
    printable = " ".join(
        shlex.quote(part)
        for part in command
    )

    print(
        f"[RUN] {printable}",
        flush=True,
    )

    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.stderr.strip():
        print(
            completed.stderr.strip(),
            file=sys.stderr,
            flush=True,
        )

    return completed.stdout.strip()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: {path}"
        )


def static_checks() -> None:
    for directory in (
        PROJECT_DIR,
        DATA_DIR,
        MEMOQ_RUN_DIR,
        VANILLA4_RUN_DIR,
        VANILLA8_RUN_DIR,
    ):
        require_dir(directory)

    for path in (
        GPU_WORKER,
        CPU_WORKER,
        PROJECT_DIR
        / "eval/analyze_recurrent_memory.py",
        PROJECT_DIR
        / "eval/analyze_lifetime_binned_error.py",
        PROJECT_DIR
        / "eval/build_recurrent_memory_results.py",
    ):
        require_file(path)

    for fidelity in (
        P2E_FIDELITY,
        P2F_FIDELITY,
        P3_FIDELITY,
        V4_FIDELITY,
        V8_FIDELITY,
    ):
        require_file(fidelity)

    for run_dir in (
        MEMOQ_RUN_DIR,
        VANILLA4_RUN_DIR,
        VANILLA8_RUN_DIR,
    ):
        require_file(
            run_dir
            / "student_args.json"
        )

    for path in (
        MEMOQ_RUN_DIR
        / "stage2e_best.weights.h5",
        MEMOQ_RUN_DIR
        / "stage2f_best.weights.h5",
        MEMOQ_RUN_DIR
        / "student_best.weights.h5",
        VANILLA4_RUN_DIR
        / "student_best.weights.h5",
        VANILLA8_RUN_DIR
        / "student_best.weights.h5",
    ):
        require_file(path)

    run_checked(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(
                PROJECT_DIR
                / "eval/analyze_recurrent_memory.py"
            ),
            str(
                PROJECT_DIR
                / "eval/analyze_lifetime_binned_error.py"
            ),
            str(
                PROJECT_DIR
                / "eval/build_recurrent_memory_results.py"
            ),
            str(
                PROJECT_DIR
                / "eval/submit_recurrent_memory_matrix.py"
            ),
        ]
    )

    run_checked(
        [
            "bash",
            "-n",
            str(GPU_WORKER),
        ]
    )

    run_checked(
        [
            "bash",
            "-n",
            str(CPU_WORKER),
        ]
    )


def conditions() -> List[Condition]:
    result: List[Condition] = []

    result.append(
        Condition(
            name="P2E_identity",
            run_dir=MEMOQ_RUN_DIR,
            phase="P2E",
            method="identity",
            state_bits=4,
            fidelity_json=P2E_FIDELITY,
            out_root=MEMOQ_ROOT,
        )
    )

    for phase, fidelity in (
        (
            "P2F",
            P2F_FIDELITY,
        ),
        (
            "P3",
            P3_FIDELITY,
        ),
    ):
        result.extend(
            [
                Condition(
                    name=f"{phase}_det_B4",
                    run_dir=MEMOQ_RUN_DIR,
                    phase=phase,
                    method="deterministic",
                    state_bits=4,
                    fidelity_json=fidelity,
                    out_root=MEMOQ_ROOT,
                ),
                Condition(
                    name=f"{phase}_ef_B4",
                    run_dir=MEMOQ_RUN_DIR,
                    phase=phase,
                    method="error_feedback",
                    state_bits=4,
                    fidelity_json=fidelity,
                    out_root=MEMOQ_ROOT,
                ),
                Condition(
                    name=f"{phase}_sr_B4",
                    run_dir=MEMOQ_RUN_DIR,
                    phase=phase,
                    method="stochastic",
                    state_bits=4,
                    fidelity_json=fidelity,
                    out_root=MEMOQ_ROOT,
                ),
                Condition(
                    name=(
                        f"{phase}_"
                        "residual_FULL_HALFSTEP_B4"
                    ),
                    run_dir=MEMOQ_RUN_DIR,
                    phase=phase,
                    method=(
                        "full_halfstep_residual"
                    ),
                    state_bits=4,
                    fidelity_json=fidelity,
                    out_root=MEMOQ_ROOT,
                ),
            ]
        )

        for bits in (
            2,
            3,
            4,
        ):
            result.append(
                Condition(
                    name=(
                        f"{phase}_"
                        f"residual_R{bits}_B4"
                    ),
                    run_dir=MEMOQ_RUN_DIR,
                    phase=phase,
                    method="quantized_residual",
                    state_bits=4,
                    residual_bits=bits,
                    fidelity_json=fidelity,
                    out_root=MEMOQ_ROOT,
                )
            )

            for (
                deadzone_label,
                deadzone_fraction,
            ) in (
                (
                    "TH0",
                    0.0,
                ),
                (
                    "TH1_8",
                    0.125,
                ),
            ):
                result.append(
                    Condition(
                        name=(
                            f"{phase}_"
                            f"scw_K{bits}_"
                            f"{deadzone_label}_B4"
                        ),
                        run_dir=MEMOQ_RUN_DIR,
                        phase=phase,
                        method="scw",
                        state_bits=4,
                        counter_bits=bits,
                        deadzone_fraction=(
                            deadzone_fraction
                        ),
                        fidelity_json=fidelity,
                        out_root=MEMOQ_ROOT,
                    )
                )

    result.extend(
        [
            Condition(
                name="V4_det_B4",
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method="deterministic",
                state_bits=4,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            ),
            Condition(
                name="V4_ef_B4",
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method="error_feedback",
                state_bits=4,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            ),
            Condition(
                name="V4_sr_B4",
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method="stochastic",
                state_bits=4,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            ),
            Condition(
                name=(
                    "V4_residual_"
                    "FULL_HALFSTEP_B4"
                ),
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method=(
                    "full_halfstep_residual"
                ),
                state_bits=4,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            ),
        ]
    )

    for bits in (
        2,
        3,
        4,
    ):
        result.append(
            Condition(
                name=f"V4_residual_R{bits}_B4",
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method="quantized_residual",
                state_bits=4,
                residual_bits=bits,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            )
        )

        for (
            deadzone_label,
            deadzone_fraction,
        ) in (
            (
                "TH0",
                0.0,
            ),
            (
                "TH1_8",
                0.125,
            ),
        ):
            result.append(
                Condition(
                    name=(
                        f"V4_scw_K{bits}_"
                        f"{deadzone_label}_B4"
                    ),
                    run_dir=VANILLA4_RUN_DIR,
                    phase="VANILLA",
                    method="scw",
                    state_bits=4,
                    counter_bits=bits,
                    deadzone_fraction=(
                        deadzone_fraction
                    ),
                    fidelity_json=V4_FIDELITY,
                    out_root=V4_ROOT,
                )
            )

    for state_bits in (
        6,
        7,
        8,
    ):
        result.append(
            Condition(
                name=f"V4_det_B{state_bits}",
                run_dir=VANILLA4_RUN_DIR,
                phase="VANILLA",
                method="deterministic",
                state_bits=state_bits,
                fidelity_json=V4_FIDELITY,
                out_root=V4_ROOT,
            )
        )

    result.extend(
        [
            Condition(
                name="V8_det_B8",
                run_dir=VANILLA8_RUN_DIR,
                phase="VANILLA",
                method="deterministic",
                state_bits=8,
                fidelity_json=V8_FIDELITY,
                out_root=V8_ROOT,
            ),
            Condition(
                name="V8_forced_B4",
                run_dir=VANILLA8_RUN_DIR,
                phase="VANILLA",
                method="deterministic",
                state_bits=4,
                fidelity_json=V8_FIDELITY,
                out_root=V8_ROOT,
            ),
        ]
    )

    names = [
        condition.name
        for condition in result
    ]

    if len(names) != len(set(names)):
        raise RuntimeError(
            "Condition matrix contains duplicate names"
        )

    return result


def submit_condition(
    condition: Condition,
) -> None:
    condition.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if condition.complete_flag.is_file():
        print(
            f"[SKIP] complete: {condition.name}",
            flush=True,
        )
        return

    command = [
        "sbatch",
        "--parsable",
        f"--job-name=rm_{condition.name}",
        str(GPU_WORKER),
        str(condition.run_dir),
        condition.phase,
        condition.name,
        condition.method,
        str(condition.state_bits),
        str(condition.residual_bits),
        str(condition.counter_bits),
        format(
            condition.deadzone_fraction,
            ".9g",
        ),
        str(condition.out_dir),
        str(condition.fidelity_json),
    ]

    job_id = run_checked(
        command
    )

    print(
        f"[SUBMITTED] {condition.name}: {job_id}",
        flush=True,
    )


def stage_conditions() -> None:
    all_conditions = conditions()

    for root in (
        MEMOQ_ROOT,
        V4_ROOT,
        V8_ROOT,
    ):
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

    for condition in all_conditions:
        submit_condition(
            condition
        )

    print(
        "[DONE] condition submission pass complete. "
        "Re-run this stage safely after failed jobs; "
        "completed conditions are skipped.",
        flush=True,
    )


def require_condition_complete(
    path: Path,
) -> None:
    require_file(
        path
        / "recurrent_memory_complete.flag"
    )

    require_file(
        path
        / "recurrent_memory_per_sequence.npz"
    )

    require_file(
        path
        / "recurrent_memory_manifest.json"
    )


def stage_lifetime() -> None:
    reference_dir = (
        MEMOQ_ROOT
        / "P2E_identity"
    )

    condition_dir = (
        MEMOQ_ROOT
        / "P2F_det_B4"
    )

    require_condition_complete(
        reference_dir
    )

    require_condition_complete(
        condition_dir
    )

    LIFETIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_flag = (
        LIFETIME_DIR
        / "lifetime_binned_excess_error_complete.flag"
    )

    if complete_flag.is_file():
        print(
            "[SKIP] lifetime analysis complete: "
            f"{LIFETIME_DIR}",
            flush=True,
        )
        return

    command = [
        "sbatch",
        "--parsable",
        "--job-name=rm_lifetime",
        str(CPU_WORKER),
        "lifetime",
        str(
            reference_dir
            / "recurrent_memory_per_sequence.npz"
        ),
        str(
            condition_dir
            / "recurrent_memory_per_sequence.npz"
        ),
        str(
            reference_dir
            / "recurrent_memory_manifest.json"
        ),
        str(
            condition_dir
            / "recurrent_memory_manifest.json"
        ),
        str(LIFETIME_DIR),
    ]

    job_id = run_checked(
        command
    )

    print(
        f"[SUBMITTED] lifetime analysis: {job_id}",
        flush=True,
    )


def stage_aggregate() -> None:
    all_conditions = conditions()

    missing = []

    for condition in all_conditions:
        if not condition.complete_flag.is_file():
            missing.append(
                str(
                    condition.complete_flag
                )
            )

    lifetime_flag = (
        LIFETIME_DIR
        / "lifetime_binned_excess_error_complete.flag"
    )

    if not lifetime_flag.is_file():
        missing.append(
            str(lifetime_flag)
        )

    if missing:
        raise RuntimeError(
            "Aggregation prerequisites are incomplete:\n  "
            + "\n  ".join(missing)
        )

    STUDY_OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_flag = (
        STUDY_OUT_DIR
        / "recurrent_memory_study_complete.flag"
    )

    if complete_flag.is_file():
        print(
            "[SKIP] aggregation already complete: "
            f"{STUDY_OUT_DIR}",
            flush=True,
        )
        return

    command = [
        "sbatch",
        "--parsable",
        "--job-name=rm_aggregate",
        str(CPU_WORKER),
        "aggregate",
        str(MEMOQ_RUN_DIR),
        str(VANILLA4_RUN_DIR),
        str(VANILLA8_RUN_DIR),
        str(LIFETIME_DIR),
        str(STUDY_OUT_DIR),
    ]

    job_id = run_checked(
        command
    )

    print(
        f"[SUBMITTED] aggregation: {job_id}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    static_checks()

    if args.stage == "conditions":
        stage_conditions()
        return

    if args.stage == "lifetime":
        stage_lifetime()
        return

    if args.stage == "aggregate":
        stage_aggregate()
        return

    raise RuntimeError(
        f"Unhandled stage: {args.stage}"
    )


if __name__ == "__main__":
    main()