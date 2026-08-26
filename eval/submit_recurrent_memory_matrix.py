#!/usr/bin/env python3
"""
eval/submit_recurrent_memory_matrix.py

SLURM submission controller for the frozen recurrent-memory study.

Stages
------
smoke
    Submit only V4_det_B4. This is the mandatory end-to-end reconstruction
    smoke test.

smoke-check
    Compare the completed V4_det_B4 result against the repository's already
    validated VANILLA4S4_native writeback analysis. The full matrix is blocked
    until this check passes.

conditions
    Submit the complete frozen-checkpoint condition matrix. Completed
    conditions are skipped safely. This stage requires a passed smoke check.

lifetime
    Submit two same-checkpoint paired lifetime analyses:
        P2E_identity vs P2E_det_B4
        P2F_identity vs P2F_det_B4

aggregate
    Submit the final fail-closed aggregation after every required condition,
    both lifetime analyses, and the smoke validation are complete.

No stage trains or modifies a model.
"""

import argparse
import shlex
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import (
    List,
    Sequence,
)


PROJECT_DIR = Path(
    "/gpfs/u/home/HBNN/HBNNrbss/scratch/nmi"
)

DATA_DIR = Path(
    "/gpfs/u/scratch/HBNN/HBNNrbss/nmi"
)

MEMOQ_RUN_DIR = (
    PROJECT_DIR
    / (
        "results/"
        "paper_main_qkeras_matched_memoq_"
        "b4k4r4a4s4_gru32_dense3_effbs1024_"
        "microbs1024_lr1e-04_p1-40_2a30_"
        "2b30_2c10_2d10_2e30_2f30_p3-170_"
        "mse_q1_curr_nodither_aux0"
    )
)

VANILLA4_RUN_DIR = (
    PROJECT_DIR
    / (
        "results/"
        "vanilla_kd_T4.0_a0.6_"
        "b4k4r4a4s4_gru32x1_dense3_"
        "effbs1024_microbs1024_lr1e-04"
    )
)

VANILLA8_RUN_DIR = (
    PROJECT_DIR
    / (
        "results/"
        "vanilla_kd_T4.0_a0.6_"
        "b8k8r8a8_gru32x1_dense3_"
        "effbs1024_microbs1024_lr1e-04"
    )
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
    / (
        "writeback_analysis/"
        "P2E_native/"
        "native_fidelity.json"
    )
)

P2F_FIDELITY = (
    MEMOQ_RUN_DIR
    / (
        "writeback_analysis/"
        "P2F_native/"
        "native_fidelity.json"
    )
)

P3_FIDELITY = (
    MEMOQ_RUN_DIR
    / (
        "writeback_analysis/"
        "P3_native/"
        "native_fidelity.json"
    )
)

V4_FIDELITY = (
    VANILLA4_RUN_DIR
    / (
        "writeback_analysis/"
        "VANILLA4S4_native/"
        "native_fidelity.json"
    )
)

V8_FIDELITY = (
    VANILLA8_RUN_DIR
    / (
        "writeback_analysis/"
        "VANILLA8_native/"
        "native_fidelity.json"
    )
)

V4_NATIVE_REFERENCE_DIR = (
    VANILLA4_RUN_DIR
    / (
        "writeback_analysis/"
        "VANILLA4S4_native"
    )
)

SMOKE_DIR = (
    V4_ROOT
    / "smoke_validation"
)

SMOKE_FLAG = (
    SMOKE_DIR
    / "recurrent_memory_smoke_validation_complete.flag"
)

LIFETIME_P2E_DIR = (
    MEMOQ_ROOT
    / "lifetime_binned_P2E_det_vs_identity"
)

LIFETIME_P2F_DIR = (
    MEMOQ_ROOT
    / "lifetime_binned_P2F_det_vs_identity"
)

STUDY_OUT_DIR = (
    MEMOQ_RUN_DIR
    / "recurrent_memory_study"
)


@dataclass(
    frozen=True
)
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
    def out_dir(
        self,
    ) -> Path:
        return (
            self.out_root
            / self.name
        )

    @property
    def complete_flag(
        self,
    ) -> Path:
        return (
            self.out_dir
            / "recurrent_memory_complete.flag"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit and validate the frozen "
            "recurrent-memory experiment matrix."
        )
    )

    parser.add_argument(
        "stage",
        choices=(
            "smoke",
            "smoke-check",
            "conditions",
            "lifetime",
            "aggregate",
        ),
    )

    return parser.parse_args()


def run_checked(
    command: Sequence[
        str
    ],
) -> str:
    printable = " ".join(
        shlex.quote(
            str(
                part
            )
        )
        for part
        in command
    )

    print(
        f"[RUN] {printable}",
        flush=True,
    )

    completed = subprocess.run(
        [
            str(
                part
            )
            for part
            in command
        ],
        check=True,
        stdout=(
            subprocess.PIPE
        ),
        stderr=(
            subprocess.PIPE
        ),
        text=True,
    )

    if (
        completed.stderr.strip()
    ):
        print(
            completed.stderr.strip(),
            file=sys.stderr,
            flush=True,
        )

    return (
        completed.stdout.strip()
    )


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: "
            f"{path}"
        )


def require_dir(
    path: Path,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: "
            f"{path}"
        )


def static_checks() -> None:
    for directory in (
        PROJECT_DIR,
        DATA_DIR,
        MEMOQ_RUN_DIR,
        VANILLA4_RUN_DIR,
        VANILLA8_RUN_DIR,
    ):
        require_dir(
            directory
        )

    required_scripts = (
        GPU_WORKER,
        CPU_WORKER,
        PROJECT_DIR
        / "eval/analyze_recurrent_memory.py",
        PROJECT_DIR
        / "eval/analyze_lifetime_binned_error.py",
        PROJECT_DIR
        / "eval/build_recurrent_memory_results.py",
        PROJECT_DIR
        / "eval/recurrent_memory_stats.py",
        PROJECT_DIR
        / "eval/validate_recurrent_memory_smoke.py",
        PROJECT_DIR
        / "eval/submit_recurrent_memory_matrix.py",
    )

    for path in (
        required_scripts
    ):
        require_file(
            path
        )

    for fidelity in (
        P2E_FIDELITY,
        P2F_FIDELITY,
        P3_FIDELITY,
        V4_FIDELITY,
        V8_FIDELITY,
    ):
        require_file(
            fidelity
        )

    for run_dir in (
        MEMOQ_RUN_DIR,
        VANILLA4_RUN_DIR,
        VANILLA8_RUN_DIR,
    ):
        require_file(
            run_dir
            / "student_args.json"
        )

    for checkpoint in (
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
        require_file(
            checkpoint
        )

    for path in (
        V4_NATIVE_REFERENCE_DIR
        / "writeback_summary.json",
        V4_NATIVE_REFERENCE_DIR
        / "writeback_manifest.json",
        V4_NATIVE_REFERENCE_DIR
        / "native_fidelity.json",
    ):
        require_file(
            path
        )

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
                / "eval/recurrent_memory_stats.py"
            ),
            str(
                PROJECT_DIR
                / "eval/validate_recurrent_memory_smoke.py"
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
            str(
                GPU_WORKER
            ),
        ]
    )

    run_checked(
        [
            "bash",
            "-n",
            str(
                CPU_WORKER
            ),
        ]
    )


def conditions() -> List[
    Condition
]:
    result: List[
        Condition
    ] = []

    result.extend(
        [
            Condition(
                name=(
                    "P2E_identity"
                ),
                run_dir=(
                    MEMOQ_RUN_DIR
                ),
                phase="P2E",
                method="identity",
                state_bits=4,
                fidelity_json=(
                    P2E_FIDELITY
                ),
                out_root=(
                    MEMOQ_ROOT
                ),
            ),
            Condition(
                name=(
                    "P2E_det_B4"
                ),
                run_dir=(
                    MEMOQ_RUN_DIR
                ),
                phase="P2E",
                method=(
                    "deterministic"
                ),
                state_bits=4,
                fidelity_json=(
                    P2E_FIDELITY
                ),
                out_root=(
                    MEMOQ_ROOT
                ),
            ),
        ]
    )

    for (
        phase,
        fidelity,
    ) in (
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
                    name=(
                        f"{phase}_identity"
                    ),
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "identity"
                    ),
                    state_bits=4,
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
                ),
                Condition(
                    name=(
                        f"{phase}_det_B4"
                    ),
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "deterministic"
                    ),
                    state_bits=4,
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
                ),
                Condition(
                    name=(
                        f"{phase}_ef_B4"
                    ),
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "error_feedback"
                    ),
                    state_bits=4,
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
                ),
                Condition(
                    name=(
                        f"{phase}_sr_B4"
                    ),
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "stochastic"
                    ),
                    state_bits=4,
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
                ),
                Condition(
                    name=(
                        f"{phase}_residual_"
                        "FULL_HALFSTEP_B4"
                    ),
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "full_halfstep_residual"
                    ),
                    state_bits=4,
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
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
                    run_dir=(
                        MEMOQ_RUN_DIR
                    ),
                    phase=(
                        phase
                    ),
                    method=(
                        "quantized_residual"
                    ),
                    state_bits=4,
                    residual_bits=(
                        bits
                    ),
                    fidelity_json=(
                        fidelity
                    ),
                    out_root=(
                        MEMOQ_ROOT
                    ),
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
                        run_dir=(
                            MEMOQ_RUN_DIR
                        ),
                        phase=(
                            phase
                        ),
                        method=(
                            "scw"
                        ),
                        state_bits=4,
                        counter_bits=(
                            bits
                        ),
                        deadzone_fraction=(
                            deadzone_fraction
                        ),
                        fidelity_json=(
                            fidelity
                        ),
                        out_root=(
                            MEMOQ_ROOT
                        ),
                    )
                )

    result.extend(
        [
            Condition(
                name="V4_det_B4",
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "deterministic"
                ),
                state_bits=4,
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
            ),
            Condition(
                name="V4_ef_B4",
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "error_feedback"
                ),
                state_bits=4,
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
            ),
            Condition(
                name="V4_sr_B4",
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "stochastic"
                ),
                state_bits=4,
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
            ),
            Condition(
                name=(
                    "V4_residual_"
                    "FULL_HALFSTEP_B4"
                ),
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "full_halfstep_residual"
                ),
                state_bits=4,
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
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
                    f"V4_residual_R{bits}_B4"
                ),
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "quantized_residual"
                ),
                state_bits=4,
                residual_bits=(
                    bits
                ),
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
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
                    run_dir=(
                        VANILLA4_RUN_DIR
                    ),
                    phase="VANILLA",
                    method="scw",
                    state_bits=4,
                    counter_bits=(
                        bits
                    ),
                    deadzone_fraction=(
                        deadzone_fraction
                    ),
                    fidelity_json=(
                        V4_FIDELITY
                    ),
                    out_root=(
                        V4_ROOT
                    ),
                )
            )

    for state_bits in (
        6,
        7,
        8,
    ):
        result.append(
            Condition(
                name=(
                    f"V4_det_B{state_bits}"
                ),
                run_dir=(
                    VANILLA4_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "deterministic"
                ),
                state_bits=(
                    state_bits
                ),
                fidelity_json=(
                    V4_FIDELITY
                ),
                out_root=(
                    V4_ROOT
                ),
            )
        )

    result.extend(
        [
            Condition(
                name="V8_det_B8",
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "deterministic"
                ),
                state_bits=8,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
            Condition(
                name=(
                    "V8_forced_B4"
                ),
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "deterministic"
                ),
                state_bits=4,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
            Condition(
                name="V8_ef_B4",
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "error_feedback"
                ),
                state_bits=4,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
            Condition(
                name=(
                    "V8_residual_R2_B4"
                ),
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "quantized_residual"
                ),
                state_bits=4,
                residual_bits=2,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
            Condition(
                name=(
                    "V8_residual_R4_B4"
                ),
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method=(
                    "quantized_residual"
                ),
                state_bits=4,
                residual_bits=4,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
            Condition(
                name=(
                    "V8_scw_K4_TH1_8_B4"
                ),
                run_dir=(
                    VANILLA8_RUN_DIR
                ),
                phase="VANILLA",
                method="scw",
                state_bits=4,
                counter_bits=4,
                deadzone_fraction=0.125,
                fidelity_json=(
                    V8_FIDELITY
                ),
                out_root=(
                    V8_ROOT
                ),
            ),
        ]
    )

    names = [
        condition.name
        for condition
        in result
    ]

    if (
        len(
            names
        )
        != len(
            set(
                names
            )
        )
    ):
        raise RuntimeError(
            "Condition matrix contains duplicate names"
        )

    return result


def condition_by_name(
    name: str,
) -> Condition:
    matches = [
        condition
        for condition
        in conditions()
        if condition.name
        == name
    ]

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Expected exactly one "
            f"condition named {name}, "
            f"found {len(matches)}"
        )

    return matches[
        0
    ]


def submit_condition(
    condition: Condition,
) -> None:
    condition.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        condition.complete_flag.is_file()
    ):
        print(
            f"[SKIP] complete: "
            f"{condition.name}",
            flush=True,
        )

        return

    command = [
        "sbatch",
        "--parsable",
        (
            "--job-name="
            f"rm_{condition.name}"
        ),
        str(
            GPU_WORKER
        ),
        str(
            condition.run_dir
        ),
        condition.phase,
        condition.name,
        condition.method,
        str(
            condition.state_bits
        ),
        str(
            condition.residual_bits
        ),
        str(
            condition.counter_bits
        ),
        format(
            condition.deadzone_fraction,
            ".9g",
        ),
        str(
            condition.out_dir
        ),
        str(
            condition.fidelity_json
        ),
    ]

    job_id = run_checked(
        command
    )

    print(
        f"[SUBMITTED] "
        f"{condition.name}: "
        f"{job_id}",
        flush=True,
    )


def stage_smoke() -> None:
    V4_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    smoke_condition = (
        condition_by_name(
            "V4_det_B4"
        )
    )

    submit_condition(
        smoke_condition
    )

    print(
        "[NEXT] after V4_det_B4 "
        "finishes successfully, run: "
        "python "
        "eval/submit_recurrent_memory_matrix.py "
        "smoke-check",
        flush=True,
    )


def stage_smoke_check() -> None:
    smoke_condition = (
        condition_by_name(
            "V4_det_B4"
        )
    )

    require_file(
        smoke_condition.complete_flag
    )

    SMOKE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    script = (
        PROJECT_DIR
        / "eval/validate_recurrent_memory_smoke.py"
    )

    run_checked(
        [
            sys.executable,
            str(
                script
            ),
            "--reference-dir",
            str(
                V4_NATIVE_REFERENCE_DIR
            ),
            "--new-dir",
            str(
                smoke_condition.out_dir
            ),
            "--out-dir",
            str(
                SMOKE_DIR
            ),
            "--absolute-tolerance",
            "1e-6",
        ]
    )

    require_file(
        SMOKE_FLAG
    )

    print(
        "[PASS] smoke validation passed. "
        "Full condition matrix is authorized.",
        flush=True,
    )


def stage_conditions() -> None:
    require_file(
        SMOKE_FLAG
    )

    for root in (
        MEMOQ_ROOT,
        V4_ROOT,
        V8_ROOT,
    ):
        root.mkdir(
            parents=True,
            exist_ok=True,
        )

    for condition in (
        conditions()
    ):
        submit_condition(
            condition
        )

    print(
        "[DONE] condition submission "
        "pass complete. Re-running "
        "this stage is safe; completed "
        "conditions are skipped.",
        flush=True,
    )


def require_condition_complete(
    condition_name: str,
) -> Condition:
    condition = (
        condition_by_name(
            condition_name
        )
    )

    require_file(
        condition.complete_flag
    )

    require_file(
        condition.out_dir
        / "recurrent_memory_per_sequence.npz"
    )

    require_file(
        condition.out_dir
        / "recurrent_memory_manifest.json"
    )

    return condition


def submit_lifetime_pair(
    reference_name: str,
    condition_name: str,
    out_dir: Path,
) -> None:
    reference = (
        require_condition_complete(
            reference_name
        )
    )

    condition = (
        require_condition_complete(
            condition_name
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    complete_flag = (
        out_dir
        / "lifetime_binned_excess_error_complete.flag"
    )

    if complete_flag.is_file():
        print(
            "[SKIP] lifetime analysis "
            f"complete: {out_dir}",
            flush=True,
        )

        return

    command = [
        "sbatch",
        "--parsable",
        (
            "--job-name="
            f"rm_life_{condition_name}"
        ),
        str(
            CPU_WORKER
        ),
        "lifetime",
        str(
            reference.out_dir
            / "recurrent_memory_per_sequence.npz"
        ),
        str(
            condition.out_dir
            / "recurrent_memory_per_sequence.npz"
        ),
        str(
            reference.out_dir
            / "recurrent_memory_manifest.json"
        ),
        str(
            condition.out_dir
            / "recurrent_memory_manifest.json"
        ),
        reference_name,
        condition_name,
        str(
            out_dir
        ),
    ]

    job_id = run_checked(
        command
    )

    print(
        "[SUBMITTED] lifetime "
        f"{condition_name} vs "
        f"{reference_name}: "
        f"{job_id}",
        flush=True,
    )


def stage_lifetime() -> None:
    require_file(
        SMOKE_FLAG
    )

    submit_lifetime_pair(
        "P2E_identity",
        "P2E_det_B4",
        LIFETIME_P2E_DIR,
    )

    submit_lifetime_pair(
        "P2F_identity",
        "P2F_det_B4",
        LIFETIME_P2F_DIR,
    )


def stage_aggregate() -> None:
    require_file(
        SMOKE_FLAG
    )

    missing = []

    for condition in (
        conditions()
    ):
        if not (
            condition.complete_flag.is_file()
        ):
            missing.append(
                str(
                    condition.complete_flag
                )
            )

    for flag in (
        LIFETIME_P2E_DIR
        / "lifetime_binned_excess_error_complete.flag",
        LIFETIME_P2F_DIR
        / "lifetime_binned_excess_error_complete.flag",
    ):
        if not flag.is_file():
            missing.append(
                str(
                    flag
                )
            )

    if missing:
        raise RuntimeError(
            "Aggregation prerequisites "
            "are incomplete:\n  "
            + "\n  ".join(
                missing
            )
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
            "[SKIP] aggregation "
            "already complete: "
            f"{STUDY_OUT_DIR}",
            flush=True,
        )

        return

    command = [
        "sbatch",
        "--parsable",
        "--job-name=rm_aggregate",
        str(
            CPU_WORKER
        ),
        "aggregate",
        str(
            MEMOQ_RUN_DIR
        ),
        str(
            VANILLA4_RUN_DIR
        ),
        str(
            VANILLA8_RUN_DIR
        ),
        str(
            LIFETIME_P2E_DIR
        ),
        str(
            LIFETIME_P2F_DIR
        ),
        str(
            SMOKE_DIR
        ),
        str(
            STUDY_OUT_DIR
        ),
    ]

    job_id = run_checked(
        command
    )

    print(
        "[SUBMITTED] aggregation: "
        f"{job_id}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    static_checks()

    if (
        args.stage
        == "smoke"
    ):
        stage_smoke()
        return

    if (
        args.stage
        == "smoke-check"
    ):
        stage_smoke_check()
        return

    if (
        args.stage
        == "conditions"
    ):
        stage_conditions()
        return

    if (
        args.stage
        == "lifetime"
    ):
        stage_lifetime()
        return

    if (
        args.stage
        == "aggregate"
    ):
        stage_aggregate()
        return

    raise RuntimeError(
        f"Unhandled stage: "
        f"{args.stage}"
    )


if __name__ == "__main__":
    main()