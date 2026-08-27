#!/usr/bin/env python3
"""
eval/build_v8_allocation_results.py

Fail-closed extension of the recurrent-memory study for the independently
trained B8 checkpoint.

The analysis completes an equal-stored-bits ladder under one frozen B8-trained
checkpoint:

    6 stored bits: B6 state vs B4 + R2 vs B4 + SCW K2
    7 stored bits: B7 state vs B4 + R3 vs B4 + SCW K3
    8 stored bits: B8 state vs B4 + R4 vs B4 + SCW K4

SCW is evaluated at theta=0 and theta=Delta/8. All comparisons use the exact
same checkpoint and test partition. Paired sequence-level bootstrap confidence
intervals are calculated from saved per-sequence predictions using the existing
validated recurrent_memory_stats.py implementation.

No performance value is hard-coded in this file.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from build_recurrent_memory_results import (
    checkpoint_sha,
    load_condition,
    paired_equal_storage_row,
    performance_row,
    require_same_checkpoint,
    require_same_test_partition,
    sha256_file,
    write_csv,
)


ALLOCATION_CONDITIONS = (
    "V8_det_B6",
    "V8_det_B7",
    "V8_det_B8",
    "V8_residual_R2_B4",
    "V8_residual_R3_B4",
    "V8_residual_R4_B4",
    "V8_scw_K2_TH0_B4",
    "V8_scw_K2_TH1_8_B4",
    "V8_scw_K3_TH0_B4",
    "V8_scw_K3_TH1_8_B4",
    "V8_scw_K4_TH0_B4",
    "V8_scw_K4_TH1_8_B4",
)

CONTEXT_CONDITIONS = (
    "V8_forced_B4",
    "V8_ef_B4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the B8-trained equal-storage "
            "recurrent-memory allocation study."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--v8-run-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--paired-bootstrap-reps",
        default=2000,
        type=int,
    )

    parser.add_argument(
        "--paired-bootstrap-seed",
        default=20000,
        type=int,
    )

    parser.add_argument(
        "--paired-bootstrap-batch-reps",
        default=8,
        type=int,
    )

    args = parser.parse_args()

    if (
        args.paired_bootstrap_reps
        <= 0
    ):
        raise ValueError(
            "--paired-bootstrap-reps must be > 0"
        )

    if (
        args.paired_bootstrap_batch_reps
        <= 0
    ):
        raise ValueError(
            "--paired-bootstrap-batch-reps must be > 0"
        )

    return args


def atomic_write_json(
    path: Path,
    payload: Dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

        handle.write(
            "\n"
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        tmp,
        path,
    )


def build_equal_storage_rows(
    records: Dict[
        str,
        Dict,
    ],
    performance_by_name: Dict[
        str,
        Dict,
    ],
    reps: int,
    seed: int,
    batch_reps: int,
) -> List[
    Dict
]:
    rows: List[
        Dict
    ] = []

    cache: Dict = {}

    row_seed = int(
        seed
    )

    for total_bits in (
        6,
        7,
        8,
    ):
        extra_bits = (
            total_bits
            - 4
        )

        state_name = (
            f"V8_det_B{total_bits}"
        )

        alternatives = (
            (
                "quantized_residual",
                (
                    f"V8_residual_"
                    f"R{extra_bits}_B4"
                ),
            ),
            (
                "scw_theta0",
                (
                    f"V8_scw_"
                    f"K{extra_bits}_"
                    f"TH0_B4"
                ),
            ),
            (
                "scw_theta_delta_over_8",
                (
                    f"V8_scw_"
                    f"K{extra_bits}_"
                    f"TH1_8_B4"
                ),
            ),
        )

        for (
            comparison_family,
            alternative_name,
        ) in alternatives:
            row = (
                paired_equal_storage_row(
                    total_bits=total_bits,
                    comparison_family=(
                        comparison_family
                    ),
                    state_record=(
                        records[
                            state_name
                        ]
                    ),
                    alternative_record=(
                        records[
                            alternative_name
                        ]
                    ),
                    state_summary=(
                        performance_by_name[
                            state_name
                        ]
                    ),
                    alternative_summary=(
                        performance_by_name[
                            alternative_name
                        ]
                    ),
                    cache=cache,
                    reps=reps,
                    seed=row_seed,
                    batch_reps=(
                        batch_reps
                    ),
                )
            )

            row[
                "checkpoint_group"
            ] = "Native B8"

            rows.append(
                row
            )

            row_seed += 1000

    return rows


def main() -> None:
    args = parse_args()

    v8_run_dir = Path(
        args.v8_run_dir
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    root = (
        v8_run_dir
        / "recurrent_memory_analysis"
    )

    if not v8_run_dir.is_dir():
        raise FileNotFoundError(
            f"V8 run directory does "
            f"not exist: {v8_run_dir}"
        )

    if not root.is_dir():
        raise FileNotFoundError(
            f"V8 recurrent-memory "
            f"directory does not exist: "
            f"{root}"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: Dict[
        str,
        Dict,
    ] = {}

    for name in (
        ALLOCATION_CONDITIONS
        + CONTEXT_CONDITIONS
    ):
        records[
            name
        ] = load_condition(
            root,
            name,
        )

    allocation_records = [
        records[
            name
        ]
        for name
        in ALLOCATION_CONDITIONS
    ]

    checkpoint_hash = (
        require_same_checkpoint(
            "B8-trained allocation grid",
            allocation_records,
        )
    )

    test_idx_hash = (
        require_same_test_partition(
            allocation_records
        )
    )

    for name in (
        CONTEXT_CONDITIONS
    ):
        if (
            checkpoint_sha(
                records[
                    name
                ]
            )
            != checkpoint_hash
        ):
            raise RuntimeError(
                f"Context condition "
                f"{name} does not use "
                f"the allocation-grid "
                f"checkpoint"
            )

    all_records = [
        records[
            name
        ]
        for name in (
            ALLOCATION_CONDITIONS
            + CONTEXT_CONDITIONS
        )
    ]

    require_same_test_partition(
        all_records
    )

    performance_rows = [
        performance_row(
            records[
                name
            ]
        )
        for name in (
            ALLOCATION_CONDITIONS
            + CONTEXT_CONDITIONS
        )
    ]

    performance_by_name = {
        row[
            "condition"
        ]: row
        for row in (
            performance_rows
        )
    }

    equal_storage_rows = (
        build_equal_storage_rows(
            records=records,
            performance_by_name=(
                performance_by_name
            ),
            reps=(
                args.paired_bootstrap_reps
            ),
            seed=(
                args.paired_bootstrap_seed
            ),
            batch_reps=(
                args.paired_bootstrap_batch_reps
            ),
        )
    )

    method_sweep_path = (
        out_dir
        / "v8_allocation_method_sweep.csv"
    )

    equal_storage_path = (
        out_dir
        / "v8_equal_storage_comparison.csv"
    )

    write_csv(
        method_sweep_path,
        performance_rows,
    )

    write_csv(
        equal_storage_path,
        equal_storage_rows,
    )

    input_files = {}

    for name in (
        ALLOCATION_CONDITIONS
        + CONTEXT_CONDITIONS
    ):
        record = (
            records[
                name
            ]
        )

        for key in (
            "summary_path",
            "manifest_path",
            "prediction_path",
        ):
            path = Path(
                record[
                    key
                ]
            ).resolve()

            input_files[
                str(
                    path
                )
            ] = sha256_file(
                path
            )

    study = {
        "passed": (
            True
        ),
        "analysis": (
            "B8-trained equal-storage "
            "recurrent-memory allocation"
        ),
        "checkpoint_sha256": (
            checkpoint_hash
        ),
        "test_idx_sha256": (
            test_idx_hash
        ),
        "allocation_conditions": (
            list(
                ALLOCATION_CONDITIONS
            )
        ),
        "context_conditions": (
            list(
                CONTEXT_CONDITIONS
            )
        ),
        "paired_bootstrap_reps": (
            int(
                args.paired_bootstrap_reps
            )
        ),
        "paired_bootstrap_seed": (
            int(
                args.paired_bootstrap_seed
            )
        ),
        "paired_bootstrap_batch_reps": (
            int(
                args.paired_bootstrap_batch_reps
            )
        ),
        "comparison_definition": {
            "6_bits": {
                "state_only": (
                    "V8_det_B6"
                ),
                "alternatives": [
                    "V8_residual_R2_B4",
                    "V8_scw_K2_TH0_B4",
                    "V8_scw_K2_TH1_8_B4",
                ],
            },
            "7_bits": {
                "state_only": (
                    "V8_det_B7"
                ),
                "alternatives": [
                    "V8_residual_R3_B4",
                    "V8_scw_K3_TH0_B4",
                    "V8_scw_K3_TH1_8_B4",
                ],
            },
            "8_bits": {
                "state_only": (
                    "V8_det_B8"
                ),
                "alternatives": [
                    "V8_residual_R4_B4",
                    "V8_scw_K4_TH0_B4",
                    "V8_scw_K4_TH1_8_B4",
                ],
            },
        },
        "outputs": {
            "method_sweep_csv": (
                str(
                    method_sweep_path
                )
            ),
            "equal_storage_csv": (
                str(
                    equal_storage_path
                )
            ),
        },
        "input_files": (
            input_files
        ),
        "analysis_script": (
            str(
                Path(
                    __file__
                ).resolve()
            )
        ),
        "analysis_script_sha256": (
            sha256_file(
                Path(
                    __file__
                ).resolve()
            )
        ),
        "base_aggregator_script": (
            str(
                (
                    Path(
                        __file__
                    ).resolve().parent
                    / "build_recurrent_memory_results.py"
                ).resolve()
            )
        ),
        "base_aggregator_script_sha256": (
            sha256_file(
                (
                    Path(
                        __file__
                    ).resolve().parent
                    / "build_recurrent_memory_results.py"
                ).resolve()
            )
        ),
    }

    study_path = (
        out_dir
        / "v8_allocation_study.json"
    )

    atomic_write_json(
        study_path,
        study,
    )

    (
        out_dir
        / "v8_allocation_study_complete.flag"
    ).write_text(
        "passed\n",
        encoding="utf-8",
    )

    print(
        "[V8 ALLOCATION] passed",
        flush=True,
    )

    print(
        f"[V8 ALLOCATION] "
        f"checkpoint_sha256="
        f"{checkpoint_hash}",
        flush=True,
    )

    print(
        f"[V8 ALLOCATION] "
        f"test_idx_sha256="
        f"{test_idx_hash}",
        flush=True,
    )

    print(
        f"[V8 ALLOCATION] "
        f"{method_sweep_path}",
        flush=True,
    )

    print(
        f"[V8 ALLOCATION] "
        f"{equal_storage_path}",
        flush=True,
    )

    print(
        f"[V8 ALLOCATION] "
        f"{study_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()