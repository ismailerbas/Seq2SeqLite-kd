#!/usr/bin/env python3
"""
eval/validate_recurrent_memory_smoke.py

Cross-check the new recurrent-memory unroll against the repository's already
validated native writeback analysis on the native B4 checkpoint.

No paper numbers are hard-coded. The reference is the existing
VANILLA4S4_native/writeback_summary.json produced by eval/analyze_writeback.py
and authorized by its native_fidelity.json. The new V4_det_B4 condition must
match that validated reference within a strict absolute tolerance before the
full recurrent-memory matrix is allowed to run.
"""

import argparse
import json
import os

from pathlib import Path
from typing import Dict, List


COMPARISON_PATHS = (
    "metrics.mae_seq",
    "metrics.rmse_tau1",
    "metrics.rmse_tau2",
    "metrics.r_tau1",
    "metrics.r_tau2",
    "encoder.p_w0",
    "encoder.p_nwrite0",
    "encoder.mean_nwrite",
    "decoder.p_w0",
    "decoder.p_nwrite0",
    "decoder.mean_nwrite",
    "handoff.mean_abs_handoff_quantization_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate V4_det_B4 against "
            "the established native B4 analysis."
        )
    )

    parser.add_argument(
        "--reference-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--new-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--absolute-tolerance",
        default=1e-6,
        type=float,
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON does not exist: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


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

    os.replace(
        tmp,
        path,
    )


def nested_get(
    payload: Dict,
    path: str,
):
    value = payload

    for key in path.split(
        "."
    ):
        if (
            not isinstance(
                value,
                dict,
            )
            or key
            not in value
        ):
            raise RuntimeError(
                f"Missing required path "
                f"{path}"
            )

        value = value[
            key
        ]

    return value


def first_realization(
    payload: Dict,
) -> Dict:
    realizations = payload.get(
        "realizations"
    )

    if (
        not isinstance(
            realizations,
            list,
        )
        or len(
            realizations
        )
        != 1
    ):
        raise RuntimeError(
            "Smoke validation requires "
            "exactly one deterministic realization"
        )

    return realizations[
        0
    ]


def main() -> None:
    args = parse_args()

    if (
        args.absolute_tolerance
        <= 0.0
    ):
        raise ValueError(
            "--absolute-tolerance must be > 0"
        )

    reference_dir = Path(
        args.reference_dir
    ).resolve()

    new_dir = Path(
        args.new_dir
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    if not reference_dir.is_dir():
        raise FileNotFoundError(
            "Reference directory "
            "does not exist: "
            f"{reference_dir}"
        )

    if not new_dir.is_dir():
        raise FileNotFoundError(
            "New-condition directory "
            "does not exist: "
            f"{new_dir}"
        )

    reference_summary = load_json(
        reference_dir
        / "writeback_summary.json"
    )

    reference_manifest = load_json(
        reference_dir
        / "writeback_manifest.json"
    )

    reference_fidelity = load_json(
        reference_dir
        / "native_fidelity.json"
    )

    new_summary = load_json(
        new_dir
        / "recurrent_memory_summary.json"
    )

    new_manifest = load_json(
        new_dir
        / "recurrent_memory_manifest.json"
    )

    if (
        reference_fidelity.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Reference native_fidelity.json "
            "does not report passed=true"
        )

    if (
        new_summary.get(
            "condition_name"
        )
        != "V4_det_B4"
    ):
        raise RuntimeError(
            "Smoke validation must "
            "run on V4_det_B4"
        )

    if (
        new_summary.get(
            "method"
        )
        != "deterministic"
    ):
        raise RuntimeError(
            "Smoke condition is not deterministic"
        )

    if (
        int(
            new_summary[
                "operator"
            ][
                "state_bits"
            ]
        )
        != 4
    ):
        raise RuntimeError(
            "Smoke condition is not B4"
        )

    reference_checkpoint = (
        reference_manifest.get(
            "checkpoint_sha256"
        )
    )

    new_checkpoint = (
        new_manifest.get(
            "checkpoint_sha256"
        )
    )

    if (
        reference_checkpoint
        != new_checkpoint
    ):
        raise RuntimeError(
            "Smoke reference and new "
            "condition use different checkpoints"
        )

    reference_analysis_sha = (
        reference_manifest.get(
            "analysis_script_sha256"
        )
    )

    new_base_analysis_sha = (
        new_manifest.get(
            "base_analysis_script_sha256"
        )
    )

    if (
        reference_analysis_sha
        != new_base_analysis_sha
    ):
        raise RuntimeError(
            "New condition was not built "
            "against the same analyze_writeback.py "
            "source as the validated reference"
        )

    adapter = new_summary.get(
        "adapter_equivalence"
    )

    if (
        not isinstance(
            adapter,
            dict,
        )
        or adapter.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "New recurrent-memory adapter "
            "equivalence did not pass"
        )

    native_equivalence = (
        new_summary.get(
            "native_tensor_equivalence"
        )
    )

    if (
        not isinstance(
            native_equivalence,
            dict,
        )
        or native_equivalence.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "New V4_det_B4 native tensor "
            "equivalence did not pass"
        )

    reference_realization = (
        first_realization(
            reference_summary
        )
    )

    new_realization = (
        first_realization(
            new_summary
        )
    )

    checks: List[
        Dict
    ] = []

    for path in (
        COMPARISON_PATHS
    ):
        reference_value = float(
            nested_get(
                reference_realization,
                path,
            )
        )

        new_value = float(
            nested_get(
                new_realization,
                path,
            )
        )

        difference = abs(
            new_value
            - reference_value
        )

        passed = (
            difference
            <= args.absolute_tolerance
        )

        checks.append(
            {
                "path": (
                    path
                ),
                "reference": (
                    reference_value
                ),
                "new": (
                    new_value
                ),
                "absolute_difference": (
                    difference
                ),
                "absolute_tolerance": (
                    args.absolute_tolerance
                ),
                "passed": (
                    passed
                ),
            }
        )

    failed = [
        row
        for row
        in checks
        if not row[
            "passed"
        ]
    ]

    if failed:
        raise RuntimeError(
            "Recurrent-memory smoke "
            "validation failed: "
            + "; ".join(
                (
                    f"{row['path']} "
                    "diff="
                    f"{row['absolute_difference']:.9g}"
                )
                for row
                in failed
            )
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "passed": True,
        "reference_dir": str(
            reference_dir
        ),
        "new_dir": str(
            new_dir
        ),
        "checkpoint_sha256": (
            reference_checkpoint
        ),
        "base_analysis_script_sha256": (
            reference_analysis_sha
        ),
        "absolute_tolerance": (
            args.absolute_tolerance
        ),
        "checks": (
            checks
        ),
    }

    atomic_write_json(
        out_dir
        / "recurrent_memory_smoke_validation.json",
        payload,
    )

    with (
        out_dir
        / "recurrent_memory_smoke_validation_complete.flag"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "passed\n"
        )

    print(
        "[DONE] recurrent-memory "
        "smoke validation passed",
        flush=True,
    )


if __name__ == "__main__":
    main()