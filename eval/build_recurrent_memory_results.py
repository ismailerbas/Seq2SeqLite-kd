#!/usr/bin/env python3
"""
eval/build_recurrent_memory_results.py

Fail-closed aggregation for the recurrent-margin, residual-memory, and SCW
study.

Every expected frozen-checkpoint condition must be complete. Within-checkpoint
comparison families are checked by checkpoint SHA-256, all conditions are
checked against the same held-out test-index hash, and equal-storage claims use
paired bootstrap confidence intervals computed from per-sequence predictions.

No performance value is hard-coded in this file.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time

from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

THIS_FILE = Path(
    __file__
).resolve()

EVAL_DIR = (
    THIS_FILE.parent
)

if str(
    EVAL_DIR
) not in sys.path:
    sys.path.insert(
        0,
        str(
            EVAL_DIR
        ),
    )

from recurrent_memory_stats import (
    load_prediction_npz,
    paired_bootstrap_mean_difference,
    paired_bootstrap_prediction_difference,
    validate_prediction_pair,
)


MEMOQ_CONDITIONS = (
    "P2E_identity",
    "P2E_det_B4",

    "P2F_identity",
    "P2F_det_B4",
    "P2F_ef_B4",
    "P2F_sr_B4",
    "P2F_residual_R2_B4",
    "P2F_residual_R3_B4",
    "P2F_residual_R4_B4",
    "P2F_residual_FULL_HALFSTEP_B4",
    "P2F_scw_K2_TH0_B4",
    "P2F_scw_K2_TH1_8_B4",
    "P2F_scw_K3_TH0_B4",
    "P2F_scw_K3_TH1_8_B4",
    "P2F_scw_K4_TH0_B4",
    "P2F_scw_K4_TH1_8_B4",

    "P3_identity",
    "P3_det_B4",
    "P3_ef_B4",
    "P3_sr_B4",
    "P3_residual_R2_B4",
    "P3_residual_R3_B4",
    "P3_residual_R4_B4",
    "P3_residual_FULL_HALFSTEP_B4",
    "P3_scw_K2_TH0_B4",
    "P3_scw_K2_TH1_8_B4",
    "P3_scw_K3_TH0_B4",
    "P3_scw_K3_TH1_8_B4",
    "P3_scw_K4_TH0_B4",
    "P3_scw_K4_TH1_8_B4",
)

V4_CONDITIONS = (
    "V4_det_B4",
    "V4_det_B6",
    "V4_det_B7",
    "V4_det_B8",
    "V4_ef_B4",
    "V4_sr_B4",
    "V4_residual_R2_B4",
    "V4_residual_R3_B4",
    "V4_residual_R4_B4",
    "V4_residual_FULL_HALFSTEP_B4",
    "V4_scw_K2_TH0_B4",
    "V4_scw_K2_TH1_8_B4",
    "V4_scw_K3_TH0_B4",
    "V4_scw_K3_TH1_8_B4",
    "V4_scw_K4_TH0_B4",
    "V4_scw_K4_TH1_8_B4",
)

V8_CONDITIONS = (
    "V8_det_B8",
    "V8_forced_B4",
    "V8_ef_B4",
    "V8_residual_R2_B4",
    "V8_residual_R4_B4",
    "V8_scw_K4_TH1_8_B4",
)

PRIMARY_MECHANISM_CONDITIONS = (
    "P2E_identity",
    "P2E_det_B4",
    "P2F_identity",
    "P2F_det_B4",
    "P3_identity",
    "P3_det_B4",
    "V4_det_B4",
    "V8_det_B8",
    "V8_forced_B4",
)

METHOD_LABELS = {
    "identity": (
        "Identity"
    ),
    "deterministic": (
        "Deterministic"
    ),
    "stochastic": (
        "Stochastic rounding"
    ),
    "error_feedback": (
        "Error feedback"
    ),
    "quantized_residual": (
        "Quantized residual"
    ),
    "full_halfstep_residual": (
        "Full half-step residual"
    ),
    "scw": (
        "SCW"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the complete "
            "recurrent-memory study."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--memoq-run-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--vanilla4-run-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--vanilla8-run-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--lifetime-p2e-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--lifetime-p2f-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--smoke-validation-dir",
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
        default=7000,
        type=int,
    )

    parser.add_argument(
        "--paired-bootstrap-batch-reps",
        default=8,
        type=int,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
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


def sha256_file(
    path: Path,
    chunk_size: int = (
        1024
        * 1024
    ),
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


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


def require_dir(
    path: Path,
) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: "
            f"{path}"
        )

    return path


def require_file(
    path: Path,
) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: "
            f"{path}"
        )

    return path


def load_condition(
    root: Path,
    name: str,
) -> Dict:
    condition_dir = require_dir(
        root
        / name
    )

    require_file(
        condition_dir
        / "recurrent_memory_complete.flag"
    )

    summary_path = require_file(
        condition_dir
        / "recurrent_memory_summary.json"
    )

    manifest_path = require_file(
        condition_dir
        / "recurrent_memory_manifest.json"
    )

    prediction_path = require_file(
        condition_dir
        / "recurrent_memory_per_sequence.npz"
    )

    summary = load_json(
        summary_path
    )

    manifest = load_json(
        manifest_path
    )

    if (
        summary.get(
            "condition_name"
        )
        != name
    ):
        raise RuntimeError(
            "Condition-name mismatch in "
            f"{summary_path}: "
            f"{summary.get('condition_name')!r} "
            f"!= {name!r}"
        )

    if (
        manifest.get(
            "condition_name"
        )
        != name
    ):
        raise RuntimeError(
            "Condition-name mismatch in "
            f"{manifest_path}: "
            f"{manifest.get('condition_name')!r} "
            f"!= {name!r}"
        )

    adapter = summary.get(
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
            "Adapter equivalence did "
            f"not pass for {name}"
        )

    return {
        "name": (
            name
        ),
        "dir": (
            condition_dir
        ),
        "summary_path": (
            summary_path
        ),
        "manifest_path": (
            manifest_path
        ),
        "prediction_path": (
            prediction_path
        ),
        "summary": (
            summary
        ),
        "manifest": (
            manifest
        ),
    }


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
            or key not in value
        ):
            return None

        value = value[
            key
        ]

    return value


def optional_realization_mean_std(
    record: Dict,
    path: str,
) -> Tuple[
    Optional[
        float
    ],
    Optional[
        float
    ],
]:
    realizations = record[
        "summary"
    ].get(
        "realizations"
    )

    if (
        not isinstance(
            realizations,
            list,
        )
        or not realizations
    ):
        raise RuntimeError(
            f"No realizations in "
            f"{record['summary_path']}"
        )

    values = [
        nested_get(
            realization,
            path,
        )
        for realization
        in realizations
    ]

    if all(
        value is None
        for value
        in values
    ):
        return (
            None,
            None,
        )

    if any(
        value is None
        for value
        in values
    ):
        raise RuntimeError(
            "Mixed null/non-null "
            f"values for {path} in "
            f"{record['summary_path']}"
        )

    array = np.asarray(
        [
            float(
                value
            )
            for value
            in values
        ],
        dtype=np.float64,
    )

    return (
        float(
            np.mean(
                array
            )
        ),
        (
            float(
                np.std(
                    array,
                    ddof=1,
                )
            )
            if len(
                array
            )
            > 1
            else 0.0
        ),
    )


def required_realization_mean_std(
    record: Dict,
    path: str,
) -> Tuple[
    float,
    float,
]:
    (
        mean,
        std,
    ) = optional_realization_mean_std(
        record,
        path,
    )

    if mean is None:
        raise RuntimeError(
            "Required realization metric "
            f"{path} is null in "
            f"{record['summary_path']}"
        )

    return (
        mean,
        float(
            std
        ),
    )


def optional_mechanism_mean(
    record: Dict,
    region: str,
    key: str,
) -> Optional[
    float
]:
    mechanisms = record[
        "summary"
    ].get(
        "mechanism_realizations"
    )

    if (
        not isinstance(
            mechanisms,
            list,
        )
        or not mechanisms
    ):
        raise RuntimeError(
            f"No mechanism realizations in "
            f"{record['summary_path']}"
        )

    values = []

    for realization in mechanisms:
        region_payload = realization.get(
            region
        )

        if not isinstance(
            region_payload,
            dict,
        ):
            raise RuntimeError(
                f"Missing mechanism region "
                f"{region} in "
                f"{record['summary_path']}"
            )

        if key not in region_payload:
            raise RuntimeError(
                f"Missing mechanism "
                f"{region}.{key} in "
                f"{record['summary_path']}"
            )

        values.append(
            region_payload[
                key
            ]
        )

    if all(
        value is None
        for value
        in values
    ):
        return None

    if any(
        value is None
        for value
        in values
    ):
        raise RuntimeError(
            "Mixed null/non-null "
            "mechanism values for "
            f"{region}.{key}"
        )

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def checkpoint_sha(
    record: Dict,
) -> str:
    value = record[
        "manifest"
    ].get(
        "checkpoint_sha256"
    )

    if (
        not isinstance(
            value,
            str,
        )
        or len(
            value
        )
        != 64
    ):
        raise RuntimeError(
            "Invalid checkpoint_sha256 in "
            f"{record['manifest_path']}"
        )

    return value


def test_idx_sha(
    record: Dict,
) -> str:
    value = record[
        "manifest"
    ].get(
        "test_idx_sha256"
    )

    if (
        not isinstance(
            value,
            str,
        )
        or len(
            value
        )
        != 64
    ):
        raise RuntimeError(
            "Invalid test_idx_sha256 in "
            f"{record['manifest_path']}"
        )

    return value


def require_same_checkpoint(
    name: str,
    records: Sequence[
        Dict
    ],
) -> str:
    hashes = sorted(
        set(
            checkpoint_sha(
                record
            )
            for record
            in records
        )
    )

    if len(
        hashes
    ) != 1:
        raise RuntimeError(
            "Checkpoint identity failure "
            f"for {name}: {hashes}"
        )

    return hashes[
        0
    ]


def require_same_test_partition(
    records: Sequence[
        Dict
    ],
) -> str:
    hashes = sorted(
        set(
            test_idx_sha(
                record
            )
            for record
            in records
        )
    )

    if len(
        hashes
    ) != 1:
        raise RuntimeError(
            "Held-out test partition "
            f"mismatch: {hashes}"
        )

    return hashes[
        0
    ]


def condition_group(
    name: str,
) -> str:
    if name.startswith(
        "P2E_"
    ):
        return "P2E"

    if name.startswith(
        "P2F_"
    ):
        return "P2F"

    if name.startswith(
        "P3_"
    ):
        return "P3"

    if name.startswith(
        "V4_"
    ):
        return "Native B4"

    if name.startswith(
        "V8_"
    ):
        return "Native B8"

    raise RuntimeError(
        "Cannot infer group from "
        f"condition name {name}"
    )


def performance_row(
    record: Dict,
) -> Dict:
    operator = record[
        "summary"
    ].get(
        "operator"
    )

    if not isinstance(
        operator,
        dict,
    ):
        raise RuntimeError(
            "Missing operator metadata in "
            f"{record['summary_path']}"
        )

    method = record[
        "summary"
    ].get(
        "method"
    )

    if method not in METHOD_LABELS:
        raise RuntimeError(
            f"Unsupported method "
            f"{method!r} in "
            f"{record['summary_path']}"
        )

    (
        mae_seq,
        mae_seq_std,
    ) = required_realization_mean_std(
        record,
        "metrics.mae_seq",
    )

    (
        tau1,
        tau1_std,
    ) = required_realization_mean_std(
        record,
        "metrics.rmse_tau1",
    )

    (
        tau2,
        tau2_std,
    ) = required_realization_mean_std(
        record,
        "metrics.rmse_tau2",
    )

    (
        r_tau1,
        r_tau1_std,
    ) = required_realization_mean_std(
        record,
        "metrics.r_tau1",
    )

    (
        r_tau2,
        r_tau2_std,
    ) = required_realization_mean_std(
        record,
        "metrics.r_tau2",
    )

    (
        encoder_p_w0,
        encoder_p_w0_std,
    ) = optional_realization_mean_std(
        record,
        "encoder.p_w0",
    )

    (
        encoder_p_nwrite0,
        encoder_p_nwrite0_std,
    ) = optional_realization_mean_std(
        record,
        "encoder.p_nwrite0",
    )

    (
        encoder_mean_nwrite,
        encoder_mean_nwrite_std,
    ) = optional_realization_mean_std(
        record,
        "encoder.mean_nwrite",
    )

    (
        decoder_p_w0,
        decoder_p_w0_std,
    ) = optional_realization_mean_std(
        record,
        "decoder.p_w0",
    )

    (
        decoder_p_nwrite0,
        decoder_p_nwrite0_std,
    ) = optional_realization_mean_std(
        record,
        "decoder.p_nwrite0",
    )

    (
        decoder_mean_nwrite,
        decoder_mean_nwrite_std,
    ) = optional_realization_mean_std(
        record,
        "decoder.mean_nwrite",
    )

    (
        handoff_mae,
        handoff_mae_std,
    ) = required_realization_mean_std(
        record,
        "handoff.mean_abs_handoff_quantization_error",
    )

    (
        n_test,
        _,
    ) = required_realization_mean_std(
        record,
        "metrics.n_test",
    )

    return {
        "condition": (
            record[
                "name"
            ]
        ),
        "group": (
            condition_group(
                record[
                    "name"
                ]
            )
        ),
        "method": (
            method
        ),
        "method_label": (
            METHOD_LABELS[
                method
            ]
        ),
        "state_bits": int(
            operator[
                "state_bits"
            ]
        ),
        "residual_bits": (
            operator.get(
                "residual_bits"
            )
        ),
        "counter_bits": (
            operator.get(
                "counter_bits"
            )
        ),
        "total_stored_bits_per_unit": (
            operator.get(
                "total_stored_bits_per_unit"
            )
        ),
        "deadzone_fraction_of_delta": (
            operator.get(
                "deadzone_fraction_of_delta"
            )
        ),
        "trigger_votes": (
            operator.get(
                "trigger_votes"
            )
        ),
        "emitted_velocity_per_consistent_vote_step": (
            operator.get(
                "emitted_velocity_per_consistent_vote_step"
            )
        ),
        "n_realizations": int(
            record[
                "summary"
            ][
                "n_realizations"
            ]
        ),
        "n_test": int(
            round(
                n_test
            )
        ),
        "mae_seq": (
            mae_seq
        ),
        "mae_seq_realization_std": (
            mae_seq_std
        ),
        "rmse_tau1": (
            tau1
        ),
        "rmse_tau1_realization_std": (
            tau1_std
        ),
        "rmse_tau2": (
            tau2
        ),
        "rmse_tau2_realization_std": (
            tau2_std
        ),
        "r_tau1": (
            r_tau1
        ),
        "r_tau1_realization_std": (
            r_tau1_std
        ),
        "r_tau2": (
            r_tau2
        ),
        "r_tau2_realization_std": (
            r_tau2_std
        ),
        "encoder_p_w0": (
            encoder_p_w0
        ),
        "encoder_p_w0_realization_std": (
            encoder_p_w0_std
        ),
        "encoder_p_nwrite0": (
            encoder_p_nwrite0
        ),
        "encoder_p_nwrite0_realization_std": (
            encoder_p_nwrite0_std
        ),
        "encoder_mean_nwrite": (
            encoder_mean_nwrite
        ),
        "encoder_mean_nwrite_realization_std": (
            encoder_mean_nwrite_std
        ),
        "decoder_p_w0": (
            decoder_p_w0
        ),
        "decoder_p_w0_realization_std": (
            decoder_p_w0_std
        ),
        "decoder_p_nwrite0": (
            decoder_p_nwrite0
        ),
        "decoder_p_nwrite0_realization_std": (
            decoder_p_nwrite0_std
        ),
        "decoder_mean_nwrite": (
            decoder_mean_nwrite
        ),
        "decoder_mean_nwrite_realization_std": (
            decoder_mean_nwrite_std
        ),
        "handoff_mean_abs_quantization_error": (
            handoff_mae
        ),
        "handoff_mean_abs_quantization_error_realization_std": (
            handoff_mae_std
        ),
        "encoder_deadband_fraction": (
            optional_mechanism_mean(
                record,
                "encoder",
                "deadband_fraction",
            )
        ),
        "decoder_deadband_fraction": (
            optional_mechanism_mean(
                record,
                "decoder",
                "deadband_fraction",
            )
        ),
        "encoder_deadband_is_counterfactual": (
            nested_get(
                record[
                    "summary"
                ][
                    "mechanism_realizations"
                ][
                    0
                ],
                "encoder.deadband_is_counterfactual",
            )
        ),
        "decoder_deadband_is_counterfactual": (
            nested_get(
                record[
                    "summary"
                ][
                    "mechanism_realizations"
                ][
                    0
                ],
                "decoder.deadband_is_counterfactual",
            )
        ),
        "encoder_subthreshold_write_fraction_given_deadband": (
            optional_mechanism_mean(
                record,
                "encoder",
                "subthreshold_write_fraction_given_deadband",
            )
        ),
        "decoder_subthreshold_write_fraction_given_deadband": (
            optional_mechanism_mean(
                record,
                "decoder",
                "subthreshold_write_fraction_given_deadband",
            )
        ),
        "encoder_suprathreshold_nowrite_fraction_given_suprathreshold": (
            optional_mechanism_mean(
                record,
                "encoder",
                "suprathreshold_nowrite_fraction_given_suprathreshold",
            )
        ),
        "decoder_suprathreshold_nowrite_fraction_given_suprathreshold": (
            optional_mechanism_mean(
                record,
                "decoder",
                "suprathreshold_nowrite_fraction_given_suprathreshold",
            )
        ),
        "encoder_scw_forced_write_fraction": (
            optional_mechanism_mean(
                record,
                "encoder",
                "scw_forced_write_fraction",
            )
        ),
        "decoder_scw_forced_write_fraction": (
            optional_mechanism_mean(
                record,
                "decoder",
                "scw_forced_write_fraction",
            )
        ),
        "encoder_scw_forced_rail_block_fraction": (
            optional_mechanism_mean(
                record,
                "encoder",
                "scw_forced_rail_block_fraction",
            )
        ),
        "decoder_scw_forced_rail_block_fraction": (
            optional_mechanism_mean(
                record,
                "decoder",
                "scw_forced_rail_block_fraction",
            )
        ),
        "encoder_gate_factor_mean": (
            optional_mechanism_mean(
                record,
                "encoder",
                "gate_factor_mean",
            )
        ),
        "decoder_gate_factor_mean": (
            optional_mechanism_mean(
                record,
                "decoder",
                "gate_factor_mean",
            )
        ),
        "encoder_candidate_displacement_mean": (
            optional_mechanism_mean(
                record,
                "encoder",
                "candidate_displacement_mean",
            )
        ),
        "decoder_candidate_displacement_mean": (
            optional_mechanism_mean(
                record,
                "decoder",
                "candidate_displacement_mean",
            )
        ),
        "checkpoint_sha256": (
            checkpoint_sha(
                record
            )
        ),
        "test_idx_sha256": (
            test_idx_sha(
                record
            )
        ),
        "source_dir": str(
            record[
                "dir"
            ]
        ),
    }


def write_csv(
    path: Path,
    rows: Sequence[
        Dict
    ],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: "
            f"{path}"
        )

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(
                    key
                )

                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def load_prediction_cached(
    record: Dict,
    cache: Dict[
        str,
        Dict[
            str,
            np.ndarray,
        ],
    ],
) -> Dict[
    str,
    np.ndarray,
]:
    name = record[
        "name"
    ]

    if name not in cache:
        cache[
            name
        ] = load_prediction_npz(
            record[
                "prediction_path"
            ]
        )

    return cache[
        name
    ]


def assert_close(
    label: str,
    measured: float,
    expected: float,
    tolerance: float = 1e-6,
) -> None:
    if (
        abs(
            float(
                measured
            )
            - float(
                expected
            )
        )
        > tolerance
    ):
        raise RuntimeError(
            "Per-sequence recomputation mismatch "
            f"for {label}: "
            f"{measured:.12g} vs "
            f"summary {expected:.12g}"
        )


def paired_equal_storage_row(
    total_bits: int,
    comparison_family: str,
    state_record: Dict,
    alternative_record: Dict,
    state_summary: Dict,
    alternative_summary: Dict,
    cache: Dict[
        str,
        Dict[
            str,
            np.ndarray,
        ],
    ],
    reps: int,
    seed: int,
    batch_reps: int,
) -> Dict:
    require_same_checkpoint(
        (
            "equal-storage "
            f"{comparison_family} "
            f"B{total_bits}"
        ),
        [
            state_record,
            alternative_record,
        ],
    )

    state_prediction = (
        load_prediction_cached(
            state_record,
            cache,
        )
    )

    alternative_prediction = (
        load_prediction_cached(
            alternative_record,
            cache,
        )
    )

    validate_prediction_pair(
        state_prediction,
        alternative_prediction,
    )

    tau1_stats = (
        paired_bootstrap_prediction_difference(
            state_prediction[
                "gt_tau1"
            ],
            state_prediction[
                "tau1_pred"
            ],
            alternative_prediction[
                "tau1_pred"
            ],
            reps,
            seed,
            batch_reps,
        )
    )

    tau2_stats = (
        paired_bootstrap_prediction_difference(
            state_prediction[
                "gt_tau2"
            ],
            state_prediction[
                "tau2_pred"
            ],
            alternative_prediction[
                "tau2_pred"
            ],
            reps,
            (
                seed
                + 100
            ),
            batch_reps,
        )
    )

    seq_stats = (
        paired_bootstrap_mean_difference(
            state_prediction[
                "seq_mae_per_sequence"
            ],
            alternative_prediction[
                "seq_mae_per_sequence"
            ],
            reps,
            (
                seed
                + 200
            ),
            batch_reps,
        )
    )

    assert_close(
        f"{state_record['name']} tau1",
        tau1_stats[
            "reference_rmse"
        ],
        state_summary[
            "rmse_tau1"
        ],
    )

    assert_close(
        f"{alternative_record['name']} tau1",
        tau1_stats[
            "alternative_rmse"
        ],
        alternative_summary[
            "rmse_tau1"
        ],
    )

    assert_close(
        f"{state_record['name']} tau2",
        tau2_stats[
            "reference_rmse"
        ],
        state_summary[
            "rmse_tau2"
        ],
    )

    assert_close(
        f"{alternative_record['name']} tau2",
        tau2_stats[
            "alternative_rmse"
        ],
        alternative_summary[
            "rmse_tau2"
        ],
    )

    assert_close(
        f"{state_record['name']} sequence MAE",
        seq_stats[
            "reference_mean"
        ],
        state_summary[
            "mae_seq"
        ],
    )

    assert_close(
        f"{alternative_record['name']} sequence MAE",
        seq_stats[
            "alternative_mean"
        ],
        alternative_summary[
            "mae_seq"
        ],
    )

    tau1_low = tau1_stats[
        "alternative_minus_reference_rmse_ci95_low"
    ]

    tau1_high = tau1_stats[
        "alternative_minus_reference_rmse_ci95_high"
    ]

    tau2_low = tau2_stats[
        "alternative_minus_reference_rmse_ci95_low"
    ]

    tau2_high = tau2_stats[
        "alternative_minus_reference_rmse_ci95_high"
    ]

    seq_low = seq_stats[
        "alternative_minus_reference_mean_ci95_low"
    ]

    seq_high = seq_stats[
        "alternative_minus_reference_mean_ci95_high"
    ]

    return {
        "total_stored_bits_per_unit": (
            total_bits
        ),
        "comparison_family": (
            comparison_family
        ),
        "state_only_condition": (
            state_record[
                "name"
            ]
        ),
        "alternative_condition": (
            alternative_record[
                "name"
            ]
        ),
        "state_only_rmse_tau1": (
            tau1_stats[
                "reference_rmse"
            ]
        ),
        "alternative_rmse_tau1": (
            tau1_stats[
                "alternative_rmse"
            ]
        ),
        "alternative_minus_state_rmse_tau1": (
            tau1_stats[
                "alternative_minus_reference_rmse"
            ]
        ),
        "alternative_minus_state_rmse_tau1_ci95_low": (
            tau1_low
        ),
        "alternative_minus_state_rmse_tau1_ci95_high": (
            tau1_high
        ),
        "alternative_beats_state_tau1_ci95": bool(
            tau1_high
            < 0.0
        ),
        "state_only_rmse_tau2": (
            tau2_stats[
                "reference_rmse"
            ]
        ),
        "alternative_rmse_tau2": (
            tau2_stats[
                "alternative_rmse"
            ]
        ),
        "alternative_minus_state_rmse_tau2": (
            tau2_stats[
                "alternative_minus_reference_rmse"
            ]
        ),
        "alternative_minus_state_rmse_tau2_ci95_low": (
            tau2_low
        ),
        "alternative_minus_state_rmse_tau2_ci95_high": (
            tau2_high
        ),
        "alternative_beats_state_tau2_ci95": bool(
            tau2_high
            < 0.0
        ),
        "state_only_mae_seq": (
            seq_stats[
                "reference_mean"
            ]
        ),
        "alternative_mae_seq": (
            seq_stats[
                "alternative_mean"
            ]
        ),
        "alternative_minus_state_mae_seq": (
            seq_stats[
                "alternative_minus_reference_mean"
            ]
        ),
        "alternative_minus_state_mae_seq_ci95_low": (
            seq_low
        ),
        "alternative_minus_state_mae_seq_ci95_high": (
            seq_high
        ),
        "alternative_beats_state_mae_seq_ci95": bool(
            seq_high
            < 0.0
        ),
        "bootstrap_reps": (
            reps
        ),
        "bootstrap_seed": (
            seed
        ),
        "bootstrap_batch_reps": (
            batch_reps
        ),
        "checkpoint_sha256": (
            checkpoint_sha(
                state_record
            )
        ),
        "test_idx_sha256": (
            test_idx_sha(
                state_record
            )
        ),
    }


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
    rows = []

    cache: Dict[
        str,
        Dict[
            str,
            np.ndarray,
        ],
    ] = {}

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
            f"V4_det_B{total_bits}"
        )

        alternatives = (
            (
                "quantized_residual",
                (
                    "V4_residual_"
                    f"R{extra_bits}_B4"
                ),
            ),
            (
                "scw_theta0",
                (
                    "V4_scw_"
                    f"K{extra_bits}_"
                    "TH0_B4"
                ),
            ),
            (
                "scw_theta_delta_over_8",
                (
                    "V4_scw_"
                    f"K{extra_bits}_"
                    "TH1_8_B4"
                ),
            ),
        )

        for (
            comparison_family,
            alternative_name,
        ) in alternatives:
            rows.append(
                paired_equal_storage_row(
                    total_bits,
                    comparison_family,
                    records[
                        state_name
                    ],
                    records[
                        alternative_name
                    ],
                    performance_by_name[
                        state_name
                    ],
                    performance_by_name[
                        alternative_name
                    ],
                    cache,
                    reps,
                    row_seed,
                    batch_reps,
                )
            )

            row_seed += 1000

    return rows


def build_margin_rows(
    records: Dict[
        str,
        Dict,
    ],
) -> List[
    Dict
]:
    rows = []

    for name in (
        PRIMARY_MECHANISM_CONDITIONS
    ):
        record = records[
            name
        ]

        mechanisms = record[
            "summary"
        ].get(
            "mechanism_realizations"
        )

        if (
            not isinstance(
                mechanisms,
                list,
            )
            or len(
                mechanisms
            )
            != 1
        ):
            raise RuntimeError(
                "Primary mechanism condition "
                f"{name} must have one realization"
            )

        for region in (
            "encoder",
            "decoder",
        ):
            payload = mechanisms[
                0
            ][
                region
            ]

            checks = payload[
                "internal_consistency_checks"
            ]

            rows.append(
                {
                    "condition": (
                        name
                    ),
                    "group": (
                        condition_group(
                            name
                        )
                    ),
                    "region": (
                        region
                    ),
                    "state_bits": (
                        payload[
                            "state_bits"
                        ]
                    ),
                    "delta": (
                        payload[
                            "delta"
                        ]
                    ),
                    "half_step": (
                        payload[
                            "half_step"
                        ]
                    ),
                    "deadband_fraction": (
                        payload[
                            "deadband_fraction"
                        ]
                    ),
                    "deadband_is_counterfactual": (
                        payload[
                            "deadband_is_counterfactual"
                        ]
                    ),
                    "state_change_fraction": (
                        payload[
                            "state_change_fraction"
                        ]
                    ),
                    "live_write_fraction": (
                        payload[
                            "live_write_fraction"
                        ]
                    ),
                    "subthreshold_write_fraction_given_deadband": (
                        payload[
                            "subthreshold_write_fraction_given_deadband"
                        ]
                    ),
                    "suprathreshold_nowrite_fraction_given_suprathreshold": (
                        payload[
                            "suprathreshold_nowrite_fraction_given_suprathreshold"
                        ]
                    ),
                    "rail_fraction": (
                        payload[
                            "rail_fraction"
                        ]
                    ),
                    "tie_fraction": (
                        payload[
                            "tie_fraction"
                        ]
                    ),
                    "margin_median": (
                        payload[
                            "margin_median"
                        ]
                    ),
                    "margin_p90": (
                        payload[
                            "margin_p90"
                        ]
                    ),
                    "margin_p99": (
                        payload[
                            "margin_p99"
                        ]
                    ),
                    "gate_factor_mean": (
                        payload[
                            "gate_factor_mean"
                        ]
                    ),
                    "candidate_displacement_mean": (
                        payload[
                            "candidate_displacement_mean"
                        ]
                    ),
                    "innovation_abs_mean": (
                        payload[
                            "innovation_abs_mean"
                        ]
                    ),
                    "deadband_gate_factor_mean": (
                        payload[
                            "deadband_gate_factor_mean"
                        ]
                    ),
                    "deadband_candidate_displacement_mean": (
                        payload[
                            "deadband_candidate_displacement_mean"
                        ]
                    ),
                    "gru_decomposition_max_abs_error_check": (
                        checks[
                            "gru_decomposition_max_abs_error"
                        ]
                    ),
                    "deterministic_threshold_mismatch_fraction_check": (
                        checks[
                            "deterministic_threshold_mismatch_fraction"
                        ]
                    ),
                    "checkpoint_sha256": (
                        checkpoint_sha(
                            record
                        )
                    ),
                }
            )

    return rows


def merge_condition_csvs(
    records: Dict[
        str,
        Dict,
    ],
    filename_template: str,
    primary_conditions: Sequence[
        str
    ],
) -> List[
    Dict
]:
    rows = []

    for condition in primary_conditions:
        record = records[
            condition
        ]

        for region in (
            "encoder",
            "decoder",
        ):
            path = require_file(
                record[
                    "dir"
                ]
                / filename_template.format(
                    region=region
                )
            )

            with path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                reader = csv.DictReader(
                    handle
                )

                for source_row in reader:
                    row = {
                        "condition": (
                            condition
                        ),
                        "group": (
                            condition_group(
                                condition
                            )
                        ),
                        "region": (
                            region
                        ),
                    }

                    row.update(
                        source_row
                    )

                    rows.append(
                        row
                    )

    return rows


def read_csv_rows(
    path: Path,
) -> List[
    Dict
]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def validate_lifetime_dir(
    path: Path,
) -> Tuple[
    Dict,
    List[
        Dict
    ],
]:
    require_file(
        path
        / "lifetime_binned_excess_error_complete.flag"
    )

    json_path = require_file(
        path
        / "lifetime_binned_excess_error.json"
    )

    csv_path = require_file(
        path
        / "lifetime_binned_excess_error.csv"
    )

    payload = load_json(
        json_path
    )

    if (
        payload.get(
            "same_checkpoint_required"
        )
        is not True
    ):
        raise RuntimeError(
            "Lifetime analysis did not "
            "enforce same checkpoint: "
            f"{json_path}"
        )

    rows = read_csv_rows(
        csv_path
    )

    if not rows:
        raise RuntimeError(
            "Lifetime analysis CSV is empty: "
            f"{csv_path}"
        )

    return (
        payload,
        rows,
    )


def main() -> None:
    started = time.time()

    args = parse_args()

    validate_args(
        args
    )

    memoq_run = require_dir(
        Path(
            args.memoq_run_dir
        ).resolve()
    )

    vanilla4_run = require_dir(
        Path(
            args.vanilla4_run_dir
        ).resolve()
    )

    vanilla8_run = require_dir(
        Path(
            args.vanilla8_run_dir
        ).resolve()
    )

    lifetime_p2e_dir = require_dir(
        Path(
            args.lifetime_p2e_dir
        ).resolve()
    )

    lifetime_p2f_dir = require_dir(
        Path(
            args.lifetime_p2f_dir
        ).resolve()
    )

    smoke_dir = require_dir(
        Path(
            args.smoke_validation_dir
        ).resolve()
    )

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    require_file(
        smoke_dir
        / "recurrent_memory_smoke_validation_complete.flag"
    )

    smoke_json = require_file(
        smoke_dir
        / "recurrent_memory_smoke_validation.json"
    )

    smoke_payload = load_json(
        smoke_json
    )

    if (
        smoke_payload.get(
            "passed"
        )
        is not True
    ):
        raise RuntimeError(
            "Smoke validation did not pass"
        )

    memoq_root = require_dir(
        memoq_run
        / "recurrent_memory_analysis"
    )

    v4_root = require_dir(
        vanilla4_run
        / "recurrent_memory_analysis"
    )

    v8_root = require_dir(
        vanilla8_run
        / "recurrent_memory_analysis"
    )

    records: Dict[
        str,
        Dict,
    ] = {}

    for name in (
        MEMOQ_CONDITIONS
    ):
        records[
            name
        ] = load_condition(
            memoq_root,
            name,
        )

    for name in (
        V4_CONDITIONS
    ):
        records[
            name
        ] = load_condition(
            v4_root,
            name,
        )

    for name in (
        V8_CONDITIONS
    ):
        records[
            name
        ] = load_condition(
            v8_root,
            name,
        )

    require_same_checkpoint(
        "P2E",
        [
            records[
                name
            ]
            for name
            in MEMOQ_CONDITIONS
            if name.startswith(
                "P2E_"
            )
        ],
    )

    require_same_checkpoint(
        "P2F",
        [
            records[
                name
            ]
            for name
            in MEMOQ_CONDITIONS
            if name.startswith(
                "P2F_"
            )
        ],
    )

    require_same_checkpoint(
        "P3",
        [
            records[
                name
            ]
            for name
            in MEMOQ_CONDITIONS
            if name.startswith(
                "P3_"
            )
        ],
    )

    require_same_checkpoint(
        "Native B4",
        [
            records[
                name
            ]
            for name
            in V4_CONDITIONS
        ],
    )

    require_same_checkpoint(
        "Native B8",
        [
            records[
                name
            ]
            for name
            in V8_CONDITIONS
        ],
    )

    common_test_hash = (
        require_same_test_partition(
            list(
                records.values()
            )
        )
    )

    performance_rows = [
        performance_row(
            records[
                name
            ]
        )
        for name
        in (
            MEMOQ_CONDITIONS
            + V4_CONDITIONS
            + V8_CONDITIONS
        )
    ]

    performance_by_name = {
        row[
            "condition"
        ]: row
        for row
        in performance_rows
    }

    equal_storage_rows = (
        build_equal_storage_rows(
            records,
            performance_by_name,
            args.paired_bootstrap_reps,
            args.paired_bootstrap_seed,
            args.paired_bootstrap_batch_reps,
        )
    )

    margin_rows = (
        build_margin_rows(
            records
        )
    )

    margin_histogram_rows = (
        merge_condition_csvs(
            records,
            "{region}_margin_histogram.csv",
            PRIMARY_MECHANISM_CONDITIONS,
        )
    )

    joint_rows = (
        merge_condition_csvs(
            records,
            "{region}_mechanism_joint_gate_candidate.csv",
            PRIMARY_MECHANISM_CONDITIONS,
        )
    )

    (
        p2e_lifetime_payload,
        p2e_lifetime_rows,
    ) = validate_lifetime_dir(
        lifetime_p2e_dir
    )

    (
        p2f_lifetime_payload,
        p2f_lifetime_rows,
    ) = validate_lifetime_dir(
        lifetime_p2f_dir
    )

    if (
        p2e_lifetime_payload.get(
            "checkpoint_sha256"
        )
        != checkpoint_sha(
            records[
                "P2E_identity"
            ]
        )
    ):
        raise RuntimeError(
            "P2E lifetime analysis checkpoint "
            "does not match P2E conditions"
        )

    if (
        p2f_lifetime_payload.get(
            "checkpoint_sha256"
        )
        != checkpoint_sha(
            records[
                "P2F_identity"
            ]
        )
    ):
        raise RuntimeError(
            "P2F lifetime analysis checkpoint "
            "does not match P2F conditions"
        )

    if (
        p2e_lifetime_payload.get(
            "test_idx_sha256"
        )
        != common_test_hash
    ):
        raise RuntimeError(
            "P2E lifetime analysis "
            "test partition mismatch"
        )

    if (
        p2f_lifetime_payload.get(
            "test_idx_sha256"
        )
        != common_test_hash
    ):
        raise RuntimeError(
            "P2F lifetime analysis "
            "test partition mismatch"
        )

    lifetime_rows = (
        p2e_lifetime_rows
        + p2f_lifetime_rows
    )

    write_csv(
        out_dir
        / "method_sweep.csv",
        performance_rows,
    )

    write_csv(
        out_dir
        / "equal_storage_comparison.csv",
        equal_storage_rows,
    )

    write_csv(
        out_dir
        / "margin_decomposition.csv",
        margin_rows,
    )

    write_csv(
        out_dir
        / "margin_histograms.csv",
        margin_histogram_rows,
    )

    write_csv(
        out_dir
        / "margin_joint_gate_candidate.csv",
        joint_rows,
    )

    write_csv(
        out_dir
        / "lifetime_binned_excess_error.csv",
        lifetime_rows,
    )

    shutil.copy2(
        lifetime_p2e_dir
        / "lifetime_binned_excess_error.json",
        out_dir
        / "lifetime_binned_P2E_det_vs_identity.json",
    )

    shutil.copy2(
        lifetime_p2f_dir
        / "lifetime_binned_excess_error.json",
        out_dir
        / "lifetime_binned_P2F_det_vs_identity.json",
    )

    shutil.copy2(
        smoke_json,
        out_dir
        / "recurrent_memory_smoke_validation.json",
    )

    input_files = {}

    for record in records.values():
        for key in (
            "summary_path",
            "manifest_path",
            "prediction_path",
        ):
            path = record[
                key
            ]

            input_files[
                str(
                    path
                )
            ] = sha256_file(
                path
            )

    for path in (
        lifetime_p2e_dir
        / "lifetime_binned_excess_error.csv",
        lifetime_p2e_dir
        / "lifetime_binned_excess_error.json",
        lifetime_p2f_dir
        / "lifetime_binned_excess_error.csv",
        lifetime_p2f_dir
        / "lifetime_binned_excess_error.json",
        smoke_json,
    ):
        input_files[
            str(
                path
            )
        ] = sha256_file(
            path
        )

    payload = {
        "passed": True,
        "test_idx_sha256": (
            common_test_hash
        ),
        "checkpoint_sha256": {
            "P2E": (
                checkpoint_sha(
                    records[
                        "P2E_identity"
                    ]
                )
            ),
            "P2F": (
                checkpoint_sha(
                    records[
                        "P2F_identity"
                    ]
                )
            ),
            "P3": (
                checkpoint_sha(
                    records[
                        "P3_identity"
                    ]
                )
            ),
            "Native_B4": (
                checkpoint_sha(
                    records[
                        "V4_det_B4"
                    ]
                )
            ),
            "Native_B8": (
                checkpoint_sha(
                    records[
                        "V8_det_B8"
                    ]
                )
            ),
        },
        "paired_equal_storage_bootstrap": {
            "reps": (
                args.paired_bootstrap_reps
            ),
            "seed": (
                args.paired_bootstrap_seed
            ),
            "batch_reps": (
                args.paired_bootstrap_batch_reps
            ),
            "pairing": (
                "same held-out sequence indices "
                "are resampled jointly for "
                "state-only and alternative "
                "writeback conditions"
            ),
        },
        "smoke_validation": (
            smoke_payload
        ),
        "outputs": {
            "method_sweep": str(
                out_dir
                / "method_sweep.csv"
            ),
            "equal_storage_comparison": str(
                out_dir
                / "equal_storage_comparison.csv"
            ),
            "margin_decomposition": str(
                out_dir
                / "margin_decomposition.csv"
            ),
            "margin_histograms": str(
                out_dir
                / "margin_histograms.csv"
            ),
            "margin_joint_gate_candidate": str(
                out_dir
                / "margin_joint_gate_candidate.csv"
            ),
            "lifetime_binned_excess_error": str(
                out_dir
                / "lifetime_binned_excess_error.csv"
            ),
        },
        "input_files": (
            input_files
        ),
        "analysis_script_sha256": (
            sha256_file(
                THIS_FILE
            )
        ),
        "elapsed_seconds": float(
            time.time()
            - started
        ),
    }

    atomic_write_json(
        out_dir
        / "recurrent_memory_study.json",
        payload,
    )

    with (
        out_dir
        / "recurrent_memory_study_complete.flag"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "passed\n"
        )

    print(
        "[DONE] recurrent-memory "
        "study aggregation passed",
        flush=True,
    )

    print(
        f"[OUT] {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()