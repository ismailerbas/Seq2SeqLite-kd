#!/usr/bin/env python3
"""
eval/build_recurrent_memory_results.py

Fail-closed aggregation for the recurrent-margin, residual-memory, and
Saturating-Counter Writeback study.

The script expects the exact condition directory names documented in the
submission matrix below. It refuses to aggregate partial runs, checkpoint
mixes, or mismatched held-out partitions.

No numerical result is hard-coded. Every performance and mechanism value is
read from completed recurrent-memory analyses.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

THIS_FILE = Path(__file__).resolve()


MEMOQ_CONDITIONS = (
    "P2E_identity",

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
)

MARGIN_PRIMARY = (
    "P2F_det_B4",
    "P3_det_B4",
    "V4_det_B4",
    "V8_det_B8",
    "V8_forced_B4",
)

METHOD_LABELS = {
    "deterministic": "Deterministic",
    "error_feedback": "Error feedback",
    "stochastic": "Stochastic rounding",
    "quantized_residual": "Quantized residual",
    "full_halfstep_residual": "Full half-step residual",
    "scw": "SCW",
    "identity": "Identity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the complete recurrent-memory study."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--lifetime-analysis-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
    )

    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
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
        handle.write("\n")

    os.replace(tmp, path)


def require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory does not exist: {path}"
        )
    return path


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )
    return path


def load_condition(root: Path, name: str) -> Dict:
    condition_dir = require_dir(
        root / name
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

    summary = load_json(
        summary_path
    )
    manifest = load_json(
        manifest_path
    )

    if summary.get("condition_name") != name:
        raise RuntimeError(
            f"Condition-name mismatch in {summary_path}: "
            f"{summary.get('condition_name')!r} != {name!r}"
        )

    if manifest.get("condition_name") != name:
        raise RuntimeError(
            f"Condition-name mismatch in {manifest_path}: "
            f"{manifest.get('condition_name')!r} != {name!r}"
        )

    adapter = summary.get(
        "adapter_equivalence"
    )

    if (
        not isinstance(adapter, dict)
        or adapter.get("passed") is not True
    ):
        raise RuntimeError(
            f"Adapter equivalence did not pass for {name}"
        )

    return {
        "name": name,
        "dir": condition_dir,
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary": summary,
        "manifest": manifest,
    }


def nested_get(payload: Dict, path: str):
    value = payload

    for key in path.split("."):
        if (
            not isinstance(value, dict)
            or key not in value
        ):
            return None
        value = value[key]

    return value


def realization_values(record: Dict, path: str) -> np.ndarray:
    realizations = record["summary"].get(
        "realizations"
    )

    if (
        not isinstance(realizations, list)
        or not realizations
    ):
        raise RuntimeError(
            f"No realizations in {record['summary_path']}"
        )

    values = []

    for realization in realizations:
        value = nested_get(
            realization,
            path,
        )

        if value is None:
            raise RuntimeError(
                f"Missing {path} in {record['summary_path']}"
            )

        values.append(
            float(value)
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def realization_mean_std(
    record: Dict,
    path: str,
) -> Tuple[float, float]:
    values = realization_values(
        record,
        path,
    )

    mean = float(
        np.mean(values)
    )

    std = (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0
    )

    return (
        mean,
        std,
    )


def mechanism_mean(
    record: Dict,
    region: str,
    key: str,
) -> float:
    mechanisms = record["summary"].get(
        "mechanism_realizations"
    )

    if (
        not isinstance(mechanisms, list)
        or not mechanisms
    ):
        raise RuntimeError(
            f"No mechanism realizations in {record['summary_path']}"
        )

    values = []

    for realization in mechanisms:
        region_payload = realization.get(
            region
        )

        if (
            not isinstance(region_payload, dict)
            or key not in region_payload
        ):
            raise RuntimeError(
                f"Missing mechanism {region}.{key} "
                f"in {record['summary_path']}"
            )

        value = region_payload[key]

        if value is None:
            raise RuntimeError(
                f"Mechanism {region}.{key} is null "
                f"in {record['summary_path']}"
            )

        values.append(
            float(value)
        )

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def checkpoint_sha(record: Dict) -> str:
    value = record["manifest"].get(
        "checkpoint_sha256"
    )

    if (
        not isinstance(value, str)
        or len(value) != 64
    ):
        raise RuntimeError(
            f"Invalid checkpoint_sha256 "
            f"in {record['manifest_path']}"
        )

    return value


def test_idx_sha(record: Dict) -> str:
    value = record["manifest"].get(
        "test_idx_sha256"
    )

    if (
        not isinstance(value, str)
        or len(value) != 64
    ):
        raise RuntimeError(
            f"Invalid test_idx_sha256 "
            f"in {record['manifest_path']}"
        )

    return value


def require_same_checkpoint(
    name: str,
    records: Sequence[Dict],
) -> str:
    hashes = sorted(
        set(
            checkpoint_sha(record)
            for record in records
        )
    )

    if len(hashes) != 1:
        raise RuntimeError(
            f"Checkpoint identity failure for {name}: {hashes}"
        )

    return hashes[0]


def require_same_test_partition(
    records: Sequence[Dict],
) -> str:
    hashes = sorted(
        set(
            test_idx_sha(record)
            for record in records
        )
    )

    if len(hashes) != 1:
        raise RuntimeError(
            f"Held-out test partition mismatch: {hashes}"
        )

    return hashes[0]


def condition_group(name: str) -> str:
    if name.startswith("P2E_"):
        return "P2E"

    if name.startswith("P2F_"):
        return "P2F"

    if name.startswith("P3_"):
        return "P3"

    if name.startswith("V4_"):
        return "Native B4"

    if name.startswith("V8_"):
        return "Native B8"

    raise RuntimeError(
        f"Cannot infer group from condition name {name}"
    )


def performance_row(record: Dict) -> Dict:
    operator = record["summary"].get(
        "operator"
    )

    if not isinstance(operator, dict):
        raise RuntimeError(
            f"Missing operator metadata in {record['summary_path']}"
        )

    method = record["summary"].get(
        "method"
    )

    if method not in METHOD_LABELS:
        raise RuntimeError(
            f"Unsupported method {method!r} "
            f"in {record['summary_path']}"
        )

    (
        mae_seq,
        mae_seq_std,
    ) = realization_mean_std(
        record,
        "metrics.mae_seq",
    )

    (
        tau1,
        tau1_std,
    ) = realization_mean_std(
        record,
        "metrics.rmse_tau1",
    )

    (
        tau2,
        tau2_std,
    ) = realization_mean_std(
        record,
        "metrics.rmse_tau2",
    )

    (
        r_tau1,
        r_tau1_std,
    ) = realization_mean_std(
        record,
        "metrics.r_tau1",
    )

    (
        r_tau2,
        r_tau2_std,
    ) = realization_mean_std(
        record,
        "metrics.r_tau2",
    )

    (
        decoder_p_w0,
        decoder_p_w0_std,
    ) = realization_mean_std(
        record,
        "decoder.p_w0",
    )

    (
        decoder_mean_nwrite,
        decoder_mean_nwrite_std,
    ) = realization_mean_std(
        record,
        "decoder.mean_nwrite",
    )

    (
        encoder_p_w0,
        encoder_p_w0_std,
    ) = realization_mean_std(
        record,
        "encoder.p_w0",
    )

    (
        encoder_mean_nwrite,
        encoder_mean_nwrite_std,
    ) = realization_mean_std(
        record,
        "encoder.mean_nwrite",
    )

    n_test_values = realization_values(
        record,
        "metrics.n_test",
    )

    if not np.all(
        n_test_values == n_test_values[0]
    ):
        raise RuntimeError(
            f"n_test differs across realizations "
            f"in {record['summary_path']}"
        )

    residual_bits = operator.get(
        "residual_bits"
    )

    counter_bits = operator.get(
        "counter_bits"
    )

    total_bits = operator.get(
        "total_stored_bits_per_unit"
    )

    return {
        "condition": record["name"],
        "group": condition_group(record["name"]),
        "method": method,
        "method_label": METHOD_LABELS[method],
        "state_bits": int(operator["state_bits"]),
        "residual_bits": residual_bits,
        "counter_bits": counter_bits,
        "total_stored_bits_per_unit": total_bits,
        "deadzone_fraction_of_delta": operator.get(
            "deadzone_fraction_of_delta"
        ),
        "trigger_votes": operator.get(
            "trigger_votes"
        ),
        "n_realizations": int(
            record["summary"]["n_realizations"]
        ),
        "n_test": int(
            round(n_test_values[0])
        ),
        "mae_seq": mae_seq,
        "mae_seq_realization_std": mae_seq_std,
        "rmse_tau1": tau1,
        "rmse_tau1_realization_std": tau1_std,
        "rmse_tau2": tau2,
        "rmse_tau2_realization_std": tau2_std,
        "r_tau1": r_tau1,
        "r_tau1_realization_std": r_tau1_std,
        "r_tau2": r_tau2,
        "r_tau2_realization_std": r_tau2_std,
        "encoder_p_w0": encoder_p_w0,
        "encoder_p_w0_realization_std": encoder_p_w0_std,
        "encoder_mean_nwrite": encoder_mean_nwrite,
        "encoder_mean_nwrite_realization_std": (
            encoder_mean_nwrite_std
        ),
        "decoder_p_w0": decoder_p_w0,
        "decoder_p_w0_realization_std": decoder_p_w0_std,
        "decoder_mean_nwrite": decoder_mean_nwrite,
        "decoder_mean_nwrite_realization_std": (
            decoder_mean_nwrite_std
        ),
        "encoder_deadband_fraction": mechanism_mean(
            record,
            "encoder",
            "deadband_fraction",
        ),
        "decoder_deadband_fraction": mechanism_mean(
            record,
            "decoder",
            "deadband_fraction",
        ),
        "encoder_gate_factor_mean": mechanism_mean(
            record,
            "encoder",
            "gate_factor_mean",
        ),
        "decoder_gate_factor_mean": mechanism_mean(
            record,
            "decoder",
            "gate_factor_mean",
        ),
        "encoder_candidate_displacement_mean": mechanism_mean(
            record,
            "encoder",
            "candidate_displacement_mean",
        ),
        "decoder_candidate_displacement_mean": mechanism_mean(
            record,
            "decoder",
            "candidate_displacement_mean",
        ),
        "checkpoint_sha256": checkpoint_sha(record),
        "test_idx_sha256": test_idx_sha(record),
        "source_dir": str(record["dir"]),
    }


def write_csv(
    path: Path,
    rows: Sequence[Dict],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

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

        for row in rows:
            writer.writerow(row)


def build_equal_storage_rows(
    performance_by_name: Dict[str, Dict],
) -> List[Dict]:
    rows = []

    for total_bits in (
        6,
        7,
        8,
    ):
        extra_bits = total_bits - 4

        state_name = f"V4_det_B{total_bits}"
        residual_name = (
            f"V4_residual_R{extra_bits}_B4"
        )

        state_row = performance_by_name[
            state_name
        ]
        residual_row = performance_by_name[
            residual_name
        ]

        rows.append(
            {
                "total_stored_bits_per_unit": total_bits,
                "comparison_family": "quantized_residual",
                "state_only_condition": state_name,
                "alternative_condition": residual_name,
                "state_only_rmse_tau1": state_row[
                    "rmse_tau1"
                ],
                "alternative_rmse_tau1": residual_row[
                    "rmse_tau1"
                ],
                "alternative_minus_state_rmse_tau1": (
                    residual_row["rmse_tau1"]
                    - state_row["rmse_tau1"]
                ),
                "state_only_rmse_tau2": state_row[
                    "rmse_tau2"
                ],
                "alternative_rmse_tau2": residual_row[
                    "rmse_tau2"
                ],
                "alternative_minus_state_rmse_tau2": (
                    residual_row["rmse_tau2"]
                    - state_row["rmse_tau2"]
                ),
                "state_only_mae_seq": state_row[
                    "mae_seq"
                ],
                "alternative_mae_seq": residual_row[
                    "mae_seq"
                ],
                "alternative_minus_state_mae_seq": (
                    residual_row["mae_seq"]
                    - state_row["mae_seq"]
                ),
                "checkpoint_sha256": state_row[
                    "checkpoint_sha256"
                ],
            }
        )

        for (
            deadzone_label,
            comparison_family,
        ) in (
            (
                "TH0",
                "scw_theta0",
            ),
            (
                "TH1_8",
                "scw_theta_delta_over_8",
            ),
        ):
            scw_name = (
                f"V4_scw_K{extra_bits}_"
                f"{deadzone_label}_B4"
            )

            scw_row = performance_by_name[
                scw_name
            ]

            rows.append(
                {
                    "total_stored_bits_per_unit": total_bits,
                    "comparison_family": comparison_family,
                    "state_only_condition": state_name,
                    "alternative_condition": scw_name,
                    "state_only_rmse_tau1": state_row[
                        "rmse_tau1"
                    ],
                    "alternative_rmse_tau1": scw_row[
                        "rmse_tau1"
                    ],
                    "alternative_minus_state_rmse_tau1": (
                        scw_row["rmse_tau1"]
                        - state_row["rmse_tau1"]
                    ),
                    "state_only_rmse_tau2": state_row[
                        "rmse_tau2"
                    ],
                    "alternative_rmse_tau2": scw_row[
                        "rmse_tau2"
                    ],
                    "alternative_minus_state_rmse_tau2": (
                        scw_row["rmse_tau2"]
                        - state_row["rmse_tau2"]
                    ),
                    "state_only_mae_seq": state_row[
                        "mae_seq"
                    ],
                    "alternative_mae_seq": scw_row[
                        "mae_seq"
                    ],
                    "alternative_minus_state_mae_seq": (
                        scw_row["mae_seq"]
                        - state_row["mae_seq"]
                    ),
                    "checkpoint_sha256": state_row[
                        "checkpoint_sha256"
                    ],
                }
            )

    return rows


def build_margin_rows(
    records: Dict[str, Dict],
) -> List[Dict]:
    rows = []

    for name in MARGIN_PRIMARY:
        record = records[name]
        summary = record["summary"]
        mechanisms = summary.get(
            "mechanism_realizations"
        )

        if (
            not isinstance(mechanisms, list)
            or len(mechanisms) != 1
        ):
            raise RuntimeError(
                f"Primary deterministic margin condition {name} "
                "must have exactly one realization"
            )

        for region in (
            "encoder",
            "decoder",
        ):
            payload = mechanisms[0][region]

            rows.append(
                {
                    "condition": name,
                    "group": condition_group(name),
                    "region": region,
                    "state_bits": payload[
                        "state_bits"
                    ],
                    "delta": payload[
                        "delta"
                    ],
                    "half_step": payload[
                        "half_step"
                    ],
                    "deadband_fraction": payload[
                        "deadband_fraction"
                    ],
                    "write_fraction": payload[
                        "write_fraction"
                    ],
                    "rail_fraction": payload[
                        "rail_fraction"
                    ],
                    "tie_fraction": payload[
                        "tie_fraction"
                    ],
                    "margin_median": payload[
                        "margin_median"
                    ],
                    "margin_p90": payload[
                        "margin_p90"
                    ],
                    "margin_p99": payload[
                        "margin_p99"
                    ],
                    "gate_factor_mean": payload[
                        "gate_factor_mean"
                    ],
                    "candidate_displacement_mean": payload[
                        "candidate_displacement_mean"
                    ],
                    "innovation_abs_mean": payload[
                        "innovation_abs_mean"
                    ],
                    "deadband_gate_factor_mean": payload[
                        "deadband_gate_factor_mean"
                    ],
                    "deadband_candidate_displacement_mean": payload[
                        "deadband_candidate_displacement_mean"
                    ],
                    "decomposition_max_abs_error": payload[
                        "decomposition_max_abs_error"
                    ],
                    "deterministic_criterion_mismatch_fraction": payload[
                        "deterministic_criterion_mismatch_fraction"
                    ],
                    "checkpoint_sha256": checkpoint_sha(record),
                }
            )

    return rows


def read_csv_rows(path: Path) -> List[Dict]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def main() -> None:
    started = time.time()

    args = parse_args()

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

    lifetime_dir = require_dir(
        Path(
            args.lifetime_analysis_dir
        ).resolve()
    )

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    records: Dict[str, Dict] = {}

    for name in MEMOQ_CONDITIONS:
        records[name] = load_condition(
            memoq_root,
            name,
        )

    for name in V4_CONDITIONS:
        records[name] = load_condition(
            v4_root,
            name,
        )

    for name in V8_CONDITIONS:
        records[name] = load_condition(
            v8_root,
            name,
        )

    require_same_checkpoint(
        "P2F",
        [
            records[name]
            for name in MEMOQ_CONDITIONS
            if name.startswith("P2F_")
        ],
    )

    require_same_checkpoint(
        "P3",
        [
            records[name]
            for name in MEMOQ_CONDITIONS
            if name.startswith("P3_")
        ],
    )

    require_same_checkpoint(
        "Native B4",
        [
            records[name]
            for name in V4_CONDITIONS
        ],
    )

    require_same_checkpoint(
        "Native B8",
        [
            records[name]
            for name in V8_CONDITIONS
        ],
    )

    common_test_hash = require_same_test_partition(
        list(records.values())
    )

    performance_rows = [
        performance_row(records[name])
        for name in (
            MEMOQ_CONDITIONS
            + V4_CONDITIONS
            + V8_CONDITIONS
        )
        if name != "P2E_identity"
    ]

    performance_by_name = {
        row["condition"]: row
        for row in performance_rows
    }

    equal_storage_rows = build_equal_storage_rows(
        performance_by_name
    )

    margin_rows = build_margin_rows(
        records
    )

    lifetime_flag = require_file(
        lifetime_dir
        / "lifetime_binned_excess_error_complete.flag"
    )

    lifetime_csv = require_file(
        lifetime_dir
        / "lifetime_binned_excess_error.csv"
    )

    lifetime_json = require_file(
        lifetime_dir
        / "lifetime_binned_excess_error.json"
    )

    lifetime_payload = load_json(
        lifetime_json
    )

    lifetime_rows = read_csv_rows(
        lifetime_csv
    )

    if int(
        lifetime_payload.get(
            "n_test",
            -1,
        )
    ) <= 0:
        raise RuntimeError(
            "Lifetime analysis has invalid n_test"
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
        / "lifetime_binned_excess_error.csv",
        lifetime_rows,
    )

    shutil.copy2(
        lifetime_json,
        out_dir
        / "lifetime_binned_excess_error.json",
    )

    input_files = {}

    for record in records.values():
        for key in (
            "summary_path",
            "manifest_path",
        ):
            path = record[key]
            input_files[str(path)] = sha256_file(path)

    input_files[
        str(lifetime_csv)
    ] = sha256_file(
        lifetime_csv
    )

    input_files[
        str(lifetime_json)
    ] = sha256_file(
        lifetime_json
    )

    input_files[
        str(lifetime_flag)
    ] = sha256_file(
        lifetime_flag
    )

    payload = {
        "passed": True,
        "test_idx_sha256": common_test_hash,
        "checkpoint_sha256": {
            "P2E": checkpoint_sha(
                records["P2E_identity"]
            ),
            "P2F": checkpoint_sha(
                records["P2F_det_B4"]
            ),
            "P3": checkpoint_sha(
                records["P3_det_B4"]
            ),
            "Native_B4": checkpoint_sha(
                records["V4_det_B4"]
            ),
            "Native_B8": checkpoint_sha(
                records["V8_det_B8"]
            ),
        },
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
            "lifetime_binned_excess_error_csv": str(
                out_dir
                / "lifetime_binned_excess_error.csv"
            ),
            "lifetime_binned_excess_error_json": str(
                out_dir
                / "lifetime_binned_excess_error.json"
            ),
        },
        "input_files": input_files,
        "analysis_script_sha256": sha256_file(THIS_FILE),
        "elapsed_seconds": float(
            time.time() - started
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
        "[DONE] recurrent-memory study aggregation passed",
        flush=True,
    )

    print(
        f"[OUT] {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()