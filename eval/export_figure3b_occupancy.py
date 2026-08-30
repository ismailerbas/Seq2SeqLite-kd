#!/usr/bin/env python3

"""
eval/export_figure3b_occupancy.py

Fail-closed extraction of the paired decoder per-unit occupied-state-level
vectors required for Figure 3b.

The extraction uses the completed recurrent-memory analysis products for the
same trained vanilla B4 model under:

    V4_det_B4
        deterministic 4-bit recurrent-state writeback

    V4_det_B8
        deterministic 8-bit recurrent-state writeback

No model is trained.
No checkpoint is modified.
No recurrent computation is reimplemented.
No manuscript values are substituted for missing per-unit data.

The script searches the existing completed condition outputs for explicit
per-unit decoder occupied-level products, validates the condition provenance,
aligns the same 32 decoder units by unit index, and verifies the resulting
medians against the already reported manuscript values:

    V4_det_B4 decoder median occupied levels = 12.0
    V4_det_B8 decoder median occupied levels = 176.5

The manuscript medians are validation targets only. They are never used to
construct, impute, interpolate, or alter any per-unit value.

If the required per-unit data cannot be identified unambiguously, the script
fails without producing a Figure 3 input artifact. A discovery inventory is
still written so the exact available result schema can be inspected.

Successful outputs:

    results/figure3_inputs/
        figure3b_decoder_occupied_levels.csv
        figure3b_decoder_occupied_levels.npz
        figure3b_decoder_occupied_levels.mat
        figure3b_decoder_occupied_levels_manifest.json
        figure3b_occupancy_discovery.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scipy.io import savemat


THIS_FILE = Path(__file__).resolve()
EVAL_DIR = THIS_FILE.parent
REPO_ROOT = EVAL_DIR.parent

EXPECTED_UNITS = 32

V4_RUN_NAME = (
    "vanilla_kd_T4.0_a0.6_b4k4r4a4s4_gru32x1_dense3_"
    "effbs1024_microbs1024_lr1e-04"
)

CONDITION_B4 = "V4_det_B4"
CONDITION_B8 = "V4_det_B8"

EXPECTED_MEDIAN_B4 = 12.0
EXPECTED_MEDIAN_B8 = 176.5

EXPECTED_PHASE = "VANILLA"
EXPECTED_METHOD = "deterministic"

FLOAT_TOLERANCE = 1.0e-9
INTEGER_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class Candidate:
    condition_name: str
    source_path: Path
    source_kind: str
    locator: str
    explicit_unit_ids: bool
    unit_indices_zero_based: np.ndarray
    values: np.ndarray

    def as_dict(self) -> Dict[str, Any]:
        return {
            "condition_name": self.condition_name,
            "source_path": str(self.source_path),
            "source_kind": self.source_kind,
            "locator": self.locator,
            "explicit_unit_ids": bool(self.explicit_unit_ids),
            "unit_indices_zero_based": (
                self.unit_indices_zero_based.astype(
                    np.int64
                ).tolist()
            ),
            "values": (
                self.values.astype(
                    np.float64
                ).tolist()
            ),
            "median": float(
                np.median(
                    self.values
                )
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the paired per-unit decoder occupied-level vectors "
            "required for Figure 3b from completed recurrent-memory results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(
            REPO_ROOT
        ),
        help=(
            "Seq2SeqLite-kd project root containing eval/ and results/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory. If omitted, uses "
            "<project-root>/results/figure3_inputs."
        ),
    )

    return parser.parse_args()


def normalize_name(
    value: Any,
) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(
            value
        ).lower(),
    )


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                chunk_size
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            f"Expected JSON object in {path}, "
            f"got {type(payload).__name__}"
        )

    return payload


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
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


def is_numeric_scalar(
    value: Any,
) -> bool:
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return False

    return isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    )


def parse_float_value(
    value: Any,
) -> float:
    if value is None:
        raise ValueError(
            "value is None"
        )

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        raise ValueError(
            "boolean is not a numeric occupied-level count"
        )

    parsed = float(
        value
    )

    if not math.isfinite(
        parsed
    ):
        raise ValueError(
            f"non-finite numeric value: {value!r}"
        )

    return parsed


def parse_integer_value(
    value: Any,
) -> int:
    parsed = parse_float_value(
        value
    )

    rounded = int(
        round(
            parsed
        )
    )

    if not math.isclose(
        parsed,
        float(
            rounded
        ),
        rel_tol=0.0,
        abs_tol=INTEGER_TOLERANCE,
    ):
        raise ValueError(
            f"expected integer-valued index, got {value!r}"
        )

    return rounded


def occupied_level_field(
    name: Any,
) -> bool:
    norm = normalize_name(
        name
    )

    return (
        "occupied" in norm
        and "level" in norm
        and "median" not in norm
        and "mean" not in norm
    )


def unit_field_priority(
    name: Any,
) -> Optional[int]:
    norm = normalize_name(
        name
    )

    priorities = {
        "unitindex": 0,
        "hiddenunitindex": 1,
        "unitid": 2,
        "hiddenunitid": 3,
        "unit": 4,
        "hiddenunit": 5,
        "neuronindex": 6,
        "neuronid": 7,
        "neuron": 8,
    }

    return priorities.get(
        norm
    )


def region_field(
    name: Any,
) -> bool:
    norm = normalize_name(
        name
    )

    return norm in {
        "region",
        "recurrentregion",
        "recurrentlayer",
        "layer",
        "section",
    }


def is_decoder_value(
    value: Any,
) -> bool:
    norm = normalize_name(
        value
    )

    return norm in {
        "decoder",
        "dec",
        "sdecgru",
        "decodergru",
    }


def canonicalize_unit_indices(
    raw_indices: Sequence[Any],
) -> np.ndarray:
    parsed = np.asarray(
        [
            parse_integer_value(
                value
            )
            for value in raw_indices
        ],
        dtype=np.int64,
    )

    if parsed.shape != (
        EXPECTED_UNITS,
    ):
        raise ValueError(
            "unit-index vector does not contain exactly "
            f"{EXPECTED_UNITS} entries"
        )

    if len(
        np.unique(
            parsed
        )
    ) != EXPECTED_UNITS:
        raise ValueError(
            "unit-index vector contains duplicate units"
        )

    sorted_raw = np.sort(
        parsed
    )

    zero_based_expected = np.arange(
        EXPECTED_UNITS,
        dtype=np.int64,
    )

    one_based_expected = np.arange(
        1,
        EXPECTED_UNITS
        + 1,
        dtype=np.int64,
    )

    if np.array_equal(
        sorted_raw,
        zero_based_expected,
    ):
        return parsed

    if np.array_equal(
        sorted_raw,
        one_based_expected,
    ):
        return (
            parsed
            - 1
        )

    raise ValueError(
        "unit indices are neither exactly 0..31 nor exactly 1..32"
    )


def canonicalize_candidate(
    condition_name: str,
    source_path: Path,
    source_kind: str,
    locator: str,
    explicit_unit_ids: bool,
    raw_unit_indices: Sequence[Any],
    raw_values: Sequence[Any],
) -> Candidate:
    unit_indices = canonicalize_unit_indices(
        raw_unit_indices
    )

    values = np.asarray(
        [
            parse_float_value(
                value
            )
            for value in raw_values
        ],
        dtype=np.float64,
    )

    if values.shape != (
        EXPECTED_UNITS,
    ):
        raise ValueError(
            "occupied-level vector does not contain exactly "
            f"{EXPECTED_UNITS} values"
        )

    order = np.argsort(
        unit_indices
    )

    sorted_units = unit_indices[
        order
    ]

    sorted_values = values[
        order
    ]

    if not np.array_equal(
        sorted_units,
        np.arange(
            EXPECTED_UNITS,
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "canonical unit ordering failed"
        )

    return Candidate(
        condition_name=condition_name,
        source_path=source_path.resolve(),
        source_kind=source_kind,
        locator=locator,
        explicit_unit_ids=explicit_unit_ids,
        unit_indices_zero_based=sorted_units,
        values=sorted_values,
    )


def candidate_path_text(
    path_parts: Sequence[Any],
) -> str:
    return ".".join(
        str(
            part
        )
        for part in path_parts
    )


def candidate_path_is_decoder_occupancy(
    path_parts: Sequence[Any],
) -> bool:
    norm = normalize_name(
        candidate_path_text(
            path_parts
        )
    )

    return (
        "decoder" in norm
        and "occupied" in norm
        and "level" in norm
    )


def discover_json_candidates(
    condition_name: str,
    path: Path,
) -> List[Candidate]:
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(
                handle
            )
    except Exception:
        return []

    candidates: List[Candidate] = []

    def visit(
        node: Any,
        node_path: Tuple[Any, ...],
    ) -> None:
        if isinstance(
            node,
            list,
        ):
            if (
                len(
                    node
                )
                == EXPECTED_UNITS
                and all(
                    is_numeric_scalar(
                        value
                    )
                    for value in node
                )
                and candidate_path_is_decoder_occupancy(
                    node_path
                )
            ):
                try:
                    candidate = canonicalize_candidate(
                        condition_name=condition_name,
                        source_path=path,
                        source_kind="json_numeric_vector",
                        locator=candidate_path_text(
                            node_path
                        ),
                        explicit_unit_ids=False,
                        raw_unit_indices=np.arange(
                            EXPECTED_UNITS,
                            dtype=np.int64,
                        ),
                        raw_values=node,
                    )

                    candidates.append(
                        candidate
                    )
                except ValueError:
                    pass

            if (
                len(
                    node
                )
                > 0
                and all(
                    isinstance(
                        item,
                        dict,
                    )
                    for item in node
                )
            ):
                records: List[Dict[str, Any]] = [
                    dict(
                        item
                    )
                    for item in node
                ]

                key_names = sorted(
                    {
                        str(
                            key
                        )
                        for record in records
                        for key in record.keys()
                    }
                )

                occupied_fields = [
                    key
                    for key in key_names
                    if occupied_level_field(
                        key
                    )
                ]

                unit_fields_ranked = sorted(
                    (
                        (
                            unit_field_priority(
                                key
                            ),
                            key,
                        )
                        for key in key_names
                        if unit_field_priority(
                            key
                        )
                        is not None
                    ),
                    key=lambda item: (
                        int(
                            item[0]
                        ),
                        item[1],
                    ),
                )

                region_fields = [
                    key
                    for key in key_names
                    if region_field(
                        key
                    )
                ]

                if (
                    occupied_fields
                    and unit_fields_ranked
                ):
                    unit_key = unit_fields_ranked[
                        0
                    ][1]

                    parent_is_decoder = (
                        "decoder"
                        in normalize_name(
                            candidate_path_text(
                                node_path
                            )
                        )
                    )

                    for occupied_key in occupied_fields:
                        filtered_records: List[
                            Dict[str, Any]
                        ] = []

                        if region_fields:
                            selected_region_key = (
                                region_fields[
                                    0
                                ]
                            )

                            for record in records:
                                if (
                                    selected_region_key
                                    not in record
                                ):
                                    continue

                                if is_decoder_value(
                                    record[
                                        selected_region_key
                                    ]
                                ):
                                    filtered_records.append(
                                        record
                                    )

                        elif parent_is_decoder:
                            filtered_records = records

                        if len(
                            filtered_records
                        ) != EXPECTED_UNITS:
                            continue

                        if any(
                            unit_key
                            not in record
                            or occupied_key
                            not in record
                            for record in filtered_records
                        ):
                            continue

                        try:
                            candidate = canonicalize_candidate(
                                condition_name=condition_name,
                                source_path=path,
                                source_kind="json_record_table",
                                locator=(
                                    candidate_path_text(
                                        node_path
                                    )
                                    + "::"
                                    + occupied_key
                                ),
                                explicit_unit_ids=True,
                                raw_unit_indices=[
                                    record[
                                        unit_key
                                    ]
                                    for record in filtered_records
                                ],
                                raw_values=[
                                    record[
                                        occupied_key
                                    ]
                                    for record in filtered_records
                                ],
                            )

                            candidates.append(
                                candidate
                            )
                        except ValueError:
                            pass

            for index, item in enumerate(
                node
            ):
                visit(
                    item,
                    node_path
                    + (
                        index,
                    ),
                )

            return

        if isinstance(
            node,
            dict,
        ):
            if (
                len(
                    node
                )
                == EXPECTED_UNITS
                and candidate_path_is_decoder_occupancy(
                    node_path
                )
                and all(
                    is_numeric_scalar(
                        value
                    )
                    for value in node.values()
                )
            ):
                try:
                    candidate = canonicalize_candidate(
                        condition_name=condition_name,
                        source_path=path,
                        source_kind="json_unit_mapping",
                        locator=candidate_path_text(
                            node_path
                        ),
                        explicit_unit_ids=True,
                        raw_unit_indices=list(
                            node.keys()
                        ),
                        raw_values=list(
                            node.values()
                        ),
                    )

                    candidates.append(
                        candidate
                    )
                except ValueError:
                    pass

            for key, value in node.items():
                visit(
                    value,
                    node_path
                    + (
                        key,
                    ),
                )

    visit(
        payload,
        tuple(),
    )

    return candidates


def read_csv_header(
    path: Path,
) -> List[str]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.reader(
                handle
            )

            return next(
                reader,
                [],
            )
    except Exception:
        return []


def discover_csv_candidates(
    condition_name: str,
    path: Path,
) -> List[Candidate]:
    header = read_csv_header(
        path
    )

    if not header:
        return []

    occupied_columns = [
        column
        for column in header
        if occupied_level_field(
            column
        )
    ]

    unit_columns_ranked = sorted(
        (
            (
                unit_field_priority(
                    column
                ),
                column,
            )
            for column in header
            if unit_field_priority(
                column
            )
            is not None
        ),
        key=lambda item: (
            int(
                item[0]
            ),
            item[1],
        ),
    )

    region_columns = [
        column
        for column in header
        if region_field(
            column
        )
    ]

    if (
        not occupied_columns
        or not unit_columns_ranked
    ):
        return []

    unit_column = unit_columns_ranked[
        0
    ][1]

    path_is_decoder = (
        "decoder"
        in normalize_name(
            str(
                path
            )
        )
    )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(
                handle
            )

            records = [
                row
                for row in reader
            ]
    except Exception:
        return []

    candidates: List[Candidate] = []

    for occupied_column in occupied_columns:
        filtered_records: List[
            Mapping[str, Any]
        ] = []

        if region_columns:
            selected_region_column = (
                region_columns[
                    0
                ]
            )

            for record in records:
                if is_decoder_value(
                    record.get(
                        selected_region_column,
                        "",
                    )
                ):
                    filtered_records.append(
                        record
                    )

        elif path_is_decoder:
            filtered_records = records

        if len(
            filtered_records
        ) != EXPECTED_UNITS:
            continue

        try:
            candidate = canonicalize_candidate(
                condition_name=condition_name,
                source_path=path,
                source_kind="csv_direct_per_unit",
                locator=(
                    f"column:{occupied_column}"
                ),
                explicit_unit_ids=True,
                raw_unit_indices=[
                    record[
                        unit_column
                    ]
                    for record in filtered_records
                ],
                raw_values=[
                    record[
                        occupied_column
                    ]
                    for record in filtered_records
                ],
            )

            candidates.append(
                candidate
            )
        except (
            KeyError,
            ValueError,
        ):
            pass

    return candidates


def discover_npz_candidates(
    condition_name: str,
    path: Path,
) -> List[Candidate]:
    candidates: List[Candidate] = []

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            for key in archive.files:
                norm = normalize_name(
                    key
                )

                if not (
                    "decoder" in norm
                    and "occupied" in norm
                    and "level" in norm
                ):
                    continue

                try:
                    value = np.asarray(
                        archive[
                            key
                        ]
                    )

                    value = np.squeeze(
                        value
                    )
                except Exception:
                    continue

                if value.shape != (
                    EXPECTED_UNITS,
                ):
                    continue

                if not np.issubdtype(
                    value.dtype,
                    np.number,
                ):
                    continue

                try:
                    candidate = canonicalize_candidate(
                        condition_name=condition_name,
                        source_path=path,
                        source_kind="npz_numeric_vector",
                        locator=f"key:{key}",
                        explicit_unit_ids=False,
                        raw_unit_indices=np.arange(
                            EXPECTED_UNITS,
                            dtype=np.int64,
                        ),
                        raw_values=value.tolist(),
                    )

                    candidates.append(
                        candidate
                    )
                except ValueError:
                    pass

    except Exception:
        return []

    return candidates


def discover_npy_candidates(
    condition_name: str,
    path: Path,
) -> List[Candidate]:
    norm = normalize_name(
        str(
            path
        )
    )

    if not (
        "decoder" in norm
        and "occupied" in norm
        and "level" in norm
    ):
        return []

    try:
        value = np.asarray(
            np.load(
                path,
                allow_pickle=False,
            )
        )

        value = np.squeeze(
            value
        )
    except Exception:
        return []

    if value.shape != (
        EXPECTED_UNITS,
    ):
        return []

    if not np.issubdtype(
        value.dtype,
        np.number,
    ):
        return []

    try:
        candidate = canonicalize_candidate(
            condition_name=condition_name,
            source_path=path,
            source_kind="npy_numeric_vector",
            locator="array",
            explicit_unit_ids=False,
            raw_unit_indices=np.arange(
                EXPECTED_UNITS,
                dtype=np.int64,
            ),
            raw_values=value.tolist(),
        )
    except ValueError:
        return []

    return [
        candidate
    ]


def discover_candidates(
    condition_name: str,
    condition_dir: Path,
) -> List[Candidate]:
    candidates: List[Candidate] = []

    for path in sorted(
        condition_dir.rglob(
            "*"
        )
    ):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if suffix == ".json":
            candidates.extend(
                discover_json_candidates(
                    condition_name=condition_name,
                    path=path,
                )
            )

        elif suffix == ".csv":
            candidates.extend(
                discover_csv_candidates(
                    condition_name=condition_name,
                    path=path,
                )
            )

        elif suffix == ".npz":
            candidates.extend(
                discover_npz_candidates(
                    condition_name=condition_name,
                    path=path,
                )
            )

        elif suffix == ".npy":
            candidates.extend(
                discover_npy_candidates(
                    condition_name=condition_name,
                    path=path,
                )
            )

    return candidates


def validate_condition_products(
    condition_dir: Path,
    condition_name: str,
    expected_state_bits: int,
) -> Dict[str, Any]:
    if not condition_dir.is_dir():
        raise FileNotFoundError(
            f"Condition directory does not exist: {condition_dir}"
        )

    complete_flag = (
        condition_dir
        / "recurrent_memory_complete.flag"
    )

    summary_path = (
        condition_dir
        / "recurrent_memory_summary.json"
    )

    manifest_path = (
        condition_dir
        / "recurrent_memory_manifest.json"
    )

    per_sequence_path = (
        condition_dir
        / "recurrent_memory_per_sequence.npz"
    )

    required_files = (
        complete_flag,
        summary_path,
        manifest_path,
        per_sequence_path,
    )

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required completed-condition product is missing: {path}"
            )

    flag_value = complete_flag.read_text(
        encoding="utf-8"
    ).strip()

    if flag_value != condition_name:
        raise RuntimeError(
            f"{complete_flag} contains {flag_value!r}; "
            f"expected {condition_name!r}"
        )

    summary = load_json(
        summary_path
    )

    manifest = load_json(
        manifest_path
    )

    if summary.get(
        "condition_name"
    ) != condition_name:
        raise RuntimeError(
            f"{summary_path}: condition_name does not equal "
            f"{condition_name!r}"
        )

    if manifest.get(
        "condition_name"
    ) != condition_name:
        raise RuntimeError(
            f"{manifest_path}: condition_name does not equal "
            f"{condition_name!r}"
        )

    if summary.get(
        "phase"
    ) != EXPECTED_PHASE:
        raise RuntimeError(
            f"{summary_path}: phase={summary.get('phase')!r}; "
            f"expected {EXPECTED_PHASE!r}"
        )

    if manifest.get(
        "phase"
    ) != EXPECTED_PHASE:
        raise RuntimeError(
            f"{manifest_path}: phase={manifest.get('phase')!r}; "
            f"expected {EXPECTED_PHASE!r}"
        )

    if summary.get(
        "method"
    ) != EXPECTED_METHOD:
        raise RuntimeError(
            f"{summary_path}: method={summary.get('method')!r}; "
            f"expected {EXPECTED_METHOD!r}"
        )

    operator = summary.get(
        "operator"
    )

    if not isinstance(
        operator,
        dict,
    ):
        raise RuntimeError(
            f"{summary_path}: missing operator object"
        )

    try:
        actual_state_bits = int(
            operator[
                "state_bits"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"{summary_path}: operator.state_bits is missing or invalid"
        ) from exc

    if actual_state_bits != int(
        expected_state_bits
    ):
        raise RuntimeError(
            f"{summary_path}: operator.state_bits={actual_state_bits}; "
            f"expected {expected_state_bits}"
        )

    return {
        "condition_name": condition_name,
        "condition_dir": str(
            condition_dir.resolve()
        ),
        "expected_state_bits": int(
            expected_state_bits
        ),
        "complete_flag": str(
            complete_flag.resolve()
        ),
        "complete_flag_sha256": sha256_file(
            complete_flag
        ),
        "summary": str(
            summary_path.resolve()
        ),
        "summary_sha256": sha256_file(
            summary_path
        ),
        "manifest": str(
            manifest_path.resolve()
        ),
        "manifest_sha256": sha256_file(
            manifest_path
        ),
        "per_sequence": str(
            per_sequence_path.resolve()
        ),
        "per_sequence_sha256": sha256_file(
            per_sequence_path
        ),
    }


def validate_occupancy_candidate(
    candidate: Candidate,
    state_bits: int,
) -> None:
    if candidate.values.shape != (
        EXPECTED_UNITS,
    ):
        raise ValueError(
            f"{candidate.locator}: expected {EXPECTED_UNITS} values, "
            f"got shape {candidate.values.shape}"
        )

    if candidate.unit_indices_zero_based.shape != (
        EXPECTED_UNITS,
    ):
        raise ValueError(
            f"{candidate.locator}: invalid unit-index shape"
        )

    if not np.array_equal(
        candidate.unit_indices_zero_based,
        np.arange(
            EXPECTED_UNITS,
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            f"{candidate.locator}: units are not aligned as 0..31"
        )

    if not np.all(
        np.isfinite(
            candidate.values
        )
    ):
        raise ValueError(
            f"{candidate.locator}: non-finite occupied-level count"
        )

    rounded = np.rint(
        candidate.values
    )

    if not np.allclose(
        candidate.values,
        rounded,
        rtol=0.0,
        atol=INTEGER_TOLERANCE,
    ):
        raise ValueError(
            f"{candidate.locator}: occupied-level counts are not integers"
        )

    if np.any(
        rounded
        < 1
    ):
        raise ValueError(
            f"{candidate.locator}: occupied-level count below 1"
        )

    maximum_possible = 2 ** int(
        state_bits
    )

    if np.any(
        rounded
        > maximum_possible
    ):
        raise ValueError(
            f"{candidate.locator}: occupied-level count exceeds "
            f"2^{state_bits}={maximum_possible}"
        )


def candidate_vector_key(
    candidate: Candidate,
) -> Tuple[int, ...]:
    rounded = np.rint(
        candidate.values
    ).astype(
        np.int64
    )

    return tuple(
        int(
            value
        )
        for value in rounded.tolist()
    )


def source_priority(
    candidate: Candidate,
) -> Tuple[int, int, str, str]:
    kind_priority = {
        "csv_direct_per_unit": 0,
        "json_record_table": 1,
        "json_unit_mapping": 2,
        "json_numeric_vector": 3,
        "npz_numeric_vector": 4,
        "npy_numeric_vector": 5,
    }

    return (
        0
        if candidate.explicit_unit_ids
        else 1,
        kind_priority.get(
            candidate.source_kind,
            99,
        ),
        str(
            candidate.source_path
        ),
        candidate.locator,
    )


def resolve_candidate(
    candidates: Sequence[Candidate],
    expected_median: float,
    state_bits: int,
    condition_name: str,
) -> Tuple[
    Candidate,
    List[Candidate],
    List[Dict[str, Any]],
]:
    accepted: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []

    for candidate in candidates:
        try:
            validate_occupancy_candidate(
                candidate=candidate,
                state_bits=state_bits,
            )

            median_value = float(
                np.median(
                    candidate.values
                )
            )

            if not math.isclose(
                median_value,
                float(
                    expected_median
                ),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise ValueError(
                    f"median={median_value} does not match "
                    f"manuscript validation target {expected_median}"
                )

            accepted.append(
                candidate
            )

        except ValueError as exc:
            rejected.append(
                {
                    **candidate.as_dict(),
                    "rejection_reason": str(
                        exc
                    ),
                }
            )

    if not accepted:
        raise RuntimeError(
            f"No unambiguous per-unit decoder occupied-level vector "
            f"for {condition_name} passed validation. "
            "Inspect figure3b_occupancy_discovery.json."
        )

    vectors: Dict[
        Tuple[int, ...],
        List[Candidate],
    ] = {}

    for candidate in accepted:
        key = candidate_vector_key(
            candidate
        )

        vectors.setdefault(
            key,
            [],
        ).append(
            candidate
        )

    if len(
        vectors
    ) != 1:
        details = []

        for vector_key, grouped_candidates in vectors.items():
            details.append(
                {
                    "values": list(
                        vector_key
                    ),
                    "sources": [
                        candidate.as_dict()
                        for candidate in grouped_candidates
                    ],
                }
            )

        raise RuntimeError(
            f"Multiple distinct per-unit vectors for {condition_name} "
            "pass all validation checks. Refusing to choose between them. "
            f"Candidates: {json.dumps(details, indent=2)}"
        )

    equivalent_candidates = next(
        iter(
            vectors.values()
        )
    )

    canonical = sorted(
        equivalent_candidates,
        key=source_priority,
    )[0]

    return (
        canonical,
        sorted(
            equivalent_candidates,
            key=source_priority,
        ),
        rejected,
    )


def collect_inventory(
    condition_name: str,
    condition_dir: Path,
    candidates: Sequence[Candidate],
) -> Dict[str, Any]:
    files: List[
        Dict[str, Any]
    ] = []

    for path in sorted(
        condition_dir.rglob(
            "*"
        )
    ):
        if not path.is_file():
            continue

        entry: Dict[str, Any] = {
            "path": str(
                path.resolve()
            ),
            "relative_path": str(
                path.relative_to(
                    condition_dir
                )
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
            "suffix": path.suffix.lower(),
        }

        if path.suffix.lower() == ".csv":
            entry[
                "csv_header"
            ] = read_csv_header(
                path
            )

        elif path.suffix.lower() == ".npz":
            try:
                with np.load(
                    path,
                    allow_pickle=False,
                ) as archive:
                    entry[
                        "npz_arrays"
                    ] = {
                        key: {
                            "shape": list(
                                np.asarray(
                                    archive[
                                        key
                                    ]
                                ).shape
                            ),
                            "dtype": str(
                                np.asarray(
                                    archive[
                                        key
                                    ]
                                ).dtype
                            ),
                        }
                        for key in archive.files
                    }
            except Exception as exc:
                entry[
                    "npz_inventory_error"
                ] = str(
                    exc
                )

        files.append(
            entry
        )

    return {
        "condition_name": condition_name,
        "condition_dir": str(
            condition_dir.resolve()
        ),
        "files": files,
        "discovered_occupancy_candidates": [
            candidate.as_dict()
            for candidate in candidates
        ],
    }


def write_csv_output(
    path: Path,
    b4_values: np.ndarray,
    b8_values: np.ndarray,
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
        newline="",
    ) as handle:
        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "unit_index_zero_based",
                "unit_index_one_based",
                "native_b4_occupied_levels",
                "b8_state_occupied_levels",
                "occupied_level_increase",
                "occupied_level_ratio",
            ]
        )

        for unit_index in range(
            EXPECTED_UNITS
        ):
            native_value = int(
                round(
                    float(
                        b4_values[
                            unit_index
                        ]
                    )
                )
            )

            b8_value = int(
                round(
                    float(
                        b8_values[
                            unit_index
                        ]
                    )
                )
            )

            increase = (
                b8_value
                - native_value
            )

            ratio = (
                float(
                    b8_value
                )
                / float(
                    native_value
                )
            )

            writer.writerow(
                [
                    unit_index,
                    unit_index
                    + 1,
                    native_value,
                    b8_value,
                    increase,
                    f"{ratio:.12g}",
                ]
            )

    os.replace(
        tmp,
        path,
    )


def main() -> None:
    args = parse_args()

    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    if not project_root.is_dir():
        raise FileNotFoundError(
            f"Project root does not exist: {project_root}"
        )

    eval_script = (
        project_root
        / "eval"
        / "analyze_recurrent_memory.py"
    )

    base_script = (
        project_root
        / "eval"
        / "analyze_writeback.py"
    )

    if not eval_script.is_file():
        raise FileNotFoundError(
            f"Missing analysis script: {eval_script}"
        )

    if not base_script.is_file():
        raise FileNotFoundError(
            f"Missing base analysis script: {base_script}"
        )

    v4_run = (
        project_root
        / "results"
        / V4_RUN_NAME
    )

    if not v4_run.is_dir():
        raise FileNotFoundError(
            f"Vanilla B4 production run does not exist: {v4_run}"
        )

    recurrent_memory_root = (
        v4_run
        / "recurrent_memory_analysis"
    )

    condition_b4_dir = (
        recurrent_memory_root
        / CONDITION_B4
    )

    condition_b8_dir = (
        recurrent_memory_root
        / CONDITION_B8
    )

    if args.output_dir is None:
        output_dir = (
            project_root
            / "results"
            / "figure3_inputs"
        )
    else:
        output_dir = Path(
            args.output_dir
        ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery_path = (
        output_dir
        / "figure3b_occupancy_discovery.json"
    )

    csv_path = (
        output_dir
        / "figure3b_decoder_occupied_levels.csv"
    )

    npz_path = (
        output_dir
        / "figure3b_decoder_occupied_levels.npz"
    )

    mat_path = (
        output_dir
        / "figure3b_decoder_occupied_levels.mat"
    )

    manifest_path = (
        output_dir
        / "figure3b_decoder_occupied_levels_manifest.json"
    )

    print(
        "[FIGURE 3B] validating completed recurrent-memory conditions",
        flush=True,
    )

    b4_provenance = validate_condition_products(
        condition_dir=condition_b4_dir,
        condition_name=CONDITION_B4,
        expected_state_bits=4,
    )

    b8_provenance = validate_condition_products(
        condition_dir=condition_b8_dir,
        condition_name=CONDITION_B8,
        expected_state_bits=8,
    )

    print(
        "[FIGURE 3B] discovering per-unit decoder occupancy outputs",
        flush=True,
    )

    b4_candidates = discover_candidates(
        condition_name=CONDITION_B4,
        condition_dir=condition_b4_dir,
    )

    b8_candidates = discover_candidates(
        condition_name=CONDITION_B8,
        condition_dir=condition_b8_dir,
    )

    discovery_payload = {
        "analysis": (
            "Figure 3b paired decoder occupied-state-level extraction"
        ),
        "project_root": str(
            project_root
        ),
        "analysis_script": str(
            THIS_FILE
        ),
        "analysis_script_sha256": sha256_file(
            THIS_FILE
        ),
        "source_analysis_script": str(
            eval_script
        ),
        "source_analysis_script_sha256": sha256_file(
            eval_script
        ),
        "source_base_analysis_script": str(
            base_script
        ),
        "source_base_analysis_script_sha256": sha256_file(
            base_script
        ),
        CONDITION_B4: collect_inventory(
            condition_name=CONDITION_B4,
            condition_dir=condition_b4_dir,
            candidates=b4_candidates,
        ),
        CONDITION_B8: collect_inventory(
            condition_name=CONDITION_B8,
            condition_dir=condition_b8_dir,
            candidates=b8_candidates,
        ),
    }

    atomic_write_json(
        discovery_path,
        discovery_payload,
    )

    print(
        f"[FIGURE 3B] discovery inventory written: {discovery_path}",
        flush=True,
    )

    b4_candidate, b4_equivalent_sources, b4_rejected = (
        resolve_candidate(
            candidates=b4_candidates,
            expected_median=EXPECTED_MEDIAN_B4,
            state_bits=4,
            condition_name=CONDITION_B4,
        )
    )

    b8_candidate, b8_equivalent_sources, b8_rejected = (
        resolve_candidate(
            candidates=b8_candidates,
            expected_median=EXPECTED_MEDIAN_B8,
            state_bits=8,
            condition_name=CONDITION_B8,
        )
    )

    if not np.array_equal(
        b4_candidate.unit_indices_zero_based,
        b8_candidate.unit_indices_zero_based,
    ):
        raise RuntimeError(
            "B4 and B8 occupied-level vectors do not have identical "
            "decoder-unit indexing"
        )

    unit_indices_zero_based = np.arange(
        EXPECTED_UNITS,
        dtype=np.int32,
    )

    unit_indices_one_based = (
        unit_indices_zero_based
        + 1
    )

    native_b4_occupied_levels = np.rint(
        b4_candidate.values
    ).astype(
        np.int32
    )

    b8_state_occupied_levels = np.rint(
        b8_candidate.values
    ).astype(
        np.int32
    )

    native_b4_median = float(
        np.median(
            native_b4_occupied_levels
        )
    )

    b8_state_median = float(
        np.median(
            b8_state_occupied_levels
        )
    )

    if not math.isclose(
        native_b4_median,
        EXPECTED_MEDIAN_B4,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise RuntimeError(
            "Native-B4 per-unit vector failed final median validation: "
            f"{native_b4_median} != {EXPECTED_MEDIAN_B4}"
        )

    if not math.isclose(
        b8_state_median,
        EXPECTED_MEDIAN_B8,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise RuntimeError(
            "B8-state per-unit vector failed final median validation: "
            f"{b8_state_median} != {EXPECTED_MEDIAN_B8}"
        )

    occupied_level_increase = (
        b8_state_occupied_levels
        - native_b4_occupied_levels
    ).astype(
        np.int32
    )

    occupied_level_ratio = (
        b8_state_occupied_levels.astype(
            np.float64
        )
        / native_b4_occupied_levels.astype(
            np.float64
        )
    )

    write_csv_output(
        path=csv_path,
        b4_values=native_b4_occupied_levels,
        b8_values=b8_state_occupied_levels,
    )

    np.savez_compressed(
        npz_path,
        unit_index_zero_based=unit_indices_zero_based,
        unit_index_one_based=unit_indices_one_based,
        native_b4_occupied_levels=native_b4_occupied_levels,
        b8_state_occupied_levels=b8_state_occupied_levels,
        occupied_level_increase=occupied_level_increase,
        occupied_level_ratio=occupied_level_ratio,
        native_b4_median_occupied_levels=np.asarray(
            native_b4_median,
            dtype=np.float64,
        ),
        b8_state_median_occupied_levels=np.asarray(
            b8_state_median,
            dtype=np.float64,
        ),
    )

    savemat(
        mat_path,
        {
            "unit_index_zero_based": (
                unit_indices_zero_based
            ),
            "unit_index_one_based": (
                unit_indices_one_based
            ),
            "native_b4_occupied_levels": (
                native_b4_occupied_levels
            ),
            "b8_state_occupied_levels": (
                b8_state_occupied_levels
            ),
            "occupied_level_increase": (
                occupied_level_increase
            ),
            "occupied_level_ratio": (
                occupied_level_ratio
            ),
            "native_b4_median_occupied_levels": np.asarray(
                native_b4_median,
                dtype=np.float64,
            ),
            "b8_state_median_occupied_levels": np.asarray(
                b8_state_median,
                dtype=np.float64,
            ),
        },
        do_compression=True,
        long_field_names=True,
        oned_as="column",
    )

    manifest = {
        "analysis": (
            "Figure 3b paired decoder occupied-state-level extraction"
        ),
        "repository": (
            "https://github.com/ismailerbas/Seq2SeqLite-kd"
        ),
        "project_root": str(
            project_root
        ),
        "same_trained_network": True,
        "trained_model": (
            "vanilla 4-bit Seq2SeqLite"
        ),
        "unit_count": EXPECTED_UNITS,
        "unit_alignment": (
            "Decoder hidden units are aligned by canonical model unit "
            "index 0 through 31 in both conditions."
        ),
        "conditions": {
            "native_b4": {
                "condition_name": CONDITION_B4,
                "state_bits": 4,
                "expected_manuscript_median": EXPECTED_MEDIAN_B4,
                "observed_median": native_b4_median,
                "canonical_source": (
                    b4_candidate.as_dict()
                ),
                "equivalent_sources": [
                    candidate.as_dict()
                    for candidate in b4_equivalent_sources
                ],
                "rejected_candidates": (
                    b4_rejected
                ),
                "condition_provenance": (
                    b4_provenance
                ),
            },
            "b8_state": {
                "condition_name": CONDITION_B8,
                "state_bits": 8,
                "expected_manuscript_median": EXPECTED_MEDIAN_B8,
                "observed_median": b8_state_median,
                "canonical_source": (
                    b8_candidate.as_dict()
                ),
                "equivalent_sources": [
                    candidate.as_dict()
                    for candidate in b8_equivalent_sources
                ],
                "rejected_candidates": (
                    b8_rejected
                ),
                "condition_provenance": (
                    b8_provenance
                ),
            },
        },
        "validation": {
            "native_b4_vector_length": int(
                native_b4_occupied_levels.size
            ),
            "b8_state_vector_length": int(
                b8_state_occupied_levels.size
            ),
            "native_b4_all_integer_counts": bool(
                np.all(
                    native_b4_occupied_levels
                    == np.rint(
                        native_b4_occupied_levels
                    )
                )
            ),
            "b8_state_all_integer_counts": bool(
                np.all(
                    b8_state_occupied_levels
                    == np.rint(
                        b8_state_occupied_levels
                    )
                )
            ),
            "identical_unit_indexing": True,
            "native_b4_median_matches_manuscript": True,
            "b8_state_median_matches_manuscript": True,
            "passed": True,
        },
        "outputs": {
            "csv": str(
                csv_path
            ),
            "csv_sha256": sha256_file(
                csv_path
            ),
            "npz": str(
                npz_path
            ),
            "npz_sha256": sha256_file(
                npz_path
            ),
            "mat": str(
                mat_path
            ),
            "mat_sha256": sha256_file(
                mat_path
            ),
            "discovery": str(
                discovery_path
            ),
            "discovery_sha256": sha256_file(
                discovery_path
            ),
        },
        "analysis_script": str(
            THIS_FILE
        ),
        "analysis_script_sha256": sha256_file(
            THIS_FILE
        ),
        "source_analysis_script": str(
            eval_script
        ),
        "source_analysis_script_sha256": sha256_file(
            eval_script
        ),
        "source_base_analysis_script": str(
            base_script
        ),
        "source_base_analysis_script_sha256": sha256_file(
            base_script
        ),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] extraction passed",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] native B4 decoder occupied levels:",
        flush=True,
    )

    print(
        ", ".join(
            str(
                int(
                    value
                )
            )
            for value in native_b4_occupied_levels
        ),
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] B8-state decoder occupied levels:",
        flush=True,
    )

    print(
        ", ".join(
            str(
                int(
                    value
                )
            )
            for value in b8_state_occupied_levels
        ),
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] medians:",
        flush=True,
    )

    print(
        f"  native B4 = {native_b4_median:.6f}",
        flush=True,
    )

    print(
        f"  B8 state  = {b8_state_median:.6f}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] canonical source files:",
        flush=True,
    )

    print(
        f"  native B4 = {b4_candidate.source_path}",
        flush=True,
    )

    print(
        f"  B8 state  = {b8_candidate.source_path}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )

    print(
        "[FIGURE 3B] saved:",
        flush=True,
    )

    print(
        f"  {csv_path}",
        flush=True,
    )

    print(
        f"  {npz_path}",
        flush=True,
    )

    print(
        f"  {mat_path}",
        flush=True,
    )

    print(
        f"  {manifest_path}",
        flush=True,
    )

    print(
        f"  {discovery_path}",
        flush=True,
    )

    print(
        "",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "",
            file=sys.stderr,
            flush=True,
        )

        print(
            "[FIGURE 3B] FAILED",
            file=sys.stderr,
            flush=True,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

        print(
            "",
            file=sys.stderr,
            flush=True,
        )

        raise