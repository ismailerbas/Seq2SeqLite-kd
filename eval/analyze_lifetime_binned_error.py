#!/usr/bin/env python3
"""
eval/analyze_lifetime_binned_error.py

Paired same-checkpoint lifetime-error analysis for the recurrent writeback
study.

The reference and condition prediction files must contain the exact same
held-out test indices and the manifests must point to the exact same checkpoint
SHA-256. This prevents a training-trajectory difference from being mislabeled
as a writeback effect.

For tau1 and tau2 independently, held-out samples are sorted by the real
reported ground-truth lifetime and partitioned into equal-count bins. Binning
never uses model error. Within each bin the script reports paired reference and
condition RMSE/MAE plus condition-minus-reference confidence intervals obtained
by sequence-level paired bootstrap resampling.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

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
    paired_bootstrap_prediction_difference,
    validate_prediction_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired same-checkpoint "
            "lifetime-binned excess-error analysis."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--reference-npz",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--condition-npz",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--reference-manifest",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--condition-manifest",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--reference-name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--condition-name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--n-bins",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-reps",
        default=2000,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-seed",
        default=42,
        type=int,
    )

    parser.add_argument(
        "--bootstrap-batch-reps",
        default=16,
        type=int,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    for (
        value,
        name,
    ) in (
        (
            args.reference_npz,
            "--reference-npz",
        ),
        (
            args.condition_npz,
            "--condition-npz",
        ),
        (
            args.reference_manifest,
            "--reference-manifest",
        ),
        (
            args.condition_manifest,
            "--condition-manifest",
        ),
    ):
        path = Path(
            value
        ).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} does not exist: "
                f"{path}"
            )

    if (
        args.reference_name
        == args.condition_name
    ):
        raise ValueError(
            "Reference and condition names must differ"
        )

    if args.n_bins < 4:
        raise ValueError(
            "--n-bins must be >= 4"
        )

    if args.bootstrap_reps <= 0:
        raise ValueError(
            "--bootstrap-reps must be > 0"
        )

    if args.bootstrap_batch_reps <= 0:
        raise ValueError(
            "--bootstrap-batch-reps must be > 0"
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


def validate_manifests(
    reference_manifest: Dict,
    condition_manifest: Dict,
    reference_name: str,
    condition_name: str,
) -> None:
    if (
        reference_manifest.get(
            "condition_name"
        )
        != reference_name
    ):
        raise RuntimeError(
            "Reference manifest "
            "condition_name mismatch: "
            f"{reference_manifest.get('condition_name')!r} "
            f"!= {reference_name!r}"
        )

    if (
        condition_manifest.get(
            "condition_name"
        )
        != condition_name
    ):
        raise RuntimeError(
            "Condition manifest "
            "condition_name mismatch: "
            f"{condition_manifest.get('condition_name')!r} "
            f"!= {condition_name!r}"
        )

    ref_run = reference_manifest.get(
        "run_dir"
    )

    cond_run = condition_manifest.get(
        "run_dir"
    )

    if (
        not isinstance(
            ref_run,
            str,
        )
        or not isinstance(
            cond_run,
            str,
        )
        or ref_run
        != cond_run
    ):
        raise RuntimeError(
            "Reference and condition must "
            "come from the same run directory"
        )

    ref_checkpoint = (
        reference_manifest.get(
            "checkpoint_sha256"
        )
    )

    cond_checkpoint = (
        condition_manifest.get(
            "checkpoint_sha256"
        )
    )

    if (
        not isinstance(
            ref_checkpoint,
            str,
        )
        or not isinstance(
            cond_checkpoint,
            str,
        )
        or len(
            ref_checkpoint
        )
        != 64
        or len(
            cond_checkpoint
        )
        != 64
        or ref_checkpoint
        != cond_checkpoint
    ):
        raise RuntimeError(
            "Reference and condition must "
            "use the exact same checkpoint SHA-256"
        )

    ref_test = reference_manifest.get(
        "test_idx_sha256"
    )

    cond_test = condition_manifest.get(
        "test_idx_sha256"
    )

    if (
        not isinstance(
            ref_test,
            str,
        )
        or not isinstance(
            cond_test,
            str,
        )
        or len(
            ref_test
        )
        != 64
        or len(
            cond_test
        )
        != 64
        or ref_test
        != cond_test
    ):
        raise RuntimeError(
            "Reference and condition manifests "
            "must carry the same test_idx_sha256"
        )


def equal_count_bins(
    ground_truth: np.ndarray,
    n_bins: int,
) -> List[
    np.ndarray
]:
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float64,
    )

    n = len(
        ground_truth
    )

    if n < n_bins:
        raise RuntimeError(
            f"Cannot split {n} "
            f"samples into {n_bins} "
            "non-empty bins"
        )

    order = np.argsort(
        ground_truth,
        kind="mergesort",
    )

    bins = [
        np.asarray(
            chunk,
            dtype=np.int64,
        )
        for chunk
        in np.array_split(
            order,
            n_bins,
        )
    ]

    if any(
        len(
            chunk
        )
        == 0
        for chunk
        in bins
    ):
        raise RuntimeError(
            "Equal-count lifetime binning "
            "produced an empty bin"
        )

    flattened = np.concatenate(
        bins
    )

    if not np.array_equal(
        np.sort(
            flattened
        ),
        np.arange(
            n,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Lifetime binning did not "
            "partition every sample exactly once"
        )

    return bins


def analyze_target(
    comparison_name: str,
    target_name: str,
    gt: np.ndarray,
    reference_pred: np.ndarray,
    condition_pred: np.ndarray,
    reference_name: str,
    condition_name: str,
    n_bins: int,
    bootstrap_reps: int,
    bootstrap_seed: int,
    bootstrap_batch_reps: int,
) -> Tuple[
    List[
        Dict
    ],
    Dict,
]:
    bins = equal_count_bins(
        gt,
        n_bins,
    )

    rows = []

    for (
        bin_index,
        index,
    ) in enumerate(
        bins
    ):
        gt_bin = gt[
            index
        ]

        reference_bin = (
            reference_pred[
                index
            ]
        )

        condition_bin = (
            condition_pred[
                index
            ]
        )

        stats = (
            paired_bootstrap_prediction_difference(
                gt_bin,
                reference_bin,
                condition_bin,
                bootstrap_reps,
                (
                    bootstrap_seed
                    + bin_index
                ),
                bootstrap_batch_reps,
            )
        )

        rows.append(
            {
                "comparison": (
                    comparison_name
                ),
                "reference_name": (
                    reference_name
                ),
                "condition_name": (
                    condition_name
                ),
                "target": (
                    target_name
                ),
                "bin": (
                    bin_index
                    + 1
                ),
                "n": int(
                    len(
                        index
                    )
                ),
                "gt_min": float(
                    np.min(
                        gt_bin
                    )
                ),
                "gt_max": float(
                    np.max(
                        gt_bin
                    )
                ),
                "gt_mean": float(
                    np.mean(
                        gt_bin
                    )
                ),
                "gt_median": float(
                    np.median(
                        gt_bin
                    )
                ),
                "reference_rmse": (
                    stats[
                        "reference_rmse"
                    ]
                ),
                "condition_rmse": (
                    stats[
                        "alternative_rmse"
                    ]
                ),
                "condition_minus_reference_rmse": (
                    stats[
                        "alternative_minus_reference_rmse"
                    ]
                ),
                "condition_minus_reference_rmse_ci95_low": (
                    stats[
                        "alternative_minus_reference_rmse_ci95_low"
                    ]
                ),
                "condition_minus_reference_rmse_ci95_high": (
                    stats[
                        "alternative_minus_reference_rmse_ci95_high"
                    ]
                ),
                "reference_mae": (
                    stats[
                        "reference_mae"
                    ]
                ),
                "condition_mae": (
                    stats[
                        "alternative_mae"
                    ]
                ),
                "condition_minus_reference_mae": (
                    stats[
                        "alternative_minus_reference_mae"
                    ]
                ),
                "condition_minus_reference_mae_ci95_low": (
                    stats[
                        "alternative_minus_reference_mae_ci95_low"
                    ]
                ),
                "condition_minus_reference_mae_ci95_high": (
                    stats[
                        "alternative_minus_reference_mae_ci95_high"
                    ]
                ),
            }
        )

    x = np.asarray(
        [
            row[
                "gt_median"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    y_rmse = np.asarray(
        [
            row[
                "condition_minus_reference_rmse"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    y_mae = np.asarray(
        [
            row[
                "condition_minus_reference_mae"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    (
        rho_rmse,
        p_rmse,
    ) = spearmanr(
        x,
        y_rmse,
    )

    (
        rho_mae,
        p_mae,
    ) = spearmanr(
        x,
        y_mae,
    )

    summary = {
        "comparison": (
            comparison_name
        ),
        "reference_name": (
            reference_name
        ),
        "condition_name": (
            condition_name
        ),
        "target": (
            target_name
        ),
        "n_bins": (
            n_bins
        ),
        "spearman_gt_median_vs_condition_minus_reference_rmse": {
            "rho": float(
                rho_rmse
            ),
            "pvalue": float(
                p_rmse
            ),
        },
        "spearman_gt_median_vs_condition_minus_reference_mae": {
            "rho": float(
                rho_mae
            ),
            "pvalue": float(
                p_mae
            ),
        },
        "nondecreasing_adjacent_excess_rmse_steps": int(
            np.count_nonzero(
                np.diff(
                    y_rmse
                )
                >= 0.0
            )
        ),
        "total_adjacent_steps": int(
            len(
                y_rmse
            )
            - 1
        ),
    }

    return (
        rows,
        summary,
    )


def write_csv(
    path: Path,
    rows: List[
        Dict
    ],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: "
            f"{path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main() -> None:
    started = time.time()

    args = parse_args()

    validate_args(
        args
    )

    reference_npz_path = Path(
        args.reference_npz
    ).resolve()

    condition_npz_path = Path(
        args.condition_npz
    ).resolve()

    reference_manifest_path = Path(
        args.reference_manifest
    ).resolve()

    condition_manifest_path = Path(
        args.condition_manifest
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = load_prediction_npz(
        reference_npz_path
    )

    condition = load_prediction_npz(
        condition_npz_path
    )

    validate_prediction_pair(
        reference,
        condition,
    )

    reference_manifest = load_json(
        reference_manifest_path
    )

    condition_manifest = load_json(
        condition_manifest_path
    )

    validate_manifests(
        reference_manifest,
        condition_manifest,
        args.reference_name,
        args.condition_name,
    )

    comparison_name = (
        f"{args.condition_name}_vs_"
        f"{args.reference_name}"
    )

    (
        tau1_rows,
        tau1_summary,
    ) = analyze_target(
        comparison_name,
        "tau1",
        reference[
            "gt_tau1"
        ],
        reference[
            "tau1_pred"
        ],
        condition[
            "tau1_pred"
        ],
        args.reference_name,
        args.condition_name,
        args.n_bins,
        args.bootstrap_reps,
        args.bootstrap_seed,
        args.bootstrap_batch_reps,
    )

    (
        tau2_rows,
        tau2_summary,
    ) = analyze_target(
        comparison_name,
        "tau2",
        reference[
            "gt_tau2"
        ],
        reference[
            "tau2_pred"
        ],
        condition[
            "tau2_pred"
        ],
        args.reference_name,
        args.condition_name,
        args.n_bins,
        args.bootstrap_reps,
        (
            args.bootstrap_seed
            + 1000
        ),
        args.bootstrap_batch_reps,
    )

    rows = (
        tau1_rows
        + tau2_rows
    )

    write_csv(
        out_dir
        / "lifetime_binned_excess_error.csv",
        rows,
    )

    payload = {
        "comparison": (
            comparison_name
        ),
        "reference_name": (
            args.reference_name
        ),
        "condition_name": (
            args.condition_name
        ),
        "same_checkpoint_required": True,
        "checkpoint_sha256": (
            reference_manifest[
                "checkpoint_sha256"
            ]
        ),
        "test_idx_sha256": (
            reference_manifest[
                "test_idx_sha256"
            ]
        ),
        "n_test": int(
            len(
                reference[
                    "test_idx"
                ]
            )
        ),
        "binning": {
            "method": (
                "ground-truth-sorted equal-count "
                "bins, constructed independently "
                "for tau1 and tau2 without using "
                "prediction error"
            ),
            "n_bins": (
                args.n_bins
            ),
        },
        "bootstrap": {
            "method": (
                "paired sequence-level resampling "
                "within each lifetime bin"
            ),
            "reps": (
                args.bootstrap_reps
            ),
            "seed": (
                args.bootstrap_seed
            ),
            "batch_reps": (
                args.bootstrap_batch_reps
            ),
        },
        "tau1": (
            tau1_summary
        ),
        "tau2": (
            tau2_summary
        ),
        "reference_npz": str(
            reference_npz_path
        ),
        "condition_npz": str(
            condition_npz_path
        ),
        "reference_npz_sha256": (
            sha256_file(
                reference_npz_path
            )
        ),
        "condition_npz_sha256": (
            sha256_file(
                condition_npz_path
            )
        ),
        "reference_manifest": str(
            reference_manifest_path
        ),
        "condition_manifest": str(
            condition_manifest_path
        ),
        "reference_manifest_sha256": (
            sha256_file(
                reference_manifest_path
            )
        ),
        "condition_manifest_sha256": (
            sha256_file(
                condition_manifest_path
            )
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
        / "lifetime_binned_excess_error.json",
        payload,
    )

    with (
        out_dir
        / "lifetime_binned_excess_error_complete.flag"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{comparison_name}\n"
        )

    print(
        f"[DONE] "
        f"{comparison_name}",
        flush=True,
    )

    print(
        f"[OUT] "
        f"{out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()