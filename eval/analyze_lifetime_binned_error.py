#!/usr/bin/env python3
"""
eval/analyze_lifetime_binned_error.py

Paired held-out lifetime-error analysis for the deadband hypothesis.

The intended primary comparison is the same MemoQ trajectory at:

    P2E identity / continuous recurrent state
    P2F deterministic B4 recurrent writeback

The script never invents lifetime bins and never tunes them using model error.
For tau1 and tau2 independently, held-out samples are sorted by the real
ground-truth lifetime and partitioned into equal-count bins. All comparisons
are paired because the two prediction files must contain exactly the same test
indices in exactly the same order.

For each bin the script reports:

    reference RMSE
    condition RMSE
    condition - reference excess RMSE
    reference MAE
    condition MAE
    condition - reference excess MAE

with paired sequence-level bootstrap confidence intervals for the excess
metrics. It also reports Spearman association between bin lifetime and excess
error as a descriptive test of the slow-trajectory prediction.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

THIS_FILE = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired lifetime-binned excess-error analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=32,
        type=int,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for value, name in (
        (args.reference_npz, "--reference-npz"),
        (args.condition_npz, "--condition-npz"),
        (args.reference_manifest, "--reference-manifest"),
        (args.condition_manifest, "--condition-manifest"),
    ):
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} does not exist: {path}"
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


def require_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    required = (
        "test_idx",
        "gt_tau1",
        "gt_tau2",
        "tau1_pred",
        "tau2_pred",
    )

    with np.load(
        str(path),
        allow_pickle=False,
    ) as payload:
        missing = [
            key
            for key in required
            if key not in payload.files
        ]

        if missing:
            raise RuntimeError(
                f"{path} is missing required arrays: {missing}"
            )

        arrays = {
            key: np.asarray(payload[key])
            for key in required
        }

    n = len(arrays["test_idx"])

    if n <= 0:
        raise RuntimeError(
            f"{path} contains no held-out sequences"
        )

    for key in required:
        array = arrays[key]

        if array.ndim != 1:
            raise RuntimeError(
                f"{path}:{key} must be one-dimensional, got {array.shape}"
            )

        if len(array) != n:
            raise RuntimeError(
                f"{path}:{key} length {len(array)} != test_idx length {n}"
            )

        if key != "test_idx" and not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"{path}:{key} contains non-finite values"
            )

    arrays["test_idx"] = np.asarray(
        arrays["test_idx"],
        dtype=np.int64,
    )

    for key in (
        "gt_tau1",
        "gt_tau2",
        "tau1_pred",
        "tau2_pred",
    ):
        arrays[key] = np.asarray(
            arrays[key],
            dtype=np.float64,
        )

    return arrays


def validate_pair(
    reference: Dict[str, np.ndarray],
    condition: Dict[str, np.ndarray],
    reference_manifest: Dict,
    condition_manifest: Dict,
) -> None:
    if not np.array_equal(
        reference["test_idx"],
        condition["test_idx"],
    ):
        raise RuntimeError(
            "Reference and condition test_idx arrays are not identical"
        )

    for key in (
        "gt_tau1",
        "gt_tau2",
    ):
        if not np.array_equal(
            reference[key],
            condition[key],
        ):
            max_abs = float(
                np.max(
                    np.abs(
                        reference[key]
                        - condition[key]
                    )
                )
            )

            raise RuntimeError(
                f"Reference and condition {key} arrays differ; "
                f"max_abs={max_abs:.9g}"
            )

    ref_test_hash = reference_manifest.get(
        "test_idx_sha256"
    )
    cond_test_hash = condition_manifest.get(
        "test_idx_sha256"
    )

    if (
        not isinstance(ref_test_hash, str)
        or not isinstance(cond_test_hash, str)
        or ref_test_hash != cond_test_hash
    ):
        raise RuntimeError(
            "Reference and condition manifests do not carry the same "
            "test_idx_sha256"
        )

    ref_run = reference_manifest.get(
        "run_dir"
    )
    cond_run = condition_manifest.get(
        "run_dir"
    )

    if ref_run != cond_run:
        raise RuntimeError(
            "Primary lifetime-binned excess-error comparison must use "
            "the same MemoQ run directory"
        )


def rmse(error: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    error,
                    dtype=np.float64,
                )
            )
        )
    )


def mae(error: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(error),
            dtype=np.float64,
        )
    )


def paired_bootstrap_excess(
    gt: np.ndarray,
    reference_pred: np.ndarray,
    condition_pred: np.ndarray,
    reps: int,
    seed: int,
    batch_reps: int,
) -> Dict:
    n = len(gt)

    if n < 10:
        raise RuntimeError(
            f"Bootstrap bin contains only {n} samples"
        )

    rng = np.random.default_rng(seed)

    excess_rmse = np.empty(
        reps,
        dtype=np.float64,
    )
    excess_mae = np.empty(
        reps,
        dtype=np.float64,
    )

    ref_error = (
        reference_pred - gt
    ).astype(np.float64)
    cond_error = (
        condition_pred - gt
    ).astype(np.float64)

    done = 0

    while done < reps:
        take = min(
            batch_reps,
            reps - done,
        )

        index = rng.integers(
            0,
            n,
            size=(take, n),
        )

        ref_sample = ref_error[index]
        cond_sample = cond_error[index]

        ref_rmse = np.sqrt(
            np.mean(
                ref_sample * ref_sample,
                axis=1,
            )
        )
        cond_rmse = np.sqrt(
            np.mean(
                cond_sample * cond_sample,
                axis=1,
            )
        )

        ref_mae = np.mean(
            np.abs(ref_sample),
            axis=1,
        )
        cond_mae = np.mean(
            np.abs(cond_sample),
            axis=1,
        )

        excess_rmse[done : done + take] = (
            cond_rmse - ref_rmse
        )
        excess_mae[done : done + take] = (
            cond_mae - ref_mae
        )

        done += take

    return {
        "excess_rmse_ci95_low": float(
            np.percentile(excess_rmse, 2.5)
        ),
        "excess_rmse_ci95_high": float(
            np.percentile(excess_rmse, 97.5)
        ),
        "excess_mae_ci95_low": float(
            np.percentile(excess_mae, 2.5)
        ),
        "excess_mae_ci95_high": float(
            np.percentile(excess_mae, 97.5)
        ),
    }


def equal_count_bins(
    ground_truth: np.ndarray,
    n_bins: int,
) -> List[np.ndarray]:
    n = len(ground_truth)

    if n < n_bins:
        raise RuntimeError(
            f"Cannot split {n} samples into {n_bins} non-empty bins"
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
        for chunk in np.array_split(
            order,
            n_bins,
        )
    ]

    if any(len(chunk) == 0 for chunk in bins):
        raise RuntimeError(
            "Equal-count lifetime binning produced an empty bin"
        )

    flattened = np.concatenate(bins)

    if not np.array_equal(
        np.sort(flattened),
        np.arange(
            n,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Lifetime binning did not partition every sample exactly once"
        )

    return bins


def analyze_target(
    target_name: str,
    gt: np.ndarray,
    reference_pred: np.ndarray,
    condition_pred: np.ndarray,
    n_bins: int,
    bootstrap_reps: int,
    bootstrap_seed: int,
    bootstrap_batch_reps: int,
) -> Tuple[List[Dict], Dict]:
    bins = equal_count_bins(
        gt,
        n_bins,
    )

    rows = []

    for bin_index, index in enumerate(bins):
        gt_bin = gt[index]
        ref_bin = reference_pred[index]
        cond_bin = condition_pred[index]

        ref_error = ref_bin - gt_bin
        cond_error = cond_bin - gt_bin

        ref_rmse = rmse(ref_error)
        cond_rmse = rmse(cond_error)
        ref_mae = mae(ref_error)
        cond_mae = mae(cond_error)

        bootstrap = paired_bootstrap_excess(
            gt_bin,
            ref_bin,
            cond_bin,
            bootstrap_reps,
            bootstrap_seed + bin_index,
            bootstrap_batch_reps,
        )

        rows.append(
            {
                "target": target_name,
                "bin": bin_index + 1,
                "n": int(len(index)),
                "gt_min": float(np.min(gt_bin)),
                "gt_max": float(np.max(gt_bin)),
                "gt_mean": float(np.mean(gt_bin)),
                "gt_median": float(np.median(gt_bin)),
                "reference_rmse": ref_rmse,
                "condition_rmse": cond_rmse,
                "excess_rmse": cond_rmse - ref_rmse,
                "excess_rmse_ci95_low": bootstrap[
                    "excess_rmse_ci95_low"
                ],
                "excess_rmse_ci95_high": bootstrap[
                    "excess_rmse_ci95_high"
                ],
                "reference_mae": ref_mae,
                "condition_mae": cond_mae,
                "excess_mae": cond_mae - ref_mae,
                "excess_mae_ci95_low": bootstrap[
                    "excess_mae_ci95_low"
                ],
                "excess_mae_ci95_high": bootstrap[
                    "excess_mae_ci95_high"
                ],
            }
        )

    x = np.asarray(
        [row["gt_median"] for row in rows],
        dtype=np.float64,
    )

    y_rmse = np.asarray(
        [row["excess_rmse"] for row in rows],
        dtype=np.float64,
    )

    y_mae = np.asarray(
        [row["excess_mae"] for row in rows],
        dtype=np.float64,
    )

    rho_rmse, p_rmse = spearmanr(
        x,
        y_rmse,
    )

    rho_mae, p_mae = spearmanr(
        x,
        y_mae,
    )

    summary = {
        "target": target_name,
        "n_bins": n_bins,
        "spearman_gt_median_vs_excess_rmse": {
            "rho": float(rho_rmse),
            "pvalue": float(p_rmse),
        },
        "spearman_gt_median_vs_excess_mae": {
            "rho": float(rho_mae),
            "pvalue": float(p_mae),
        },
        "nondecreasing_adjacent_excess_rmse_steps": int(
            np.count_nonzero(
                np.diff(y_rmse) >= 0.0
            )
        ),
        "total_adjacent_steps": int(
            len(y_rmse) - 1
        ),
    }

    return (
        rows,
        summary,
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise RuntimeError(
            f"Refusing to write empty CSV: {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.time()

    args = parse_args()
    validate_args(args)

    reference_npz = Path(
        args.reference_npz
    ).resolve()
    condition_npz = Path(
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

    reference = require_npz_arrays(
        reference_npz
    )
    condition = require_npz_arrays(
        condition_npz
    )

    reference_manifest = load_json(
        reference_manifest_path
    )
    condition_manifest = load_json(
        condition_manifest_path
    )

    validate_pair(
        reference,
        condition,
        reference_manifest,
        condition_manifest,
    )

    tau1_rows, tau1_summary = analyze_target(
        "tau1",
        reference["gt_tau1"],
        reference["tau1_pred"],
        condition["tau1_pred"],
        args.n_bins,
        args.bootstrap_reps,
        args.bootstrap_seed,
        args.bootstrap_batch_reps,
    )

    tau2_rows, tau2_summary = analyze_target(
        "tau2",
        reference["gt_tau2"],
        reference["tau2_pred"],
        condition["tau2_pred"],
        args.n_bins,
        args.bootstrap_reps,
        args.bootstrap_seed + 1000,
        args.bootstrap_batch_reps,
    )

    rows = tau1_rows + tau2_rows

    write_csv(
        out_dir
        / "lifetime_binned_excess_error.csv",
        rows,
    )

    payload = {
        "reference_name": args.reference_name,
        "condition_name": args.condition_name,
        "n_test": int(
            len(reference["test_idx"])
        ),
        "binning": {
            "method": (
                "ground-truth-sorted equal-count bins, constructed independently "
                "for tau1 and tau2 without using prediction error"
            ),
            "n_bins": args.n_bins,
        },
        "bootstrap": {
            "method": (
                "paired sequence-level resampling within each lifetime bin"
            ),
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "batch_reps": args.bootstrap_batch_reps,
        },
        "tau1": tau1_summary,
        "tau2": tau2_summary,
        "reference_npz": str(reference_npz),
        "condition_npz": str(condition_npz),
        "reference_npz_sha256": sha256_file(reference_npz),
        "condition_npz_sha256": sha256_file(condition_npz),
        "reference_manifest": str(reference_manifest_path),
        "condition_manifest": str(condition_manifest_path),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "condition_manifest_sha256": sha256_file(condition_manifest_path),
        "analysis_script_sha256": sha256_file(THIS_FILE),
        "elapsed_seconds": float(
            time.time() - started
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
            f"{args.reference_name} -> {args.condition_name}\n"
        )

    print(
        "[DONE] lifetime-binned excess-error analysis",
        flush=True,
    )
    print(
        f"[OUT] {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()