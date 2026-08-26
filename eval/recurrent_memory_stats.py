#!/usr/bin/env python3
"""
Shared paired-statistics helpers for the recurrent-memory study.

All comparison functions operate on the same held-out sequence indices and
preserve pairing during bootstrap resampling. No training or model execution
occurs in this module.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


PREDICTION_KEYS = (
    "test_idx",
    "gt_tau1",
    "gt_tau2",
    "tau1_pred",
    "tau2_pred",
    "seq_mae_per_sequence",
)


def load_prediction_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction NPZ does not exist: {path}"
        )

    with np.load(
        str(path),
        allow_pickle=False,
    ) as payload:
        missing = [
            key
            for key in PREDICTION_KEYS
            if key not in payload.files
        ]

        if missing:
            raise RuntimeError(
                f"{path} is missing required arrays: {missing}"
            )

        arrays = {
            key: np.asarray(payload[key])
            for key in PREDICTION_KEYS
        }

    n = len(arrays["test_idx"])

    if n <= 0:
        raise RuntimeError(
            f"{path} contains no held-out samples"
        )

    for key, array in arrays.items():
        if array.ndim != 1:
            raise RuntimeError(
                f"{path}:{key} must be one-dimensional, got {array.shape}"
            )

        if len(array) != n:
            raise RuntimeError(
                f"{path}:{key} length {len(array)} != test_idx length {n}"
            )

        if (
            key != "test_idx"
            and not np.all(np.isfinite(array))
        ):
            raise RuntimeError(
                f"{path}:{key} contains non-finite values"
            )

    arrays["test_idx"] = np.asarray(
        arrays["test_idx"],
        dtype=np.int64,
    )

    for key in PREDICTION_KEYS:
        if key == "test_idx":
            continue

        arrays[key] = np.asarray(
            arrays[key],
            dtype=np.float64,
        )

    return arrays


def validate_prediction_pair(
    reference: Dict[str, np.ndarray],
    alternative: Dict[str, np.ndarray],
) -> None:
    if not np.array_equal(
        reference["test_idx"],
        alternative["test_idx"],
    ):
        raise RuntimeError(
            "Paired prediction files do not contain identical test_idx arrays"
        )

    for key in (
        "gt_tau1",
        "gt_tau2",
    ):
        if not np.array_equal(
            reference[key],
            alternative[key],
        ):
            max_abs = float(
                np.max(
                    np.abs(
                        reference[key]
                        - alternative[key]
                    )
                )
            )

            raise RuntimeError(
                f"Paired prediction files disagree on {key}; "
                f"max_abs={max_abs:.9g}"
            )


def rmse_from_error(
    error: np.ndarray,
) -> float:
    error = np.asarray(
        error,
        dtype=np.float64,
    )

    return float(
        np.sqrt(
            np.mean(
                error * error,
                dtype=np.float64,
            )
        )
    )


def mae_from_error(
    error: np.ndarray,
) -> float:
    error = np.asarray(
        error,
        dtype=np.float64,
    )

    return float(
        np.mean(
            np.abs(error),
            dtype=np.float64,
        )
    )


def paired_bootstrap_prediction_difference(
    ground_truth: np.ndarray,
    reference_pred: np.ndarray,
    alternative_pred: np.ndarray,
    reps: int,
    seed: int,
    batch_reps: int,
) -> Dict:
    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float64,
    )

    reference_pred = np.asarray(
        reference_pred,
        dtype=np.float64,
    )

    alternative_pred = np.asarray(
        alternative_pred,
        dtype=np.float64,
    )

    if not (
        ground_truth.ndim
        == reference_pred.ndim
        == alternative_pred.ndim
        == 1
    ):
        raise ValueError(
            "Paired bootstrap prediction inputs must be one-dimensional"
        )

    if not (
        len(ground_truth)
        == len(reference_pred)
        == len(alternative_pred)
    ):
        raise ValueError(
            "Paired bootstrap prediction inputs must have identical lengths"
        )

    if len(ground_truth) < 10:
        raise ValueError(
            "Paired bootstrap requires at least 10 samples"
        )

    if reps <= 0:
        raise ValueError(
            "reps must be > 0"
        )

    if batch_reps <= 0:
        raise ValueError(
            "batch_reps must be > 0"
        )

    if not (
        np.all(np.isfinite(ground_truth))
        and np.all(np.isfinite(reference_pred))
        and np.all(np.isfinite(alternative_pred))
    ):
        raise ValueError(
            "Paired bootstrap prediction inputs contain non-finite values"
        )

    reference_error = (
        reference_pred
        - ground_truth
    )

    alternative_error = (
        alternative_pred
        - ground_truth
    )

    observed_reference_rmse = rmse_from_error(
        reference_error
    )

    observed_alternative_rmse = rmse_from_error(
        alternative_error
    )

    observed_reference_mae = mae_from_error(
        reference_error
    )

    observed_alternative_mae = mae_from_error(
        alternative_error
    )

    n = len(ground_truth)

    rng = np.random.default_rng(
        seed
    )

    rmse_difference = np.empty(
        reps,
        dtype=np.float64,
    )

    mae_difference = np.empty(
        reps,
        dtype=np.float64,
    )

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
            dtype=np.int64,
        )

        reference_sample = (
            reference_error[index]
        )

        alternative_sample = (
            alternative_error[index]
        )

        reference_rmse = np.sqrt(
            np.mean(
                reference_sample
                * reference_sample,
                axis=1,
                dtype=np.float64,
            )
        )

        alternative_rmse = np.sqrt(
            np.mean(
                alternative_sample
                * alternative_sample,
                axis=1,
                dtype=np.float64,
            )
        )

        reference_mae = np.mean(
            np.abs(
                reference_sample
            ),
            axis=1,
            dtype=np.float64,
        )

        alternative_mae = np.mean(
            np.abs(
                alternative_sample
            ),
            axis=1,
            dtype=np.float64,
        )

        rmse_difference[
            done:
            done + take
        ] = (
            alternative_rmse
            - reference_rmse
        )

        mae_difference[
            done:
            done + take
        ] = (
            alternative_mae
            - reference_mae
        )

        done += take

    return {
        "n": int(n),
        "reference_rmse": (
            observed_reference_rmse
        ),
        "alternative_rmse": (
            observed_alternative_rmse
        ),
        "alternative_minus_reference_rmse": (
            observed_alternative_rmse
            - observed_reference_rmse
        ),
        "alternative_minus_reference_rmse_ci95_low": float(
            np.percentile(
                rmse_difference,
                2.5,
            )
        ),
        "alternative_minus_reference_rmse_ci95_high": float(
            np.percentile(
                rmse_difference,
                97.5,
            )
        ),
        "reference_mae": (
            observed_reference_mae
        ),
        "alternative_mae": (
            observed_alternative_mae
        ),
        "alternative_minus_reference_mae": (
            observed_alternative_mae
            - observed_reference_mae
        ),
        "alternative_minus_reference_mae_ci95_low": float(
            np.percentile(
                mae_difference,
                2.5,
            )
        ),
        "alternative_minus_reference_mae_ci95_high": float(
            np.percentile(
                mae_difference,
                97.5,
            )
        ),
        "bootstrap_reps": int(
            reps
        ),
        "bootstrap_seed": int(
            seed
        ),
        "bootstrap_batch_reps": int(
            batch_reps
        ),
    }


def paired_bootstrap_mean_difference(
    reference_values: np.ndarray,
    alternative_values: np.ndarray,
    reps: int,
    seed: int,
    batch_reps: int,
) -> Dict:
    reference_values = np.asarray(
        reference_values,
        dtype=np.float64,
    )

    alternative_values = np.asarray(
        alternative_values,
        dtype=np.float64,
    )

    if not (
        reference_values.ndim
        == alternative_values.ndim
        == 1
    ):
        raise ValueError(
            "Paired bootstrap mean inputs must be one-dimensional"
        )

    if (
        len(reference_values)
        != len(alternative_values)
    ):
        raise ValueError(
            "Paired bootstrap mean inputs must have identical lengths"
        )

    if len(reference_values) < 10:
        raise ValueError(
            "Paired bootstrap requires at least 10 samples"
        )

    if reps <= 0:
        raise ValueError(
            "reps must be > 0"
        )

    if batch_reps <= 0:
        raise ValueError(
            "batch_reps must be > 0"
        )

    if not (
        np.all(
            np.isfinite(
                reference_values
            )
        )
        and np.all(
            np.isfinite(
                alternative_values
            )
        )
    ):
        raise ValueError(
            "Paired bootstrap mean inputs contain non-finite values"
        )

    n = len(
        reference_values
    )

    rng = np.random.default_rng(
        seed
    )

    difference = np.empty(
        reps,
        dtype=np.float64,
    )

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
            dtype=np.int64,
        )

        reference_mean = np.mean(
            reference_values[index],
            axis=1,
            dtype=np.float64,
        )

        alternative_mean = np.mean(
            alternative_values[index],
            axis=1,
            dtype=np.float64,
        )

        difference[
            done:
            done + take
        ] = (
            alternative_mean
            - reference_mean
        )

        done += take

    reference_mean_observed = float(
        np.mean(
            reference_values,
            dtype=np.float64,
        )
    )

    alternative_mean_observed = float(
        np.mean(
            alternative_values,
            dtype=np.float64,
        )
    )

    return {
        "n": int(n),
        "reference_mean": (
            reference_mean_observed
        ),
        "alternative_mean": (
            alternative_mean_observed
        ),
        "alternative_minus_reference_mean": (
            alternative_mean_observed
            - reference_mean_observed
        ),
        "alternative_minus_reference_mean_ci95_low": float(
            np.percentile(
                difference,
                2.5,
            )
        ),
        "alternative_minus_reference_mean_ci95_high": float(
            np.percentile(
                difference,
                97.5,
            )
        ),
        "bootstrap_reps": int(
            reps
        ),
        "bootstrap_seed": int(
            seed
        ),
        "bootstrap_batch_reps": int(
            batch_reps
        ),
    }