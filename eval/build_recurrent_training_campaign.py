#!/usr/bin/env python3
"""Aggregate the pre-registered matched recurrent-memory training campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


CONDITIONS = ("b4", "b6", "r2", "scw_k2")
SEEDS = (42, 43, 44)
EXPECTED_ALPHA = 0.6
EXPECTED_TEMPERATURE = 4.0
EXPECTED_SPLIT_SEED = 42


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate all 12 matched recurrent-memory training runs."
    )
    parser.add_argument("--campaign-root", required=True, type=str)
    parser.add_argument("--training-script", required=True, type=str)
    args = parser.parse_args()
    root = Path(args.campaign_root).resolve()
    training_script = Path(args.training_script).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Campaign root does not exist: {root}")
    if not training_script.is_file():
        raise FileNotFoundError(f"Training script does not exist: {training_script}")
    args.campaign_root = str(root)
    args.training_script = str(training_script)
    return args


def job_dir(root: Path, condition: str, seed: int) -> Path:
    return root / "results" / f"vanilla_memory_{condition}_seed{seed}"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    return path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scalar_stats(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty metric list")
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": [float(value) for value in arr],
    }


def validate_condition_manifest(
    manifest: Dict,
    condition: str,
    seed: int,
    training_script_sha256: str,
) -> None:
    expected = {
        "training_complete": True,
        "condition": condition,
        "init_seed": seed,
        "split_seed": EXPECTED_SPLIT_SEED,
        "alpha": EXPECTED_ALPHA,
        "temperature": EXPECTED_TEMPERATURE,
        "training_script_sha256": training_script_sha256,
        "epoch_shuffle_policy": "split_seed_plus_zero_based_epoch",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Manifest mismatch for {condition} seed {seed}, {key}: "
                f"expected={value!r}, got={manifest.get(key)!r}"
            )
    expected_storage = {
        "b4": (4, 0, 4),
        "b6": (6, 0, 6),
        "r2": (4, 2, 6),
        "scw_k2": (4, 2, 6),
    }[condition]
    actual_storage = (
        manifest.get("recurrence_visible_state_bits"),
        manifest.get("auxiliary_memory_bits"),
        manifest.get("total_stored_bits_per_unit"),
    )
    if actual_storage != expected_storage:
        raise RuntimeError(
            f"Storage mismatch for {condition} seed {seed}: "
            f"expected={expected_storage}, got={actual_storage}"
        )
    if condition == "scw_k2":
        operator = manifest.get("operator", {})
        if operator.get("counter_bits") != 2:
            raise RuntimeError(f"SCW counter_bits mismatch for seed {seed}")
        if operator.get("trigger_votes") != 2:
            raise RuntimeError(f"SCW trigger_votes mismatch for seed {seed}")
        if not np.isclose(
            float(operator.get("deadzone_fraction_of_delta")),
            0.125,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(f"SCW dead-zone mismatch for seed {seed}")
    if condition == "r2":
        operator = manifest.get("operator", {})
        if operator.get("residual_bits") != 2:
            raise RuntimeError(f"R2 residual_bits mismatch for seed {seed}")
        if operator.get("residual_levels") != 4:
            raise RuntimeError(f"R2 residual_levels mismatch for seed {seed}")


def main() -> None:
    args = parse_args()
    root = Path(args.campaign_root)
    output_dir = root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_script = Path(args.training_script)
    training_script_sha256 = sha256_file(training_script)
    campaign_spec_path = require_file(root / "campaign_spec.json")
    campaign_spec = load_json(campaign_spec_path)
    if campaign_spec.get("locked_before_training") is not True:
        raise RuntimeError("campaign_spec.json is not marked locked_before_training")
    if campaign_spec.get("conditions") != list(CONDITIONS):
        raise RuntimeError("campaign_spec.json condition list does not match aggregator")
    if campaign_spec.get("init_seeds") != list(SEEDS):
        raise RuntimeError("campaign_spec.json seed list does not match aggregator")
    if campaign_spec.get("split_seed") != EXPECTED_SPLIT_SEED:
        raise RuntimeError("campaign_spec.json split seed mismatch")
    if campaign_spec.get("alpha") != EXPECTED_ALPHA:
        raise RuntimeError("campaign_spec.json alpha mismatch")
    if campaign_spec.get("temperature") != EXPECTED_TEMPERATURE:
        raise RuntimeError("campaign_spec.json temperature mismatch")
    if campaign_spec.get("epoch_shuffle_policy") != "split_seed_plus_zero_based_epoch":
        raise RuntimeError("campaign_spec.json shuffle policy mismatch")
    if campaign_spec.get("script_hashes", {}).get("training") != training_script_sha256:
        raise RuntimeError("campaign_spec.json training script hash mismatch")

    rows: List[Dict] = []
    manifests: Dict[Tuple[str, int], Dict] = {}
    reference_test_idx = None
    reference_gt_tau1 = None
    reference_gt_tau2 = None
    reference_gt_fret = None
    reference_teacher_sha = None
    reference_teacher_cache_sha = None
    reference_test_index_sha = None
    initial_weights_by_seed: Dict[int, Dict[str, np.ndarray]] = {}

    for condition in CONDITIONS:
        for seed in SEEDS:
            directory = job_dir(root, condition, seed)
            require_file(directory / "campaign_training_complete.flag")
            manifest_path = require_file(directory / "training_manifest.json")
            metrics_path = require_file(directory / "test_metrics.json")
            per_sequence_path = require_file(directory / "test_per_sequence.npz")
            checkpoint_path = require_file(directory / "student_best.weights.h5")
            initial_weights_path = require_file(directory / "initial_weights.npz")
            manifest = load_json(manifest_path)
            metrics = load_json(metrics_path)
            validate_condition_manifest(
                manifest,
                condition,
                seed,
                training_script_sha256,
            )
            if sha256_file(checkpoint_path) != manifest["selected_checkpoint_sha256"]:
                raise RuntimeError(
                    f"Checkpoint hash mismatch for {condition} seed {seed}"
                )
            if sha256_file(initial_weights_path) != manifest["initial_weights_sha256"]:
                raise RuntimeError(
                    f"Initial-weight file hash mismatch for {condition} seed {seed}"
                )
            with np.load(initial_weights_path, allow_pickle=False) as initial_payload:
                current_initial = {
                    key: np.asarray(initial_payload[key], dtype=np.float32)
                    for key in initial_payload.files
                }
            if seed not in initial_weights_by_seed:
                initial_weights_by_seed[seed] = current_initial
            else:
                reference_initial = initial_weights_by_seed[seed]
                if set(current_initial) != set(reference_initial):
                    raise RuntimeError(
                        f"Initial-weight keys differ for {condition} seed {seed}"
                    )
                for key in sorted(reference_initial):
                    if not np.array_equal(current_initial[key], reference_initial[key]):
                        max_abs = float(
                            np.max(np.abs(current_initial[key] - reference_initial[key]))
                        )
                        raise RuntimeError(
                            f"Matched-initialization audit failed for {condition} "
                            f"seed {seed}, {key}: max_abs={max_abs:.9g}"
                        )
            teacher_sha = manifest["teacher_checkpoint_sha256"]
            teacher_cache_sha = manifest["teacher_prediction_cache_sha256"]
            test_index_sha = manifest["test_index_sha256"]
            if reference_teacher_sha is None:
                reference_teacher_sha = teacher_sha
            elif teacher_sha != reference_teacher_sha:
                raise RuntimeError("Teacher checkpoint hash differs across campaign runs")
            if reference_teacher_cache_sha is None:
                reference_teacher_cache_sha = teacher_cache_sha
            elif teacher_cache_sha != reference_teacher_cache_sha:
                raise RuntimeError("Teacher prediction cache hash differs across campaign runs")
            if reference_test_index_sha is None:
                reference_test_index_sha = test_index_sha
            elif test_index_sha != reference_test_index_sha:
                raise RuntimeError("Test-index hash differs across campaign runs")

            with np.load(per_sequence_path, allow_pickle=False) as data:
                test_idx = np.asarray(data["test_idx"], dtype=np.int64)
                gt_tau1 = np.asarray(data["gt_tau1"], dtype=np.float32)
                gt_tau2 = np.asarray(data["gt_tau2"], dtype=np.float32)
                gt_fret = np.asarray(data["gt_fret"], dtype=np.float32)
            if reference_test_idx is None:
                reference_test_idx = test_idx
                reference_gt_tau1 = gt_tau1
                reference_gt_tau2 = gt_tau2
                reference_gt_fret = gt_fret
            else:
                if not np.array_equal(test_idx, reference_test_idx):
                    raise RuntimeError(
                        f"Test-index array mismatch for {condition} seed {seed}"
                    )
                if not np.array_equal(gt_tau1, reference_gt_tau1):
                    raise RuntimeError(
                        f"tau1 ground truth mismatch for {condition} seed {seed}"
                    )
                if not np.array_equal(gt_tau2, reference_gt_tau2):
                    raise RuntimeError(
                        f"tau2 ground truth mismatch for {condition} seed {seed}"
                    )
                if not np.array_equal(gt_fret, reference_gt_fret):
                    raise RuntimeError(
                        f"FRET ground truth mismatch for {condition} seed {seed}"
                    )

            row = {
                "condition": condition,
                "init_seed": seed,
                "visible_state_bits": manifest["recurrence_visible_state_bits"],
                "auxiliary_bits": manifest["auxiliary_memory_bits"],
                "total_stored_bits_per_unit": manifest["total_stored_bits_per_unit"],
                "best_validation_loss": float(manifest["best_validation_loss"]),
                "mae_seq": float(metrics["mae_seq"]),
                "tau1_rmse": float(metrics["tau1"]["rmse"]),
                "tau1_r": float(metrics["tau1"]["r"]),
                "tau2_rmse": float(metrics["tau2"]["rmse"]),
                "tau2_r": float(metrics["tau2"]["r"]),
                "fret_rmse": float(metrics["fret"]["rmse"]),
                "fret_r": float(metrics["fret"]["r"]),
                "job_dir": str(directory),
                "checkpoint_sha256": manifest["selected_checkpoint_sha256"],
            }
            rows.append(row)
            manifests[(condition, seed)] = manifest

    csv_path = output_dir / "trained_memory_seed_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_condition: Dict[str, Dict] = {}
    metric_names = (
        "best_validation_loss",
        "mae_seq",
        "tau1_rmse",
        "tau1_r",
        "tau2_rmse",
        "tau2_r",
        "fret_rmse",
        "fret_r",
    )
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            metric: scalar_stats([float(row[metric]) for row in condition_rows])
            for metric in metric_names
        }
        by_condition[condition]["storage"] = {
            "recurrence_visible_state_bits": int(
                condition_rows[0]["visible_state_bits"]
            ),
            "auxiliary_memory_bits": int(condition_rows[0]["auxiliary_bits"]),
            "total_stored_bits_per_unit": int(
                condition_rows[0]["total_stored_bits_per_unit"]
            ),
        }

    paired_comparisons: Dict[str, Dict] = {}
    comparisons = (
        ("r2_minus_b6", "r2", "b6"),
        ("scw_k2_minus_b6", "scw_k2", "b6"),
        ("r2_minus_b4", "r2", "b4"),
        ("scw_k2_minus_b4", "scw_k2", "b4"),
        ("scw_k2_minus_r2", "scw_k2", "r2"),
        ("b6_minus_b4", "b6", "b4"),
    )
    comparison_metrics = (
        "best_validation_loss",
        "mae_seq",
        "tau1_rmse",
        "tau2_rmse",
        "fret_rmse",
    )
    row_map = {(row["condition"], row["init_seed"]): row for row in rows}
    for label, lhs, rhs in comparisons:
        payload = {
            "lhs": lhs,
            "rhs": rhs,
            "definition": "lhs minus rhs; negative error differences favor lhs",
            "metrics": {},
        }
        for metric in comparison_metrics:
            differences = [
                float(row_map[(lhs, seed)][metric])
                - float(row_map[(rhs, seed)][metric])
                for seed in SEEDS
            ]
            payload["metrics"][metric] = scalar_stats(differences)
            payload["metrics"][metric]["by_seed"] = {
                str(seed): float(
                    row_map[(lhs, seed)][metric] - row_map[(rhs, seed)][metric]
                )
                for seed in SEEDS
            }
        paired_comparisons[label] = payload

    summary = {
        "passed": True,
        "campaign_root": str(root),
        "campaign_spec": str(campaign_spec_path),
        "campaign_spec_sha256": sha256_file(campaign_spec_path),
        "training_script": str(training_script),
        "training_script_sha256": training_script_sha256,
        "conditions": list(CONDITIONS),
        "init_seeds": list(SEEDS),
        "n_training_runs": len(rows),
        "matched_initialization_audit": {
            "passed": True,
            "definition": "Raw trainable initialization arrays are exactly equal across all four conditions within each init seed.",
            "seeds": list(SEEDS),
        },
        "alpha": EXPECTED_ALPHA,
        "temperature": EXPECTED_TEMPERATURE,
        "split_seed": EXPECTED_SPLIT_SEED,
        "epoch_shuffle_policy": "split_seed_plus_zero_based_epoch",
        "teacher_checkpoint_sha256": reference_teacher_sha,
        "teacher_prediction_cache_sha256": reference_teacher_cache_sha,
        "test_index_sha256": reference_test_index_sha,
        "n_test": int(len(reference_test_idx)),
        "condition_summaries": by_condition,
        "paired_seed_comparisons": paired_comparisons,
        "seed_metrics_csv": str(csv_path),
        "seed_metrics_csv_sha256": sha256_file(csv_path),
    }
    summary_path = output_dir / "trained_memory_campaign_summary.json"
    atomic_write_json(summary_path, summary)
    (output_dir / "trained_memory_campaign_complete.flag").write_text(
        "passed\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
