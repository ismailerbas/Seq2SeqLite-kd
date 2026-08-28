#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_DIR_DEFAULT = Path(
    "/gpfs/u/home/HBNN/HBNNrbss/scratch/nmi"
)
JOB_LOG_DIR_DEFAULT = Path(
    "/gpfs/u/scratch/HBNN/HBNNrbss/nmi"
)

CONDITIONS = (
    "b4",
    "b6",
    "r2",
    "scw_k2",
)

SEEDS = (
    42,
    43,
    44,
)

KNOWN_SIGNAL_LINE = (
    "SIGUSR1 received by training worker."
)

KNOWN_EXIT_LINE = (
    "Python exited with code 138"
)

KNOWN_CLASSIFICATION_LINE = (
    "This was not classified as a controlled walltime continuation."
)


class RepairError(RuntimeError):
    pass


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    if not path.is_file():
        raise RepairError(
            f"Required file does not exist: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(
    path: Path,
) -> Dict[str, Any]:
    if not path.is_file():
        raise RepairError(
            f"Required JSON file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(
        payload,
        dict,
    ):
        raise RepairError(
            f"Expected a JSON object in {path}"
        )

    return payload


def atomic_write_json(
    path: Path,
    payload: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
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

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def atomic_write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(text)

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def require_equal(
    actual: Any,
    expected: Any,
    label: str,
) -> None:
    if isinstance(
        expected,
        float,
    ):
        if not isinstance(
            actual,
            (
                int,
                float,
            ),
        ):
            raise RepairError(
                f"{label} mismatch: "
                f"expected={expected!r} "
                f"got={actual!r}"
            )

        if not math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RepairError(
                f"{label} mismatch: "
                f"expected={expected!r} "
                f"got={actual!r}"
            )

        return

    if actual != expected:
        raise RepairError(
            f"{label} mismatch: "
            f"expected={expected!r} "
            f"got={actual!r}"
        )


def job_dir_for(
    campaign_root: Path,
    condition: str,
    seed: int,
) -> Path:
    return (
        campaign_root
        / "results"
        / f"vanilla_memory_{condition}_seed{seed}"
    )


def current_script_hashes(
    project_dir: Path,
) -> Dict[str, str]:
    return {
        "training": sha256_file(
            project_dir
            / "train_student_vanilla_kd_memory_campaign.py"
        ),
        "baseline": sha256_file(
            project_dir
            / "train_student_vanilla_kd.py"
        ),
        "scw": sha256_file(
            project_dir
            / "train_student_vanilla_kd_scw.py"
        ),
        "aggregate": sha256_file(
            project_dir
            / "eval"
            / "build_recurrent_training_campaign.py"
        ),
        "worker": sha256_file(
            project_dir
            / "studentvanilla_memory_train"
        ),
    }

def verify_campaign_configuration(
    spec: Dict[str, Any],
) -> None:
    expected = {
        "locked_before_training": True,
        "conditions": list(CONDITIONS),
        "init_seeds": list(SEEDS),
        "split_seed": 42,
        "epoch_shuffle_policy": (
            "split_seed_plus_zero_based_epoch"
        ),
        "alpha": 0.6,
        "temperature": 4.0,
        "batch_size": 1024,
        "report_all_conditions": True,
    }

    for key, value in expected.items():
        require_equal(
            spec.get(key),
            value,
            f"campaign_spec.{key}",
        )


def migrate_worker_hash_only(
    spec_path: Path,
    spec: Dict[str, Any],
    hashes: Dict[str, str],
) -> Dict[str, Any]:
    locked_hashes = spec.get(
        "script_hashes"
    )

    if not isinstance(
        locked_hashes,
        dict,
    ):
        raise RepairError(
            "campaign_spec.script_hashes "
            "is missing or invalid"
        )

    immutable_keys = (
        "training",
        "baseline",
        "scw",
        "aggregate",
    )

    for key in immutable_keys:
        require_equal(
            locked_hashes.get(
                key
            ),
            hashes[
                key
            ],
            f"locked script hash {key}",
        )

    old_worker_hash = (
        locked_hashes.get(
            "worker"
        )
    )

    new_worker_hash = (
        hashes[
            "worker"
        ]
    )

    if old_worker_hash == new_worker_hash:
        print(
            "[REPAIR] Campaign worker hash already "
            "matches the current worker."
        )

        return spec

    if (
        not isinstance(
            old_worker_hash,
            str,
        )
        or len(
            old_worker_hash
        )
        != 64
    ):
        raise RepairError(
            "Locked worker hash "
            "is missing or malformed"
        )

    migrations = spec.get(
        "operational_migrations",
        [],
    )

    if not isinstance(
        migrations,
        list,
    ):
        raise RepairError(
            "campaign_spec.operational_migrations "
            "must be a list"
        )

    migration = {
        "utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "kind": (
            "recurrent_worker_runtime_recovery"
        ),
        "scientific_training_code_changed": False,
        "old_worker_sha256": (
            old_worker_hash
        ),
        "new_worker_sha256": (
            new_worker_hash
        ),
    }

    updated = dict(
        spec
    )

    updated_hashes = dict(
        locked_hashes
    )

    updated_hashes[
        "worker"
    ] = new_worker_hash

    updated[
        "script_hashes"
    ] = updated_hashes

    updated[
        "operational_migrations"
    ] = [
        *migrations,
        migration,
    ]

    updated[
        "last_operational_update_utc"
    ] = migration[
        "utc"
    ]

    atomic_write_json(
        spec_path,
        updated,
    )

    print(
        "[REPAIR] Updated "
        "campaign_spec.script_hashes.worker."
    )

    print(
        "[REPAIR] Scientific training "
        "source hashes were not changed."
    )

    print(
        "[REPAIR] Old worker SHA256: "
        f"{old_worker_hash}"
    )

    print(
        "[REPAIR] New worker SHA256: "
        f"{new_worker_hash}"
    )

    return updated

def validate_provenance(
    provenance: Dict[str, Any],
    condition: str,
    seed: int,
    hashes: Dict[str, str],
) -> None:
    expected = {
        "condition": condition,
        "init_seed": seed,
        "split_seed": 42,
        "alpha": 0.6,
        "temperature": 4.0,
        "training_script_sha256": hashes[
            "training"
        ],
        "baseline_source_sha256": hashes[
            "baseline"
        ],
        "scw_source_sha256": hashes[
            "scw"
        ],
        "epoch_shuffle_policy": (
            "split_seed_plus_zero_based_epoch"
        ),
    }

    for key, value in expected.items():
        require_equal(
            provenance.get(key),
            value,
            (
                f"provenance {condition} "
                f"seed {seed} {key}"
            ),
        )


def validate_resume_state(
    state: Dict[str, Any],
    condition: str,
    seed: int,
) -> None:
    require_equal(
        state.get(
            "condition"
        ),
        condition,
        (
            f"resume state {condition} "
            f"seed {seed} condition"
        ),
    )

    require_equal(
        state.get(
            "init_seed"
        ),
        seed,
        (
            f"resume state {condition} "
            f"seed {seed} init_seed"
        ),
    )

    require_equal(
        state.get(
            "epoch_shuffle_policy"
        ),
        "split_seed_plus_zero_based_epoch",
        (
            f"resume state {condition} "
            f"seed {seed} epoch_shuffle_policy"
        ),
    )

    epoch = state.get(
        "epoch"
    )

    if (
        not isinstance(
            epoch,
            int,
        )
        or not (
            1 <= epoch <= 300
        )
    ):
        raise RepairError(
            f"Invalid resume epoch for "
            f"{condition} seed {seed}: "
            f"{epoch!r}"
        )

    best_val = state.get(
        "best_val"
    )

    if (
        not isinstance(
            best_val,
            (
                int,
                float,
            ),
        )
        or not math.isfinite(
            float(best_val)
        )
    ):
        raise RepairError(
            f"Invalid best_val for "
            f"{condition} seed {seed}: "
            f"{best_val!r}"
        )


def read_text_lossy(
    path: Path,
) -> str:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        return handle.read()


def matching_known_failure_logs(
    project_dir: Path,
    condition: str,
    seed: int,
) -> List[Path]:
    pattern = (
        f"slurm-*_mem_{condition}_s{seed}.out"
    )

    search_roots = (
        project_dir,
        JOB_LOG_DIR_DEFAULT,
    )

    candidates: List[Path] = []

    seen_paths = set()

    for search_root in search_roots:
        if not search_root.is_dir():
            continue

        for candidate in search_root.glob(
            pattern
        ):
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate

            resolved_key = str(
                resolved
            )

            if resolved_key in seen_paths:
                continue

            seen_paths.add(
                resolved_key
            )

            candidates.append(
                candidate
            )

    matches: List[Path] = []

    for path in candidates:
        if not path.is_file():
            continue

        text = read_text_lossy(
            path
        )

        if (
            KNOWN_SIGNAL_LINE in text
            and KNOWN_EXIT_LINE in text
            and KNOWN_CLASSIFICATION_LINE in text
        ):
            matches.append(
                path
            )

    matches.sort(
        key=lambda item: (
            item.stat().st_mtime
        ),
        reverse=True,
    )

    return matches

def repair_resume_markers(
    project_dir: Path,
    campaign_root: Path,
    hashes: Dict[str, str],
) -> Tuple[
    List[str],
    List[str],
]:
    repaired: List[str] = []

    already_ready: List[str] = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            label = (
                f"{condition} seed {seed}"
            )

            job_dir = job_dir_for(
                campaign_root,
                condition,
                seed,
            )

            complete_flag = (
                job_dir
                / "campaign_training_complete.flag"
            )

            resume_flag = (
                job_dir
                / "pipeline_resume_requested.flag"
            )

            resume_state_path = (
                job_dir
                / "resume_state.json"
            )

            provenance_path = (
                job_dir
                / "run_provenance.json"
            )

            if complete_flag.is_file():
                print(
                    f"[REPAIR] Complete: {label}"
                )

                continue

            if not job_dir.is_dir():
                raise RepairError(
                    "Incomplete run directory is "
                    f"missing for {label}: "
                    f"{job_dir}"
                )

            provenance = load_json(
                provenance_path
            )

            validate_provenance(
                provenance,
                condition,
                seed,
                hashes,
            )

            state = load_json(
                resume_state_path
            )

            validate_resume_state(
                state,
                condition,
                seed,
            )

            if resume_flag.is_file():
                already_ready.append(
                    label
                )

                print(
                    "[REPAIR] Resume marker "
                    f"already present: {label}"
                )

                continue

            logs = matching_known_failure_logs(
                project_dir,
                condition,
                seed,
            )

            if not logs:
                raise RepairError(
                    "Refusing to mark an incomplete "
                    "run for resume because the known "
                    "SIGUSR1/wait-status-138 failure "
                    "signature was not found for "
                    f"{label}. Inspect its SLURM log."
                )

            newest_log = logs[
                0
            ]

            atomic_write_text(
                resume_flag,
                "requested\n",
            )

            repaired.append(
                label
            )

            print(
                "[REPAIR] Restored controlled "
                f"resume marker: {label}"
            )

            print(
                "[REPAIR] Verified failure log: "
                f"{newest_log}"
            )

    return (
        repaired,
        already_ready,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair recurrent-memory campaign "
            "resume metadata after the known "
            "Bash wait/SIGUSR1 status-138 "
            "orchestration bug."
        )
    )

    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR_DEFAULT,
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    project_dir = (
        args.project_dir.resolve()
    )

    campaign_root = (
        project_dir
        / "recurrent_training_campaign"
    )

    spec_path = (
        campaign_root
        / "campaign_spec.json"
    )

    print(
        f"[REPAIR] Project: "
        f"{project_dir}"
    )

    print(
        f"[REPAIR] Campaign: "
        f"{campaign_root}"
    )

    hashes = current_script_hashes(
        project_dir
    )

    spec = load_json(
        spec_path
    )

    verify_campaign_configuration(
        spec
    )

    spec = migrate_worker_hash_only(
        spec_path,
        spec,
        hashes,
    )

    locked_hashes = spec.get(
        "script_hashes"
    )

    if not isinstance(
        locked_hashes,
        dict,
    ):
        raise RepairError(
            "campaign_spec.script_hashes "
            "is missing or invalid"
        )

    required_hash_keys = (
        "training",
        "baseline",
        "scw",
        "aggregate",
        "worker",
    )

    for key in required_hash_keys:
        require_equal(
            locked_hashes.get(
                key
            ),
            hashes[
                key
            ],
            (
                "campaign script hash "
                f"{key} after worker migration"
            ),
        )

    repaired, already_ready = (
        repair_resume_markers(
            project_dir,
            campaign_root,
            hashes,
        )
    )

    print(
        "[REPAIR] Validation passed."
    )

    print(
        "[REPAIR] Restored markers: "
        f"{len(repaired)}"
    )

    print(
        "[REPAIR] Existing valid markers: "
        f"{len(already_ready)}"
    )

    print(
        "[REPAIR] The root-level "
        "recurrent_training_pipeline "
        "can now be submitted in gate mode."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except RepairError as exc:
        print(
            f"FATAL: {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise SystemExit(1)