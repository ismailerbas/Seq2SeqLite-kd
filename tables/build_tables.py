#!/usr/bin/env python3
"""
tables/build_tables.py

Reads test_sdf_metrics.json files produced by:
  - eval/eval_teacher_sdf.py          → Table 1 (Teacher ablation)
  - eval/eval_ptq_sdf.py              → Table 2 (PTQ 16-bit and 8-bit)
  - eval/eval_student_vanilla_sdf.py  → Table 3 (Seq2SeqLite KD, uniform quantization)
                                      → Table 4 (Seq2SeqLite KD, hybrid quantization)

Prints four LaTeX tables to stdout and saves them to:
  tables/table1_teacher.tex
  tables/table2_ptq.tex
  tables/table3_student.tex
  tables/table4_student_hybrid.tex

Only vanilla_kd_* directories are processed for Tables 3 and 4.

Usage:
    python tables/build_tables.py --results-dir /scratch/nmi/results

Arguments:
    --results-dir   Root directory containing teacher training runs,
                    the ptq/ subdirectory, and vanilla_kd* subdirectories.
                    All four table sources are discovered from this root.
    --out-dir       Directory to write .tex files into.
                    Defaults to the same directory as this script (tables/).
"""

import argparse
import glob
import json
import os
import re as _re
import sys


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Tables 1, 2, 3, 4 from test_sdf_metrics.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Root directory containing teacher training runs, "
            "ptq/ subdirectory, and vanilla_kd* subdirectories."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write table1_teacher.tex, table2_ptq.tex, table3_student.tex, table4_student_hybrid.tex into.",
    )
    return p.parse_args()


# ==============================================================================
# JSON loading helpers
# ==============================================================================

def load_sdf_metrics(json_path):
    """
    Load test_sdf_metrics.json and return the dict.
    Returns None and prints a warning if the file does not exist or is malformed.
    """
    if not os.path.exists(json_path):
        print(f"  [WARN] Missing: {json_path}", flush=True)
        return None
    with open(json_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON decode error in {json_path}: {exc}", flush=True)
            return None


def extract_ch0_metrics(sdf_metrics):
    """
    Extract RMSE, R2, L2-norm from the ch0_full channel of test_sdf_metrics.json.
    Returns (rmse, r2, l2) as floats, or (None, None, None) if keys are missing.

    The paper tables show one row per model with three metrics.
    ch0_full is the composite (full SDF) channel used as the representative
    scalar metric in Tables 1, 2, 3, and 4.
    """
    ch = sdf_metrics.get("ch0_full")
    if ch is None:
        print(
            f"  [WARN] 'ch0_full' key missing in metrics dict: {list(sdf_metrics.keys())}",
            flush=True,
        )
        return None, None, None
    rmse = ch.get("rmse")
    r2   = ch.get("r2_score")
    l2   = ch.get("l2_norm")
    return rmse, r2, l2


# ==============================================================================
# Table 1: Teacher ablation
#
# Relevant folders: any directory under --results-dir whose basename starts with
# "teacher_training_" and contains teacher_best*.weights.h5.
#
# Row order matches the paper:
#   64×64, 64×16, 45×45, 32×32, 16×16  (compressed models)
#   128×128  (baseline — printed last, labeled "128×128 (baseline)")
#
# Best value per metric is bolded among the non-baseline rows only
# (128×128 is the teacher baseline — bolding it is trivial and uninformative).
# The second-best (i.e. best among non-baseline) gets the bold.
#
# Arrows in column headers: RMSE ↓, R² ↑, L2 ↓
# ==============================================================================

MODEL_ORDER_TABLE1 = [
    [64, 64],
    [64, 16],
    [45, 45],
    [32, 32],
    [16, 16],
    [128, 128],   # baseline — always last
]

BASELINE_LABEL_TABLE1 = "128×128"


def layers_to_label(layers_list):
    """
    Convert a list of ints to the display label used in the paper.
    [64, 16] → "64×16"
    [128, 128] → "128×128"
    """
    return "×".join(str(u) for u in layers_list)


def find_teacher_run_dirs_for_table1(results_dir):
    """
    Walk results_dir (non-recursively at top level) and collect every directory
    whose basename starts with "teacher_training_" and contains at least one
    file matching teacher_best*.weights.h5.

    PTQ and vanilla_kd subdirectories are excluded.
    """
    run_dirs = []
    try:
        entries = os.listdir(results_dir)
    except OSError as exc:
        print(f"  [ERROR] Cannot list {results_dir}: {exc}", flush=True)
        return run_dirs

    for name in sorted(entries):
        full_path = os.path.join(results_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith("teacher_training_"):
            continue
        ckpt_matches = glob.glob(os.path.join(full_path, "teacher_best*.weights.h5"))
        if ckpt_matches:
            run_dirs.append(full_path)
        else:
            print(
                f"  [WARN] teacher_training_ dir has no teacher_best*.weights.h5: {full_path}",
                flush=True,
            )
    return run_dirs


def build_table1(results_dir):
    """
    Discover all teacher run directories, load their test_sdf_metrics.json,
    and assemble Table 1 rows.

    Returns a list of dicts:
        {
            "label":       str   — e.g. "64×16"
            "is_baseline": bool  — True only for 128×128
            "rmse":        float
            "r2":          float
            "l2":          float
        }
    Rows are ordered according to MODEL_ORDER_TABLE1 (baseline last).
    Runs whose layers_teacher does not match any expected model are appended at the end.
    """
    run_dirs = find_teacher_run_dirs_for_table1(results_dir)
    print(f"\n[Table 1] Found {len(run_dirs)} teacher_training_* director(y/ies):", flush=True)
    for d in run_dirs:
        print(f"  {d}", flush=True)

    rows_by_label = {}

    for run_dir in run_dirs:
        json_path = os.path.join(run_dir, "test_sdf_metrics.json")
        sdf = load_sdf_metrics(json_path)
        if sdf is None:
            continue

        layers_teacher = sdf.get("layers_teacher")
        if layers_teacher is None:
            # Fall back: parse from directory name e.g. teacher_training_gru64x16
            dirname = os.path.basename(run_dir)
            m = _re.search(r"gru(\d+)x(\d+)", dirname)
            if m:
                layers_teacher = [int(m.group(1)), int(m.group(2))]
                print(
                    f"  [INFO] layers_teacher inferred from dirname '{dirname}': {layers_teacher}",
                    flush=True,
                )
            else:
                m2 = _re.search(r"gru(\d+)", dirname)
                if m2:
                    u = int(m2.group(1))
                    layers_teacher = [u, u]
                    print(
                        f"  [INFO] layers_teacher inferred from dirname '{dirname}': {layers_teacher}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [WARN] 'layers_teacher' missing in {json_path} and cannot parse from dirname — skipping.",
                        flush=True,
                    )
                    continue

        label = layers_to_label(layers_teacher)
        rmse, r2, l2 = extract_ch0_metrics(sdf)
        if any(v is None for v in [rmse, r2, l2]):
            print(f"  [WARN] Incomplete metrics in {json_path} — skipping.", flush=True)
            continue

        if label in rows_by_label:
            print(
                f"  [WARN] Duplicate model label '{label}' from {run_dir}; "
                f"keeping first encountered.",
                flush=True,
            )
            continue

        is_baseline = (label == BASELINE_LABEL_TABLE1)
        rows_by_label[label] = {
            "label":       label,
            "is_baseline": is_baseline,
            "rmse":        rmse,
            "r2":          r2,
            "l2":          l2,
        }
        print(
            f"  [{label}]  RMSE={rmse:.4f}  R²={r2:.4f}  L2={l2:.4f}",
            flush=True,
        )

    ordered_rows = []
    for layers in MODEL_ORDER_TABLE1:
        lbl = layers_to_label(layers)
        if lbl in rows_by_label:
            ordered_rows.append(rows_by_label.pop(lbl))
        else:
            print(f"  [WARN] Expected model '{lbl}' not found in results.", flush=True)

    for extra_label in sorted(rows_by_label.keys()):
        ordered_rows.append(rows_by_label[extra_label])
        print(f"  [INFO] Extra model appended: {extra_label}", flush=True)

    return ordered_rows


# ==============================================================================
# Table 2: PTQ (16-bit and 8-bit)
#
# Relevant folders: everything under <results-dir>/ptq/ that matches
# ptq_{bits}bit_* and contains test_sdf_metrics.json.
#
# The model label is parsed from the directory name:
#   ptq_16bit_teacher_training_gru64x16  → model "64×16", bits 16
#   ptq_8bit_teacher_training_gru64x16   → model "64×16", bits 8
#
# Row order: same MODEL_ORDER_TABLE1 for model axis, 16-bit rows first then 8-bit.
# Baseline 128×128 goes last within each bit-width group.
#
# Bolding: best per metric across the ENTIRE table (both bit-width groups combined).
# ==============================================================================

def parse_model_label_from_ptq_dirname(dirname):
    """
    Extract model label from a PTQ directory name.

    Patterns tried in order:
      1. gruXxY  →  "X×Y"   (e.g. gru64x16 → "64×16")
      2. gruX    →  "X×X"   (e.g. gru128   → "128×128")
      3. Fall back to ptq_args.json inside the directory.

    dirname is just the basename, not the full path.
    Returns (label_str, bits_int) or (None, None) on failure.
    """
    bits_match = _re.search(r"ptq_(\d+)bit_", dirname)
    if not bits_match:
        return None, None
    bits = int(bits_match.group(1))

    gru_match = _re.search(r"gru(\d+)x(\d+)", dirname)
    if gru_match:
        u1 = int(gru_match.group(1))
        u2 = int(gru_match.group(2))
        return f"{u1}×{u2}", bits

    gru_single_match = _re.search(r"gru(\d+)", dirname)
    if gru_single_match:
        u = int(gru_single_match.group(1))
        return f"{u}×{u}", bits

    return None, bits


def find_ptq_run_dirs(results_dir):
    """
    Return all subdirectories of <results_dir>/ptq/ whose basename matches
    ptq_{bits}bit_* and that contain test_sdf_metrics.json.
    """
    ptq_root = os.path.join(results_dir, "ptq")
    if not os.path.isdir(ptq_root):
        print(f"  [WARN] PTQ root not found: {ptq_root}", flush=True)
        return []

    run_dirs = []
    for name in sorted(os.listdir(ptq_root)):
        full_path = os.path.join(ptq_root, name)
        if not os.path.isdir(full_path):
            continue
        if not _re.match(r"ptq_\d+bit_", name):
            continue
        json_path = os.path.join(full_path, "test_sdf_metrics.json")
        if os.path.exists(json_path):
            run_dirs.append(full_path)
        else:
            print(
                f"  [WARN] PTQ dir has no test_sdf_metrics.json (eval not run): {full_path}",
                flush=True,
            )
    return run_dirs


def build_table2(results_dir):
    """
    Discover all PTQ run directories, load their test_sdf_metrics.json,
    and assemble Table 2 rows grouped by (model_label, bits).

    Returns a list of dicts:
        {
            "label":       str   — e.g. "64×16"
            "is_baseline": bool  — True only for 128×128
            "bits":        int   — 16 or 8
            "rmse":        float
            "r2":          float
            "l2":          float
        }
    Rows are ordered: MODEL_ORDER_TABLE1 × [16, 8].
    """
    run_dirs = find_ptq_run_dirs(results_dir)
    print(f"\n[Table 2] Found {len(run_dirs)} PTQ run director(y/ies):", flush=True)
    for d in run_dirs:
        print(f"  {d}", flush=True)

    rows_by_key = {}

    for run_dir in run_dirs:
        dirname   = os.path.basename(run_dir)
        json_path = os.path.join(run_dir, "test_sdf_metrics.json")
        sdf = load_sdf_metrics(json_path)
        if sdf is None:
            continue

        label, bits = parse_model_label_from_ptq_dirname(dirname)

        if label is None:
            ptq_args_path = os.path.join(run_dir, "ptq_args.json")
            if os.path.exists(ptq_args_path):
                with open(ptq_args_path, "r") as f:
                    ptq_args = json.load(f)
                teacher_units  = ptq_args.get("teacher_units",  128)
                teacher_layers = ptq_args.get("teacher_layers", 2)
                label = "×".join([str(teacher_units)] * teacher_layers)
                bits  = ptq_args.get("bits", bits)
                print(
                    f"  [INFO] Label resolved from ptq_args.json: '{label}' bits={bits}",
                    flush=True,
                )
            else:
                print(
                    f"  [WARN] Cannot determine model label for {run_dir} — skipping.",
                    flush=True,
                )
                continue

        if bits is None:
            bits_from_json = sdf.get("bits")
            if bits_from_json is not None:
                bits = int(bits_from_json)
            else:
                print(
                    f"  [WARN] Cannot determine bit-width for {run_dir} — skipping.",
                    flush=True,
                )
                continue

        bits = int(bits)

        rmse, r2, l2 = extract_ch0_metrics(sdf)
        if any(v is None for v in [rmse, r2, l2]):
            print(f"  [WARN] Incomplete metrics in {json_path} — skipping.", flush=True)
            continue

        key = (label, bits)
        if key in rows_by_key:
            print(
                f"  [WARN] Duplicate key {key} from {run_dir}; keeping first.",
                flush=True,
            )
            continue

        is_baseline = (label == BASELINE_LABEL_TABLE1)
        rows_by_key[key] = {
            "label":       label,
            "is_baseline": is_baseline,
            "bits":        bits,
            "rmse":        rmse,
            "r2":          r2,
            "l2":          l2,
        }
        print(
            f"  [{label} {bits}-bit]  RMSE={rmse:.4f}  R²={r2:.4f}  L2={l2:.4f}",
            flush=True,
        )

    ordered_rows = []
    for bits in [16, 8]:
        for layers in MODEL_ORDER_TABLE1:
            lbl = layers_to_label(layers)
            key = (lbl, bits)
            if key in rows_by_key:
                ordered_rows.append(rows_by_key.pop(key))
            else:
                print(
                    f"  [WARN] Expected PTQ entry '{lbl}' {bits}-bit not found.",
                    flush=True,
                )

    for key in sorted(rows_by_key.keys()):
        ordered_rows.append(rows_by_key[key])
        print(f"  [INFO] Extra PTQ row appended: {key}", flush=True)

    return ordered_rows


# ==============================================================================
# Tables 3 & 4: Seq2SeqLite (student) with/without KD
#
# Relevant folders: any directory under --results-dir whose basename starts
# with "vanilla_kd" (ONLY vanilla_kd — no other prefixes processed).
# Must contain student_best.weights.h5, student_args.json,
# and test_sdf_metrics.json.
#
# Key parsing from the directory name (format produced by train_student_vanilla_kd.py):
#
#   vanilla_kd_T{temp}_a{alpha}_b{bb}k{bk}r{br}a{ba}_gru{units}x1_dense3_...
#
#   alpha  : parsed from a{float} field between _T..._ and _b...
#            a0.0 → KD weight = 0 → WITHOUT KD (no_kd)
#            a0.7 (or any non-zero) → WITH KD
#
#   bits_kernel     : parsed from k{bk} in the b{bb}k{bk}r{br}a{ba} segment.
#   bits_recurrent  : parsed from r{br} in the b{bb}k{bk}r{br}a{ba} segment.
#
#   is_hybrid : True when bits_kernel != bits_recurrent
#               False when bits_kernel == bits_recurrent (uniform quantization)
#
#   bits (for uniform rows only): bits_kernel == bits_recurrent
#
#   units  : parsed from gru{N}x1 in the directory name.
#
# Table 3: uniform quantization rows (bits_kernel == bits_recurrent)
# Table 4: hybrid quantization rows  (bits_kernel != bits_recurrent)
#
# When multiple directories map to the same key, keep the one with the lowest RMSE.
#
# Table 3 columns:
#   Seq2SeqLite Models | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓
#
# Table 4 columns:
#   Seq2SeqLite Models | Kernel bits | Recurrent bits | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓
#
# Row groups in Table 3: sorted by bits ascending, then within each group sorted
# by units ascending, no-KD before with-KD.
#
# Row groups in Table 4: sorted by (bits_kernel, bits_recurrent) ascending,
# then units ascending, no-KD before with-KD.
#
# Bolding: best per metric across the ENTIRE respective table.
# ==============================================================================

def parse_student_run_info_from_dirname(dirname):
    """
    Parse (student_units, bits_kernel, bits_recurrent, has_kd) from a vanilla_kd
    directory name.

    Directory name format:
      vanilla_kd_T{temp}_a{alpha}_b{bb}k{bk}r{br}a{ba}_gru{units}x1_dense3_...

    Returns (student_units, bits_kernel, bits_recurrent, has_kd) or raises
    ValueError on parse failure.

    student_units   : int  — parsed from gru{N}x1
    bits_kernel     : int  — parsed from k{bk} in b{bb}k{bk}r{br}a{ba}
    bits_recurrent  : int  — parsed from r{br} in b{bb}k{bk}r{br}a{ba}
    has_kd          : bool — True if alpha > 0 (a{float} where float != 0.0)
    """
    # Parse alpha (determines KD on/off) — matches _a{float}_b pattern
    alpha_match = _re.search(r"_a([\d.]+)_b", dirname)
    if not alpha_match:
        raise ValueError(f"Cannot parse alpha from dirname: {dirname}")
    alpha = float(alpha_match.group(1))
    has_kd = alpha != 0.0

    # Parse bits_kernel and bits_recurrent from b{bb}k{bk}r{br}a{ba}
    bkr_match = _re.search(r"_b\d+k(\d+)r(\d+)a\d+_", dirname)
    if not bkr_match:
        raise ValueError(f"Cannot parse bits_kernel/bits_recurrent from dirname: {dirname}")
    bits_kernel    = int(bkr_match.group(1))
    bits_recurrent = int(bkr_match.group(2))

    # Parse student units from gru{N}x1
    gru_match = _re.search(r"_gru(\d+)x1_", dirname)
    if not gru_match:
        raise ValueError(f"Cannot parse student units from dirname: {dirname}")
    student_units = int(gru_match.group(1))

    return student_units, bits_kernel, bits_recurrent, has_kd


def find_student_run_dirs(results_dir):
    """
    Walk results_dir recursively and return every directory that:
    1. Has a basename starting with "vanilla_kd" (only vanilla_kd — strictly)
    2. Contains student_best.weights.h5
    3. Contains student_args.json
    4. Contains test_sdf_metrics.json

    Non-vanilla_kd directories (e.g. memoq, other prefixes) are explicitly
    excluded by the basename check.
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        basename = os.path.basename(root)
        if (
            basename.startswith("vanilla_kd")
            and "student_best.weights.h5" in files
            and "student_args.json" in files
            and "test_sdf_metrics.json" in files
        ):
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


def build_table3_and_table4(results_dir):
    """
    Discover all vanilla_kd student run directories, load their
    test_sdf_metrics.json, and split rows into:
      - uniform quantization rows (bits_kernel == bits_recurrent) → Table 3
      - hybrid quantization rows  (bits_kernel != bits_recurrent) → Table 4

    When multiple directories map to the same key, keep the one with the lowest RMSE.

    Returns (rows_uniform, rows_hybrid) where each is a list of dicts:

    Uniform dict:
        {
            "units":   int   — e.g. 32
            "bits":    int   — 4, 6, or 8  (== bits_kernel == bits_recurrent)
            "has_kd":  bool
            "label":   str   — e.g. "32 w/KD" or "32"
            "rmse":    float
            "r2":      float
            "l2":      float
        }

    Hybrid dict:
        {
            "units":          int
            "bits_kernel":    int
            "bits_recurrent": int
            "has_kd":         bool
            "label":          str   — e.g. "32 w/KD" or "32"
            "rmse":           float
            "r2":             float
            "l2":             float
        }

    Uniform rows: bits ascending → units ascending → no-KD before with-KD.
    Hybrid rows:  (bits_kernel, bits_recurrent) ascending → units ascending → no-KD before with-KD.
    """
    run_dirs = find_student_run_dirs(results_dir)
    print(f"\n[Tables 3 & 4] Found {len(run_dirs)} vanilla_kd student run director(y/ies):", flush=True)
    for d in run_dirs:
        print(f"  {d}", flush=True)

    uniform_by_key = {}
    hybrid_by_key  = {}

    for run_dir in run_dirs:
        dirname   = os.path.basename(os.path.normpath(run_dir))
        json_path = os.path.join(run_dir, "test_sdf_metrics.json")
        sdf = load_sdf_metrics(json_path)
        if sdf is None:
            continue

        try:
            student_units, bits_kernel, bits_recurrent, has_kd = parse_student_run_info_from_dirname(dirname)
        except ValueError as exc:
            print(
                f"  [WARN] Cannot parse run info for {run_dir}: {exc} — skipping.",
                flush=True,
            )
            continue

        rmse, r2, l2 = extract_ch0_metrics(sdf)
        if any(v is None for v in [rmse, r2, l2]):
            print(f"  [WARN] Incomplete metrics in {json_path} — skipping.", flush=True)
            continue

        is_hybrid = (bits_kernel != bits_recurrent)
        kd_suffix = " w/KD" if has_kd else ""
        label     = f"{student_units}{kd_suffix}"

        if is_hybrid:
            key = (student_units, bits_kernel, bits_recurrent, has_kd)
            kd_tag = "w/KD" if has_kd else "no KD"
            if key in hybrid_by_key:
                existing_rmse = hybrid_by_key[key]["rmse"]
                if rmse < existing_rmse:
                    print(
                        f"  [INFO] Better hybrid run found for {key}: RMSE {existing_rmse:.4f} -> {rmse:.4f}"
                        f"  ({dirname})",
                        flush=True,
                    )
                    hybrid_by_key[key].update({"rmse": rmse, "r2": r2, "l2": l2})
                else:
                    print(
                        f"  [INFO] Keeping existing hybrid run for {key} (RMSE {existing_rmse:.4f} <= {rmse:.4f})",
                        flush=True,
                    )
                continue
            hybrid_by_key[key] = {
                "units":          student_units,
                "bits_kernel":    bits_kernel,
                "bits_recurrent": bits_recurrent,
                "has_kd":         has_kd,
                "label":          label,
                "rmse":           rmse,
                "r2":             r2,
                "l2":             l2,
            }
            print(
                f"  [HYBRID {label} k{bits_kernel}r{bits_recurrent}]  RMSE={rmse:.4f}  R²={r2:.4f}  L2={l2:.4f}",
                flush=True,
            )
        else:
            bits = bits_kernel  # == bits_recurrent
            key  = (student_units, bits, has_kd)
            kd_tag = "w/KD" if has_kd else "no KD"
            if key in uniform_by_key:
                existing_rmse = uniform_by_key[key]["rmse"]
                if rmse < existing_rmse:
                    print(
                        f"  [INFO] Better uniform run found for {key}: RMSE {existing_rmse:.4f} -> {rmse:.4f}"
                        f"  ({dirname})",
                        flush=True,
                    )
                    uniform_by_key[key].update({"rmse": rmse, "r2": r2, "l2": l2})
                else:
                    print(
                        f"  [INFO] Keeping existing uniform run for {key} (RMSE {existing_rmse:.4f} <= {rmse:.4f})",
                        flush=True,
                    )
                continue
            uniform_by_key[key] = {
                "units":  student_units,
                "bits":   bits,
                "has_kd": has_kd,
                "label":  label,
                "rmse":   rmse,
                "r2":     r2,
                "l2":     l2,
            }
            print(
                f"  [UNIFORM {label} {bits}-bit]  RMSE={rmse:.4f}  R²={r2:.4f}  L2={l2:.4f}",
                flush=True,
            )

    # Sort uniform: bits ascending, then units ascending, then no-KD (False) before with-KD (True)
    rows_uniform = sorted(
        uniform_by_key.values(),
        key=lambda r: (r["bits"], r["units"], r["has_kd"]),
    )

    # Sort hybrid: (bits_kernel, bits_recurrent) ascending, then units ascending, then no-KD before with-KD
    rows_hybrid = sorted(
        hybrid_by_key.values(),
        key=lambda r: (r["bits_kernel"], r["bits_recurrent"], r["units"], r["has_kd"]),
    )

    return rows_uniform, rows_hybrid


# ==============================================================================
# LaTeX table renderers
# ==============================================================================

def fmt(value, bold=False, decimals=3):
    """Format a float to `decimals` decimal places for LaTeX.
    If bold=True, wraps the value in \\textbf{}.
    """
    if value is None:
        return "--"
    s = f"{value:.{decimals}f}"
    if bold:
        return r"\textbf{" + s + r"}"
    return s


def find_best_values(rows, exclude_baseline=True):
    """
    Find best RMSE (min), best R2 (max), best L2 (min) across rows.
    If exclude_baseline=True, rows with is_baseline=True are excluded from
    the best-value computation (so we bold the best non-baseline row).
    Returns (best_rmse, best_r2, best_l2).
    """
    candidates = rows
    if exclude_baseline:
        candidates = [r for r in rows if not r.get("is_baseline", False)]

    rmse_vals = [r["rmse"] for r in candidates if r["rmse"] is not None]
    r2_vals   = [r["r2"]   for r in candidates if r["r2"]   is not None]
    l2_vals   = [r["l2"]   for r in candidates if r["l2"]   is not None]

    best_rmse = min(rmse_vals) if rmse_vals else None
    best_r2   = max(r2_vals)   if r2_vals   else None
    best_l2   = min(l2_vals)   if l2_vals   else None

    return best_rmse, best_r2, best_l2


def render_table1_latex(rows):
    """
    Render Table 1 as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2Seq model size | RMSE ↓ | R² Score ↑ | L2 norm ↓

    128×128 is the baseline and is printed last without being bolded.
    The best values among non-baseline rows are bolded.
    """
    best_rmse, best_r2, best_l2 = find_best_values(rows, exclude_baseline=True)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Performance metrics for different Seq2Seq model sizes "
        r"on experimental data.}"
    )
    lines.append(r"  \label{tab:teacher_ablation}")
    lines.append(r"  \begin{tabular}{lccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Seq2Seq model size & RMSE $\downarrow$ & "
        r"R\textsuperscript{2} Score $\uparrow$ & L2 norm $\downarrow$ \\"
    )
    lines.append(r"    \midrule")
    for row in rows:
        is_baseline = row.get("is_baseline", False)
        bold_rmse = (not is_baseline and row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (not is_baseline and row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (not is_baseline and row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
        display_label = row["label"] + (" (baseline)" if is_baseline else "")
        lines.append(
            f"    {display_label} & "
            f"{fmt(row['rmse'], bold=bold_rmse)} & "
            f"{fmt(row['r2'],   bold=bold_r2)} & "
            f"{fmt(row['l2'],   bold=bold_l2)} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_table2_latex(rows):
    """
    Render Table 2 as a LaTeX tabular.

    The paper groups rows by bit-width (16-bit block, then 8-bit block)
    with a \\midrule between blocks. Within each block rows follow
    MODEL_ORDER_TABLE1 (baseline 128×128 last).

    Column layout:
      Seq2Seq Models | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓

    The 128×128 baseline is not bolded.
    Best value per metric among non-baseline rows across the ENTIRE table
    (both bit-width blocks) is bolded.
    """
    best_rmse, best_r2, best_l2 = find_best_values(rows, exclude_baseline=True)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Performance metrics for Seq2Seq models (16-bit and 8-bit) "
        r"across various model sizes on experimental data.}"
    )
    lines.append(r"  \label{tab:ptq}")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Seq2Seq Models & Type & RMSE $\downarrow$ & "
        r"R\textsuperscript{2} Score $\uparrow$ & L2 norm $\downarrow$ \\"
    )
    lines.append(r"    \midrule")

    rows_16 = [r for r in rows if r["bits"] == 16]
    rows_8  = [r for r in rows if r["bits"] == 8]

    for i, row in enumerate(rows_16):
        type_cell = "16-bit" if i == 0 else ""
        is_baseline = row.get("is_baseline", False)
        bold_rmse = (not is_baseline and row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (not is_baseline and row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (not is_baseline and row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'], bold=bold_rmse)} & "
            f"{fmt(row['r2'],   bold=bold_r2)} & "
            f"{fmt(row['l2'],   bold=bold_l2)} \\\\"
        )

    if rows_16 and rows_8:
        lines.append(r"    \midrule")

    for i, row in enumerate(rows_8):
        type_cell = "8-bit" if i == 0 else ""
        is_baseline = row.get("is_baseline", False)
        bold_rmse = (not is_baseline and row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (not is_baseline and row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (not is_baseline and row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'], bold=bold_rmse)} & "
            f"{fmt(row['r2'],   bold=bold_r2)} & "
            f"{fmt(row['l2'],   bold=bold_l2)} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_table3_latex(rows):
    """
    Render Table 3 (uniform quantization) as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2SeqLite Models | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓

    Rows are grouped by bit-width (ascending), within each group ordered by
    units ascending with no-KD before with-KD.

    The first row in each bit-width group gets the bit-width label in the
    Type column; subsequent rows in the same group get an empty Type cell.

    Best value per metric across the ENTIRE Table 3 is bolded.
    """
    best_rmse, best_r2, best_l2 = find_best_values(rows, exclude_baseline=False)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Performance metrics for Seq2SeqLite uniformly quantized models "
        r"(with and without knowledge distillation (KD)) on experimental data.}"
    )
    lines.append(r"  \label{tab:student_kd}")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Seq2SeqLite Models & Type & RMSE $\downarrow$ & "
        r"R\textsuperscript{2} Score $\uparrow$ & L2 norm $\downarrow$ \\"
    )
    lines.append(r"    \midrule")

    # Group rows by bits, maintaining sorted order
    bits_seen   = []
    bits_groups = {}
    for row in rows:
        b = row["bits"]
        if b not in bits_groups:
            bits_groups[b] = []
            bits_seen.append(b)
        bits_groups[b].append(row)

    first_group = True
    for b in bits_seen:
        group = bits_groups[b]
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False
        for i, row in enumerate(group):
            type_cell = f"{b}-bit" if i == 0 else ""
            bold_rmse = (row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
            bold_r2   = (row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
            bold_l2   = (row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
            lines.append(
                f"    {row['label']} & {type_cell} & "
                f"{fmt(row['rmse'], bold=bold_rmse)} & "
                f"{fmt(row['r2'],   bold=bold_r2)} & "
                f"{fmt(row['l2'],   bold=bold_l2)} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_table4_latex(rows):
    """
    Render Table 4 (hybrid quantization) as a LaTeX tabular.

    Column layout:
      Seq2SeqLite Models | Kernel bits | Recurrent bits | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓

    Rows are grouped by (bits_kernel, bits_recurrent) pair (ascending), within
    each group ordered by units ascending with no-KD before with-KD.

    The first row in each (bits_kernel, bits_recurrent) group shows those values
    in their columns; subsequent rows in the same group get empty cells for those columns.

    Best value per metric across the ENTIRE Table 4 is bolded.
    """
    best_rmse, best_r2, best_l2 = find_best_values(rows, exclude_baseline=False)

    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Performance metrics for Seq2SeqLite hybrid quantized models "
        r"(mixed kernel/recurrent bit-widths, with and without knowledge distillation (KD)) "
        r"on experimental data.}"
    )
    lines.append(r"  \label{tab:student_kd_hybrid}")
    lines.append(r"  \begin{tabular}{llllccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Seq2SeqLite Models & Kernel bits & Recurrent bits & Type & RMSE $\downarrow$ & "
        r"R\textsuperscript{2} Score $\uparrow$ & L2 norm $\downarrow$ \\"
    )
    lines.append(r"    \midrule")

    # Group rows by (bits_kernel, bits_recurrent), maintaining sorted order
    pair_seen   = []
    pair_groups = {}
    for row in rows:
        pair = (row["bits_kernel"], row["bits_recurrent"])
        if pair not in pair_groups:
            pair_groups[pair] = []
            pair_seen.append(pair)
        pair_groups[pair].append(row)

    first_group = True
    for pair in pair_seen:
        group = pair_groups[pair]
        if not first_group:
            lines.append(r"    \midrule")
        first_group = False
        bk, br = pair
        for i, row in enumerate(group):
            kernel_cell    = str(bk) if i == 0 else ""
            recurrent_cell = str(br) if i == 0 else ""
            type_cell      = f"{bk}/{br}-bit" if i == 0 else ""
            bold_rmse = (row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
            bold_r2   = (row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
            bold_l2   = (row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
            lines.append(
                f"    {row['label']} & {kernel_cell} & {recurrent_cell} & {type_cell} & "
                f"{fmt(row['rmse'], bold=bold_rmse)} & "
                f"{fmt(row['r2'],   bold=bold_r2)} & "
                f"{fmt(row['l2'],   bold=bold_l2)} \\\\"
            )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("build_tables.py — constructing Tables 1, 2, 3, 4", flush=True)
    print(f"  results-dir : {args.results_dir}", flush=True)
    print(f"  out-dir     : {args.out_dir}", flush=True)
    print("=" * 70, flush=True)

    # ── Table 1: Teacher ablation ──────────────────────────────────────────────
    rows1 = build_table1(args.results_dir)
    tex1  = render_table1_latex(rows1)
    path1 = os.path.join(args.out_dir, "table1_teacher.tex")
    with open(path1, "w") as f:
        f.write(tex1 + "\n")
    print(f"\n[Table 1] Saved -> {path1}", flush=True)
    print(tex1, flush=True)

    # ── Table 2: PTQ ──────────────────────────────────────────────────────────
    rows2 = build_table2(args.results_dir)
    tex2  = render_table2_latex(rows2)
    path2 = os.path.join(args.out_dir, "table2_ptq.tex")
    with open(path2, "w") as f:
        f.write(tex2 + "\n")
    print(f"\n[Table 2] Saved -> {path2}", flush=True)
    print(tex2, flush=True)

    # ── Tables 3 & 4: Student KD (uniform + hybrid) ───────────────────────────
    rows3_uniform, rows4_hybrid = build_table3_and_table4(args.results_dir)

    tex3  = render_table3_latex(rows3_uniform)
    path3 = os.path.join(args.out_dir, "table3_student.tex")
    with open(path3, "w") as f:
        f.write(tex3 + "\n")
    print(f"\n[Table 3] Saved -> {path3}", flush=True)
    print(tex3, flush=True)

    tex4  = render_table4_latex(rows4_hybrid)
    path4 = os.path.join(args.out_dir, "table4_student_hybrid.tex")
    with open(path4, "w") as f:
        f.write(tex4 + "\n")
    print(f"\n[Table 4] Saved -> {path4}", flush=True)
    print(tex4, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Done. All four tables written.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()