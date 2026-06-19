#!/usr/bin/env python3
"""
tables/build_tables.py

Reads test_sdf_metrics.json files produced by:
  - eval/eval_teacher_sdf.py          → Table 1 (Teacher ablation)
  - eval/eval_ptq_sdf.py              → Table 2 (PTQ 16-bit and 8-bit)
  - eval/eval_student_vanilla_sdf.py  → Table 3 (Seq2SeqLite KD)

Prints three LaTeX tables to stdout and saves them to:
  tables/table1_teacher.tex
  tables/table2_ptq.tex
  tables/table3_student.tex

Usage:
    python tables/build_tables.py --results-dir /scratch/nmi/results

Arguments:
    --results-dir   Root directory containing teacher training runs,
                    the ptq/ subdirectory, and vanilla_kd* subdirectories.
                    All three table sources are discovered from this root.
    --out-dir       Directory to write .tex files into.
                    Defaults to the same directory as this script (tables/).
"""

import argparse
import glob
import json
import os
import sys


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Tables 1, 2, 3 from test_sdf_metrics.json files.",
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
        help="Directory to write table1_teacher.tex, table2_ptq.tex, table3_student.tex into.",
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
    scalar metric in Tables 1, 2, and 3.
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
# Relevant folders: any directory under --results-dir that contains
# teacher_best*.weights.h5 (same discovery logic as eval_teacher_sdf.py).
#
# The model label is derived from the layers_teacher list saved in
# test_sdf_metrics.json by eval_teacher_sdf.py.
# Example: layers_teacher=[64, 16] → label "64×16"
#          layers_teacher=[128, 128] → label "128×128"
#
# Table columns (matching the paper):
#   Seq2Seq model size | RMSE | R² Score | L2 norm
#
# Row order matches the paper: 64×64, 64×32, 64×16, 45×45, 32×32, 16×16
# ==============================================================================

MODEL_ORDER_TABLE1 = [
    [64, 64],
    [64, 32],
    [64, 16],
    [45, 45],
    [32, 32],
    [16, 16],
]


def layers_to_label(layers_list):
    """
    Convert a list of ints to the display label used in the paper.
    [64, 16] → "64×16"
    [128, 128] → "128×128"
    """
    return "×".join(str(u) for u in layers_list)


def find_teacher_run_dirs_for_table1(results_dir):
    """
    Walk results_dir recursively and collect every directory that contains
    at least one file matching teacher_best*.weights.h5.

    This is the SAME discovery logic used by eval_teacher_sdf.py so that
    the table builder finds exactly the same runs that were evaluated.
    PTQ subdirectories are excluded because they live under results_dir/ptq/
    and their basenames start with "ptq_"; they are for Table 2.
    vanilla_kd* directories are for Table 3 and are excluded here.
    """
    run_dirs = []
    ptq_root = os.path.join(os.path.normpath(results_dir), "ptq")
    for root, dirs, files in os.walk(results_dir):
        norm_root = os.path.normpath(root)
        if norm_root.startswith(ptq_root):
            continue
        basename = os.path.basename(norm_root)
        if basename.startswith("vanilla_kd"):
            continue
        ckpt_matches = glob.glob(os.path.join(root, "teacher_best*.weights.h5"))
        if ckpt_matches:
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


def build_table1(results_dir):
    """
    Discover all teacher run directories, load their test_sdf_metrics.json,
    and assemble Table 1 rows.

    Returns a list of dicts:
        {
            "label":  str   — e.g. "64×16"
            "rmse":   float
            "r2":     float
            "l2":     float
        }
    Rows are ordered according to MODEL_ORDER_TABLE1.
    Runs whose layers_teacher does not match any expected model are appended at the end.
    """
    run_dirs = find_teacher_run_dirs_for_table1(results_dir)
    print(f"\n[Table 1] Found {len(run_dirs)} teacher run director(y/ies):", flush=True)
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
            print(
                f"  [WARN] 'layers_teacher' missing in {json_path} — skipping.",
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

        rows_by_label[label] = {"label": label, "rmse": rmse, "r2": r2, "l2": l2}
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
# If the run basename does not contain a gruxY or gruXxY pattern, the
# teacher_units/teacher_layers from ptq_args.json is used as fallback.
#
# Table columns:
#   Seq2Seq Models | Type | RMSE | R² Score | L2 norm
#
# Row order: same MODEL_ORDER_TABLE1 for model axis,
# 16-bit rows first then 8-bit rows.
# ==============================================================================

import re as _re


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
                f"  [WARN] PTQ dir has no test_sdf_metrics.json: {full_path}",
                flush=True,
            )
    return run_dirs


def build_table2(results_dir):
    """
    Discover all PTQ run directories, load their test_sdf_metrics.json,
    and assemble Table 2 rows grouped by (model_label, bits).

    Returns a list of dicts:
        {
            "label":  str   — e.g. "64×16"
            "bits":   int   — 16 or 8
            "rmse":   float
            "r2":     float
            "l2":     float
        }
    Rows are ordered: MODEL_ORDER_TABLE1 × [16, 8].
    """
    run_dirs = find_ptq_run_dirs(results_dir)
    print(f"\n[Table 2] Found {len(run_dirs)} PTQ run director(y/ies):", flush=True)
    for d in run_dirs:
        print(f"  {d}", flush=True)

    rows_by_key = {}

    for run_dir in run_dirs:
        dirname  = os.path.basename(run_dir)
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

        rows_by_key[key] = {"label": label, "bits": bits, "rmse": rmse, "r2": r2, "l2": l2}
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
# Table 3: Seq2SeqLite (student) with/without KD
#
# Relevant folders: any directory under --results-dir whose basename starts
# with "vanilla_kd" and that contains student_best.weights.h5 and
# student_args.json — the same discovery logic as eval_student_vanilla_sdf.py.
#
# The model label is derived from student_units in student_args.json.
# Since Seq2SeqLite is a single-GRU-layer architecture, label is just "X"
# (e.g. student_units=32 → "32").
#
# KD vs. no-KD is determined by the presence of "kd" in the directory name
# (dirs with "nokd" or not containing "kd" after the prefix → no-KD).
# Specifically:
#   vanilla_kd_gru32_8bit   → WITH KD
#   vanilla_kd_nokd_gru32_8bit   → WITHOUT KD  (if such dirs exist)
#
# Bit-width is parsed from the directory name (8bit / 16bit).
# If not in the dirname, student_args.json keys bits_kernel / bits_recurrent
# are used: if max(bits_kernel, bits_recurrent) <= 8 → 8-bit, else 16-bit.
#
# Table columns (matching the paper):
#   Seq2SeqLite Models | Type | RMSE | R² Score | L2 norm
#
# Columns in paper:
#   128 (no KD, 16-bit), 32×32 (no KD, 16-bit), 32×32 w/KD (16-bit),
#   16×16 (no KD, 16-bit), 16×16 w/KD (16-bit)
#   same set repeated for 8-bit.
#
# The paper Table 3 column headers are:
#   128  |  32  |  32 w/KD  |  16  |  16 w/KD
# The 128 model is a single-layer 128-unit Seq2SeqLite (the student at
# student_units=128, without KD, trained by vanilla KD training script).
#
# Row order (bits): 16-bit first, then 8-bit.
# Column order (models): 128 (noKD), 32 (noKD), 32 (wKD), 16 (noKD), 16 (wKD)
# ==============================================================================

MODEL_ORDER_TABLE3 = [
    (128, False),
    (32,  False),
    (32,  True),
    (16,  False),
    (16,  True),
]


def parse_student_run_info(run_dir):
    """
    Parse student_units, bits (8 or 16), and has_kd (bool) from a vanilla_kd
    run directory.

    Strategy:
      1. Load student_args.json for student_units, bits_kernel, bits_recurrent.
      2. Parse the directory basename for "8bit" / "16bit" to determine precision.
         If neither is in the name, infer from max(bits_kernel, bits_recurrent):
           <= 8  → 8-bit
           else  → 16-bit
      3. has_kd = True if the directory name contains "nokd" is NOT present and
         "kd" IS present in the non-prefix portion of the name. More precisely:
           - Strip the leading "vanilla_kd" prefix (which is always present).
           - If "nokd" is in the remainder → has_kd = False
           - Else → has_kd = True (vanilla KD training always trains with KD
             by default; nokd runs are explicitly labeled)

    Returns (student_units, bits, has_kd) or raises ValueError on parse failure.
    """
    args_path = os.path.join(run_dir, "student_args.json")
    if not os.path.exists(args_path):
        raise ValueError(f"student_args.json not found in {run_dir}")

    with open(args_path, "r") as f:
        run_args = json.load(f)

    student_units   = int(run_args["student_units"])
    bits_kernel     = int(run_args.get("bits_kernel",     8))
    bits_recurrent  = int(run_args.get("bits_recurrent",  8))

    dirname = os.path.basename(os.path.normpath(run_dir))

    if "16bit" in dirname or "16_bit" in dirname:
        bits = 16
    elif "8bit" in dirname or "8_bit" in dirname:
        bits = 8
    else:
        max_bits = max(bits_kernel, bits_recurrent)
        bits = 8 if max_bits <= 8 else 16

    suffix = dirname[len("vanilla_kd"):]
    has_kd = "nokd" not in suffix.lower()

    return student_units, bits, has_kd


def find_student_run_dirs_for_table3(results_dir):
    """
    Walk results_dir recursively and return every directory that:
    1. Has a basename starting with "vanilla_kd"
    2. Contains student_best.weights.h5
    3. Contains student_args.json
    4. Contains test_sdf_metrics.json

    This is the same discovery logic as eval_student_vanilla_sdf.py plus
    the requirement that test_sdf_metrics.json already exists (eval is done).
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


def build_table3(results_dir):
    """
    Discover all vanilla_kd student run directories, load their
    test_sdf_metrics.json, and assemble Table 3 rows.

    Returns a list of dicts:
        {
            "units":  int   — e.g. 32
            "bits":   int   — 16 or 8
            "has_kd": bool
            "label":  str   — e.g. "32 w/KD" or "32"
            "rmse":   float
            "r2":     float
            "l2":     float
        }
    Rows are ordered: [16-bit, 8-bit] × MODEL_ORDER_TABLE3.
    """
    run_dirs = find_student_run_dirs_for_table3(results_dir)
    print(f"\n[Table 3] Found {len(run_dirs)} vanilla_kd student run director(y/ies):", flush=True)
    for d in run_dirs:
        print(f"  {d}", flush=True)

    rows_by_key = {}

    for run_dir in run_dirs:
        json_path = os.path.join(run_dir, "test_sdf_metrics.json")
        sdf = load_sdf_metrics(json_path)
        if sdf is None:
            continue

        try:
            student_units, bits, has_kd = parse_student_run_info(run_dir)
        except (ValueError, KeyError) as exc:
            print(
                f"  [WARN] Cannot parse run info for {run_dir}: {exc} — skipping.",
                flush=True,
            )
            continue

        rmse, r2, l2 = extract_ch0_metrics(sdf)
        if any(v is None for v in [rmse, r2, l2]):
            print(f"  [WARN] Incomplete metrics in {json_path} — skipping.", flush=True)
            continue

        key = (student_units, bits, has_kd)
        if key in rows_by_key:
            print(
                f"  [WARN] Duplicate key {key} from {run_dir}; keeping first.",
                flush=True,
            )
            continue

        kd_suffix = " w/KD" if has_kd else ""
        label     = f"{student_units}{kd_suffix}"

        rows_by_key[key] = {
            "units":  student_units,
            "bits":   bits,
            "has_kd": has_kd,
            "label":  label,
            "rmse":   rmse,
            "r2":     r2,
            "l2":     l2,
        }
        print(
            f"  [{label} {bits}-bit]  RMSE={rmse:.4f}  R²={r2:.4f}  L2={l2:.4f}",
            flush=True,
        )

    ordered_rows = []
    for bits in [16, 8]:
        for (units, with_kd) in MODEL_ORDER_TABLE3:
            key = (units, bits, with_kd)
            if key in rows_by_key:
                ordered_rows.append(rows_by_key.pop(key))
            else:
                kd_tag = "w/KD" if with_kd else "no KD"
                print(
                    f"  [WARN] Expected student entry {units} {kd_tag} {bits}-bit not found.",
                    flush=True,
                )

    for key in sorted(rows_by_key.keys()):
        ordered_rows.append(rows_by_key[key])
        print(f"  [INFO] Extra student row appended: {key}", flush=True)

    return ordered_rows


# ==============================================================================
# LaTeX table renderers
# ==============================================================================

def fmt(value, decimals=3):
    """Format a float to `decimals` decimal places for LaTeX."""
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def render_table1_latex(rows):
    """
    Render Table 1 as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2Seq model size | RMSE | R² Score | L2 norm
    """
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
    lines.append(r"    Seq2Seq model size & RMSE & R\textsuperscript{2} Score & L2 norm \\")
    lines.append(r"    \midrule")
    for row in rows:
        lines.append(
            f"    {row['label']} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['r2'])} & "
            f"{fmt(row['l2'])} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_table2_latex(rows):
    """
    Render Table 2 as a LaTeX tabular.

    The paper groups rows by bit-width (16-bit block, then 8-bit block)
    with a \midrule between blocks. Within each block rows follow
    MODEL_ORDER_TABLE1.

    Column layout:
      Seq2Seq Models | 16-bit/8-bit Type | RMSE | R² Score | L2 norm

    The "Type" column carries the bit-width label; it is shown as a
    multi-row header spanning all rows in each block, matching the paper.
    """
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
        r"    Seq2Seq Models & Type & RMSE & R\textsuperscript{2} Score & L2 norm \\"
    )
    lines.append(r"    \midrule")

    rows_16 = [r for r in rows if r["bits"] == 16]
    rows_8  = [r for r in rows if r["bits"] == 8]

    for i, row in enumerate(rows_16):
        type_cell = "16-bit" if i == 0 else ""
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['r2'])} & "
            f"{fmt(row['l2'])} \\\\"
        )

    if rows_16 and rows_8:
        lines.append(r"    \midrule")

    for i, row in enumerate(rows_8):
        type_cell = "8-bit" if i == 0 else ""
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['r2'])} & "
            f"{fmt(row['l2'])} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def render_table3_latex(rows):
    """
    Render Table 3 as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2SeqLite Models | Type | RMSE | R² Score | L2 norm

    Row groups: 16-bit block then 8-bit block, each containing the
    MODEL_ORDER_TABLE3 sequence: 128, 32, 32 w/KD, 16, 16 w/KD.
    """
    lines = []
    lines.append(r"\begin{table}[!ht]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{Performance metrics for Seq2SeqLite quantized models "
        r"(16-bit and 8-bit) with and without knowledge distillation (KD) "
        r"on experimental data.}"
    )
    lines.append(r"  \label{tab:student_kd}")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Seq2SeqLite Models & Type & RMSE & R\textsuperscript{2} Score & L2 norm \\"
    )
    lines.append(r"    \midrule")

    rows_16 = [r for r in rows if r["bits"] == 16]
    rows_8  = [r for r in rows if r["bits"] == 8]

    for i, row in enumerate(rows_16):
        type_cell = "16-bit" if i == 0 else ""
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['r2'])} & "
            f"{fmt(row['l2'])} \\\\"
        )

    if rows_16 and rows_8:
        lines.append(r"    \midrule")

    for i, row in enumerate(rows_8):
        type_cell = "8-bit" if i == 0 else ""
        lines.append(
            f"    {row['label']} & {type_cell} & "
            f"{fmt(row['rmse'])} & "
            f"{fmt(row['r2'])} & "
            f"{fmt(row['l2'])} \\\\"
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
    print("build_tables.py — constructing Tables 1, 2, 3", flush=True)
    print(f"  results-dir : {args.results_dir}", flush=True)
    print(f"  out-dir     : {args.out_dir}", flush=True)
    print("=" * 70, flush=True)

    # ── Table 1: Teacher ablation ──────────────────────────────────────────────
    rows1 = build_table1(args.results_dir)
    tex1  = render_table1_latex(rows1)
    path1 = os.path.join(args.out_dir, "table1_teacher.tex")
    with open(path1, "w") as f:
        f.write(tex1 + "\n")
    print(f"\n[Table 1] Saved → {path1}", flush=True)
    print(tex1, flush=True)

    # ── Table 2: PTQ ──────────────────────────────────────────────────────────
    rows2 = build_table2(args.results_dir)
    tex2  = render_table2_latex(rows2)
    path2 = os.path.join(args.out_dir, "table2_ptq.tex")
    with open(path2, "w") as f:
        f.write(tex2 + "\n")
    print(f"\n[Table 2] Saved → {path2}", flush=True)
    print(tex2, flush=True)

    # ── Table 3: Student KD ───────────────────────────────────────────────────
    rows3 = build_table3(args.results_dir)
    tex3  = render_table3_latex(rows3)
    path3 = os.path.join(args.out_dir, "table3_student.tex")
    with open(path3, "w") as f:
        f.write(tex3 + "\n")
    print(f"\n[Table 3] Saved → {path3}", flush=True)
    print(tex3, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Done. All three tables written.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()