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
    python tables/build_tables.py --results-dir /scratch/nmi

Arguments:
    --results-dir   Root directory containing teacher_training_* runs,
                    the ptq/ subdirectory, and results/vanilla_kd* subdirectories.
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
        description="Build Tables 1, 2, 3 from test_sdf_metrics.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Root directory containing teacher_training_* runs, "
            "ptq/ subdirectory, and results/vanilla_kd* subdirectories."
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
# FIXED: only directories whose basename starts with "teacher_training_" are
# collected.  This excludes the root nmi dir, m740bp, teacher_gru128x128 12vials,
# and teklayer_gru128 — which are all non-canonical runs.
#
# Row order matches the paper: 64×64, 64×32, 64×16, 45×45, 32×32, 16×16, 128×128
# 128×128 is the baseline full-size teacher, shown last in the paper.
# ==============================================================================

MODEL_ORDER_TABLE1 = [
    [64, 64],
    [64, 32],
    [64, 16],
    [45, 45],
    [32, 32],
    [16, 16],
    [128, 128],
]


def layers_to_label(layers_list):
    """
    Convert a list of ints to the display label used in the paper.
    [64, 16] -> "64x16"
    [128, 128] -> "128x128"
    Uses literal 'x' for LaTeX (not the Unicode times symbol) so that
    LaTeX renders it cleanly inside tabular without needing math mode.
    """
    return "x".join(str(u) for u in layers_list)


def find_teacher_run_dirs_for_table1(results_dir):
    """
    Walk results_dir recursively and collect every directory whose basename
    starts with "teacher_training_" AND contains at least one file matching
    teacher_best*.weights.h5.

    This is the EXACT same filter used by eval_teacher_sdf.py and
    eval_ptq_sdf.py: find_teacher_run_dirs in both scripts requires
    basename.startswith("teacher_training_").

    Directories such as the root results_dir, m740bp, teklayer_gru128,
    and "teacher_gru128x128 12vials" are intentionally excluded because
    they do not start with "teacher_training_".
    """
    run_dirs = []
    for root, dirs, files in os.walk(results_dir):
        basename = os.path.basename(os.path.normpath(root))
        if not basename.startswith("teacher_training_"):
            continue
        ckpt_matches = glob.glob(os.path.join(root, "teacher_best*.weights.h5"))
        if ckpt_matches:
            run_dirs.append(root)
    run_dirs.sort()
    return run_dirs


def parse_layers_from_teacher_dirname(dirname):
    """
    Parse the GRU layer sizes from a teacher_training_* directory name.

    Patterns tried:
      teacher_training_gruXxY  ->  [X, Y]   (e.g. gru64x16 -> [64, 16])
      teacher_training_gruX    ->  [X, X]   (e.g. gru128   -> [128, 128])

    Returns a list of ints or None on failure.
    """
    m = _re.search(r"gru(\d+)x(\d+)", dirname)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    m = _re.search(r"gru(\d+)", dirname)
    if m:
        u = int(m.group(1))
        return [u, u]
    return None


def build_table1(results_dir):
    """
    Discover all teacher_training_* run directories, load their
    test_sdf_metrics.json, and assemble Table 1 rows.

    Returns a list of dicts:
        {
            "label":  str   -- e.g. "64x16"
            "rmse":   float
            "r2":     float
            "l2":     float
        }
    Rows are ordered according to MODEL_ORDER_TABLE1.
    Runs whose layers do not match any expected model are appended at the end.
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

        dirname = os.path.basename(os.path.normpath(run_dir))

        layers = parse_layers_from_teacher_dirname(dirname)
        if layers is None:
            layers_from_json = sdf.get("layers_teacher")
            if layers_from_json is not None:
                layers = list(layers_from_json)
            else:
                print(
                    f"  [WARN] Cannot determine layers for {run_dir} — skipping.",
                    flush=True,
                )
                continue

        label = layers_to_label(layers)
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
            f"  [{label}]  RMSE={rmse:.4f}  R2={r2:.4f}  L2={l2:.4f}",
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
# Relevant folders: everything under <results-dir>/ptq/ whose basename matches
# ptq_{bits}bit_* and contains test_sdf_metrics.json.
#
# Row order: MODEL_ORDER_TABLE1 for model axis, 16-bit rows first then 8-bit.
# ==============================================================================

MODEL_ORDER_TABLE2 = [
    [64, 64],
    [64, 32],
    [64, 16],
    [45, 45],
    [32, 32],
    [16, 16],
    [128, 128],
]


def parse_model_label_from_ptq_dirname(dirname):
    """
    Extract model label and bit-width from a PTQ directory name.

    Patterns tried in order:
      1. gruXxY  ->  "XxY"   (e.g. gru64x16 -> "64x16")
      2. gruX    ->  "XxX"   (e.g. gru128   -> "128x128")
      3. Fall back to ptq_args.json.

    Returns (label_str, bits_int) or (None, bits_int) if label cannot be parsed.
    bits_int is None if the bit-width cannot be parsed from the directory name.
    """
    bits_match = _re.search(r"ptq_(\d+)bit_", dirname)
    bits = int(bits_match.group(1)) if bits_match else None

    gru_match = _re.search(r"gru(\d+)x(\d+)", dirname)
    if gru_match:
        u1 = int(gru_match.group(1))
        u2 = int(gru_match.group(2))
        return f"{u1}x{u2}", bits

    gru_single_match = _re.search(r"gru(\d+)", dirname)
    if gru_single_match:
        u = int(gru_single_match.group(1))
        return f"{u}x{u}", bits

    return None, bits


def find_ptq_run_dirs(results_dir):
    """
    Return all subdirectories of <results_dir>/ptq/ whose basename matches
    ptq_{bits}bit_* and that contain test_sdf_metrics.json.

    Dirs that exist but lack test_sdf_metrics.json are reported as warnings
    (eval was never run for them).
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
            "label":  str   -- e.g. "64x16"
            "bits":   int   -- 16 or 8
            "rmse":   float
            "r2":     float
            "l2":     float
        }
    Rows are ordered: MODEL_ORDER_TABLE2 x [16, 8].
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
                label = "x".join([str(teacher_units)] * teacher_layers)
                if bits is None:
                    bits = ptq_args.get("bits", None)
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
            f"  [{label} {bits}-bit]  RMSE={rmse:.4f}  R2={r2:.4f}  L2={l2:.4f}",
            flush=True,
        )

    ordered_rows = []
    for bits in [16, 8]:
        for layers in MODEL_ORDER_TABLE2:
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
# FIXED: KD detection now parses _a{alpha}_ from the directory name.
#   alpha == 0.0  ->  has_kd = False  (no knowledge distillation)
#   alpha >  0.0  ->  has_kd = True   (knowledge distillation active)
#
# FIXED: bits detection now reads bits_kernel and bits_recurrent exclusively
# from student_args.json.  The max of the two determines the effective
# bit-width for the table row (8 or 4 for the paper).
#
# Directory naming convention:
#   vanilla_kd_T{temp}_a{alpha}_b{bias_bits}k{kernel_bits}r{recurrent_bits}a{act_bits}_
#   gru{units}x{layers}_dense{n_out}_effbs{bs}_microbs{mbs}_lr{lr}
#
# The paper Table 3 shows:
#   Rows: 8-bit block and 4-bit block
#   Cols: 128 (no KD), 32 (no KD), 32 w/KD, 16 (no KD), 16 w/KD
#
# MODEL_ORDER_TABLE3 defines (units, has_kd) pairs in paper column order.
# ==============================================================================

MODEL_ORDER_TABLE3 = [
    (128, False),
    (32,  False),
    (32,  True),
    (16,  False),
    (16,  True),
]

BIT_ORDER_TABLE3 = [8, 4]


def parse_student_run_info(run_dir):
    """
    Parse student_units, effective_bits (int), and has_kd (bool) from a
    vanilla_kd run directory.

    Strategy:
      1. Load student_args.json for student_units, bits_kernel, bits_recurrent.
      2. effective_bits = max(bits_kernel, bits_recurrent).
         This is the representative precision for the table row.
      3. has_kd: parse _a{alpha}_ from the directory basename.
           alpha parsed as float.
           alpha == 0.0  ->  has_kd = False
           alpha >  0.0  ->  has_kd = True
         If the alpha token is not found in the dirname, fall back to False
         (conservative: assume no KD rather than misclassify).

    Returns (student_units, effective_bits, has_kd) or raises ValueError.
    """
    args_path = os.path.join(run_dir, "student_args.json")
    if not os.path.exists(args_path):
        raise ValueError(f"student_args.json not found in {run_dir}")

    with open(args_path, "r") as f:
        run_args = json.load(f)

    student_units  = int(run_args["student_units"])
    bits_kernel    = int(run_args.get("bits_kernel",    8))
    bits_recurrent = int(run_args.get("bits_recurrent", 8))
    effective_bits = max(bits_kernel, bits_recurrent)

    dirname = os.path.basename(os.path.normpath(run_dir))

    alpha_match = _re.search(r"_a([\d]+\.[\d]+)_", dirname)
    if alpha_match:
        alpha = float(alpha_match.group(1))
        has_kd = alpha > 0.0
    else:
        print(
            f"  [WARN] Cannot parse alpha from dirname '{dirname}'; assuming no KD.",
            flush=True,
        )
        has_kd = False

    return student_units, effective_bits, has_kd


def find_student_run_dirs_for_table3(results_dir):
    """
    Walk results_dir recursively and return every directory that:
      1. Has a basename starting with "vanilla_kd"
      2. Contains student_best.weights.h5
      3. Contains student_args.json
      4. Contains test_sdf_metrics.json

    The vanilla_kd runs live under results_dir/results/ based on the
    observed output paths shown in the job log.
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
            "units":          int   -- e.g. 32
            "effective_bits": int   -- e.g. 8 or 4
            "has_kd":         bool
            "label":          str   -- e.g. "32 w/KD" or "32"
            "rmse":           float
            "r2":             float
            "l2":             float
        }
    Rows are ordered: BIT_ORDER_TABLE3 x MODEL_ORDER_TABLE3.
    When multiple runs share the same (units, bits, has_kd) key, the
    best (lowest RMSE) run is kept rather than the first encountered,
    so that hyperparameter sweep variants do not arbitrarily determine
    the table value.
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
            student_units, effective_bits, has_kd = parse_student_run_info(run_dir)
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

        key = (student_units, effective_bits, has_kd)
        kd_suffix = " w/KD" if has_kd else ""
        label     = f"{student_units}{kd_suffix}"

        if key in rows_by_key:
            existing_rmse = rows_by_key[key]["rmse"]
            if rmse < existing_rmse:
                print(
                    f"  [INFO] Better run found for {key}: "
                    f"RMSE {existing_rmse:.4f} -> {rmse:.4f}  ({os.path.basename(run_dir)})",
                    flush=True,
                )
                rows_by_key[key] = {
                    "units":          student_units,
                    "effective_bits": effective_bits,
                    "has_kd":         has_kd,
                    "label":          label,
                    "rmse":           rmse,
                    "r2":             r2,
                    "l2":             l2,
                }
            else:
                print(
                    f"  [INFO] Keeping existing run for {key} "
                    f"(RMSE {existing_rmse:.4f} <= {rmse:.4f})",
                    flush=True,
                )
            continue

        rows_by_key[key] = {
            "units":          student_units,
            "effective_bits": effective_bits,
            "has_kd":         has_kd,
            "label":          label,
            "rmse":           rmse,
            "r2":             r2,
            "l2":             l2,
        }
        print(
            f"  [{label} {effective_bits}-bit]  RMSE={rmse:.4f}  R2={r2:.4f}  L2={l2:.4f}",
            flush=True,
        )

    ordered_rows = []
    for bits in BIT_ORDER_TABLE3:
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

def render_table1_latex(rows):
    """
    Render Table 1 as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2Seq model size | RMSE ↓ | R² Score ↑ | L2 norm ↓

    Best value per metric column is bolded.
    """
    # Determine best values across all rows (skip None)
    rmse_vals = [r["rmse"] for r in rows if r["rmse"] is not None]
    r2_vals   = [r["r2"]   for r in rows if r["r2"]   is not None]
    l2_vals   = [r["l2"]   for r in rows if r["l2"]   is not None]

    best_rmse = min(rmse_vals) if rmse_vals else None
    best_r2   = max(r2_vals)   if r2_vals   else None
    best_l2   = min(l2_vals)   if l2_vals   else None

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
        bold_rmse = (row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
        lines.append(
            f"    {row['label']} & "
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
    MODEL_ORDER_TABLE1.

    Column layout:
      Seq2Seq Models | 16-bit/8-bit Type | RMSE ↓ | R² Score ↑ | L2 norm ↓

    Best value per metric column is bolded across the entire table
    (matching the paper, which bolds the single best across both bit-width blocks).
    """
    # Determine best values across ALL rows (both 16-bit and 8-bit combined)
    rmse_vals = [r["rmse"] for r in rows if r["rmse"] is not None]
    r2_vals   = [r["r2"]   for r in rows if r["r2"]   is not None]
    l2_vals   = [r["l2"]   for r in rows if r["l2"]   is not None]

    best_rmse = min(rmse_vals) if rmse_vals else None
    best_r2   = max(r2_vals)   if r2_vals   else None
    best_l2   = min(l2_vals)   if l2_vals   else None

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
        bold_rmse = (row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
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

def render_table3_latex(rows):
    """
    Render Table 3 as a LaTeX tabular.

    Column layout (matching the paper):
      Seq2SeqLite Models | Type | RMSE ↓ | R² Score ↑ | L2 norm ↓

    Row groups: 16-bit block then 8-bit block, each containing the
    MODEL_ORDER_TABLE3 sequence: 128, 32, 32 w/KD, 16, 16 w/KD.

    Best value per metric column is bolded across the entire table
    (matching the paper, which bolds the single best across both bit-width blocks).
    """
    # Determine best values across ALL rows (both 16-bit and 8-bit combined)
    rmse_vals = [r["rmse"] for r in rows if r["rmse"] is not None]
    r2_vals   = [r["r2"]   for r in rows if r["r2"]   is not None]
    l2_vals   = [r["l2"]   for r in rows if r["l2"]   is not None]

    best_rmse = min(rmse_vals) if rmse_vals else None
    best_r2   = max(r2_vals)   if r2_vals   else None
    best_l2   = min(l2_vals)   if l2_vals   else None

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
        r"    Seq2SeqLite Models & Type & RMSE $\downarrow$ & "
        r"R\textsuperscript{2} Score $\uparrow$ & L2 norm $\downarrow$ \\"
    )
    lines.append(r"    \midrule")

    rows_16 = [r for r in rows if r["bits"] == 16]
    rows_8  = [r for r in rows if r["bits"] == 8]

    for i, row in enumerate(rows_16):
        type_cell = "16-bit" if i == 0 else ""
        bold_rmse = (row["rmse"] is not None and best_rmse is not None and row["rmse"] == best_rmse)
        bold_r2   = (row["r2"]   is not None and best_r2   is not None and row["r2"]   == best_r2)
        bold_l2   = (row["l2"]   is not None and best_l2   is not None and row["l2"]   == best_l2)
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

    rows1 = build_table1(args.results_dir)
    tex1  = render_table1_latex(rows1)
    path1 = os.path.join(args.out_dir, "table1_teacher.tex")
    with open(path1, "w") as f:
        f.write(tex1 + "\n")
    print(f"\n[Table 1] Saved -> {path1}", flush=True)
    print(tex1, flush=True)

    rows2 = build_table2(args.results_dir)
    tex2  = render_table2_latex(rows2)
    path2 = os.path.join(args.out_dir, "table2_ptq.tex")
    with open(path2, "w") as f:
        f.write(tex2 + "\n")
    print(f"\n[Table 2] Saved -> {path2}", flush=True)
    print(tex2, flush=True)

    rows3 = build_table3(args.results_dir)
    tex3  = render_table3_latex(rows3)
    path3 = os.path.join(args.out_dir, "table3_student.tex")
    with open(path3, "w") as f:
        f.write(tex3 + "\n")
    print(f"\n[Table 3] Saved -> {path3}", flush=True)
    print(tex3, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("Done. All three tables written.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()