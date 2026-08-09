#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Evaluate Prompts - Quality metrics for prompt_lab.py results
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
"""
Compute sensitivity-focused quality metrics from tools/prompt_lab.py's JSONL
output, comparing each prompt variant's classifications against radiologist
ground truth. Deliberately dependency-free (stdlib math/statistics only) --
data volume here is tens of rows, not a DataFrame problem.

Ground truth positive: rad.severity >= threshold (fetched from the DB per uid).
Predicted positive:    classification['severity'] >= threshold (mirrors
                        production's actual gate in send_exam_to_llm; the
                        'pathologic' field is reported separately as a
                        secondary view, since disagreement between the two
                        surfaces classifier self-contradiction bugs too).

READ-ONLY. Never writes to the database (only reads rad_reports/patients for
ground truth and age).

Usage:
    python tools/evaluate_prompts.py --results results.jsonl \\
        --baseline-variant baseline --candidate-variant candidate \\
        --md-out evaluation_report.md
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrayvision import db_get_exams, SEVERITY_THRESHOLD


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(paths):
    rows = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def load_case_sets(specs):
    """['worst=mismatches_chest_fn.json', 'random=mismatches_chest_random.json']
    -> {uid: 'worst'|'random'|...}. Accepts either find_mismatches.py's JSON
    (list of case dicts with a 'uid' key) or a plain JSON array of uid strings."""
    uid_to_set = {}
    for spec in specs or []:
        if '=' not in spec:
            raise ValueError(f"--case-set must be NAME=PATH, got: {spec}")
        name, path = spec.split('=', 1)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            uid = item['uid'] if isinstance(item, dict) else item
            uid_to_set[uid] = name
    return uid_to_set


def fetch_ground_truth(uids, threshold):
    """uid -> {'rad_severity': int, 'rad_positive': bool, 'age_years': int|None}"""
    truth = {}
    for uid in uids:
        exams, _ = db_get_exams(uid=uid, limit=1)
        if not exams:
            continue
        exam = exams[0]
        rad = exam['report']['rad']
        rad_severity = rad.get('severity')
        age = exam['patient'].get('age')
        truth[uid] = {
            'rad_severity': rad_severity,
            'rad_positive': rad_severity is not None and rad_severity >= threshold,
            'age_years': age if age is not None and age >= 0 else None,
        }
    return truth


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_counts(pairs):
    """pairs: list of (predicted_bool, actual_bool) -> dict of TP/TN/FP/FN."""
    tp = sum(1 for p, a in pairs if p and a)
    tn = sum(1 for p, a in pairs if not p and not a)
    fp = sum(1 for p, a in pairs if p and not a)
    fn = sum(1 for p, a in pairs if not p and a)
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'n': len(pairs)}


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def safe_div(num, den):
    return num / den if den else None


def compute_metrics(cm):
    tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)
    accuracy = safe_div(tp + tn, cm['n'])
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if precision and sensitivity and (precision + sensitivity) > 0 else None)
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if cm['n'] else 0
    mcc = ((tp * tn - fp * fn) / mcc_denom) if mcc_denom else 0.0

    sens_ci = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else None
    spec_ci = wilson_ci(tn, tn + fp) if (tn + fp) > 0 else None

    return {
        'sensitivity': sensitivity, 'sensitivity_ci': sens_ci,
        'specificity': specificity, 'specificity_ci': spec_ci,
        'precision': precision, 'npv': npv, 'accuracy': accuracy,
        'f1': f1, 'mcc': mcc,
        **cm,
    }


def mcnemar_or_binomial(b, c):
    """Paired baseline-vs-candidate significance test on discordant pairs.
    b = baseline-correct/candidate-wrong, c = baseline-wrong/candidate-correct.
    Falls back to an exact two-sided binomial sign test (stdlib math.comb,
    no scipy needed) when b+c is too small for the chi-square approximation
    to be reliable (commonly-cited floor: b+c >= 25)."""
    n = b + c
    if n == 0:
        return {'test': 'none', 'n_discordant': 0, 'statistic': None, 'p_value': None,
                'note': 'No discordant pairs -- variants agree on every case.'}

    if n >= 25:
        chi2 = ((abs(b - c) - 1) ** 2) / n
        # p-value for chi-square with df=1 via the complementary error function
        p_value = math.erfc(math.sqrt(chi2 / 2))
        return {'test': 'mcnemar_chi2_corrected', 'n_discordant': n, 'statistic': chi2,
                'p_value': p_value, 'note': None}

    # Exact two-sided binomial sign test at p=0.5
    k = min(b, c)
    total = sum(math.comb(n, i) for i in range(0, k + 1))
    p_value = min(1.0, 2 * total / (2 ** n))
    return {
        'test': 'exact_binomial_sign', 'n_discordant': n, 'statistic': None,
        'p_value': p_value,
        'note': f'b+c={n} < 25: chi-square approximation unreliable at this n, '
                f'using an exact binomial sign test instead. Treat any p-value '
                f'here as indicative only.',
    }


# ---------------------------------------------------------------------------
# Aggregation views
# ---------------------------------------------------------------------------

def per_sample_pairs(rows, variant, truth, threshold, use_pathologic=False):
    pairs = []
    parse_failures = 0
    for row in rows:
        if row['variant'] != variant:
            continue
        uid = row['uid']
        if uid not in truth:
            continue
        cls = row.get('classification')
        if not cls or 'error' in cls:
            parse_failures += 1
            continue
        actual = truth[uid]['rad_positive']
        if use_pathologic:
            predicted = cls.get('pathologic') == 'yes'
        else:
            predicted = cls.get('severity', -1) >= threshold
        pairs.append((predicted, actual))
    return pairs, parse_failures


def per_uid_max_severity(rows, variant, threshold, use_pathologic=False):
    """uid -> predicted_positive, taking the max severity (or any 'yes'
    pathologic) across all samples for that (variant, uid)."""
    by_uid = defaultdict(list)
    for row in rows:
        if row['variant'] != variant:
            continue
        cls = row.get('classification')
        if not cls or 'error' in cls:
            continue
        by_uid[row['uid']].append(cls)

    result = {}
    for uid, classifications in by_uid.items():
        if use_pathologic:
            result[uid] = any(c.get('pathologic') == 'yes' for c in classifications)
        else:
            max_sev = max((c.get('severity', -1) for c in classifications), default=-1)
            result[uid] = max_sev >= threshold
    return result


def per_uid_pairs(rows, variant, truth, threshold, use_pathologic=False):
    preds = per_uid_max_severity(rows, variant, threshold, use_pathologic)
    pairs = []
    for uid, predicted in preds.items():
        if uid not in truth:
            continue
        pairs.append((predicted, truth[uid]['rad_positive']))
    return pairs


def stratify(rows, truth, key_fn):
    """Split rows into groups by key_fn(uid) -> label, dropping rows whose
    uid isn't in truth (no ground truth) or whose key is None."""
    groups = defaultdict(list)
    for row in rows:
        uid = row['uid']
        if uid not in truth:
            continue
        label = key_fn(uid)
        if label is None:
            continue
        groups[label].append(row)
    return groups


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def fmt_pct(x):
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def fmt_ci(ci):
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci else "n/a"


def render_metrics_table(label, metrics_by_variant, out_lines):
    out_lines.append(f"\n### {label}\n")
    out_lines.append("| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |")
    out_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for variant, m in metrics_by_variant.items():
        f1_str = f"{m['f1']:.3f}" if m['f1'] is not None else 'n/a'
        out_lines.append(
            f"| {variant} | {m['n']} | {m['TP']} | {m['TN']} | {m['FP']} | {m['FN']} | "
            f"{fmt_pct(m['sensitivity'])} {fmt_ci(m['sensitivity_ci'])} | "
            f"{fmt_pct(m['specificity'])} {fmt_ci(m['specificity_ci'])} | "
            f"{fmt_pct(m['precision'])} | "
            f"{f1_str} | "
            f"{m['mcc']:.3f} |"
        )


def render_paired_test(baseline, candidate, pairs_baseline_by_uid, pairs_candidate_by_uid, out_lines):
    common_uids = set(pairs_baseline_by_uid) & set(pairs_candidate_by_uid)
    b = c = 0
    for uid in common_uids:
        base_correct = pairs_baseline_by_uid[uid][0] == pairs_baseline_by_uid[uid][1]
        cand_correct = pairs_candidate_by_uid[uid][0] == pairs_candidate_by_uid[uid][1]
        if base_correct and not cand_correct:
            b += 1
        elif not base_correct and cand_correct:
            c += 1
    test = mcnemar_or_binomial(b, c)
    out_lines.append(f"\n**{baseline} vs {candidate}** (paired on {len(common_uids)} common uids, per-uid max-severity view):")
    out_lines.append(f"- discordant pairs: b({baseline} right / {candidate} wrong)={b}, "
                      f"c({baseline} wrong / {candidate} right)={c}")
    if test['p_value'] is not None:
        stat_str = f"{test['statistic']:.3f}" if test['statistic'] is not None else "n/a"
        out_lines.append(f"- test: {test['test']}, statistic={stat_str}, p-value={test['p_value']:.4f}")
    else:
        out_lines.append(f"- test: {test['test']} (undefined -- no discordant pairs)")
    if test['note']:
        out_lines.append(f"- **caveat:** {test['note']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute quality metrics from prompt_lab.py JSONL results against radiologist ground truth.")
    parser.add_argument('--results', action='append', required=True, help="JSONL file(s) from prompt_lab.py")
    parser.add_argument('--threshold', type=int, default=None,
                         help=f"Severity threshold (default: config SEVERITY_THRESHOLD={SEVERITY_THRESHOLD})")
    parser.add_argument('--baseline-variant', default='baseline')
    parser.add_argument('--candidate-variant', action='append', default=[],
                         help="Variant name(s) to compare against baseline (repeatable)")
    parser.add_argument('--case-set', action='append', default=[],
                         help="NAME=PATH (repeatable): tag uids with a case-set label from a "
                              "find_mismatches.py JSON file, for worst-case-set vs broader-sample stratification")
    parser.add_argument('--md-out', default=None)
    parser.add_argument('--csv-out', default=None)
    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else SEVERITY_THRESHOLD
    rows = load_results(args.results)
    if not rows:
        print("No result rows loaded.", file=sys.stderr)
        return 1

    all_uids = sorted({row['uid'] for row in rows})
    print(f"Fetching ground truth for {len(all_uids)} uids...", file=sys.stderr)
    truth = fetch_ground_truth(all_uids, threshold)
    missing = set(all_uids) - set(truth)
    if missing:
        print(f"WARNING: {len(missing)} uids not found in DB, excluded: {sorted(missing)[:5]}...", file=sys.stderr)

    case_set_map = load_case_sets(args.case_set)
    variants = sorted({row['variant'] for row in rows})

    out_lines = ["# Prompt Evaluation Report\n"]
    out_lines.append(f"Threshold: {threshold}  |  Variants: {', '.join(variants)}  |  "
                      f"Total result rows: {len(rows)}  |  Unique uids: {len(all_uids)}\n")

    def build_metrics(row_subset, use_pathologic):
        per_sample = {}
        per_uid = {}
        for v in variants:
            pairs, failures = per_sample_pairs(row_subset, v, truth, threshold, use_pathologic)
            m = compute_metrics(confusion_counts(pairs))
            m['parse_failures'] = failures
            per_sample[v] = m
            per_uid[v] = compute_metrics(confusion_counts(per_uid_pairs(row_subset, v, truth, threshold, use_pathologic)))
        return per_sample, per_uid

    # --- Overall ---
    out_lines.append("\n## Overall\n")
    for use_path, label in [(False, "severity >= threshold (primary, matches production gate)"),
                             (True, "pathologic == yes (secondary view)")]:
        per_sample, per_uid = build_metrics(rows, use_path)
        render_metrics_table(f"Per-sample view -- {label}", per_sample, out_lines)
        render_metrics_table(f"Per-uid max-severity view -- {label}", per_uid, out_lines)

    # --- Paired significance test (severity-based, per-uid view) ---
    if args.candidate_variant:
        out_lines.append("\n## Paired significance (baseline vs candidate)\n")
        for candidate in args.candidate_variant:
            base_map = per_uid_max_severity(rows, args.baseline_variant, threshold)
            cand_map = per_uid_max_severity(rows, candidate, threshold)
            base_pairs_by_uid = {uid: (pred, truth[uid]['rad_positive']) for uid, pred in base_map.items() if uid in truth}
            cand_pairs_by_uid = {uid: (pred, truth[uid]['rad_positive']) for uid, pred in cand_map.items() if uid in truth}
            render_paired_test(args.baseline_variant, candidate,
                                base_pairs_by_uid, cand_pairs_by_uid, out_lines)

    # --- Stratified: pediatric vs adult ---
    def age_key(uid):
        age = truth.get(uid, {}).get('age_years')
        if age is None:
            return None
        return 'pediatric' if age < 18 else 'adult'

    age_groups = stratify(rows, truth, age_key)
    if age_groups:
        out_lines.append("\n## Stratified: pediatric vs adult\n")
        for group_label, group_rows in age_groups.items():
            per_sample, _ = build_metrics(group_rows, False)
            render_metrics_table(f"{group_label} (n_uids={len({r['uid'] for r in group_rows})}) -- per-sample, severity-based",
                                  per_sample, out_lines)

    # --- Stratified: case set (worst-case vs broader sample) ---
    if case_set_map:
        def case_set_key(uid):
            return case_set_map.get(uid)

        cs_groups = stratify(rows, truth, case_set_key)
        if cs_groups:
            out_lines.append("\n## Stratified: case set (checks the candidate isn't overfit to only the worst cases)\n")
            for group_label, group_rows in cs_groups.items():
                per_sample, _ = build_metrics(group_rows, False)
                render_metrics_table(f"case_set={group_label} (n_uids={len({r['uid'] for r in group_rows})}) -- per-sample, severity-based",
                                      per_sample, out_lines)

    report = "\n".join(out_lines)
    print(report)

    if args.md_out:
        with open(args.md_out, 'w', encoding='utf-8') as f:
            f.write(report + "\n")
        print(f"\nWrote {args.md_out}", file=sys.stderr)

    if args.csv_out:
        import csv
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['view', 'variant', 'use_pathologic', 'n', 'TP', 'TN', 'FP', 'FN',
                              'sensitivity', 'specificity', 'precision', 'f1', 'mcc'])
            for use_path in (False, True):
                per_sample, per_uid = build_metrics(rows, use_path)
                for view_name, metrics_by_variant in [('per_sample', per_sample), ('per_uid_max', per_uid)]:
                    for variant, m in metrics_by_variant.items():
                        writer.writerow([view_name, variant, use_path, m['n'], m['TP'], m['TN'], m['FP'], m['FN'],
                                          m['sensitivity'], m['specificity'], m['precision'], m['f1'], m['mcc']])
        print(f"Wrote {args.csv_out}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
