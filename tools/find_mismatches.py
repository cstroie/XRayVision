#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Find Mismatches - Locate cases of profound AI-vs-radiologist disagreement
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
"""
Find exams where the AI report and radiologist report disagree most, with a
focus on false negatives (radiologist found something positive, AI called it
normal) -- the highest-stakes failure mode for a diagnostic aid.

Also supports --mode unassessed, a distinct and arguably more severe failure:
exams where the AI pipeline never produced a verdict at all (ar.severity
stuck at -1) despite a positive radiologist report -- these are excluded from
every severity-comparison mode (fn/fp/all/random) since there's no AI
severity to compare against. See the check_report() bare-JSON parsing fix in
xrayvision.py for one confirmed cause of this category.

READ-ONLY. Never writes to the database.

PHI NOTE: output includes only exam uid, computed age, sex, and clinical free
text (AI findings, radiologist report, clinical justification) -- never
patient name or CNP. The clinical free text itself is still sensitive
information; review the output before pasting it into a remote/cloud session.

Usage:
    python tools/find_mismatches.py --region chest --limit 15
    python tools/find_mismatches.py --mode random --stratify-severity --limit 20
    python tools/find_mismatches.py --mode fn --audit-log xrayvision_audit.log --format json --json-out fn.json
    python tools/find_mismatches.py --mode unassessed --region chest --limit 20
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

# Add parent directory to path to import xrayvision module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrayvision import DB_FILE, SEVERITY_THRESHOLD, db_execute_query

# Hedge phrases that indicate the vision stage (rep_prompt) DID describe
# something uncertain, but the classifier stage (chk_prompt) may have scored
# it as non-actionable anyway.
HEDGE_PHRASES = [
    'suggestive of', 'cannot exclude', 'can not exclude', 'cannot rule out',
    'can not rule out', 'possible', 'probable', 'likely', 'concerning for',
    'raises the possibility', 'may represent', 'questionable', 'equivocal',
    'not excluded', 'not be excluded',
]

# Boilerplate normal-reading phrases lifted from prompts/rep_prompt.txt's own
# worked "normal" example -- their presence with no hedge phrase suggests the
# vision stage itself produced an essentially-normal read.
NORMAL_BOILERPLATE = [
    'clear lung fields', 'no focal consolidation', 'within normal limits',
    'no acute osseous abnormality', 'normal in size', 'no pleural effusion',
]


def compute_age_years(birthdate_str):
    """Compute age in whole years from a YYYY-MM-DD birthdate string."""
    if not birthdate_str:
        return None
    try:
        birth = datetime.strptime(birthdate_str, "%Y-%m-%d")
    except ValueError:
        return None
    today = datetime.now()
    age = today.year - birth.year
    if (today.month, today.day) < (birth.month, birth.day):
        age -= 1
    return age


def compute_flags(ai_text, ai_positive, ai_severity, threshold):
    """Python-side heuristics pointing at which pipeline stage likely failed."""
    flags = []

    if ai_severity is None or ai_severity == -1:
        # AI never produced a verdict for this exam at all -- distinct from a
        # false negative (AI actively said "normal"). The STAGE1/STAGE2
        # heuristics below assume a real classification happened, so skip
        # them here; find_mismatches.py --mode unassessed surfaces this case
        # directly.
        flags.append('UNASSESSED')
        return flags

    text_lower = (ai_text or '').lower()
    word_count = len((ai_text or '').split())

    has_hedge = any(phrase in text_lower for phrase in HEDGE_PHRASES)
    has_boilerplate = any(phrase in text_lower for phrase in NORMAL_BOILERPLATE)

    if has_hedge and ai_severity < threshold:
        flags.append('STAGE2_SUSPECT')

    if (word_count < 12 or (has_boilerplate and not has_hedge)) and not has_hedge:
        flags.append('STAGE1_SUSPECT')

    if ai_positive is not None:
        severity_says_positive = ai_severity >= threshold
        if bool(ai_positive) != severity_says_positive:
            flags.append('POS_SEV_INCONSISTENT')

    return flags


def check_requeued(uid, audit_log_path):
    """Grep the audit log for a REQUEUE line referencing this uid."""
    if not audit_log_path or not os.path.exists(audit_log_path):
        return None
    pattern = re.compile(r'REQUEUE\s+uid=' + re.escape(uid) + r'(\s|$)')
    try:
        with open(audit_log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if pattern.search(line):
                    return True
    except IOError:
        return None
    return False


def build_query(mode, region, threshold, limit, stratify_severity, pediatric_only, adult_only):
    conditions = [
        "e.status = 'done'",
        "rr.severity >= 0",
    ]
    params = []

    if region and region != 'all':
        conditions.append("LOWER(e.region) = ?")
        params.append(region.lower())

    if mode == 'fn':
        conditions.append("ar.severity >= 0")
        conditions.append("rr.severity >= ?")
        conditions.append("ar.severity < ?")
        params.extend([threshold, threshold])
        order_by = "(rr.severity - ar.severity) DESC, rr.severity DESC"
    elif mode == 'fp':
        conditions.append("ar.severity >= 0")
        conditions.append("ar.severity >= ?")
        conditions.append("rr.severity < ?")
        params.extend([threshold, threshold])
        order_by = "(ar.severity - rr.severity) DESC, ar.severity DESC"
    elif mode == 'all':
        conditions.append("ar.severity >= 0")
        order_by = "ABS(rr.severity - ar.severity) DESC"
    elif mode == 'random':
        conditions.append("ar.severity >= 0")
        order_by = "RANDOM()"
    elif mode == 'unassessed':
        # Radiologist found a positive case but the AI classification pass
        # (chk_prompt) never produced a valid severity for it -- ar.severity
        # stuck at its -1 "not assessed" default. Distinct from a false
        # negative (AI actively said "normal"): here the AI pipeline
        # silently failed to produce any verdict at all. See the
        # check_report() bare-JSON parsing fix in xrayvision.py.
        conditions.append("(ar.severity IS NULL OR ar.severity = -1)")
        conditions.append("rr.severity >= ?")
        params.append(threshold)
        order_by = "rr.severity DESC"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if pediatric_only:
        conditions.append("(julianday('now') - julianday(p.birthdate)) / 365.25 < 18")
    elif adult_only:
        conditions.append("(julianday('now') - julianday(p.birthdate)) / 365.25 >= 18")

    where = "WHERE " + " AND ".join(conditions)

    select = f"""
        SELECT
            e.uid, e.region, e.created,
            p.sex, p.birthdate,
            ar.text, ar.summary, ar.severity, ar.positive, ar.model, ar.created, ar.updated,
            rr.text, rr.text_en, rr.summary, rr.severity, rr.justification,
            (rr.severity - ar.severity) AS severity_diff
        FROM exams e
        INNER JOIN patients p ON e.cnp = p.cnp
        LEFT JOIN ai_reports ar ON e.uid = ar.uid
        INNER JOIN rad_reports rr ON e.uid = rr.uid
        {where}
    """

    if mode == 'random' and stratify_severity:
        # Evenly sample across radiologist severity buckets to avoid only
        # ever surfacing the extreme worst cases. Each bucket is its own
        # "SELECT ... LIMIT ?" subquery inside a UNION ALL, so the parameter
        # list must interleave each bucket's WHERE params immediately
        # followed by that bucket's own LIMIT param, in the exact order the
        # placeholders appear in the generated SQL text.
        buckets = [(0, 2), (3, 5), (6, 8), (9, 10)]
        per_bucket_limit = max(1, limit // len(buckets))
        parts = []
        full_params = []
        for lo, hi in buckets:
            bucket_where = where + f" AND rr.severity BETWEEN {lo} AND {hi}"
            parts.append(f"""
                SELECT * FROM (
                    {select.replace(where, bucket_where)}
                    ORDER BY RANDOM()
                    LIMIT ?
                )
            """)
            full_params.extend(params)
            full_params.append(per_bucket_limit)
        query = " UNION ALL ".join(parts)
        return query, full_params, order_by, True

    query = select + f" ORDER BY {order_by} LIMIT ?"
    return query, params + [limit], order_by, False


def find_mismatches(mode='fn', region='chest', threshold=None, limit=15,
                     stratify_severity=False, pediatric_only=False,
                     adult_only=False, audit_log=None):
    if threshold is None:
        threshold = SEVERITY_THRESHOLD

    query, all_params, order_by, stratified = build_query(
        mode, region, threshold, limit, stratify_severity, pediatric_only, adult_only)

    rows = db_execute_query(query, tuple(all_params), fetch_mode='all') or []

    cases = []
    for row in rows:
        (uid, region_val, created, sex, birthdate,
         ai_text, ai_summary, ai_severity, ai_positive, ai_model, ai_created, ai_updated,
         rad_text, rad_text_en, rad_summary, rad_severity, rad_justification,
         severity_diff) = row

        flags = compute_flags(ai_text, ai_positive, ai_severity, threshold)
        was_requeued = check_requeued(uid, audit_log)
        if was_requeued:
            flags.append('WAS_REQUEUED')

        cases.append({
            'uid': uid,
            'region': region_val,
            'created': created,
            'age_years': compute_age_years(birthdate),
            'sex': sex,
            'ai_text': ai_text,
            'ai_summary': ai_summary,
            'ai_severity': ai_severity,
            'ai_positive': ai_positive,
            'ai_model': ai_model,
            'rad_text': rad_text,
            'rad_text_en': rad_text_en,
            'rad_summary': rad_summary,
            'rad_severity': rad_severity,
            'rad_justification': rad_justification,
            'severity_diff': severity_diff,
            'flags': flags,
            'was_requeued': was_requeued,
        })

    # Stratified random query issues one ORDER BY RANDOM() per bucket via
    # UNION ALL, so re-shuffle the combined result for a non-bucket-grouped
    # final order (still capped at `limit`).
    if stratified:
        import random
        random.shuffle(cases)
        cases = cases[:limit]

    return cases


def format_case_text(case, index, total):
    age = f"{case['age_years']}y" if case['age_years'] is not None else "age unknown"
    pediatric = "yes" if (case['age_years'] is not None and case['age_years'] < 18) else "no"
    diff_str = f"{case['severity_diff']:+d}" if case['severity_diff'] is not None else "n/a (AI never assessed)"
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"Case {index}/{total}  uid={case['uid']}   region={case['region']}   "
        f"severity_diff={diff_str} (rad={case['rad_severity']}, ai={case['ai_severity']})"
    )
    lines.append(f"Patient: {age} {case['sex'] or '?'}  (pediatric={pediatric})   AI model: {case['ai_model'] or 'n/a'}")
    lines.append("-" * 80)
    lines.append(f"RADIOLOGIST REPORT (severity={case['rad_severity']}, summary=\"{case['rad_summary']}\")")
    if case['rad_justification']:
        lines.append(f"  Clinical indication: {case['rad_justification'].strip()}")
    rad_text = case['rad_text_en'] or case['rad_text'] or ''
    source = 'text_en' if case['rad_text_en'] else 'text (untranslated)'
    lines.append(f"  [{source}] {rad_text.strip()}")
    lines.append("")
    if case['ai_text'] is None and case['ai_severity'] is None:
        lines.append("AI FINDINGS: none (no ai_reports row at all for this exam)")
    else:
        lines.append(f"AI FINDINGS (severity={case['ai_severity']}, positive={case['ai_positive']}, summary=\"{case['ai_summary']}\")")
        lines.append(f"  {(case['ai_text'] or '').strip()}")
    lines.append("")
    flags_str = ", ".join(case['flags']) if case['flags'] else "none"
    lines.append(f"FLAGS: {flags_str}")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find worst AI-vs-radiologist mismatches, false-negative-focused by default.")
    parser.add_argument('--limit', type=int, default=15)
    parser.add_argument('--threshold', type=int, default=None,
                         help=f"Severity threshold (default: config SEVERITY_THRESHOLD={SEVERITY_THRESHOLD})")
    parser.add_argument('--region', default='chest', help="Region to filter on, or 'all'")
    parser.add_argument('--mode', choices=['fn', 'fp', 'all', 'random', 'unassessed'], default='fn',
                         help="fn=false negative (default), fp=false positive, all=worst mismatches either way, "
                              "random=unfiltered sample, unassessed=rad positive but AI never produced a "
                              "severity at all (ar.severity stuck at -1) -- a pipeline reliability failure "
                              "distinct from a false negative")
    parser.add_argument('--stratify-severity', action='store_true',
                         help="Only meaningful with --mode random: sample evenly across rad severity buckets")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pediatric-only', action='store_true')
    group.add_argument('--adult-only', action='store_true')
    parser.add_argument('--audit-log', default=None,
                         help="Path to xrayvision_audit.log to cross-reference REQUEUE events")
    parser.add_argument('--format', choices=['text', 'json', 'both'], default='both')
    parser.add_argument('--json-out', default=None,
                         help="Output path for JSON (default: mismatches_<region>_<mode>.json)")
    args = parser.parse_args()

    print(f"Using database: {DB_FILE}", file=sys.stderr)
    print(f"Mode: {args.mode}  Region: {args.region}  Threshold: {args.threshold or SEVERITY_THRESHOLD}", file=sys.stderr)

    cases = find_mismatches(
        mode=args.mode,
        region=args.region,
        threshold=args.threshold,
        limit=args.limit,
        stratify_severity=args.stratify_severity,
        pediatric_only=args.pediatric_only,
        adult_only=args.adult_only,
        audit_log=args.audit_log,
    )

    if not cases:
        print("No matching cases found.", file=sys.stderr)
        return

    if args.format in ('text', 'both'):
        for i, case in enumerate(cases, 1):
            print(format_case_text(case, i, len(cases)))

    if args.format in ('json', 'both'):
        json_out = args.json_out or f"mismatches_{args.region}_{args.mode}.json"
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(cases)} cases to {json_out}", file=sys.stderr)


if __name__ == '__main__':
    main()
