#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Check Report Probe - Test chk_prompt.txt classification stability on given texts
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
"""
Replay xrayvision.check_report() against a fixed list of report texts, N
times each, to measure classification stability/hallucination rate. No
image, no exam DB lookup needed -- check_report() is text-only.

Motivating finding: several radiologist reports in mismatches_chest_fn.json
that read as trivially normal in the original Romanian ("Cord, pulmon
normale radiologic.") were scored by check_rad_report_and_update() ->
check_report() as severe pathology entirely disconnected from the text
(severity 10, summary "stemi"/"small cell carcinoma"/"hemoragie cerebrala").
This tool re-runs those exact texts against the live endpoint to see how
often that reproduces, and to A/B a candidate chk_prompt.txt fix against it.

SAFETY: check_report() itself never writes to the DB (see xrayvision.py).
This script only reads report text from an input JSON/text file and never
touches the database at all.

Usage:
    python tools/check_report_probe.py --variant baseline \\
        --variant candidate=tools/prompt_variants/v6_chk_grounding \\
        --texts-file suspect_texts.json --samples 5 --output probe_results.jsonl

    python tools/check_report_probe.py --variant baseline \\
        --text "Cord, pulmon normale radiologic." --samples 5 --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import aiohttp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xrayvision
from xrayvision import check_report, OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY, USER_AGENT

_FORBIDDEN_NAMES = ('send_exam_to_openai', 'check_ai_report_and_update', 'check_rad_report_and_update',
                     'db_insert', 'db_update', 'db_execute_query_retry')
for _name in _FORBIDDEN_NAMES:
    assert _name not in globals(), f"check_report_probe.py must never import {_name} -- it writes to the DB"

PROMPT_FILES = {'CHK_PROMPT': 'chk_prompt.txt'}


def load_variant_overrides(variant_dir):
    overrides = {}
    for key, filename in PROMPT_FILES.items():
        path = os.path.join(variant_dir, filename)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                overrides[key] = f.read()
    return overrides


def apply_prompts(baseline_prompts, overrides):
    xrayvision.PROMPTS.clear()
    xrayvision.PROMPTS.update(baseline_prompts)
    xrayvision.PROMPTS.update(overrides)


def parse_variant_args(variant_args):
    variants = []
    for spec in variant_args:
        if '=' in spec:
            name, path = spec.split('=', 1)
        else:
            name, path = spec, None
        if name != 'baseline' and path is None:
            raise ValueError(f"Variant '{name}' must be given as NAME=DIR (only 'baseline' needs no path)")
        variants.append((name, path))
    return variants


def load_texts(args):
    """Returns a list of (label, text) pairs."""
    texts = []
    if args.text:
        for i, t in enumerate(args.text):
            texts.append((f"cli-{i}", t))
    if args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    texts.append((item[:40], item))
                elif isinstance(item, dict):
                    # Accept find_mismatches.py's case dicts directly: use rad_text.
                    label = item.get('uid') or item.get('label') or str(len(texts))
                    text = item.get('text') or item.get('rad_text')
                    if text:
                        texts.append((label, text))
        else:
            raise ValueError(f"Unrecognized JSON shape in {args.texts_file}")
    return texts


def probe_active_openai_url(override_url=None):
    if override_url:
        xrayvision.active_openai_url = override_url
        return override_url

    async def _probe():
        for url in [OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY]:
            base_url = url.split('/v1/')[0] if '/v1/' in url else url.rstrip('/')
            models_url = f"{base_url}/v1/models"
            try:
                async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                    async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return url
            except Exception:
                continue
        return None

    active = asyncio.get_event_loop().run_until_complete(_probe())
    xrayvision.active_openai_url = active
    return active


def summarize(rows):
    """Per (variant, label): distribution of severity values and summaries,
    flagged as UNSTABLE if severity range > 3 or summaries disagree wildly."""
    by_key = {}
    for row in rows:
        key = (row['variant'], row['label'])
        by_key.setdefault(key, []).append(row)

    print("\n" + "=" * 100)
    for (variant, label), samples in by_key.items():
        severities = [s['classification'].get('severity') for s in samples
                      if s['classification'] and 'error' not in s['classification']]
        summaries = [s['classification'].get('summary') for s in samples
                     if s['classification'] and 'error' not in s['classification']]
        errors = sum(1 for s in samples if not s['classification'] or 'error' in s['classification'])
        print(f"variant={variant}  label={label}")
        print(f"  text: {samples[0]['text'][:120]}")
        if severities:
            print(f"  severities: {severities}  (range={max(severities)-min(severities)})")
            print(f"  summaries: {dict(Counter(summaries))}")
        if errors:
            print(f"  parse errors: {errors}/{len(samples)}")
        if severities and (max(severities) - min(severities)) >= 4:
            print(f"  ** UNSTABLE: severity range >= 4 across {len(severities)} samples **")
        print("-" * 100)


async def main_async(args):
    variants = parse_variant_args(args.variant)
    texts = load_texts(args)
    if not texts:
        print("No texts given (use --text and/or --texts-file).", file=sys.stderr)
        return 1

    baseline_prompts = dict(xrayvision.PROMPTS)

    if not args.dry_run:
        active = probe_active_openai_url(args.openai_url)
        if not active:
            print("No healthy OpenAI-compatible endpoint found.", file=sys.stderr)
            return 1
        print(f"Using AI endpoint: {active}", file=sys.stderr)

    out_f = open(args.output, 'a', encoding='utf-8') if (args.output and not args.dry_run) else None
    all_rows = []

    try:
        for variant_name, variant_dir in variants:
            overrides = load_variant_overrides(variant_dir) if variant_dir else {}
            apply_prompts(baseline_prompts, overrides)

            if args.dry_run:
                print(f"\n===== variant={variant_name} =====")
                print(f"--- CHK_PROMPT ---\n{xrayvision.PROMPTS['CHK_PROMPT'].strip()}\n")
                continue

            for label, text in texts:
                for sample_idx in range(args.samples):
                    t0 = time.monotonic()
                    classification = await check_report(text)
                    latency_ms = int((time.monotonic() - t0) * 1000)
                    row = {
                        'variant': variant_name, 'label': label, 'text': text,
                        'sample_idx': sample_idx, 'classification': classification,
                        'latency_ms': latency_ms,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    }
                    all_rows.append(row)
                    if out_f:
                        out_f.write(json.dumps(row, ensure_ascii=False) + '\n')
                        out_f.flush()
                    print(f"[{variant_name}] {label} sample={sample_idx} -> {classification}", file=sys.stderr)
    finally:
        xrayvision.PROMPTS.clear()
        xrayvision.PROMPTS.update(baseline_prompts)
        if out_f:
            out_f.close()

    if all_rows:
        summarize(all_rows)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Probe check_report()'s classification stability on given report texts.")
    parser.add_argument('--variant', action='append', required=True,
                         help="NAME=DIR (repeatable). 'baseline' needs no DIR. DIR should contain chk_prompt.txt.")
    parser.add_argument('--text', action='append', default=[], help="Report text (repeatable)")
    parser.add_argument('--texts-file', default=None,
                         help="JSON array of strings, or of {label,text} / find_mismatches.py case dicts (uses rad_text)")
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--output', default=None, help="JSONL output path")
    parser.add_argument('--openai-url', default=None)
    parser.add_argument('--dry-run', action='store_true', help="Print assembled CHK_PROMPT, skip the HTTP call")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == '__main__':
    sys.exit(main())
