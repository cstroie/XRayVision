#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Prompt Lab - Replay real exams through candidate prompt variants
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
"""
Replay real exams (by uid) through xrayvision's real prompt-assembly and AI
call pipeline, using one or more candidate prompt variants, and score each
generated report with the same chk_prompt classification step used in
production. Used to A/B candidate prompts against the live MedGemma endpoint
before promoting them into prompts/*.txt.

SAFETY: this script must NEVER call xrayvision.send_exam_to_openai() or
xrayvision.check_ai_report_and_update() -- both write to the production
database. It only uses the side-effect-free primitives: db_get_exams (read),
prepare_exam_data, create_exam_prompt, prepare_ai_request_data, send_to_openai
(a bare HTTP call), parse_ai_report_text, and check_report (returns a dict,
does not touch the DB).

Usage:
    python tools/prompt_lab.py --variant baseline \\
        --variant candidate=tools/prompt_variants/v4_combined \\
        --uids-file mismatches_chest_fn.json --samples 3 --output results.jsonl

    # --uids-file is repeatable: combine the worst-case set and a random
    # sample in one run, to check a candidate isn't just overfit to the
    # hardest cases
    python tools/prompt_lab.py --variant baseline --variant v9=tools/prompt_variants/v9_vision_combined \\
        --uids-file mismatches_chest_fn.json --uids-file mismatches_chest_random.json \\
        --samples 3 --output results.jsonl

    python tools/prompt_lab.py --variant baseline --uids fn-1 fn-2 --dry-run
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import aiohttp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xrayvision
from xrayvision import (
    db_get_exams, prepare_exam_data, create_exam_prompt, prepare_ai_request_data,
    send_to_openai, parse_ai_report_text, check_report,
    OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY, USER_AGENT,
)

# Safety guard: fail loudly if this script is ever edited to import the
# DB-writing functions it must not call.
_FORBIDDEN_NAMES = ('send_exam_to_openai', 'check_ai_report_and_update')
for _name in _FORBIDDEN_NAMES:
    assert _name not in globals(), (
        f"prompt_lab.py must never import {_name} -- it writes to the production DB"
    )

PROMPT_FILES = {
    'REP_PROMPT': 'rep_prompt.txt',
    'USR_PROMPT': 'usr_prompt.txt',
    'CHK_PROMPT': 'chk_prompt.txt',
    'REV_PROMPT': 'rev_prompt.txt',
}


def short_hash(text):
    if not text:
        return None
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]


def load_variant_overrides(variant_dir):
    """Read whichever of rep/usr/chk/rev prompt files exist in variant_dir."""
    overrides = {}
    for key, filename in PROMPT_FILES.items():
        path = os.path.join(variant_dir, filename)
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                overrides[key] = f.read()
    return overrides


def apply_prompts(baseline_prompts, overrides):
    """Mutate xrayvision.PROMPTS in place: baseline + this variant's overrides."""
    xrayvision.PROMPTS.clear()
    xrayvision.PROMPTS.update(baseline_prompts)
    xrayvision.PROMPTS.update(overrides)


def parse_variant_args(variant_args):
    """['baseline', 'candidate=tools/prompt_variants/v4'] -> [(name, dir_or_None), ...]"""
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


def load_uids(args):
    """--uids-file is repeatable, so e.g. the worst-case set and a random
    sample can be combined in one prompt_lab.py run without pre-merging."""
    uids = list(args.uids or [])
    for uids_file in (args.uids_file or []):
        with open(uids_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Unrecognized JSON shape in {uids_file}")
        for item in data:
            if isinstance(item, str):
                uids.append(item)
            elif isinstance(item, dict) and 'uid' in item:
                uids.append(item['uid'])
    # de-duplicate, preserve order
    seen = set()
    result = []
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


async def probe_active_openai_url(override_url=None):
    """One-shot mirror of openai_health_check()'s body -- picks a working
    endpoint without starting the background loop (which never runs here).
    Must be awaited from within an already-running event loop (main_async
    runs under asyncio.run()) -- do not try to spin up a nested loop here."""
    if override_url:
        xrayvision.active_openai_url = override_url
        return override_url

    for url in [OPENAI_URL_PRIMARY, OPENAI_URL_SECONDARY]:
        base_url = url.split('/v1/')[0] if '/v1/' in url else url.rstrip('/')
        models_url = f"{base_url}/v1/models"
        try:
            async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
                async with session.get(models_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        xrayvision.active_openai_url = url
                        return url
        except Exception:
            continue

    xrayvision.active_openai_url = None
    return None


def load_completed_keys(output_path):
    """Read an existing JSONL output file and return the set of
    (variant, uid, sample_idx) triples already completed, for resumability."""
    completed = set()
    if not output_path or not os.path.exists(output_path):
        return completed
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add((row['variant'], row['uid'], row['sample_idx']))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


async def run_one_sample(exam, region, question, subject, anatomy, image_bytes,
                          do_review, max_retries, sleep_seconds, impression_max_words=3):
    """Run one vision + check pass (optionally + review pass) for the
    currently-applied PROMPTS. Returns a result dict, never writes to the DB."""
    prompt = create_exam_prompt(exam, region, question, subject, anatomy)
    headers, data = prepare_ai_request_data(prompt, image_bytes)

    result_row = {
        'raw_response': None, 'findings': None, 'impression': None,
        'check_input_text': None, 'classification': None,
        'latency_ms_vision': None, 'latency_ms_check': None,
        'reviewed_classification': None, 'error': None,
    }

    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
        attempt = 1
        api_result = None
        t0 = time.monotonic()
        while attempt <= max_retries:
            api_result = await send_to_openai(session, headers, data)
            if api_result:
                break
            await asyncio.sleep(2 ** attempt)
            attempt += 1
        result_row['latency_ms_vision'] = int((time.monotonic() - t0) * 1000)

        if not api_result:
            result_row['error'] = 'No response from AI after retries'
            return result_row

        try:
            response_text = api_result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            result_row['error'] = f'Malformed AI response: {e}'
            return result_row

        result_row['raw_response'] = response_text
        findings, impression = parse_ai_report_text(response_text, impression_max_words=impression_max_words)
        result_row['findings'] = findings
        result_row['impression'] = impression

        report_text = f"FINDINGS: {findings}\n\nIMPRESSION: {impression}" if impression else findings
        result_row['check_input_text'] = report_text

        if sleep_seconds:
            await asyncio.sleep(sleep_seconds)

        t1 = time.monotonic()
        result_row['classification'] = await check_report(report_text)
        result_row['latency_ms_check'] = int((time.monotonic() - t1) * 1000)

        if do_review:
            data['messages'].append({'role': 'assistant', 'content': report_text})
            data['messages'].append({'role': 'user', 'content': xrayvision.PROMPTS['REV_PROMPT'].strip()})
            review_result = await send_to_openai(session, headers, data)
            if review_result:
                try:
                    review_text = review_result["choices"][0]["message"]["content"].strip()
                    r_findings, r_impression = parse_ai_report_text(review_text, impression_max_words=impression_max_words)
                    r_report_text = f"FINDINGS: {r_findings}\n\nIMPRESSION: {r_impression}" if r_impression else r_findings
                    result_row['reviewed_classification'] = await check_report(r_report_text)
                except (KeyError, IndexError, TypeError):
                    pass

    return result_row


def dry_run_preview(exam, region, question, subject, anatomy):
    prompt = create_exam_prompt(exam, region, question, subject, anatomy)
    print(f"--- system (REP_PROMPT) ---\n{xrayvision.PROMPTS['REP_PROMPT'].strip()}\n")
    print(f"--- user (assembled) ---\n{prompt}\n")
    print(f"--- CHK_PROMPT (system for classification pass) ---\n{xrayvision.PROMPTS['CHK_PROMPT'].strip()}\n")


async def main_async(args):
    variants = parse_variant_args(args.variant)
    uids = load_uids(args)
    if not uids:
        print("No uids given (use --uids and/or --uids-file).", file=sys.stderr)
        return 1

    baseline_prompts = dict(xrayvision.PROMPTS)  # pristine copy, captured once at startup

    if not args.dry_run:
        active = await probe_active_openai_url(args.openai_url)
        if not active:
            print("No healthy OpenAI-compatible endpoint found (PRIMARY/SECONDARY both failed).",
                  file=sys.stderr)
            return 1
        print(f"Using AI endpoint: {active}", file=sys.stderr)

    completed = load_completed_keys(args.output) if not args.dry_run else set()

    out_f = None
    if not args.dry_run and args.output:
        out_f = open(args.output, 'a', encoding='utf-8')

    try:
        for variant_name, variant_dir in variants:
            overrides = load_variant_overrides(variant_dir) if variant_dir else {}
            apply_prompts(baseline_prompts, overrides)

            rep_hash = short_hash(xrayvision.PROMPTS.get('REP_PROMPT'))
            usr_hash = short_hash(xrayvision.PROMPTS.get('USR_PROMPT'))
            chk_hash = short_hash(xrayvision.PROMPTS.get('CHK_PROMPT'))

            for uid in uids:
                exams, _ = db_get_exams(uid=uid, limit=1)
                if not exams:
                    print(f"[{variant_name}] uid={uid}: not found in DB, skipping", file=sys.stderr)
                    continue
                exam = exams[0]

                try:
                    region, question, subject, anatomy, image_bytes = prepare_exam_data(exam)
                except FileNotFoundError:
                    print(f"[{variant_name}] uid={uid}: image file not found under --images-dir, skipping",
                          file=sys.stderr)
                    continue
                if region is None:
                    print(f"[{variant_name}] uid={uid}: region not supported, skipping", file=sys.stderr)
                    continue

                if args.dry_run:
                    print(f"\n===== variant={variant_name} uid={uid} =====")
                    dry_run_preview(exam, region, question, subject, anatomy)
                    continue

                for sample_idx in range(args.samples):
                    key = (variant_name, uid, sample_idx)
                    if key in completed:
                        continue

                    row = await run_one_sample(
                        exam, region, question, subject, anatomy, image_bytes,
                        do_review=args.review, max_retries=args.max_retries,
                        sleep_seconds=args.sleep, impression_max_words=args.impression_max_words,
                    )
                    row.update({
                        'variant': variant_name, 'uid': uid, 'sample_idx': sample_idx,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'rep_prompt_sha256_8': rep_hash, 'usr_prompt_sha256_8': usr_hash,
                        'chk_prompt_sha256_8': chk_hash,
                    })
                    line = json.dumps(row, ensure_ascii=False)
                    if out_f:
                        out_f.write(line + '\n')
                        out_f.flush()
                    else:
                        print(line)
                    status = 'error' if row['error'] else (row['classification'] or {}).get('severity', '?')
                    print(f"[{variant_name}] uid={uid} sample={sample_idx} -> severity={status}", file=sys.stderr)
    finally:
        # Always restore the original prompts regardless of how we exit.
        xrayvision.PROMPTS.clear()
        xrayvision.PROMPTS.update(baseline_prompts)
        if out_f:
            out_f.close()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Replay real exams through candidate prompt variants against the live AI endpoint.")
    parser.add_argument('--variant', action='append', required=True,
                         help="NAME=DIR (repeatable). 'baseline' needs no DIR.")
    parser.add_argument('--uids', nargs='*', default=[])
    parser.add_argument('--uids-file', action='append', default=[],
                         help="JSON array of uid strings, or find_mismatches.py's JSON output "
                              "(repeatable -- e.g. pass the worst-case set and a random sample together)")
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--output', default=None, help="Resumable JSONL output path")
    parser.add_argument('--review', action='store_true',
                         help="Also run the rev_prompt self-review second turn")
    parser.add_argument('--sleep', type=float, default=0.0)
    parser.add_argument('--openai-url', default=None)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true',
                         help="Print assembled prompts, skip the HTTP call entirely")
    parser.add_argument('--images-dir', default=None,
                         help="Override xrayvision.IMAGES_DIR (default: use the app's configured value)")
    parser.add_argument('--impression-max-words', type=int, default=3,
                         help="Words allowed in IMPRESSION before parse_ai_report_text() discards it "
                              "(default: 3, matches production). Raise this to test whether the "
                              "3-word truncation is suppressing real short diagnoses.")
    parser.add_argument('--model', default=None,
                         help="Override xrayvision.MODEL_NAME for this run (e.g. to A/B a different "
                              "backend model against the same prompts/uids). Must already be loaded "
                              "and servable at the configured endpoint.")
    args = parser.parse_args()

    if args.images_dir:
        xrayvision.IMAGES_DIR = args.images_dir
    if args.model:
        xrayvision.MODEL_NAME = args.model

    return asyncio.run(main_async(args))


if __name__ == '__main__':
    sys.exit(main())
