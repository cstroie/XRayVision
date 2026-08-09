#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cheap gut-check for a "generic pass, then specific checklist pass if normal"
vision-stage architecture. Unlike targeted_probe.py (which asked about the
one pathology the radiologist actually found -- ground truth leaked into
the question), this asks a single GENERIC per-category checklist, the same
for every uid, with no foreknowledge of which finding to look for. This is
the realistic version of a "specific" second pass: production doesn't know
the answer in advance either.

Tests the same 9 worst-set uids as targeted_probe.py (all scored severity 0
by both baseline and v9 in the original vision-stage run) for direct
comparison against that probe's result (0/9 recoverable under neutral
phrasing).

Usage:
    python tools/checklist_probe.py --model medgemma-4b-it
    python tools/checklist_probe.py --model medgemma-1.5-4b-it

Read-only: only prepare_exam_data/prepare_ai_request_data/send_to_llm
(bare HTTP call), never send_exam_to_llm/check_ai_report_and_update.
"""
import argparse
import asyncio
import json
import os
import sys

import aiohttp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import xrayvision
from xrayvision import db_get_exams, prepare_exam_data, prepare_ai_request_data, send_to_llm, USER_AGENT
from prompt_lab import probe_active_llm_backend
from targeted_probe import CASES  # same 9 uids as the ground-truth-informed probe

CHECKLIST_PROMPT = """For EACH of the following categories, state "present" or "absent" based only on this image, one line per category, in this exact format "Category: present/absent - brief note":
- Pneumothorax
- Pleural effusion
- Consolidation / infiltrate
- Mediastinal shift
- Cardiomegaly
- Rib or bone fracture
- Foreign body / support device / stent / catheter
- Interstitial pattern (accentuated or prominent markings)
- Other acute abnormality (specify if present)

Then on a final line write "OVERALL: normal" or "OVERALL: abnormal - <finding>"."""


async def probe_one(session, uid):
    exams, _ = db_get_exams(uid=uid, limit=1)
    if not exams:
        return {"uid": uid, "error": "not found in DB"}
    exam = exams[0]
    try:
        region, _q, subject, anatomy, image_bytes = prepare_exam_data(exam)
    except FileNotFoundError:
        return {"uid": uid, "error": "image not found"}
    if region is None:
        return {"uid": uid, "error": "region not supported"}

    exam_backend = xrayvision.TASK_ACTIVE['exam']
    headers, payload = prepare_ai_request_data(CHECKLIST_PROMPT, image_bytes, exam_backend['model'], exam_backend['api_key'])
    resp = await send_to_llm(session, headers, payload, url=exam_backend['url'])
    if not resp:
        return {"uid": uid, "error": "no response"}
    try:
        text = resp['choices'][0]['message']['content']
    except (KeyError, IndexError):
        return {"uid": uid, "error": "malformed response", "raw": resp}
    return {"uid": uid, "answer": text}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Model name to request (must already be loaded/servable at the configured endpoint)")
    parser.add_argument('--output', default=None, help="Output jsonl path (default: checklist_probe_<model>.jsonl)")
    args = parser.parse_args()

    default_backend = xrayvision.LLM_BACKENDS[xrayvision.LLM_BACKEND_NAMES[0]]
    default_backend['models']['exam'] = args.model
    xrayvision.MODEL_NAME = args.model
    output = args.output or f"checklist_probe_{args.model.replace('/', '_')}.jsonl"

    active = await probe_active_llm_backend()
    if not active:
        print("No active AI endpoint reachable", file=sys.stderr)
        sys.exit(1)

    results = []
    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
        for uid in CASES:
            r = await probe_one(session, uid)
            print(f"[{uid[-14:]}] {(r.get('answer') or r.get('error'))[:300]}", file=sys.stderr)
            print("---", file=sys.stderr)
            results.append(r)

    with open(output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} results to {output}", file=sys.stderr)


if __name__ == '__main__':
    asyncio.run(main())
