#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elicitation-vs-capability probe: for uids where the free-text vision report
is 100% negative sentences, ask the model a direct, pathology-specific
yes/no question about the same image instead of "write a report."

If the model says yes to a pointed question but never volunteers it in
free text -> elicitation problem, prompt refactor toward targeted
interrogation is justified.
If it says no even when asked directly -> capability ceiling on this
model, no prompt wording fixes it.

Read-only: uses only prepare_exam_data/prepare_ai_request_data/send_to_openai
(bare HTTP call), never send_exam_to_openai/check_ai_report_and_update.
"""
import asyncio
import json
import os
import sys

import aiohttp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrayvision import db_get_exams, prepare_exam_data, prepare_ai_request_data, send_to_openai, USER_AGENT
from prompt_lab import probe_active_openai_url

# uid suffix -> targeted yes/no question, derived from mismatches_chest_fn.json rad_text_en.
# Resolved to full uids at runtime against mismatches_chest_fn.json so a
# hand-transcribed prefix can never silently point at the wrong exam.
_QUESTIONS_BY_SUFFIX = {
    "03190851280575": "Is there a pneumothorax on this image?",
    "03191707160140": "Are there prominent interstitial markings in the lower lung fields bilaterally on this image?",
    "03212110190140": "Are there peribronchovascular perihilar interstitial markings on this image?",
    "04300312170723": "Is there a pneumothorax on this image?",
    "07181000300192": "Is the entire hemithorax hyperlucent with mediastinal shift on this image (tension pneumothorax)?",
    "08202346260949": "Is there a metallic stent or foreign body projected over the abdomen/mesentery on this image?",
    "10260711340683": "Is there a right-sided tension pneumothorax with lung collapse and mediastinal shift on this image?",
    "01291930030785": "Are there bilateral accentuated perihilar interstitial lung markings on this image?",
    "02051458040185": "Are there bilateral perihilar interstitial lung markings on this image?",
}


def resolve_cases():
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mismatches_chest_fn.json'), encoding='utf-8') as f:
        known_uids = [c['uid'] for c in json.load(f)]
    cases = {}
    for suffix, question in _QUESTIONS_BY_SUFFIX.items():
        matches = [u for u in known_uids if u.endswith(suffix)]
        assert len(matches) == 1, f"suffix {suffix} matched {len(matches)} uids, expected exactly 1"
        cases[matches[0]] = question
    return cases


CASES = resolve_cases()

async def probe_one(session, uid, question):
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

    # Uses the current production REP_PROMPT (radiologist role) as system
    # message, unchanged -- only the user-turn task is replaced with a
    # direct pathology-specific question instead of "write a report".
    user_prompt = f"{question} Answer yes or no first, then briefly describe what you see."
    headers, payload = prepare_ai_request_data(user_prompt, image_bytes)

    resp = await send_to_openai(session, headers, payload)
    if not resp:
        return {"uid": uid, "error": "no response"}
    try:
        text = resp['choices'][0]['message']['content']
    except (KeyError, IndexError):
        return {"uid": uid, "error": "malformed response", "raw": resp}
    return {"uid": uid, "question": question, "answer": text}


async def main():
    active = await probe_active_openai_url()
    if not active:
        print("No active AI endpoint reachable", file=sys.stderr)
        sys.exit(1)
    results = []
    async with aiohttp.ClientSession(headers={'User-Agent': USER_AGENT}) as session:
        for uid, question in CASES.items():
            r = await probe_one(session, uid, question)
            print(f"[{uid[-14:]}] {r.get('answer', r.get('error'))[:200] if r.get('answer') else r.get('error')}", file=sys.stderr)
            results.append(r)
    with open('targeted_probe_results.jsonl', 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    asyncio.run(main())
