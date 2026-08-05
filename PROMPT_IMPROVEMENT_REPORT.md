# Prompt Improvement: Reducing False Negatives in Chest X-Ray AI Reporting

This document summarizes the investigation into false-negative AI reports on
chest exams (branch `xray-prompt-improvements-dls3gr`), what was shipped,
what was tested and not promoted, and what remains open.

## Background

The pipeline has two AI stages per exam:

1. **Vision stage** (`rep_prompt.txt`/`usr_prompt.txt`) — MedGemma
   (`medgemma-4b-it`, served locally via LM Studio) reads the image and
   produces a free-text report (`ai_text`).
2. **Classifier stage** (`chk_prompt.txt`) — the same model reads `ai_text`
   and outputs `{"pathologic": ..., "severity": ..., "summary": ...}`.
   `ai_reports.severity >= SEVERITY_THRESHOLD` (5) is the production gate
   used for triage.

A false negative occurs when the radiologist's report is positive but the
pipeline's final severity stays below threshold. These fall into two
failure modes with different fixes:

- **Classifier miscounts a real finding as normal** — the vision-stage text
  actually describes the pathology, but the classifier scores it 0. Fixable
  by improving `chk_prompt.txt` alone (text-only, no images needed).
- **Vision stage never describes the finding at all** — `ai_text` is 100%
  negative sentences, no signal for any classifier to catch. Requires
  fixing `rep_prompt.txt`/`usr_prompt.txt` and re-running images through
  the model.

## What was shipped

### 1. `check_report()` bug fix (`xrayvision.py`)

`check_report()` silently discarded classifier responses that weren't
wrapped in ` ```json``` ` fences, even though `chk_prompt.txt` only asks for
"valid JSON," not fenced JSON. This left `ai_reports.severity` stuck at -1
(never assessed) for affected exams indefinitely — a distinct and more
severe failure than a false negative, since the exam never got a
classification at all. Fixed to parse unfenced JSON.

### 2. `chk_prompt.txt` (production, promoted from `v8_chk_refined`)

Real bug: a report with one positive finding followed by several negative
sentences (e.g. *"Consolidation... No pleural effusion... normal
limits."*) was scored `severity=0/normal` **100% of the time** in the
mismatch sample. The prompt's existing rule ("normal statements don't
negate pathological ones") wasn't enough on its own — the model needed a
worked example to generalize it.

Validated via three rounds of iteration (v6 → v7 → v8) using
`tools/check_report_probe.py` against real `ai_text` from 27 mismatch
cases (text-only replay, no images, no live vision calls needed):

- 2 real cases converted from below-threshold (missed) to above-threshold
  (correctly flagged)
- 0 new above-threshold misclassifications introduced in the validated
  sample
- v7 fixed the negation bug but introduced new hallucination-driven
  false positives on vague/terse text; v8 added guardrails for that
  without regressing the negation fix

### 3. Refactor: `parse_ai_report_text()`

Extracted the FINDINGS:/IMPRESSION: split logic out of
`send_exam_to_openai()` into a standalone, testable function, with
`impression_max_words` (default 3, unchanged behavior) as an explicit
parameter — a lever for future testing of whether 3-word IMPRESSION
truncation drops real short diagnoses.

### 4. New read-only investigation tools (`tools/`)

All verified at import time (assert) to never call
`send_exam_to_openai`/`check_ai_report_and_update`/`check_rad_report_and_update`
directly against the database — analysis only, no side effects on
production data:

- **`find_mismatches.py`** — ranks exams by rad-vs-AI severity gap.
  `--mode fn|fp|all|random|unassessed`, PHI-safe output (uid, age, sex,
  clinical text only — no name/CNP).
- **`prompt_lab.py`** — replays real exams (by uid) through candidate
  prompt variants against the live vision + classifier endpoint.
  `--variant NAME=DIR` (repeatable), `--dry-run` for a no-network preview.
- **`check_report_probe.py`** — classifier-only probe, text-in/JSON-out,
  no images or vision calls needed.
- **`evaluate_prompts.py`** — sensitivity/specificity/precision/F1/MCC with
  Wilson confidence intervals, McNemar/exact-binomial paired significance
  testing, pediatric/adult and case-set stratification. Stdlib only.

## Vision-stage candidate: tested, NOT promoted

`tools/prompt_variants/v9_vision_combined/` was built to address the
majority of remaining false negatives — cases where `ai_text` never
mentions the finding at all, so no classifier fix can help. It combines
two changes to `rep_prompt.txt`/`usr_prompt.txt`:

1. Wires the previously-dead `{question}` placeholder into
   `usr_prompt.txt` (region-specific prompt from `xrayvision.cfg`
   `[questions]`, e.g. for chest: *"Are there any lung consolidations,
   infiltrates, opacities, pleural effusion, pneumothorax or
   pneumoperitoneum"*).
2. Makes `rep_prompt.txt`'s FINDINGS:/IMPRESSION: labeling mandatory
   (previously self-contradictory: said "no labels" but its own examples
   used them, and the downstream parser depends on the labels being
   present), and encourages naming a suspected diagnosis even when hedged.

### Method

27 real chest uids — 15 worst rad-vs-AI severity gaps (`--mode fn`) plus 12
severity-stratified random cases — replayed through the **live**
`medgemma-4b-it` endpoint, 3 samples per uid per variant (162 vision +
classification calls total, 0 errors), then scored with
`evaluate_prompts.py` against the promoted `chk_prompt.txt` (both variants
use the same, current classifier — only the vision-stage prompts differ).

### Mechanism-level result: the labeling fix worked

FINDINGS:/IMPRESSION: parse compliance across all 81 samples per variant:

| Variant  | Findings+Impression parsed  |
|----------|-----------------------------|
| baseline |  3 / 81 (3.7%)              |
| v9       | 71 / 81 (87.7%)             |

And it's not a no-op: every one of the 15 worst-set uids produced
genuinely different report text between baseline and v9 (0/15 identical).
Both changes — the `{question}` wiring and the mandatory labeling — were
verifiably exercised on real exams, not silently skipped.

### Outcome-level result: no detectable sensitivity gain

| View (per-uid max-severity, severity≥3 gate) | baseline | v9 |
|---|---|---|
| Sensitivity (95% CI) | 25.0% [12.0%, 44.9%] | 20.8% [9.2%, 40.5%] |
| Specificity (95% CI) | 100% [43.8%, 100%] | 100% [43.8%, 100%] |
| F1 | 0.400 | 0.345 |

Paired significance (27 common uids, exact binomial sign test — used
instead of McNemar because discordant pairs b+c=3 is below the n=25
threshold for the chi-square approximation): **p = 1.0**. With only 3
discordant uids (baseline-right/v9-wrong=2, baseline-wrong/v9-right=1),
this sample cannot distinguish "v9 is worse" from "no difference" — the
correct reading is **no evidence of benefit at this sample size**, not
evidence of harm.

On the worst-case set specifically (the 15 uids v9 was designed to fix):
baseline flags 3/15 at least once across its 3 samples, v9 flags 1/15.
This set was selected as cases the *original* pipeline missed, so
baseline is structurally advantaged here — some of its 3/15 hits are
likely the already-shipped `chk_prompt.txt` negation fix recovering
signal that was already present in `ai_text`, not a vision-stage
improvement. v9 still shows no gain despite that headwind, which is a
real (if statistically underpowered) strike against it.

One severe case (bilateral pneumothorax with pneumomediastinum) was
caught by both variants, with v9 scoring it higher-severity (9 vs 7).
Two other worst-set cases baseline caught (severity 4) that v9 missed
entirely (severity 0) — consistent with the labeling change reshuffling
which findings get emphasized rather than reliably fixing vision-stage
blindness.

**Caveat on specificity**: computed on only 3 rad-negative uids (CI
[43.8%, 100%]) — this case set is worst-case/random-stratified for
catching false negatives, not enriched for negatives, so it cannot
meaningfully bound the false-positive risk a more sensitive vision prompt
would introduce. Any future vision candidate needs a negative-enriched
evaluation set before promotion, not just this false-negative-focused one.

### Conclusion

`v9_vision_combined` is **not promoted**. It measurably fixes a real
mechanical bug (missing/inconsistent FINDINGS:/IMPRESSION: labels feeding
a parser that depends on them) but does not measurably improve detection
of the false-negative cases it targeted, at n=27. It's left in
`tools/prompt_variants/v9_vision_combined/` for reference; the labeling
fix specifically may be worth re-testing in isolation from the `{question}`
change, since combining both makes it impossible to attribute the (null)
outcome to either one.

## Known residual limitation (not fully solved)

The backend (`medgemma-4b-it`, confirmed via `/v1/models`) occasionally
hallucinates a diagnosis or fails to produce valid classifier JSON on
vague/terse/content-free vision-stage text (e.g. *"Obvious acute
abnormalities are identified."* with no specific finding named). Three
rounds of `chk_prompt.txt` wording fixes (v6/v7/v8) each fixed some
instances and occasionally introduced new ones elsewhere. In the validated
sample this stays below `SEVERITY_THRESHOLD` so it doesn't flip real
classifications, but it's a bounded, not eliminated, risk — most likely a
capability limit of this specific 4B model rather than something prompt
wording alone resolves. `medgemma-1.5-4b-it` was tried on the same
endpoint and performed worse; not recommended as a substitute.

## Recommendations

1. **Ship**: the `check_report()` fence-parsing fix and the promoted
   `chk_prompt.txt` — both are shipped already, real-data validated, and
   contained to the classifier text-in/JSON-out step (no vision risk).
2. **Do not ship** `v9_vision_combined` as-is. If vision-stage work
   continues, isolate the FINDINGS:/IMPRESSION: labeling fix from the
   `{question}` change and re-test each independently.
3. **Before any future vision-prompt promotion**, build a
   negative-enriched evaluation set (rad-confirmed-normal exams) alongside
   the false-negative set — the current sets can measure sensitivity gains
   but not the false-positive cost.
4. **Statistical power**: 27 uids is enough to validate a targeted
   classifier-text fix (where the specific failure mode is known and
   checked directly) but not enough to detect realistic effect sizes in
   a vision-stage sensitivity/specificity comparison. A future vision
   candidate evaluation should budget for a substantially larger uid set
   (low hundreds) if a statistically decisive answer is required — that's
   a scope/cost decision for the user, not something to run inline.
5. **Vision-stage blindness remains the dominant open problem.** Most
   false negatives are still cases where `ai_text` contains no signal at
   all for any classifier to act on. This is a model-capability question
   more than a prompt-wording one, and the cheapest next lever (prompt
   iteration) has now been tried without a measurable win.

## Artifacts

- `mismatches_chest_fn.json` / `mismatches_chest_random.json` — the 27-uid
  evaluation set (15 worst false-negative gaps + 12 severity-stratified
  random cases), PHI-safe.
- `results_vision.jsonl` — raw per-sample results (162 rows) from the
  baseline vs v9 `prompt_lab.py` run.
- `evaluation_report.md` — full `evaluate_prompts.py` output (all views:
  per-sample, per-uid max-severity, pediatric/adult and case-set
  stratification, paired significance).
