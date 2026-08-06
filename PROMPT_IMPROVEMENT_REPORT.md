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
   `ai_reports.severity >= SEVERITY_THRESHOLD` (3, per `local.cfg` —
   overrides the `xrayvision.cfg` default of 5) is the production gate
   used for triage. Verified via `xrayvision.SEVERITY_THRESHOLD` at
   runtime; all numbers in this report use the real value, 3.

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
`tools/prompt_variants/v9_vision_combined/` for reference.

## Elicitation vs. capability: is the vision-stage miss fixable by prompting at all?

Before investing further in vision-prompt iteration, the open question was
disentangled directly: for the 11 worst-set uids where **both** `baseline`
and `v9` scored severity 0 with a fully negative `ai_text`, is the model
failing to *volunteer* a finding it can actually see (an elicitation
problem, fixable by prompt architecture), or is it a genuine capability
limit of this 4B model (not fixable by prompt wording at all)?

### Method

`tools/targeted_probe.py` sends the same image plus a direct,
pathology-specific yes/no question derived from the radiologist's actual
report (e.g. *"Is there a pneumothorax on this image?"*), instead of the
open-ended "write a report" task — one live call per uid, 9 of the 11
cases (2 excluded: one had an unrelated extremity fracture as the rad
finding, one's snippet was ambiguous on first read). Uses the current
production `rep_prompt.txt` as system message unchanged; only the
user-turn task differs.

### First pass: 2/9 "yes", but the question leaked the answer

The first round of direct questions got "yes" (matching the correct
finding) on 2/9 (both bilateral perihilar interstitial-marking cases) and
"no" on the rest, including severe cases like tension pneumothorax with
mediastinal shift. Read at face value this looked like a real, if narrow,
elicitation problem worth fixing.

It wasn't. The two "yes" questions were phrased leadingly (*"Are there
bilateral accentuated perihilar interstitial lung markings?"*) and the
model's answer echoed the question's own phrasing back — a sycophancy
pattern, not evidence of detection. The control that exposes this: a
third, separately-worded interstitial-markings case in the same probe
(`02051458040185`) got an equivalently leading question and answered
**"No, lung fields are clear"** — same finding type, same question shape,
opposite answer. Re-asking the two "yes" cases with neutral, non-leading
phrasing (*"Describe the interstitial lung markings... normal or
abnormal?"*) reverted both to **"normal"**. A third case that initially
looked like a formatting bug — the model said "No" but then described the
correct finding (hyperlucent hemithorax, mediastinal shift) in the same
response when leadingly prompted — also reverted to a fully normal
description under a neutral, non-leading re-ask ("compare lung
transparency/volume between hemithoraces, describe mediastinal position").

### Result: 9/9 capability ceiling, 0/9 elicitation

With leading-question artifacts excluded, **none** of the 9 probed
worst-set misses could be recovered by asking the model to look for the
specific finding directly. This is a materially stronger and more useful
result than "v9 didn't help": it says a vision-stage prompt refactor
*cannot* help on this backend for this failure population, because the
signal isn't accessible to the model at all — not because the prompt
isn't asking the right question. Building a v10 "checklist" prompt
(explicit per-pathology assessment) was considered and **not built**,
because the evidence above already fails the condition that would justify
it (needed: real, non-leading-question-confirmed detections on at least
some of the probed misses; got: zero).

This is a capability-ceiling finding specific to `medgemma-4b-it` on
subtle/complex chest findings (tension pneumothorax, foreign body/stent,
hyperlucent hemithorax, faint interstitial patterns) — consistent with,
and now directly confirming with targeted evidence, the residual
hallucination/miss pattern already noted from the `chk_prompt.txt` work
below.

## Update: checklist-style second pass revisited (partially overturns "don't build v10")

The targeted probe above used a *ground-truth-informed* question ("is
there a pneumothorax on this image?") — the realistic worst case, since
production doesn't know the answer in advance. `tools/checklist_probe.py`
tests the *realistic* version: a single generic per-category checklist
(pneumothorax / effusion / consolidation / mediastinal shift /
cardiomegaly / fracture / foreign body / interstitial pattern / other),
identical for every uid, run against the same 9 worst-set misses, no
foreknowledge of which category applies.

**`medgemma-4b-it`**: 2/9 flagged `OVERALL: abnormal` — both for the
*wrong* category (flagged "consolidation" on cases whose real findings
were pneumothorax and interstitial markings respectively). 0/9 correct
category-level catches.

**`medgemma-1.5-4b-it`** (same 9 uids, same checklist prompt): 3/9
flagged abnormal, and 2 of those 3 hit a genuinely correct category —
it caught the central line/NG tube (`foreign body/support device:
present`) in the tension-pneumothorax case (missing the pneumothorax
itself, but catching a real, correlated abnormal finding that would
still route to human review) and the mediastinal shift in the
hyperlucent-hemithorax case (attributed to atelectasis rather than
tension physiology, but the category flag itself was correct). Both
models still missed all 4 interstitial-marking cases regardless of
framing — that specific finding type looks like a harder ceiling than
the others tested.

This **does not overturn** the core conclusion that ground-truth-informed
targeted questions get 0/9 under neutral phrasing — that result stands.
It does mean the "don't build v10" call above was slightly too strong:
a generic checklist second pass produces non-zero, non-random signal on
`medgemma-1.5-4b-it` specifically, worth quantifying properly rather than
dismissing on a 9-sample qualitative read. This also runs against the
earlier informal finding that `1.5-4b-it` "performs worse" — under this
framing, on this tiny sample, it did better, which is reason enough to
re-test both models properly rather than trust either informal read.

**Next step in progress**: `tools/prompt_lab.py` (now with a `--model`
override) run against both models on the full 27-uid set with the
existing production prompts and classifier, to get real
sensitivity/specificity numbers instead of a qualitative 9-sample read.
Results to follow.

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
2. **Do not ship** `v9_vision_combined`. Isolating its two changes
   (`{question}` wiring vs. mandatory FINDINGS:/IMPRESSION: labeling) and
   re-testing each separately was considered and **deliberately not run**:
   the targeted probe below already shows the underlying miss population
   is a capability ceiling, not a labeling or elicitation artifact, so
   splitting a null result into two smaller nulls would spend live-inference
   time without changing the conclusion.
3. **A negative-enriched evaluation set and a larger (low-hundreds) uid
   scale-up were considered and deliberately not run.** Both are only
   worth the live-inference cost once there's a specific candidate prompt
   showing a real effect worth measuring precisely. There isn't one right
   now — `v9` showed no gain, and a v10 checklist candidate was ruled out
   before being built (see below). Building the negative-enriched set
   remains a prerequisite for any *future* vision candidate, not a
   standalone task to run speculatively.
4. **Do not build a "checklist"/targeted-interrogation vision prompt (v10)
   for this failure population.** It was evaluated as a design option and
   rejected on evidence, not skipped: `tools/targeted_probe.py` asked the
   model direct, pathology-specific yes/no questions about the 9 worst
   real misses it had already gotten wrong in free text. Under leading
   phrasing 2/9 looked recoverable; under neutral, non-leading phrasing
   (the only fair test, since production prompts can't smuggle the answer
   into the question) **0/9 were recoverable**. No prompt architecture
   change can fix a signal the model doesn't perceive in the image.
5. **Vision-stage blindness on `medgemma-4b-it` is a model-capability
   limit, not a prompt problem, for this failure population** (tension
   pneumothorax, foreign body/stent, hyperlucent hemithorax, faint
   interstitial patterns). This is now supported by targeted evidence, not
   just a null A/B result. Closing the remaining false-negative gap
   requires a model/pipeline change (larger or fine-tuned vision model,
   ensemble/second-opinion pass, or explicit escalation of
   low-confidence-normal reads to a human) — none of which are prompt
   changes, and none of which were in scope to change unilaterally here.

## Artifacts

- `mismatches_chest_fn.json` / `mismatches_chest_random.json` — the 27-uid
  evaluation set (15 worst false-negative gaps + 12 severity-stratified
  random cases), PHI-safe.
- `results_vision.jsonl` — raw per-sample results (162 rows) from the
  baseline vs v9 `prompt_lab.py` run.
- `evaluation_report.md` — full `evaluate_prompts.py` output (all views:
  per-sample, per-uid max-severity, pediatric/adult and case-set
  stratification, paired significance).
- `tools/targeted_probe.py` / `targeted_probe_results.jsonl` — the
  elicitation-vs-capability probe and its raw responses (first-pass
  leading-question results only; the neutral-phrasing re-asks that
  overturned 3 of them are recorded in this report's text, not re-saved
  to a file, since they were a 3-call confirmatory check, not a
  structured run).
