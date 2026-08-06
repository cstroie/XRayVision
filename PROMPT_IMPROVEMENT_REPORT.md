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
Results below.

## `medgemma-1.5-4b-it` vs `medgemma-4b-it`: real sensitivity/specificity comparison

Both models run through the full production pipeline (current promoted
`rep_prompt.txt`/`chk_prompt.txt`, 3 samples each) against the same
27-uid set (`results_model_compare.jsonl`,
`evaluation_report_model_compare.md`). This directly contradicts an
earlier informal read (from before this investigation) that
`medgemma-1.5-4b-it` "performs worse" — under this measured comparison it
does not, though it trades one failure mode for another.

| View (per-uid max-severity, severity≥3 gate) | `medgemma-4b-it` | `medgemma-1.5-4b-it` |
|---|---|---|
| Sensitivity (95% CI) | 25.0% [12.0%, 44.9%] | **62.5% [42.7%, 78.8%]** |
| Specificity (95% CI) | 100% [43.8%, 100%] | 66.7% [20.8%, 93.9%] |
| Precision | 100% | 93.8% |
| F1 | 0.400 | **0.750** |

Paired significance (27 uids, exact binomial sign test): b(4b right /
1.5 wrong)=1, c(4b wrong / 1.5 right)=9, **p=0.0215** — statistically
meaningful at this n, not just noise, unlike the `v9` comparison earlier
in this report.

### What it fixes

9 of the 27 uids flip from missed to caught, including cases from
several distinct pathology categories: pneumothorax (`04300312170723`),
bilateral perihilar interstitial markings (`03191707160140`,
`02051458040185`), hyperlucent hemithorax with mediastinal shift
(`07181000300192`), the tension-pneumothorax-with-lines case
(`10260711340683`), and others. This lines up with what
`checklist_probe.py` hinted at on a 9-sample qualitative read, now
confirmed quantitatively across the full 27-uid set.

### What it breaks

1 of 3 rad-negative uids in the set becomes a new, non-borderline false
positive: `...06081121550212` (rad severity 1 — tubes/lines present but
clinically unremarkable) gets scored severity 8 by `medgemma-1.5-4b-it`.
Inspecting the raw text shows this is **not** a vision hallucination —
the model correctly identifies a central venous catheter and endotracheal
tube, closely matching the radiologist's own description ("Right
subclavian central venous catheter" vs. rad's "Right jugular central
venous catheter" — same finding category, different vessel). The
`chk_prompt` classifier then scores "catheters present" as severity 8, as
if routine line/tube placement were a major pathological finding, which
it isn't in this clinical context. This is a **classifier calibration
gap specific to incidental device/line findings**, not a vision-stage
capability issue, and it's a different bug class than anything
`chk_prompt.txt` v6-v8 targeted (those were about negation handling, not
severity calibration for non-pathological incidental findings).

Sample-to-sample variance on this case is also large (severity 8, 1, 5
across 3 samples of the same image) — consistent with the
higher-variance, occasionally-hallucinating behavior noted below,
here manifesting as severity instability rather than content
hallucination.

### Reading this result

n=3 rad-negative uids is far too small to trust the 66.7% specificity
point estimate (CI spans 20.8%-93.9%) — this is exactly the
negative-enriched-set gap flagged earlier in this report, now with a
concrete reason it matters: there's a real, identified false-positive
mechanism (device/line severity miscalibration) that a 3-negative-uid
set can only barely detect, not properly bound. The sensitivity gain
(p=0.0215, 9 recovered uids across multiple pathology types) is much
better supported by this sample size than the specificity number is.

**This is not a recommendation to switch production models.** It's
enough evidence to say `medgemma-1.5-4b-it` deserves a properly powered,
negative-enriched re-evaluation before any such decision — the earlier
informal "it's worse" verdict does not survive this measured comparison,
but neither does a clean "switch to it," because the false-positive risk
it introduces is real, specific, and currently unquantified.

## Large-sample follow-up: 90 uids, 20 rad-negative (properly powered)

The negative-enriched re-evaluation flagged above was run:
`mismatches_chest_random_v2.json` (75 new uids, severity-stratified with
20 per bucket including the 0-2/negative bucket, deduplicated against the
existing worst-case set) combined with the original 15-uid worst-case
set, both models run through the full pipeline again
(`results_model_compare_v2.jsonl`, `evaluation_report_model_compare_v2.md`,
90 uids total, 20 genuinely rad-negative).

| View (per-uid max-severity, severity≥3 gate) | `medgemma-4b-it` | `medgemma-1.5-4b-it` |
|---|---|---|
| Sensitivity (95% CI) | 27.1% [18.1%, 38.5%] | **51.4% [40.0%, 62.8%]** |
| Specificity (95% CI) | 80.0% [58.4%, 91.9%] | 70.0% [48.1%, 85.5%] |
| Precision | 82.6% | 85.7% |
| F1 | 0.409 | **0.643** |
| MCC | 0.068 | **0.179** |

Paired significance (90 uids): b(4b right / 1.5 wrong)=5, c(4b wrong /
1.5 right)=20, **McNemar chi-square (corrected) = 7.840, p=0.0051** — this
time with enough discordant pairs (25) for the full chi-square test
itself, not the small-sample sign-test fallback used earlier in this
report.

### This corrects, not just confirms, the earlier read

**`medgemma-4b-it`'s "100% specificity" claimed earlier in this report
was an artifact of n=3 negative uids, not a real property of the model.**
At n=20 its actual specificity is 80%, with 4 real false positives. The
sensitivity gap direction and its statistical reality both hold up at
scale; what changes is the size of the specificity trade-off, which
turns out to be real but far more moderate (80%→70%, a 10-point gap) than
the earlier 3-sample estimate implied (100%→67%, a 33-point gap that
looked far scarier than it turns out to be).

### The false-positive mechanism is different at scale than the small sample suggested

The n=3 false positive (device/line severity miscalibration) does **not**
turn out to be the dominant pattern. Across all 10 false positives found
in this run (4 from `4b-it`, 6 from `1.5-4b-it`), the classifier summary
field shows:

| uid (suffix) | model | model severity | rad severity | classifier summary |
|---|---|---|---|---|
| 05141702220795 | 4b | 4 | 0 | cardiomegaly |
| 03090505000632 | 4b | 7 | 0 | cardiomegaly |
| 03031218570842 | 4b | 4 | 0 | pulmonary edema |
| 03251748580321 | 4b | 7 | 0 | cardiomegaly |
| 03090505000632 | 1.5 | 8 | 0 | cardiomegaly |
| 03031218570842 | 1.5 | 4 | 0 | diaphragm |
| 08110214080352 | 1.5 | 7 | 0 | bowel obstruction |
| 12062315060643 | 1.5 | 8 | 0 | enlarged heart |
| 03251748580321 | 1.5 | 7 | 0 | cardiomegaly |
| 03061431100704 | 1.5 | 5 | 2 | multiple abnormalities |

**Cardiomegaly (over-called heart enlargement) is the dominant false-positive
driver for both models** — 3/4 of `4b-it`'s false positives and 3/6 of
`1.5-4b-it`'s. Two uids (`03090505000632`, `03251748580321`) are false
positives for *both* models on the same cardiomegaly call, suggesting a
shared, systematic weakness (plausibly cardiac-silhouette magnification
on portable/AP chest films, a known radiographic pitfall) rather than a
per-model quirk. This is a more actionable, more generalizable target
than the earlier device/line theory, and it's a `chk_prompt.txt`/
`rep_prompt.txt` calibration question (how confidently to call
cardiomegaly from a single AP view) rather than a vision-capability
question.

### Updated reading

The sensitivity advantage of `medgemma-1.5-4b-it` is now well-supported,
not just suggestive (p=0.0051 at proper n). The specificity cost is real
but moderate, and both models share the same dominant failure mode
(cardiomegaly over-calling) rather than `1.5-4b-it` introducing a new one
— it just also inherits `4b-it`'s existing weakness while fixing more of
its misses. On F1 and MCC, `1.5-4b-it` is the stronger model on this
dataset. See Recommendations for what this changes.

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

## Decisions made (2026-08-06)

- **Production switched to `medgemma-1.5-4b-it`** (`local.cfg`
  `MODEL_NAME`, untracked/environment-specific). Accepted on the strength
  of the large-sample comparison above: the decision-maker's risk
  tolerance is that false positives are acceptable for an early-warning
  system, which resolves the sensitivity/specificity tradeoff in favor of
  the model that catches more real misses.
- **Two-look (generic pass, then a conditional specific/checklist pass if
  the first is normal) architecture: rejected, not built.** The
  checklist probe's marginal, mostly-overlapping-with-the-model-switch
  gains (2/9 partial catches on `1.5-4b-it`, 0/9 clean category matches)
  don't justify 2x inference cost and added complexity, especially once
  false positives are an accepted tradeoff — the entire rationale for a
  conditional second pass (get more sensitivity while limiting FP cost by
  only escalating on a "normal" first read) evaporates once FP cost is
  no longer the constraint. A cheaper lever exists in the same direction
  if more sensitivity is later wanted: lowering `SEVERITY_THRESHOLD`
  trades specificity for sensitivity directly, with a config edit, no new
  inference calls, and no new failure surface.
- **Cardiomegaly-calibration prompt fix: deferred, not done.** Still a
  real, identified, shared weakness in both models (see the large-sample
  false-positive breakdown above), but explicitly lower priority now that
  false positives are an accepted cost of the early-warning design. Worth
  revisiting later for alert-quality/radiologist-trust reasons, not
  urgent.

## Recommendations

1. **Shipped**: the `check_report()` fence-parsing fix, the promoted
   `chk_prompt.txt`, and the FINDINGS:/IMPRESSION: labeling half of
   `rep_prompt.txt` — all real-data validated, all in production.
2. **Do not ship `v9`'s `{question}` wiring.** The targeted probe shows
   the worst-miss population on `medgemma-4b-it` is a capability ceiling,
   not a labeling or elicitation artifact — no prompt change recovers it
   on that model. Isolating `v9`'s two changes further was considered and
   deliberately not run, since this conclusion doesn't depend on which
   half of `v9` is "responsible" for a null result.
3. **The model-capability-ceiling conclusion is `medgemma-4b-it`-specific,
   not universal — it does not extend to `medgemma-1.5-4b-it`.** The
   claim in earlier sections of this report that vision-stage blindness
   requires "a model/pipeline change" turned out to be checkable, not just
   speculative, and checking it reversed the informal prior belief that
   `1.5-4b-it` is worse: on a real, quantified 27-uid comparison it
   recovers 9 of the false-negative misses `4b-it` gets wrong
   (sensitivity 62.5% vs. 25.0%, p=0.0215), at the cost of one new,
   non-borderline false positive traced to a specific, identified
   mechanism (classifier severity miscalibration on incidental line/tube
   findings, not vision hallucination). See the comparison section above.
4. **`medgemma-1.5-4b-it` is now a real, evidence-backed candidate for
   production, pending a decision, not just a research finding.** The
   90-uid, 20-negative follow-up (see "Large-sample follow-up" above)
   confirms the sensitivity gain at proper statistical power (p=0.0051)
   and shows the specificity cost is real but moderate (80%→70%, not the
   33-point gap the 3-negative sample implied) — F1 (0.409→0.643) and MCC
   (0.068→0.179) both favor `1.5-4b-it`. This report does not make the
   switch decision — that's a clinical-risk-tolerance call (is a 10-point
   specificity drop, i.e. more false alarms needing radiologist
   dismissal, an acceptable trade for catching roughly twice as many real
   misses) that belongs to whoever owns that tradeoff, not to prompt
   engineering. The quantitative basis to make that call now exists.
5. **The concrete, contained next fix — independent of the model
   decision — is cardiomegaly-severity calibration in `chk_prompt.txt`
   and/or `rep_prompt.txt`.** The large-sample false-positive breakdown
   shows cardiomegaly over-calling is the dominant false-positive driver
   for *both* models (3/4 of `4b-it`'s FPs, 3/6 of `1.5-4b-it`'s, with 2
   uids failing identically on both), not a `1.5-4b-it`-specific problem
   and not the device/line miscalibration the earlier small sample
   suggested. This is a well-known radiographic pitfall (cardiac
   silhouette magnification on portable/AP films) and a plausible,
   scoped prompt fix: add explicit calibration guidance (e.g. "AP/portable
   views inflate apparent heart size; downgrade cardiomegaly confidence
   unless the ratio is clearly outside normal limits, or supine/AP
   technique is not stated as excluded") — this is a smaller, more
   contained change than a vision-prompt rewrite, and would benefit
   whichever model is running in production.
6. **A "checklist"/targeted-interrogation vision prompt (v10) was
   evaluated on both models and is not currently worth building as a
   standalone architecture change**, independent of the model question
   above: under neutral, non-leading phrasing (the only fair test), 0/9
   ground-truth-informed direct questions were recoverable on `4b-it`,
   and the realistic no-foreknowledge checklist version recovered 0/9
   correct categories on `4b-it` and a partial 2/9 on `1.5-4b-it` (both
   qualitative, n=9). The larger, better-supported lever is the model
   comparison in point 3-4, not a v10 prompt rewrite on either model.

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
- `tools/checklist_probe.py` / `checklist_probe_medgemma-4b-it.jsonl` /
  `checklist_probe_medgemma-1.5-4b-it.jsonl` — the generic (no
  foreknowledge) per-category checklist probe, run against both models.
- `results_model_compare.jsonl` — combined per-sample results
  (`model_4b` + `model_15b` variants, 162 rows) from the initial 27-uid
  `medgemma-4b-it` vs `medgemma-1.5-4b-it` comparison, via
  `prompt_lab.py`'s new `--model` override.
- `evaluation_report_model_compare.md` — full `evaluate_prompts.py`
  output for that initial comparison.
- `mismatches_chest_random_v2.json` — the 75-uid negative-enriched,
  severity-stratified follow-up random set (20 per severity bucket
  0-2/3-5/6-8/9-10), deduplicated against the worst-case set.
- `results_model_compare_v2.jsonl` — combined per-sample results (540
  rows, 90 uids) from the large-sample follow-up comparison.
- `evaluation_report_model_compare_v2.md` — full `evaluate_prompts.py`
  output for the large-sample comparison, the properly-powered numbers
  this report's conclusions are based on.
