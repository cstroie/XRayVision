# Prompt Evaluation Report

Threshold: 3  |  Variants: model_15b, model_4b  |  Total result rows: 540  |  Unique uids: 90


## Overall


### Per-sample view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 270 | 78 | 50 | 10 | 132 | 37.1% [30.9%, 43.9%] | 83.3% [72.0%, 90.7%] | 88.6% | 0.523 | 0.182 |
| model_4b | 270 | 43 | 52 | 8 | 167 | 20.5% [15.6%, 26.4%] | 86.7% [75.8%, 93.1%] | 84.3% | 0.330 | 0.076 |

### Per-uid max-severity view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 90 | 36 | 14 | 6 | 34 | 51.4% [40.0%, 62.8%] | 70.0% [48.1%, 85.5%] | 85.7% | 0.643 | 0.179 |
| model_4b | 90 | 19 | 16 | 4 | 51 | 27.1% [18.1%, 38.5%] | 80.0% [58.4%, 91.9%] | 82.6% | 0.409 | 0.068 |

### Per-sample view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 270 | 123 | 31 | 29 | 87 | 58.6% [51.8%, 65.0%] | 51.7% [39.3%, 63.8%] | 80.9% | 0.680 | 0.086 |
| model_4b | 270 | 43 | 52 | 8 | 167 | 20.5% [15.6%, 26.4%] | 86.7% [75.8%, 93.1%] | 84.3% | 0.330 | 0.076 |

### Per-uid max-severity view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 90 | 57 | 5 | 15 | 13 | 81.4% [70.8%, 88.8%] | 25.0% [11.2%, 46.9%] | 79.2% | 0.803 | 0.067 |
| model_4b | 90 | 19 | 16 | 4 | 51 | 27.1% [18.1%, 38.5%] | 80.0% [58.4%, 91.9%] | 82.6% | 0.409 | 0.068 |

## Paired significance (baseline vs candidate)


**model_4b vs model_15b** (paired on 90 common uids, per-uid max-severity view):
- discordant pairs: b(model_4b right / model_15b wrong)=5, c(model_4b wrong / model_15b right)=20
- test: mcnemar_chi2_corrected, statistic=7.840, p-value=0.0051

## Stratified: pediatric vs adult


### pediatric (n_uids=83) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 249 | 65 | 47 | 10 | 127 | 33.9% [27.5%, 40.8%] | 82.5% [70.6%, 90.2%] | 86.7% | 0.487 | 0.149 |
| model_4b | 249 | 35 | 49 | 8 | 157 | 18.2% [13.4%, 24.3%] | 86.0% [74.7%, 92.7%] | 81.4% | 0.298 | 0.047 |

### adult (n_uids=5) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 15 | 7 | 3 | 0 | 5 | 58.3% [32.0%, 80.7%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.737 | 0.468 |
| model_4b | 15 | 6 | 3 | 0 | 6 | 50.0% [25.4%, 74.6%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.667 | 0.408 |

## Stratified: case set (checks the candidate isn't overfit to only the worst cases)


### case_set=worst (n_uids=15) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 45 | 19 | 0 | 0 | 26 | 42.2% [29.0%, 56.7%] | n/a n/a | 100.0% | 0.594 | 0.000 |
| model_4b | 45 | 5 | 0 | 0 | 40 | 11.1% [4.8%, 23.5%] | n/a n/a | 100.0% | 0.200 | 0.000 |

### case_set=random (n_uids=75) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 225 | 59 | 50 | 10 | 106 | 35.8% [28.8%, 43.3%] | 83.3% [72.0%, 90.7%] | 85.5% | 0.504 | 0.183 |
| model_4b | 225 | 38 | 52 | 8 | 127 | 23.0% [17.3%, 30.0%] | 86.7% [75.8%, 93.1%] | 82.6% | 0.360 | 0.106 |
