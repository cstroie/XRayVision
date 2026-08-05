# Prompt Evaluation Report

Threshold: 3  |  Variants: baseline, v9  |  Total result rows: 162  |  Unique uids: 27


## Overall


### Per-sample view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 81 | 14 | 9 | 0 | 58 | 19.4% [12.0%, 30.0%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.326 | 0.162 |
| v9 | 81 | 13 | 9 | 0 | 59 | 18.1% [10.9%, 28.5%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.306 | 0.155 |

### Per-uid max-severity view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 27 | 6 | 3 | 0 | 18 | 25.0% [12.0%, 44.9%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.400 | 0.189 |
| v9 | 27 | 5 | 3 | 0 | 19 | 20.8% [9.2%, 40.5%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.345 | 0.169 |

### Per-sample view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 81 | 16 | 9 | 0 | 56 | 22.2% [14.2%, 33.1%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.364 | 0.175 |
| v9 | 81 | 13 | 9 | 0 | 59 | 18.1% [10.9%, 28.5%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.306 | 0.155 |

### Per-uid max-severity view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 27 | 7 | 3 | 0 | 17 | 29.2% [14.9%, 49.2%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.452 | 0.209 |
| v9 | 27 | 5 | 3 | 0 | 19 | 20.8% [9.2%, 40.5%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.345 | 0.169 |

## Paired significance (baseline vs candidate)


**baseline vs v9** (paired on 27 common uids, per-uid max-severity view):
- discordant pairs: b(baseline right / v9 wrong)=2, c(baseline wrong / v9 right)=1
- test: exact_binomial_sign, statistic=n/a, p-value=1.0000
- **caveat:** b+c=3 < 25: chi-square approximation unreliable at this n, using an exact binomial sign test instead. Treat any p-value here as indicative only.

## Stratified: pediatric vs adult


### pediatric (n_uids=24) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 72 | 11 | 9 | 0 | 52 | 17.5% [10.0%, 28.6%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.297 | 0.161 |
| v9 | 72 | 10 | 9 | 0 | 53 | 15.9% [8.9%, 26.8%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.274 | 0.152 |

### adult (n_uids=3) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 9 | 3 | 0 | 0 | 6 | 33.3% [12.1%, 64.6%] | n/a n/a | 100.0% | 0.500 | 0.000 |
| v9 | 9 | 3 | 0 | 0 | 6 | 33.3% [12.1%, 64.6%] | n/a n/a | 100.0% | 0.500 | 0.000 |

## Stratified: case set (checks the candidate isn't overfit to only the worst cases)


### case_set=worst (n_uids=15) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 45 | 5 | 0 | 0 | 40 | 11.1% [4.8%, 23.5%] | n/a n/a | 100.0% | 0.200 | 0.000 |
| v9 | 45 | 3 | 0 | 0 | 42 | 6.7% [2.3%, 17.9%] | n/a n/a | 100.0% | 0.125 | 0.000 |

### case_set=random (n_uids=12) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 36 | 9 | 9 | 0 | 18 | 33.3% [18.6%, 52.2%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.500 | 0.333 |
| v9 | 36 | 10 | 9 | 0 | 17 | 37.0% [21.5%, 55.8%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.541 | 0.358 |
