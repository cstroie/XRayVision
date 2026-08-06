# Prompt Evaluation Report

Threshold: 3  |  Variants: model_15b, model_4b  |  Total result rows: 162  |  Unique uids: 27


## Overall


### Per-sample view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 81 | 30 | 7 | 2 | 42 | 41.7% [31.0%, 53.2%] | 77.8% [45.3%, 93.7%] | 93.8% | 0.577 | 0.125 |
| model_4b | 81 | 14 | 9 | 0 | 58 | 19.4% [12.0%, 30.0%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.326 | 0.162 |

### Per-uid max-severity view -- severity >= threshold (primary, matches production gate)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 27 | 15 | 2 | 1 | 9 | 62.5% [42.7%, 78.8%] | 66.7% [20.8%, 93.9%] | 93.8% | 0.750 | 0.187 |
| model_4b | 27 | 6 | 3 | 0 | 18 | 25.0% [12.0%, 44.9%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.400 | 0.189 |

### Per-sample view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 81 | 42 | 6 | 3 | 30 | 58.3% [46.8%, 69.0%] | 66.7% [35.4%, 87.9%] | 93.3% | 0.718 | 0.158 |
| model_4b | 81 | 16 | 9 | 0 | 56 | 22.2% [14.2%, 33.1%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.364 | 0.175 |

### Per-uid max-severity view -- pathologic == yes (secondary view)

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 27 | 20 | 2 | 1 | 4 | 83.3% [64.1%, 93.3%] | 66.7% [20.8%, 93.9%] | 95.2% | 0.889 | 0.378 |
| model_4b | 27 | 7 | 3 | 0 | 17 | 29.2% [14.9%, 49.2%] | 100.0% [43.8%, 100.0%] | 100.0% | 0.452 | 0.209 |

## Paired significance (baseline vs candidate)


**model_4b vs model_15b** (paired on 27 common uids, per-uid max-severity view):
- discordant pairs: b(model_4b right / model_15b wrong)=1, c(model_4b wrong / model_15b right)=9
- test: exact_binomial_sign, statistic=n/a, p-value=0.0215
- **caveat:** b+c=10 < 25: chi-square approximation unreliable at this n, using an exact binomial sign test instead. Treat any p-value here as indicative only.

## Stratified: pediatric vs adult


### pediatric (n_uids=24) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 72 | 29 | 7 | 2 | 34 | 46.0% [34.3%, 58.2%] | 77.8% [45.3%, 93.7%] | 93.5% | 0.617 | 0.159 |
| model_4b | 72 | 11 | 9 | 0 | 52 | 17.5% [10.0%, 28.6%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.297 | 0.161 |

### adult (n_uids=3) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 9 | 1 | 0 | 0 | 8 | 11.1% [2.0%, 43.5%] | n/a n/a | 100.0% | 0.200 | 0.000 |
| model_4b | 9 | 3 | 0 | 0 | 6 | 33.3% [12.1%, 64.6%] | n/a n/a | 100.0% | 0.500 | 0.000 |

## Stratified: case set (checks the candidate isn't overfit to only the worst cases)


### case_set=worst (n_uids=15) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 45 | 17 | 0 | 0 | 28 | 37.8% [25.1%, 52.4%] | n/a n/a | 100.0% | 0.548 | 0.000 |
| model_4b | 45 | 5 | 0 | 0 | 40 | 11.1% [4.8%, 23.5%] | n/a n/a | 100.0% | 0.200 | 0.000 |

### case_set=random (n_uids=12) -- per-sample, severity-based

| Variant | n | TP | TN | FP | FN | Sensitivity (95% CI) | Specificity (95% CI) | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| model_15b | 36 | 13 | 7 | 2 | 14 | 48.1% [30.7%, 66.0%] | 77.8% [45.3%, 93.7%] | 86.7% | 0.619 | 0.228 |
| model_4b | 36 | 9 | 9 | 0 | 18 | 33.3% [18.6%, 52.2%] | 100.0% [70.1%, 100.0%] | 100.0% | 0.500 | 0.333 |
