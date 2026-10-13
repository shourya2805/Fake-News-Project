# Confirmatory test: microsoft/deberta-v3-base vs bert-base-uncased

Pre-registered in commit `56399e3` (`preregistration/deberta_vs_bert.md`). Seeds [47, 48, 49, 50, 51, 52, 53, 54, 55, 56], frozen split `splits/full_split.json`, without_flag, d_k 32, best validation macro F1 checkpoint.

Environment: cuda (Tesla T4), torch 2.11.0+cu130, transformers 5.16.1, CUDA 13.0. Both arms extracted and trained here.

## Primary endpoint

Test macro F1, microsoft/deberta-v3-base − bert-base-uncased, paired two-sided t-test, alpha 0.05, one test.

| Mean diff | 95% CI | t | p | W/L | Per-seed diffs |
|---|---|---|---|---|---|
| +2.87 | [+1.51, +4.23] | 4.77 | 0.0010 | 9/1 | +2.14 +1.45 +1.38 +2.69 +4.99 -0.15 +5.45 +2.80 +2.38 +5.57 |

**Decision: CONFIRMED** (pre-registered prediction: confirmed).

## Descriptive only (no tests)

| Metric | bert-base-uncased | microsoft/deberta-v3-base | Mean diff | W/L |
|---|---|---|---|---|
| Macro F1 (primary) | 74.17 ± 1.52 | 77.04 ± 1.35 | +2.87 | 9/1 |
| Accuracy | 76.88 ± 1.28 | 79.11 ± 1.02 | +2.23 | 9/1 |
| CD F1 | 69.82 ± 2.91 | 70.87 ± 1.67 | +1.05 | 5/5 |
| CE F1 | 65.55 ± 2.41 | 72.26 ± 2.63 | +6.71 | 10/0 |
| SV F1 | 67.99 ± 2.98 | 72.44 ± 3.68 | +4.45 | 8/2 |
| CA F1 | 60.37 ± 3.16 | 64.90 ± 3.88 | +4.53 | 8/2 |

Same 359-sample test set as the exploratory study: this confirms robustness to training randomness and a fresh environment, not generalisation to unseen data.
