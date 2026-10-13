# Frozen text-encoder comparison

Full dataset (2,393 records), frozen split `splits/full_split.json` (train 1675 / val 359 / test 359). Phase 1 configuration, `without_flag`, d_k 32. Test set, best-validation-macro-F1 checkpoint. Seeds [42, 43, 44, 45, 46], mean ± sample SD over seeds.

Environment: cuda (Tesla T4), torch 2.11.0+cu130, transformers 5.16.1, CUDA 13.0, python 3.13.15. Every row was extracted and trained in this one environment.

| Encoder | Accuracy | Macro F1 | CD F1 | CE F1 | SV F1 | CA F1 |
|---|---|---|---|---|---|---|
| bert-base-uncased (baseline) | 77.44 ± 1.20 | 75.14 ± 1.40 | 71.24 ± 2.19 | 66.92 ± 2.05 | 70.30 ± 3.41 | 62.76 ± 3.31 |
| roberta-base | 75.71 ± 0.91 | 74.09 ± 0.64 | 67.61 ± 0.91 | 70.77 ± 1.48 | 73.98 ± 1.97 | 64.77 ± 2.21 |
| microsoft/deberta-v3-base | 80.39 ± 0.54 | 78.41 ± 0.84 | 73.14 ± 1.81 | 72.57 ± 1.68 | 73.34 ± 2.96 | 65.89 ± 3.02 |
| sentence-transformers/all-mpnet-base-v2 | 70.42 ± 2.58 | 68.87 ± 1.90 | 64.94 ± 1.39 | 65.00 ± 0.82 | 68.24 ± 3.50 | 62.08 ± 5.38 |

Paired t-test across seeds against `bert-base-uncased` (difference = encoder − baseline). p (Holm) adjusts over all 18 tests in this table.

| Encoder | Metric | Mean diff | 95% CI | t | p | p (Holm) | W/L |
|---|---|---|---|---|---|---|---|
| roberta-base | Accuracy | -1.73 | [-3.66, +0.20] | -2.48 | 0.068 | 0.613 | 0/4 |
| roberta-base | Macro F1 | -1.05 | [-2.97, +0.87] | -1.52 | 0.203 | 0.884 | 2/3 |
| roberta-base | CD F1 | -3.64 | [-5.60, -1.67] | -5.13 | 0.007 | 0.095 | 0/5 |
| roberta-base | CE F1 | +3.86 | [+0.64, +7.07] | 3.33 | 0.029 | 0.320 | 5/0 |
| roberta-base | SV F1 | +3.68 | [-2.56, +9.92] | 1.64 | 0.177 | 0.884 | 4/1 |
| roberta-base | CA F1 | +2.01 | [-4.62, +8.64] | 0.84 | 0.447 | 1.000 | 4/1 |
| microsoft/deberta-v3-base | Accuracy | +2.95 | [+1.47, +4.43] | 5.54 | 0.005 | 0.078 | 5/0 |
| microsoft/deberta-v3-base | Macro F1 | +3.26 | [+1.27, +5.26] | 4.54 | 0.011 | 0.137 | 5/0 |
| microsoft/deberta-v3-base | CD F1 | +1.89 | [+0.31, +3.47] | 3.33 | 0.029 | 0.320 | 5/0 |
| microsoft/deberta-v3-base | CE F1 | +5.65 | [+1.89, +9.41] | 4.17 | 0.014 | 0.168 | 5/0 |
| microsoft/deberta-v3-base | SV F1 | +3.04 | [-1.12, +7.19] | 2.03 | 0.112 | 0.785 | 4/1 |
| microsoft/deberta-v3-base | CA F1 | +3.13 | [-0.69, +6.94] | 2.28 | 0.085 | 0.681 | 4/1 |
| sentence-transformers/all-mpnet-base-v2 | Accuracy | -7.02 | [-8.82, -5.22] | -10.80 | 0.000 | 0.007 | 0/5 |
| sentence-transformers/all-mpnet-base-v2 | Macro F1 | -6.27 | [-7.72, -4.82] | -12.02 | 0.000 | 0.005 | 0/5 |
| sentence-transformers/all-mpnet-base-v2 | CD F1 | -6.30 | [-8.88, -3.73] | -6.79 | 0.002 | 0.039 | 0/5 |
| sentence-transformers/all-mpnet-base-v2 | CE F1 | -1.91 | [-4.81, +0.99] | -1.83 | 0.141 | 0.846 | 1/4 |
| sentence-transformers/all-mpnet-base-v2 | SV F1 | -2.06 | [-9.42, +5.30] | -0.78 | 0.480 | 1.000 | 1/4 |
| sentence-transformers/all-mpnet-base-v2 | CA F1 | -0.68 | [-10.12, +8.76] | -0.20 | 0.851 | 1.000 | 1/4 |

Per-category F1 = macro F1 on all real test samples plus only that category's fakes, scored as its own binary problem (the convention of the Phase 1 per-category table).

Token budgets (32 title / 64 transcript tokens) cover slightly different amounts of text under each tokenizer; encoder differences are confounded with that (see DEVIATIONS.md).

Not comparable to the cited Phase 1 figure (77.16 / 74.34), which was produced on a Mac (mps) under an unrecorded torch version.
