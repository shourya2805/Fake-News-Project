# Pre-registration: DeBERTa-v3 vs BERT, confirmatory test

Registered 2026-10-13. This document is committed **before** the experiment
that tests it is run, and before any code exists to run it. The commit that
adds this file is the evidence that the endpoint, the test and the decision
rule were fixed before the data existed. `deberta_vs_bert.json` in this
directory holds the same parameters in machine-readable form, and the runner
reads them from there.

## Why this test exists

In the exploratory comparison (`results/encoder_comparison.md`, seeds 42–46),
frozen `microsoft/deberta-v3-base` beat frozen `bert-base-uncased` on every
seed: macro F1 +3.26 (raw p = 0.011), accuracy +2.95 (raw p = 0.005). With
Holm over the 18 tests of that study, neither survived (p = 0.137 and 0.078).
Re-analysing those data with macro F1 as the only endpoint would clear α, but
that endpoint would have been chosen after seeing the result. This test fixes
the endpoint first and uses fresh seeds.

## Hypothesis

H1: In the Phase 1 configuration, frozen `microsoft/deberta-v3-base` features
give higher test macro F1 than frozen `bert-base-uncased` features.

## Design (all fixed)

- **Arms:** `bert-base-uncased` (baseline) and `microsoft/deberta-v3-base`.
  No other encoders.
- **Seeds:** 47, 48, 49, 50, 51, run once each for both arms (paired by seed).
  None of these seeds has been used for any full-dataset run, by any encoder,
  in any environment.
- **Data:** the frozen split `splits/full_split.json`, sha256
  `1a4adcada317524dbe5c3498cb9a82bdbea96995cadf06b39c209c07950d5ff9`
  (train 1675 / val 359 / test 359).
- **Model and training:** unchanged from the exploratory study and Phase 1:
  `without_flag`, d_k = 32, Adam 1e-3, batch 128, 30 epochs, cross-entropy,
  checkpoint = best validation macro F1, token budgets 32/64.
- **Code:** training and evaluation run through `multiseed.run_study` →
  `train.run_experiment`. The runner refuses to run if `train.py`, `model.py`,
  `dataset.py`, `multiseed.py`, `extract_features.py`, `compare_encoders.py`,
  `determinism.py`, `utils.py`, `modules/` or the split file differ from the
  commit that adds this pre-registration, or if this pre-registration has been
  modified since that commit.
- **Environment:** a single Colab GPU runtime. Both encoders' features are
  extracted fresh in that runtime, and both arms are trained there. The GPU type
  may differ from the exploratory run (Tesla T4); versions and GPU are recorded.

## Primary endpoint and test (the only inferential test)

- **Endpoint:** for each seed, test-set macro F1 (macro over real/fake, all 359
  test samples, best-validation checkpoint) of DeBERTa minus that of BERT.
- **Test:** paired two-sided t-test on the five differences (df = 4),
  α = 0.05. There is one test, so no multiplicity correction is applied.

## Decision rule

- **Confirmed:** mean difference > 0 and p < 0.05.
- **Not confirmed:** p ≥ 0.05, whatever the sign.
- **Contradicted:** mean difference < 0 and p < 0.05.

## Prediction and what falsifies it

We predict **Confirmed**. We expect the effect to be smaller than the
exploratory +3.26, because DeBERTa was picked for this test as the best of
three encoders and its exploratory estimate is probably inflated by that
selection. Our point expectation is +2 to +3 macro-F1 points.

The prediction is falsified by a **Not confirmed** or **Contradicted** outcome.
A Contradicted outcome would also show the exploratory result to be a seed
artefact.

## Power, stated so a null is read correctly

Using the exploratory SD of the paired differences (1.61), a two-sided
paired t-test with n = 5 at α = 0.05 has power 0.92 if the true effect is
+3.26, 0.72 at +2.45, 0.41 at +1.63, and 0.19 at +1.00. A Not confirmed
outcome is therefore weak evidence of no effect if the true effect is small.
We will report it as "not confirmed", not as "no difference".

## Reported descriptively only, with no tests

Accuracy, per-category F1 (CD/CE/SV/CA, using the REAL-plus-category
convention), per-seed values, and wins/losses. None of these is used to decide
the outcome.

## Fixed in advance to rule out flexible analysis

- Exactly five seeds. No seeds added or dropped after looking. A run that
  crashes is rerun with the same seed; it is not replaced.
- Exploratory seeds 42–46 are not pooled with 47–51.
- The decision is reported exactly as the rule above gives it, whichever way it
  falls.

## Limitation of this confirmation

The test set is the same 359 samples as in the exploratory study, and choosing
DeBERTa for confirmation depended on its score on that test set. A Confirmed
outcome therefore shows that the advantage is robust to fresh training
randomness and to a fresh extraction environment. It does **not** show that
the advantage generalises to unseen data. That would need a test split that
has never been scored, and the current split cannot provide one.
