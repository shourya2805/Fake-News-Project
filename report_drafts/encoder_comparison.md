# Draft: report section on the frozen text-encoder comparison

Sources: `results/encoder_comparison.{json,md}` (exploratory, commit 791774a) and
`results/confirmatory_deberta_vs_bert.{json,md}` (confirmatory, commit 9360def).
Cross-references use report section names, not numbers, because the report chapters are kept
outside this repository.

---

## Frozen text-encoder comparison

We replaced the frozen `bert-base-uncased` text encoder with three alternatives
and left everything else in the Phase 1 configuration unchanged: the
`without_flag` model, d_k = 32, Adam at 1e-3, batch 128, 30 epochs, the
best-validation-macro-F1 checkpoint, the frozen 70/15/15 split of all 2,393
records, and the authors' 32/64 token budgets. Each encoder was trained with
five seeds (42–46). All four were extracted and trained in a single Colab
environment (Tesla T4, torch 2.11.0+cu130, transformers 5.16.1). We compared
each encoder with BERT using a paired t-test across seeds on six metrics
(accuracy, macro F1, and per-category F1 for CD, CE, SV and CA). We applied a
Holm correction over all 18 tests.

| Encoder | Accuracy | Macro F1 | CD F1 | CE F1 | SV F1 | CA F1 |
|---|---|---|---|---|---|---|
| bert-base-uncased (baseline) | 77.44 ± 1.20 | 75.14 ± 1.40 | 71.24 ± 2.19 | 66.92 ± 2.05 | 70.30 ± 3.41 | 62.76 ± 3.31 |
| roberta-base | 75.71 ± 0.91 | 74.09 ± 0.64 | 67.61 ± 0.91 | 70.77 ± 1.48 | 73.98 ± 1.97 | 64.77 ± 2.21 |
| deberta-v3-base | 80.39 ± 0.54 | 78.41 ± 0.84 | 73.14 ± 1.81 | 72.57 ± 1.68 | 73.34 ± 2.96 | 65.89 ± 3.02 |
| all-mpnet-base-v2 | 70.42 ± 2.58 | 68.87 ± 1.90 | 64.94 ± 1.39 | 65.00 ± 0.82 | 68.24 ± 3.50 | 62.08 ± 5.38 |

Mean ± sample SD over 5 seeds, test set (359 samples).

### MPNet is significantly worse, and this refutes our own hypothesis

The clearest result is negative. `all-mpnet-base-v2` is worse than BERT on
macro F1 by 6.27 points (95% CI [−7.72, −4.82], raw p = 0.0003, Holm
p = 0.005). It loses on all five seeds, and is also worse on accuracy (−7.02,
Holm p = 0.007). Its largest per-category loss is on Contextual Dishonesty:
CD F1 falls by 6.30 points (Holm p = 0.039, 0/5 seeds), the worst CD score of
the four encoders.

This contradicts a hypothesis we put forward in Future work, "Stronger text
encoders": that an encoder trained for semantic similarity would suit CD especially well, since
CD fakes pair genuine footage with a title that misrepresents it. On this
evidence the hypothesis is wrong. The similarity-trained encoder is the worst
of the four on CD, and the effect survives correction.

We have a possible mechanism, which we have not tested. Sentence-transformers
models are trained so that the **pooled** sentence vector (the mean of the
token states) carries the sentence's meaning; the individual token states are
not themselves the training target. FMNVD does not use the pooled vector. It
takes the token-level `last_hidden_state`, projects each token through
Linear + ReLU, and passes the token sequences through co-attention before
pooling. If the useful information in MPNet sits in the linear mean of its
tokens rather than in the tokens themselves, this pipeline discards it. The
result would then reflect a mismatch between MPNet's training objective and
our architecture, not a verdict on similarity training as such. One way to
test this is to feed MPNet's pooled embedding to the model as a single-token
sequence; we have not run that.

### DeBERTa-v3: an exploratory advantage, then a pre-registered confirmation

**Exploratory finding (uncorrected).** In the exploratory study,
`deberta-v3-base` was the strongest encoder. Against BERT it gained 2.95
accuracy points (raw p = 0.005) and 3.26 macro-F1 points (95% CI
[+1.27, +5.26], raw p = 0.011), beating BERT on all five seeds for accuracy,
macro F1, CD F1 (+1.89) and CE F1 (+5.65). It did **not** survive the correction
we had committed to for that study: Holm over the 18 tests gave p = 0.078 for
accuracy and p = 0.137 for macro F1. Within the exploratory study it is
therefore a consistent but not statistically significant improvement, and we
report it that way.

**Declined re-analysis.** A narrower analysis of the same data would have
cleared it. With macro F1 as the only primary endpoint, DeBERTa's raw
p = 0.011 passes at α = 0.05, and still passes (p = 0.021) with Holm across
the three encoders' primary tests alone. We did not adopt that analysis,
because we would have been choosing the endpoint after seeing which one
favoured DeBERTa. This is the same reasoning by which we declined the visual
branch's epoch-30 result in Results, "Contribution of the visual branch".
There, the full model beat text-only by 3.08 macro-F1 points at the epoch-30
checkpoint (p = 0.018, 5/5 seeds), but by only 0.50 (p = 0.616) at the
best-validation checkpoint we had specified in advance, and we reported the
pre-specified one.

**Pre-registration.** Instead of re-analysing, we tested the result again on new
data, fixing the test in advance (see "Pre-registered confirmatory test"). The
test is DeBERTa against BERT only, with test macro F1 as the single primary
endpoint and one paired two-sided t-test at α = 0.05. It uses fresh seeds
(47–56), the same frozen split and checkpoint rule, and features re-extracted
after registration. The decision rule was committed before any confirmatory
code or data existed. We raised the seed count from five to ten for power
before any confirmatory run, in a separate, documented revision.

**Confirmation.** The pre-registered test came back **Confirmed**. DeBERTa beats
BERT on test macro F1 by +2.87 points (95% CI [+1.51, +4.23], t = 4.77, df = 9,
p = 0.0010), winning on 9 of 10 seeds. Accuracy rises by +2.23 (9/10 seeds;
descriptive, not tested).

**The prediction held, inside its own range.** The pre-registration did not
simply predict a positive result. It predicted that the effect would be
**smaller** than the exploratory +3.26, because DeBERTa had been selected as the
best of three encoders and its exploratory estimate was likely inflated by that
selection (the winner's curse). It gave a point expectation of +2 to +3 macro F1.
The confirmed effect, +2.87, shrank from the exploratory estimate by 0.39
points and landed inside the stated range. We read this as evidence that the
discount for selection was about right, not only that the direction was.

The confirmation has the limitation stated in the pre-registration. The test
set is the same 359 samples as in the exploratory study, and choosing DeBERTa
for confirmation depended on its score on that test set. The result shows
that the advantage is robust to fresh training randomness and a fresh
extraction environment. It does not show that the advantage generalises to
unseen data.

### RoBERTa: no overall difference, but a different category profile

`roberta-base` is close to BERT overall (macro F1 −1.05, raw p = 0.203). The
per-category picture is more varied. It is consistently worse on CD
(−3.64, 0/5 seeds, raw p = 0.007, Holm p = 0.095) and consistently better on
CE (+3.86, 5/5 seeds, raw p = 0.029, Holm p = 0.320). It is also better on SV
(+3.68, 4/1, not significant). None of these survives Holm, but the opposite
signs on CD and CE, each consistent across seeds, suggest that encoders differ
in which manipulation types they detect well, not only in overall quality. A
single overall score would hide this.

### Caveats

- **Token budgets.** The 32/64-token budgets cover slightly different amounts
  of text under each tokenizer (about 21–22.5 title words and 50–52 transcript
  words), so encoder differences are confounded with that. More importantly,
  every encoder sees truncated input: 84.1% of transcripts and 46.2% of titles
  are cut (see "The paper's token budgets truncate most of the input"). The
  ranking above is a ranking on truncated input. Before running the comparison
  we predicted that truncation would flatten the differences between encoders.
  That prediction failed (see the outcome note in the same section).
- **Not comparable to the Phase 1 figure.** The BERT row here (77.44 / 75.14,
  mean of five seeds on CUDA) cannot be compared with the cited Phase 1
  reproduction (77.16 / 74.34). That was a single seed on a Mac (mps) under a
  torch version that was not recorded. Each table's baseline is valid only
  within that table.
- **The baseline moves between studies.** BERT's mean test macro F1 was
  75.14 ± 1.40 over seeds 42–46 and 74.17 ± 1.52 over seeds 47–56. The split,
  code, GPU type (Tesla T4) and library versions (torch 2.11.0+cu130,
  transformers 5.16.1) were identical; only the seeds differed. A 0.97-point
  shift in the *baseline* from seed choice alone is as large as some of the
  per-encoder differences in the exploratory table. A single-seed comparison
  could have shown an encoder winning or losing on seed luck alone, which is
  why every comparison here is reported over several seeds.
- **Same test set for every seed.** All seeds are scored on the same 359 test
  samples. The paired tests therefore measure robustness to training
  randomness, not to sampling of the test data.

---

## Pre-registered confirmatory test

Pre-registration: `preregistration/deberta_vs_bert.md`, registered in commit
2c84655 and revised once before any confirmatory run (56399e3, five seeds raised
to ten for power). Results: `results/confirmatory_deberta_vs_bert.md`
(commit 9360def). Both encoders' features were re-extracted after registration,
and both arms were trained in one Colab environment (Tesla T4, torch
2.11.0+cu130, transformers 5.16.1).

| | bert-base-uncased | deberta-v3-base | Mean diff | Seeds won/lost |
|---|---|---|---|---|
| **Macro F1 (primary, tested)** | 74.17 ± 1.52 | 77.04 ± 1.35 | **+2.87** | 9/1 |
| Accuracy | 76.88 ± 1.28 | 79.11 ± 1.02 | +2.23 | 9/1 |
| CD F1 | 69.82 ± 2.91 | 70.87 ± 1.67 | +1.05 | 5/5 |
| CE F1 | 65.55 ± 2.41 | 72.26 ± 2.63 | +6.71 | 10/0 |
| SV F1 | 67.99 ± 2.98 | 72.44 ± 3.68 | +4.45 | 8/2 |
| CA F1 | 60.37 ± 3.16 | 64.90 ± 3.88 | +4.53 | 8/2 |

Mean ± sample SD over seeds 47–56. Primary endpoint: paired two-sided t-test,
+2.87, 95% CI [+1.51, +4.23], p = 0.0010, **Confirmed** under the pre-registered
decision rule. All other rows are descriptive; none was tested.

**CE complicates our explanation (descriptive only).** The largest per-category
gain is on Cherry-picked Editing: +6.71 CE F1, with DeBERTa ahead on all 10
seeds. In the exploratory study it was also the largest (+5.65, 5/5). This sits
awkwardly with how we have explained CE so far. CE is a visual manipulation,
since the footage is cut to change its meaning, and we had argued that
detecting it needs the visual branch. Yet a better text encoder, with no access
to the video, gives the largest gain on exactly this category. We did not test
this and do not claim a mechanism. It may mean that CE clips leave traces in
their titles or transcripts that a stronger encoder can pick up, or it may
reflect how this dataset's CE samples were constructed. Either way, "CE needs
vision" is not supported as stated, and the explanation should be revisited
rather than repeated.

---

## Draft for the conclusion: CD is the project's most robust finding

Contextual Dishonesty, where real footage is paired with a misleading title or
claim, is the category that nothing we tried has improved by more than two
points. At every pre-specified checkpoint, the change in CD F1 from each
intervention stays below +2:

| Intervention | CD F1 change | Seeds won/lost | Checkpoint |
|---|---|---|---|
| Visual branch (full vs text-only, d_k 32, visual split) | +0.80 | 4/1 | best validation |
| Class-balanced loss + val-tuned threshold, text-only model | −1.13 | 0/5 | best validation |
| Class-balanced loss + val-tuned threshold, full model | +0.79 | 4/1 | best validation |
| DeBERTa-v3 encoder, exploratory (seeds 42–46) | +1.89 | 5/0 | best validation |
| DeBERTa-v3 encoder, confirmatory (seeds 47–56) | +1.05 | 5 won / 5 lost | best validation |

The first three rows are on the visual split (REAL+CD+CA, 227 test samples);
the DeBERTa rows are on the full split (359 test samples). Each change is
measured against its own study's baseline.

Where CD was tested (in the visual-branch, Tier 1 and exploratory encoder
studies), no gain survived the study's own correction; the confirmatory CD
figure is descriptive only. The point is the pattern, not any single p-value:
no intervention moves CD by more than two points. The same encoder that improves
macro F1 by a confirmed +2.87 and CE by +6.71 changes CD by +1.05, winning on
exactly half the seeds.

Two qualifications belong next to this claim. First, the one larger CD gain we
observed, +3.41 for the visual branch at the epoch-30 checkpoint (5/5 seeds),
came at a checkpoint we declined to report because it was not specified in
advance; under our own rule it does not count. Second, CD *can* move, but only
downward: RoBERTa costs 3.64 points and MPNet 6.30. Nothing we tried has
improved CD by more than two points.

We think the reason is structural. Detecting CD means checking whether a
title's claim matches the content it accompanies. Every model here encodes the
title and the transcript (and, in the full model, the video) and lets them
interact only through co-attention and a pooled classifier. None computes an
explicit measure of whether title and content agree. A **title–transcript
consistency feature**, such as a similarity or entailment score between the
title and the transcript fed to the classifier, is the remaining untested
hypothesis for moving CD. It is also the hypothesis this work leaves most
directly open. The MPNet result is a warning about how to build it: consistency
probably needs to be computed explicitly, not hoped for from a
similarity-trained encoder's token states.

