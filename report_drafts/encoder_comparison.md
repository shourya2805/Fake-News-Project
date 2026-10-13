# Draft: report section on the frozen text-encoder comparison

Source: `results/encoder_comparison.{json,md}` (commit 791774a). Placeholders in
[square brackets] mark cross-references to report sections that are not in this
repository.

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

This contradicts a hypothesis we put forward in [Section X, Conclusion]: that
an encoder trained for semantic similarity would suit CD especially well, since
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

### DeBERTa-v3 is better on every seed, but we do not claim significance

`deberta-v3-base` is the strongest encoder. Against BERT it gains 2.95
accuracy points (raw p = 0.005) and 3.26 macro-F1 points (95% CI
[+1.27, +5.26], raw p = 0.011). The gains are consistent: it beats BERT on all
five seeds for accuracy, macro F1, CD F1 (+1.89) and CE F1 (+5.65). The
macro-F1 gain is 2.3 times BERT's seed SD, and the accuracy gain is 2.5 times.

It does not survive the correction we committed to in advance. Holm over the
18 tests gives p = 0.078 for accuracy and p = 0.137 for macro F1. We therefore
report it as a consistent but **not statistically significant** improvement.

A narrower analysis would clear it. If macro F1 had been declared as the only
primary endpoint, DeBERTa's raw p = 0.011 would pass at α = 0.05, and it
would still pass (p = 0.021) with Holm applied across the three encoders'
primary tests alone. We do not adopt that analysis, because we would be
choosing the endpoint after seeing which one favoured DeBERTa. This is the
same reasoning by which we declined the visual branch's epoch-30 result in
[Section Y, visual multi-seed study]. There, the full model beat text-only by
3.08 macro-F1 points at the epoch-30 checkpoint (p = 0.018, 5/5 seeds), but by
only 0.50 (p = 0.616) at the best-validation checkpoint we had specified in
advance, and we reported the pre-specified one. We apply the same rule here
even though it costs us a positive result.

Instead of re-analysing these data, we pre-registered a confirmatory
experiment on fresh seeds with macro F1 as its single primary endpoint
(Section [Z]).

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
  are cut (Section [truncation]). The ranking above is a ranking on truncated
  input. Before running the comparison we predicted that truncation would
  flatten the differences between encoders. That prediction failed (Section
  [truncation], outcome).
- **Not comparable to the Phase 1 figure.** The BERT row here (77.44 / 75.14,
  mean of five seeds on CUDA) cannot be compared with the cited Phase 1
  reproduction (77.16 / 74.34). That was a single seed on a Mac (mps) under a
  torch version that was not recorded. Each table's baseline is valid only
  within that table.
- **Same test set for every seed.** All seeds are scored on the same 359 test
  samples. The paired tests therefore measure robustness to training
  randomness, not to sampling of the test data.
