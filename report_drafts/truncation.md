# Draft — report subsection and limitation on input truncation

Written 2026-10-13, before any result of the frozen-encoder comparison
(`results/encoder_comparison.*`) existed. The limitation is stated in advance on
purpose, as a prediction rather than an explanation after the fact.

---

## Subsection: The paper's token budgets truncate most of the input

The authors' released `FMNVD.py` fixes the text inputs at 32 tokens for the title
and 64 tokens for the audio transcript. We measured what these budgets keep from
the 2,393 records of the paper's own `data.json`, counting tokens with each
model's tokenizer (including special tokens, which use part of the budget).

**Transcripts.** 1,476 records have a non-empty transcript. Their median length is
118 words (149 `bert-base-uncased` tokens). **1,242 of them (84.1%) exceed 64
tokens.** About 50 words fit in that budget, so the median transcript reaches
the model with roughly 58% of its words removed, and only its opening survives.

**Titles.** The median title is 19 words (30 tokens), close to the limit.
**1,106 of 2,393 titles (46.2%) exceed 32 tokens** and lose their end. About 21
words fit.

**Other tokenizers.** These figures barely change across the four encoders we
compare. For RoBERTa, DeBERTa-v3 and MPNet, 1,058–1,113 titles and 1,236–1,242
transcripts are cut, and the budgets hold about 21–22.5 title words and
50–52 transcript words.

All of our results so far, the Phase 1 reproduction (77.16 accuracy /
74.34 macro F1) included, were obtained under this truncation, and so were the
paper's own. The model has never seen most of the transcript text. The paper
does not discuss the choice of lengths.

## Limitation: the encoder comparison may measure truncation, not encoder quality

Every encoder in the comparison sees the same truncated input: the first ~50
words of each transcript and the first ~21 words of each title. Differences
between encoders can only come from how well each one represents those
fragments. If the information that separates real from fake news sits mostly
in text that is cut off, every encoder is equally limited, and the comparison
will show small differences regardless of how good the encoders are.

**Prediction, made before seeing the results.** We expect the four frozen
encoders to land within about one seed standard deviation of each other on
macro F1, with no difference surviving the Holm correction. If that happens,
it should not be read as "the encoders are equivalent for this task". The
measured truncation is a sufficient explanation on its own, and this
experiment cannot tell the two explanations apart. A clear difference between
encoders, on the other hand, would show that the encoder matters even on
truncated input, but would still say nothing about input the model never
receives.

The comparison holds the token budget fixed to stay faithful to the authors'
specification. The direct test of this limitation is to repeat the comparison
with a longer transcript budget (for example 256 tokens, which would cover the
median transcript entirely). We have not run that test.

---

## Outcome, added 2026-10-13: the prediction failed

The comparison ran after the prediction above was committed (prediction: commit
b066470, replayed as 8c32373; results: commit 791774a). **Both halves of the
prediction were wrong.**

- **"Within about one seed standard deviation of each other on macro F1."**
  Wrong by a wide margin. Macro F1 ranges from 68.87 (MPNet) to 78.41 (DeBERTa-v3),
  a spread of 9.54 points, which is 6.8 times the baseline's seed SD of 1.40.
  Accuracy ranges from 70.42 to 80.39, close to 10 points.
- **"No difference surviving the Holm correction."** Wrong. MPNet is worse than
  BERT after Holm adjustment over all 18 tests on accuracy (−7.02, p = 0.007),
  macro F1 (−6.27, p = 0.005) and CD F1 (−6.30, p = 0.039), and loses on all 5
  seeds for each.

**What the prediction got wrong.** It assumed that, with every encoder limited
to the same ~50 transcript words and ~21 title words, the truncation would
dominate and flatten the differences between encoders. It did not. The
encoders represent the same truncated fragments very differently, enough to
move macro F1 by almost ten points. The pre-registered alternative applies:
"the encoder matters even on truncated input".

**What still stands.** The truncation itself is measured and does not change:
84.1% of transcripts and 46.2% of titles are cut. All four encoders were
compared at the authors' 32/64 budgets, so the ranking holds for truncated
input only. Whether it holds when the model sees whole transcripts is untested;
the 256-token comparison is still the direct test. The limitation should
therefore be reworded from "the comparison may measure truncation rather than
encoder quality" to "the comparison measures encoder quality on truncated
input, which need not match encoder quality on full input".
