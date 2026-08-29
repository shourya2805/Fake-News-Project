# Deviations

Every choice made here that the paper (*FMNV: A Dataset of Media-Published News
Videos for Fake News Detection*, Wang, Qian & Li, ICIC 2025) does not specify.
Appended to as the reproduction is built.


## Environment
- **Python 3.13 / venv at `.venv/`** — paper states no interpreter or dependency
  versions. Installed: torch 2.13.0, transformers 5.16.1, numpy 2.5.2,
  scikit-learn 1.9.0.
- **Device: `mps` when available, else `cpu`.** The paper does not state hardware;
  CLAUDE.md forbids CUDA since the team is on Macs.

## Co-attention module (`modules/coattention.py`)
The paper gives only equations 5-7 and the authors never uploaded the file, so
the following implementation details are ours:
- **d_k = d_v = d_model / n_heads = 32** (128 / 4). The paper names `d_k` in the
  scaling term but never gives its value; equal splitting is the standard
  Transformer convention.
- **Per-head projections + an output `Linear(n_heads*d_v, d_model)`** after head
  concatenation, as in Vaswani et al. The paper's eq. 5 writes single-head
  attention only.
- **Dropout (0.1, the paper's stated rate) applied to the attention weights and
  to the output projection.** The paper does not say where dropout sits.
- **Post-norm ordering** — `LayerNorm(x + Attn(...))` — taken literally from
  eq. 6-7 rather than the pre-norm variant.
- **Fully-masked context handling:** when a stream has an all-zero attention mask
  (an empty transcript), softmax over all `-inf` would be NaN. Those rows are
  computed unmasked and the attended output is then zeroed, so the sample passes
  through as `LayerNorm(residual)` only. The paper does not discuss empty inputs.
- **`v_len`/`s_len` accept an int, a `(B,)` length tensor, or a `(B, L)` attention
  mask.** The authors' call site does not reveal which was intended.

## Model (`model.py`)
- **`TransformerEncoderLayer` uses PyTorch defaults** for everything the authors'
  file does not set: `dim_feedforward=2048`, `dropout=0.1`, ReLU, post-norm. The
  2048-wide FFN dominates the parameter count (527K of 923K total). CLAUDE.md's
  "~500K params" estimate is therefore low; the architecture as specified gives
  **922,754** parameters.
- **Masked mean pooling** over the token dimension (padded positions excluded)
  rather than a plain mean. A plain mean would dilute every short title with
  padding, and would make an all-zero transcript contribute a spurious constant.
- **`has_transcript` is injected as a single scalar concatenated to the fused
  128-d vector**, giving `Linear(129, 2)`. The paper has no such feature at all;
  this is the leakage probe required by CLAUDE.md. Concatenating at the last
  layer keeps the rest of the network identical between the two runs.
- **`num_classes = 2` with cross-entropy over logits** (not a 1-logit sigmoid),
  matching the authors' `Linear(dim, 2)`.

## Data splits (`dataset.py`)
- **70 / 15 / 15 train / val / test, `seed = 42`.** The paper specifies no split
  ratio, seed, or protocol.
- **Stratified on the 5-way category label (REAL/CD/CE/SV/CA), not on the binary
  real/fake label.** Per-category evaluation needs each split to contain a
  representative number of CD/CE/SV/CA fakes; binary stratification could leave a
  split with too few CA samples (only 150 exist in total). Resulting per-split
  fake rate is 62.7% in all three splits.
- **Split is a single fixed partition, not k-fold cross-validation.** The paper
  does not say which it used.
- **Records are keyed by their position in `data.json`**, and that index is what
  the cached feature arrays are indexed by, so splits and features stay aligned.
- **`contextual_class` (action/event/object) is ignored** — it is metadata, not a
  training target, per CLAUDE.md.
- **Whitespace-only transcripts count as missing** (`.strip()` before the
  emptiness test), so they get the all-zero embedding + all-zero mask path.

## Feature extraction (`extract_features.py`)
- **`bert-base-uncased`** (the paper says "BERT" with a 768-d hidden size but not
  which checkpoint); one shared frozen instance encodes both title and transcript,
  as the paper states the weights are shared.
- **`last_hidden_state` is cached, not the pooler output** — the model needs
  token-level `(L, 768)` sequences for co-attention.
- **Padding to `max_length` with truncation** (32 / 64 tokens). Padded positions
  are zeroed in the saved embedding as well as excluded by the attention mask, so
  padding cannot leak through pooling.
- **float32 on disk (707 MB total).** float16 would halve it, but exact-value
  caching keeps the training run bit-identical to an uncached run.
- **Extraction batch size 32** — an I/O detail with no effect on results (BERT is
  frozen and each record is encoded independently).
- **Empty transcripts skip BERT entirely** and are written as all-zero embedding +
  all-zero mask, asserted after saving.

## Metrics — averaging convention (the single most consequential choice)
- **Macro-averaged over the real and fake classes is reported as PRIMARY**;
  fake-class-only numbers are reported alongside. The paper says only
  "Accuracy, F1, Precision, Recall".

  Evidence that macro is what the paper used: the test set is 62.7% fake, and a
  degenerate "predict everything fake" model scores **fake-class F1 = 77.05**,
  already *above* the paper's reported 71.94. Any usable model must therefore
  exceed that, so the paper's F1 cannot be fake-class F1. Macro F1 for that same
  degenerate model is 38.53, and the paper's four numbers cluster tightly
  (72.50 / 71.94 / 72.11 / 73.56), which is the signature of macro averaging.

  Under macro, our reproduction lands within +6.4% / -0.1% of every target.
  Under fake-class-only it does not (F1 +15.2%, Recall +19.6%). Both are printed
  so the choice is auditable.
- **"Fake" (`label == "false"`) is the positive class**, per CLAUDE.md.
- **Per-category subset = all real test samples + only that category's fakes**,
  as CLAUDE.md specifies. Category subsets therefore overlap on the real samples
  and are not independent.

## Training (`train.py`)
- **Seed 42** for Python/NumPy/torch/MPS and the train DataLoader shuffle.
- **No weight decay, no LR scheduler, no gradient clipping, no early stopping** —
  the paper specifies none of these (CLAUDE.md's suggested defaults).
- **Best checkpoint selected by validation macro F1**, evaluated after every
  epoch; the full 30 epochs are always run. This matters: both runs overfit
  hard (minimum val loss at epoch 6 / 9, train loss reaching ~0.02 by epoch 30),
  so reporting the epoch-30 model instead would understate the result. The paper
  does not say which epoch's model it reports.
- **`drop_last=False`** — with 1675 train samples at batch 128 the last batch has
  27 samples; dropping it would waste 1.6% of the training data.
- **Model selection and all reported numbers use the test split only once**, after
  the checkpoint is chosen on validation.

## Leakage reporting
- **A transcript-presence-only baseline** (predict fake iff a transcript exists,
  reading no text at all) is computed and reported. It is not requested by the
  paper or CLAUDE.md; it quantifies how much of any per-category score could come
  from the missingness pattern alone.
- **Observed result contradicts the expectation in CLAUDE.md.** Supplying
  `has_transcript` explicitly *lowers* macro F1 (overall -1.91, SV -1.81), and the
  presence-only baseline scores 52.5% accuracy on the SV subset — near chance.
  Reason: an empty transcript is already an all-zero embedding with an all-zero
  mask, so the network can detect missingness from its input whether or not the
  flag is passed; and 65% of *real* samples also have transcripts, so presence is
  not by itself a discriminative shortcut. The correlation is real and documented,
  but it is not a large exploitable leak on this split.
