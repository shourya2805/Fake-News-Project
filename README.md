# FMNVD Reproduction — title + audio

Reproduction of the FMNVD baseline from *"FMNV: A Dataset of Media-Published News
Videos for Fake News Detection"* (Wang, Qian & Li, ICIC 2025, arXiv:2504.07687v3),
restricted to the **title + audio-transcript** configuration — the paper's
"w/o Frames" ablation row. No visual branch is built.

Binary task: is a news video real (0) or fake (1). The 1500 fake samples fall into
four manipulation categories that are scored separately: CD, CE, SV, CA.

## Setup

Requires Python 3.9+ (developed on 3.13) and ~1 GB of free disk for the feature
cache. `data/data.json` (2393 records) must already be in place; it comes from the
paper's own repo, <https://raw.githubusercontent.com/DennisIW/FMNV/main/data.json>.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Device selection is automatic: `mps` on Apple Silicon, otherwise `cpu`. CUDA is
never used. `python utils.py` prints the detected device.

## Run

Everything runs from the repo root, in this order:

```bash
python test_smoke.py          # random tensors through the model, shape check
python dataset.py             # split sizes, label distribution, missingness table
python extract_features.py    # frozen BERT -> features/*.npy  (~20 s on MPS)
python train.py               # both runs + all result tables -> results.json
```

`extract_features.py` is cached — a second run is a no-op. Use `--force` to
rebuild. `train.py` accepts `--epochs`, `--seed`, and `--device`.

`python modules/coattention.py` runs the co-attention module standalone and prints
its input/output shapes.

## Layout

```
CLAUDE.md              project spec (architecture, dimensions, data schema)
README.md              this file
DEVIATIONS.md          every choice the paper does not specify
requirements.txt
data/data.json         2393 records (gitignored)
features/              cached BERT embeddings, .npy (gitignored, ~707 MB)
modules/coattention.py bidirectional multi-head co-attention, paper eq. 5-7
model.py               FMNVD, title + audio path
dataset.py             loading, label/category mapping, stratified splits
extract_features.py    frozen bert-base-uncased -> features/
train.py               training loop + overall/per-category evaluation
test_smoke.py          shape smoke test
results.json           written by train.py (gitignored)
```

## Architecture

```
title  (B, 32, 768) --linear_title--> (B, 32, 128)
speech (B, 64, 768) --linear_speech-> (B, 64, 128)
        |
        +-- co_attention_ts (bidirectional, residual + LayerNorm, 4 heads)
        |
  masked mean over tokens -> (B, 128) each
  concat                  -> (B, 2, 128)
  TransformerEncoderLayer(d_model=128, nhead=2)
  mean over sequence      -> (B, 128)
  Linear(128, 2)          -> logits
```

922,754 parameters. `modules/coattention.py` is imported by the authors' released
`FMNVD.py` but was never uploaded to their repo, so it is reconstructed here from
the paper's equations 5-7.

Training: Adam, lr 1e-3, batch 128, 30 epochs, cross-entropy (paper Section 4.2).
Split: 70/15/15, seed 42, stratified on the 5-way category label.

## Results

Test set, macro-averaged, best-val-F1 checkpoint, without the `has_transcript`
feature (see below):

| Category | n | fakes | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| CD (ft) | 224 | 90 | 69.20 | 69.19 | 71.48 | 71.70 |
| CE (fv) | 201 | 67 | 66.17 | 65.64 | 67.71 | 69.78 |
| SV (fa) | 179 | 45 | 69.27 | 68.12 | 72.50 | 79.48 |
| CA (fc) | 157 | 23 | 64.97 | 59.86 | 64.74 | 79.48 |
| **Overall** | 359 | | **77.16** | **74.34** | **76.39** | **73.48** |

Against the paper's "w/o Frames" targets:

| Metric | Paper | Ours | Difference |
|---|---|---|---|
| Accuracy | 72.50 | 77.16 | +6.4% |
| F1 | 71.94 | 74.34 | +3.3% |
| Precision | 72.11 | 76.39 | +5.9% |
| Recall | 73.56 | 73.48 | −0.1% |

All four are inside the ±5–10% reproduction tolerance.

**Averaging matters here.** Metrics are macro-averaged over the real and fake
classes. The test set is 62.7% fake, so a degenerate "predict everything fake"
model scores a fake-class F1 of 77.05 — already above the paper's 71.94, which
rules that convention out. `train.py` prints both conventions; see DEVIATIONS.md.

## The transcript-missingness issue

`audio_transcript` is empty on 917 of 2393 records (38.3%) and the missingness is
strongly class-correlated — SV 0%, CA 72.7%, CD 51.8%, CE 41.3%, real 34.8%. In
principle a model could score on SV by learning "transcript exists → SV" without
reading any text.

Mitigations, all in place:

1. Empty transcripts get an **all-zero embedding and an all-zero attention mask**,
   never silent padding. BERT is not run on them, and the co-attention module
   handles a fully-masked stream by falling back to a pure residual rather than
   producing NaN.
2. Every sample carries a `has_transcript` flag, and `train.py` trains **twice** —
   with and without it as a model input — reporting both.
3. A **transcript-presence-only baseline** (predict fake iff a transcript exists,
   reading no text) is reported per category.

What the runs actually show, which is not what was anticipated:

- Supplying `has_transcript` explicitly *lowers* macro F1 (overall −1.91,
  SV −1.81). It is redundant — an all-zero transcript embedding already tells the
  network that the transcript is missing.
- The presence-only baseline reaches just 52.5% accuracy on the SV subset, near
  chance, because 65% of real samples also have transcripts.

So the correlation is real and is documented, but on this split it is not a large
exploitable shortcut. Per-category SV numbers should still be read with it in mind.

## Deviations

Every choice the paper leaves unspecified — the averaging convention, split
protocol, co-attention internals, checkpoint selection, and the empty-transcript
handling — is recorded in [DEVIATIONS.md](DEVIATIONS.md).
