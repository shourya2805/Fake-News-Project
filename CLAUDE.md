# FMNVD Reproduction — Project Context

## What this project is

Academic capstone reproducing **FMNVD**, the baseline model from the paper
*"FMNV: A Dataset of Media-Published News Videos for Fake News Detection"*
(Wang, Qian & Li — ICIC 2025, arXiv:2504.07687v3).

Binary classification: is a news video real or fake. Fake samples fall into four
manipulation categories that we must score separately.

**Goal for this phase:** reproduce the paper's reported numbers within ±5–10%.

---

## SCOPE — read this before writing any code

We are reproducing the **title + audio** configuration only. This corresponds to
the paper's "w/o Frames" ablation row.

**Target metrics (paper Table 4, "w/o Frames"):**

| Metric | Target |
|---|---|
| Accuracy | 72.50 |
| F1 | 71.94 |
| Precision | 72.11 |
| Recall | 73.56 |

Full-modality numbers (74.17 / 73.76) are **not** our target. Do not build the
visual branch.

**Therefore DO NOT:**
- download or process any video files
- use Whisper (transcripts are already provided — see below)
- use CLIP, 3D ResNeXt-101, ffmpeg, opencv, or decord
- implement `linear_motion`, `linear_clip`, the gated visual fusion, or `co_attention_tv`

If you think we need video features, we don't. Stop and ask.

---

## Data

`data/data.json` — 1.9 MB, from the paper's own public GitHub repo
(`https://raw.githubusercontent.com/DennisIW/FMNV/main/data.json`).

Verified contents: **2,393 records**, matching the paper exactly.

### Schema

Each record is a flat object with these keys:

```json
{
  "video_id": "__HPCmizGfc",
  "class": "fv",
  "contextual_class": "",
  "label": "false",
  "subject": "2024 US presidential election",
  "title": "Sky polling puts Kamala Harris ahead | US Election 2024",
  "audio_transcript": "Kamala Harris is gaining support as ..."
}
```

- `label` is the **string** `"true"` or `"false"`. `"false"` = fake = positive class = 1.
- `class` is `""` for real news.
- `contextual_class` is only populated for `ft` records (`action` / `event` / `object`).
  It is metadata, not a training target.

### Verified distribution

| `label` | count |
|---|---|
| `false` (fake) | 1500 |
| `true` (real) | 893 |

| `class` | count | Paper name | Code |
|---|---|---|---|
| `""` | 893 | Real | — |
| `ft` | 600 | Contextual Dishonesty | CD |
| `fv` | 450 | Cherry-picked Editing | CE |
| `fa` | 300 | Synthetic Voiceover | SV |
| `fc` | 150 | Contrived Absurdity | CA |

The `ft/fv/fa/fc` → `CD/CE/SV/CA` mapping above is confirmed by the counts, which
match the paper's stated 600/450/300/150. Use it.

### CRITICAL — data leakage risk

`title` is present on all 2,393 records. **`audio_transcript` is empty on 917
records (38.3%), and the missingness is strongly class-correlated:**

| Category | Missing transcript |
|---|---|
| SV (`fa`) | **0%** |
| CA (`fc`) | **73%** |
| CD (`ft`) | ~52% |
| CE (`fv`) | 41% |
| Real | 35% |

Every SV sample has a transcript. A model can score well on SV by learning
"transcript exists → SV" without reading any text.

**Required mitigation, non-negotiable:**
1. Add a `has_transcript` boolean feature to every sample in the dataset class.
2. Empty transcripts must be handled explicitly (all-zero embedding + attention
   mask of zeros), never silently padded as if they were real text.
3. Report per-category metrics both with and without the flag so the effect is
   visible.

Do not defer this. It must exist before the first training run produces numbers.

---

## Model architecture

Source: the paper's `FMNVD.py` plus paper Section 3. Reproduce the **title + audio
path only**.

### Dimensions (exact, from the authors' file)

```python
text_dim            = 768    # BERT-base hidden size
title_length        = 32     # tokens
speech_text_length  = 64     # tokens
dim                 = 128    # shared projection dim
dropout             = 0.1
num_heads           = 4      # co-attention heads
```

### Forward pass (our scope)

```
title            (B, 32, 768)  ──linear_title──▶  (B, 32, 128)
audio_transcript (B, 64, 768)  ──linear_speech─▶  (B, 64, 128)

        ↓ co_attention_ts (bidirectional, title ↔ speech)

fea_speech, fea_title  ──mean over token dim──▶ (B, 128) each
        ↓ unsqueeze + concat  →  (B, 2, 128)
        ↓ TransformerEncoderLayer(d_model=128, nhead=2, batch_first=True)
        ↓ mean over sequence dim  →  (B, 128)
        ↓ Linear(128, 2)
     logits (B, 2)
```

Each `linear_*` block is `Linear(in, 128) → ReLU → Dropout(0.1)`.

Note: the original concatenates three modalities into `(B, 3, 128)`. We drop the
visual one, so ours is `(B, 2, 128)`. This is the intended consequence of the
ablation scope, not a bug.

### co_attention module — YOU MUST WRITE THIS

`modules/coattention.py` is **imported by the authors' file but was never
uploaded to their repo.** Nothing runs without it. Build it from the paper's
equations 5–7:

```
Attn(Q, K, V) = softmax(QKᵀ / √d_k) V

F_t' = LayerNorm(F_t + Attn(a→t))     # title attends to audio
F_a' = LayerNorm(F_a + Attn(t→a))     # audio attends to title
```

Signature must match the authors' call site:

```python
co_attention(d_k, d_v, n_heads, dropout, d_model,
             visual_len, sen_len, fea_v, fea_s, pos=False)
# called as: module(v=..., s=..., v_len=..., s_len=...)
# returns:   (v_out, s_out)
```

Here `v` = audio/speech stream, `s` = title stream. Multi-head with 4 heads,
residual + LayerNorm on both directions. `pos=False` means no positional encoding.

---

## Training hyperparameters

All from paper Section 4.2. These are the only ones the paper specifies:

```
optimizer     Adam
learning rate 1e-3
batch size    128
epochs        30
loss          cross-entropy
```

Everything else (weight decay, scheduler, seed, split ratio, early stopping) is
**undocumented in the paper**. Pick sensible defaults, but log every such choice
to `DEVIATIONS.md` — these become the documented-deviation section of the report.

Suggested defaults to record: 70/15/15 stratified split, seed 42, no scheduler,
no weight decay.

---

## Metrics

Report Accuracy, F1, Precision, Recall — computed **overall** and **per category**
(CD, CE, SV, CA), matching the layout of paper Table 3.

Per-category evaluation means: take the real samples plus only that category's
fake samples, and score that subset.

---

## Environment

Team is on **Mac laptops**, mixed Apple Silicon and possibly Intel.

- Device selection must be: `mps` if available, else `cpu`. Never hardcode `cuda`.
- Some ops fall back to CPU under MPS; if training is unstable or errors on MPS,
  fall back to CPU. The model is tiny (~500K params) so CPU is acceptable.
- ~19 steps per epoch at batch 128. 30 epochs is under 600 steps. This trains in
  minutes.
- BERT encoding of 2,393 titles + transcripts is the slowest step, roughly 10–20
  minutes on CPU. **Cache it to `.npy` and never recompute.**

### Dependencies

Keep `requirements.txt` minimal:

```
torch
transformers
numpy
scikit-learn
tqdm
```

---

## Target repo structure

```
fmnvd/
  CLAUDE.md              # this file
  README.md              # setup + run instructions
  DEVIATIONS.md          # every undocumented choice we made
  requirements.txt
  .gitignore             # must exclude data/ and features/
  data/
    data.json
  features/              # cached BERT embeddings (.npy)
  modules/
    __init__.py
    coattention.py       # written by us, from paper eq 5-7
  model.py               # FMNVD, title+audio scope
  dataset.py             # JSON loading, has_transcript flag, splits
  extract_features.py    # BERT encode → features/
  train.py               # train loop + overall/per-category eval
  test_smoke.py          # random tensors through model, shape check
```

---

## Working conventions

- **Build in order, verify each step before moving on.** Do not scaffold
  everything at once and debug at the end.
- After writing `modules/coattention.py` and `model.py`, run `test_smoke.py`
  (random tensors of the right shape) and confirm output is `(B, 2)` before
  touching real data.
- Every file must run standalone from the repo root.
- Commit after each working step. Supervisor tracks git activity — commit
  history is part of the assessment.
- If something in the paper is ambiguous, do not guess silently. Write the
  assumption into `DEVIATIONS.md` and mention it.

## Things that are already decided — do not revisit

- Scope is title + audio. Settled.
- Data source is the repo's `data.json`, not the Baidu Netdisk archive. Settled.
- `has_transcript` flag is required. Settled.
