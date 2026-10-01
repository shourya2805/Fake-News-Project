# FMNVD Reproduction — Project Context

## What this project is

Academic capstone reproducing **FMNVD**, the baseline model from the paper
*"FMNV: A Dataset of Media-Published News Videos for Fake News Detection"*
(Wang, Qian & Li — ICIC 2025, arXiv:2504.07687v3).

Binary classification: is a news video real or fake. Fake samples fall into four
manipulation categories that we must score separately.

**Goal:** reproduce the paper's reported numbers within ±5–10%.

---

## SCOPE

**Phase 1 — title + audio (DONE).** The paper's "w/o Frames" ablation is
reproduced, deterministic, and committed. It stays in the repo as the ablation
comparison and must keep running unchanged.

**Phase 2 — visual branch (CURRENT).** We are now building the visual half of
FMNVD (the top half of paper Fig. 5) to reach the full-modality model:
video download, frame sampling, CLIP frame features, 3D ResNeXt-101 motion
features, `linear_clip`, `linear_motion`, the gated visual fusion, and
`co_attention_tv`. With the visual stream added, the fusion sequence goes from
`(B, 2, 128)` back to the authors' `(B, 3, 128)`.

### Targets

**Full model (paper Table 4, full FMNVD) — primary target for Phase 2:**

| Metric | Target |
|---|---|
| Accuracy | 74.17 |
| F1 | 73.76 |
| Precision | 74.17 |
| Recall | 75.78 |

**Text-only (paper Table 4, "w/o Frames") — kept as the ablation comparison:**

| Metric | Target |
|---|---|
| Accuracy | 72.50 |
| F1 | 71.94 |
| Precision | 72.11 |
| Recall | 73.56 |

Both rows must be reported side by side in the final results so that the
contribution of the visual branch is visible.

### Still do NOT

- use Whisper as a transcription step. Model inputs always come from the
  `audio_transcript` field in `data.json`. Whisper (faster-whisper, base) is a
  dataset verification tool only, used by `verify_downloads.py` to check that a
  downloaded video matches the version the dataset was built from.
- use the audio track of any downloaded video as a model input
- commit videos or extracted features to git

### Video provenance (see DEVIATIONS.md, "Downloaded video provenance")

Public videos match the dataset for REAL, CD and CA. SV public videos carry the
original audio, not the dataset's VITS voiceover. CE public videos are the
uncut originals, not the dataset's TransNetV2-cut clips. Do not start the full
download or build visual features until how CE (and SV) are handled is decided.

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
- `video_id` is the YouTube id used by the video download step.

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

The same rule applies to videos: some YouTube ids will be unavailable. Missing
videos must get an explicit all-zero visual feature + zero mask, their count must
be reported per category, and the missingness must be checked for class
correlation the same way transcripts were.

---

## Model architecture

Source: the paper's `FMNVD.py` plus paper Section 3.

### Dimensions (exact, from the authors' file)

```python
text_dim            = 768    # BERT-base hidden size
title_length        = 32     # tokens
speech_text_length  = 64     # tokens
dim                 = 128    # shared projection dim
dropout             = 0.1
num_heads           = 4      # co-attention heads
```

Visual dimensions (frame count, CLIP and motion feature sizes) are taken from the
authors' `FMNVD.py` when the visual branch is implemented. Anything not fixed by
that file or the paper goes into `DEVIATIONS.md`.

### Forward pass — text + audio path (done)

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

### Forward pass — full model (Phase 2)

The visual stream (CLIP frame features via `linear_clip`, motion features via
`linear_motion`, combined by the gated visual fusion) is co-attended with the
title via `co_attention_tv`, mean-pooled to `(B, 128)`, and concatenated with the
title and speech vectors into `(B, 3, 128)` before the same
`TransformerEncoderLayer` → mean → `Linear(128, 2)` head. The text-only model
must remain available as a separate mode for the ablation row.

### co_attention module

`modules/coattention.py` was written by us from the paper's equations 5–7
(the authors never uploaded theirs):

```
Attn(Q, K, V) = softmax(QKᵀ / √d_k) V

F_t' = LayerNorm(F_t + Attn(a→t))     # title attends to audio
F_a' = LayerNorm(F_a + Attn(t→a))     # audio attends to title
```

```python
co_attention(d_k, d_v, n_heads, dropout, d_model,
             visual_len, sen_len, fea_v, fea_s, pos=False)
# called as: module(v=..., s=..., v_len=..., s_len=...)
# returns:   (v_out, s_out)
```

`co_attention_ts` passes speech as `v` and title as `s`. `co_attention_tv` reuses
the same module with the visual stream as `v` and the title as `s`.

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

Current choices: 70/15/15 stratified split, seed 42, no scheduler, no weight
decay, best checkpoint by validation macro F1.

### Determinism

Runs must be bit-reproducible. `determinism.py` pins Python, NumPy and torch
(CPU / CUDA / MPS) seeds, sets cuDNN deterministic mode and
`torch.use_deterministic_algorithms(True, warn_only=True)`, and every DataLoader
gets a seeded generator and `worker_init_fn`. Two runs of `train.py` must produce
identical results; check this after any change to the training path.

---

## Metrics

Report Accuracy, F1, Precision, Recall — computed **overall** and **per category**
(CD, CE, SV, CA), matching the layout of paper Table 3.

Per-category evaluation means: take the real samples plus only that category's
fake samples, and score that subset.

---

## Environment

Team is on **Mac laptops** (mixed Apple Silicon and possibly Intel), and heavy
video feature extraction may run on **Google Colab**.

- Device selection (`utils.get_device()`): `cuda` if available, else `mps`, else
  `cpu`. The same code must run unchanged on a Mac and on Colab.
- Some ops fall back to CPU under MPS; if training is unstable or errors on MPS,
  fall back to CPU.
- BERT features are cached to `features/*.npy` and never recomputed.
- Video features are cached to `features/video/` and never recomputed.

### Dependencies

```
torch
transformers
numpy
scikit-learn
tqdm
yt-dlp
opencv-python
open_clip_torch
faster-whisper
```

`ffmpeg` must be on PATH (`brew install ffmpeg` on macOS; preinstalled on Colab).

---

## Target repo structure

```
fmnvd/
  CLAUDE.md                  # this file
  README.md                  # setup + run instructions
  DEVIATIONS.md              # every undocumented choice we made
  requirements.txt
  .gitignore                 # excludes data/, features/, videos/, features/video/
  data/
    data.json
  features/                  # cached BERT embeddings (.npy)
    video/                   # cached CLIP frame + motion features
  videos/                    # downloaded YouTube videos (never committed)
  modules/
    __init__.py
    coattention.py           # written by us, from paper eq 5-7
  utils.py                   # get_device(): cuda > mps > cpu
  determinism.py             # seeds, deterministic flags, DataLoader seeding
  model.py                   # FMNVD
  dataset.py                 # JSON loading, has_transcript flag, splits
  extract_features.py        # BERT encode → features/
  train.py                   # train loop + overall/per-category eval
  test_smoke.py              # random tensors through model, shape check
  multiseed.py               # 5-seed study on the frozen visual split → results_multiseed.json

  # video pipeline (Phase 2), one job per file, in run order
  download_videos.py         # yt-dlp: video_id → videos/
  verify_downloads.py        # whisper vs data.json transcript, per category
  verify_results.txt         # output table of verify_downloads.py
  extract_clip.py            # videos/ → 55 sampled frames → CLIP features in features/video/
  extract_motion.py          # videos/ → 55 aligned 8-frame clips → 3D ResNeXt-101 features in features/video/
  modules/resnext3d.py       # 3D ResNeXt-101 (Hara et al.) definition + checkpoint loader
  weights/                   # pretrained checkpoints (never committed)
```

File names in the video pipeline are the plan; update this section if they
change.

---

## Working conventions

- **Build in order, verify each step before moving on.** Do not scaffold
  everything at once and debug at the end.
- **Separate clean files, one job per file.**
- **No comments in code** written from Phase 2 onward.
- **No shell scripts** — Python entry points only.
- **Every script is resumable** and skips work already on disk.
- When an existing file is changed, show the full file.
- After any model change, run `test_smoke.py` and confirm output is `(B, 2)`
  before touching real data.
- Every file must run standalone from the repo root.
- Commit after each working step. Supervisor tracks git activity — commit
  history is part of the assessment.
- If something in the paper is ambiguous, do not guess silently. Write the
  assumption into `DEVIATIONS.md` and mention it.

## Output conventions

- When you change CLAUDE.md, DEVIATIONS.md, README.md or any other .md file,
  do NOT print the file contents. Make the edit and report it in one line:
  which file, what changed.
- When you change a .py file, show the full file.
- Never print a file just to confirm you wrote it.
- Do not echo results.json, coverage.json, failed.csv or any other generated
  file. Print the summary table only.

## Things that are already decided — do not revisit

- Phase 1 (title + audio) is complete and is the ablation baseline. Settled.
- Phase 2 builds the visual branch toward the full-model target. Settled.
- Data source is the repo's `data.json`, not the Baidu Netdisk archive. Settled.
- `has_transcript` flag is required. Settled.
