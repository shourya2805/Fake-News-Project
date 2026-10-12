# Deviations

Every choice made here that the paper (*FMNV: A Dataset of Media-Published News
Videos for Fake News Detection*, Wang, Qian & Li, ICIC 2025) does not specify.
Appended to as the reproduction is built.


## Environment
- **Python 3.13 / venv at `.venv/`** — paper states no interpreter or dependency
  versions. Installed: torch 2.14.0 (pinned in `requirements.txt` since
  2026-10-12), transformers 5.16.1, numpy 2.5.2, scikit-learn 1.9.0. The torch
  that produced the committed Phase 1 `results.json` was not recorded (see
  "Environment drift").
- **Colab runs use Colab's preinstalled torch**, not the pin, and install only
  `requirements-text.txt`. Exact torch / transformers / CUDA versions are written
  into each Colab results file instead.
- **Device: `cuda` if available, else `mps`, else `cpu`.** The paper does not
  state hardware. CUDA was added so the same code runs on Google Colab; the
  Phase 1 results were produced on `mps`.
- **Determinism (`determinism.py`).** Seeds are pinned for Python, NumPy and
  torch (CPU/CUDA/MPS), `PYTHONHASHSEED` is set, cuDNN is deterministic with
  benchmarking off, and `torch.use_deterministic_algorithms(True,
  warn_only=True)` is on (`warn_only` because MPS lacks deterministic versions
  of some ops). Every DataLoader gets a seeded generator and `worker_init_fn`.
  Two consecutive runs of `train.py` on `mps` produce byte-identical
  `results.json`. Results across devices (CPU vs MPS vs CUDA) are not
  guaranteed to match.

## Co-attention module (`modules/coattention.py`)
The paper gives only equations 5-7 and the authors never uploaded the file, so
the following implementation details are ours:
- **d_k = d_v = 32 (d_model / n_heads). ERROR IN OUR RECORD — corrected.** This
  entry originally said the value was unspecified. That was wrong: the authors'
  released `FMNVD.py` passes `d_k = d_v = dim = 128` to both co-attention modules
  (with `n_heads = 4`). Phase 1 used 32 per head, which does not match the
  authors' code. `model.py` now takes a `d_k` option; both 32 and 128 are run on
  the frozen visual split (see "Visual-split training").
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
- **Frozen to `splits/full_split.json` (2026-10-12), sha256
  `1a4adcada317524dbe5c3498cb9a82bdbea96995cadf06b39c209c07950d5ff9`.** Generated
  by `python dataset.py --freeze-full` from `make_splits(load_records(), seed=42)`,
  i.e. the exact partition Phase 1 was trained on. The file stores both the
  data.json `index` and the `video_id` of every record; `load_full_split` refuses
  to load if either disagrees with data.json or the three splits are not a
  disjoint cover of all 2,393 records. `train.py --subset full` now loads this file
  instead of calling `make_splits`.
  - **Equivalence check, two parts.** (a) The file reproduces the in-code
    partition record for record: train 1675 / val 359 / test 359, identical
    ordered index and `video_id` lists. (b) Under torch 2.14.0, training from the
    file reproduces `results_torch214.json` bit for bit — both runs' full epoch
    history, every test metric, and the transcript-only baseline. Since the file
    changes only how the partition is obtained, (a) + (b) together show the
    committed 77.16 / 74.34 result is unaffected by the switch; that number itself
    is not reproducible in the current environment (see "Environment drift").
  - **Behaviour change:** before, `train.py --subset full --seed N` also re-drew
    the split with seed N. Now the split is fixed and `--seed` changes only model
    init, shuffle order and dropout — the same convention the visual split already
    uses, and what the multi-seed encoder comparison needs.
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

### Per-encoder cache (2026-10-12)
- **`--encoder` selects one of four frozen encoders:** `bert-base-uncased`,
  `roberta-base`, `microsoft/deberta-v3-base`,
  `sentence-transformers/all-mpnet-base-v2`. All four are 768-d, so the
  projection layers are unchanged; `FMNVD` now takes `text_dim` as a required
  argument, read from the cache's `meta.json` rather than a module constant.
- **Each encoder caches to `features/<slug>/`** (`/` → `__`). Before this, a fixed
  `features/*.npy` path meant a new encoder would silently have been served the
  BERT arrays. `meta.json` records encoder, tokenizer class, hidden size, max
  lengths, truncation counts, data.json sha256, device, library versions and a
  sha256 of every array.
- **The Phase 1 BERT cache was moved, not re-extracted**, into
  `features/bert-base-uncased/` (array sha256s identical before and after). Its
  `meta.json` says it predates meta.json: extraction date, device and versions
  were never recorded. Phase 1 still reproduces `results_torch214.json` bit for
  bit from the moved cache.
- **The same extraction path is used for every encoder:** `last_hidden_state`,
  same shared-weights rule, same padding zeroing, same empty-transcript rule
  (never tokenized, all-zero embedding + all-zero mask, asserted after saving;
  917 such rows for every encoder). For `all-mpnet-base-v2` this means its token
  states are used directly, not the mean-pooled sentence embedding it was trained
  to produce — co-attention needs token sequences.
- **Token budgets are a confound in the encoder comparison.** 32 / 64 *tokens*
  cover different amounts of text under different tokenizers (WordPiece,
  byte-level BPE, SentencePiece). Measured on data.json (median words per budget):

  | encoder | titles > 32 tokens | words in 32 | transcripts > 64 tokens | words in 64 |
  |---|---|---|---|---|
  | bert-base-uncased | 1106 / 2393 | 21.5 | 1242 / 1476 | 50.3 |
  | roberta-base | 1113 | 21.2 | 1238 | 51.7 |
  | deberta-v3-base | 1058 | 22.5 | 1236 | 51.9 |
  | all-mpnet-base-v2 | 1106 | 21.5 | 1242 | 50.3 |

  The spread is about one word of title and 1.6 words of transcript, so it is
  small, but it is not zero, and any encoder difference in the comparison is
  confounded with it. The larger fact the table exposes holds for every encoder,
  Phase 1 included: **64 tokens truncates 84% of non-empty transcripts** to
  roughly their first 50 of a median 118 words, and 46% of titles are cut at 32
  tokens. The lengths are the authors' and are kept.

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

## Downloaded video provenance (`download_videos.py`, `verify_downloads.py`)
The paper's own video files are distributed only via Baidu Netdisk. We re-download
from the public YouTube / X posts named by `video_id`, which is only valid if the
public post is the same version the dataset was built from. This was checked.

- **Method.** For each downloaded video, ffmpeg extracts 16 kHz mono audio,
  faster-whisper (`base`, int8 on CPU, beam 5, language auto-detected) transcribes
  it, and the result is compared to that record's `audio_transcript` in
  `data.json` after lower-casing and stripping punctuation. Two scores:
  *token overlap* = multiset intersection / stored-transcript token count (how
  much of the stored transcript is recovered from our audio), and the `difflib`
  `SequenceMatcher` ratio on the token sequences (also penalises length and order
  differences). Records with an empty stored transcript are skipped.
- **Sample.** YouTube IDs only (no cookies needed), records with a non-empty
  transcript only, first N in `data.json` order: 10 each of REAL, CD, CE, SV, and
  10 of the 12 CA YouTube IDs. 50 videos scored. X-hosted videos (1,795 of
  2,393) are untested.
- **Chance baseline.** Each stored transcript scored against the Whisper output
  of every *other* downloaded video: token overlap 0.217-0.254, difflib
  0.047-0.053. Token overlap has a high floor because common words are shared
  between any two English news transcripts.

| Category | n | overlap mean | overlap median | overlap chance | difflib mean | difflib median | difflib chance |
|---|---|---|---|---|---|---|---|
| REAL | 10 | 0.781 | 0.940 | 0.217 | 0.780 | 0.924 | 0.047 |
| CD | 10 | 0.843 | 0.921 | 0.253 | 0.841 | 0.931 | 0.053 |
| CE | 10 | 0.620 | 0.675 | 0.239 | 0.325 | 0.344 | 0.047 |
| SV | 10 | 0.410 | 0.456 | 0.254 | 0.216 | 0.168 | 0.052 |
| CA | 10 | 0.961 | 0.966 | 0.227 | 0.958 | 0.963 | 0.052 |

Per-video numbers are in `verify_results.txt`.

- **Explanation from the paper's generation tools.**
  - **CD** — ERNIE 4.0 rewrote only the title; video and audio are untouched.
    Scores match REAL, as expected. Public videos are the dataset's videos.
  - **CA** — highest agreement of all (0.96 / 0.96): public videos are the
    dataset's videos. Only 12 CA records are on YouTube; the other 138 are on X
    and untested.
  - **SV** — audio replaced with a VITS voiceover of rewritten text. Overlap is
    only ~0.16-0.20 above chance and difflib ~0.17 above chance. Side-by-side
    reading confirms the stored text is a rewrite over the same footage (e.g.
    `2AKezIUdMSc`: public audio "the best female breaker in Australia", stored
    "a world record in underwater basket weaving"). The public post carries the
    original audio; the frames are expected to be the original ones. Several SV
    public videos also transcribe to ~2x the stored token count, so some SV
    clips may additionally be shorter than the public post; frame identity is
    not directly verified by this audio test.
  - **CE** — cut with TransNetV2. Stored transcripts are ~half the length of
    the public audio and paraphrased, giving moderate token overlap (0.62) but
    low difflib (0.33). The public post is the uncut original; the dataset's CE
    frames are a different, shorter edit.
- **Consequence for the visual branch.** Public videos are usable as-is for
  REAL, CD and CA. SV is usable for frames only; its downloaded audio must never
  be used (speech features always come from `data.json`). CE public videos do
  not match the dataset's frames.
  In `data.json` this is REAL 893 + CD 600 + CA 150 = 1,643 records whose public
  video is expected to match (REAL 217 + CD 90 + CA 12 = 319 on YouTube), plus
  SV 300 (82 on YouTube) usable for frames only. CE (450; 197 on YouTube) is not
  reconstructible from public sources.
- **Leakage risk if CE videos are left out.** Giving CE samples an all-zero
  visual feature would make "no video => CE" a perfect shortcut, the same
  failure mode as the transcript missingness. CE must not be handled by simply
  masking out its videos.
- **Measurement artefacts, not provenance failures.** `0zC0K0YZEzg` (REAL) and
  `_qwLHlVjRyw` (CE) have stored transcripts of 10 and 5 tokens. Whisper stopped
  partway through `_HC36OnL-6c` (REAL). `_mJs-T-aH14` (SV) was detected as
  Norwegian Nynorsk and returned 23 tokens. These lower the means but no
  category's conclusion depends on them.

### X-hosted videos
- **Sample.** First X-hosted records (in `data.json` order) with a non-empty
  transcript, downloaded with `--cookies-from-browser chrome` until 10 per class
  succeeded: REAL, CD, CA.
- **Download failures.** 7 of 37 attempts (18.9%): REAL 5/15, CD 1/11, CA 1/11.
  Every failure was yt-dlp "No video could be found in this tweet", i.e. the
  post or its media is gone. No auth or rate-limit failures. Failed posts
  date from 2019-2022 (REAL 2019-2020). Candidates were taken in `data.json`
  order, which sorts X IDs by age, so this sample over-represents older posts
  and the full-download failure rate may differ.
- **Results (chance baseline computed within each host).**

| Category | Host | n | overlap mean | overlap median | overlap chance | difflib mean | difflib median | difflib chance |
|---|---|---|---|---|---|---|---|---|
| REAL | YouTube | 10 | 0.781 | 0.940 | 0.189 | 0.780 | 0.924 | 0.046 |
| REAL | X | 10 | 0.925 | 0.951 | 0.236 | 0.926 | 0.947 | 0.056 |
| CD | YouTube | 10 | 0.843 | 0.921 | 0.221 | 0.841 | 0.931 | 0.053 |
| CD | X | 10 | 0.858 | 0.939 | 0.248 | 0.845 | 0.929 | 0.054 |
| CA | YouTube | 10 | 0.961 | 0.966 | 0.196 | 0.958 | 0.963 | 0.050 |
| CA | X | 10 | 0.606 | 0.739 | 0.248 | 0.494 | 0.528 | 0.041 |

- **CA on X is a measurement limit, not a mismatch.** Six of the ten CA X
  records have stored transcripts of 5-26 tokens (mostly non-speech clips), and
  one is Chinese, which the ASCII-only normaliser reduces to zero tokens.
  Restricted to stored transcripts of 30+ tokens, CA X scores 0.955 overlap /
  0.772 difflib (n=4), REAL X 0.925 / 0.926 (n=10), CD X 0.949 / 0.940 (n=8).
  X posts match the dataset for REAL and CD; CA X is consistent with a match but
  only 4 videos are testable.

### SV trim check
Ratio of Whisper token count on the public video to the stored transcript's
token count. REAL/CD/CA (stored >= 20 tokens) land at median 1.00, range
0.53-1.21.

- SV, sorted: 0.54 (`_mJs-T-aH14`, re-transcribed with language forced to
  English; auto-detection gave 0.17), 0.96, 0.97, 1.16, 1.27, 1.33, 1.35, 1.42,
  2.06, 2.23. Median 1.30.
- **3 of 10 look intact** (0.96-1.16, inside the matched-category range),
  2 look trimmed (~2x), 4 are ambiguous (1.27-1.42), 1 is unmeasurable (public
  audio has little recognisable speech).
- Supporting measure: stored-transcript tokens per second of the *public*
  video's duration is median 1.67 for SV vs 2.27-2.91 for REAL/CD/CA, consistent
  with most SV dataset clips being shorter than the public post.
- Caveat: a VITS voiceover of rewritten text need not match the original's word
  rate, so this is a proxy for clip length, not a frame-level check.

## Visual-branch experiment design
- **CE (450 records) is EXCLUDED from the visual-branch experiment entirely,
  not masked.** Giving CE all-zero visual features would make video-absence a
  near-perfect CE predictor — the same class of shortcut as the transcript gap,
  but stronger. Excluding CE removes the shortcut at the cost of sample size.
- **The visual experiment runs on REAL + CD + CA (1,643 records), plus SV (300)
  only if the trim check supports it, and only on records whose video downloaded
  successfully.** Every sample in that experiment has real video, so there is no
  visual missingness and no mask leakage.
- **SV status: the trim check does not support inclusion** (3 of 10 intact).
  SV is out of the visual experiment unless a per-video length filter is adopted.
- **The text-only reproduction on all 2,393 records is unaffected and remains
  the headline result.**
- **Final visual subset: 1,510 records (REAL 820, CD 552, CA 138)** after 8%
  download attrition, uniform across the three categories (91.8% / 92.0% /
  92.0% retained). Class balance is 45.7% fake against 62.7% in the full
  dataset, because CE and SV are excluded (majority baseline 54.3% vs 62.7%).
  The visual experiment is therefore not comparable to the full reproduction;
  the text-only model is retrained on the identical frozen split
  (`splits/visual_split.json`, seed 42, 70/15/15 stratified on REAL/CD/CA:
  1,057 / 226 / 227) to provide the baseline.

## CLIP frame features (`extract_clip.py`)

- **Frame sampling:** 55 frames per video at indices
  `round(linspace(0, N-1, 55))` over the OpenCV frame count N, decoded
  sequentially (grab every frame, decode only the selected ones). If the stream
  ends before the reported count, indices are recomputed over the frames
  actually decoded.
- **Padding rule:** videos with N < 55 frames get repeated indices from the same
  linspace, i.e. frames are repeated in temporal order (not zero-padded, not
  last-frame padded). Such videos are listed in `features/video/clip_meta.json`
  under `padded`.
- **Preprocessing / encoder:** open_clip `ViT-B-32-quickgelu` with OpenAI
  weights (the QuickGELU variant matches how those weights were trained), frozen,
  eval mode. Standard CLIP transform: BGR→RGB, bicubic resize of the short side
  to 224, center crop 224×224, CLIP mean/std normalisation. Feature is the
  projected `encode_image` output (512-d, not L2-normalised), computed in
  float32 and stored as float16. Output per video `(55, 512)`; consolidated
  `features/video/clip_features.npy` is `(1510, 55, 512)` float16 in frozen-split
  order, with ids in `clip_ids.json`.

## Motion features (`extract_motion.py`)

- **Checkpoint:** `resnext-101-kinetics.pth` from Hara et al. (CVPR 2018,
  3D-ResNets-PyTorch, "old pretrained models" Google Drive folder, file id
  `1cULocPe5YvPGWU4tV5t6fC9rJdGLfkWe`, sha256 `f82e4e51…ffe48326`).
  ResNeXt-101 32×4d, shortcut B, trained on Kinetics-400 with 16-frame 112×112
  clips (epoch 251). Architecture reimplemented in `modules/resnext3d.py` to
  match the authors' `models/resnext.py`; weights load with `strict=True`.
  The paper does not name a checkpoint; this is the standard one from that work.
  The 64-frame variant (`resnext-101-64f-kinetics.pth`) was not used.
- **Clip sampling:** 55 clips of 8 consecutive frames, centred on the same 55
  indices `extract_clip.py` samples (the window is `c-4 … c+3`, shifted inward
  at the video's ends; the frame count N is taken from `clip_meta.json` so both
  streams use identical indices). Videos shorter than 8 frames loop-pad the
  clip, as the authors' `LoopPadding` does.
- **Clip length differs from training:** the network was trained on 16-frame
  clips; we feed 8. Its temporal stride collapses both to a single temporal
  position before pooling, so the 2048-d output is well-defined, but the
  features are not identical to 16-frame features.
- **Feature:** 2048-d global-average-pooled output of `layer4`, before `fc`
  (equivalent to the authors' `--mode feature`); adaptive pooling replaces their
  fixed `AvgPool3d((1, 4, 4))`, which gives the same result at 112×112.
- **Preprocessing (differs from CLIP):** RGB, bilinear resize of the short side
  to 112, center crop 112×112, pixel values kept in 0–255 and mean-subtracted
  with the authors' ActivityNet mean `[114.7748, 107.7354, 99.4750]`, no std
  division — exactly the transform in Hara's `video-classification-3d-cnn-pytorch`
  feature extractor. CLIP instead uses bicubic 224 crops, [0,1] scaling and
  mean/std normalisation. Stored as float16, `(55, 2048)` per video.

## Visual branch (`model.py`, `modalities="full"`)

- **The video stays a 55-step sequence through the gate and `co_attention_tv`;
  it is mean-pooled only afterwards (option a).** The authors' `FMNVD.py`
  mean-pools motion and CLIP features to length 1 *before* the gate, yet builds
  `co_attention_tv` with `visual_len=55`. We follow the paper and the
  `visual_len` setting rather than the pooling line: eq. 1–4 define motion, CLIP
  and the gate per time step `t`, and eq. 8 has the title attend to "visual
  frame features F_v ∈ ℝ^{L_v×d}", a sequence. Pooling first would also make the
  title→visual attention degenerate: with a single key, softmax is always 1, so
  every title token receives the same projected vector and the co-attention
  carries no alignment information. The literal code path (option b) is
  available as a one-line change if an ablation is wanted.
- **Gate is computed per time step:** `α_t = σ(W[F_m,t; F_s,t] + b)` with
  `linear_attn = Linear(256, 1) → Sigmoid`, giving a `(B, 55, 1)` gate.
- **`co_attention_tv` uses the same `d_k` as `co_attention_ts`**, set by the
  model's `d_k` option (default 32, the Phase 1 value; the authors' file uses
  128 — see the corrected co-attention entry above).
- **No visual mask.** Every record in the frozen visual subset has a real video,
  so `co_attention_tv` gets `v_len=None` and the visual stream is a plain mean.
- **Fusion order `(title, speech, visual)` → `(B, 3, 128)`**, taken from the
  authors' `torch.cat((fea_title, fea_speech, fea_visual), 1)`. The title fed to
  `co_attention_tv` is the output of `co_attention_ts` (the "enhanced title" F_t′
  of eq. 8).
- **Classifier is a single `Linear(128, 2)`** as in the authors' code; the
  paper's eq. 10 says "MLP" but does not define one.
- **Parameter counts:** text 922,754; full 1,383,555 (+460,801: `linear_motion`
  262,272, `linear_clip` 65,664, gate 257, `co_attention_tv` 132,608). The
  text-only model is bit-identical to Phase 1 (same init, same outputs, existing
  checkpoints load strictly).

## Visual-split training (`train.py --subset visual`)

- **Four configurations on the frozen visual split** (1,057 / 226 / 227, seed
  42): text-only and full model, each at `d_k = 32` (Phase 1) and `d_k = 128`
  (authors' file). Same seed, same DataLoader generator, same hyperparameters as
  Phase 1 (Adam 1e-3, batch 128, 30 epochs, cross-entropy). Two runs produce
  identical `results_visual.json`.
- **No `has_transcript` input** in these runs; the flag stays a Phase 1 leakage
  probe.
- **Both the best-val-macro-F1 checkpoint and the epoch-30 model are reported.**
  The paper does not say which it reports; with a 226-sample validation set the
  two differ by up to ~5 points, so neither is presented alone.
- **Per-category rows are REAL+CD and REAL+CA** (real samples plus that
  category's fakes, as in Phase 1). A "REAL only" row reports accuracy on real
  samples (specificity); F1/P/R are undefined there with a single class.
- **Majority baseline printed beside every accuracy:** accuracy of always
  predicting real on that exact subset (54.19% overall on the test split,
  59.7% on REAL+CD, 85.4% on REAL+CA).
- **Single seed, 227 test samples:** one sample is 0.44 points and the 95%
  binomial interval on ~75% accuracy is roughly ±5.6 points. Differences
  between the four configurations are within that range.

## Environment drift: torch upgrade breaks bit-reproduction of Phase 1

- Installing `open_clip_torch` on 2026-09-30 upgraded torch to 2.14.0. Rerunning
  the unchanged Phase 1 path (`python train.py`) no longer reproduces the
  committed `results.json` bit for bit: epoch-1 losses differ at ~1e-8, which
  compounds over 30 epochs (without flag: best epoch 21 → 27, test accuracy
  77.16 → 76.60, macro F1 74.34 → 74.23; with flag: accuracy 75.21 → 76.60,
  macro F1 72.43 → 74.13).
- The pre- and post-change `train.py` produce bit-identical epoch-1 results
  under torch 2.14.0, so the drift is from the library, not from the Phase 2
  code. Runs remain deterministic within one environment. `results.json` is
  kept as the recorded Phase 1 result; the torch version used for it was not
  recorded.
- **Two stored Phase 1 results, one cited figure.**

  | File | torch | without flag (acc / macro F1) | with flag (acc / macro F1) |
  |---|---|---|---|
  | `results.json` | not recorded (pre-2026-09-30 environment) | 77.16 / 74.34 | 75.21 / 72.43 |
  | `results_torch214.json` | 2.14.0 | 76.60 / 74.23 | 76.60 / 74.13 |

  Both are seed 42, `mps`, the same split and the same code. **77.16 / 74.34
  remains the cited Phase 1 figure.** It is not reproducible in the current
  environment, and the torch version that produced it was never recorded, so
  it cannot be recovered by reinstalling a known version. This is the
  framework-drift limitation: within one environment runs are bit-identical;
  across torch versions they are not. `results_torch214.json` is the reference
  for any check that has to be re-run today.

## Multi-seed study (`multiseed.py`)

- **Seeds 42–46, all four visual-split configurations, same frozen split** for
  every seed (split file sha256 checked per seed). The seed changes only model
  init, training shuffle order and dropout. Seed 42 reproduces
  `results_visual.json` exactly.
- **mean ± sample std (ddof = 1)** over 5 seeds, for the best-val-F1 checkpoint
  and the epoch-30 model.
- **Paired test for text vs full at d_k = 32:** paired t-test and Wilcoxon
  signed-rank on the per-seed differences, with a t-based 95% CI. With n = 5
  the smallest possible two-sided Wilcoxon p is 0.0625 (all five differences
  the same sign), so it cannot reach 0.05 by construction.
- **What the test does and does not cover:** it measures variation from training
  randomness only. Every seed is scored on the same 227 test samples, so it says
  nothing about how the difference would hold on a different test sample.
  Eight comparisons are reported (2 checkpoints × 4 metrics); no
  multiple-comparison correction is applied, and none survives Bonferroni
  (0.05 / 8 = 0.006).

## Tier 1 enhancement: class-balanced loss + threshold tuning (`enhance.py`)

Architecture unchanged (d_k = 32). Neither technique is in the paper; both are
our additions, evaluated as an ablation on the frozen visual split.

- **Class weights are computed from the training split only:** inverse class
  frequency, normalised so the mean of the two class weights is 1. Train counts
  real/fake 574/483 give weights **real 0.9139, fake 1.0861**. The imbalance in
  the visual subset is mild (45.7% fake), so the weights are close to 1.
  Weighted and unweighted runs use the same code path and the same RNG
  sequence; only the `CrossEntropyLoss` weight differs.
- **Threshold is tuned on the validation split only:** predict fake iff
  `p_fake > t`, sweeping t from 0.05 to 0.95 in steps of 0.01 (91 values) and
  keeping the t with the highest validation macro F1. Ties go to the t closest
  to 0.5. The chosen t is then applied unchanged to the test split; the test
  split is never used for tuning.
- **Checkpoint selection is unchanged** (best validation macro F1 at argmax,
  i.e. t = 0.5). The threshold is tuned afterwards on the same validation set,
  so "threshold only" differs from the baseline in the threshold alone. The
  validation F1 at the tuned threshold is optimistic by construction and is not
  used for any comparison.
- **The four combinations need only two trainings per model and seed**
  (unweighted, weighted); thresholding is post-hoc. The full model was run on
  all four combinations rather than only the "best" one, so no combination was
  selected by looking at test results.
- **Reproduces the multi-seed study:** the baseline cells equal
  `results_multiseed.json` (text_dk32, full_dk32, best checkpoint) exactly for
  every seed.
- **Chosen thresholds vary widely across seeds** (text 0.28–0.68, full
  0.16–0.67), so the tuned threshold is not a stable property of the model on a
  226-sample validation set.
- **Eight cells (four combinations × two models) and three paired comparisons
  are reported without multiple-comparison correction.**

## Output naming and run isolation (2026-10-12)

Before the encoder comparison, every artefact that a non-Phase-1 run could
overwrite was namespaced. None of this changes any result.

- **The encoder slug is in every training tag**, e.g.
  `bert-base-uncased_without_flag`, `bert-base-uncased_visual_full_dk32`. The
  existing local checkpoints were renamed to match (no retraining).
- **`--out` is mandatory except for the canonical runs:** `train.py` with
  `--subset full` or `--subset visual` at the defaults (`bert-base-uncased`,
  seed 42, 30 epochs, default `--modalities`/`--d-k`), `multiseed.py` on the
  visual split at its defaults, and `enhance.py` at its defaults. Only these may
  write `results.json`, `results_visual.json`, `results_multiseed.json` and
  `results_enhance.json`. Anything else exits with an error unless `--out` is given.
- **Checkpoints of any run given `--out` go to `checkpoints/<out-stem>/`**, so only
  a canonical run without `--out` writes the top-level checkpoint files.
- **Per-seed caches are namespaced** as `runs/multiseed/<subset>/<encoder-slug>/`
  and `runs/enhance/<encoder-slug>/` (the existing visual caches were moved there).
  A cached seed is refused if it records a different encoder or different feature
  hashes. Re-summarising the moved caches reproduces `results_multiseed.json` and
  `results_enhance.json` exactly (summary and paired tests).
- **`multiseed.py --subset full`** runs the Phase 1 configuration on the frozen
  full split through `train.run_experiment`; no new training loop. Seed 42 with
  `bert-base-uncased` reproduces the `without_flag` run of `results_torch214.json`
  bit for bit. The full subset reports the best-val-F1 checkpoint only.
- **`enhance.py` still has its own training loop** (`train_one`), a known
  duplication. It was given `--encoder` and the namespacing above, nothing else.

## Enhancement: frozen text-encoder comparison (`compare_encoders.py`)

Enhancement work with its own baseline; it does not touch the Phase 1
reproduction numbers.

- **Four frozen encoders × five seeds (42–46)** on the full 2,393-record dataset
  and the frozen split `splits/full_split.json`: `bert-base-uncased` (baseline),
  `roberta-base`, `microsoft/deberta-v3-base`,
  `sentence-transformers/all-mpnet-base-v2`. Only the cached features change
  between rows; the head, its hyperparameters (Adam 1e-3, batch 128, 30 epochs)
  and the training path (`multiseed.run_study` → `train.run_experiment`) are the
  same for every encoder.
- **`without_flag` configuration only, d_k = 32.** This is the Phase 1 headline
  configuration (the 77.16 / 74.34 row), so the comparison stays in the
  reproduction lineage. The `has_transcript` input is left out because Phase 1
  showed it lowers macro F1 and is redundant with the all-zero transcript
  embedding.
- **Best-validation-macro-F1 checkpoint**, the same selection rule as Phase 1.
- **Reported:** mean ± sample SD (ddof = 1) over seeds for accuracy, macro F1, and
  per-category F1 for CD / CE / SV / CA. Per-category F1 uses the established
  convention: all real test samples plus only that category's fakes, scored as
  its own binary problem with macro F1 (`train.per_category_metrics`). It is not
  class-wise F1 on the full test set.
- **Paired t-test of each encoder against `bert-base-uncased` only**, not all
  pairs: 3 encoders × 6 metrics = 18 tests. Raw p-values are reported with a
  Holm adjustment over all 18 beside them. The Wilcoxon signed-rank test is
  stored in the JSON but cannot reach p < 0.05 with n = 5 (minimum two-sided
  p = 0.0625). As in the multi-seed study, the test covers training randomness
  only: every seed is scored on the same 359 test samples.
- **One environment for every row.** The published table is produced on Colab.
  All four caches, BERT included, are extracted fresh there; the August BERT cache
  is not used. `compare_encoders.py` compares each cache's `meta.json`
  (torch / transformers / tokenizers versions and device type) with the running
  environment and refuses to write a table on any mismatch.
  `--allow-mixed-env` exists only for local smoke tests and stamps the output
  "not for publication".
- **Colab uses its preinstalled CUDA torch**, plus `requirements-text.txt`
  (transformers pinned to 5.16.1, matching the local environment). The exact
  versions, CUDA/cuDNN, GPU name and git commit are recorded in
  `results/encoder_comparison.json`. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set
  before Python starts so deterministic cuBLAS is in effect from the first
  matmul.
- **The BERT row is not comparable to 77.16 / 74.34.** That figure came from a
  Mac (mps) under an unrecorded torch version. Within the table, BERT is
  re-run under exactly the same conditions as the other encoders, and that
  re-run is the only valid baseline for it.
- **Token budgets confound the comparison**; see "Per-encoder cache".
