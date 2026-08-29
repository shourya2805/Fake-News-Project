"""Encode titles and audio transcripts with a frozen BERT into cached .npy arrays.

- `bert-base-uncased`, frozen (eval mode, no_grad, no fine-tuning).
- The *same* BERT encodes both streams — the paper shares the weights.
- Titles -> 32 tokens, transcripts -> 64 tokens.
- Token-level last_hidden_state is saved, plus the attention mask.
- Empty transcripts get an all-zero embedding AND an all-zero attention mask;
  BERT is never run on them, so they are never silently padded into real text.
- Re-running is a no-op once `features/` is populated (use --force to rebuild).

    python extract_features.py
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset import FEATURE_DIR, FEATURE_FILES, features_exist, load_records
from model import SPEECH_TEXT_LENGTH, TEXT_DIM, TITLE_LENGTH
from utils import get_device

BERT_NAME = "bert-base-uncased"
BATCH_SIZE = 32


@torch.no_grad()
def encode(texts, tokenizer, bert, max_length, device, desc, skip_empty):
    """Return (embeddings (N, L, 768) float32, masks (N, L) float32)."""
    n = len(texts)
    emb = np.zeros((n, max_length, TEXT_DIM), dtype=np.float32)
    mask = np.zeros((n, max_length), dtype=np.float32)

    todo = [i for i, t in enumerate(texts) if t.strip()] if skip_empty \
        else list(range(n))

    for start in tqdm(range(0, len(todo), BATCH_SIZE), desc=desc, unit="batch"):
        idx = todo[start:start + BATCH_SIZE]
        batch = tokenizer(
            [texts[i] for i in idx],
            padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = bert(**batch).last_hidden_state          # (b, L, 768)
        am = batch["attention_mask"]                   # (b, L)
        # Zero out padded positions so no padding vector reaches the model.
        out = out * am.unsqueeze(-1).to(out.dtype)
        emb[idx] = out.float().cpu().numpy()
        mask[idx] = am.float().cpu().numpy()

    return emb, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    os.makedirs(FEATURE_DIR, exist_ok=True)
    if features_exist(FEATURE_DIR) and not args.force:
        print(f"cache hit: {FEATURE_DIR}/ already populated — nothing to do "
              f"(use --force to rebuild)")
        report(FEATURE_DIR)
        return

    records = load_records()
    device = get_device()
    print(f"records: {len(records)}   device: {device}   model: {BERT_NAME} (frozen)")

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    bert = AutoModel.from_pretrained(BERT_NAME).to(device).eval()
    for p in bert.parameters():
        p.requires_grad = False

    titles = [r["title"] for r in records]
    transcripts = [r["transcript"] for r in records]
    n_empty = sum(1 for t in transcripts if not t.strip())
    print(f"empty transcripts: {n_empty} -> all-zero embedding + all-zero mask "
          f"(BERT not run on them)")

    title_emb, title_mask = encode(
        titles, tokenizer, bert, TITLE_LENGTH, device, "titles   ", skip_empty=False
    )
    tr_emb, tr_mask = encode(
        transcripts, tokenizer, bert, SPEECH_TEXT_LENGTH, device,
        "transcripts", skip_empty=True,
    )

    arrays = {
        "title_emb": title_emb, "title_mask": title_mask,
        "transcript_emb": tr_emb, "transcript_mask": tr_mask,
    }
    for name, arr in arrays.items():
        np.save(os.path.join(FEATURE_DIR, f"{name}.npy"), arr)

    # Contract check: empty transcripts must be all-zero on both arrays.
    empty_idx = [i for i, t in enumerate(transcripts) if not t.strip()]
    assert not tr_emb[empty_idx].any(), "empty transcript got a non-zero embedding"
    assert not tr_mask[empty_idx].any(), "empty transcript got a non-zero mask"
    print(f"\ncontract OK: all {len(empty_idx)} empty transcripts are all-zero "
          f"in both embedding and mask")
    report(FEATURE_DIR)


def report(feature_dir):
    print("\n=== CACHED FEATURES ===")
    print(f"{'file':<26}{'shape':>22}{'dtype':>10}{'size':>12}")
    total = 0
    for name in FEATURE_FILES:
        path = os.path.join(feature_dir, f"{name}.npy")
        arr = np.load(path, mmap_mode="r")
        size = os.path.getsize(path)
        total += size
        print(f"{name + '.npy':<26}{str(tuple(arr.shape)):>22}"
              f"{str(arr.dtype):>10}{size / 1e6:>10.1f} MB")
    print(f"{'TOTAL':<26}{'':>22}{'':>10}{total / 1e6:>10.1f} MB")

    tr_mask = np.load(os.path.join(feature_dir, "transcript_mask.npy"), mmap_mode="r")
    tr_emb = np.load(os.path.join(feature_dir, "transcript_emb.npy"), mmap_mode="r")
    zero_mask = int((np.asarray(tr_mask).sum(axis=1) == 0).sum())
    zero_emb = int((np.abs(np.asarray(tr_emb)).sum(axis=(1, 2)) == 0).sum())
    print(f"\nall-zero transcript masks      : {zero_mask}")
    print(f"all-zero transcript embeddings : {zero_emb}")


if __name__ == "__main__":
    main()
