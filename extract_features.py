"""Encode titles and audio transcripts with a frozen text encoder into cached .npy arrays.

- `--encoder` picks one of `dataset.ENCODERS` (default `bert-base-uncased`),
  frozen (eval mode, no_grad, no fine-tuning).
- Each encoder caches to its own directory, `features/<encoder-slug>/`, with a
  `meta.json` recording encoder, tokenizer, max lengths and library versions.
- The *same* encoder encodes both streams — the paper shares the weights.
- Titles -> 32 tokens, transcripts -> 64 tokens.
- Token-level last_hidden_state is saved, plus the attention mask.
- Empty transcripts get an all-zero embedding AND an all-zero attention mask;
  the encoder is never run on them, so they are never silently padded into real text.
- Re-running is a no-op once the encoder's directory is populated (use --force to rebuild).

    python extract_features.py --encoder roberta-base
"""

import argparse
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset import (DATA_PATH, DEFAULT_ENCODER, ENCODERS, FEATURE_FILES, FEATURE_META,
                     feature_dir_for, features_exist, file_sha256, load_records)
from model import SPEECH_TEXT_LENGTH, TITLE_LENGTH
from utils import get_device

BATCH_SIZE = 32
VERSIONED_PACKAGES = ["torch", "transformers", "tokenizers", "sentencepiece", "protobuf", "numpy"]


@torch.no_grad()
def encode(texts, tokenizer, encoder, max_length, device, desc, skip_empty):
    """Return (embeddings (N, L, H) float32, masks (N, L) float32)."""
    n = len(texts)
    emb = np.zeros((n, max_length, encoder.config.hidden_size), dtype=np.float32)
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
        out = encoder(**batch).last_hidden_state       # (b, L, H)
        am = batch["attention_mask"]                   # (b, L)
        # Zero out padded positions so no padding vector reaches the model.
        out = out * am.unsqueeze(-1).to(out.dtype)
        emb[idx] = out.float().cpu().numpy()
        mask[idx] = am.float().cpu().numpy()

    return emb, mask


def package_versions():
    out = {"python": platform.python_version()}
    for name in VERSIONED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    out["cuda"] = torch.version.cuda
    return out


def write_meta(feature_dir, meta):
    meta["files_sha256"] = {
        f"{name}.npy": file_sha256(feature_dir / f"{name}.npy") for name in FEATURE_FILES
    }
    (feature_dir / FEATURE_META).write_text(json.dumps(meta, indent=2))


def legacy_meta(feature_dir, encoder, records):
    title_emb = np.load(feature_dir / "title_emb.npy", mmap_mode="r")
    return {
        "encoder": encoder,
        "tokenizer": encoder,
        "tokenizer_class": None,
        "hidden_size": int(title_emb.shape[-1]),
        "title_length": int(title_emb.shape[1]),
        "speech_text_length": int(np.load(feature_dir / "transcript_emb.npy", mmap_mode="r").shape[1]),
        "num_records": int(title_emb.shape[0]),
        "num_empty_transcripts": sum(1 for r in records if not r["has_transcript"]),
        "data_sha256": file_sha256(DATA_PATH),
        "extracted": None,
        "device": None,
        "versions": None,
        "provenance": "extracted before meta.json existed; extraction date, device and versions "
                      "were not recorded. Fields above are read back from the arrays.",
        "meta_written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=ENCODERS, default=DEFAULT_ENCODER)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    feature_dir = feature_dir_for(args.encoder)
    feature_dir.mkdir(parents=True, exist_ok=True)
    meta_path = feature_dir / FEATURE_META
    if features_exist(feature_dir) and not args.force:
        if not meta_path.exists():
            write_meta(feature_dir, legacy_meta(feature_dir, args.encoder, load_records()))
            print(f"wrote {meta_path} for a cache that predates meta.json")
        meta = json.loads(meta_path.read_text())
        if meta["encoder"] != args.encoder:
            raise ValueError(f"{meta_path} records encoder {meta['encoder']!r}, expected {args.encoder!r}")
        print(f"cache hit: {feature_dir}/ already populated — nothing to do "
              f"(use --force to rebuild)")
        report(feature_dir)
        return

    records = load_records()
    device = get_device()
    print(f"records: {len(records)}   device: {device}   model: {args.encoder} (frozen)")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    titles = [r["title"] for r in records]
    transcripts = [r["transcript"] for r in records]
    n_empty = sum(1 for t in transcripts if not t.strip())
    print(f"empty transcripts: {n_empty} -> all-zero embedding + all-zero mask "
          f"(encoder not run on them)")

    title_emb, title_mask = encode(
        titles, tokenizer, encoder, TITLE_LENGTH, device, "titles   ", skip_empty=False
    )
    tr_emb, tr_mask = encode(
        transcripts, tokenizer, encoder, SPEECH_TEXT_LENGTH, device,
        "transcripts", skip_empty=True,
    )

    arrays = {
        "title_emb": title_emb, "title_mask": title_mask,
        "transcript_emb": tr_emb, "transcript_mask": tr_mask,
    }
    for name, arr in arrays.items():
        np.save(feature_dir / f"{name}.npy", arr)

    # Contract check: empty transcripts must be all-zero on both arrays.
    empty_idx = [i for i, t in enumerate(transcripts) if not t.strip()]
    assert not tr_emb[empty_idx].any(), "empty transcript got a non-zero embedding"
    assert not tr_mask[empty_idx].any(), "empty transcript got a non-zero mask"
    print(f"\ncontract OK: all {len(empty_idx)} empty transcripts are all-zero "
          f"in both embedding and mask")

    write_meta(feature_dir, {
        "encoder": args.encoder,
        "tokenizer": args.encoder,
        "tokenizer_class": type(tokenizer).__name__,
        "hidden_size": int(encoder.config.hidden_size),
        "title_length": TITLE_LENGTH,
        "speech_text_length": SPEECH_TEXT_LENGTH,
        "num_records": len(records),
        "num_empty_transcripts": n_empty,
        "title_truncated": int(sum(len(tokenizer(t)["input_ids"]) > TITLE_LENGTH for t in titles)),
        "transcript_truncated": int(sum(len(tokenizer(t)["input_ids"]) > SPEECH_TEXT_LENGTH
                                        for t in transcripts if t.strip())),
        "data_sha256": file_sha256(DATA_PATH),
        "extracted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": package_versions(),
        "provenance": "extract_features.py",
    })
    report(feature_dir)


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

    meta = json.loads((Path(feature_dir) / FEATURE_META).read_text())
    print(f"\nmeta: encoder {meta['encoder']} | tokenizer {meta['tokenizer_class']} | "
          f"hidden {meta['hidden_size']} | lengths {meta['title_length']}/{meta['speech_text_length']} | "
          f"truncated titles {meta.get('title_truncated')} transcripts {meta.get('transcript_truncated')}")


if __name__ == "__main__":
    main()
