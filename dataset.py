"""FMNV data loading, label/category mapping, has_transcript flag, and splits.

Record schema (flat objects in `data/data.json`, 2393 of them):

    video_id, class, contextual_class, label, subject, title, audio_transcript

Mappings (confirmed by the counts, which match the paper's 893/600/450/300/150):

    label  "false" -> 1 (fake, positive class)      "true" -> 0 (real)
    class  ""  -> REAL    ft -> CD    fv -> CE    fa -> SV    fc -> CA
"""

import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DATA_PATH = "data/data.json"
FEATURE_DIR = "features"

LABEL_MAP = {"false": 1, "true": 0}          # "false" = fake = positive class
CATEGORY_MAP = {                              # paper name -> code
    "": "REAL",
    "ft": "CD",   # Contextual Dishonesty
    "fv": "CE",   # Cherry-picked Editing
    "fa": "SV",   # Synthetic Voiceover
    "fc": "CA",   # Contrived Absurdity
}
FAKE_CATEGORIES = ["CD", "CE", "SV", "CA"]
CATEGORIES = ["REAL"] + FAKE_CATEGORIES

SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)             # train / val / test


def load_records(path=DATA_PATH):
    """Read data.json into normalised records, preserving file order.

    `index` is the row's position in the file and is what the cached BERT
    feature arrays are indexed by.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for i, r in enumerate(raw):
        label_str = r["label"]
        class_str = r["class"]
        if label_str not in LABEL_MAP:
            raise ValueError(f"record {i}: unexpected label {label_str!r}")
        if class_str not in CATEGORY_MAP:
            raise ValueError(f"record {i}: unexpected class {class_str!r}")

        transcript = str(r.get("audio_transcript") or "").strip()
        records.append({
            "index": i,
            "video_id": r["video_id"],
            "title": str(r["title"]).strip(),
            "transcript": transcript,
            "has_transcript": bool(transcript),
            "label": LABEL_MAP[label_str],
            "category": CATEGORY_MAP[class_str],
            "subject": r.get("subject", ""),
        })

    consistent = all(
        (rec["label"] == 0) == (rec["category"] == "REAL") for rec in records
    )
    if not consistent:
        raise ValueError("label and class disagree on real/fake for some record")
    return records


def make_splits(records, ratios=SPLIT_RATIOS, seed=SEED):
    """Stratified train/val/test split on the 5-way *category* label.

    Stratifying on category (not just real/fake) keeps the CD/CE/SV/CA mix
    representative in every split, which the per-category evaluation needs.
    """
    train_ratio, val_ratio, test_ratio = ratios
    strata = [r["category"] for r in records]
    idx = np.arange(len(records))

    train_idx, hold_idx = train_test_split(
        idx, train_size=train_ratio, random_state=seed, shuffle=True,
        stratify=strata,
    )
    hold_strata = [strata[i] for i in hold_idx]
    val_idx, test_idx = train_test_split(
        hold_idx, train_size=val_ratio / (val_ratio + test_ratio),
        random_state=seed, shuffle=True, stratify=hold_strata,
    )

    return {
        "train": [records[i] for i in sorted(train_idx)],
        "val": [records[i] for i in sorted(val_idx)],
        "test": [records[i] for i in sorted(test_idx)],
    }


class FMNVDDataset(Dataset):
    """Serves cached BERT token embeddings + attention masks for a split.

    Empty transcripts carry an all-zero embedding and an all-zero attention
    mask (written that way by `extract_features.py`) — never silent padding.
    """

    def __init__(self, records, feature_dir=FEATURE_DIR, features=None):
        self.records = records
        self.features = features if features is not None else load_features(feature_dir)
        self.indices = np.array([r["index"] for r in records], dtype=np.int64)
        self.labels = np.array([r["label"] for r in records], dtype=np.int64)
        self.has_transcript = np.array(
            [float(r["has_transcript"]) for r in records], dtype=np.float32
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        j = self.indices[i]
        f = self.features
        return {
            "title": torch.from_numpy(np.asarray(f["title_emb"][j], dtype=np.float32)),
            "title_mask": torch.from_numpy(
                np.asarray(f["title_mask"][j], dtype=np.float32)
            ),
            "speech": torch.from_numpy(
                np.asarray(f["transcript_emb"][j], dtype=np.float32)
            ),
            "speech_mask": torch.from_numpy(
                np.asarray(f["transcript_mask"][j], dtype=np.float32)
            ),
            "has_transcript": torch.tensor(self.has_transcript[i]),
            "label": torch.tensor(self.labels[i]),
        }


FEATURE_FILES = ["title_emb", "title_mask", "transcript_emb", "transcript_mask"]


def features_exist(feature_dir=FEATURE_DIR):
    return all(
        os.path.exists(os.path.join(feature_dir, f"{n}.npy")) for n in FEATURE_FILES
    )


def load_features(feature_dir=FEATURE_DIR, mmap=True):
    if not features_exist(feature_dir):
        raise FileNotFoundError(
            f"cached features missing in {feature_dir}/ — run: python extract_features.py"
        )
    return {
        n: np.load(os.path.join(feature_dir, f"{n}.npy"),
                   mmap_mode="r" if mmap else None)
        for n in FEATURE_FILES
    }


# --------------------------------------------------------------------------- #
# verification report
# --------------------------------------------------------------------------- #

def _counts(records, key):
    out = {}
    for r in records:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def main():
    records = load_records()
    splits = make_splits(records)

    print(f"total records                 : {len(records)}")
    print(f"split seed                    : {SEED}")
    print(f"split ratios (train/val/test) : "
          f"{SPLIT_RATIOS[0]:.0%}/{SPLIT_RATIOS[1]:.0%}/{SPLIT_RATIOS[2]:.0%}"
          f"  (stratified on category)")
    print()

    print("=== SPLIT SIZES ===")
    for name in ("train", "val", "test"):
        n = len(splits[name])
        print(f"  {name:<6} {n:>5}  ({n / len(records):.1%})")
    print(f"  {'TOTAL':<6} {sum(len(s) for s in splits.values()):>5}")
    print()

    print("=== LABEL DISTRIBUTION PER SPLIT ===")
    print(f"{'split':<8}{'real(0)':>9}{'fake(1)':>9}{'%fake':>8}   "
          + "".join(f"{c:>7}" for c in CATEGORIES))
    for name in ("train", "val", "test"):
        recs = splits[name]
        lab = _counts(recs, "label")
        cat = _counts(recs, "category")
        real, fake = lab.get(0, 0), lab.get(1, 0)
        print(f"{name:<8}{real:>9}{fake:>9}{fake / len(recs):>8.1%}   "
              + "".join(f"{cat.get(c, 0):>7}" for c in CATEGORIES))
    lab = _counts(records, "label")
    cat = _counts(records, "category")
    print(f"{'ALL':<8}{lab[0]:>9}{lab[1]:>9}{lab[1] / len(records):>8.1%}   "
          + "".join(f"{cat.get(c, 0):>7}" for c in CATEGORIES))
    print()

    print("=== TRANSCRIPT MISSINGNESS BY CATEGORY ===")
    print(f"{'category':<10}{'n':>6}{'missing':>9}{'% missing':>11}{'expected':>10}")
    expected = {"SV": "0%", "CA": "~73%", "CD": "~52%", "CE": "41%", "REAL": "35%"}
    for c in ["REAL"] + FAKE_CATEGORIES:
        recs = [r for r in records if r["category"] == c]
        miss = sum(1 for r in recs if not r["has_transcript"])
        print(f"{c:<10}{len(recs):>6}{miss:>9}{miss / len(recs):>10.1%}"
              f"{expected[c]:>10}")
    total_missing = sum(1 for r in records if not r["has_transcript"])
    print(f"{'ALL':<10}{len(records):>6}{total_missing:>9}"
          f"{total_missing / len(records):>10.1%}{'38.3%':>10}")
    print()

    sv_missing = sum(
        1 for r in records if r["category"] == "SV" and not r["has_transcript"]
    )
    ca = [r for r in records if r["category"] == "CA"]
    ca_missing = sum(1 for r in ca if not r["has_transcript"]) / len(ca)
    print(f"CHECK SV missing == 0%        : {sv_missing == 0}  (SV missing = {sv_missing})")
    print(f"CHECK CA missing ~= 73%       : {0.70 <= ca_missing <= 0.76}"
          f"  (CA missing = {ca_missing:.1%})")
    print()

    print("=== has_transcript BY SPLIT (leakage sanity) ===")
    print(f"{'split':<8}" + "".join(f"{c:>9}" for c in CATEGORIES))
    for name in ("train", "val", "test"):
        row = []
        for c in CATEGORIES:
            recs = [r for r in splits[name] if r["category"] == c]
            frac = sum(r["has_transcript"] for r in recs) / len(recs) if recs else 0
            row.append(f"{frac:>8.0%} ")
        print(f"{name:<8}" + "".join(row))


if __name__ == "__main__":
    main()
