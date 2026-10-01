"""FMNV data loading, label/category mapping, has_transcript flag, and splits.

Record schema (flat objects in `data/data.json`, 2393 of them):

    video_id, class, contextual_class, label, subject, title, audio_transcript

Mappings (confirmed by the counts, which match the paper's 893/600/450/300/150):

    label  "false" -> 1 (fake, positive class)      "true" -> 0 (real)
    class  ""  -> REAL    ft -> CD    fv -> CE    fa -> SV    fc -> CA
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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

VIDEO_DIR = Path("videos")
VISUAL_CATEGORIES = ["REAL", "CD", "CA"]
VISUAL_SPLIT_PATH = Path("splits/visual_split.json")
VISUAL_FEATURE_DIR = Path("features/video")
VISUAL_FEATURE_FILES = {"clip": "clip", "motion": "motion"}

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


def video_path(record, video_dir=VIDEO_DIR):
    return Path(video_dir) / f"{record['video_id']}.mp4"


def scan_visual_subset(records, video_dir=VIDEO_DIR):
    return [
        r for r in records
        if r["category"] in VISUAL_CATEGORIES
        and video_path(r, video_dir).is_file()
        and video_path(r, video_dir).stat().st_size > 0
    ]


def freeze_visual_split(splits, path=VISUAL_SPLIT_PATH):
    subset = [r for recs in splits.values() for r in recs]
    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "ratios": list(SPLIT_RATIOS),
        "stratified_on": VISUAL_CATEGORIES,
        "subset_size": len(subset),
        "category_counts": {c: _counts(subset, "category").get(c, 0) for c in VISUAL_CATEGORIES},
        "split_counts": {
            name: {c: _counts(recs, "category").get(c, 0) for c in VISUAL_CATEGORIES}
            for name, recs in splits.items()
        },
        "splits": {name: [r["video_id"] for r in recs] for name, recs in splits.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_visual_split(records=None, path=VISUAL_SPLIT_PATH, video_dir=VIDEO_DIR, refresh=False):
    records = records if records is not None else load_records()
    path = Path(path)
    if refresh or not path.exists():
        splits = make_splits(scan_visual_subset(records, video_dir))
        freeze_visual_split(splits, path)
    frozen = json.loads(path.read_text())
    by_id = {r["video_id"]: r for r in records}
    unknown = [v for ids in frozen["splits"].values() for v in ids if v not in by_id]
    if unknown:
        raise ValueError(f"{len(unknown)} video_ids in {path} are not in {DATA_PATH}")
    return {
        name: sorted((by_id[v] for v in ids), key=lambda r: r["index"])
        for name, ids in frozen["splits"].items()
    }


def build_visual_subset(records=None, path=VISUAL_SPLIT_PATH, video_dir=VIDEO_DIR, refresh=False):
    splits = load_visual_split(records, path, video_dir, refresh)
    return sorted((r for recs in splits.values() for r in recs), key=lambda r: r["index"])


def make_visual_splits(records=None, path=VISUAL_SPLIT_PATH, video_dir=VIDEO_DIR, refresh=False):
    return load_visual_split(records, path, video_dir, refresh)


class FMNVDDataset(Dataset):
    """Serves cached BERT token embeddings + attention masks for a split.

    Empty transcripts carry an all-zero embedding and an all-zero attention
    mask (written that way by `extract_features.py`) — never silent padding.
    """

    def __init__(self, records, feature_dir=FEATURE_DIR, features=None, visual=None):
        self.records = records
        self.features = features if features is not None else load_features(feature_dir)
        self.visual = visual
        if visual is not None:
            missing = [r["video_id"] for r in records if r["video_id"] not in visual["row"]]
            if missing:
                raise ValueError(f"{len(missing)} records have no visual features, e.g. {missing[:3]}")
            self.visual_rows = np.array([visual["row"][r["video_id"]] for r in records], dtype=np.int64)
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
        item = {
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
        if self.visual is not None:
            k = self.visual_rows[i]
            item["clip"] = torch.from_numpy(np.asarray(self.visual["clip"][k], dtype=np.float32))
            item["motion"] = torch.from_numpy(np.asarray(self.visual["motion"][k], dtype=np.float32))
        return item


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


def load_visual_features(records=None, feature_dir=VISUAL_FEATURE_DIR, mmap=False):
    order = [r["video_id"] for r in (records if records is not None else build_visual_subset())]
    out = {}
    for key, stem in VISUAL_FEATURE_FILES.items():
        ids = json.loads((Path(feature_dir) / f"{stem}_ids.json").read_text())
        if ids != order:
            raise ValueError(f"{stem}_ids.json does not match the frozen visual split order")
        out[key] = np.load(Path(feature_dir) / f"{stem}_features.npy", mmap_mode="r" if mmap else None)
        if len(out[key]) != len(order):
            raise ValueError(f"{stem}_features.npy has {len(out[key])} rows, expected {len(order)}")
    out["ids"] = order
    out["row"] = {v: i for i, v in enumerate(order)}
    return out


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


def _balance(records):
    fake = sum(r["label"] for r in records) / len(records)
    return fake, max(fake, 1 - fake), "fake" if fake >= 0.5 else "real"


def visual_report(refresh=False):
    records = load_records()
    existed = VISUAL_SPLIT_PATH.exists() and not refresh
    splits = make_visual_splits(records, refresh=refresh)
    subset = [r for recs in splits.values() for r in recs]
    eligible = [r for r in records if r["category"] in VISUAL_CATEGORIES]
    frozen = json.loads(VISUAL_SPLIT_PATH.read_text())
    missing = sum(1 for r in subset if not video_path(r).is_file())

    print(f"frozen split                  : {VISUAL_SPLIT_PATH}  "
          f"({'loaded from file' if existed else 'built from videos/ and written'}, created {frozen['created']})")
    print(f"visual subset                 : {len(subset)} of {len(eligible)} REAL+CD+CA records with video")
    print(f"split seed                    : {SEED}")
    print(f"split ratios (train/val/test) : "
          f"{SPLIT_RATIOS[0]:.0%}/{SPLIT_RATIOS[1]:.0%}/{SPLIT_RATIOS[2]:.0%}"
          f"  (stratified on REAL/CD/CA)")
    print()

    print(f"{'split':<8}{'n':>6}{'%':>7}   "
          + "".join(f"{c:>6}" for c in VISUAL_CATEGORIES)
          + f"{'real':>7}{'fake':>7}{'%fake':>8}")
    for name, recs in list(splits.items()) + [("ALL", subset)]:
        cat = _counts(recs, "category")
        lab = _counts(recs, "label")
        real, fake = lab.get(0, 0), lab.get(1, 0)
        print(f"{name:<8}{len(recs):>6}{len(recs) / len(subset):>7.1%}   "
              + "".join(f"{cat.get(c, 0):>6}" for c in VISUAL_CATEGORIES)
              + f"{real:>7}{fake:>7}{fake / len(recs):>8.1%}")
    print()

    ids = [set(r["index"] for r in recs) for recs in splits.values()]
    disjoint = not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    covered = set().union(*ids) == {r["index"] for r in subset}
    print(f"CHECK splits disjoint         : {disjoint}")
    print(f"CHECK splits cover subset     : {covered}")
    print(f"CHECK no CE/SV in subset      : {all(r['category'] in VISUAL_CATEGORIES for r in subset)}")
    print(f"CHECK frozen videos on disk   : {missing == 0}  ({missing} missing)")
    print()

    print(f"{'dataset':<26}{'n':>6}{'%fake':>8}{'majority':>10}{'baseline acc':>14}")
    for name, recs in [("visual subset (REAL+CD+CA)", subset), ("full dataset", records)]:
        fake, acc, majority = _balance(recs)
        print(f"{name:<26}{len(recs):>6}{fake:>8.1%}{majority:>10}{acc:>14.1%}")


if __name__ == "__main__":
    if "--visual" in sys.argv[1:]:
        visual_report(refresh="--refresh" in sys.argv[1:])
    else:
        main()
