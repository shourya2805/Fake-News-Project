import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset import build_visual_subset, video_path
from extract_clip import META as CLIP_META
from extract_clip import NUM_FRAMES, frame_indices
from modules.resnext3d import resnext101_kinetics
from utils import get_device

CLIP_LEN = 8
SAMPLE_SIZE = 112
MOTION_DIM = 2048
MEAN = np.array([114.7748, 107.7354, 99.4750], dtype=np.float32)
WEIGHTS = Path("weights/resnext-101-kinetics.pth")
WEIGHTS_DRIVE_ID = "1cULocPe5YvPGWU4tV5t6fC9rJdGLfkWe"
WEIGHTS_SHA256 = "f82e4e519723fc7b2ff3761ea35600bdaf796fb7a4e62ee4c5591da7ffe48326"
OUT_DIR = Path("features/video")
PER_VIDEO_DIR = OUT_DIR / "motion"
CONSOLIDATED = OUT_DIR / "motion_features.npy"
CONSOLIDATED_IDS = OUT_DIR / "motion_ids.json"
META = OUT_DIR / "motion_meta.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weights():
    if not WEIGHTS.exists():
        import gdown
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=WEIGHTS_DRIVE_ID, output=str(WEIGHTS), quiet=False)
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise RuntimeError(f"{WEIGHTS} does not match the expected sha256")
    return WEIGHTS


def load_model(device):
    model = resnext101_kinetics(ensure_weights())
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def clip_windows(total, centres):
    starts = np.clip(centres - CLIP_LEN // 2, 0, max(total - CLIP_LEN, 0))
    return (starts[:, None] + np.arange(CLIP_LEN)[None, :]) % max(total, 1)


def preprocess(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    w, h = image.size
    if w < h:
        size = (SAMPLE_SIZE, int(SAMPLE_SIZE * h / w))
    else:
        size = (int(SAMPLE_SIZE * w / h), SAMPLE_SIZE)
    image = image.resize(size, Image.BILINEAR)
    w, h = image.size
    left, top = int(round((w - SAMPLE_SIZE) / 2.0)), int(round((h - SAMPLE_SIZE) / 2.0))
    image = image.crop((left, top, left + SAMPLE_SIZE, top + SAMPLE_SIZE))
    return np.asarray(image, dtype=np.float32) - MEAN


def read_frames(path, wanted):
    cap = cv2.VideoCapture(str(path))
    frames, pos, last = {}, 0, max(wanted)
    while pos <= last and cap.grab():
        if pos in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[pos] = preprocess(frame)
        pos += 1
    cap.release()
    return frames, pos


def reported_frames(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def sample_clips(path, known_total=None):
    total = known_total or reported_frames(path)
    for _ in range(2):
        if total <= 0:
            break
        windows = clip_windows(total, frame_indices(total))
        frames, seen = read_frames(path, set(int(i) for i in windows.ravel()))
        if all(int(i) in frames for i in windows.ravel()):
            clips = np.stack([np.stack([frames[int(i)] for i in w]) for w in windows])
            return clips.transpose(0, 4, 1, 2, 3), total
        if seen >= total:
            break
        total = seen
    raise RuntimeError(f"could not decode clips (frame count {known_total or reported_frames(path)})")


@torch.no_grad()
def encode(model, clips, device, batch_size):
    out = []
    for i in range(0, len(clips), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(clips[i:i + batch_size])).to(device)
        out.append(model.features(batch).float().cpu().numpy())
    return np.concatenate(out)


def per_video_path(vid):
    return PER_VIDEO_DIR / f"{vid}.npy"


def save_atomic(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def consolidate(records):
    ids = [r["video_id"] for r in records]
    missing = [v for v in ids if not per_video_path(v).exists()]
    if missing:
        return None, len(missing)
    array = np.stack([np.load(per_video_path(v)) for v in ids]).astype(np.float16)
    np.save(CONSOLIDATED, array)
    CONSOLIDATED_IDS.write_text(json.dumps(ids))
    return array, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=NUM_FRAMES)
    parser.add_argument("--device")
    args = parser.parse_args()

    PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    records = build_visual_subset()
    todo = records[: args.limit] if args.limit else records
    clip_counts = load_json(CLIP_META, {}).get("frame_counts", {})
    meta = load_json(META, {
        "model": "resnext-101-kinetics (Hara et al., CVPR 2018)",
        "num_clips": NUM_FRAMES, "clip_len": CLIP_LEN, "sample_size": SAMPLE_SIZE,
        "frame_counts": {}, "short": {}, "unaligned": [], "failed": {},
    })

    device = torch.device(args.device) if args.device else get_device()
    model = None
    processed, skipped, failed = 0, 0, 0
    work_seconds = 0.0

    for r in tqdm(todo, desc="motion", unit="video"):
        vid = r["video_id"]
        if per_video_path(vid).exists() and not args.force:
            skipped += 1
            continue
        if model is None:
            model = load_model(device)
        start = time.time()
        try:
            clips, total = sample_clips(video_path(r), clip_counts.get(vid))
            features = encode(model, clips, device, args.batch_size)
        except Exception as e:
            meta["failed"][vid] = str(e)[:200]
            failed += 1
            continue
        assert features.shape == (NUM_FRAMES, MOTION_DIM), features.shape
        save_atomic(per_video_path(vid), features.astype(np.float16))
        work_seconds += time.time() - start
        meta["frame_counts"][vid] = total
        meta["failed"].pop(vid, None)
        if total < CLIP_LEN:
            meta["short"][vid] = total
        else:
            meta["short"].pop(vid, None)
        if clip_counts.get(vid) != total:
            meta["unaligned"] = sorted(set(meta["unaligned"]) | {vid})
        else:
            meta["unaligned"] = [v for v in meta["unaligned"] if v != vid]
        processed += 1
        META.write_text(json.dumps(meta, indent=2))

    META.write_text(json.dumps(meta, indent=2))
    remaining = sum(1 for r in records if not per_video_path(r["video_id"]).exists())

    print()
    print(f"device                : {device}")
    print(f"selected / subset     : {len(todo)} / {len(records)}")
    print(f"processed             : {processed}")
    print(f"skipped (on disk)     : {skipped}")
    print(f"failed                : {failed}")
    print(f"short (<{CLIP_LEN} frames)     : {len(meta['short'])}")
    print(f"not aligned with CLIP : {len(meta['unaligned'])}")
    if processed:
        rate = work_seconds / processed
        print(f"throughput            : {rate:.2f} s/video  ({processed / work_seconds * 60:.1f} videos/min)")
        print(f"estimate, remaining   : {remaining} videos ~ {remaining * rate / 60:.1f} min")
        print(f"estimate, all {len(records)}   : ~ {len(records) * rate / 60:.1f} min")

    array, missing = consolidate(records)
    if array is None:
        print(f"consolidation         : skipped, {missing} of {len(records)} videos not yet extracted")
    else:
        print(f"consolidated          : {CONSOLIDATED}  shape {array.shape}  dtype {array.dtype}  "
              f"{CONSOLIDATED.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
