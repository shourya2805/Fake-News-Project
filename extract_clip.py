import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from dataset import build_visual_subset, video_path
from utils import get_device

NUM_FRAMES = 55
CLIP_DIM = 512
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
OUT_DIR = Path("features/video")
PER_VIDEO_DIR = OUT_DIR / "clip"
CONSOLIDATED = OUT_DIR / "clip_features.npy"
CONSOLIDATED_IDS = OUT_DIR / "clip_ids.json"
META = OUT_DIR / "clip_meta.json"


def frame_indices(total, count=NUM_FRAMES):
    return np.round(np.linspace(0, total - 1, count)).astype(int)


def read_indices(path, indices):
    cap = cv2.VideoCapture(str(path))
    wanted = set(int(i) for i in indices)
    frames, pos, last = {}, 0, int(indices.max())
    while pos <= last and cap.grab():
        if pos in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[pos] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pos += 1
    cap.release()
    return frames, pos


def sample_frames(path):
    cap = cv2.VideoCapture(str(path))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    total = reported
    for _ in range(2):
        if total <= 0:
            break
        indices = frame_indices(total)
        frames, seen = read_indices(path, indices)
        if all(int(i) in frames for i in indices):
            return [frames[int(i)] for i in indices], total
        if seen >= total:
            break
        total = seen
    raise RuntimeError(f"could not decode frames (reported {reported})")


def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess


@torch.no_grad()
def encode(model, preprocess, frames, device):
    batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
    return model.encode_image(batch).float().cpu().numpy()


def per_video_path(vid):
    return PER_VIDEO_DIR / f"{vid}.npy"


def save_atomic(path, array):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def load_meta():
    if META.exists():
        return json.loads(META.read_text())
    return {"model": f"{MODEL_NAME}/{PRETRAINED}", "num_frames": NUM_FRAMES, "frame_counts": {}, "padded": {}, "failed": {}}


def consolidate(records, meta):
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
    args = parser.parse_args()

    PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    records = build_visual_subset()
    todo = records[: args.limit] if args.limit else records
    meta = load_meta()

    device = get_device()
    model, preprocess = None, None
    processed, skipped, failed = 0, 0, 0
    work_seconds = 0.0

    for r in tqdm(todo, desc="clip", unit="video"):
        vid = r["video_id"]
        if per_video_path(vid).exists() and not args.force:
            skipped += 1
            continue
        if model is None:
            model, preprocess = load_model(device)
        start = time.time()
        try:
            frames, total = sample_frames(video_path(r))
            features = encode(model, preprocess, frames, device)
        except Exception as e:
            meta["failed"][vid] = str(e)[:200]
            failed += 1
            continue
        assert features.shape == (NUM_FRAMES, CLIP_DIM), features.shape
        save_atomic(per_video_path(vid), features.astype(np.float16))
        work_seconds += time.time() - start
        meta["frame_counts"][vid] = total
        meta["failed"].pop(vid, None)
        if total < NUM_FRAMES:
            meta["padded"][vid] = total
        else:
            meta["padded"].pop(vid, None)
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
    print(f"padded (<{NUM_FRAMES} frames)   : {len(meta['padded'])}")
    if processed:
        rate = work_seconds / processed
        print(f"throughput            : {rate:.2f} s/video  ({processed / work_seconds * 60:.1f} videos/min)")
        print(f"estimate, remaining   : {remaining} videos ~ {remaining * rate / 60:.1f} min")
        print(f"estimate, all {len(records)}   : ~ {len(records) * rate / 60:.1f} min")

    array, missing = consolidate(records, meta)
    if array is None:
        print(f"consolidation         : skipped, {missing} of {len(records)} videos not yet extracted")
    else:
        print(f"consolidated          : {CONSOLIDATED}  shape {array.shape}  dtype {array.dtype}  "
              f"{CONSOLIDATED.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
