import argparse
import csv
import json
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

from yt_dlp import YoutubeDL

URL_FILES = [Path("data/youtube_urls.txt"), Path("data/twitter_urls.txt")]
DATA_JSON = Path("data/data.json")
OUT_DIR = Path("videos")
FAIL_LOG = OUT_DIR / "failed.csv"
COVERAGE = OUT_DIR / "coverage.json"
CATEGORY = {"": "REAL", "ft": "CD", "fv": "CE", "fa": "SV", "fc": "CA"}
CLASS_CODES = {"real": "", "cd": "ft", "ce": "fv", "sv": "fa", "ca": "fc"}
PROGRESS_EVERY = 25
VIDEO_TIMEOUT = 300
MIN_VIDEO_BYTES = 10 * 1024
PERMANENT_MARKERS = ["No video could be found", "Private video", "Video unavailable", "Video is unavailable"]
TRANSIENT_MARKERS = [
    "SSL", "CERTIFICATE_VERIFY_FAILED", "nodename nor servname", "Name or service not known",
    "Temporary failure in name resolution", "timed out", "HTTP Error 403",
]


def video_id(url):
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url) or re.search(r"status/(\d+)", url)
    return match.group(1)


def load_urls():
    urls = []
    for path in URL_FILES:
        urls += [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return urls


def options(vid, browser):
    opts = {
        "outtmpl": str(OUT_DIR / f"{vid}.%(ext)s"),
        "format": "bv*[height<=480]+ba/b[height<=480]/b",
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noplaylist": True,
        "playlist_items": "1",
        "retries": 3,
        "socket_timeout": 30,
        "sleep_interval": 1,
        "max_sleep_interval": 3,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": True,
    }
    if browser:
        opts["cookiesfrombrowser"] = (browser,)
    return opts


def classify(reason):
    text = reason.lower()
    if any(m.lower() in text for m in PERMANENT_MARKERS):
        return "permanent"
    if any(m.lower() in text for m in TRANSIENT_MARKERS):
        return "transient"
    return "unknown"


def has_video(vid):
    path = OUT_DIR / f"{vid}.mp4"
    return path.is_file() and path.stat().st_size > MIN_VIDEO_BYTES


def load_failures():
    if not FAIL_LOG.exists():
        return {}
    with FAIL_LOG.open(newline="") as f:
        rows = {row["video_id"]: row for row in csv.DictReader(f)}
    for row in rows.values():
        row["kind"] = classify(row["reason"])
    return rows


def save_failures(failures):
    with FAIL_LOG.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "url", "kind", "reason"])
        writer.writeheader()
        writer.writerows(failures[k] for k in sorted(failures))


def fetch(url, vid, browser, queue):
    try:
        with YoutubeDL(options(vid, browser)) as ydl:
            ydl.download([url])
        queue.put(None)
    except Exception as e:
        queue.put(str(e).splitlines()[0][:200] or type(e).__name__)


def attempt(url, vid, browser):
    queue = mp.Queue()
    proc = mp.Process(target=fetch, args=(url, vid, browser, queue))
    proc.start()
    proc.join(VIDEO_TIMEOUT)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return f"timed out after {VIDEO_TIMEOUT}s"
    try:
        return queue.get(timeout=5)
    except Exception:
        return f"worker exited with code {proc.exitcode}"


def should_skip(vid, failures, retry_failed):
    if has_video(vid):
        return True
    return not retry_failed and failures.get(vid, {}).get("kind") == "permanent"


def download(urls, browser, retry_failed=False):
    failures = load_failures()
    new_failures, ok, skipped = [], 0, 0
    for i, url in enumerate(urls, 1):
        vid = video_id(url)
        if should_skip(vid, failures, retry_failed):
            skipped += 1
        else:
            reason = attempt(url, vid, browser)
            if reason is None and has_video(vid):
                ok += 1
                if failures.pop(vid, None):
                    save_failures(failures)
            else:
                reason = reason or f"output missing or under {MIN_VIDEO_BYTES} bytes"
                failures[vid] = {"video_id": vid, "url": url, "kind": classify(reason), "reason": reason}
                new_failures.append(failures[vid])
                save_failures(failures)
        if i % PROGRESS_EVERY == 0 or i == len(urls):
            print(f"[{i}/{len(urls)}] ok {ok}  failed {len(new_failures)}  skipped {skipped}", flush=True)
    save_failures(failures)
    return new_failures


def report():
    records = json.loads(DATA_JSON.read_text())
    have, total = Counter(), Counter()
    for r in records:
        keys = [CATEGORY[r["class"]], "YouTube" if len(r["video_id"]) == 11 else "X", "ALL"]
        present = (OUT_DIR / f"{r['video_id']}.mp4").exists()
        for k in keys:
            total[k] += 1
            have[k] += present

    rows = {k: {"have": have[k], "total": total[k], "pct": round(100 * have[k] / total[k], 1)}
            for k in ["REAL", "CD", "CE", "SV", "CA", "YouTube", "X", "ALL"]}
    COVERAGE.write_text(json.dumps(rows, indent=2))

    print(f"\n{'group':<8}{'have':>7}{'total':>7}{'pct':>8}")
    for k, v in rows.items():
        print(f"{k:<8}{v['have']:>7}{v['total']:>7}{v['pct']:>7}%")


def filter_classes(urls, classes):
    wanted = {CLASS_CODES[c] for c in classes}
    records = json.loads(DATA_JSON.read_text())
    keep = {r["video_id"] for r in records if r["class"] in wanted}
    return [u for u in urls if video_id(u) in keep]


def filter_categories(urls, categories):
    records = json.loads(DATA_JSON.read_text())
    keep = {r["video_id"] for r in records if CATEGORY[r["class"]] in categories}
    return [u for u in urls if video_id(u) in keep]


def parse_categories(value):
    categories = {c.strip().upper() for c in value.split(",") if c.strip()}
    unknown = categories - set(CATEGORY.values())
    if unknown or not categories:
        raise argparse.ArgumentTypeError(
            f"unknown categories {sorted(unknown)}; choose from {','.join(CATEGORY.values())}"
        )
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cookies-from-browser", dest="browser")
    parser.add_argument("--source", choices=["youtube", "x", "all"], default="all")
    parser.add_argument("--classes", nargs="+", choices=sorted(CLASS_CODES))
    parser.add_argument("--categories", type=parse_categories, default=set(CATEGORY.values()))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    urls = load_urls()
    if args.source == "youtube":
        urls = [u for u in urls if "youtube.com" in u]
    elif args.source == "x":
        urls = [u for u in urls if "x.com" in u]
    if args.classes:
        urls = filter_classes(urls, args.classes)
    urls = filter_categories(urls, args.categories)
    if args.limit:
        urls = urls[: args.limit]

    failures = download(urls, args.browser, args.retry_failed)
    print(f"\n{len(failures)} failed this run -> {FAIL_LOG}")
    report()


if __name__ == "__main__":
    main()
