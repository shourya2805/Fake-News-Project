import difflib
import json
import re
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import torch
from faster_whisper import WhisperModel

DATA_JSON = Path("data/data.json")
VIDEO_DIR = Path("videos")
WORK_DIR = VIDEO_DIR / "verify"
AUDIO_DIR = WORK_DIR / "audio"
TEXT_DIR = WORK_DIR / "whisper"
REPORT = WORK_DIR / "report.json"
RESULTS_TABLE = Path("verify_results.txt")
CATEGORY = {"": "REAL", "ft": "CD", "fv": "CE", "fa": "SV", "fc": "CA"}
ORDER = ["REAL", "CD", "CE", "SV", "CA"]
WHISPER_SIZE = "base"
FLAG_GAP = 0.15


def normalise(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def token_overlap(reference, hypothesis):
    ref, hyp = Counter(reference), Counter(hypothesis)
    if not ref:
        return 0.0
    return sum((ref & hyp).values()) / sum(ref.values())


def seq_ratio(reference, hypothesis):
    return difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False).ratio()


def extract_audio(vid):
    wav = AUDIO_DIR / f"{vid}.wav"
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(VIDEO_DIR / f"{vid}.mp4"),
             "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
    return wav


def transcribe(vid, model_holder):
    txt = TEXT_DIR / f"{vid}.json"
    if txt.exists():
        return json.loads(txt.read_text())
    if "model" not in model_holder:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        model_holder["model"] = WhisperModel(WHISPER_SIZE, device=device, compute_type=compute)
    segments, info = model_holder["model"].transcribe(str(extract_audio(vid)), beam_size=5)
    result = {"language": info.language, "duration": info.duration,
              "text": " ".join(s.text.strip() for s in segments)}
    txt.write_text(json.dumps(result))
    return result


def evaluate():
    records = json.loads(DATA_JSON.read_text())
    targets = [r for r in records if (VIDEO_DIR / f"{r['video_id']}.mp4").exists()]
    skipped = [r["video_id"] for r in targets if not r["audio_transcript"].strip()]
    targets = [r for r in targets if r["audio_transcript"].strip()]

    holder, rows, tokens = {}, [], {}
    for i, r in enumerate(targets, 1):
        vid = r["video_id"]
        out = transcribe(vid, holder)
        ref, hyp = normalise(r["audio_transcript"]), normalise(out["text"])
        tokens[vid] = (ref, hyp)
        row = {
            "video_id": vid,
            "category": CATEGORY[r["class"]],
            "language": out["language"],
            "duration": round(out["duration"], 1),
            "ref_tokens": len(ref),
            "hyp_tokens": len(hyp),
            "token_overlap": token_overlap(ref, hyp),
            "difflib": seq_ratio(ref, hyp),
        }
        rows.append(row)
        print(f"[{i}/{len(targets)}] {vid} {row['category']:<4} "
              f"overlap {row['token_overlap']:.3f} difflib {row['difflib']:.3f}")
    return rows, skipped, tokens


def chance_baseline(rows, tokens):
    out = {}
    for c in ORDER:
        own = [r["video_id"] for r in rows if r["category"] == c]
        if not own:
            continue
        pairs = [(a, b) for a in own for b in tokens if b != a]
        out[c] = {
            "token_overlap": statistics.mean(token_overlap(tokens[a][0], tokens[b][1]) for a, b in pairs),
            "difflib": statistics.mean(seq_ratio(tokens[a][0], tokens[b][1]) for a, b in pairs),
        }
    return out


def row_lines(rows):
    lines = [f"{'video_id':<14}{'cat':<6}{'lang':<6}{'dur_s':>7}{'ref_tok':>9}{'whisper_tok':>13}"
             f"{'overlap':>9}{'difflib':>9}"]
    for r in sorted(rows, key=lambda r: (ORDER.index(r["category"]), r["video_id"])):
        lines.append(f"{r['video_id']:<14}{r['category']:<6}{r['language']:<6}{r['duration']:>7}"
                     f"{r['ref_tokens']:>9}{r['hyp_tokens']:>13}{r['token_overlap']:>9.3f}{r['difflib']:>9.3f}")
    return lines


def summarise(rows, chance):
    cats = [c for c in ORDER if any(r["category"] == c for r in rows)]
    summary = {}
    for c in cats:
        sel = [r for r in rows if r["category"] == c]
        summary[c] = {
            "n": len(sel),
            "token_overlap_mean": statistics.mean(r["token_overlap"] for r in sel),
            "token_overlap_median": statistics.median(r["token_overlap"] for r in sel),
            "difflib_mean": statistics.mean(r["difflib"] for r in sel),
            "difflib_median": statistics.median(r["difflib"] for r in sel),
            "chance_token_overlap": chance[c]["token_overlap"],
            "chance_difflib": chance[c]["difflib"],
        }
    for c in cats:
        flags = []
        for metric in ("token_overlap_mean", "difflib_mean"):
            others = [summary[o][metric] for o in cats if o != c]
            if others and summary[c][metric] < statistics.mean(others) - FLAG_GAP:
                flags.append(metric.replace("_mean", ""))
        summary[c]["flagged"] = flags

    lines = [f"{'category':<10}{'n':>4}{'overlap_mean':>14}{'overlap_med':>13}{'overlap_chance':>16}"
             f"{'difflib_mean':>14}{'difflib_med':>13}{'difflib_chance':>16}  flag"]
    for c in cats:
        s = summary[c]
        flag = "LOW (" + ", ".join(s["flagged"]) + ")" if s["flagged"] else ""
        lines.append(f"{c:<10}{s['n']:>4}{s['token_overlap_mean']:>14.3f}{s['token_overlap_median']:>13.3f}"
                     f"{s['chance_token_overlap']:>16.3f}{s['difflib_mean']:>14.3f}{s['difflib_median']:>13.3f}"
                     f"{s['chance_difflib']:>16.3f}  {flag}")
    lines.append("")
    lines.append(f"flag rule: category mean more than {FLAG_GAP} below the mean of the other categories")
    lines.append("chance: stored transcript scored against whisper output of every other downloaded video")
    return summary, lines


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    rows, skipped, tokens = evaluate()
    chance = chance_baseline(rows, tokens)
    summary, summary_lines = summarise(rows, chance)

    lines = [f"whisper model: {WHISPER_SIZE}   videos scored: {len(rows)}",
             f"skipped (empty stored transcript): {len(skipped)} {skipped}", ""]
    lines += row_lines(rows) + [""] + summary_lines
    text = "\n".join(lines)
    print("\n" + text)

    RESULTS_TABLE.write_text(text + "\n")
    REPORT.write_text(json.dumps({"rows": rows, "skipped": skipped, "summary": summary}, indent=2))
    print(f"\nsaved -> {REPORT}, {RESULTS_TABLE}")


if __name__ == "__main__":
    main()
