import argparse
import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from dataset import VISUAL_SPLIT_PATH, load_features, load_records, load_visual_features, make_visual_splits
from train import BATCH_SIZE, EPOCHS, LR, VISUAL_CONFIGS, run_experiment
from utils import get_device

SEEDS = [42, 43, 44, 45, 46]
RUN_DIR = Path("runs/multiseed")
OUT = Path("results_multiseed.json")
CHECKPOINTS = {"best": "test", "epoch30": "test_last"}
METRICS = {
    "acc": lambda t: t["overall"]["accuracy"],
    "f1": lambda t: t["overall"]["f1"],
    "cd_f1": lambda t: t["per_category"]["CD"]["f1"],
    "ca_f1": lambda t: t["per_category"]["CA"]["f1"],
}


def split_digest(splits):
    ids = {k: [r["video_id"] for r in v] for k, v in splits.items()}
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def run_seed(seed, splits, features, visual, device, epochs):
    path = RUN_DIR / f"seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    runs = {}
    log = RUN_DIR / f"seed{seed}.log"
    with open(log, "w") as f, redirect_stdout(f):
        for modalities, d_k in VISUAL_CONFIGS:
            tag = f"{modalities}_dk{d_k}"
            r = run_experiment(False, splits, features, device, epochs, seed,
                               modalities=modalities, d_k=d_k, visual=visual,
                               tag=tag, label=f"{tag} seed {seed}", save_checkpoints=False)
            runs[tag] = {
                "best_epoch": r["best_epoch"], "params": r["params"],
                "history": r["history"], "test": r["test"], "test_last": r["test_last"],
            }
    record = {"seed": seed, "split_sha256": split_digest(splits), "runs": runs}
    path.write_text(json.dumps(record, indent=2))
    return record


def summarise(records):
    tags = [f"{m}_dk{k}" for m, k in VISUAL_CONFIGS]
    out = {}
    for tag in tags:
        out[tag] = {}
        for ck, key in CHECKPOINTS.items():
            out[tag][ck] = {}
            for name, get in METRICS.items():
                values = [get(r["runs"][tag][key]) for r in records]
                out[tag][ck][name] = {
                    "values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                }
            out[tag][ck]["best_epochs"] = [r["runs"][tag]["best_epoch"] for r in records] if ck == "best" else None
    return out


def paired(summary, a, b):
    out = {}
    for ck in CHECKPOINTS:
        out[ck] = {}
        for name in METRICS:
            x = np.array(summary[a][ck][name]["values"])
            y = np.array(summary[b][ck][name]["values"])
            d = y - x
            t = stats.ttest_rel(y, x)
            w = stats.wilcoxon(y, x) if np.any(d != 0) else None
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            out[ck][name] = {
                "diffs": d.tolist(), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
                "ci95": [float(d.mean() - half), float(d.mean() + half)],
                "t": float(t.statistic), "p_ttest": float(t.pvalue),
                "p_wilcoxon": float(w.pvalue) if w is not None else None,
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            }
    return out


def print_summary(summary, seeds):
    for ck, title in (("best", "best-val-F1 checkpoint"), ("epoch30", "epoch-30 model")):
        print(f"\n--- {len(seeds)} seeds {seeds} — test set — {title} — mean ± std (sample std) ---")
        print(f"{'config':<12}{'Accuracy':>16}{'macro F1':>16}{'CD F1':>16}{'CA F1':>16}")
        print("-" * 76)
        for tag, s in summary.items():
            cells = "".join(f"{s[ck][m]['mean']:>9.2f} ± {s[ck][m]['std']:<4.2f}" for m in METRICS)
            print(f"{tag:<12}{cells}")


def print_paired(test, a, b, n):
    print(f"\n--- Paired test across {n} seeds: {b} − {a} ---")
    print(f"{'checkpoint':<11}{'metric':<8}{'mean diff':>10}{'std':>7}{'95% CI':>18}{'t':>7}"
          f"{'p (t)':>8}{'p (W)':>8}{'W/L':>6}  per-seed diffs")
    print("-" * 110)
    for ck in CHECKPOINTS:
        for name, r in test[ck].items():
            ci = f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
            pw = f"{r['p_wilcoxon']:.3f}" if r["p_wilcoxon"] is not None else "—"
            diffs = " ".join(f"{d:+.2f}" for d in r["diffs"])
            print(f"{ck:<11}{name:<8}{r['mean_diff']:>+10.2f}{r['std_diff']:>7.2f}{ci:>18}{r['t']:>7.2f}"
                  f"{r['p_ttest']:>8.3f}{pw:>8}{r['wins']:>3}/{r['losses']:<2}  {diffs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else get_device()
    records = load_records()
    splits = make_visual_splits(records)
    digest = split_digest(splits)
    visual = load_visual_features(sorted([r for v in splits.values() for r in v], key=lambda r: r["index"]))
    features = load_features(mmap=False)
    print(f"frozen split {VISUAL_SPLIT_PATH} sha256 {digest[:16]} | "
          f"train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])} | device {device}")

    results = []
    for seed in args.seeds:
        cached = (RUN_DIR / f"seed{seed}.json").exists()
        rec = run_seed(seed, splits, features, visual, device, args.epochs)
        if rec["split_sha256"] != digest:
            raise RuntimeError(f"seed {seed} was run on a different split")
        print(f"seed {seed}: {'loaded from cache' if cached else 'trained'}")
        results.append(rec)

    summary = summarise(results)
    a, b = "text_dk32", "full_dk32"
    test = paired(summary, a, b)
    print_summary(summary, args.seeds)
    print_paired(test, a, b, len(args.seeds))
    print(f"\nall seeds on the same split: {len({r['split_sha256'] for r in results}) == 1}")

    OUT.write_text(json.dumps({
        "config": {
            "seeds": args.seeds, "epochs": args.epochs, "lr": LR, "batch_size": BATCH_SIZE,
            "device": str(device), "split_file": str(VISUAL_SPLIT_PATH), "split_sha256": digest,
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "varies_with_seed": "model init, training shuffle order, dropout",
            "std": "sample std (ddof=1)",
            "paired_test": "paired t-test and Wilcoxon signed-rank across seeds, full_dk32 - text_dk32",
        },
        "summary": summary,
        "paired_text_vs_full_dk32": test,
        "per_seed": results,
    }, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
