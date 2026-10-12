import argparse
import hashlib
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from dataset import (DEFAULT_ENCODER, ENCODERS, FULL_SPLIT_PATH, VISUAL_SPLIT_PATH, encoder_slug,
                     feature_dir_for, file_sha256, load_feature_meta, load_features, load_full_split,
                     load_records, load_visual_features, make_visual_splits)
from train import BATCH_SIZE, EPOCHS, LR, VISUAL_CONFIGS, run_experiment
from utils import get_device

SEEDS = [42, 43, 44, 45, 46]
RUN_ROOT = Path("runs/multiseed")
CANONICAL_OUT = Path("results_multiseed.json")
SPLIT_FILES = {"visual": VISUAL_SPLIT_PATH, "full": FULL_SPLIT_PATH}
CHECKPOINTS = {
    "visual": {"best": "test", "epoch30": "test_last"},
    "full": {"best": "test"},
}
METRICS = {
    "visual": {
        "acc": lambda t: t["overall"]["accuracy"],
        "f1": lambda t: t["overall"]["f1"],
        "cd_f1": lambda t: t["per_category"]["CD"]["f1"],
        "ca_f1": lambda t: t["per_category"]["CA"]["f1"],
    },
    "full": {
        "acc": lambda t: t["overall"]["accuracy"],
        "f1": lambda t: t["overall"]["f1"],
        "cd_f1": lambda t: t["per_category"]["CD"]["f1"],
        "ce_f1": lambda t: t["per_category"]["CE"]["f1"],
        "sv_f1": lambda t: t["per_category"]["SV"]["f1"],
        "ca_f1": lambda t: t["per_category"]["CA"]["f1"],
    },
}


def configs(subset):
    if subset == "visual":
        return [(f"{m}_dk{k}", {"modalities": m, "d_k": k}) for m, k in VISUAL_CONFIGS]
    return [("without_flag", {"modalities": "text", "d_k": None})]


def run_dir(subset, encoder):
    return RUN_ROOT / subset / encoder_slug(encoder)


def split_digest(splits):
    ids = {k: [r["video_id"] for r in v] for k, v in splits.items()}
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()


def load_study_inputs(subset, encoder):
    records = load_records()
    splits = make_visual_splits(records) if subset == "visual" else load_full_split(records)
    visual = None
    if subset == "visual":
        visual = load_visual_features(sorted([r for v in splits.values() for r in v], key=lambda r: r["index"]))
    feature_dir = feature_dir_for(encoder)
    meta = load_feature_meta(feature_dir)
    features = load_features(feature_dir, mmap=False)
    return splits, features, meta, visual


def run_seed(seed, subset, encoder, splits, features, meta, visual, device, epochs):
    directory = run_dir(subset, encoder)
    path = directory / f"seed{seed}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("encoder", DEFAULT_ENCODER) != encoder:
            raise RuntimeError(f"{path} was run with encoder {rec.get('encoder')}, expected {encoder}")
        if rec.get("feature_files_sha256") not in (None, meta["files_sha256"]):
            raise RuntimeError(f"{path} was run on different {encoder} features; delete it to rerun")
        return rec
    directory.mkdir(parents=True, exist_ok=True)
    slug = encoder_slug(encoder)
    runs = {}
    with open(directory / f"seed{seed}.log", "w") as f, redirect_stdout(f):
        for key, cfg in configs(subset):
            r = run_experiment(False, splits, features, device, epochs, seed, text_dim=meta["hidden_size"],
                               visual=visual if subset == "visual" else None,
                               tag=f"{slug}_{subset}_{key}", label=f"{slug} {subset} {key} seed {seed}",
                               save_checkpoints=False, **cfg)
            runs[key] = {k: r[k] for k in ("best_epoch", "params", "history", "test", "test_last") if k in r}
    record = {"seed": seed, "subset": subset, "encoder": encoder,
              "feature_files_sha256": meta["files_sha256"], "split_sha256": split_digest(splits), "runs": runs}
    path.write_text(json.dumps(record, indent=2))
    return record


def run_study(subset, encoder, seeds, epochs, device):
    splits, features, meta, visual = load_study_inputs(subset, encoder)
    digest = split_digest(splits)
    print(f"{subset} split {SPLIT_FILES[subset]} sha256 {digest[:16]} | encoder {encoder} | "
          f"train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])} | device {device}")
    records = []
    for seed in seeds:
        cached = (run_dir(subset, encoder) / f"seed{seed}.json").exists()
        rec = run_seed(seed, subset, encoder, splits, features, meta, visual, device, epochs)
        if rec["split_sha256"] != digest:
            raise RuntimeError(f"seed {seed} was run on a different split")
        print(f"seed {seed}: {'loaded from cache' if cached else 'trained'}")
        records.append(rec)
    config = {
        "subset": subset, "encoder": encoder, "seeds": list(seeds), "epochs": epochs, "lr": LR,
        "batch_size": BATCH_SIZE, "device": str(device), "split_file": str(SPLIT_FILES[subset]),
        "split_file_sha256": file_sha256(SPLIT_FILES[subset]), "split_sha256": digest,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "feature_files_sha256": meta["files_sha256"],
        "varies_with_seed": "model init, training shuffle order, dropout",
        "std": "sample std (ddof=1)",
    }
    return records, summarise(records, subset), config


def summarise(records, subset):
    out = {}
    for key, _ in configs(subset):
        out[key] = {}
        for ck, test_key in CHECKPOINTS[subset].items():
            out[key][ck] = {}
            for name, get in METRICS[subset].items():
                values = [get(r["runs"][key][test_key]) for r in records]
                out[key][ck][name] = {
                    "values": values, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)),
                }
            out[key][ck]["best_epochs"] = [r["runs"][key]["best_epoch"] for r in records] if ck == "best" else None
    return out


def paired_diffs(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = y - x
    t = stats.ttest_rel(y, x)
    w = stats.wilcoxon(y, x) if np.any(d != 0) else None
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {
        "diffs": d.tolist(), "mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)),
        "ci95": [float(d.mean() - half), float(d.mean() + half)],
        "t": float(t.statistic), "p_ttest": float(t.pvalue),
        "p_wilcoxon": float(w.pvalue) if w is not None else None,
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
    }


def paired(summary, a, b, subset="visual"):
    return {
        ck: {name: paired_diffs(summary[a][ck][name]["values"], summary[b][ck][name]["values"])
             for name in METRICS[subset]}
        for ck in CHECKPOINTS[subset]
    }


def print_summary(summary, seeds, subset):
    names = list(METRICS[subset])
    for ck in CHECKPOINTS[subset]:
        title = "best-val-F1 checkpoint" if ck == "best" else "epoch-30 model"
        print(f"\n--- {len(seeds)} seeds {seeds} — test set — {title} — mean ± std (sample std) ---")
        print(f"{'config':<14}" + "".join(f"{n:>16}" for n in names))
        print("-" * (14 + 16 * len(names)))
        for key, s in summary.items():
            cells = "".join(f"{s[ck][m]['mean']:>9.2f} ± {s[ck][m]['std']:<4.2f}" for m in names)
            print(f"{key:<14}{cells}")


def print_paired(test, a, b, n):
    print(f"\n--- Paired test across {n} seeds: {b} − {a} ---")
    print(f"{'checkpoint':<11}{'metric':<8}{'mean diff':>10}{'std':>7}{'95% CI':>18}{'t':>7}"
          f"{'p (t)':>8}{'p (W)':>8}{'W/L':>6}  per-seed diffs")
    print("-" * 110)
    for ck, per_metric in test.items():
        for name, r in per_metric.items():
            ci = f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
            pw = f"{r['p_wilcoxon']:.3f}" if r["p_wilcoxon"] is not None else "—"
            diffs = " ".join(f"{d:+.2f}" for d in r["diffs"])
            print(f"{ck:<11}{name:<8}{r['mean_diff']:>+10.2f}{r['std_diff']:>7.2f}{ci:>18}{r['t']:>7.2f}"
                  f"{r['p_ttest']:>8.3f}{pw:>8}{r['wins']:>3}/{r['losses']:<2}  {diffs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["visual", "full"], default="visual")
    parser.add_argument("--encoder", choices=ENCODERS, default=DEFAULT_ENCODER)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device")
    parser.add_argument("--out")
    args = parser.parse_args()

    canonical = (args.subset == "visual" and args.encoder == DEFAULT_ENCODER
                 and args.seeds == SEEDS and args.epochs == EPOCHS)
    if not args.out and not canonical:
        parser.error(f"--out is required except for the canonical visual study ({DEFAULT_ENCODER}, "
                     f"seeds {SEEDS}, {EPOCHS} epochs), which writes {CANONICAL_OUT}")
    out = Path(args.out) if args.out else CANONICAL_OUT

    device = torch.device(args.device) if args.device else get_device()
    records, summary, config = run_study(args.subset, args.encoder, args.seeds, args.epochs, device)
    print_summary(summary, args.seeds, args.subset)

    payload = {"config": config, "summary": summary}
    if args.subset == "visual":
        a, b = "text_dk32", "full_dk32"
        test = paired(summary, a, b)
        print_paired(test, a, b, len(args.seeds))
        config["paired_test"] = "paired t-test and Wilcoxon signed-rank across seeds, full_dk32 - text_dk32"
        payload["paired_text_vs_full_dk32"] = test
    print(f"\nall seeds on the same split: {len({r['split_sha256'] for r in records}) == 1}")

    payload["per_seed"] = records
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
