import argparse
import copy
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import (FMNVDDataset, VISUAL_SPLIT_PATH, load_features, load_records,
                     load_visual_features, make_visual_splits)
from determinism import make_generator, seed_worker, set_seed
from model import FMNVD
from multiseed import SEEDS, split_digest
from train import BATCH_SIZE, EPOCHS, LR, metrics, run_epoch, visual_metrics
from utils import get_device

D_K = 32
MODELS = ["text", "full"]
THRESHOLDS = np.round(np.arange(0.05, 0.95 + 1e-9, 0.01), 2)
COMBOS = {
    "baseline": (False, False),
    "weighted": (True, False),
    "threshold": (False, True),
    "weighted+threshold": (True, True),
}
RUN_DIR = Path("runs/enhance")
OUT = Path("results_enhance.json")
METRICS = {
    "acc": lambda t: t["overall"]["accuracy"],
    "f1": lambda t: t["overall"]["f1"],
    "cd_f1": lambda t: t["per_category"]["CD"]["f1"],
    "ca_f1": lambda t: t["per_category"]["CA"]["f1"],
}
PAIRED_METRICS = ["f1", "cd_f1"]


def class_weights(records):
    labels = np.array([r["label"] for r in records])
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    inverse = 1.0 / counts
    return inverse / inverse.mean(), counts


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        video = {k: batch[k].to(device) for k in ("clip", "motion") if k in batch}
        logits = model(batch["title"].to(device), batch["speech"].to(device),
                       batch["title_mask"].to(device), batch["speech_mask"].to(device),
                       batch["has_transcript"].to(device), **video)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def tune_threshold(p_val, y_val):
    scores = np.array([f1_score(y_val, (p_val > t).astype(int), average="macro", zero_division=0)
                       for t in THRESHOLDS])
    best = np.flatnonzero(scores == scores.max())
    pick = best[np.argmin(np.abs(THRESHOLDS[best] - 0.5))]
    return float(THRESHOLDS[pick]), float(100 * scores[pick])


def train_one(modalities, weighted, seed, splits, features, visual, device, epochs):
    set_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        ds = FMNVDDataset(splits[name], features=features,
                          visual=visual if modalities == "full" else None)
        loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(name == "train"),
                                   generator=make_generator(seed), worker_init_fn=seed_worker,
                                   drop_last=False)

    model = FMNVD(modalities=modalities, d_k=D_K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weights, counts = class_weights(splits["train"])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device) if weighted else None
    )

    best_f1, best_epoch, best_state = -1.0, -1, None
    for epoch in range(1, epochs + 1):
        run_epoch(model, loaders["train"], device, criterion, optimizer)
        _, va_y, va_p = run_epoch(model, loaders["val"], device, criterion)
        va_f1 = metrics(va_y, va_p)["f1"]
        print(f"{modalities} weighted={weighted} seed={seed} epoch {epoch:>2} val macro F1 {va_f1:.2f}")
        if va_f1 > best_f1:
            best_f1, best_epoch, best_state = va_f1, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    p_val, y_val = predict_proba(model, loaders["val"], device)
    p_test, y_test = predict_proba(model, loaders["test"], device)
    threshold, val_f1_tuned = tune_threshold(p_val, y_val)
    out = {
        "best_epoch": best_epoch, "best_val_f1": best_f1,
        "class_weights": weights.tolist() if weighted else None,
        "train_class_counts": counts.tolist(),
        "threshold": threshold, "val_f1_at_threshold": val_f1_tuned,
        "val_f1_at_0.5": float(100 * f1_score(y_val, (p_val > 0.5).astype(int), average="macro", zero_division=0)),
    }
    out["test_0.5"] = visual_metrics(y_test, (p_test > 0.5).astype(int), splits["test"])
    out["test_tuned"] = visual_metrics(y_test, (p_test > threshold).astype(int), splits["test"])
    return out


def run_seed(seed, splits, features, visual, device, epochs):
    path = RUN_DIR / f"seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    runs = {}
    with open(RUN_DIR / f"seed{seed}.log", "w") as f, redirect_stdout(f):
        for modalities in MODELS:
            for weighted in (False, True):
                runs[f"{modalities}_{'weighted' if weighted else 'unweighted'}"] = train_one(
                    modalities, weighted, seed, splits, features, visual, device, epochs)
    record = {"seed": seed, "split_sha256": split_digest(splits), "runs": runs}
    path.write_text(json.dumps(record, indent=2))
    return record


def combo_result(record, modalities, combo):
    weighted, tuned = COMBOS[combo]
    run = record["runs"][f"{modalities}_{'weighted' if weighted else 'unweighted'}"]
    return {"test": run["test_tuned" if tuned else "test_0.5"],
            "threshold": run["threshold"] if tuned else 0.5, "best_epoch": run["best_epoch"]}


def summarise(records):
    out = {}
    for modalities in MODELS:
        for combo in COMBOS:
            rows = [combo_result(r, modalities, combo) for r in records]
            entry = {name: {"values": [get(x["test"]) for x in rows]} for name, get in METRICS.items()}
            for v in entry.values():
                v["mean"], v["std"] = float(np.mean(v["values"])), float(np.std(v["values"], ddof=1))
            entry["thresholds"] = [x["threshold"] for x in rows]
            entry["best_epochs"] = [x["best_epoch"] for x in rows]
            entry["majority_acc"] = rows[0]["test"]["overall"]["majority_acc"]
            out[f"{modalities}|{combo}"] = entry
    return out


def paired_diffs(x, y):
    x, y = np.asarray(x), np.asarray(y)
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


def paired_tests(summary):
    tests = {}
    for modalities in MODELS:
        a, b = f"{modalities}|baseline", f"{modalities}|weighted+threshold"
        tests[f"{b} vs {a}"] = {m: paired_diffs(summary[a][m]["values"], summary[b][m]["values"])
                                for m in PAIRED_METRICS}
    gain = {m: {mod: np.array(summary[f"{mod}|weighted+threshold"][m]["values"])
                - np.array(summary[f"{mod}|baseline"][m]["values"]) for mod in MODELS}
            for m in PAIRED_METRICS}
    tests["interaction: (full gain) vs (text gain), weighted+threshold over baseline"] = {
        m: paired_diffs(gain[m]["text"], gain[m]["full"]) for m in PAIRED_METRICS
    }
    return tests


def print_summary(summary, seeds):
    print(f"\n--- {len(seeds)} seeds {seeds} — test set — best-val-F1 checkpoint, d_k={D_K} — mean ± std ---")
    print(f"{'model | combination':<28}{'Accuracy':>15}{'Maj':>6}{'macro F1':>15}{'CD F1':>15}{'CA F1':>15}"
          f"   thresholds")
    print("-" * 124)
    for key, e in summary.items():
        if key.endswith("|baseline") and not key.startswith(MODELS[0]):
            print("-" * 124)
        cells = f"{e['acc']['mean']:>8.2f} ± {e['acc']['std']:<4.2f}{e['majority_acc']:>6.1f}"
        cells += "".join(f"{e[m]['mean']:>8.2f} ± {e[m]['std']:<4.2f}" for m in ("f1", "cd_f1", "ca_f1"))
        print(f"{key:<28}{cells}   {' '.join(f'{t:.2f}' for t in e['thresholds'])}")


def print_paired(tests, n):
    print(f"\n--- Paired tests across {n} seeds (test set, best-val-F1 checkpoint) ---")
    print(f"{'comparison':<44}{'metric':<7}{'mean diff':>10}{'std':>7}{'95% CI':>18}{'t':>7}"
          f"{'p (t)':>8}{'p (W)':>8}{'W/L':>6}  per-seed diffs")
    print("-" * 140)
    for name, per_metric in tests.items():
        label = name if len(name) <= 43 else name[:40] + "..."
        for m, r in per_metric.items():
            ci = f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
            pw = f"{r['p_wilcoxon']:.3f}" if r["p_wilcoxon"] is not None else "—"
            print(f"{label:<44}{m:<7}{r['mean_diff']:>+10.2f}{r['std_diff']:>7.2f}{ci:>18}{r['t']:>7.2f}"
                  f"{r['p_ttest']:>8.3f}{pw:>8}{r['wins']:>3}/{r['losses']:<2}  "
                  + " ".join(f"{d:+.2f}" for d in r["diffs"]))


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
    weights, counts = class_weights(splits["train"])
    print(f"frozen split {VISUAL_SPLIT_PATH} sha256 {digest[:16]} | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])} | device {device}")
    print(f"train class counts real/fake {counts.astype(int).tolist()} -> weights {np.round(weights, 4).tolist()} "
          f"(mean {weights.mean():.2f})")
    print(f"threshold sweep on val: {THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f} step 0.01 ({len(THRESHOLDS)} values)")

    results = []
    for seed in args.seeds:
        cached = (RUN_DIR / f"seed{seed}.json").exists()
        rec = run_seed(seed, splits, features, visual, device, args.epochs)
        if rec["split_sha256"] != digest:
            raise RuntimeError(f"seed {seed} was run on a different split")
        print(f"seed {seed}: {'loaded from cache' if cached else 'trained'}")
        results.append(rec)

    summary = summarise(results)
    tests = paired_tests(summary)
    print_summary(summary, args.seeds)
    print_paired(tests, len(args.seeds))
    print(f"\nFour combinations x two models are compared without multiple-comparison correction.")

    OUT.write_text(json.dumps({
        "config": {
            "seeds": args.seeds, "epochs": args.epochs, "lr": LR, "batch_size": BATCH_SIZE, "d_k": D_K,
            "device": str(device), "split_file": str(VISUAL_SPLIT_PATH), "split_sha256": digest,
            "class_weights": {"train_counts_real_fake": counts.tolist(), "weights_real_fake": weights.tolist(),
                              "rule": "inverse class frequency on the training split, normalised to mean 1"},
            "threshold_sweep": {"from": float(THRESHOLDS[0]), "to": float(THRESHOLDS[-1]), "step": 0.01,
                                "fit_on": "validation", "objective": "macro F1",
                                "tie_break": "closest to 0.5", "rule": "predict fake iff p_fake > threshold"},
            "checkpoint": "best val macro F1 at argmax (threshold 0.5), then threshold tuned on val",
            "std": "sample std (ddof=1)",
            "multiple_comparisons": "none applied",
        },
        "summary": summary,
        "paired_tests": tests,
        "per_seed": results,
    }, indent=2))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
