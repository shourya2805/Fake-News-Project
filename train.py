"""Train FMNVD (title + audio) and evaluate overall and per category.

Hyperparameters from paper Section 4.2 — Adam, lr 1e-3, batch 128, 30 epochs,
cross-entropy. Everything else is recorded in DEVIATIONS.md.

Metrics are reported under BOTH averaging conventions:
  * macro  — unweighted mean over the real and fake classes (PRIMARY; see
             DEVIATIONS.md for why this is what the paper must have used)
  * fake   — the fake class alone treated as the positive class

Runs the whole thing twice — once with `has_transcript` as a model input and once
without — plus a transcript-presence-only baseline, so the leakage from the
class-correlated transcript missingness is visible. Results -> `results.json`.

    python train.py

Visual split (frozen REAL+CD+CA subset, 1,510 records): text and full
modalities at d_k 32 and 128, best-val-F1 and epoch-30 test results, majority
baseline beside every accuracy. Results -> `results_visual.json`.

    python train.py --subset visual
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from dataset import (FAKE_CATEGORIES, FMNVDDataset, SEED, VISUAL_CATEGORIES,
                     load_features, load_records, load_visual_features,
                     make_splits, make_visual_splits)
from determinism import make_generator, seed_worker, set_seed
from model import FMNVD
from utils import get_device

LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 30
CHECKPOINT_DIR = "checkpoints"
RESULTS_PATH = "results.json"
VISUAL_RESULTS_PATH = "results_visual.json"
VISUAL_FAKE_CATEGORIES = [c for c in VISUAL_CATEGORIES if c != "REAL"]

# paper Table 4, "w/o Frames" row
TARGETS = {"accuracy": 72.50, "f1": 71.94, "precision": 72.11, "recall": 73.56}
METRIC_KEYS = ("accuracy", "f1", "precision", "recall")


def metrics(y_true, y_pred):
    """Percent metrics under both averaging conventions.

    Top-level keys are macro (primary); `*_fake` are for the fake class alone.
    """
    kw = dict(zero_division=0)
    return {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "f1": 100 * f1_score(y_true, y_pred, average="macro", **kw),
        "precision": 100 * precision_score(y_true, y_pred, average="macro", **kw),
        "recall": 100 * recall_score(y_true, y_pred, average="macro", **kw),
        "f1_fake": 100 * f1_score(y_true, y_pred, pos_label=1, **kw),
        "precision_fake": 100 * precision_score(y_true, y_pred, pos_label=1, **kw),
        "recall_fake": 100 * recall_score(y_true, y_pred, pos_label=1, **kw),
    }


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass. Returns (mean loss, y_true, y_pred)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []

    for batch in loader:
        title = batch["title"].to(device)
        speech = batch["speech"].to(device)
        title_mask = batch["title_mask"].to(device)
        speech_mask = batch["speech_mask"].to(device)
        flag = batch["has_transcript"].to(device)
        labels = batch["label"].to(device)
        video = {k: batch[k].to(device) for k in ("clip", "motion") if k in batch}

        with torch.set_grad_enabled(train):
            logits = model(title, speech, title_mask, speech_mask, flag, **video)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

    return total_loss / n, np.concatenate(y_true), np.concatenate(y_pred)


def per_category_metrics(y_true, y_pred, records, fake_categories=FAKE_CATEGORIES):
    """Per-category = the real samples plus only that category's fakes."""
    categories = np.array([r["category"] for r in records])
    out = {}
    for cat in fake_categories:
        sel = (categories == "REAL") | (categories == cat)
        m = metrics(y_true[sel], y_pred[sel])
        m["n"] = int(sel.sum())
        m["n_fake"] = int((categories[sel] == cat).sum())
        out[cat] = m
    return out


def evaluate(model, loader, device, criterion, records, fake_categories=FAKE_CATEGORIES):
    loss, y_true, y_pred = run_epoch(model, loader, device, criterion)
    overall = metrics(y_true, y_pred)
    overall["n"] = int(len(y_true))
    return {
        "loss": loss,
        "overall": overall,
        "per_category": per_category_metrics(y_true, y_pred, records, fake_categories),
    }


def evaluate_visual(model, loader, device, criterion, records):
    loss, y_true, y_pred = run_epoch(model, loader, device, criterion)
    return {"loss": loss, **visual_metrics(y_true, y_pred, records)}


def visual_metrics(y_true, y_pred, records):
    categories = np.array([r["category"] for r in records])
    overall = metrics(y_true, y_pred)
    overall.update(n=int(len(y_true)), n_fake=int(y_true.sum()),
                   majority_acc=100 * float((y_true == 0).mean()))
    per_category = per_category_metrics(y_true, y_pred, records, VISUAL_FAKE_CATEGORIES)
    for cat, m in per_category.items():
        sel = (categories == "REAL") | (categories == cat)
        m["majority_acc"] = 100 * float((y_true[sel] == 0).mean())
    real = categories == "REAL"
    per_category["REAL"] = {
        "accuracy": 100 * accuracy_score(y_true[real], y_pred[real]),
        "n": int(real.sum()), "n_fake": 0, "majority_acc": 100.0,
    }
    return {"overall": overall, "per_category": per_category}


def transcript_only_baseline(records):
    """Leakage probe: predict `fake` iff a transcript exists. No text read at all.

    If this scores well on a category, that category's result is partly an
    artefact of the missingness pattern rather than of any language understanding.
    """
    y_true = np.array([r["label"] for r in records])
    y_pred = np.array([int(r["has_transcript"]) for r in records])
    overall = metrics(y_true, y_pred)
    overall["n"] = int(len(y_true))
    return {
        "overall": overall,
        "per_category": per_category_metrics(y_true, y_pred, records),
    }


def run_experiment(use_flag, splits, features, device, epochs, seed,
                   modalities="text", d_k=None, visual=None, tag=None, label=None,
                   save_checkpoints=True):
    tag = tag or ("with_flag" if use_flag else "without_flag")
    label = label or ("WITH has_transcript" if use_flag else "WITHOUT has_transcript")
    print(f"\n{'=' * 74}\nRUN: {label}\n{'=' * 74}")

    set_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        ds = FMNVDDataset(splits[name], features=features,
                          visual=visual if modalities == "full" else None)
        loaders[name] = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=(name == "train"),
            generator=make_generator(seed), worker_init_fn=seed_worker,
            drop_last=False,
        )

    model = FMNVD(use_has_transcript=use_flag, modalities=modalities, d_k=d_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params:,} | device {device} | lr {LR} | batch {BATCH_SIZE} "
          f"| epochs {epochs} | steps/epoch {len(loaders['train'])}")
    print(f"\n{'epoch':>5}{'train_loss':>12}{'val_loss':>10}{'val_acc':>9}"
          f"{'val_f1_macro':>14}{'best':>6}")

    history, best_f1, best_epoch, best_state = [], -1.0, -1, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_y, tr_p = run_epoch(model, loaders["train"], device,
                                        criterion, optimizer)
        va_loss, va_y, va_p = run_epoch(model, loaders["val"], device, criterion)
        va = metrics(va_y, va_p)
        is_best = va["f1"] > best_f1          # selection on macro F1
        if is_best:
            best_f1, best_epoch = va["f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_acc": 100 * accuracy_score(tr_y, tr_p),
            "val_acc": va["accuracy"], "val_f1_macro": va["f1"],
            "val_f1_fake": va["f1_fake"],
        })
        print(f"{epoch:>5}{tr_loss:>12.4f}{va_loss:>10.4f}{va['accuracy']:>9.2f}"
              f"{va['f1']:>14.2f}{'  <--' if is_best else '':>6}")

    print(f"\nbest checkpoint: epoch {best_epoch} (val macro F1 {best_f1:.2f})")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    result = {
        "tag": tag, "label": label, "use_has_transcript": use_flag,
        "params": n_params, "best_epoch": best_epoch, "best_val_f1_macro": best_f1,
        "history": history,
    }
    if visual is not None:
        result.update(modalities=modalities, d_k=model.d_k)
        if save_checkpoints:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"last_{tag}.pt"))
        result["test_last"] = evaluate_visual(model, loaders["test"], device, criterion, splits["test"])

    model.load_state_dict(best_state)
    if save_checkpoints:
        torch.save(best_state, os.path.join(CHECKPOINT_DIR, f"best_{tag}.pt"))

    if visual is not None:
        result["test"] = evaluate_visual(model, loaders["test"], device, criterion, splits["test"])
    else:
        result["test"] = evaluate(model, loaders["test"], device, criterion, splits["test"])
    return result


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

CAT_NAMES = {"CD": "CD (ft)", "CE": "CE (fv)", "SV": "SV (fa)", "CA": "CA (fc)"}


def _row(name, m, suffix, n=None, n_fake=None):
    f1 = m["f1" + suffix]
    pr = m["precision" + suffix]
    rc = m["recall" + suffix]
    head = f"{name:<12}"
    head += f"{n:>6}{n_fake:>7}" if n is not None else f"{'':>6}{'':>7}"
    return head + f"{m['accuracy']:>10.2f}{f1:>9.2f}{pr:>11.2f}{rc:>9.2f}"


def print_table3(result, suffix, avg_name):
    """Paper Table 3 layout: per-category rows plus an overall row."""
    t = result["test"]
    print(f"\n--- Table 3 layout — test set — {result['label']} — {avg_name} ---")
    print(f"{'Category':<12}{'n':>6}{'fakes':>7}{'Accuracy':>10}{'F1':>9}"
          f"{'Precision':>11}{'Recall':>9}")
    print("-" * 64)
    for cat in FAKE_CATEGORIES:
        m = t["per_category"][cat]
        print(_row(CAT_NAMES[cat], m, suffix, m["n"], m["n_fake"]))
    print("-" * 64)
    o = t["overall"]
    print(_row("Overall", o, suffix, o["n"], 0).replace(f"{0:>7}", f"{'':>7}"))


def print_target_table(results, suffix, avg_name):
    print(f"\n--- Ours vs paper Table 4 \"w/o Frames\" — {avg_name} ---")
    print(f"{'Metric':<12}{'Paper':>8}" + "".join(
        f"{r['tag']:>16}{'diff':>9}" for r in results))
    print("-" * (20 + 25 * len(results)))
    for key in METRIC_KEYS:
        k = key if key == "accuracy" else key + suffix
        row = f"{key.capitalize():<12}{TARGETS[key]:>8.2f}"
        for r in results:
            ours = r["test"]["overall"][k]
            diff = 100 * (ours - TARGETS[key]) / TARGETS[key]
            row += f"{ours:>16.2f}{diff:>+8.1f}%"
        print(row)


def print_leakage_tables(results, baseline):
    by_tag = {r["tag"]: r for r in results}
    a, b = by_tag["without_flag"], by_tag["with_flag"]

    print(f"\n--- Leakage effect of the has_transcript flag (test macro F1) ---")
    print(f"{'Subset':<12}{'without flag':>14}{'with flag':>12}{'delta':>9}")
    print("-" * 47)
    for cat in FAKE_CATEGORIES:
        x = a["test"]["per_category"][cat]["f1"]
        y = b["test"]["per_category"][cat]["f1"]
        print(f"{cat:<12}{x:>14.2f}{y:>12.2f}{y - x:>+9.2f}")
    print("-" * 47)
    x, y = a["test"]["overall"]["f1"], b["test"]["overall"]["f1"]
    print(f"{'Overall':<12}{x:>14.2f}{y:>12.2f}{y - x:>+9.2f}")

    print(f"\n--- Transcript-presence-only baseline (predict fake iff a "
          f"transcript exists; reads no text) ---")
    print(f"{'Subset':<12}{'n':>6}{'fakes':>7}{'Accuracy':>10}{'macro F1':>10}"
          f"{'fake F1':>10}")
    print("-" * 55)
    for cat in FAKE_CATEGORIES:
        m = baseline["per_category"][cat]
        print(f"{CAT_NAMES[cat]:<12}{m['n']:>6}{m['n_fake']:>7}"
              f"{m['accuracy']:>10.2f}{m['f1']:>10.2f}{m['f1_fake']:>10.2f}")
    print("-" * 55)
    o = baseline["overall"]
    print(f"{'Overall':<12}{o['n']:>6}{'':>7}{o['accuracy']:>10.2f}"
          f"{o['f1']:>10.2f}{o['f1_fake']:>10.2f}")


VISUAL_CONFIGS = [("text", 32), ("text", 128), ("full", 32), ("full", 128)]


def print_visual_table(result, key, title):
    t = result[key]
    print(f"\n--- {result['label']} — {title} ---")
    print(f"{'Subset':<14}{'n':>5}{'fakes':>6}{'Acc':>8}{'Maj':>7}"
          f"{'F1':>8}{'P':>8}{'R':>8}{'F1fake':>9}{'Pfake':>8}{'Rfake':>8}")
    print("-" * 89)
    r = t["per_category"]["REAL"]
    print(f"{'REAL only':<14}{r['n']:>5}{0:>6}{r['accuracy']:>8.2f}{r['majority_acc']:>7.1f}"
          f"{'—':>8}{'—':>8}{'—':>8}{'—':>9}{'—':>8}{'—':>8}")
    rows = [(f"REAL+{c}", t["per_category"][c]) for c in VISUAL_FAKE_CATEGORIES]
    rows.append(("Overall", t["overall"]))
    for name, m in rows:
        if name == "Overall":
            print("-" * 89)
        print(f"{name:<14}{m['n']:>5}{m['n_fake']:>6}{m['accuracy']:>8.2f}{m['majority_acc']:>7.1f}"
              f"{m['f1']:>8.2f}{m['precision']:>8.2f}{m['recall']:>8.2f}"
              f"{m['f1_fake']:>9.2f}{m['precision_fake']:>8.2f}{m['recall_fake']:>8.2f}")


def print_visual_comparison(results):
    maj = results[0]["test"]["overall"]["majority_acc"]
    for key, title in (("test", "best-val-F1 checkpoint"), ("test_last", "epoch-30 model")):
        print(f"\n--- Four-way comparison — test set — {title} — majority baseline {maj:.2f}% (predict real) ---")
        print(f"{'config':<16}{'params':>11}{'ep':>4}{'Acc':>8}{'Maj':>7}{'F1':>8}{'P':>8}{'R':>8}"
              f"{'F1fake':>9}{'CD F1':>8}{'CA F1':>8}{'REALacc':>9}")
        print("-" * 104)
        for r in results:
            t = r[key]
            o, pc = t["overall"], t["per_category"]
            ep = r["best_epoch"] if key == "test" else len(r["history"])
            print(f"{r['tag']:<16}{r['params']:>11,}{ep:>4}{o['accuracy']:>8.2f}{o['majority_acc']:>7.1f}"
                  f"{o['f1']:>8.2f}{o['precision']:>8.2f}{o['recall']:>8.2f}{o['f1_fake']:>9.2f}"
                  f"{pc['CD']['f1']:>8.2f}{pc['CA']['f1']:>8.2f}{pc['REAL']['accuracy']:>9.2f}")


def main_visual(args, device):
    records = load_records()
    splits = make_visual_splits(records)
    subset = [r for recs in splits.values() for r in recs]
    visual = load_visual_features(sorted(subset, key=lambda r: r["index"]))
    features = load_features(mmap=False)
    print(f"visual subset {len(subset)} | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])} | seed {args.seed}")
    print(f"clip {visual['clip'].shape} {visual['clip'].dtype} | motion {visual['motion'].shape} "
          f"{visual['motion'].dtype} | id lists match frozen split: True")

    configs = [(m, k) for m, k in VISUAL_CONFIGS if m in args.modalities and k in args.d_k]
    results = []
    for modalities, d_k in configs:
        tag = f"visual_{modalities}_dk{d_k}"
        label = f"{modalities.upper()} | d_k={d_k} | visual split"
        results.append(run_experiment(False, splits, features, device, args.epochs, args.seed,
                                      modalities=modalities, d_k=d_k, visual=visual,
                                      tag=tag, label=label))

    print(f"\n\n{'#' * 74}\n# FINAL RESULTS — visual split — test set\n{'#' * 74}")
    print("Acc/Maj in %. Maj = accuracy of always predicting real on that subset. "
          "F1/P/R macro (PRIMARY); *fake = fake class only.")
    for r in results:
        print_visual_table(r, "test", f"best-val-F1 checkpoint (epoch {r['best_epoch']})")
        print_visual_table(r, "test_last", f"epoch-{len(r['history'])} model")
    print_visual_comparison(results)

    out = args.out or VISUAL_RESULTS_PATH
    with open(out, "w") as f:
        json.dump({
            "config": {
                "lr": LR, "batch_size": BATCH_SIZE, "epochs": args.epochs,
                "optimizer": "Adam", "loss": "cross_entropy", "seed": args.seed,
                "device": str(device), "subset": "frozen visual split (REAL+CD+CA)",
                "split_file": "splits/visual_split.json", "split_sizes": {k: len(v) for k, v in splits.items()},
                "primary_averaging": "macro",
                "checkpoint_selection": "best val macro F1; epoch-30 model also reported",
                "has_transcript_input": False,
            },
            "majority_baseline_test_acc": results[0]["test"]["overall"]["majority_acc"] if results else None,
            "runs": results,
        }, f, indent=2)
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None, help="override (mps/cpu)")
    ap.add_argument("--subset", choices=["full", "visual"], default="full")
    ap.add_argument("--modalities", nargs="+", choices=["text", "full"])
    ap.add_argument("--d-k", dest="d_k", nargs="+", type=int, choices=[32, 128])
    ap.add_argument("--out")
    args = ap.parse_args()

    set_seed(args.seed)
    if args.subset == "visual":
        args.modalities = args.modalities or ["text", "full"]
        args.d_k = args.d_k or [32, 128]
        device = torch.device(args.device) if args.device else get_device()
        return main_visual(args, device)
    if args.modalities not in (None, ["text"]):
        ap.error("--subset full has no video features; only --modalities text is available")
    if args.d_k not in (None, [32]) and not args.out:
        ap.error("--subset full with d_k other than 32 needs --out, to keep results.json as the Phase 1 record")

    device = torch.device(args.device) if args.device else get_device()
    records = load_records()
    splits = make_splits(records, seed=args.seed)
    features = load_features(mmap=False)

    print(f"records {len(records)} | train {len(splits['train'])} "
          f"val {len(splits['val'])} test {len(splits['test'])} | seed {args.seed}")

    d_k = args.d_k[0] if args.d_k else None
    results = [
        run_experiment(False, splits, features, device, args.epochs, args.seed, d_k=d_k),
        run_experiment(True, splits, features, device, args.epochs, args.seed, d_k=d_k),
    ]
    baseline = transcript_only_baseline(splits["test"])

    print(f"\n\n{'#' * 74}\n# FINAL RESULTS — test set\n{'#' * 74}")
    for suffix, avg_name in (("", "macro-averaged (PRIMARY)"),
                             ("_fake", "fake class only")):
        for r in results:
            print_table3(r, suffix, avg_name)
    print()
    print_target_table(results, "", "macro-averaged (PRIMARY)")
    print_target_table(results, "_fake", "fake class only")
    print("\n(diff = (ours - paper) / paper x 100; CLAUDE.md tolerance is +/-5-10%)")
    print_leakage_tables(results, baseline)

    with open(args.out or RESULTS_PATH, "w") as f:
        json.dump({
            "config": {
                "lr": LR, "batch_size": BATCH_SIZE, "epochs": args.epochs,
                "optimizer": "Adam", "loss": "cross_entropy", "seed": args.seed,
                "device": str(device), "split": "70/15/15 stratified on category",
                "scope": "title + audio (paper 'w/o Frames')",
                "primary_averaging": "macro",
                "checkpoint_selection": "best val macro F1",
            },
            "targets": TARGETS,
            "runs": results,
            "transcript_only_baseline": baseline,
        }, f, indent=2)
    print(f"\nsaved -> {args.out or RESULTS_PATH}")


if __name__ == "__main__":
    main()
