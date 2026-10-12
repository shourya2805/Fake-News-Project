import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import (DEFAULT_ENCODER, ENCODERS, FEATURE_META, feature_dir_for, features_exist,
                     load_feature_meta)
from extract_features import package_versions
from multiseed import METRICS, SEEDS, paired_diffs, run_study
from train import EPOCHS
from utils import get_device

BASELINE = DEFAULT_ENCODER
OUT_DIR = Path("results")
METRIC_NAMES = list(METRICS["full"])
METRIC_LABELS = {"acc": "Accuracy", "f1": "Macro F1", "cd_f1": "CD F1", "ce_f1": "CE F1",
                 "sv_f1": "SV F1", "ca_f1": "CA F1"}
ENV_KEYS = ["torch", "transformers", "tokenizers"]


def environment(device):
    versions = package_versions()
    for name in ("scikit-learn", "scipy"):
        versions[name] = importlib.metadata.version(name)
    versions["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "versions": versions, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(), "git_commit": commit, "git_dirty": dirty,
    }


def ensure_features(encoder):
    feature_dir = feature_dir_for(encoder)
    if not (features_exist(feature_dir) and (feature_dir / FEATURE_META).exists()):
        subprocess.run([sys.executable, "extract_features.py", "--encoder", encoder], check=True)
    return load_feature_meta(feature_dir)


def environment_mismatches(metas, env):
    problems = []
    for encoder, meta in metas.items():
        versions = meta.get("versions") or {}
        for key in ENV_KEYS:
            if versions.get(key) != env["versions"].get(key):
                problems.append(f"{encoder}: features extracted with {key} {versions.get(key)}, "
                                f"this run has {env['versions'].get(key)}")
        if (meta.get("device") or "").split(":")[0] != env["device"].split(":")[0]:
            problems.append(f"{encoder}: features extracted on {meta.get('device')}, this run is on {env['device']}")
    return problems


def holm(pvalues):
    keys = [k for k, p in pvalues.items() if p is not None and np.isfinite(p)]
    order = sorted(keys, key=lambda k: pvalues[k])
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def compare(summaries, seeds):
    tests = {}
    if len(seeds) < 2:
        return tests
    base = summaries[BASELINE]
    for encoder, s in summaries.items():
        if encoder == BASELINE:
            continue
        tests[encoder] = {m: paired_diffs(base[m]["values"], s[m]["values"]) for m in METRIC_NAMES}
    adjusted = holm({(e, m): r["p_ttest"] for e, per in tests.items() for m, r in per.items()})
    for (e, m), p in adjusted.items():
        tests[e][m]["p_ttest_holm"] = p
    return tests


def cell(stat):
    return f"{stat['mean']:.2f} ± {stat['std']:.2f}" if np.isfinite(stat["std"]) else f"{stat['mean']:.2f}"


def markdown(summaries, tests, config, env):
    v = env["versions"]
    lines = [
        "# Frozen text-encoder comparison",
        "",
        f"Full dataset (2,393 records), frozen split `{config['split_file']}` "
        f"(train {config['split_sizes']['train']} / val {config['split_sizes']['val']} / "
        f"test {config['split_sizes']['test']}). Phase 1 configuration, `without_flag`, d_k 32. "
        f"Test set, best-validation-macro-F1 checkpoint. Seeds {config['seeds']}, "
        f"mean ± sample SD over seeds.",
        "",
        f"Environment: {env['device']}{' (' + env['gpu'] + ')' if env['gpu'] else ''}, "
        f"torch {v['torch']}, transformers {v['transformers']}, CUDA {v['cuda']}, python {v['python']}. "
        + ("Feature caches come from more than one environment." if env.get("mixed_environment")
           else "Every row was extracted and trained in this one environment."),
        "",
        "| Encoder | " + " | ".join(METRIC_LABELS[m] for m in METRIC_NAMES) + " |",
        "|---|" + "---|" * len(METRIC_NAMES),
    ]
    for encoder, s in summaries.items():
        name = f"{encoder} (baseline)" if encoder == BASELINE else encoder
        lines.append(f"| {name} | " + " | ".join(cell(s[m]) for m in METRIC_NAMES) + " |")
    if tests:
        lines += [
            "",
            f"Paired t-test across seeds against `{BASELINE}` (difference = encoder − baseline). "
            f"p (Holm) adjusts over all {sum(len(t) for t in tests.values())} tests in this table.",
            "",
            "| Encoder | Metric | Mean diff | 95% CI | t | p | p (Holm) | W/L |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for encoder, per in tests.items():
            for m, r in per.items():
                lines.append(
                    f"| {encoder} | {METRIC_LABELS[m]} | {r['mean_diff']:+.2f} | "
                    f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['t']:.2f} | {r['p_ttest']:.3f} | "
                    f"{r.get('p_ttest_holm', float('nan')):.3f} | {r['wins']}/{r['losses']} |")
    lines += [
        "",
        "Per-category F1 = macro F1 on all real test samples plus only that category's fakes, "
        "scored as its own binary problem (the convention of the Phase 1 per-category table).",
        "",
        "Token budgets (32 title / 64 transcript tokens) cover slightly different amounts of text "
        "under each tokenizer; encoder differences are confounded with that (see DEVIATIONS.md).",
        "",
        "Not comparable to the cited Phase 1 figure (77.16 / 74.34), which was produced on a Mac "
        "(mps) under an unrecorded torch version.",
    ]
    if env.get("mixed_environment"):
        lines += ["", "**WARNING: run with --allow-mixed-env; feature caches do not all come from this "
                      "environment. Not for publication.**"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", choices=ENCODERS, default=ENCODERS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--allow-mixed-env", action="store_true")
    args = parser.parse_args()
    if BASELINE not in args.encoders:
        parser.error(f"the baseline {BASELINE} must be included")
    encoders = [BASELINE] + [e for e in args.encoders if e != BASELINE]

    device = torch.device(args.device) if args.device else get_device()
    env = environment(device)
    metas = {e: ensure_features(e) for e in encoders}
    problems = environment_mismatches(metas, env)
    if problems and not args.allow_mixed_env:
        sys.exit("feature caches do not come from this environment:\n  " + "\n  ".join(problems)
                 + "\nre-extract them here (extract_features.py --force) or pass --allow-mixed-env")
    env["mixed_environment"] = bool(problems)
    env["environment_mismatches"] = problems

    summaries, per_seed, config = {}, {}, None
    for encoder in encoders:
        records, summary, config = run_study("full", encoder, args.seeds, args.epochs, device)
        summaries[encoder] = {m: summary["without_flag"]["best"][m] for m in METRIC_NAMES}
        summaries[encoder]["best_epochs"] = summary["without_flag"]["best"]["best_epochs"]
        per_seed[encoder] = [{"seed": r["seed"], "test": r["runs"]["without_flag"]["test"],
                              "best_epoch": r["runs"]["without_flag"]["best_epoch"],
                              "history": r["runs"]["without_flag"]["history"]} for r in records]
    tests = compare(summaries, args.seeds)

    study = {k: config[k] for k in ("seeds", "epochs", "lr", "batch_size", "split_file", "split_file_sha256",
                                    "split_sizes", "std", "varies_with_seed")}
    study.update(subset="full", configuration="without_flag (Phase 1 headline), d_k 32",
                 checkpoint="best validation macro F1", baseline=BASELINE,
                 per_category="macro F1 on REAL + that category's fakes, as its own binary problem",
                 paired_test="paired t-test (and Wilcoxon) across seeds, encoder - baseline; "
                             "Holm adjustment over every encoder x metric test")
    md = markdown(summaries, tests, study, env)
    print("\n" + md)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "encoder_comparison.json").write_text(json.dumps({
        "config": study, "environment": env,
        "feature_meta": {e: {k: m.get(k) for k in ("encoder", "tokenizer_class", "hidden_size", "title_length",
                                                    "speech_text_length", "title_truncated", "transcript_truncated",
                                                    "extracted", "device", "gpu", "versions", "files_sha256")}
                         for e, m in metas.items()},
        "summary": summaries, "paired_vs_baseline": tests, "per_seed": per_seed,
    }, indent=2))
    (out_dir / "encoder_comparison.md").write_text(md)
    print(f"saved -> {out_dir / 'encoder_comparison.json'}, {out_dir / 'encoder_comparison.md'}")


if __name__ == "__main__":
    main()
