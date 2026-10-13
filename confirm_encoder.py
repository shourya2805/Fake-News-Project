import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from compare_encoders import environment, environment_mismatches
from dataset import FEATURE_META, feature_dir_for, features_exist, file_sha256, load_feature_meta
from multiseed import METRICS, paired_diffs, run_study
from utils import get_device

DEFAULT_PREREG = Path("preregistration/deberta_vs_bert.json")
OUT_DIR = Path("results")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def registration_commit(prereg_path):
    added = git("log", "--diff-filter=A", "--format=%H %cI", "--", str(prereg_path)).splitlines()
    if not added:
        raise SystemExit(f"{prereg_path} is not committed; a pre-registration must be committed before it is run")
    commit, when = added[-1].split()
    return commit, datetime.fromisoformat(when)


def check_registration(prereg, prereg_path, commit):
    problems = []
    registered = [str(prereg_path), str(prereg_path.with_suffix(".md"))]
    for label, paths in (("pre-registration", registered), ("frozen code/data", prereg["frozen_paths"])):
        if subprocess.run(["git", "diff", "--quiet", commit, "--", *paths]).returncode != 0:
            changed = git("diff", "--name-only", commit, "--", *paths).splitlines()
            problems.append(f"{label} changed since registration commit {commit[:7]}: {changed}")
    if file_sha256(prereg["split_file"]) != prereg["split_file_sha256"]:
        problems.append(f"{prereg['split_file']} sha256 does not match the registered value")
    return problems


def ensure_fresh_features(encoder, registered_at):
    feature_dir = feature_dir_for(encoder)
    meta_path = feature_dir / FEATURE_META
    stale = True
    if features_exist(feature_dir) and meta_path.exists():
        extracted = json.loads(meta_path.read_text()).get("extracted")
        stale = extracted is None or datetime.fromisoformat(extracted) < registered_at
    if stale:
        subprocess.run([sys.executable, "extract_features.py", "--encoder", encoder, "--force"], check=True)
    return load_feature_meta(feature_dir)


def decide(result, alpha):
    if result["p_ttest"] < alpha:
        return "confirmed" if result["mean_diff"] > 0 else "contradicted"
    return "not_confirmed"


def describe(summaries, baseline, treatment, names):
    out = {}
    for m in names:
        x = np.array(summaries[baseline][m]["values"])
        y = np.array(summaries[treatment][m]["values"])
        d = y - x
        out[m] = {"baseline_mean": float(x.mean()), "baseline_std": float(x.std(ddof=1)),
                  "treatment_mean": float(y.mean()), "treatment_std": float(y.std(ddof=1)),
                  "diffs": d.tolist(), "mean_diff": float(d.mean()),
                  "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    return out


def markdown(prereg, commit, env, primary, decision, descriptive):
    v = env["versions"]
    labels = {"f1": "Macro F1 (primary)", "acc": "Accuracy", "cd_f1": "CD F1", "ce_f1": "CE F1",
              "sv_f1": "SV F1", "ca_f1": "CA F1"}
    lines = [
        f"# Confirmatory test: {prereg['treatment']} vs {prereg['baseline']}",
        "",
        f"Pre-registered in commit `{commit[:7]}` (`preregistration/{prereg['name']}.md`). "
        f"Seeds {prereg['seeds']}, frozen split `{prereg['split_file']}`, {prereg['configuration']}, "
        f"{prereg['checkpoint']} checkpoint.",
        "",
        f"Environment: {env['device']}{' (' + env['gpu'] + ')' if env['gpu'] else ''}, torch {v['torch']}, "
        f"transformers {v['transformers']}, CUDA {v['cuda']}. Both arms extracted and trained here.",
        "",
        "## Primary endpoint",
        "",
        f"Test macro F1, {prereg['treatment']} − {prereg['baseline']}, paired two-sided t-test, "
        f"alpha {prereg['alpha']}, one test.",
        "",
        "| Mean diff | 95% CI | t | p | W/L | Per-seed diffs |",
        "|---|---|---|---|---|---|",
        f"| {primary['mean_diff']:+.2f} | [{primary['ci95'][0]:+.2f}, {primary['ci95'][1]:+.2f}] | "
        f"{primary['t']:.2f} | {primary['p_ttest']:.4f} | {primary['wins']}/{primary['losses']} | "
        + " ".join(f"{d:+.2f}" for d in primary["diffs"]) + " |",
        "",
        f"**Decision: {decision.replace('_', ' ').upper()}** (pre-registered prediction: "
        f"{prereg['prediction'].replace('_', ' ')}).",
        "",
        "## Descriptive only (no tests)",
        "",
        f"| Metric | {prereg['baseline']} | {prereg['treatment']} | Mean diff | W/L |",
        "|---|---|---|---|---|",
    ]
    for m, r in descriptive.items():
        lines.append(f"| {labels[m]} | {r['baseline_mean']:.2f} ± {r['baseline_std']:.2f} | "
                     f"{r['treatment_mean']:.2f} ± {r['treatment_std']:.2f} | {r['mean_diff']:+.2f} | "
                     f"{r['wins']}/{r['losses']} |")
    lines += ["", "Same 359-sample test set as the exploratory study: this confirms robustness to training "
                  "randomness and a fresh environment, not generalisation to unseen data."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", default=str(DEFAULT_PREREG))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text())
    commit, registered_at = registration_commit(prereg_path)
    problems = check_registration(prereg, prereg_path, commit)
    print(f"registration commit {commit[:7]} at {registered_at.isoformat()} | HEAD {git('rev-parse', '--short', 'HEAD')}")
    if problems:
        raise SystemExit("refusing to run:\n  " + "\n  ".join(problems))
    print("pre-registration and frozen paths unchanged since registration; split sha256 matches")
    if args.check_only:
        return

    device = torch.device(args.device) if args.device else get_device()
    env = environment(device)
    baseline, treatment = prereg["baseline"], prereg["treatment"]
    metas = {e: ensure_fresh_features(e, registered_at) for e in (baseline, treatment)}
    mismatches = environment_mismatches(metas, env)
    if mismatches:
        raise SystemExit("feature caches do not come from this environment:\n  " + "\n  ".join(mismatches))

    summaries, per_seed = {}, {}
    for encoder in (baseline, treatment):
        records, summary, _ = run_study(prereg["subset"], encoder, prereg["seeds"], prereg["epochs"], device)
        summaries[encoder] = summary["without_flag"]["best"]
        per_seed[encoder] = [{"seed": r["seed"], "test": r["runs"]["without_flag"]["test"],
                              "best_epoch": r["runs"]["without_flag"]["best_epoch"]} for r in records]

    key = prereg["primary_metric_key"]
    primary = paired_diffs(summaries[baseline][key]["values"], summaries[treatment][key]["values"])
    decision = decide(primary, prereg["alpha"])
    names = [key] + [m for m in prereg["descriptive_only"] if m in METRICS["full"]]
    descriptive = describe(summaries, baseline, treatment, names)
    md = markdown(prereg, commit, env, primary, decision, descriptive)
    print("\n" + md)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"confirmatory_{prereg['name']}"
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "preregistration": prereg, "registration_commit": commit,
        "registered_at": registered_at.isoformat(),
        "run_finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
        "feature_meta": {e: {k: m.get(k) for k in ("extracted", "device", "gpu", "versions", "files_sha256")}
                         for e, m in metas.items()},
        "primary": primary, "decision": decision, "descriptive": descriptive, "per_seed": per_seed,
    }, indent=2))
    (out_dir / f"{stem}.md").write_text(md)
    print(f"saved -> {out_dir / (stem + '.json')}, {out_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
