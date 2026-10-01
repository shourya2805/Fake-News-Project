import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import chi2_contingency

DATA_JSON = Path("data/data.json")
VIDEO_DIR = Path("videos")
FAIL_LOG = VIDEO_DIR / "failed.csv"
REPORT_TABLE = Path("download_report.txt")
CATEGORY = {"": "REAL", "ft": "CD", "fc": "CA"}
CATS = ["REAL", "CD", "CA"]
HOSTS = ["YouTube", "X"]
TWITTER_EPOCH_MS = 1288834974657


def host(vid):
    return "YouTube" if len(vid) == 11 else "X"


def post_year(vid):
    if host(vid) == "YouTube":
        return "YouTube"
    ms = (int(vid) >> 22) + TWITTER_EPOCH_MS
    return str(datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year)


def reason_key(reason):
    return re.sub(r"^ERROR:\s*(\[[^\]]+\]\s*)?[\w-]+:\s*", "", reason).strip()


def load():
    with FAIL_LOG.open(newline="") as f:
        failed = {row["video_id"]: row["reason"] for row in csv.DictReader(f)}
    rows = []
    for r in json.loads(DATA_JSON.read_text()):
        if r["class"] not in CATEGORY:
            continue
        vid = r["video_id"]
        if (VIDEO_DIR / f"{vid}.mp4").exists():
            status = "have"
        elif vid in failed:
            status = "failed"
        else:
            status = "missing"
        rows.append({"vid": vid, "cat": CATEGORY[r["class"]], "host": host(vid), "year": post_year(vid),
                     "status": status, "reason": reason_key(failed.get(vid, ""))})
    return rows


def tally(rows):
    c = Counter(r["status"] for r in rows)
    attempted = c["have"] + c["failed"]
    return {"n": len(rows), "have": c["have"], "failed": c["failed"], "missing": c["missing"],
            "cov": 100 * c["have"] / len(rows) if rows else 0.0,
            "fail": 100 * c["failed"] / attempted if attempted else 0.0}


def table(title, groups):
    lines = [title, f"{'group':<16}{'records':>9}{'have':>7}{'failed':>8}{'missing':>9}{'coverage':>10}{'fail_rate':>11}"]
    for name, rows in groups:
        t = tally(rows)
        lines.append(f"{name:<16}{t['n']:>9}{t['have']:>7}{t['failed']:>8}{t['missing']:>9}"
                     f"{t['cov']:>9.1f}%{t['fail']:>10.1f}%")
    return lines + [""]


def year_table(rows):
    years = sorted({r["year"] for r in rows}, key=lambda y: (y == "YouTube", y))
    lines = ["FAILURE RATE BY POST YEAR (X snowflake timestamp; YouTube ids carry no date)",
             f"{'year':<10}{'records':>9}{'have':>7}{'failed':>8}{'fail_rate':>11}" + "".join(f"{c + '_fail':>11}" for c in CATS)]
    for y in years:
        sel = [r for r in rows if r["year"] == y]
        t = tally(sel)
        per = []
        for c in CATS:
            sc = [r for r in sel if r["cat"] == c]
            per.append(f"{tally(sc)['fail']:>10.1f}%" if sc else f"{'-':>11}")
        lines.append(f"{y:<10}{t['n']:>9}{t['have']:>7}{t['failed']:>8}{t['fail']:>10.1f}%" + "".join(per))
    return lines + [""]


def category_test(rows):
    lines = ["DOES FAILURE RATE DIFFER BY CATEGORY? (chi-square on have vs failed)"]
    for label, sel in [("all hosts", rows), ("X only", [r for r in rows if r["host"] == "X"]),
                       ("YouTube only", [r for r in rows if r["host"] == "YouTube"])]:
        table_ = [[tally([r for r in sel if r["cat"] == c])[k] for k in ("have", "failed")] for c in CATS]
        table_ = [t for t in table_ if sum(t)]
        if len(table_) < 2 or not all(sum(col) for col in zip(*table_)):
            lines.append(f"  {label:<14} not testable")
            continue
        chi2, p, dof, _ = chi2_contingency(table_)
        rates = "  ".join(f"{c} {tally([r for r in sel if r['cat'] == c])['fail']:.1f}%" for c in CATS)
        lines.append(f"  {label:<14} {rates}   chi2 {chi2:.2f}  dof {dof}  p {p:.3g}")
    real = [r for r in rows if r["cat"] == "REAL"]
    fake = [r for r in rows if r["cat"] != "REAL"]
    chi2, p, _, _ = chi2_contingency([[tally(real)["have"], tally(real)["failed"]],
                                      [tally(fake)["have"], tally(fake)["failed"]]])
    lines.append(f"  REAL vs fake   REAL {tally(real)['fail']:.1f}%  fake(CD+CA) {tally(fake)['fail']:.1f}%   "
                 f"chi2 {chi2:.2f}  p {p:.3g}")
    return lines + [""]


def reasons(rows):
    c = Counter((r["host"], r["reason"]) for r in rows if r["status"] == "failed")
    lines = ["FAILURE REASONS", f"{'host':<9}{'count':>6}  reason"]
    for (h, reason), n in c.most_common():
        lines.append(f"{h:<9}{n:>6}  {reason[:110]}")
    return lines + [""]


def balance(rows):
    have = [r for r in rows if r["status"] == "have"]
    n_real = sum(r["cat"] == "REAL" for r in have)
    n_fake = len(have) - n_real
    all_real = sum(r["cat"] == "REAL" for r in rows)
    by_cat = Counter(r["cat"] for r in have)
    return ["USABLE FOR THE VISUAL EXPERIMENT",
            f"  usable videos     {len(have)} of {len(rows)} REAL+CD+CA records ({100 * len(have) / len(rows):.1f}%)",
            f"  per category      " + "  ".join(f"{c} {by_cat[c]}" for c in CATS),
            f"  real / fake       {n_real} / {n_fake}  ({100 * n_real / len(have):.1f}% real)",
            f"  before attrition  {all_real} / {len(rows) - all_real}  ({100 * all_real / len(rows):.1f}% real)",
            "  text-only dataset 893 / 1500  (37.3% real)", ""]


def main():
    rows = load()
    lines = []
    lines += table("COVERAGE BY CATEGORY x HOST",
                   [(f"{c} {h}", [r for r in rows if r["cat"] == c and r["host"] == h]) for c in CATS for h in HOSTS])
    lines += table("BY CATEGORY", [(c, [r for r in rows if r["cat"] == c]) for c in CATS])
    lines += table("BY HOST", [(h, [r for r in rows if r["host"] == h]) for h in HOSTS] + [("ALL", rows)])
    lines += year_table(rows)
    lines += category_test(rows)
    lines += reasons(rows)
    lines += balance(rows)
    lines.append("fail_rate = failed / (have + failed); missing = neither downloaded nor logged as failed")
    text = "\n".join(lines)
    print(text)
    REPORT_TABLE.write_text(text + "\n")


if __name__ == "__main__":
    main()
