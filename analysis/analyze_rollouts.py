#!/usr/bin/env python3
"""
analyze_rollouts.py -- how similar are the 16 rollouts the model writes for ONE prompt?

Question behind it (CrossCUTS / Mixed-CUTS reproduction):
  * Does CUTS sampling really make rollouts more diverse than standard sampling?
  * Is a saturated policy writing near-copies of the same solution (=> zero GRPO advantage)?
  * Do GRPO and Mixed-CUTS differ in diversity, and does it change over training?

Two modes (run `python analyze_rollouts.py -h`):
  dumps     training rollout dumps  runs/<run>/rollout_dumps/<step>.jsonl   (4 prompts x 16 rollouts per step)
  eval      eval samples            eval/<run>/<ckpt>/<benchmark>/samples.jsonl (N problems x 16 samples)
  selftest  builds synthetic data in both formats and runs everything (smoke test, ~1 min)

Metrics, all computed PER PAIR of rollouts of the same prompt, then averaged per group of 16:
  lcp_frac    longest common PREFIX / shorter length (token level). Where do two rollouts first diverge?
  dup         1 if the two (normalised) rollouts are token-for-token identical.
Per subset (std half / CUTS half) of a group:
  unique_frac   fraction of rollouts that are not exact duplicates of another rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# one token = LaTeX command | single CJK character | word | number | any other single non-space char
TOKEN_RE = re.compile(r"\\[A-Za-z]+|[\u4e00-\u9fff]|[A-Za-z]+|\d+(?:\.\d+)?|\S")
PAIR_METRICS = ["lcp_frac", "dup"]


# ----------------------------------------------------------------------------------------------
# STEP 1  loading: turn both file formats into ONE table with the same columns
#   arm   which model/run        step   training step (0 for eval)      group  the prompt
#   idx   rollout number 0..15   kind   "std" or "cuts"                 score  1.0 correct / 0.0 wrong
#   prompt, text                 (eval only) n_tokens, finish_reason
# Why: every later step then works on "a group = all rollouts of one prompt", whatever the source.
# ----------------------------------------------------------------------------------------------
def load_dump_dir(path, arm: str) -> pd.DataFrame:
    rows = []
    for f in Path(path).glob("*.jsonl"):
        if not f.stem.isdigit():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                group, i, _ = r["uid"].rsplit("_", 2)  # uid = "<prompt-uuid>_<rollout idx>_0"
                rows.append(dict(arm=arm, step=int(r["step"]), group=group, idx=int(i),
                                 kind=r.get("rollout_kind", "std"), score=float(r["score"]),
                                 prompt=r["input"], text=r["output"]))
    if not rows:
        sys.exit(f"no rollout dumps found under {path}")
    return pd.DataFrame(rows).sort_values(["step", "group", "idx"]).reset_index(drop=True)


def load_eval(root, benchmark: str, run_filters, max_problems: int | None) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(root).rglob("samples.jsonl")):
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        run, ckpt, bench = parts[-4], parts[-3], parts[-2]  # .../eval/<run>/<ckpt>/<bench>/samples.jsonl
        if benchmark.lower() not in bench.lower():
            continue
        arm = f"{run}/{ckpt}"
        if run_filters and not any(s in arm for s in run_filters):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if max_problems is not None and r["problem"] >= max_problems:
                    continue
                rows.append(dict(arm=arm, step=0, group=str(r["problem"]), idx=int(r["sample"]), kind="std",
                                 score=float(r["correct"]), prompt="", text=r["text"],
                                 n_tokens=r.get("n_tokens"), finish=r.get("finish_reason")))
    if not rows:
        sys.exit(f"no samples.jsonl for benchmark '{benchmark}' under {root}")
    return pd.DataFrame(rows).sort_values(["arm", "group", "idx"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# STEP 2  normalise + tokenise
# Why normalise: the policy writes "\frac" in one rollout and "\dfrac" in the next, "###" headers or
# not, a check-mark emoji... That is formatting jitter, not different reasoning. Normalising makes
# two such rollouts count as identical. Run both modes (--text-mode raw|norm) and compare: the gap
# between them is "how much of the diversity is only formatting".
# ----------------------------------------------------------------------------------------------
def normalize(text: str) -> str:
    t = text
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\dbinom", "\\binom"), ("\\tbinom", "\\binom"),
                 ("\\,", " "), ("\\;", " "), ("\\!", " "), ("\\ ", " ")):
        t = t.replace(a, b)
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])", " ", t)
    t = re.sub(r"[\u2600-\u27BF\U0001F300-\U0001FAFF]", " ", t)   # emoji / check marks
    t = re.sub(r"(?m)^\s*-{3,}\s*$", " ", t)                      # markdown horizontal rules
    t = re.sub(r"[#*_`>$]+", " ", t)                                # markdown + math delimiters
    return t.lower()


def featurize(text: str, norm: bool) -> list[str]:
    """Tokenize one rollout for prefix and duplicate comparisons."""
    return TOKEN_RE.findall(normalize(text) if norm else text)


# ----------------------------------------------------------------------------------------------
# STEP 3  longest common prefix and exact duplicate
# ----------------------------------------------------------------------------------------------
def pair_metrics(fa, fb) -> dict:
    ta, tb = fa, fb
    m, i = min(len(ta), len(tb)), 0
    while i < m and ta[i] == tb[i]:
        i += 1
    return {"lcp_frac": i / m if m else 1.0, "dup": float(ta == tb)}


# ----------------------------------------------------------------------------------------------
# STEP 4  group-level computation
# A group = the (up to) 16 rollouts of one prompt at one step of one arm. For every pair of rollouts
# in the group we compute all metrics, label the pair std-std / cuts-cuts / std-cuts, and average per
# label. In Mixed-CUTS the question is: is cuts-cuts LESS similar than std-std? (more diverse) and is
# std-cuts as similar as std-std? (CUTS rollouts are just standard ones).
# ----------------------------------------------------------------------------------------------
_ORDER = {"std": 0, "cuts": 1}


def compute(df: pd.DataFrame, norm=True, seed=0):
    df = df.reset_index(drop=True)
    print(f"featurising {len(df)} rollouts (text mode: {'normalised' if norm else 'raw'}) ...", flush=True)
    feats = [featurize(t, norm) for t in df["text"]]
    cjk = df["text"].map(lambda s: len(CJK_RE.findall(s)) / max(len(s), 1)).to_numpy()
    pair_rows, subset_rows = [], []
    for (arm, step, group), g in df.groupby(["arm", "step", "group"], sort=False):
        ix = g.index.to_numpy()
        kinds = g["kind"].tolist()
        buckets = defaultdict(list)
        for a, b in combinations(range(len(ix)), 2):
            m = pair_metrics(feats[ix[a]], feats[ix[b]])
            ptype = "-".join(sorted((kinds[a], kinds[b]), key=lambda k: _ORDER.get(k, 9)))
            buckets[ptype].append(m)
        meta = dict(arm=arm, step=step, group=group,
                    prompt_hash=hashlib.md5(str(g["prompt"].iloc[0]).encode()).hexdigest()[:10],
                    score_mean=g["score"].mean(), all_correct=bool((g["score"] == 1).all()),
                    all_wrong=bool((g["score"] == 0).all()), cjk_frac=float(cjk[ix].mean()),
                    len_tokens=float(np.mean([len(feats[i][0]) for i in ix])))
        for ptype, ms in buckets.items():
            row = dict(meta, pair_type=ptype, n_pairs=len(ms))
            for k in PAIR_METRICS:
                vals = [m[k] for m in ms if k in m]
                row[k] = float(np.mean(vals)) if vals else np.nan
            pair_rows.append(row)
        for kind in sorted(set(kinds), key=lambda k: _ORDER.get(k, 9)):
            members = [ix[j] for j, k in enumerate(kinds) if k == kind]
            row = dict(meta, kind=kind, size=len(members), score_kind=float(df.loc[members, "score"].mean()),
                       unique_frac=len({hash(tuple(feats[i])) for i in members}) / len(members))
            subset_rows.append(row)
    return pd.DataFrame(pair_rows), pd.DataFrame(subset_rows)


# ----------------------------------------------------------------------------------------------
# STEP 5  statistics
# Only 4 prompts per training step are dumped, so a single step is very noisy. We therefore (1) pool
# steps into bins and (2) put a bootstrap 95% CI (resampling GROUPS, not pairs: pairs inside a group
# are not independent) on every mean, (3) use paired tests across groups instead of eyeballing.
# ----------------------------------------------------------------------------------------------
def boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray([v for v in x if not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(axis=1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def fmt(x):
    m, lo, hi = boot_ci(x)
    return "   n/a" if np.isnan(m) else f"{m:.3f} [{lo:.3f},{hi:.3f}]"


def wilcoxon(diff):
    from scipy.stats import wilcoxon as w
    d = np.asarray(diff, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    return float(w(d).pvalue)


def metrics_present(pairs):
    return [m for m in ["lcp_frac", "dup"] if pairs[m].notna().any()]


def summarize_dumps(pairs, subsets, max_step, out):
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    healthy = pairs[pairs.step <= max_step]
    ms = metrics_present(pairs)
    P(f"=== A. mean pairwise similarity, steps <= {max_step} (before the collapse), mean [95% CI over groups]")
    P("    higher = more similar = LESS diverse")
    for (arm, pt), g in healthy.groupby(["arm", "pair_type"]):
        P(f"  {arm:8s} {pt:9s} groups={len(g):4d}  " + "  ".join(f"{m}={fmt(g[m])}" for m in ms))
    P("\n=== B. Mixed-CUTS: is the CUTS half more diverse than the standard half? (paired by group)")
    for arm, g in healthy.groupby("arm"):
        w = g.pivot_table(index=["step", "group"], columns="pair_type", values=ms)
        if ("lcp_frac", "cuts-cuts") not in w.columns:
            continue
        for m in ms:
            d = w[(m, "cuts-cuts")] - w[(m, "std-std")]
            P(f"  {arm:8s} {m:10s} median(cuts-cuts - std-std) = {np.nanmedian(d):+.4f}   Wilcoxon p = {wilcoxon(d):.3g}   (n={d.notna().sum()})"
              "   negative = CUTS rollouts are more diverse")
    P("\n=== C. same prompt, same step: GRPO std-std vs Mixed std-std (both use standard sampling)")
    std = healthy[healthy.pair_type == "std-std"]
    arms = sorted(std.arm.unique())
    if len(arms) >= 2:
        a0, a1 = arms[0], arms[1]
        j = std[std.arm == a0].merge(std[std.arm == a1], on=["step", "prompt_hash"], suffixes=("_0", "_1"))
        P(f"  matched (step, prompt) groups between {a0} and {a1}: {len(j)}")
        for m in ms:
            d = j[f"{m}_1"] - j[f"{m}_0"]
            P(f"  {m:10s} median({a1} - {a0}) = {np.nanmedian(d):+.4f}   Wilcoxon p = {wilcoxon(d):.3g}")
    P("\n=== D. all-correct groups vs mixed-outcome groups (saturation): std-std similarity")
    for (arm, ac), g in std.groupby(["arm", "all_correct"]):
        P(f"  {arm:8s} all_correct={str(ac):5s} groups={len(g):4d}  lcp_frac={fmt(g.lcp_frac)}  dup={fmt(g.dup)}")
    P("\n=== E. exact-duplicate rate per subset, steps <= %d" % max_step)
    sub = subsets[subsets.step <= max_step]
    for (arm, kind), g in sub.groupby(["arm", "kind"]):
        P(f"  {arm:8s} {kind:5s} unique_frac={fmt(g.unique_frac)}")
    Path(out, "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def summarize_eval(pairs, subsets, out, ref=None):
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    ms = metrics_present(pairs)
    pairs = pairs.assign(bucket=np.where(pairs.all_correct, "all-correct", np.where(pairs.all_wrong, "all-wrong", "mixed")))
    P("=== A. within-problem similarity of the 16 samples, mean [95% CI over problems]")
    for arm, g in pairs.groupby("arm"):
        P(f"  {arm:34s} problems={len(g):4d}  " + "  ".join(f"{m}={fmt(g[m])}" for m in ms))
    P("\n=== B. same, split by how the problem was solved (all 16 correct / mixed / all wrong)")
    for (arm, b), g in pairs.groupby(["arm", "bucket"]):
        P(f"  {arm:34s} {b:11s} n={len(g):4d}  lcp_frac={fmt(g.lcp_frac)}  dup={fmt(g.dup)}")
    arms = sorted(pairs.arm.unique())
    ref = next((a for a in arms if ref and ref in a), arms[0])
    P(f"\n=== C. paired by problem against reference '{ref}' (positive = MORE similar than the reference)")
    r = pairs[pairs.arm == ref].set_index("group")
    for arm in arms:
        if arm == ref:
            continue
        g = pairs[pairs.arm == arm].set_index("group")
        j = r.join(g, lsuffix="_ref", rsuffix="_x", how="inner")
        P("  " + f"{arm:34s} " + "  ".join(f"{m}: {np.nanmedian(j[m + '_x'] - j[m + '_ref']):+.3f} (p={wilcoxon(j[m + '_x'] - j[m + '_ref']):.2g})" for m in ["lcp_frac", "dup"]))
    P("\n=== D. exact-duplicate rate per problem")
    for arm, g in subsets.groupby("arm"):
        P(f"  {arm:34s} unique_frac={fmt(g.unique_frac)}")
    Path(out, "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# STEP 6  plots
# ----------------------------------------------------------------------------------------------
def plot_dumps(pairs, out, bin_w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [m for m in ["lcp_frac", "dup"] if pairs[m].notna().any()]
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 4), squeeze=False)
    pairs = pairs.assign(bin=(pairs.step - 1) // bin_w * bin_w + bin_w / 2)
    for ax, m in zip(axes[0], ms):
        for (arm, pt), g in pairs.groupby(["arm", "pair_type"]):
            s = g.groupby("bin")[m].mean()
            ax.plot(s.index, s.values, marker="o", ms=3, label=f"{arm} {pt}")
        for arm, g in pairs.groupby("arm"):
            drift = g.groupby("step").cjk_frac.mean()
            hit = drift[drift > 0.05]
            if len(hit):
                ax.axvline(hit.index[0], ls=":", c="gray")
                ax.text(hit.index[0], ax.get_ylim()[0], f" {arm}: language drift", rotation=90, va="bottom", fontsize=7)
        ax.set_title(f"{m} (higher = more similar)")
        ax.set_xlabel(f"training step (bins of {bin_w})")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out, "similarity_vs_step.png"), dpi=140)


def plot_eval(pairs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [m for m in ["lcp_frac", "dup"] if pairs[m].notna().any()]
    arms = sorted(pairs.arm.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 4), squeeze=False)
    for ax, m in zip(axes[0], ms):
        st = [boot_ci(pairs[pairs.arm == a][m]) for a in arms]
        ax.bar(range(len(arms)), [s[0] for s in st], yerr=[[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]], capsize=3)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a.replace("/", "\n") for a in arms], fontsize=6)
        ax.set_title(f"{m} (higher = more similar)")
    fig.tight_layout()
    fig.savefig(Path(out, "eval_metrics.png"), dpi=140)


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def run_dumps(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_dump_dir(a.grpo, "grpo"), load_dump_dir(a.mixed, "mixed")], ignore_index=True)
    pairs, subsets = compute(df, a.text_mode == "norm")
    pairs.to_csv(out / "group_pair_metrics.csv", index=False)
    subsets.to_csv(out / "group_subset_metrics.csv", index=False)
    summarize_dumps(pairs, subsets, a.max_step, out)
    plot_dumps(pairs, out, a.bin)
    print(f"\nwrote CSVs, summary.txt and PNGs to {out}")


def run_eval(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_eval(a.root, a.benchmark, a.runs, a.max_problems)
    if a.drop_truncated:
        bad = df[df.finish == "length"].groupby(["arm", "group"]).size().index
        df = df[~df.set_index(["arm", "group"]).index.isin(bad)].reset_index(drop=True)
    pairs, subsets = compute(df, a.text_mode == "norm")
    pairs.to_csv(out / "problem_pair_metrics.csv", index=False)
    subsets.to_csv(out / "problem_subset_metrics.csv", index=False)
    summarize_eval(pairs, subsets, out, a.ref)
    plot_eval(pairs, out)
    print(f"\nwrote CSVs, summary.txt and PNGs to {out}")


def make_synthetic(root: Path, seed=0):
    """Tiny fake data in the REAL formats. Mixed-CUTS' cuts half gets more word-level variety, and from
    step 53 the mixed arm drifts into Chinese, so every code path (including the drift marker) runs."""
    rng = np.random.default_rng(seed)
    opens = ["We are given", "We are asked to evaluate", "Let us compute", "Consider"]
    steps_ = ["Step 1: compute the numerator", "First, multiply", "Calculate the product", "Begin by simplifying"]
    ends = ["### Final Answer:", "Final Answer:", "Therefore the answer is", "Thus"]

    def solution(variety):
        k = lambda xs: xs[int(rng.integers(0, max(1, variety)))] if variety > 1 else xs[0]
        return f"{k(opens)} the expression \\dfrac{{3 \\times 4}}{{6}}.\n\n{k(steps_)}:\n$$3 \\times 4 = 12$$\n\n{k(ends)}\n$$\\boxed{{2}}$$"

    for arm in ("grpo", "mixed"):
        d = root / arm / "rollout_dumps"
        d.mkdir(parents=True)
        for step in range(1, 61):
            with open(d / f"{step}.jsonl", "w", encoding="utf-8") as f:
                for p in range(4):
                    gid = f"{step:04d}0000-aaaa-bbbb-cccc-{p:012d}"
                    for i in range(16):
                        kind = "cuts" if (arm == "mixed" and i >= 8) else "std"
                        txt = solution(4 if kind == "cuts" else 2)
                        if arm == "mixed" and step >= 53:
                            txt = "终于，我们来解决这个题目：" + "我们现在要找的是" * 5
                        f.write(json.dumps(dict(input=f"system\nS\nuser\nprompt {step}-{p}\nassistant\n<think>\n\n</think>\n\n",
                                                output=txt, gts="2", score=float(rng.random() < 0.8), step=step,
                                                uid=f"{gid}_{i}_0", rollout_kind=kind), ensure_ascii=False) + "\n")
    for run, ckpt, variety in (("qwen3-base", "base", 4), ("grpo", "global_step_40", 2), ("mixed", "global_step_40", 3)):
        d = root / "eval" / run / ckpt / "math500"
        d.mkdir(parents=True)
        with open(d / "samples.jsonl", "w", encoding="utf-8") as f:
            for pr in range(20):
                for s in range(16):
                    f.write(json.dumps(dict(problem=pr, index=str(pr), sample=s, correct=float(rng.random() < 0.7),
                                            pred="2", has_boxed=True, valid=True, ground_truth="2", n_tokens=300,
                                            finish_reason="stop", text=solution(variety))) + "\n")


def selftest():
    fa = featurize("a b c d", True)
    assert pair_metrics(fa, fa)["lcp_frac"] == 1.0 and pair_metrics(fa, fa)["dup"] == 1.0
    m = pair_metrics(fa, featurize("w x y z", True))
    assert m["lcp_frac"] == 0.0 and m["dup"] == 0.0
    assert normalize("\\dfrac{1}{2}") == normalize("\\frac{1}{2}")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_synthetic(root)
        ns = argparse.Namespace
        run_dumps(ns(grpo=root / "grpo/rollout_dumps", mixed=root / "mixed/rollout_dumps", out=root / "o1", text_mode="norm",
                     max_step=48, bin=5))
        run_eval(ns(root=root / "eval", benchmark="math", runs=[], max_problems=None, out=root / "o2", text_mode="norm",
                    ref="base", drop_truncated=False))
        df = load_dump_dir(root / "mixed/rollout_dumps", "mixed").query("step < 4")
        compute(df, True)
        for f in ("o1/summary.txt", "o1/similarity_vs_step.png", "o2/summary.txt", "o2/eval_metrics.png"):
            assert (root / f).exists(), f
    print("\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="out")
    common.add_argument("--text-mode", choices=["norm", "raw"], default="norm", help="normalise LaTeX/markdown jitter or not")
    d = sp.add_parser("dumps", parents=[common])
    d.add_argument("--grpo", required=True, help="runs/math_grpo-h200-s1/rollout_dumps")
    d.add_argument("--mixed", required=True, help="runs/math_mixed_cuts-h200-s1/rollout_dumps")
    d.add_argument("--max-step", type=int, default=48, help="'healthy regime' cut-off for the summary tables")
    d.add_argument("--bin", type=int, default=5)
    e = sp.add_parser("eval", parents=[common])
    e.add_argument("--root", required=True, help="the eval/ folder of the HF repo (or its parent)")
    e.add_argument("--benchmark", default="math")
    e.add_argument("--runs", nargs="*", default=[], help="keep only arms whose 'run/ckpt' contains one of these strings")
    e.add_argument("--max-problems", type=int, default=200)
    e.add_argument("--ref", default="base", help="reference arm for the paired comparison")
    e.add_argument("--drop-truncated", action="store_true", help="drop problems with any sample cut at the token limit")
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"dumps": run_dumps, "eval": run_eval, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
