"""Run one or more benchmarks end to end: load parquet -> sample -> score -> aggregate -> write.

Outputs per benchmark under ``<out_dir>/<benchmark>/``:

    samples.jsonl   one line per (problem, sample): text, extracted answer, correctness
    results.json    aggregate metrics with bootstrap CIs, plus the config used

and ``<out_dir>/summary.json`` + a printed markdown table across benchmarks. Correctness uses
:func:`mixed_cuts.reward.compute_score`, i.e. exactly the training reward.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from mc_eval.generate import GenerateFn
from mc_eval.scoring import aggregate, summarize_problem
from mixed_cuts import reward

logger = logging.getLogger(__name__)


def _equal_fn(data_source: str, reward_cfg: dict[str, Any]):
    """Expensive answer equivalence for majority voting (math-verify via the reward function)."""

    def equal(a: str, b: str) -> bool:
        if data_source == "gpqa_diamond":
            return reward.score_multiple_choice(f"\\boxed{{{a}}}", b)["score"] == 1.0
        return reward.score_math(f"\\boxed{{{a}}}", b, timeout=reward_cfg.get("timeout", 30))["score"] == 1.0

    return equal


def run_benchmark(
    name: str,
    parquet_path: str | Path,
    generate: GenerateFn,
    cfg: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    if limit:
        df = df.head(limit)
    n = int(cfg["n_samples"])
    k_values = [int(k) for k in cfg.get("k_values", [1, n]) if int(k) <= n]
    reward_cfg = dict(cfg.get("reward", {}))
    sampling_cfg = dict(cfg["sampling"])
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    conversations = [list(p) for p in df["prompt"]]
    t0 = time.time()
    texts = generate(conversations, n, sampling_cfg)
    gen_secs = time.time() - t0
    assert len(texts) == len(df), f"generator returned {len(texts)} results for {len(df)} prompts"

    per_problem = []
    n_tokens_proxy = 0
    n_valid = 0
    gts = [str(rm["ground_truth"]) for rm in df["reward_model"]]
    ceiling = (
        1.0 if name == "gpqa_diamond" else (sum(reward.has_digit(g) for g in gts) / len(gts) if gts else 0.0)
    )
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for i, (row, samples) in enumerate(zip(df.itertuples(index=False), texts, strict=True)):
            ds = row.data_source
            gt = str(row.reward_model["ground_truth"])
            scores, preds = [], []
            for j, text in enumerate(samples):
                r = reward.compute_score(ds, text, gt, timeout=reward_cfg.get("timeout", 30))
                scores.append(float(r["score"]))
                preds.append(r["pred"] or None)
                n_valid += int(r["valid"])
                n_tokens_proxy += len(text)
                f.write(
                    json.dumps(
                        {
                            "problem": i,
                            "index": str(row.extra_info.get("index")),
                            "sample": j,
                            "correct": r["score"],
                            "pred": r["pred"],
                            "has_boxed": r["has_boxed"],
                            "valid": r["valid"],
                            "ground_truth": gt,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            equal = _equal_fn(ds, reward_cfg)

            def is_correct(a: str, _ds: str = ds, _gt: str = gt) -> bool:
                return reward.compute_score(_ds, f"\\boxed{{{a}}}", _gt)["score"] == 1.0

            per_problem.append(
                summarize_problem(
                    scores, preds, gt, k_values=k_values, is_correct_answer=is_correct, equal=equal
                )
            )

    n_problems = len(df)
    result = {
        "benchmark": name,
        "n_problems": n_problems,
        "n_samples": n,
        "metrics": aggregate(per_problem, seed=int(cfg.get("bootstrap_seed", 0))),
        "frac_valid_responses": n_valid / max(1, n_problems * n),
        "digit_rule_ceiling": ceiling,
        "pp_per_problem": 100.0 / n_problems if n_problems else None,
        "notes": ci_note(name, n_problems),
        "generation_seconds": gen_secs,
        "mean_response_chars": n_tokens_proxy / max(1, len(df) * n),
        "config": cfg,
        "parquet": str(parquet_path),
        "timestamp": time.time(),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    logger.info(
        "%s: %s", name, {k: round(v, 4) for k, v in result["metrics"].items() if isinstance(v, float)}
    )
    return result


def run_all(
    benchmarks: list[str],
    data_dir: str | Path,
    generate: GenerateFn,
    cfg: dict[str, Any],
    out_dir: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    results = {}
    for name in benchmarks:
        parquet = Path(data_dir) / f"{name}.parquet"
        if not parquet.exists():
            logger.warning("skipping %s: %s missing (run `make data`; GPQA needs HF_TOKEN)", name, parquet)
            continue
        results[name] = run_benchmark(name, parquet, generate, cfg, out_dir, limit=limit)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    print(markdown_table(results, cfg.get("k_values", [])))
    return results


def ci_note(name: str, n_problems: int) -> str:
    """Small benchmarks are high-variance: one problem is 100/n percentage points."""
    pp = 100.0 / n_problems if n_problems else float("nan")
    note = f"{name}: {n_problems} problems; one problem = {pp:.1f} pp of pass@1. Read the bootstrap 95% CIs."
    if n_problems <= 50:
        note += " Small benchmark: compare arms only across multiple seeds."
    return note


def markdown_table(results: dict[str, Any], k_values: list[int]) -> str:
    cols = ["pass@1"] + [f"pass@{k}" for k in k_values if k != 1] + [f"maj@{k}" for k in k_values if k != 1]
    lines = ["| benchmark | n_problems | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, r in results.items():
        m = r["metrics"]
        cells = []
        for c in cols:
            if c in m:
                ci = m.get(f"{c}_ci95")
                cells.append(f"{100 * m[c]:.1f}" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""))
            else:
                cells.append("-")
        lines.append(f"| {name} | {r['n_problems']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Numbers are pass@1 / pass@k / maj@k in %, with bootstrap 95% CIs over problems.")
    for name, r in results.items():
        if r.get("pp_per_problem"):
            lines.append(
                f"- {name}: one problem = {r['pp_per_problem']:.1f} pp; digit-rule ceiling "
                f"{100 * r.get('digit_rule_ceiling', 1.0):.1f}%; valid responses {100 * r.get('frac_valid_responses', 0):.1f}%"
            )
    return "\n".join(lines)
