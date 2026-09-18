#!/usr/bin/env python
"""Build the parquet files verl and the eval harness read, from the prefetched Hub snapshots.

Runs on a compute node (needs pyarrow/pandas/datasets from the full venv). Reads
``$MC_STAGE_ROOT/raw/<repo>`` written by ``scripts/prefetch.py`` and writes to ``$MC_DATA_DIR``::

    math_train.parquet     7.5k MATH problems (DigitalLearningGmbH/MATH-lighteval, train)
    dapo_train.parquet     DAPO-Math-17k DEDUPLICATED (~17.9k of 1,791,700 rows)
    math500.parquet aime24.parquet aime25.parquet amc23.parquet gpqa_diamond.parquet
    smoke_train.parquet    first 20 MATH problems       (smoke test)
    smoke_val.parquet      first 8 MATH-500 problems    (smoke test)
    MANIFEST.json          row counts, dedupe ratio, sha256 of every parquet

Every prompt gets the same instruction (mc_data.schema.BOXED_INSTRUCTION); the ground truth
is the last \\boxed{} of the reference solution or the benchmark's answer column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from mc_data import dapo, eval_sets, math
from mc_data.schema import DS_GPQA, EVAL_SOURCES

MATH_REPO = "DigitalLearningGmbH/MATH-lighteval"
DAPO_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"


def _raw_dir(stage_root: Path, repo: str) -> Path:
    return stage_root / "raw" / repo.replace("/", "__")


def _load_local(repo_dir: Path, config: str | None, split: str):
    """Load a Hub snapshot directory with ``datasets`` (parquet / json / csv auto-detected)."""
    from datasets import load_dataset

    if not repo_dir.is_dir():
        raise FileNotFoundError(f"{repo_dir} missing: run `make prefetch` on the login node first")
    if config == "gpqa_diamond":
        csv = repo_dir / "gpqa_diamond.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing (gated dataset; was HF_TOKEN set during prefetch?)")
        return load_dataset("csv", data_files=str(csv), split="train")
    return load_dataset(str(repo_dir), split=split)


def _write(df: pd.DataFrame, path: Path, manifest: dict, **info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    manifest[path.name] = {
        "rows": int(len(df)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **info,
    }
    print(f"   wrote {path} ({len(df)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-root", default=os.environ.get("MC_STAGE_ROOT"))
    ap.add_argument("--out", default=os.environ.get("MC_DATA_DIR"))
    ap.add_argument("--smoke-train", type=int, default=20)
    ap.add_argument("--smoke-val", type=int, default=8)
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="datasets to skip: math dapo math500 aime24 aime25 amc23 gpqa_diamond",
    )
    args = ap.parse_args()
    if not args.stage_root or not args.out:
        print("ERROR: MC_STAGE_ROOT / MC_DATA_DIR unset (source configs/ada.env.sh)", file=sys.stderr)
        return 2
    stage, out = Path(args.stage_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}

    if "math" not in args.skip:
        print("== MATH train")
        ds = _load_local(_raw_dir(stage, MATH_REPO), None, "train")
        rows = [r.to_record() for r in math.rows_from_hf(ds, "train")]
        _write(
            pd.DataFrame(rows),
            out / "math_train.parquet",
            manifest,
            source=MATH_REPO,
            skipped_unboxed=len(ds) - len(rows),
        )
        _write(
            pd.DataFrame(rows[: args.smoke_train]), out / "smoke_train.parquet", manifest, source=MATH_REPO
        )

    if "dapo" not in args.skip:
        print("== DAPO-Math-17k (deduplicating)")
        ds = _load_local(_raw_dir(stage, DAPO_REPO), None, "train")
        unique, total = dapo.deduplicate(ds)
        ratio = total / max(len(unique), 1)
        print(f"   {total} rows -> {len(unique)} unique prompts (x{ratio:.1f} duplication)")
        if ratio < 2:
            print("   WARN expected ~100x duplication on the Hub file; check the source", file=sys.stderr)
        rows = [r.to_record() for r in dapo.rows_from_hf(unique)]
        _write(
            pd.DataFrame(rows),
            out / "dapo_train.parquet",
            manifest,
            source=DAPO_REPO,
            raw_rows=total,
            dedupe_ratio=ratio,
        )

    for name in EVAL_SOURCES:
        if name in args.skip:
            continue
        repo, config, split = eval_sets.HF_SPECS[name]
        print(f"== {name} ({repo})")
        try:
            ds = _load_local(_raw_dir(stage, repo), config, split)
        except FileNotFoundError as e:
            level = "WARN" if name == DS_GPQA else "ERROR"
            print(f"   {level} {e}")
            if name != DS_GPQA:
                return 1
            continue
        rows = [r.to_record() for r in eval_sets.rows_from_hf(name, ds)]
        _write(pd.DataFrame(rows), out / f"{name}.parquet", manifest, source=repo)
        if name == "math500":
            _write(pd.DataFrame(rows[: args.smoke_val]), out / "smoke_val.parquet", manifest, source=repo)

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest written: {out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
