#!/usr/bin/env python
"""Download the model and every dataset into the durable stage root (login node, has internet).

Pure ``huggingface_hub`` downloads so it runs on the CentOS 7 login node with the tiny
``requirements/prefetch.txt`` environment. Parquet conversion happens later on a compute node
(``scripts/prepare_data.py``). Idempotent and resumable: ``snapshot_download`` skips complete
files. A ``MANIFEST.json`` records the commit hash and size (and sha256 for files below
``--checksum-max-mb``) of every downloaded file so a partial copy is detected.

Layout under ``$MC_STAGE_ROOT``::

    models/<model-basename>/           HF snapshot (config, tokenizer, safetensors)
    raw/<repo-id with / -> __>/        HF dataset repo snapshot
    MANIFEST.json

GPQA (``Idavidrein/gpqa``) is gated: exported ``HF_TOKEN`` is required, otherwise it is skipped
with a warning and GPQA evaluation is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

DATASET_REPOS = [
    "DigitalLearningGmbH/MATH-lighteval",
    "BytedTsinghua-SIA/DAPO-Math-17k",
    "HuggingFaceH4/MATH-500",
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/amc23",
    "Idavidrein/gpqa",  # gated
]


def repo_dir(root: Path, repo_id: str) -> Path:
    return root / "raw" / repo_id.replace("/", "__")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def describe_tree(root: Path, checksum_max_bytes: int) -> dict:
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            rel = str(p.relative_to(root))
            size = p.stat().st_size
            entry: dict = {"size": size}
            if size <= checksum_max_bytes:
                entry["sha256"] = sha256_of(p)
            files[rel] = entry
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--stage-root", default=os.environ.get("MC_STAGE_ROOT"), help="defaults to $MC_STAGE_ROOT"
    )
    ap.add_argument("--model", default=os.environ.get("MC_MODEL_NAME", "Qwen/Qwen3-1.7B"))
    ap.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="additional model repo(s), e.g. Qwen/Qwen3-1.7B-Base",
    )
    ap.add_argument("--datasets", nargs="*", default=DATASET_REPOS)
    ap.add_argument(
        "--checksum-max-mb",
        type=float,
        default=200.0,
        help="sha256 files up to this size (model shards are size-checked only)",
    )
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    if not args.stage_root:
        print("ERROR: --stage-root or MC_STAGE_ROOT required (source configs/ada.env.sh)", file=sys.stderr)
        return 2

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    root = Path(args.stage_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}, "datasets": {}}
    )
    token = os.environ.get("HF_TOKEN") or None
    max_bytes = int(args.checksum_max_mb * 2**20)

    models = [] if args.skip_model else [args.model, *args.extra_model]
    for repo in models:
        dst = root / "models" / repo.split("/")[-1]
        print(f"== model {repo} -> {dst}")
        t0 = time.time()
        snapshot_download(
            repo_id=repo,
            local_dir=str(dst),
            token=token,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.md"],
        )
        manifest["models"][repo] = {
            "path": str(dst),
            "files": describe_tree(dst, max_bytes),
            "fetched": time.time(),
        }
        print(f"   done in {time.time() - t0:.0f}s")

    for repo in args.datasets:
        dst = repo_dir(root, repo)
        print(f"== dataset {repo} -> {dst}")
        t0 = time.time()
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dst), token=token)
        except GatedRepoError:
            print(f"   WARN gated repo {repo}: export HF_TOKEN (accept the terms on the Hub first); skipping")
            continue
        except RepositoryNotFoundError:
            print(f"   WARN {repo} not found (or gated without a token); skipping")
            continue
        manifest["datasets"][repo] = {
            "path": str(dst),
            "files": describe_tree(dst, max_bytes),
            "fetched": time.time(),
        }
        print(f"   done in {time.time() - t0:.0f}s")

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest written: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
