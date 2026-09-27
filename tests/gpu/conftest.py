"""GPU-only tests (Task B). Run inside an allocation AFTER the smoke test:

    MC_SMOKE_RUN_DIR=/scratch/$USER/mixed-cuts/runs/smoke-s42-<job> make gpu-test

They need vLLM, one GPU and the staged Qwen3-1.7B (``MC_STAGED_MODEL_DIR`` or ``MC_MODEL_DIR``).
One vLLM engine is shared per test module (module-scoped fixture) to keep start-up cost down.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# pytest has torch (and often CUDA) initialised before a fixture builds an engine; vLLM must spawn its
# engine core, not fork it ("Cannot re-initialize CUDA in forked subprocess").
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def model_dir() -> str | None:
    for var in ("MC_STAGED_MODEL_DIR", "MC_MODEL_DIR"):
        p = os.environ.get(var)
        if p and Path(p, "config.json").exists():
            return p
    return None


def first_divergence(seqs: list[list[int]]) -> int | None:
    """First position where the token sequences are not all identical ("ended" counts as a symbol)."""
    max_len = max(len(s) for s in seqs)
    for t in range(max_len):
        if len({s[t] if t < len(s) else -1 for s in seqs}) > 1:
            return t
    return None


def unique_fraction(seqs: list[list[int]]) -> float:
    return len({tuple(s) for s in seqs}) / len(seqs)


@pytest.fixture(scope="module")
def llm(tmp_path_factory):
    vllm = pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    path = model_dir()
    if path is None:
        pytest.skip("model not staged (MC_STAGED_MODEL_DIR / MC_MODEL_DIR)")
    from cuts.vllm_logits_processor import CutsLogitsProcessor

    engine = vllm.LLM(
        model=path,
        dtype="float16",  # sm_75: no bf16
        tensor_parallel_size=1,
        # 0.6 x 11 GiB = 6.3 GiB: the 3.2 GiB fp16 model + CUDA-graph profiling + ~1.9 GiB KV. At 0.4 (a leftover from the
        # 4-way-sharded era) vLLM 0.24 reports "Available KV cache memory: -0.14 GiB" on a whole 2080 Ti (smoke job 2719792).
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        logits_processors=[CutsLogitsProcessor],
        attention_config={"backend": "TRITON_ATTN"},
        # verl builds its rollout engine with processed log-probs (the D1 hazard the demo test shows);
        # vLLM's own default is raw_logprobs, under which only singleton steps sit on the -log|S_t| lattice
        logprobs_mode="processed_logprobs",
        seed=0,
    )
    yield engine
    del engine
