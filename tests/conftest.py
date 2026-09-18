"""Shared test helpers. Tests never need a GPU, verl, or vLLM unless marked."""

from __future__ import annotations

import importlib.util

import pytest
import torch


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAS_VLLM = _has("vllm")
HAS_VERL = _has("verl")

needs_vllm = pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed (CPU dev env)")
needs_verl = pytest.mark.skipif(not HAS_VERL, reason="verl not installed (CPU dev env)")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def logits_from_probs(
    probs: list[list[float]] | list[float], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Build logits whose temperature-1 softmax equals ``probs`` exactly (up to fp error)."""
    p = torch.tensor(probs, dtype=torch.float64)
    if p.dim() == 1:
        p = p.unsqueeze(0)
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0], dtype=torch.float64)), "rows must sum to 1"
    return torch.log(p).to(dtype)
