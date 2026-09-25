"""The CUTS operator: SELECT top-K, FILTER by probability, EQUALIZE to uniform.

Given a batch of next-token logits, for every *active* row:

1. SELECT   -- keep the ``k`` most likely tokens;
2. FILTER   -- among those, keep the ones with probability ``>= delta`` (probabilities are
               computed with a temperature-1 softmax in float32, as in the paper);
3. EQUALIZE -- give every survivor logit ``0`` and every other token the most negative finite
               value of the output dtype (``torch.finfo(dtype).min``) so that a softmax yields the
               *uniform* distribution over survivors (``exp(min - 0)`` underflows to exactly 0).
               The model's relative ordering inside the candidate set is discarded on purpose;
               this is what distinguishes CUTS from top-k / nucleus / min-p sampling.
               ``-inf`` is deliberately NOT used: on fp16 hardware it propagates into NaN through
               downstream normalisation. Note the value must be the *output* dtype's min: filling
               the fp32 working tensor with fp32's min and casting to fp16 overflows back to -inf.

If the filter leaves nothing, ``empty_set_fallback`` decides between the paper's rule
(uniform over the whole top-K set) and greedy (argmax only). Either way the set is never
empty, and the argmax token always survives (it is in the top-K by construction).

Prefix protection (T_warm) is *not* handled here: whether a row is active is decided by the
caller (see :mod:`cuts.state`), because it needs per-request token counts.

The function is pure and batched; inactive rows are returned untouched, so a single call can
serve a vLLM batch mixing standard and CUTS requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cuts.config import EMPTY_SET_FALLBACKS


@dataclass
class CutsResult:
    logits: torch.Tensor
    """Same shape and dtype as the input; active rows rewritten, inactive rows untouched."""
    set_size: torch.Tensor
    """``|S_t|`` per row (int64). 0 for inactive rows."""
    used_fallback: torch.Tensor
    """bool per row: the filter emptied the set and the fallback rule fired. False for inactive rows."""


def cuts_transform(
    logits: torch.Tensor,
    k: int,
    delta: float,
    active_mask: torch.Tensor | None = None,
    empty_set_fallback: str = "topk",
    inplace: bool = False,
) -> CutsResult:
    """Apply CUTS to the active rows of a ``[batch, vocab]`` logits tensor.

    Args:
        logits: ``[B, V]`` next-token logits (any float dtype; fp16 on the 2080 Ti).
        k: SELECT size. Values larger than the vocabulary are clamped.
        delta: FILTER threshold on temperature-1 probabilities, in ``[0, 1]``.
        active_mask: ``[B]`` bool; rows with ``False`` are returned unchanged. ``None`` = all rows.
        empty_set_fallback: ``"topk"`` (paper) or ``"argmax"``.
        inplace: if True, active rows of ``logits`` are overwritten and the same tensor is returned
            (vLLM allows in-place modification; this avoids cloning a ``[batch, 152k]`` tensor per
            decoding step). Default False returns a new tensor and leaves the input untouched.

    Returns:
        :class:`CutsResult`.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [batch, vocab], got shape {tuple(logits.shape)}")
    if empty_set_fallback not in EMPTY_SET_FALLBACKS:
        raise ValueError(
            f"empty_set_fallback must be one of {EMPTY_SET_FALLBACKS}, got {empty_set_fallback!r}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    batch, vocab = logits.shape
    device = logits.device
    if active_mask is None:
        active_mask = torch.ones(batch, dtype=torch.bool, device=device)
    else:
        active_mask = active_mask.to(device=device, dtype=torch.bool)
        if active_mask.shape != (batch,):
            raise ValueError(f"active_mask must be [batch]={batch}, got {tuple(active_mask.shape)}")

    set_size = torch.zeros(batch, dtype=torch.int64, device=device)
    used_fallback = torch.zeros(batch, dtype=torch.bool, device=device)
    if not bool(active_mask.any()):
        return CutsResult(logits if inplace else logits.clone(), set_size, used_fallback)

    k_eff = min(k, vocab)

    # Work only on the active rows; float32 softmax so fp16 logits do not lose small probabilities.
    active_idx = active_mask.nonzero(as_tuple=False).squeeze(1)
    sub = logits.index_select(0, active_idx).float()
    probs = torch.softmax(sub, dim=-1)

    # SELECT: top-K, sorted descending, so column 0 is the argmax of every row.
    top_probs, top_idx = torch.topk(probs, k=k_eff, dim=-1, largest=True, sorted=True)

    # FILTER: absolute threshold on temperature-1 probabilities.
    survive = top_probs >= delta
    n_survive = survive.sum(dim=-1)

    # Never let the candidate set be empty.
    empty = n_survive == 0
    if bool(empty.any()):
        if empty_set_fallback == "topk":
            survive[empty] = True  # uniform over the whole top-K set (paper)
        else:
            survive[empty, 0] = True  # greedy: only the argmax survives
    fallback_rows = empty

    # EQUALIZE: 0 for survivors, finfo(out dtype).min elsewhere -> softmax is uniform over survivors.
    mask_value = torch.finfo(logits.dtype).min  # exactly representable after the cast below
    new_sub = torch.full_like(sub, mask_value)
    src = torch.where(survive, torch.zeros_like(top_probs), torch.full_like(top_probs, mask_value))
    new_sub.scatter_(1, top_idx, src)

    out = logits if inplace else logits.clone()
    out.index_copy_(0, active_idx, new_sub.to(logits.dtype))
    set_size.index_copy_(0, active_idx, survive.sum(dim=-1))
    used_fallback.index_copy_(0, active_idx, fallback_rows)
    return CutsResult(out, set_size, used_fallback)
