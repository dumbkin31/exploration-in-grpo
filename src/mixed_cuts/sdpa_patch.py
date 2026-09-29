"""Make transformers' SDPA attention repeat the KV heads instead of requesting ``enable_gqa`` (Turing).

Measured on the RTX 2080 Ti (sm_75, torch 2.11.0+cu130, decision 012): PyTorch's memory-efficient SDPA
kernel runs at 6,024 tokens with a mask for +0.36 GiB (forward + backward), but it has no grouped-query
variant on this card, and neither does flash. transformers 5.x calls
``F.scaled_dot_product_attention(..., enable_gqa=True)`` whenever ``attention_mask is None`` (a
single-sequence micro-batch without padding: exactly the long ones), and torch then falls back to the
math kernel: +8.8 GiB for one attention call at 6,024 tokens, which is the update-pass OOM of
smoke_maxlen job 2719481. With the KV heads repeated (what transformers does when a mask is present)
the memory-efficient kernel is used.

Installed in every Ray worker by ``mixed_cuts.worker_hooks.install``, the
``ray_kwargs.ray_init.runtime_env.worker_process_setup_hook`` (configs/train/base_grpo.yaml), so the
FSDP worker has it before the model is built. Idempotent.
"""

from __future__ import annotations

import os

_FLAG = "_mixed_cuts_no_gqa"


def install() -> None:
    import transformers.integrations.sdpa_attention as sdpa

    if getattr(sdpa, _FLAG, False):
        return

    def use_gqa_in_sdpa(attention_mask, key) -> bool:  # noqa: ARG001 - transformers' signature
        return False

    sdpa.use_gqa_in_sdpa = use_gqa_in_sdpa
    setattr(sdpa, _FLAG, True)
    print(
        f"[mixed_cuts] sdpa_patch installed in pid {os.getpid()}: HF SDPA repeats KV heads (no enable_gqa)",
        flush=True,
    )
