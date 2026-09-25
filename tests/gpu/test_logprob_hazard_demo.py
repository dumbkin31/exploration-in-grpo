"""D1 hazard, demonstrated at the engine level (informational + one assertion).

Under verl's default ``logprobs_mode = processed_logprobs`` the logprob vLLM reports for a token is
taken AFTER the logits processors, temperature and top-k/top-p
(vllm/v1/sample/ops/topk_topp_sampler.py:139). For a CUTS request the survivors all carry logit 0,
so every sampled token after T_warm reports exactly ``log(1/|S_t|)``: the proposal Q, not
``pi_theta_old``. This is why ``rollout.calculate_log_probs`` is false and the actor recomputes
``old_log_probs`` (docs/decisions/001).
"""

from __future__ import annotations

import math

import pytest

from cuts.config import CutsParams
from mc_data.schema import build_messages
from tests.conftest import needs_gpu, needs_vllm

pytestmark = [needs_gpu, needs_vllm, pytest.mark.needs_gpu, pytest.mark.needs_vllm]

T_WARM, K = 5, 5
LATTICE = [-math.log(j) for j in range(1, K + 1)]


def test_engine_reports_log_one_over_set_size_for_cuts_tokens(llm, tmp_path):
    from vllm import SamplingParams

    p = CutsParams(k=K, delta=0.03, t_warm=T_WARM, stats_dir=str(tmp_path), step=0, uid="lp", session_id=0)
    sp = SamplingParams(
        n=4,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        seed=3,
        max_tokens=64,
        ignore_eos=True,
        logprobs=0,
        extra_args=p.to_extra_args(),
    )
    outs = llm.chat(
        [build_messages("List three properties of prime numbers.")],
        sampling_params=sp,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    on_lattice = total = 0
    for o in outs[0].outputs:
        for t, (tok, lp_dict) in enumerate(zip(o.token_ids, o.logprobs, strict=True)):
            if t < T_WARM:
                continue
            lp = lp_dict[tok].logprob
            total += 1
            on_lattice += any(abs(lp - v) <= 1e-3 for v in LATTICE)
    frac = on_lattice / total
    print(
        f"{100 * frac:.1f}% of {total} CUTS tokens report a logprob of exactly -log|S_t| (processed_logprobs mode)"
    )
    assert frac >= 0.95, "expected the engine's logprobs to be the CUTS proposal Q for CUTS tokens"
