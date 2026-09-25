"""Prove the CUTS path fires in a live vLLM engine (Task B, GPU layer).

Discriminator: vLLM applies custom logits processors BEFORE temperature scaling
(vllm/v1/sample/sampler.py: ``apply_logits_processors`` at forward() line 98, ``apply_temperature``
inside sample() at line 276; verified for v0.24.0). CUTS writes a constant over S_t, and dividing a
constant by any temperature leaves it constant, so CUTS output is temperature-invariant while
standard sampling collapses toward greedy at temperature 0.01.

With n=16 and a fixed seed every child request gets seed+index (parallel_sampling.py:80), so the
16 standard samples are near-identical at T=0.01 and the 16 CUTS samples stay diverse. If vLLM
ever moved the processor after temperature this test would fail and the fallback would be a
distinct-n comparison at temperature 1.0 with a looser threshold.
"""

from __future__ import annotations

import math

import pytest

from cuts.config import CutsParams
from cuts.stats import read_cuts_stats
from mc_data.schema import build_messages
from tests.conftest import needs_gpu, needs_vllm
from tests.gpu.conftest import first_divergence, unique_fraction

pytestmark = [needs_gpu, needs_vllm, pytest.mark.needs_gpu, pytest.mark.needs_vllm]

PROMPT = "Explain, in a short paragraph, why the sum of the first n odd numbers equals n squared."
T_WARM, K, DELTA = 5, 5, 0.03


def _generate(llm, sampling_params):
    from vllm import SamplingParams  # noqa: F401 - type hint only

    outs = llm.chat(
        [build_messages(PROMPT)],
        sampling_params=sampling_params,
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert len(outs) == 1 and len(outs[0].outputs) == 16
    return [list(o.token_ids) for o in outs[0].outputs]


def test_cuts_is_temperature_invariant_and_standard_is_not(llm, tmp_path):
    from vllm import SamplingParams

    common = dict(n=16, temperature=0.01, top_p=1.0, top_k=-1, seed=1234, max_tokens=48, ignore_eos=True)
    std_seqs = _generate(llm, SamplingParams(**common))
    cuts_params = CutsParams(
        k=K, delta=DELTA, t_warm=T_WARM, stats_dir=str(tmp_path), step=0, uid="inv", session_id=0
    )
    cuts_seqs = _generate(llm, SamplingParams(**common, extra_args=cuts_params.to_extra_args()))

    std_unique, cuts_unique = unique_fraction(std_seqs), unique_fraction(cuts_seqs)
    print(f"unique fraction: standard {std_unique:.3f}  cuts {cuts_unique:.3f}")
    assert std_unique <= 0.25, (
        f"standard arm should be near-greedy at T=0.01, got unique fraction {std_unique}"
    )
    assert cuts_unique >= 0.875, f"CUTS arm should stay diverse at T=0.01, got unique fraction {cuts_unique}"

    div = first_divergence(cuts_seqs)
    lo = min(T_WARM, first_divergence(std_seqs) or T_WARM)
    print(f"first divergence: standard {first_divergence(std_seqs)}  cuts {div}  (bounds {lo}..{3 * T_WARM})")
    assert div is not None and lo <= div <= 3 * T_WARM, (
        f"CUTS completions must diverge within 3*T_warm tokens, got {div}"
    )

    # |S_t| statistics: records are flushed by the engine's NEXT forward pass, so run a tiny standard
    # request first. All 16 children share one session_id, so read without de-duplication.
    _generate(llm, SamplingParams(n=16, temperature=1.0, max_tokens=1))
    records = read_cuts_stats(tmp_path, step=0)
    assert len(records) == 16, f"expected one stats record per CUTS sample, got {len(records)}"
    steps = sum(r["n_cuts_steps"] for r in records)
    mean_set = sum(r["mean_set_size"] * r["n_cuts_steps"] for r in records) / steps
    singleton = sum(r["n_singleton"] for r in records)
    print(f"mean |S_t| = {mean_set:.3f} over {steps} CUTS steps; singleton rate {singleton / steps:.3f}")
    assert 1.0 < mean_set < float(K), (
        f"mean |S_t| = {mean_set}: pinned at 1 means delta ate everything (greedy), at K means the filter never fires"
    )
    assert singleton < steps
    assert all(r["n_generated"] == 48 and r["n_cuts_steps"] == 48 - T_WARM for r in records)


def test_uniform_over_survivors_is_exact_in_engine(llm, tmp_path):
    """Sanity: with delta=0 and k=2, the CUTS set has exactly 2 tokens for every step after T_warm."""
    from vllm import SamplingParams

    p = CutsParams(k=2, delta=0.0, t_warm=T_WARM, stats_dir=str(tmp_path), step=1, uid="k2", session_id=0)
    _generate(
        llm,
        SamplingParams(
            n=16, temperature=1.0, seed=7, max_tokens=24, ignore_eos=True, extra_args=p.to_extra_args()
        ),
    )
    _generate(llm, SamplingParams(n=16, temperature=1.0, max_tokens=1))
    records = read_cuts_stats(tmp_path, step=1)
    assert records and all(math.isclose(r["mean_set_size"], 2.0) for r in records)
