# 001: CUTS is a proposal distribution; no importance correction (D1)

**Status**: settled by the brief; implementation details recorded here.

## Decision

Mixed-CUTS rollouts are trained with the standard clipped PPO objective against
`pi_theta_old`, with **no** importance-sampling or rejection-sampling correction for the fact
that CUTS rollouts were drawn from a different behaviour policy `mu`.

Source: Liang et al. (arXiv 2604.18493), Section 2.3, paragraph "On-policy vs. Behavior Policy":
although Mixed-CUTS induces a mixed behaviour policy, the authors retain the clipped objective
with `pi_theta_old`. The off-policy bias is argued to be bounded because CUTS only redistributes
mass inside the top-K set, the `delta` threshold prunes tail tokens, and PPO clipping caps the
per-step update. The Limitations section acknowledges this as an unquantified deviation.

## The hazard this creates

vLLM computes the logprob it *reports* for a sampled token **after** the logits processors,
temperature and top-k/top-p have run (`vllm/v1/sample/ops/topk_topp_sampler.py:139`, mode
`processed_logprobs`, verl's default). Our EQUALIZE step gives every survivor logit 0, so for a
CUTS token the reported logprob is exactly `log(1/|S_t|)`: the proposal `Q`, not
`pi_theta_old`. If those numbers ever reached `old_log_probs`, the ratio
`pi_theta / old` would silently become a `pi / Q` correction and cancel the whole method.

## Implementation (verified against verl v0.9.0 source)

| Flag | Value | Why |
|---|---|---|
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | rollout logprobs never enter TransferQueue (`agent_loop.py:124-126`); verl's V1 sync trainer tolerates the missing field (TransferQueue drops unknown `select_fields`). Side effect: no `rollout_corr/*` metrics. |
| `algorithm.rollout_correction.bypass_mode` | `false` | the actor recomputes `old_log_probs` every step (`trainer_base.py:1487-1503`); `true` would copy `rollout_log_probs` |
| `algorithm.rollout_correction.rollout_is` / `rollout_rs` | `null` | no IS weights are ever built (`rollout_corr_helper.py:854`), so nothing multiplies `pg_loss` (`core_algos.py:1363`) |

Code-level guard (`src/mixed_cuts/trainer.py`): after the first `_compute_old_log_prob` of a
process, the trainer measures `cuts/frac_old_logprob_uniform_like_k2plus`, the fraction of CUTS
response tokens (after `T_warm`) whose `old_log_probs` lies within 1e-4 of `-log(j)` for
`j = 2..K`. Under actor recompute this is near chance; if engine logprobs leaked in it is ~1.0.
Above 0.5 at the first step the run stops before any update (`MixedCutsStartupError`, exit 2).
`j = 1` is excluded from the alarm because near-deterministic tokens legitimately have logprob
~0; the `..._any` variant is logged for information. Tests: `tests/test_diagnostics.py`
(synthetic), `tests/gpu/test_logprob_hazard_demo.py` (the engine really reports `-log|S_t|`),
`tests/gpu/test_smoke_run.py` (the trained run's fraction stays below 0.5).

`compose_config --check` refuses any config that flips these flags.
