# 006: fp16 stability watchlist and mitigation ladder (Task D)

Every RL paper we follow trained in bf16. On sm_75 everything is fp16 (5 exponent bits), with a
sharded grad scaler in verl's FSDP engine. "It did not crash" is not the bar.

## Automatic checks (`src/mixed_cuts/stability.py`, every step)

| Alert | Rule | Metric |
|---|---|---|
| NaN / inf | any `actor/*` metric is NaN or inf | `stability/alert_nan` |
| grad-norm spike | `actor/grad_norm` > `grad_norm_max` (50) or > `grad_norm_spike_factor` (10) x running median of the last 20 steps | `stability/alert_grad_spike`, `stability/grad_norm_over_median` |
| entropy collapse | in the first `entropy_collapse_window` (20) steps, `actor/entropy` < `entropy_collapse_ratio` (0.25) x step-1 entropy | `stability/alert_entropy_collapse`, `stability/entropy_over_step1` |

Alerts print a banner in the job log and set the metric to 1 (W&B + `metrics.jsonl`);
`stability.abort_on_nan: true` turns the NaN alert into a hard stop. History is rebuilt from
`metrics.jsonl` after a resume. The `<think>`-tag and D1 checks (001) run once at the first step.

## Ladder (apply one rung at a time; record which was used as a deviation from the paper)

1. `actor.optim.lr`: 1e-6 -> 5e-7.
2. `actor.grad_clip`: 1.0 -> 0.5.
3. Reconsider the KL term (`kl_loss_coef` 1e-3 -> 5e-3, or `kl_loss_type`).

Whatever rung ends up in a reported run goes into its config file and the README's deviations
list.
