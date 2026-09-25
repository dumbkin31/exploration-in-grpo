# 007: How |S_t| statistics travel from vLLM to the trainer

The CUTS logits processor runs inside vLLM's engine processes; the trainer that logs metrics is
a Ray actor elsewhere. The channel is an append-only JSONL file per (training step, writer):

`<run dir>/cuts_stats/cuts_stats_step<N>_<writer id>.jsonl`, one record per finished CUTS
rollout (`uid`, `session_id`, `n_cuts_steps`, `mean_set_size`, `n_singleton`, `n_fallback`,
`set_size_hist`, ...). `stats_dir` and `step` travel inside the request's
`extra_args["cuts"]`, so nothing crosses a process boundary through environment variables.

Three facts shaped it (all verified in vLLM 0.24.0 / verl 0.9.0):

1. **Tensor parallelism**: every TP rank runs the sampler, hence the processor, on the full
   batch (`gpu_model_runner.py:654-690`). Only TP rank 0 owns a writer
   (`vllm.distributed.get_tensor_model_parallel_rank()`); the others discard records.
2. **Flush timing**: a finished request is removed from the persistent batch at the engine's
   *next* forward pass, so the longest rollouts of a generation batch are flushed when the next
   batch starts. The trainer therefore summarises step N at step N+1 (`cuts/stats_step` says
   which) and the smoke test needs at least two steps. The last step of a run is flushed by the
   final validation.
3. **Restarts and PID reuse**: files are keyed by a unique writer id (PID + random suffix,
   because PIDs repeat across nodes and jobs), and the trainer snapshots the files that already
   exist at start-up and ignores them (a killed job leaves a partial step behind).

Metrics derived per step: `cuts/set_size_mean`, `cuts/set_size_hist/k=i`,
`cuts/degenerate_to_greedy_rate` (|S_t| == 1), `cuts/frac_steps_fallback`,
`cuts/cuts_steps_per_rollout`, `cuts/frac_rollouts_never_active`, `cuts/n_rollouts`.
