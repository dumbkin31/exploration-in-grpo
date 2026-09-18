"""verl integration for Mixed-CUTS GRPO.

Pure, verl-independent pieces (unit-tested on CPU):

- :mod:`mixed_cuts.scheduler`   -- ``MixedCutsConfig`` and ``plan_group``: which of a prompt's
                                   ``n`` rollouts are standard and which are CUTS.
- :mod:`mixed_cuts.diagnostics` -- group-level reward statistics (advantage collapse rate, ...).
- :mod:`mixed_cuts.reward`      -- math-verify reward with last-``\\boxed{}`` extraction.
- :mod:`mixed_cuts.jsonl_logger` -- local JSONL metrics log (primary, W&B is secondary).

verl-dependent pieces (imported inside the Ray task runner, see :mod:`mixed_cuts.main`):

- :mod:`mixed_cuts.agent_loop`  -- ``MixedCutsAgentLoopManagerTQ``: the rollout-worker hook.
- :mod:`mixed_cuts.trainer`     -- ``MixedCutsPPOTrainerSync``: the diagnostics hook.
"""
