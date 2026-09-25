"""Exceptions that must stop a run at startup with a readable message (caught in mixed_cuts.main)."""


class MixedCutsStartupError(RuntimeError):
    """A first-step invariant failed (thinking mode on, rollout logprobs leaked into old_log_probs, ...)."""
