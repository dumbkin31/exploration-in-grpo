# 008: The boxed-plus-digit validity rule, the training filter and the evaluation ceiling

**Rule** (CUTS paper Appendix B.4, settled by the brief): a math response is valid only if a
`\boxed{...}` is present and the content of the FINAL one contains at least one numeric digit;
the content is then compared with the ground truth by math-verify. Binary reward. No
`Answer:`-line fallback on the math path (GPQA has its own letter parser, `mixed_cuts.gpqa`).

**Consequence not covered by the brief**: MATH ground truths without any digit exist
(`\pi`, `e`, `\infty`, pure symbolic expressions). Under the rule such prompts can never be
rewarded, so their groups are permanently collapsed: zero gradient, wasted generation budget.

**Decisions**

1. `scripts/prepare_data.py` drops training prompts whose ground truth has no digit and records
   the count in `MANIFEST.json` (`dropped_no_digit_ground_truth`). Both arms train on the same
   filtered set, so the comparison is unaffected.
2. Evaluation sets are **not** filtered (the paper's numbers are on the full benchmarks).
   Instead the harness reports the per-benchmark **digit-rule ceiling** (fraction of ground
   truths that contain a digit) in `results.json`, the summary table and the README targets,
   together with the fraction of valid responses. A model that answers `\boxed{\pi}` correctly
   scores 0 on that problem by design.
3. `reward_extra_info` carries `has_boxed` and `valid` for every sample, so format failures are
   visible separately from wrong answers.
