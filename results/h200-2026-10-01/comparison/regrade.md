Re-grade of `jarvis-results-2026-10-01` with `scripts/regrade_eval.py` (no model re-run). Cells show the published
number, or `old → **new**` where the current grader changes it (in %). Flips count single responses.

| model | checkpoint | benchmark | pass@1 | maj@16 | pass@16 | wrong→right | right→wrong |
|---|---|---|---|---|---|---|---|
| base-qwen3-1.7b-h200 | base | aime24 | 13.8 | 20.0 | 46.7 | 0 | 0 |
| base-qwen3-1.7b-h200 | base | aime25 | 10.6 | 16.7 | 30.0 | 0 | 0 |
| base-qwen3-1.7b-h200 | base | amc23 | 42.8 | 57.5 | 80.0 | 0 | 0 |
| base-qwen3-1.7b-h200 | base | gpqa_diamond | 32.9 | 37.9 | 84.3 | 0 | 0 |
| base-qwen3-1.7b-h200 | base | math500 | 70.2 | 79.0 | 91.2 | 0 | 0 |
| math_grpo-h200-s1 | global_step_40 | aime24 | 23.5 | 36.7 | 40.0 | 0 | 0 |
| math_grpo-h200-s1 | global_step_40 | aime25 | 15.8 | 30.0 | 30.0 | 0 | 0 |
| math_grpo-h200-s1 | global_step_40 | amc23 | 53.0 | 75.0 | 87.5 | 0 | 0 |
| math_grpo-h200-s1 | global_step_40 | gpqa_diamond | 35.4 | 39.4 | 84.3 | 0 | 0 |
| math_grpo-h200-s1 | global_step_40 | math500 | 78.1 | 86.4 | 91.4 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_40 | aime24 | 16.0 | 20.0 | 30.0 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_40 | aime25 | 6.7 | 13.3 | 26.7 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_40 | amc23 | 43.1 | 55.0 | 80.0 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_40 | gpqa_diamond | 31.5 | 34.3 | 78.8 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_40 | math500 | 71.5 | 79.0 | 89.0 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_50 | aime24 | 8.5 | 20.0 | 26.7 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_50 | aime25 | 3.5 | 6.7 | 16.7 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_50 | amc23 | 34.4 | 37.5 | 72.5 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_50 | gpqa_diamond | 33.0 | 38.9 | 78.3 | 0 | 0 |
| math_mixed_cuts-h200-s1 | global_step_50 | math500 | 67.4 | 75.6 | 85.4 | 0 | 0 |

51072 responses re-graded: 0 wrong→right, 0 right→wrong.
