| model | checkpoint | benchmark | pass@1 [95% CI] | maj@16 | pass@16 | mean tokens | truncated |
|---|---|---|---|---|---|---|---|
| base-qwen3-1.7b-h200 | base | math500 | 70.2 [67.0, 73.3] | 79.0 | 91.2 | 850 | 1.2% |
| base-qwen3-1.7b-h200 | base | aime24 | 13.8 [6.5, 22.1] | 20.0 | 46.7 | 2548 | 10.8% |
| base-qwen3-1.7b-h200 | base | aime25 | 10.6 [3.8, 19.6] | 16.7 | 30.0 | 2205 | 4.4% |
| base-qwen3-1.7b-h200 | base | amc23 | 42.8 [32.3, 54.2] | 57.5 | 80.0 | 1406 | 1.4% |
| base-qwen3-1.7b-h200 | base | gpqa_diamond | 32.9 [29.1, 37.0] | 37.9 | 84.3 | 1278 | 1.6% |
| math_grpo-h200-s1 | global_step_40 | math500 | 78.1 [75.2, 80.9] | 86.4 | 91.4 | 1258 | 7.3% |
| math_grpo-h200-s1 | global_step_40 | aime24 | 23.5 [11.2, 36.9] | 36.7 | 40.0 | 3935 | 55.0% |
| math_grpo-h200-s1 | global_step_40 | aime25 | 15.8 [6.7, 26.9] | 30.0 | 30.0 | 3818 | 49.6% |
| math_grpo-h200-s1 | global_step_40 | amc23 | 53.0 [42.8, 63.9] | 75.0 | 87.5 | 2355 | 21.1% |
| math_grpo-h200-s1 | global_step_40 | gpqa_diamond | 35.4 [31.5, 39.9] | 39.4 | 84.3 | 2805 | 20.2% |
| math_mixed_cuts-h200-s1 | global_step_40 | math500 | 71.5 [68.1, 74.9] | 79.0 | 89.0 | 963 | 3.3% |
| math_mixed_cuts-h200-s1 | global_step_40 | aime24 | 16.0 [6.0, 27.9] | 20.0 | 30.0 | 2978 | 27.3% |
| math_mixed_cuts-h200-s1 | global_step_40 | aime25 | 6.7 [2.1, 13.1] | 13.3 | 26.7 | 2578 | 16.7% |
| math_mixed_cuts-h200-s1 | global_step_40 | amc23 | 43.1 [32.5, 55.5] | 55.0 | 80.0 | 1713 | 8.9% |
| math_mixed_cuts-h200-s1 | global_step_40 | gpqa_diamond | 31.5 [27.2, 35.7] | 34.3 | 78.8 | 1448 | 5.5% |
| math_mixed_cuts-h200-s1 | global_step_50 | math500 | 67.4 [63.9, 70.8] | 75.6 | 85.4 | 1075 | 7.9% |
| math_mixed_cuts-h200-s1 | global_step_50 | aime24 | 8.5 [2.3, 16.9] | 20.0 | 26.7 | 3396 | 47.7% |
| math_mixed_cuts-h200-s1 | global_step_50 | aime25 | 3.5 [0.4, 9.0] | 6.7 | 16.7 | 2973 | 36.9% |
| math_mixed_cuts-h200-s1 | global_step_50 | amc23 | 34.4 [24.1, 46.3] | 37.5 | 72.5 | 2012 | 20.9% |
| math_mixed_cuts-h200-s1 | global_step_50 | gpqa_diamond | 33.0 [29.2, 37.3] | 38.9 | 78.3 | 1765 | 18.3% |
