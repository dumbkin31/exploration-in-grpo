# Mixed-CUTS vs GRPO: results

## Evaluation

pass@1 with bootstrap 95% CI, then pass@16 and maj@16, all in %. 16 samples per problem.

| model | checkpoint | aime24 pass@1 | aime24 pass@16 | aime24 maj@16 | aime25 pass@1 | aime25 pass@16 | aime25 maj@16 | amc23 pass@1 | amc23 pass@16 | amc23 maj@16 | gpqa_diamond pass@1 | gpqa_diamond pass@16 | gpqa_diamond maj@16 | math500 pass@1 | math500 pass@16 | math500 maj@16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base-qwen3-1.7b-h200 | base | 13.8 [6.5, 22.1] | 46.7 | 20.0 | 10.6 [3.8, 19.6] | 30.0 | 16.7 | 42.8 [32.3, 54.2] | 80.0 | 57.5 | 32.9 [29.1, 37.0] | 84.3 | 37.9 | 70.2 [67.0, 73.3] | 91.2 | 79.0 |
| math_grpo-h200-s1 | global_step_40 | 23.5 [11.2, 36.9] | 40.0 | 36.7 | 15.8 [6.7, 26.9] | 30.0 | 30.0 | 53.0 [42.8, 63.9] | 87.5 | 75.0 | 35.4 [31.5, 39.9] | 84.3 | 39.4 | 78.1 [75.2, 80.9] | 91.4 | 86.4 |
| math_mixed_cuts-h200-s1 | global_step_40 | 16.0 [6.0, 27.9] | 30.0 | 20.0 | 6.7 [2.1, 13.1] | 26.7 | 13.3 | 43.1 [32.5, 55.5] | 80.0 | 55.0 | 31.5 [27.2, 35.7] | 78.8 | 34.3 | 71.5 [68.1, 74.9] | 89.0 | 79.0 |

## Response length (tokens)

Mean generated tokens per sample: all / correct / incorrect, and % cut off at max_tokens.

| model | aime24 | aime25 | amc23 | gpqa_diamond | math500 |
|---|---|---|---|---|---|
| base-qwen3-1.7b-h200 | 2548 / 1687 / 2685 (10.8%) | 2205 / 1638 / 2272 (4.4%) | 1406 / 959 / 1741 (1.4%) | 1278 / 1279 / 1278 (1.6%) | 850 / 562 / 1530 (1.2%) |
| math_grpo-h200-s1 | 3935 / 2005 / 4529 (55.0%) | 3818 / 1827 / 4193 (49.6%) | 2355 / 1362 / 3473 (21.1%) | 2805 / 2657 / 2887 (20.2%) | 1258 / 808 / 2863 (7.3%) |
| math_mixed_cuts-h200-s1 | 2978 / 1559 / 3249 (27.3%) | 2578 / 2046 / 2616 (16.7%) | 1713 / 1048 / 2217 (8.9%) | 1448 / 1377 / 1480 (5.5%) | 963 / 593 / 1891 (3.3%) |

## Training

Means over the last 10 logged steps.

| run | last step | train reward | advantage collapse rate | ACR_100 | all-correct groups | all-wrong groups | entropy | response length | last validation (MATH-500 mean@4) |
|---|---|---|---|---|---|---|---|---|---|
| math_grpo-h200-s1 | 72 | 0.740 | 0.605 | 0.591 | 0.499 | 0.105 | 0.153 | 2016.798 | 0.695 |
| math_mixed_cuts-h200-s1 | 61 | 0.212 | 0.779 | 0.544 | 0.109 | 0.670 | 4.063 | 3601.849 | 0.003 |

## Compute

| model | training GPU-hours | minutes per step | evaluation GPU-hours | cost at INR 188.73/GPU-h |
|---|---|---|---|---|
| base-qwen3-1.7b-h200 | - | - | 0.2 | 35 |
| math_grpo-h200-s1 | 10.2 | 8.6 | 0.5 | 2,013 |
| math_mixed_cuts-h200-s1 | 10.3 | 10.2 | 0.2 | 1,999 |

Total: 21.4 GPU-hours, INR 4,046. Training hours are the
sum of logged step times; evaluation hours are vLLM generation time. Setup, smoke tests, model
loading and idle time come on top: take the billed total from the provider's invoice.
