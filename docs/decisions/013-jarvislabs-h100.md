# 013: Both arms on Jarvislabs H100s (bf16, same LoRA), and the learning-rate bug it exposed

**Date**: 2026-09-29. **Status**: accepted by the user ("write the jarvis code, use the new run names",
arms in parallel). Supersedes 012 as the place the two deliverables are trained; the Ada setup stays
intact and selectable.

## Context

Ada access ended with the deadline close. Arm 1 had completed one full-scale step there (job 2721833)
and died in the weight sync three times before the fix in 012; its step-1 checkpoint is on Ada and
unreachable. The course sponsors each team with INR 5,880 of Jarvislabs credit, with no top-up and no
availability guarantee near the deadline, and the report must justify the use of the credit.

Prices (jarvislabs.ai/in, 2026-09-29, per GPU-hour): H100 SXM 80 GB INR 255.15 on demand, INR 112.59 spot;
H200 INR 378.27 / 188.73; RTX Pro 6000 INR 179.01 / 93.96; A100 80 GB INR 140.94 / 84.24. Per-minute
billing; storage INR 0.013 per GB-hour; only `/home` survives a pause, a filesystem volume survives
deletion. What happens to a spot instance's disk on preemption is not documented.

## Decision

* **One H100 per arm, both arms in parallel** (a 2-GPU instance, or two 1-GPU instances with
  `ARMS=...`). Same GPU-hours as running them in sequence, half the wall-clock time. Never both on one
  GPU. The H100 is the best value here: the H200 costs 1.5x for mostly faster decoding, the RTX Pro 6000
  is about half the speed on a newer architecture with shakier attention-kernel support, and the
  A100 80 GB is similar value at twice the wall-clock (the fallback if H100s are sold out).
* **Spot when available**: the worst-case estimate below fits the credit on spot with ~INR 1,500 to
  spare; on demand only the optimistic case fits. Checkpoints every step (0.9 GB, LoRA-only) in the
  durable run dir; `jarvis/start.sh` after a resume restarts exactly the stopped arms.
* **The same training recipe as the Ada plan**: LoRA r=64 / alpha=128 on all linear layers, lr 1e-5,
  128 prompts x G=16, mini-batch 32, 5,000-token responses, KL 1e-3, CUTS K=5 / delta=0.03 / T_warm=5,
  100 steps, the same seeds. Changed only where the hardware allows better: **bf16** instead of fp16
  (fp32 LoRA masters as before), no CPU offload (`memory=plan_b_lora_gpu`), vLLM `FLASH_ATTN` instead
  of `TRITON_ATTN`, 0.60 of the GPU for vLLM with 256 concurrent sequences, 32k/64k-token micro-batches.
  The training forward stays SDPA without remove-padding: the pinned environment has no flash-attn
  package, and it keeps the model code path identical to the one validated on Ada.
* **New run names**: `math_grpo-h100-s1` and `math_mixed_cuts-h100-s1` (`MC_RUN_TAG=h100`), new W&B runs
  in the same project (`anlp-mixed-cuts/mixed-cuts`), separate from the Ada attempts.

## Implementation

* `configs/train/hardware/{sm75,h100}.yaml`: precision + attention moved out of `base_grpo.yaml`'s body
  into a group applied after it (a layout could not override the body). With `hardware=sm75` (default)
  every Ada config composes exactly as before, verified key by key.
* `configs/train/layout/h100_1gpu.yaml`, `configs/train/memory/plan_b_lora_gpu.yaml`.
* `configs/jarvis.env.sh`, reached from `configs/ada.env.sh` when `MC_SITE=jarvis`, so the Makefile,
  `slurm/run_train.sh` and the checks work unchanged. `MC_HW_PROFILE` (sm75 / h100) drives the
  preflight (`scripts/check_env.py`: compute capability, kernel arch, bf16, attention backend) and is
  cross-checked against the composed config's `hw_profile`.
* `jarvis/setup.sh` (driver >= 580 check, the exact Ada lock, model + data, tests, preflight),
  `jarvis/smoke.sh`, `jarvis/train.sh` (one arm on one GPU, own Ray instance, restarts, stops at the
  last checkpoint), `jarvis/start.sh`, `jarvis/status.sh`, `jarvis/eval.sh` (bf16 merge + the five
  benchmarks with `configs/eval/h100.yaml`), `scripts/compare_runs.py` (report tables, GPU-hours, W&B).

## The learning-rate bug

Composing the H100 variant showed `actor.optim.lr = 1e-6` although both LoRA plans set 1e-5:
`base_grpo.yaml`'s body (`lr: 1.0e-6`, the paper's full-fine-tune value) is applied after the memory
group and overrode it. **Every LoRA run on Ada composed with 1e-6**, contrary to 012. No result depends
on it (no arm got past step 1). Fixed: the lr lives only in the memory plans (plan A 1e-6, plan B 1e-5);
`compose_config.check` refuses a composed lr that disagrees with the plan, and a test composes every
layout/memory/hardware file and fails if any value it sets is overridden.

## Budget (estimates, not measurements)

Extrapolated from Ada: a 2080 Ti step took ~2 h 20 min (rollout 59 min, training passes 81 min). The
H100 has ~5x the memory bandwidth and ~8x the concurrent sequences for generation, and 10-18x the
compute without CPU offload for training: **~5-10 minutes per step**.

| Work, one H100 | Best | Worst |
|---|---|---|
| Setup, smoke test, first steps | 1.5 h | 2.5 h |
| Base-model evaluation | 0.5 h | 1 h |
| GRPO arm, 100 steps + validation | 8 h | 17 h |
| Mixed-CUTS arm, 100 steps + validation | 8 h | 17 h |
| Merge + evaluation of both final checkpoints | 1 h | 2 h |
| **Total GPU-hours** | **19 h** | **39.5 h** |
| On spot (INR 112.59/h) | INR 2,140 | INR 4,450 |

Check `jarvis/status.sh` after the first 3 steps: above ~10 minutes per step the worst case no longer
fits comfortably, and the number of steps (the same for both arms) becomes the user's decision.

## Not verified yet (needs the instance)

The driver on Jarvislabs' images (>= 580 needed), vLLM `FLASH_ATTN` with the CUTS logits processor at
0.60 utilisation, the real step time, the spot-preemption disk behaviour, and the GPU tests under bf16.
`jarvis/smoke.sh` covers the first two before the arms start.

## For the write-up

LoRA instead of full fine-tuning remains the deviation from the paper. The runs are bf16 on H100s;
the Ada attempts produced no results and are not part of the comparison. The compute-credit section
takes its GPU-hours from `scripts/compare_runs.py` (training) plus the setup and evaluation time.
