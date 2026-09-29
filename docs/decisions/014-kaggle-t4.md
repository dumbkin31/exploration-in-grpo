# 014: Kaggle T4 support (fp16, the Ada plan), runs carried between sessions on the Hugging Face Hub

**Date**: 2026-09-30. **Status**: implemented at the user's request ("run on Kaggle as well as Jarvis",
so a teammate can use Kaggle). Complements 013; nothing about the Jarvislabs setup changes.

## What Kaggle offers (checked 2026-09-30)

* Accelerators: **2x NVIDIA T4** (16 GB each, cc 7.5) or one P100 (cc 6.0, unsupported by CUDA 13 and
  vLLM 0.24: `kaggle/setup.sh` refuses it). TPUs cannot run this vLLM/verl stack.
* **~30 GPU-hours per account per week**, counted per session: a 2x T4 session costs the same as one GPU.
* **12-hour sessions**; the disk is gone afterwards. `/kaggle/working` (20 GB) is kept with a saved
  notebook version. Internet needs a phone-verified account.
* ~30-34 GB of host RAM per session, 4 vCPUs. NVIDIA driver 560 (CUDA 12.6) has been reported, below the
  580 the pinned CUDA 13 wheels need.

## Decision

* **Hardware profile `sm75`, the one validated on Ada**: the T4 is cc 7.5 like the 2080 Ti, so fp16,
  `TRITON_ATTN`, the SDPA patch and the fused log-prob kernel carry over unchanged.
  `layout=kaggle_t4` keeps `memory=plan_b_lora` (LoRA + FSDP2 CPU offload): on Ada a training run held
  ~18 GB anonymous + ~11 GB pinned host memory, which only fits a Kaggle session's RAM once. So there is
  **one training run per session**, and the second T4 is free for an evaluation. vLLM gets 0.80 of the
  T4 (~65k KV tokens, 48 concurrent sequences).
* **Driver fallback**: on drivers below 580, `kaggle/setup.sh` installs `torch==2.11.0+cu129`
  (download.pytorch.org) and the `vllm-0.24.0+cu129` release wheel, the same versions built for CUDA
  12.9, with every other package pinned to the Ada lock as constraints. CUDA 12.x runs on drivers >= 525
  through minor-version compatibility. The variant is recorded in `.venv/.mc_cuda_variant`, and the env
  file lowers the preflight's driver floor to match.
* **Sessions**: `scripts/hub_sync.py` pushes every new checkpoint (LoRA-only, ~0.8 GB) plus the small
  run files to a private Hugging Face model repo (`MC_HUB_REPO`, token `HF_TOKEN` with write access), and
  keeps the newest 2 there. It pulls the newest one at the start of a session. `kaggle/train.sh` stops
  the trainer cleanly (SIGUSR1, the path the resume test verified on Ada) after a logged step when the
  next step would not finish before the 12-hour limit minus 30 minutes. W&B resumes by run id.
* **Run names** `math_grpo-t4-s1` / `math_mixed_cuts-t4-s1` (`MC_RUN_TAG=t4`); base-model evaluations
  are named per platform (`base-qwen3-1.7b-t4`, `base-qwen3-1.7b-h100`).

## The catch: Kaggle cannot finish the arms in time

A T4 has about half the 2080 Ti's memory bandwidth (320 vs 616 GB/s: generation) and similar fp16
compute (training). The Ada step took ~2 h 20 min, so a T4 step is estimated at **~3-3.5 hours**. One
arm is 100 steps, **~300-350 T4-hours**, which is 10-12 weeks of one account's quota, or ~3 weeks with
all four teammates' accounts on one arm. The two arms must also train on the **same** platform: fp16 on
a T4 vs bf16 on an H100, with different micro-batching, is a confound in a comparison of methods.

So the recommended split is:
* **Both training arms on Jarvislabs** (013).
* **Kaggle for what fits the quota**:
  * **Smoke tests.** Free, and the same code paths as Ada.
  * **Evaluations.** About 3 hours per model per T4, two at once. Kaggle can evaluate the H100 runs'
    final checkpoints after `hub_sync.py push` on Jarvislabs; `jarvis/train.sh` pushes automatically
    when `MC_HUB_REPO` is set. Every model in one comparison table must be evaluated on the same
    platform.
* If Kaggle trains at all, it trains both arms with the same, smaller step count, as its own
  comparison.

## Not verified (needs a Kaggle session)

The current driver, the cu129 install, the session's real RAM limit against ~29 GB per training run,
the T4 step time, and Kaggle's handling of `kill -USR1` inside notebooks. `kaggle/smoke.sh` covers the
first two, and its memory profile shows the third.
