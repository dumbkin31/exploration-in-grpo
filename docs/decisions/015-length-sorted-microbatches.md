# 015: Length-sorted micro-batches on the padded attention path

Date: 2026-10-01. Status: accepted (user-approved patch); deployed to both H200 arms mid-run.

## Problem

The first H200 steps took far longer than planned. Mixed-CUTS step 1 (`math_mixed_cuts-h200-s1`):

| Phase | Seconds |
|---|---|
| Generation (vLLM) | 113 |
| Old log-probs | 235 |
| Reference log-probs | 216 |
| Actor update | 498 |
| Whole step | 1,066 |

MFU was 0.06. At 17.8 minutes a step, the ₹5,880 credit covers about 40 of the 100 steps per arm.

## Cause

There is no flash-attn package for this torch build (2.11.0+cu130; the instance's nvcc is 12.6, so it
cannot be built either), so the actor runs with `use_remove_padding: false` and
`FSDPEngine.prepare_model_inputs` pads every micro-batch to its longest sequence
(verl/workers/engine/fsdp/transformer_impl.py:1259). verl v0.9.0 forms dynamic micro-batches in
`seqlen_balancing.rearrange_micro_batches` by balancing attention workload across them
(Karmarkar-Karp on `24576 * len + len**2`), so each one mixes long and short sequences. At the measured
length spread (prompt mean 113, response mean 918, std 903, 0.9% clipped at 5,000), a simulation gives
a padding efficiency of about 0.25 at the configured budgets (32,768 update, 65,536 log-prob): three
quarters of the training compute, and of the logits memory, is spent on pad tokens.

## Decision

`mixed_cuts.microbatch_patch` wraps `verl.workers.engine.utils.rearrange_micro_batches`: sort by length,
cut runs that fit the budget once padded (`count * longest <= max_token_len`), longest run first. The
simulated padding efficiency becomes about 0.92 for the update (512 sequences per mini-batch) and 0.95
for the log-prob passes (2,048 sequences), the micro-batch count grows by about one in eight, and every
micro-batch's padded size is below what verl gave it, so peak memory falls.

The gradient does not change: `forward_backward_batch` computes `batch_num_tokens` over the whole
mini-batch before splitting (transformer_impl.py:706) and `agg_loss` divides every micro-batch's token
sum by it. Outputs are restored to the original order by verl's `restore_dynamic_batch` from the
returned index lists. Only the floating-point summation order differs.

The patch hands the call back to verl unchanged on the remove-padding path, with more than one
data-parallel rank, grouped samples, a pipeline divisor or a forced minimum count, or a sequence longer
than the budget. `MC_SORTED_MICROBATCHES=0` disables it. A guard test fails if a verl upgrade moves the
call site, changes the signature, or stops normalising the loss over the whole mini-batch.

## Deployment and the report

Both arms were stopped right after a step's checkpoint (`save_freq: 1`) and resumed with the patch, so
no step was lost and both arms received the same change. Steps before the switch ran on verl's
partition; the training math is the same on both sides of it, and the report says so in one line.
The measured step time after the switch decides the common step cap for both arms.
