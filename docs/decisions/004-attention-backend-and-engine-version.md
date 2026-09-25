# 004: Attention backend and engine version on sm_75 (vLLM 0.24.0)

The brief asked to "fall back to XFORMERS or FLASHINFER" and to "confirm the V1 engine works,
else V0 changes the logits-processor interface". Neither option exists in the pinned vLLM:

| Brief | vLLM 0.24.0 source | What we do |
|---|---|---|
| XFORMERS | removed (`vllm/config/multimodal.py:231` mentions its removal); not a member of `AttentionBackendEnum` | n/a |
| FLASHINFER | `supports_compute_capability` requires cc >= 8.0 ("currently broken on SM75", `v1/attention/backends/flashinfer.py:426`) | n/a |
| FLASH_ATTN | requires cc >= 8.0 | n/a |
| **TRITON_ATTN** | `supports_compute_capability` returns True (`triton_attn.py:373`) | selected explicitly via `engine_kwargs.vllm.attention_backend: TRITON_ATTN` (`--attention-backend`; the old `VLLM_ATTENTION_BACKEND` env var no longer exists) |
| FLEX_ATTENTION | supported, lower priority | fallback if Triton fails |
| V0 engine | gone: `vllm/engine/llm_engine.py` aliases the V1 class; `VLLM_USE_V1` is an unknown variable | preflight asserts the alias |
| Model Runner V2 | default for Qwen3, but custom logits processors are unsupported and force **V1** (`vllm/config/vllm.py:2065-2070`); forcing V2 raises | never set `VLLM_USE_V2_MODEL_RUNNER`; jobs assert the `Using V2 Model Runner` log line is absent |

The processor therefore implements the **V1** `LogitsProcessor` interface
(`update_state(BatchUpdate)` / `apply(logits)`), which is applied before temperature
(`sampler.py:98` vs `:276`); this ordering is what the temperature-invariance GPU test relies on.

Logging: vLLM prints `Using TRITON_ATTN attention backend out of potential backends:
['TRITON_ATTN', 'FLEX_ATTENTION'].` through Ray to the job's stdout; `slurm/common.sh` greps it
into `env_facts.json`. The preflight evaluates every backend's `supports_compute_capability`
for the local GPU before any engine starts.

bf16 is rejected by vLLM below cc 8.0 (`platforms/cuda.py:610`), so every dtype is `float16`.
