#!/usr/bin/env python
"""Compose a training config exactly as `mixed_cuts.main` would, and print or check it.

Catches YAML typos, broken interpolations and wrong key paths *before* burning a GPU
allocation. Works without verl installed by pointing ``--verl-config-dir`` at a verl checkout
(``verl/trainer/config``); on the cluster it defaults to the installed package.

    python scripts/compose_config.py smoke                      # dump the composed YAML
    python scripts/compose_config.py smoke --check              # assert the invariants below
    python scripts/compose_config.py math_mixed_cuts --verl-config-dir /path/to/verl/trainer/config

Invariants checked with --check (see the [checked] marks in configs/train/base_grpo.yaml):
  * fp16 only: no 'bfloat16'/'bf16' anywhere, sdpa attention, no remove-padding (sm_75)
  * D4: n_std + n_cuts == group_size == rollout.n; CUTS K/delta/T_warm = paper values
  * D1: rollout_correction.rollout_is/rollout_rs null, bypass_mode false, calculate_log_probs false
  * D2: rollout T=1.0/top_p=1.0, validation 1.0/0.8/20, non-thinking, KL low_var_kl 1e-3
  * Task A: TP=4 on one node, TRITON_ATTN, sleep mode, gradient checkpointing, dynamic bsz with
    token budgets >= max_prompt_length + max_response_length, micro-batch keys null
  * one seed interpolated into data/rollout/actor/ref; resume_mode auto; outputs under paths.run_dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
TRAIN_CONFIG_DIR = REPO / "configs" / "train"


def compose_train_config(name: str, verl_config_dir: str | None, overrides: list[str]):
    searchpath = [f"file://{verl_config_dir}"] if verl_config_dir else ["pkg://verl.trainer.config"]
    fake_paths = [
        "paths.run_dir=/tmp/mc-compose/run",
        "paths.model_dir=/tmp/mc-compose/model",
        "paths.data_dir=/tmp/mc-compose/data",
        "paths.repo_dir=" + str(REPO),
        "run_name=compose-check-s42",
    ]
    with initialize_config_dir(config_dir=str(TRAIN_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=name,
            overrides=[f"hydra.searchpath=[{','.join(searchpath)}]", *fake_paths, *overrides],
        )
    return cfg


def _walk(node, path=""):
    if OmegaConf.is_dict(node):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif OmegaConf.is_list(node):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def check(cfg) -> list[str]:
    """Invariants that keep the two arms runnable on 4x 2080 Ti and faithful to the brief."""
    problems: list[str] = []
    arr = cfg.actor_rollout_ref

    def fail(msg: str) -> None:
        problems.append(msg)

    # --- fp16 only (sm_75) ---
    for path, value in _walk(arr, "actor_rollout_ref"):
        if isinstance(value, str) and value.lower() in ("bf16", "bfloat16"):
            fail(f"{path} = {value} (bf16 is unsupported on sm_75)")
    if arr.rollout.dtype != "float16":
        fail(f"rollout.dtype = {arr.rollout.dtype}")
    for role in ("actor", "ref"):
        mp = arr[role].fsdp_config.get("mixed_precision") or {}
        if mp.get("param_dtype") != "fp16":
            fail(f"{role}.fsdp_config.mixed_precision.param_dtype must be fp16")
    if arr.model.get("override_config", {}).get("attn_implementation") != "sdpa":
        fail("model.override_config.attn_implementation must be sdpa (no FlashAttention-2 on sm_75)")
    if arr.model.use_remove_padding:
        fail("model.use_remove_padding must be false (needs FA2 varlen kernels)")

    # --- D4: matched generation budget ---
    mc = cfg.get("mixed_cuts")
    if mc is None:
        fail("mixed_cuts block missing")
    else:
        g = mc.get("group_size")
        if g is None:
            fail("mixed_cuts.group_size must be set explicitly")
        elif mc.enabled and (mc.n_std + mc.n_cuts) != g:
            fail(f"budget mismatch: {mc.n_std} + {mc.n_cuts} != G={g}")
        if g is not None and arr.rollout.n != g:
            fail(f"rollout.n = {arr.rollout.n} != mixed_cuts.group_size = {g}")
        lps = list(arr.rollout.engine_kwargs.vllm.get("logits_processors", []) or [])
        if mc.enabled and mc.n_cuts > 0 and "cuts.vllm_logits_processor:CutsLogitsProcessor" not in lps:
            fail(f"CUTS enabled but logits processor not registered: {lps}")
        if mc.cuts.t_warm != 5 or mc.cuts.k != 5 or abs(mc.cuts.delta - 0.03) > 1e-12:
            fail(f"CUTS hyperparameters differ from the paper (K=5, delta=0.03, T_warm=5): {dict(mc.cuts)}")

    # --- D1: no importance correction; actor recomputes old_log_probs ---
    rc = cfg.algorithm.get("rollout_correction")
    if rc is None:
        fail("algorithm.rollout_correction block missing")
    else:
        if rc.get("rollout_is") is not None or rc.get("rollout_rs") is not None:
            fail(f"rollout_correction.rollout_is/rollout_rs must be null (D1): {dict(rc)}")
        if rc.get("bypass_mode"):
            fail("rollout_correction.bypass_mode must be false: the actor must recompute old_log_probs (D1)")
    if arr.rollout.calculate_log_probs:
        fail("rollout.calculate_log_probs must be false (D1: engine logprobs for CUTS rows are log(1/|S_t|))")

    # --- D2: decoding / non-thinking ---
    if arr.rollout.temperature != 1.0 or arr.rollout.top_p != 1.0:
        fail(
            f"rollout temperature/top_p must be 1.0/1.0 (paper): {arr.rollout.temperature}/{arr.rollout.top_p}"
        )
    vk = arr.rollout.val_kwargs
    if (vk.temperature, vk.top_p, vk.top_k) != (1.0, 0.8, 20):
        fail(f"val_kwargs must be T=1.0/top_p=0.8/top_k=20 (paper): {vk.temperature}/{vk.top_p}/{vk.top_k}")
    if cfg.data.get("apply_chat_template_kwargs", {}).get("enable_thinking") is not False:
        fail("data.apply_chat_template_kwargs.enable_thinking must be false (Qwen3 non-thinking)")
    if cfg.actor_rollout_ref.actor.use_kl_loss is not True or abs(arr.actor.kl_loss_coef - 0.001) > 1e-12:
        fail("actor.use_kl_loss must be true with kl_loss_coef 1e-3 (paper)")
    if arr.actor.kl_loss_type != "low_var_kl":
        fail(f"actor.kl_loss_type must be low_var_kl: {arr.actor.kl_loss_type}")

    # --- Task A: 4-GPU layout ---
    if arr.rollout.tensor_model_parallel_size != 4:
        fail(f"rollout.tensor_model_parallel_size must be 4: {arr.rollout.tensor_model_parallel_size}")
    if cfg.trainer.n_gpus_per_node != 4 or cfg.trainer.nnodes != 1:
        fail("trainer.n_gpus_per_node must be 4 and nnodes 1")
    if arr.rollout.engine_kwargs.vllm.get("attention_backend") != "TRITON_ATTN":
        fail(
            "rollout.engine_kwargs.vllm.attention_backend must be TRITON_ATTN (the only V1 backend for sm_75)"
        )
    if not arr.rollout.free_cache_engine:
        fail("rollout.free_cache_engine must be true (vLLM must sleep during the update on 11 GiB)")
    if not arr.model.enable_gradient_checkpointing:
        fail("model.enable_gradient_checkpointing must be true")
    if (
        not arr.actor.use_dynamic_bsz
        or not arr.rollout.log_prob_use_dynamic_bsz
        or not arr.ref.log_prob_use_dynamic_bsz
    ):
        fail("use_dynamic_bsz must be true for actor, rollout log-prob and ref")
    longest = cfg.data.max_prompt_length + cfg.data.max_response_length
    for path, value in (
        ("actor.ppo_max_token_len_per_gpu", arr.actor.ppo_max_token_len_per_gpu),
        ("rollout.log_prob_max_token_len_per_gpu", arr.rollout.log_prob_max_token_len_per_gpu),
        ("ref.log_prob_max_token_len_per_gpu", arr.ref.log_prob_max_token_len_per_gpu),
        ("rollout.max_model_len", arr.rollout.max_model_len),
    ):
        if value is None or int(value) < longest:
            fail(f"{path} = {value} < max_prompt_length + max_response_length = {longest}")
    for path, value in (
        ("actor.ppo_micro_batch_size_per_gpu", arr.actor.ppo_micro_batch_size_per_gpu),
        ("rollout.log_prob_micro_batch_size_per_gpu", arr.rollout.log_prob_micro_batch_size_per_gpu),
        ("ref.log_prob_micro_batch_size_per_gpu", arr.ref.log_prob_micro_batch_size_per_gpu),
    ):
        if value is not None:
            fail(f"{path} must be null in dynamic-bsz mode: {value}")
    mgr = arr.rollout.agent.get("agent_loop_manager_class")
    if mgr != "mixed_cuts.agent_loop.MixedCutsAgentLoopManagerTQ":
        fail(f"rollout.agent.agent_loop_manager_class = {mgr}")

    # --- seeds: one knob ---
    seed = cfg.get("seed")
    if seed is None:
        fail("top-level seed missing")
    else:
        for path, value in (
            ("data.seed", cfg.data.seed),
            ("rollout.seed", arr.rollout.seed),
            ("actor.data_loader_seed", arr.actor.data_loader_seed),
            ("actor.fsdp_config.seed", arr.actor.fsdp_config.seed),
            ("ref.fsdp_config.seed", arr.ref.fsdp_config.seed),
        ):
            if value != seed:
                fail(f"{path} = {value} != seed = {seed} (must interpolate ${{seed}})")

    # --- resume + durable outputs (Task C) ---
    if cfg.trainer.get("resume_mode") != "auto":
        fail(f"trainer.resume_mode must be auto: {cfg.trainer.get('resume_mode')}")
    run_dir = str(cfg.paths.run_dir)
    if not str(cfg.trainer.default_local_dir).startswith(run_dir):
        fail(
            f"trainer.default_local_dir must be under paths.run_dir (on /share1): {cfg.trainer.default_local_dir}"
        )
    if str(cfg.trainer.experiment_name) != str(cfg.run_name):
        fail("trainer.experiment_name must equal run_name (stable across resubmissions)")
    if not cfg.trainer.use_v1 or cfg.trainer.v1.trainer_mode != "mixed_cuts_sync":
        fail(f"trainer.use_v1={cfg.trainer.use_v1} trainer_mode={cfg.trainer.v1.trainer_mode}")
    if (
        cfg.reward.custom_reward_function.get("path") in (None, "")
        or cfg.reward.custom_reward_function.name != "compute_score"
    ):
        fail("reward.custom_reward_function must point at mixed_cuts.reward:compute_score")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="name under configs/train (without .yaml)")
    ap.add_argument(
        "--verl-config-dir", default=None, help="verl/trainer/config of a checkout (default: installed verl)"
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--select", default=None, help="only print this subtree, e.g. actor_rollout_ref.rollout")
    ap.add_argument("overrides", nargs="*", help="extra hydra overrides")
    args = ap.parse_args()
    cfg = compose_train_config(args.config, args.verl_config_dir, args.overrides)
    if args.check:
        problems = check(cfg)
        for p in problems:
            print("FAIL ", p)
        print(f"== {args.config}: {'OK' if not problems else str(len(problems)) + ' problem(s)'} ==")
        return 1 if problems else 0
    node = OmegaConf.select(cfg, args.select) if args.select else cfg
    print(OmegaConf.to_yaml(node, resolve=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
