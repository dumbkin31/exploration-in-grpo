#!/usr/bin/env python
"""Preflight check for the Mixed-CUTS pipeline on Ada.

Runs at the top of every SLURM job (and via `make check-env`). It fails FAST with a
readable message instead of letting a job die 40 minutes in. Every check prints one of

    PASS  <what>
    WARN  <what>: <why>          (does not fail the run)
    FAIL  <what>: <why>          (exit code 1 at the end)

Checks are grouped: python packages and pins, GPU (must be RTX 2080 Ti class, sm_75, with
sm_75 kernels present in the torch wheel), staged model/data paths, storage mounts, and
internet reachability. Storage and internet results are also written as a small JSON file
so the sbatch scripts can branch on them (e.g. HF_HUB_OFFLINE).

Usage:
    python scripts/check_env.py                 # everything
    python scripts/check_env.py --no-gpu        # login node / laptop
    python scripts/check_env.py --no-staged     # right after `make setup`, before prefetch
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The repo's src/ must be importable even when the package is not installed (login node, fresh venv).
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

# Expected pins: keep in sync with requirements/base.txt and the README table.
EXPECTED_VERSIONS = {
    "vllm": "0.24.0",
    "torch": "2.11.0",
    "transformers": "5.5.3",
    "ray": "2.56.1",
    "tensordict": "0.10.0",
    "verl": "0.9.0",
    "math_verify": "0.9.0",
}
# Import-only (no version pin enforced here).
EXPECTED_IMPORTS = ["transfer_queue", "hydra", "omegaconf", "datasets", "wandb", "cuts", "mixed_cuts"]

REQUIRED_CC = (7, 5)  # RTX 2080 Ti. The 1080 Ti nodes (6,1) are unsupported by CUDA 13 and vLLM.


class Report:
    def __init__(self) -> None:
        self.failed = False
        self.facts: dict[str, object] = {}

    def ok(self, what: str) -> None:
        print(f"PASS  {what}")

    def warn(self, what: str, why: str) -> None:
        print(f"WARN  {what}: {why}")

    def fail(self, what: str, why: str) -> None:
        self.failed = True
        print(f"FAIL  {what}: {why}")


def _version_of(mod_name: str) -> str | None:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:  # noqa: BLE001 - we want to report any import error
        return f"<import error: {type(e).__name__}: {e}>"
    return getattr(mod, "__version__", "<no __version__>")


def check_python(r: Report) -> None:
    v = sys.version_info
    if v >= (3, 10):
        r.ok(f"python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    else:
        r.fail("python", f"{v.major}.{v.minor} < 3.10 (verl requires >= 3.10, we target 3.12)")
    if "site-packages" in sys.executable or "/.venv/" in sys.executable or os.environ.get("VIRTUAL_ENV"):
        r.ok(f"virtualenv active: {os.environ.get('VIRTUAL_ENV', Path(sys.executable).parent.parent)}")
    else:
        r.warn("virtualenv", "no VIRTUAL_ENV set; are you using the project venv?")


def check_pins(r: Report) -> None:
    for name, want in EXPECTED_VERSIONS.items():
        got = _version_of(name)
        if got is None or (got and got.startswith("<import error")):
            r.fail(f"import {name}", got or "not importable")
        elif got.split("+")[0] == want:
            r.ok(f"{name}=={got}")
        else:
            r.fail(f"{name} version", f"installed {got}, expected {want} (see requirements/base.txt)")
    for name in EXPECTED_IMPORTS:
        got = _version_of(name)
        if got and got.startswith("<import error"):
            r.fail(f"import {name}", got)
        else:
            r.ok(f"import {name} ({got})")


def check_gpu(r: Report) -> None:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        r.fail("torch import", str(e))
        return
    if not torch.cuda.is_available():
        r.fail("cuda", "torch.cuda.is_available() is False (no GPU allocated, or driver/runtime mismatch)")
        return
    cuda_rt = torch.version.cuda or "?"
    if cuda_rt.startswith("13."):
        r.ok(f"torch CUDA runtime {cuda_rt} (default cu130 wheel)")
    else:
        r.warn("torch CUDA runtime", f"{cuda_rt}; the pinned default wheel is CUDA 13.0")
    arch_list = torch.cuda.get_arch_list()
    if "sm_75" in arch_list:
        r.ok(f"sm_75 kernels present in torch wheel: {arch_list}")
    else:
        r.fail("torch arch list", f"sm_75 missing from {arch_list}; this wheel cannot run on a 2080 Ti")
    n = torch.cuda.device_count()
    r.facts["gpu_count"] = n
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 2**30
        if cc == REQUIRED_CC:
            r.ok(f"gpu{i}: {name}, cc {cc[0]}.{cc[1]}, {mem_gb:.1f} GiB")
        elif cc < (7, 0):
            r.fail(
                f"gpu{i}",
                f"{name} cc {cc} is below 7.0: unsupported by vLLM and CUDA 13 (did the job land on a 1080 Ti node? use -C 2080ti)",
            )
        else:
            r.warn(f"gpu{i}", f"{name} cc {cc}; configs are tuned for the 2080 Ti (7.5, 11 GiB)")
    if n and torch.cuda.is_bf16_supported():
        r.warn("bf16", "reported as supported; configs still force fp16 for reproducibility across nodes")
    else:
        r.ok("bf16 unsupported on this GPU -> all configs use float16 (as required)")
    # Which attention backend vLLM will pick (mirrors vllm/platforms/cuda.py priority + support checks).
    try:
        from vllm.platforms import current_platform  # type: ignore

        if current_platform.has_device_capability(80):
            r.warn("vllm attention", "cc>=8.0 -> FLASH_ATTN; the plan was validated for TRITON_ATTN on sm_75")
        else:
            r.ok("vllm attention backend on cc<8.0 -> TRITON_ATTN (FLASH_ATTN/FLASHINFER need sm_80)")
    except Exception as e:  # noqa: BLE001
        r.warn("vllm attention backend probe", f"{type(e).__name__}: {e}")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") not in (None, "", "0"):
        r.fail("VLLM_USE_V2_MODEL_RUNNER", "must be unset: custom logits processors need Model Runner V1")
    else:
        r.ok("VLLM_USE_V2_MODEL_RUNNER unset (Model Runner V1 will be used for the CUTS logits processor)")


def check_staged(r: Report) -> None:
    model_dir = os.environ.get("MC_MODEL_DIR")
    data_dir = os.environ.get("MC_DATA_DIR")
    if not model_dir or not data_dir:
        r.fail("staging env", "MC_MODEL_DIR / MC_DATA_DIR unset; `source configs/ada.env.sh` first")
        return
    cfg = Path(model_dir) / "config.json"
    if cfg.exists():
        r.ok(f"model staged at {model_dir}")
    else:
        r.fail("model staged", f"{cfg} missing; run `make prefetch` on the login node")
    manifest = Path(data_dir) / "MANIFEST.json"
    if manifest.exists():
        r.ok(f"data staged at {data_dir}")
    else:
        r.fail("data staged", f"{manifest} missing; run `make prefetch` then `make data`")


def check_storage(r: Report) -> None:
    for var in ("MC_STAGE_ROOT", "MC_SCRATCH_ROOT", "MC_CACHE_ROOT"):
        p = os.environ.get(var)
        if not p:
            r.warn(var, "unset; `source configs/ada.env.sh` first")
            continue
        root = Path(p)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".mc_write_probe"
            probe.write_text("ok")
            probe.unlink()
            free_gb = shutil.disk_usage(root).free / 2**30
            r.ok(f"{var}={p} writable, {free_gb:.0f} GiB free")
            r.facts[var] = {"path": p, "writable": True, "free_gib": round(free_gb)}
        except OSError as e:
            r.fail(var, f"{p} not writable here ({e}); outputs and checkpoints must land on /share1")
            r.facts[var] = {"path": p, "writable": False}
    home = Path.home()
    try:
        used = subprocess.run(
            ["du", "-sh", str(home)], capture_output=True, text=True, timeout=60
        ).stdout.split()[0]
        r.ok(f"home usage {used} (quota 25 GB; code + venv only)")
    except Exception:  # noqa: BLE001
        pass


def check_internet(r: Report) -> None:
    proxies = {k: v for k, v in os.environ.items() if k.lower().endswith("_proxy")}
    if proxies:
        r.ok(f"proxy env present: {sorted(proxies)}")
    reach: dict[str, bool] = {}
    for host in ("https://huggingface.co", "https://api.wandb.ai"):
        try:
            urllib.request.urlopen(host, timeout=5)  # noqa: S310
            reach[host] = True
            r.ok(f"reachable {host}")
        except urllib.error.HTTPError as e:
            # Any HTTP status (403, 404, ...) means the TCP/TLS path works; only transport errors count.
            reach[host] = True
            r.ok(f"reachable {host} (HTTP {e.code})")
        except Exception as e:  # noqa: BLE001
            reach[host] = False
            r.warn(f"unreachable {host}", f"{type(e).__name__}; jobs will run offline from staged copies")
    r.facts["internet"] = reach


def check_slurm(r: Report) -> None:
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        r.warn("slurm", "not inside an allocation (fine on the login node / laptop)")
        return
    r.ok(
        f"slurm job {job} on {os.environ.get('SLURMD_NODENAME', '?')}, gres={os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
    )
    part = os.environ.get("SLURM_JOB_PARTITION", "")
    try:
        out = subprocess.run(
            ["scontrol", "show", "partition", part], capture_output=True, text=True, timeout=20
        ).stdout
        for tok in out.split():
            if tok.startswith("MaxMemPerCPU=") or tok.startswith("DefMemPerCPU="):
                r.ok(f"partition {part}: {tok}")
    except Exception:  # noqa: BLE001
        pass


def check_single_node(r: Report) -> None:
    """The 4 GPUs must be on one node: rollout TP=4 and FSDP over 4 ranks assume one NCCL host."""
    n = os.environ.get("SLURM_JOB_NUM_NODES")
    if n is None:
        r.warn("single node", "SLURM_JOB_NUM_NODES unset (not inside a SLURM job)")
    elif int(n) == 1:
        r.ok("single node allocation (SLURM_JOB_NUM_NODES=1)")
    else:
        r.fail("single node", f"SLURM_JOB_NUM_NODES={n}; add '#SBATCH -N 1' (the configs assume one node)")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        n_vis = len([x for x in vis.split(",") if x.strip()])
        (r.ok if n_vis == 4 else r.warn)(
            f"visible GPUs: {n_vis} ({vis})", "training configs expect 4 (rollout TP=4)"
        )


def check_vllm_engine(r: Report) -> None:
    """vLLM 0.24.0: the V0 engine is gone; the risk is Model Runner V2, which custom logits processors force back to V1."""
    try:
        import vllm.engine.llm_engine as legacy
        import vllm.v1.engine.llm_engine as v1
    except Exception as e:  # noqa: BLE001
        r.fail("vllm engine import", f"{type(e).__name__}: {e}")
        return
    if getattr(legacy, "LLMEngine", None) is v1.LLMEngine:
        r.ok("vLLM V1 engine (vllm.engine.llm_engine.LLMEngine is the V1 class; no V0 engine exists)")
    else:
        r.fail(
            "vLLM engine version",
            "vllm.engine.llm_engine.LLMEngine is not the V1 class; the logits-processor interface would differ",
        )
    for var in ("VLLM_USE_V1", "VLLM_USE_V2_MODEL_RUNNER"):
        if os.environ.get(var) not in (None, ""):
            r.fail(
                var,
                f"must be unset (set to {os.environ[var]!r}); custom logits processors need Model Runner V1",
            )
    try:
        from vllm.config.vllm import VllmConfig
        from vllm.v1.sample.logits_processor import LogitsProcessor

        from cuts.vllm_logits_processor import CutsLogitsProcessor

        if not issubclass(CutsLogitsProcessor, LogitsProcessor):
            r.fail("CutsLogitsProcessor", "does not subclass vllm.v1.sample.logits_processor.LogitsProcessor")
        elif not hasattr(VllmConfig, "_get_v2_model_runner_unsupported_features"):
            r.fail(
                "Model Runner V2 fallback",
                "VllmConfig._get_v2_model_runner_unsupported_features missing; cannot rely on the V1 fallback",
            )
        else:
            r.ok(
                "CutsLogitsProcessor implements the V1 interface; vLLM falls back to Model Runner V1 for custom processors"
            )
    except Exception as e:  # noqa: BLE001
        r.fail("logits processor interface", f"{type(e).__name__}: {e}")


def check_attention_backend(r: Report) -> None:
    """Which V1 attention backend vLLM will pick for this GPU (expected TRITON_ATTN on sm_75)."""
    try:
        import torch
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except Exception as e:  # noqa: BLE001
        r.warn("attention backend probe", f"{type(e).__name__}: {e}")
        return
    if not torch.cuda.is_available():
        return
    cc = DeviceCapability(*torch.cuda.get_device_capability(0))
    supported = []
    for name in ("FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"):
        try:
            ok = bool(AttentionBackendEnum[name].get_class().supports_compute_capability(cc))
        except Exception as e:  # noqa: BLE001
            ok = False
            r.warn(f"backend {name}", f"probe failed: {type(e).__name__}")
        (supported if ok else []).append(name)
    r.facts["attention_backends_supporting_cc"] = supported
    if supported and supported[0] == "TRITON_ATTN":
        r.ok(
            f"attention backend for cc {cc.major}.{cc.minor}: TRITON_ATTN (first in priority among {supported})"
        )
    elif supported:
        r.warn("attention backend", f"first supported backend is {supported[0]}, configs pin TRITON_ATTN")
    else:
        r.fail("attention backend", f"no V1 attention backend supports cc {cc.major}.{cc.minor}")


def check_chat_template(r: Report, model_dir: str | None) -> None:
    """Tokenizer-only proof that enable_thinking=False is honoured and the system prompt is rendered."""
    model_dir = model_dir or os.environ.get("MC_MODEL_DIR")
    if not model_dir or not Path(model_dir, "tokenizer_config.json").exists():
        r.warn("chat template", f"tokenizer not found under {model_dir}; skipped")
        return
    try:
        from transformers import AutoTokenizer

        from mc_data.schema import SYSTEM_PROMPT, build_messages
        from mixed_cuts.thinking import EMPTY_THINK_BLOCK, think_token_ids

        tok = AutoTokenizer.from_pretrained(model_dir)
        msgs = build_messages("What is 1+1?")
        off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except Exception as e:  # noqa: BLE001
        r.fail("chat template", f"{type(e).__name__}: {e}")
        return
    if off.endswith(EMPTY_THINK_BLOCK):
        r.ok("enable_thinking=False renders the empty <think></think> block at the end of the prompt")
    else:
        r.fail(
            "chat template",
            f"enable_thinking=False prompt does not end with the empty think block: ...{off[-60:]!r}",
        )
    if on.endswith(EMPTY_THINK_BLOCK):
        r.fail(
            "chat template",
            "enable_thinking=True renders the same tail: the kwarg is NOT honoured by this template",
        )
    else:
        r.ok("enable_thinking=True differs from False (the kwarg is honoured)")
    if SYSTEM_PROMPT in off:
        r.ok("system prompt rendered verbatim")
    else:
        r.fail("chat template", "system prompt missing from the rendered prompt")
    try:
        think_token_ids(tok)
        r.ok("<think>/</think> are single tokens (the response check works on ids)")
    except ValueError as e:
        r.fail("think tokens", str(e))


def check_config(r: Report, name: str | None) -> None:
    """Compose the config about to be launched and run the compose-check invariants."""
    if not name:
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from compose_config import check, compose_train_config

        # On the cluster verl's config tree comes from the installed package; on a laptop point
        # MC_VERL_CONFIG_DIR at a checkout's verl/trainer/config.
        cfg = compose_train_config(name, os.environ.get("MC_VERL_CONFIG_DIR") or None, [])
        problems = check(cfg)
    except Exception as e:  # noqa: BLE001
        r.fail(f"config {name}", f"does not compose: {type(e).__name__}: {e}")
        return
    for p in problems:
        r.fail(f"config {name}", p)
    if not problems:
        r.ok(
            f"config {name} composes and passes every invariant (fp16, D1, D2, D4, 4-GPU layout, seeds, resume)"
        )


def check_durable_dir(r: Report) -> None:
    run_dir = os.environ.get("MC_RUN_DIR") or os.environ.get("MC_RUNS_DIR")
    if not run_dir:
        return
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        probe = Path(run_dir) / ".mc_write_probe"
        probe.write_text("ok")
        probe.unlink()
        r.ok(f"durable run dir writable: {run_dir}")
    except OSError as e:
        r.fail("durable run dir", f"{run_dir} not writable ({e}); checkpoints must land on /share1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gpu", action="store_true", help="skip GPU checks (login node / laptop)")
    ap.add_argument("--no-staged", action="store_true", help="skip staged model/data checks")
    ap.add_argument("--no-pins", action="store_true", help="skip package pin checks (dev env)")
    ap.add_argument("--config", default=None, help="training config name to compose and check (e.g. smoke)")
    ap.add_argument(
        "--model-dir", default=None, help="tokenizer dir for the chat-template check (default $MC_MODEL_DIR)"
    )
    ap.add_argument("--only-config", action="store_true", help="run only the config + chat-template checks")
    ap.add_argument(
        "--facts-json", default=os.environ.get("MC_ENV_FACTS_JSON"), help="write detected facts here"
    )
    args = ap.parse_args()

    r = Report()
    if args.only_config:
        print("== config ==")
        check_config(r, args.config)
        print("== chat template ==")
        check_chat_template(r, args.model_dir)
        print("== RESULT:", "FAIL (see above)" if r.failed else "OK", "==")
        return 1 if r.failed else 0
    print("== python ==")
    check_python(r)
    if not args.no_pins:
        print("== pins ==")
        check_pins(r)
    if not args.no_gpu:
        print("== gpu ==")
        check_gpu(r)
        print("== vllm engine ==")
        check_vllm_engine(r)
        check_attention_backend(r)
    print("== single node ==")
    check_single_node(r)
    print("== storage ==")
    check_storage(r)
    check_durable_dir(r)
    if args.config:
        print("== config ==")
        check_config(r, args.config)
    print("== chat template ==")
    check_chat_template(r, args.model_dir)
    if not args.no_staged:
        print("== staged model/data ==")
        check_staged(r)
    print("== network ==")
    check_internet(r)
    print("== slurm ==")
    check_slurm(r)

    r.facts["slurm_job_num_nodes"] = os.environ.get("SLURM_JOB_NUM_NODES")
    if args.facts_json:
        Path(args.facts_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.facts_json).write_text(json.dumps(r.facts, indent=2))
        print(f"facts written to {args.facts_json}")

    print("== RESULT:", "FAIL (see above)" if r.failed else "OK", "==")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
