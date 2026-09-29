#!/usr/bin/env python
"""Preflight check for the Mixed-CUTS pipeline (Ada 2080 Ti, or one H100 on Jarvislabs).

Runs at the top of every SLURM job (and via `make check-env`). It fails FAST with a
readable message instead of letting a job die 40 minutes in. Every check prints one of

    PASS  <what>
    WARN  <what>: <why>          (does not fail the run)
    FAIL  <what>: <why>          (exit code 1 at the end)

Checks are grouped: python packages and pins, GPU (must match the hardware profile MC_HW_PROFILE:
sm75 = RTX 2080 Ti with sm_75 kernels in the torch wheel, the default; h100 = cc 9.0 with sm_90 kernels
and bf16; decision 013), staged model/data paths, storage mounts, and
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
import importlib.metadata
import json
import os
import re
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

# Hardware profiles (configs/train/hardware/, chosen by MC_HW_PROFILE from the env file).
PROFILES = {
    # RTX 2080 Ti. The 1080 Ti nodes (6,1) are unsupported by CUDA 13 and vLLM.
    "sm75": {
        "cc": (7, 5),
        "arch": "sm_75",
        "gpu": "RTX 2080 Ti (7.5, 11 GiB)",
        "bf16": False,
        "backend": "TRITON_ATTN",
    },
    "h100": {
        "cc": (9, 0),
        "arch": "sm_90",
        "gpu": "H100 (9.0, 80 GB)",
        "bf16": True,
        "backend": "FLASH_ATTN",
    },
}


def hw_profile() -> tuple[str, dict]:
    name = os.environ.get("MC_HW_PROFILE", "sm75")
    return name, PROFILES.get(name, PROFILES["sm75"])


REQUIRED_CC = PROFILES["sm75"]["cc"]  # kept for callers of the old constant


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
    v = getattr(mod, "__version__", None)
    if v is None:  # e.g. math_verify exposes no __version__; ask the installed distribution
        try:
            v = importlib.metadata.version(mod_name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            return "<no __version__>"
    return v


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


def check_driver(r: Report) -> None:
    """The pinned wheels are CUDA 13.0 builds: driver >= 580 (mixed generations on Ada, decision 011)."""
    need = int(os.environ.get("MC_MIN_DRIVER_MAJOR", "580"))
    drv = None
    try:
        text = Path("/proc/driver/nvidia/version").read_text()
        m = re.search(r"Kernel Module\s+([0-9.]+)", text)
        drv = m.group(1) if m else None
    except OSError:
        pass
    if drv is None:  # containers (Jarvislabs) may not expose /proc/driver/nvidia
        try:
            drv = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split()[0]
        except (OSError, IndexError, subprocess.SubprocessError):
            r.fail(
                "nvidia driver", "neither /proc/driver/nvidia/version nor nvidia-smi: no NVIDIA driver here"
            )
            r.facts["nvidia_driver"] = None
            return
    r.facts["nvidia_driver"] = drv
    major = int(drv.split(".")[0]) if drv[:1].isdigit() else 0
    if major >= need:
        r.ok(f"nvidia driver {drv} (>= {need}, runs the cu130 wheels)")
    else:
        r.fail(
            "nvidia driver",
            f"{drv} < {need}: the cu130 wheels cannot initialise CUDA here; exclude this node (decision 011)",
        )


def _check_sdpa_kernels(r: Report, torch) -> None:
    """Turing: the memory-efficient SDPA kernel exists only without enable_gqa (decision 012)."""
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as e:  # noqa: BLE001
        r.warn("sdpa probe", f"{type(e).__name__}: {e}")
        return
    dev = "cuda"
    q = torch.randn(1, 16, 256, 128, device=dev, dtype=torch.float16)
    k = torch.randn(1, 16, 256, 128, device=dev, dtype=torch.float16)
    k8 = torch.randn(1, 8, 256, 128, device=dev, dtype=torch.float16)
    mask = torch.ones(1, 1, 256, 256, device=dev, dtype=torch.bool).tril()
    facts = {}
    for name, kv, kw in (
        ("mem_efficient_repeated_kv", k, {"attn_mask": mask}),
        ("mem_efficient_gqa", k8, {"is_causal": True, "enable_gqa": True}),
    ):
        try:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                F.scaled_dot_product_attention(q, kv, kv, **kw)
            facts[name] = True
        except Exception:  # noqa: BLE001
            facts[name] = False
    r.facts["sdpa"] = facts
    if facts["mem_efficient_repeated_kv"]:
        r.ok(
            f"SDPA memory-efficient kernel available with repeated KV heads (enable_gqa: {facts['mem_efficient_gqa']}); the worker hook keeps transformers on that path"
        )
    else:
        r.fail(
            "sdpa",
            "no memory-efficient SDPA kernel on this GPU even with repeated KV heads: attention would run on the math kernel (+8.8 GiB per call at 6k tokens)",
        )


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
    prof_name, prof = hw_profile()
    r.facts["hw_profile"] = prof_name
    arch_list = torch.cuda.get_arch_list()
    if prof["arch"] in arch_list:
        r.ok(f"{prof['arch']} kernels present in torch wheel: {arch_list}")
    else:
        r.fail(
            "torch arch list",
            f"{prof['arch']} missing from {arch_list}; this wheel cannot run on a {prof['gpu']}",
        )
    try:
        torch.cuda.init()
    except RuntimeError as e:  # e.g. "The NVIDIA driver on your system is too old" on a 570 node
        r.fail("cuda init", str(e).split(". ")[0])
        return
    n = torch.cuda.device_count()
    r.facts["gpu_count"] = n
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 2**30
        if cc == prof["cc"]:
            r.ok(f"gpu{i}: {name}, cc {cc[0]}.{cc[1]}, {mem_gb:.1f} GiB")
        elif cc < (7, 0):
            r.fail(
                f"gpu{i}",
                f"{name} cc {cc} is below 7.0: unsupported by vLLM and CUDA 13 (did the job land on a 1080 Ti node? use -C 2080ti)",
            )
        elif prof["bf16"] and cc < (8, 0):
            r.fail(f"gpu{i}", f"{name} cc {cc} has no bf16: the {prof_name} profile needs cc >= 8.0")
        else:
            r.warn(f"gpu{i}", f"{name} cc {cc}; profile {prof_name} is tuned for the {prof['gpu']}")
    if n:
        _check_sdpa_kernels(r, torch)
    if prof["bf16"]:
        if n and torch.cuda.is_bf16_supported():
            r.ok(f"bf16 supported -> the {prof_name} profile trains and samples in bf16")
        else:
            r.fail("bf16", f"not supported on this GPU, but the {prof_name} profile runs bf16")
    elif n and torch.cuda.is_bf16_supported():
        r.warn("bf16", "reported as supported; configs still force fp16 for reproducibility across nodes")
    else:
        r.ok("bf16 unsupported on this GPU -> all configs use float16 (as required)")
    # Which attention backend vLLM will pick (mirrors vllm/platforms/cuda.py priority + support checks).
    try:
        from vllm.platforms import current_platform  # type: ignore

        if current_platform.has_device_capability(80):
            if prof["backend"] == "FLASH_ATTN":
                r.ok(f"vllm attention backend on cc>=8.0 -> FLASH_ATTN (the {prof_name} profile pins it)")
            else:
                r.warn(
                    "vllm attention", "cc>=8.0 -> FLASH_ATTN; the plan was validated for TRITON_ATTN on sm_75"
                )
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
            r.fail(
                var,
                f"{p} not writable here ({e}); staged data and run outputs live on /home2 (NFS), checkpoints on /scratch",
            )
            r.facts[var] = {"path": p, "writable": False}
    cache_root = os.environ.get("MC_CACHE_ROOT", "")
    if Path("/scratch").is_dir() and not cache_root.startswith("/scratch"):
        r.fail(
            "MC_CACHE_ROOT",
            f"{cache_root} is not on node-local /scratch although this node has one: a login-node environment "
            "leaked into the job (configs/ada.env.sh re-derives it by hostname; do not pin it in the environment)",
        )
    if os.environ.get("MC_SITE", "ada") != "ada":
        return
    # /home2 is the only durable file system compute nodes can see (docs/decisions/010): 25 GB and
    # 300k files per user hold the venv (~9.6 GB), the staged model/data (~5.2 GB) and every run's
    # durable outputs. Warn before the quota bites.
    home = Path.home()
    try:
        mib = int(
            subprocess.run(
                ["du", "-sm", str(home)], capture_output=True, text=True, timeout=120
            ).stdout.split()[0]
        )
        files = int(
            subprocess.run(
                ["du", "-s", "--inodes", str(home)], capture_output=True, text=True, timeout=120
            ).stdout.split()[0]
        )
        msg = f"home usage {mib / 1024:.1f} GiB in {files} files (quota 25 GB / 300k files: venv, staged data, run outputs)"
        if mib > 20_000 or files > 250_000:
            r.warn("home quota", msg)
        else:
            r.ok(msg)
        r.facts["home_usage"] = {"mib": mib, "files": files}
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
        f"slurm job {job} on {os.environ.get('SLURMD_NODENAME', '?')}, gpu ids={os.environ.get('SLURM_JOB_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
    )
    slurm = {
        k: os.environ.get(k)
        for k in (
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_QOS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_CPU",
            "SLURM_JOB_GPUS",
        )
    }
    r.facts["slurm"] = slurm
    r.ok(
        "slurm request: "
        + ", ".join(f"{k.removeprefix('SLURM_').lower()}={v}" for k, v in slurm.items() if v)
    )
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        r.facts["memlock_soft"] = soft
        if soft == resource.RLIM_INFINITY:
            r.ok("ulimit -l unlimited (pinned host RAM for the FSDP2 offload policy)")
        else:
            r.warn(
                "ulimit -l",
                f"{soft} bytes; the offload policy pins ~7 GB, CUDA falls back to pageable copies",
            )
    except Exception:  # noqa: BLE001
        pass
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
    """One node, and as many visible GPUs as the layout expects (TP == GPUs; decisions 002/012)."""
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
        r.facts["visible_gpus"] = n_vis
        want = os.environ.get("MC_SLURM_GPUS")
        layout = os.environ.get("MC_LAYOUT", "?")
        if want is None:
            r.warn(
                f"visible GPUs: {n_vis} ({vis})",
                "MC_SLURM_GPUS unset; `source configs/ada.env.sh` to compare with the layout",
            )
        elif int(want) == n_vis:
            r.ok(f"visible GPUs: {n_vis} ({vis}) == MC_SLURM_GPUS for layout {layout}")
        else:
            r.fail(
                "visible GPUs",
                f"{n_vis} ({vis}) but layout {layout} expects {want}; check MC_LAYOUT and the sbatch --gres",
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
    want = hw_profile()[1]["backend"]
    if supported and supported[0] == want:
        r.ok(f"attention backend for cc {cc.major}.{cc.minor}: {want} (first in priority among {supported})")
    elif want in supported:
        r.ok(
            f"attention backend {want} supports cc {cc.major}.{cc.minor} (configs pin it; priority order {supported})"
        )
    elif supported:
        r.fail(
            "attention backend",
            f"configs pin {want}, which does not support cc {cc.major}.{cc.minor}: {supported}",
        )
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
        # the same layout/memory overrides the launcher passes (configs/ada.env.sh: MC_TRAIN_OVERRIDES)
        overrides = os.environ.get("MC_TRAIN_OVERRIDES", "").split()
        cfg = compose_train_config(name, os.environ.get("MC_VERL_CONFIG_DIR") or None, overrides)
        problems = check(cfg)
        n_gpus = int(cfg.trainer.n_gpus_per_node)
        r.facts["n_gpus_per_node"] = n_gpus
        r.facts["train_overrides"] = overrides
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis is not None and len([x for x in vis.split(",") if x.strip()]) != n_gpus:
            problems.append(f"config expects {n_gpus} GPU(s) but CUDA_VISIBLE_DEVICES={vis}")
    except Exception as e:  # noqa: BLE001
        r.fail(f"config {name}", f"does not compose: {type(e).__name__}: {e}")
        return
    for p in problems:
        r.fail(f"config {name}", p)
    if not problems:
        r.ok(
            f"config {name} composes and passes every invariant ({cfg.get('hw_profile')} profile, D1, D2, D4, {n_gpus}-GPU layout, seeds, resume)"
        )


def _writable(path: str) -> str | None:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        probe = Path(path) / ".mc_write_probe"
        probe.write_text("ok")
        probe.unlink()
        return None
    except OSError as e:
        return str(e)


def check_durable_dir(r: Report) -> None:
    """Both homes of a run must be writable: the live run dir (checkpoints) and the durable mirror on /home2."""
    for label, var in (
        ("run dir (checkpoints, live outputs)", "MC_RUN_DIR"),
        ("durable mirror dir", "MC_DURABLE_DIR"),
    ):
        path = os.environ.get(var)
        if not path:
            continue
        err = _writable(path)
        if err is None:
            r.ok(f"{label} writable: {path}")
        else:
            r.fail(label, f"{path} not writable ({err})")


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
        check_driver(r)
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
