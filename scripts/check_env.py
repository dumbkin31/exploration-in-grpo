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
            level = r.fail if var == "MC_SCRATCH_ROOT" else r.warn
            level(var, f"{p} not writable here ({e}); on compute nodes /share1 may be master-only")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-gpu", action="store_true", help="skip GPU checks (login node / laptop)")
    ap.add_argument("--no-staged", action="store_true", help="skip staged model/data checks")
    ap.add_argument("--no-pins", action="store_true", help="skip package pin checks (dev env)")
    ap.add_argument(
        "--facts-json", default=os.environ.get("MC_ENV_FACTS_JSON"), help="write detected facts here"
    )
    args = ap.parse_args()

    r = Report()
    print("== python ==")
    check_python(r)
    if not args.no_pins:
        print("== pins ==")
        check_pins(r)
    if not args.no_gpu:
        print("== gpu ==")
        check_gpu(r)
    print("== storage ==")
    check_storage(r)
    if not args.no_staged:
        print("== staged model/data ==")
        check_staged(r)
    print("== network ==")
    check_internet(r)
    print("== slurm ==")
    check_slurm(r)

    if args.facts_json:
        Path(args.facts_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.facts_json).write_text(json.dumps(r.facts, indent=2))
        print(f"facts written to {args.facts_json}")

    print("== RESULT:", "FAIL (see above)" if r.failed else "OK", "==")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
