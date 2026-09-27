#!/usr/bin/env python
"""Per-GPU memory profile of a training run, by phase (Task A).

Two modes:

``--watch OUT.jsonl``
    Sidecar started by ``slurm/common.sh``: samples ``nvidia-smi --query-gpu=index,memory.used`` and
    the SLURM job cgroup's ``memory.current`` (host RSS of every process in the job) every ``--interval``
    seconds and appends ``{"t": ..., "job": ..., "mem_mib": [g0, g1, ...], "host_mib": h}``.
    It is killed by the job's EXIT trap, so data exists however the job ends. The host sample matters on
    research/low: the cgroup is 30 GB and kernel 5.15 has no ``memory.peak`` (decision 012).

``--report RUN_DIR``
    Joins ``RUN_DIR/gpu_mem.jsonl`` with ``RUN_DIR/phases.jsonl`` (written by the trainer:
    rollout / old_log_prob / ref_log_prob / update_actor windows per step) and prints a markdown
    table of peak memory per GPU per phase, plus verl's own ``actor/perf/max_memory_allocated_gb``
    from ``metrics.jsonl`` (a cumulative torch peak, for cross-checking). The sbatch script saves
    the output as ``RUN_DIR/memory_profile.md``.

The rollout phase is vLLM (a separate process, so torch's counters cannot see it); the other
phases are the actor/ref FSDP workers. Use the peaks to tune ``gpu_memory_utilization`` and
``ppo_max_token_len_per_gpu`` (docs/decisions/002).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PHASES = ("rollout", "old_log_prob", "ref_log_prob", "update_actor")


def _nvidia_smi_cmd() -> list[str]:
    """Only this job's GPUs: research/low jobs share a 4-GPU node with other users (decision 012)."""
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis and not vis.startswith("MIG"):
        cmd += ["-i", vis]
    return cmd


def sample_gpu_mem() -> list[int] | None:
    try:
        out = subprocess.run(
            _nvidia_smi_cmd(),
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    mems: dict[int, int] = {}
    for line in out.strip().splitlines():
        try:
            idx, mem = line.split(",")
            mems[int(idx)] = int(float(mem))
        except ValueError:
            continue
    return [mems[i] for i in sorted(mems)] if mems else None


def _host_mem_file() -> Path | None:
    """The SLURM job's cgroup-v2 memory.current (readable inside the job); None outside SLURM."""
    job = os.environ.get("SLURM_JOB_ID")
    cands = []
    if job:
        cands.append(Path(f"/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job}/memory.current"))
    try:  # walk up from our own cgroup to the job_* level
        own = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split(":", 2)[-1]
        p = Path("/sys/fs/cgroup" + own)
        while p != Path("/sys/fs/cgroup"):
            if p.name.startswith("job_"):
                cands.append(p / "memory.current")
                break
            p = p.parent
    except (OSError, IndexError):
        pass
    for c in cands:
        try:
            c.read_text()
            return c
        except OSError:
            continue
    return None


def sample_host_mib(f: Path | None) -> int | None:
    if f is None:
        return None
    try:
        return int(f.read_text().strip()) // 1048576
    except (OSError, ValueError):
        return None


def sample_top_rss(n: int = 8) -> list[list]:
    """[[rss_mib, short command], ...] of this user's biggest processes (attributes the host budget)."""
    try:
        out = subprocess.run(
            ["ps", "-u", os.environ.get("USER", str(os.getuid())), "-o", "rss=,args=", "--sort=-rss"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    rows: list[list] = []
    for line in out.splitlines()[:n]:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rss_kib, args = parts
        cmd = args.strip()
        for marker in ("ray::", "vllm", "-m mixed_cuts.main", "raylet", "gcs_server", "python"):
            if marker in cmd:
                cmd = marker if marker != "ray::" else cmd[cmd.index("ray::") : cmd.index("ray::") + 40]
                break
        rows.append([int(rss_kib) // 1024, cmd[:60]])
    return rows


def watch(path: Path, interval: float, job: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    host_file = _host_mem_file()
    every = max(1, int(20 / interval))  # a process snapshot every ~20 s
    i = 0
    with open(path, "a", encoding="utf-8") as f:
        while True:
            mems = sample_gpu_mem()
            if mems is not None:
                row = {"t": time.time(), "job": job, "mem_mib": mems}
                host = sample_host_mib(host_file)
                if host is not None:
                    row["host_mib"] = host
                if i % every == 0:
                    row["top_rss"] = sample_top_rss()
                f.write(json.dumps(row) + "\n")
                f.flush()
            i += 1
            time.sleep(interval)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def report(run_dir: Path) -> str:
    samples = _read_jsonl(run_dir / "gpu_mem.jsonl")
    phases = _read_jsonl(run_dir / "phases.jsonl")
    metrics = _read_jsonl(run_dir / "metrics.jsonl")
    lines = [f"# Memory profile: {run_dir}", ""]
    if not samples:
        lines.append("No gpu_mem.jsonl samples found (was the sampler started?).")
        return "\n".join(lines)
    n_gpu = max(len(s["mem_mib"]) for s in samples)
    lines.append(
        f"{len(samples)} nvidia-smi samples over {n_gpu} GPU(s); {len(phases)} phase windows; {len(metrics)} logged steps."
    )
    lines.append("")
    # peak per phase per GPU (max over every sample inside any window of that phase)
    peak: dict[str, list[int]] = {p: [0] * n_gpu for p in PHASES}
    count: dict[str, int] = defaultdict(int)
    windows = defaultdict(list)
    for ph in phases:
        windows[ph["phase"]].append((ph["t_start"], ph["t_end"], ph.get("step")))
    for s in samples:
        t = s["t"]
        for name, wins in windows.items():
            if any(a <= t <= b for a, b, _ in wins):
                count[name] += 1
                for i, m in enumerate(s["mem_mib"][:n_gpu]):
                    peak.setdefault(name, [0] * n_gpu)[i] = max(peak[name][i], m)
    overall = [max(s["mem_mib"][i] for s in samples if len(s["mem_mib"]) > i) for i in range(n_gpu)]
    lines.append("| phase | samples | " + " | ".join(f"GPU{i} peak MiB" for i in range(n_gpu)) + " |")
    lines.append("|---|---|" + "---|" * n_gpu)
    for name in list(PHASES) + [p for p in windows if p not in PHASES]:
        if name in peak:
            lines.append(f"| {name} | {count[name]} | " + " | ".join(str(v) for v in peak[name]) + " |")
    lines.append("| whole job | " + str(len(samples)) + " | " + " | ".join(str(v) for v in overall) + " |")
    lines.append("")
    lines.append("11 GiB cards hold 11264 MiB; leave >= 1 GiB headroom for fragmentation.")
    # host RSS of the whole job (cgroup memory.current, sampled): the 30 GB research/low cgroup (012)
    hosts = [s["host_mib"] for s in samples if s.get("host_mib") is not None]
    if hosts:
        host_peak: dict[str, int] = defaultdict(int)
        for s in samples:
            if s.get("host_mib") is None:
                continue
            for name, wins in windows.items():
                if any(a <= s["t"] <= b for a, b, _ in wins):
                    host_peak[name] = max(host_peak[name], s["host_mib"])
        lines.append("")
        # biggest processes seen (max RSS per command over the snapshots)
        top: dict[str, int] = {}
        for s_ in samples:
            for rss, cmd in s_.get("top_rss") or []:
                top[cmd] = max(top.get(cmd, 0), int(rss))
        if top:
            lines.append("")
            lines.append("| process (max RSS over the run) | MiB |")
            lines.append("|---|---|")
            for cmd, rss in sorted(top.items(), key=lambda kv: -kv[1])[:10]:
                lines.append(f"| `{cmd}` | {rss} |")
        lines.append(
            f"Host RSS of the job cgroup (sampled memory.current): peak {max(hosts)} MiB, "
            f"median {sorted(hosts)[len(hosts) // 2]} MiB"
            + (" | per phase: " + ", ".join(f"{k} {v}" for k, v in host_peak.items()) if host_peak else "")
            + ". research/low cgroup limit: 30000 MiB."
        )
    # verl's torch counters
    keys = [
        "actor/perf/max_memory_allocated_gb",
        "actor/perf/max_memory_reserved_gb",
        "actor/perf/cpu_memory_used_gb",
    ]
    found = {k: max((m[k] for m in metrics if k in m), default=None) for k in keys}
    if any(v is not None for v in found.values()):
        lines.append("")
        lines.append("verl torch counters (cumulative peaks over the run, actor workers):")
        for k, v in found.items():
            if v is not None:
                lines.append(f"- {k}: {v:.2f}")
    # per-step phase durations
    if phases:
        dur: dict[str, list[float]] = defaultdict(list)
        for ph in phases:
            dur[ph["phase"]].append(ph["t_end"] - ph["t_start"])
        lines.append("")
        lines.append("| phase | mean seconds | max seconds | n |")
        lines.append("|---|---|---|---|")
        for name, ds in dur.items():
            lines.append(f"| {name} | {sum(ds) / len(ds):.1f} | {max(ds):.1f} | {len(ds)} |")
        steps = [ph.get("step") for ph in phases if ph.get("step")]
        if steps:
            per_step = defaultdict(float)
            for ph in phases:
                per_step[ph.get("step")] += ph["t_end"] - ph["t_start"]
            mean_step = sum(per_step.values()) / len(per_step)
            lines.append("")
            lines.append(
                f"Mean measured time per step (sum of phases): {mean_step / 60:.1f} min -> "
                f"save_freq for ~30 min between checkpoints: {max(1, round(30 * 60 / mean_step))}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", metavar="OUT_JSONL", help="sample nvidia-smi forever into this file")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--job", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--report", metavar="RUN_DIR", help="print the markdown report for a run dir")
    args = ap.parse_args()
    if args.watch:
        return watch(Path(args.watch), args.interval, args.job)
    if args.report:
        print(report(Path(args.report)))
        return 0
    ap.error("one of --watch or --report is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
