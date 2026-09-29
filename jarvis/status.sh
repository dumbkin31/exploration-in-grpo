#!/usr/bin/env bash
# Progress of every H100 run: last logged step, minutes per step, ETA, newest checkpoint, launcher alive,
# GPU-hours so far and their cost at MC_GPU_RATE_INR (default 112.59: H100 spot on Jarvislabs, 2026-09-29).
#   bash jarvis/status.sh            (add -w to refresh every 60 s)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MC_SITE=jarvis
# shellcheck source=../configs/ada.env.sh
source configs/ada.env.sh
show() {
  date '+%F %T'
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
  "${MC_VENV_DIR}/bin/python" - "${MC_RUNS_DIR}" "${MC_RUN_TAG}" "${MC_GPU_RATE_INR:-112.59}" <<'PY'
import json, re, subprocess, sys
from pathlib import Path
runs, tag, rate = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
hours_all = 0.0
for run in sorted(runs.glob(f"*-{tag}-s*")):
    rows = []
    m = run / "metrics.jsonl"
    if m.exists():
        for line in m.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    last = rows[-1]["step"] if rows else 0
    secs = [r["timing_s/step"] for r in rows[-5:] if isinstance(r.get("timing_s/step"), (int, float))]
    per = sum(secs) / len(secs) if secs else None
    per_step = {r["step"]: r["timing_s/step"] for r in rows if isinstance(r.get("timing_s/step"), (int, float))}
    hours = sum(per_step.values()) / 3600
    hours_all += hours
    total = 100
    launcher_log = run / "launcher" / "launcher.log"
    if launcher_log.exists():
        m = re.search(r": (\d+) steps, layout", launcher_log.read_text())
        total = int(m.group(1)) if m else total
    ck = run / "checkpoints" / "latest_checkpointed_iteration.txt"
    ckpt = ck.read_text().strip() if ck.exists() else "-"
    arm = run.name.rsplit(f"-{tag}-s", 1)[0]
    alive = subprocess.run(["pgrep", "-f", f"jarvis/train.sh {arm} "], capture_output=True).returncode == 0
    eta = f"{(total - last) * per / 3600:.1f} h" if per else "?"
    reward = rows[-1].get("critic/score/mean") if rows else None
    acr = rows[-1].get("mixed_cuts/advantage_collapse_rate") if rows else None
    print(
        f"{run.name:28s} step {last:>3}/{total}  {per / 60 if per else float('nan'):5.1f} min/step  ETA {eta:>7s}  "
        f"ckpt {ckpt:>3}  reward {reward if reward is not None else '-'}  ACR {acr if acr is not None else '-'}  "
        f"launcher {'running' if alive else 'STOPPED'}  {hours:.1f} GPU-h"
    )
print(f"training so far: {hours_all:.1f} GPU-h = INR {hours_all * rate:,.0f} at INR {rate:g}/h (logged step times only)")
PY
}
if [ "${1:-}" = "-w" ]; then while true; do clear; show; sleep 60; done; else show; fi
