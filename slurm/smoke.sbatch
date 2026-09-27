#!/usr/bin/env bash
#SBATCH -J mc-smoke
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=05:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
# End-to-end plumbing test on the real layout: 20 MATH problems, 2 steps, groups of 4 (2 std + 2 CUTS),
# then the GPU tests (tests/gpu) against that run in the SAME allocation: a second GPU srun would compete
# with research/low's one-GPU-per-user quota (decision 012).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export MC_RUN_NAME="${MC_RUN_NAME:-smoke-s${MC_SEED:-42}-${SLURM_JOB_ID}}"   # fresh dir per smoke job
# run_train.sh installs the stage-out traps in its own shell, so it runs as a child and SLURM's
# --signal=B:SIGUSR1 (delivered to THIS shell) is forwarded to it.
bash slurm/run_train.sh smoke "$@" &
child=$!
trap 'kill -USR1 "${child}" 2>/dev/null || true' USR1
trap 'kill -TERM "${child}" 2>/dev/null || true' TERM
rc=0
while :; do
  if wait "${child}"; then rc=0; break; fi
  rc=$?
  kill -0 "${child}" 2>/dev/null || break   # a trap interrupted `wait`; keep waiting while the child lives
done
echo "[$(date '+%F %T')] smoke training finished with rc=${rc}; running the GPU tests"
source configs/ada.env.sh
if [ "${MC_CHECKPOINT_HOME:-scratch}" = "scratch" ]; then RUN_DIR="${MC_SCRATCH_RUNS_DIR}/${MC_RUN_NAME}"; else RUN_DIR="${MC_RUNS_DIR}/${MC_RUN_NAME}"; fi
JOB_DUR="${MC_RUNS_DIR}/${MC_RUN_NAME}/jobs/${SLURM_JOB_ID}"; mkdir -p "${JOB_DUR}"
trc=0
MC_SMOKE_RUN_DIR="${RUN_DIR}" MC_STAGED_MODEL_DIR="${MC_SCRATCH_ROOT}/stage/models/$(basename "${MC_MODEL_DIR}")" \
  "${MC_VENV_DIR}/bin/python" -m pytest -q tests/gpu -p no:cacheprovider 2>&1 | tee "${JOB_DUR}/gpu_test.log" || trc=$?
echo "[$(date '+%F %T')] smoke rc=${rc} gpu-tests rc=${trc} (log: ${JOB_DUR}/gpu_test.log)"
exit $(( rc != 0 ? rc : trc ))
