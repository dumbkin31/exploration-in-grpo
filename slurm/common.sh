#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared job plumbing for every slurm/*.sbatch and slurm/run_train.sh. Source it, then call:
#
#   mc_job_init [RUN_NAME]   paths, env, preflight (aborts early on FAIL), GPU memory sampler
#   mc_stage_in              model + parquet from $MC_STAGE_ROOT -> node-local /scratch
#   mc_run_with_traps CMD    run CMD; on SIGUSR1/SIGTERM/EXIT copy job logs out and prune checkpoints
#   mc_stage_out             idempotent copy of job logs into the durable run dir + engine-log facts
#   mc_train_main CONFIG [overrides...]   the whole training job (used by run_train.sh)
#
# Layout (docs/decisions/005 and 009):
#   MC_RUN_NAME    = <config>-s<seed>            stable across resubmissions (override: MC_RUN_NAME=...)
#   MC_RUN_DIR     = live run dir: checkpoints/, metrics.jsonl, cuts_stats/, rollout_dumps/,
#                    phases.jsonl, gpu_mem.jsonl, wandb/, jobs/<jobid>/
#                    MC_CHECKPOINT_HOME=scratch (default): $MC_SCRATCH_RUNS_DIR/<run>, node-local,
#                    because /home2 (the durable NFS) has a 25 GB quota and one checkpoint is ~21 GB
#   MC_DURABLE_DIR = $MC_RUNS_DIR/<run> on /home2 (NFS, every node): everything except checkpoints, mirrored every
#                    MC_MIRROR_INTERVAL seconds and at exit, plus node.txt (which node holds the
#                    checkpoints; `make sbatch-train` pins resubmissions to it with -w)
#   MC_SCRATCH_ROOT/stage                        staged model/data copies (re-rsynced per job)
# verl's resume_mode=auto reads $MC_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt, so a
# resubmitted job ON THE SAME NODE continues from the last periodic checkpoint (verl cannot
# checkpoint on SIGTERM; the grace period only flushes logs). WANDB_RUN_ID = run name keeps one W&B run.
# ---------------------------------------------------------------------------
set -euo pipefail

_MC_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MC_REPO_ROOT="${MC_REPO_ROOT:-$(cd "${_MC_COMMON_DIR}/.." && pwd)}"

mc_log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

mc_check_driver() {
  # The pinned wheels are CUDA 13.0 builds and need an NVIDIA driver >= MC_MIN_DRIVER_MAJOR (580); the
  # 2080 Ti nodes are mixed (docs/decisions/011). Runs before anything is written, so a bad node leaves no
  # node.txt behind. The node is recorded in MC_BAD_NODES_FILE, which `make sbatch-*` excludes from then on.
  local drv major min="${MC_MIN_DRIVER_MAJOR:-580}"
  drv="$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)"
  major="${drv%%.*}"
  if [ -z "${drv}" ] || ! [ "${major}" -ge "${min}" ] 2>/dev/null; then
    mc_log "ERROR: $(hostname) has NVIDIA driver '${drv:-none}' (need >= ${min} for the cu130 wheels; decision 011)"
    if [ -n "${MC_BAD_NODES_FILE:-}" ] && mkdir -p "$(dirname "${MC_BAD_NODES_FILE}")" 2>/dev/null; then
      echo "$(hostname) driver=${drv:-none} job=${SLURM_JOB_ID:-?} $(date '+%F')" >> "${MC_BAD_NODES_FILE}"
      mc_log "recorded in ${MC_BAD_NODES_FILE}: rerun the same make sbatch-* command, it now excludes this node"
    fi
    exit 6
  fi
  mc_log "NVIDIA driver ${drv} on $(hostname): OK for the cu130 wheels (>= ${min})"
}

mc_check_cuda() {
  # A driver that is new enough can still fail to initialise CUDA (a GPU that fell off the bus, nvidia-uvm
  # missing, a stuck device): the bench job died with "CUDA unknown error" on gnode065 that way. Probe with
  # the real torch before anything is written; a failing node is recorded like an old-driver one (011).
  local py="${MC_VENV_DIR:-}/bin/python" out
  [ -x "${py}" ] || { mc_log "cuda probe skipped: ${py} missing"; return 0; }
  if out="$(timeout 120 "${py}" -c 'import torch; torch.cuda.init(); n=torch.cuda.device_count(); print(n, torch.cuda.get_device_name(0) if n else "-", [round(x/2**30,1) for x in torch.cuda.mem_get_info(0)] if n else "-")' 2>&1)"; then
    mc_log "cuda probe on $(hostname): ${out##*$'\n'} (count, name, [free GiB, total GiB])"
    return 0
  fi
  mc_log "ERROR: CUDA cannot initialise on $(hostname) (GPU ${CUDA_VISIBLE_DEVICES:-?}): ${out##*$'\n'}"
  if [ -n "${MC_BAD_NODES_FILE:-}" ] && mkdir -p "$(dirname "${MC_BAD_NODES_FILE}")" 2>/dev/null; then
    echo "$(hostname) cuda_init_failed gpu=${CUDA_VISIBLE_DEVICES:-?} job=${SLURM_JOB_ID:-?} $(date '+%F')" >> "${MC_BAD_NODES_FILE}"
    mc_log "recorded in ${MC_BAD_NODES_FILE}: rerun the same make sbatch-* command, it now excludes this node"
  fi
  exit 6
}

mc_job_init() {
  # shellcheck source=/dev/null
  source "${MC_REPO_ROOT}/configs/ada.env.sh"
  # SLURM exports the AMD device lists next to CUDA_VISIBLE_DEVICES for every --gres=gpu job; verl's
  # Worker refuses to start when ROCR_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are both set
  # (verl/single_controller/base/worker.py). NVIDIA-only cluster: drop the AMD ones.
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  mc_check_driver
  mc_check_cuda
  export MC_JOB_ID="${SLURM_JOB_ID:-local-$$}"
  export MC_SEED="${MC_SEED:-42}"
  export MC_RUN_NAME="${1:-${MC_RUN_NAME:-${SLURM_JOB_NAME:-job}-s${MC_SEED}}}"
  export MC_DURABLE_DIR="${MC_DURABLE_DIR:-${MC_RUNS_DIR}/${MC_RUN_NAME}}"   # /home2 (NFS): mirror of small outputs
  if [ "${MC_CHECKPOINT_HOME:-scratch}" = "scratch" ]; then
    export MC_RUN_DIR="${MC_RUN_DIR:-${MC_SCRATCH_RUNS_DIR}/${MC_RUN_NAME}}" # node-local: checkpoints + live outputs
  else
    export MC_RUN_DIR="${MC_RUN_DIR:-${MC_DURABLE_DIR}}"
  fi
  export MC_JOB_DIR="${MC_RUN_DIR}/jobs/${MC_JOB_ID}"                        # this submission's logs
  export MC_STAGED_MODEL_DIR="${MC_SCRATCH_ROOT}/stage/models/$(basename "${MC_MODEL_DIR}")"
  export MC_STAGED_DATA_DIR="${MC_SCRATCH_ROOT}/stage/data"
  export MC_ENV_FACTS_JSON="${MC_JOB_DIR}/env_facts.json"
  export MC_JOB_STDOUT="${SLURM_SUBMIT_DIR:-$PWD}/slurm-${SLURM_JOB_NAME:-job}-${MC_JOB_ID}.out"   # matches -o slurm-%x-%j.out
  # W&B: online from the node (offline fallback below), one run id across resubmissions
  # (WANDB_RESUME=allow continues the same curves; steps re-logged after a kill are dropped by W&B;
  # metrics.jsonl is the primary log). wandb creates its own `wandb/` under WANDB_DIR, so the run
  # folders are <run dir>/wandb/{run,offline-run}-<ts>-<id>; verl's Tracking passes no dir/id/resume.
  export WANDB_DIR="${MC_RUN_DIR}"
  export WANDB_RUN_ID="${MC_RUN_NAME}"
  export WANDB_RESUME="allow"
  export WANDB_NAME="${MC_RUN_NAME}"
  mkdir -p "${MC_DURABLE_DIR}" \
    || { mc_log "ERROR: cannot create ${MC_DURABLE_DIR} (is ${MC_STAGE_ROOT} mounted and writable here?)"; exit 3; }
  mkdir -p "${MC_RUN_DIR}" "${MC_JOB_DIR}" "${WANDB_DIR}" "${MC_SCRATCH_ROOT}/stage" \
    || { mc_log "ERROR: cannot create ${MC_RUN_DIR} (no writable node-local scratch on $(hostname)?)"; exit 3; }
  mc_check_node_pin
  export PATH="${MC_VENV_DIR}/bin:${PATH}"
  export PYTHONUNBUFFERED=1

  mc_log "job ${MC_JOB_ID} run ${MC_RUN_NAME} on $(hostname); run dir ${MC_RUN_DIR} (${MC_CHECKPOINT_HOME}); durable ${MC_DURABLE_DIR}"
  mc_log "python: $(command -v python) ; GPUs: ${CUDA_VISIBLE_DEVICES:-unset} ; nodes: ${SLURM_JOB_NUM_NODES:-?}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  {
    echo "job_id=${MC_JOB_ID}"; echo "run_name=${MC_RUN_NAME}"; echo "seed=${MC_SEED}"; echo "node=$(hostname)"
    echo "git=$(git -C "${MC_REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo ?)"; date
  } > "${MC_JOB_DIR}/job_info.txt"

  # Preflight: pins, sm_75, single node, engine version, storage, internet. Aborts on FAIL.
  python "${MC_REPO_ROOT}/scripts/check_env.py" ${MC_PREFLIGHT_ARGS:---no-staged} --facts-json "${MC_ENV_FACTS_JSON}" \
    | tee "${MC_JOB_DIR}/preflight.log"
  test "${PIPESTATUS[0]}" -eq 0 || { mc_log "preflight FAILED; see ${MC_JOB_DIR}/preflight.log"; exit 4; }
  if python - "$MC_ENV_FACTS_JSON" <<'PY'
import json, sys
try:
    facts = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if facts.get("internet", {}).get("https://huggingface.co") is False else 1)
PY
  then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mc_log "no internet on this node: HF_HUB_OFFLINE=1"
  fi
  if [ "${WANDB_MODE:-online}" = "online" ] && python - "$MC_ENV_FACTS_JSON" <<'PY'
import json, sys
try:
    facts = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if facts.get("internet", {}).get("https://api.wandb.ai") is False else 1)
PY
  then
    export WANDB_MODE=offline
    mc_log "api.wandb.ai unreachable from this node: WANDB_MODE=offline (push later with scripts/wandb_sync.sh)"
  fi
  mc_start_gpu_sampler
}

mc_record_host_peak() {
  # The research/low cgroup is 30 GB (decision 012) and kernel 5.15's cgroup v2 has no memory.peak, so
  # the GPU sampler sidecar (profile_memory.py --watch) also samples the job cgroup's memory.current;
  # the peak is the max over this job's samples. sstat's MaxRSS (largest single process) is kept beside it.
  local out="${MC_JOB_DIR}/host_mem_peak.txt" now="" peak="" rss=""
  local cur="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID:-x}/memory.current"
  [ -r "${cur}" ] && now="$(( $(cat "${cur}") / 1048576 ))"
  if [ -f "${MC_RUN_DIR}/gpu_mem.jsonl" ]; then
    peak="$(python - "${MC_RUN_DIR}/gpu_mem.jsonl" "${MC_JOB_ID}" <<'PY'
import json, sys
best = 0
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if str(d.get("job")) == sys.argv[2] and d.get("host_mib"):
        best = max(best, int(d["host_mib"]))
print(best or "")
PY
)"
  fi
  rss="$(sstat -n -P -j "${SLURM_JOB_ID:-x}.batch" -o MaxRSS 2>/dev/null | head -1)"
  {
    echo "cgroup_peak_mib=${peak:-unknown} (max of the sampled memory.current of this job)"
    echo "cgroup_now_mib=${now:-unknown}"
    echo "sstat_maxrss_batch=${rss:-unknown} (largest single process)"
    echo "request=cpus:${SLURM_CPUS_PER_TASK:-?} mem_per_cpu:${SLURM_MEM_PER_CPU:-?}M"
  } > "${out}"
  mc_log "host memory: cgroup peak ${peak:-?} MiB (sampled), now ${now:-?} MiB, sstat MaxRSS ${rss:-?}"
}

mc_check_node_pin() {
  # Checkpoints on node-local scratch only resume on the node that wrote them (docs/decisions/009).
  local node_file="${MC_DURABLE_DIR}/node.txt" here; here="$(hostname)"
  if [ "${MC_CHECKPOINT_HOME:-scratch}" = "scratch" ] && [ -f "${node_file}" ]; then
    local prev; prev="$(sed -n 's/^node=//p' "${node_file}" | tail -1)"
    if [ -n "${prev}" ] && [ "${prev}" != "${here}" ] && [ ! -f "${MC_RUN_DIR}/checkpoints/latest_checkpointed_iteration.txt" ]; then
      mc_log "ERROR: run ${MC_RUN_NAME} has its checkpoints on ${prev}, but this job landed on ${here}."
      mc_log "       Resubmit with 'make sbatch-train ...' (pins -w ${prev}) or set MC_ALLOW_NODE_CHANGE=1 to start over here."
      [ "${MC_ALLOW_NODE_CHANGE:-0}" = "1" ] || exit 5
      mc_log "MC_ALLOW_NODE_CHANGE=1: starting from scratch on ${here}"
    fi
  fi
  { echo "node=${here}"; echo "run_dir=${MC_RUN_DIR}"; echo "job=${MC_JOB_ID}"; echo "date=$(date '+%F %T')"; } >> "${node_file}"
}

mc_mirror() {
  # Copy everything except checkpoints from the (possibly node-local) run dir to the durable dir on
  # /home2 (docs/decisions/010). Idempotent. The durable side has a 25 GB quota, so the two per-step
  # directories (one jsonl per step each, ~5x smaller gzipped) are packed into one archive apiece.
  [ "${MC_RUN_DIR}" = "${MC_DURABLE_DIR}" ] && return 0
  mkdir -p "${MC_DURABLE_DIR}" 2>/dev/null || return 0
  rsync -a --exclude 'checkpoints/' --exclude 'cuts_stats/' --exclude 'rollout_dumps/' \
    --exclude '*.pt' --exclude '*.safetensors' "${MC_RUN_DIR}/" "${MC_DURABLE_DIR}/" \
    || mc_log "WARN: mirror to ${MC_DURABLE_DIR} failed (quota? /home2 unreachable?)"
  local d
  for d in cuts_stats rollout_dumps; do
    if [ -d "${MC_RUN_DIR}/${d}" ]; then
      tar -czf "${MC_DURABLE_DIR}/${d}.tar.gz.tmp" -C "${MC_RUN_DIR}" "${d}" 2>/dev/null \
        && mv -f "${MC_DURABLE_DIR}/${d}.tar.gz.tmp" "${MC_DURABLE_DIR}/${d}.tar.gz" \
        || mc_log "WARN: packing ${d} failed"
    fi
  done
}

_mc_mirror_loop() {
  while sleep "${MC_MIRROR_INTERVAL:-600}"; do mc_mirror; done
}

mc_start_gpu_sampler() {
  # Per-GPU memory timeline for scripts/profile_memory.py --report (joined with phases.jsonl).
  python "${MC_REPO_ROOT}/scripts/profile_memory.py" --watch "${MC_RUN_DIR}/gpu_mem.jsonl" --job "${MC_JOB_ID}" \
    > "${MC_JOB_DIR}/gpu_sampler.log" 2>&1 &
  export _MC_SAMPLER_PID=$!
}

mc_stop_gpu_sampler() {
  if [ -n "${_MC_SAMPLER_PID:-}" ] && kill -0 "${_MC_SAMPLER_PID}" 2>/dev/null; then
    kill "${_MC_SAMPLER_PID}" 2>/dev/null || true
    wait "${_MC_SAMPLER_PID}" 2>/dev/null || true
  fi
  _MC_SAMPLER_PID=""
}

mc_stage_in() {
  if [ ! -d "${MC_STAGE_ROOT}" ]; then
    mc_log "ERROR: ${MC_STAGE_ROOT} is not visible on $(hostname). Export MC_STAGE_ROOT to a path this node can read."
    exit 3
  fi
  if [ ! -f "${MC_MODEL_DIR}/config.json" ]; then
    mc_log "ERROR: model not staged at ${MC_MODEL_DIR}; run 'make prefetch' on the login node"; exit 3
  fi
  mkdir -p "${MC_STAGED_MODEL_DIR}" "${MC_STAGED_DATA_DIR}"
  mc_log "stage-in model -> ${MC_STAGED_MODEL_DIR}"
  rsync -a --info=stats1 "${MC_MODEL_DIR}/" "${MC_STAGED_MODEL_DIR}/"
  if [ -d "${MC_DATA_DIR}" ]; then
    mc_log "stage-in data  -> ${MC_STAGED_DATA_DIR}"
    rsync -a --info=stats1 "${MC_DATA_DIR}/" "${MC_STAGED_DATA_DIR}/"
  else
    mc_log "WARN: ${MC_DATA_DIR} missing (run 'make data'); continuing without data"
  fi
}

mc_engine_log_facts() {
  # vLLM logs its choices through Ray to the driver stdout (= this job's .out file). Record them.
  local out="${MC_JOB_STDOUT}"
  [ -f "${out}" ] || return 0
  local attn mrv2
  attn="$(grep -oE "Using AttentionBackendEnum\.[A-Z_]+ backend|Using [A-Z_]+ attention backend out of potential backends: \[[^]]*\]" "${out}" | head -1 || true)"
  if grep -q "Using V2 Model Runner" "${out}"; then mrv2=true; else mrv2=false; fi
  mc_log "engine facts: attention='${attn:-not found in log}' model_runner_v2=${mrv2}"
  python - "${MC_ENV_FACTS_JSON}" "${attn}" "${mrv2}" <<'PY'
import json, sys, os
path, attn, mrv2 = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
facts = json.load(open(path)) if os.path.exists(path) else {}
facts["vllm_attention_backend_log"] = attn or None
facts["vllm_model_runner_v2"] = mrv2
json.dump(facts, open(path, "w"), indent=2)
PY
  if [ "${mrv2}" = true ]; then
    mc_log "ERROR: vLLM used Model Runner V2; the CUTS logits processor needs V1 (custom LPs force V1 unless VLLM_USE_V2_MODEL_RUNNER=1 is set)"
  fi
}

mc_prune_checkpoints() {
  # verl's max_actor_ckpt_to_keep never prunes checkpoints saved before a restart (docs/decisions/005).
  # Keep the step named in latest_checkpointed_iteration.txt plus the previous one.
  local ckdir="${MC_RUN_DIR}/checkpoints" keep="${MC_KEEP_CHECKPOINTS:-2}"
  [ -d "${ckdir}" ] || return 0
  local latest=""
  [ -f "${ckdir}/latest_checkpointed_iteration.txt" ] && latest="$(cat "${ckdir}/latest_checkpointed_iteration.txt")"
  local dirs
  dirs="$(ls -d "${ckdir}"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)"
  local n; n="$(echo "${dirs}" | grep -c . || true)"
  [ "${n}" -gt "${keep}" ] || return 0
  echo "${dirs}" | head -n "$((n - keep))" | while read -r step; do
    [ -n "${step}" ] || continue
    [ "${step}" = "${latest}" ] && continue
    mc_log "pruning old checkpoint global_step_${step}"
    rm -rf "${ckdir}/global_step_${step}"
  done
}

mc_stage_out() {
  # Idempotent. Outputs already live in the durable run dir; this copies THIS job's logs next to
  # them, records the engine facts and prunes stale checkpoints. Safe to call from the trap and EXIT.
  mc_stop_gpu_sampler
  mc_engine_log_facts || true
  mc_record_host_peak
  if [ -f "${MC_JOB_STDOUT}" ]; then cp -f "${MC_JOB_STDOUT}" "${MC_JOB_DIR}/" 2>/dev/null || true; fi
  mc_prune_checkpoints || true
  if [ -n "${_MC_MIRROR_PID:-}" ] && kill -0 "${_MC_MIRROR_PID}" 2>/dev/null; then kill "${_MC_MIRROR_PID}" 2>/dev/null || true; fi
  _MC_MIRROR_PID=""
  mc_mirror
  mc_log "stage-out done (run dir ${MC_RUN_DIR}; durable copy ${MC_DURABLE_DIR})"
}

_MC_CHILD_PID=""
_mc_on_signal() {
  local sig="$1"
  mc_log "caught ${sig}: flushing (child pid ${_MC_CHILD_PID:-none}); the run resumes from its last checkpoint"
  if [ -n "${_MC_CHILD_PID}" ] && kill -0 "${_MC_CHILD_PID}" 2>/dev/null; then
    kill -TERM "${_MC_CHILD_PID}" 2>/dev/null || true
    for _ in $(seq 1 90); do kill -0 "${_MC_CHILD_PID}" 2>/dev/null || break; sleep 2; done
    kill -KILL "${_MC_CHILD_PID}" 2>/dev/null || true
  fi
  mc_stage_out
  exit 143
}

_mc_kill_watcher() {
  # Test hook (slurm/test_resume.sbatch): send this shell SIGUSR1 once metrics.jsonl reaches a step,
  # which exercises the real SLURM signal path end to end.
  local target="$1" parent="$2"
  while sleep 10; do
    kill -0 "${parent}" 2>/dev/null || exit 0
    if [ -f "${MC_RUN_DIR}/metrics.jsonl" ] && grep -qE "\"step\": ?${target}[,}]" "${MC_RUN_DIR}/metrics.jsonl"; then
      mc_log "kill watcher: step ${target} logged; sending SIGUSR1 to ${parent}"
      kill -USR1 "${parent}"
      exit 0
    fi
  done
}

mc_run_with_traps() {
  trap '_mc_on_signal SIGUSR1' USR1
  trap '_mc_on_signal SIGTERM' TERM
  trap 'mc_stage_out' EXIT
  if [ -n "${MC_KILL_AFTER_STEP:-}" ]; then
    _mc_kill_watcher "${MC_KILL_AFTER_STEP}" "$$" &
  fi
  _mc_mirror_loop &
  export _MC_MIRROR_PID=$!
  mc_log "launch: $*"
  "$@" &
  _MC_CHILD_PID=$!
  local rc=0
  wait "${_MC_CHILD_PID}" || rc=$?
  _MC_CHILD_PID=""
  mc_log "command finished with rc=${rc}"
  return "${rc}"
}

mc_train_main() {
  # Usage: mc_train_main <config-name> [extra hydra overrides...]; needs MC_SEED (default 42).
  local config="${1:?config name}"; shift || true
  export MC_SEED="${MC_SEED:-42}"
  mc_job_init "${MC_RUN_NAME:-${config}-s${MC_SEED}}"
  mc_stage_in
  # Config-specific preflight (compose + invariants + chat template) now that the model is staged.
  python "${MC_REPO_ROOT}/scripts/check_env.py" --no-pins --no-gpu --no-staged --only-config \
    --config "${config}" --model-dir "${MC_STAGED_MODEL_DIR}" | tee -a "${MC_JOB_DIR}/preflight.log"
  test "${PIPESTATUS[0]}" -eq 0 || { mc_log "config preflight FAILED"; exit 4; }
  local extra=()
  [ -n "${MC_SAVE_FREQ:-}" ] && extra+=("save_freq=${MC_SAVE_FREQ}")
  [ -n "${MC_TOTAL_STEPS:-}" ] && extra+=("trainer.total_training_steps=${MC_TOTAL_STEPS}")
  # layout/memory overrides of $MC_LAYOUT (configs/ada.env.sh); empty for the default research_1gpu
  if [ -n "${MC_TRAIN_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206
    extra+=(${MC_TRAIN_OVERRIDES})
    mc_log "layout ${MC_LAYOUT:-?}: ${MC_TRAIN_OVERRIDES}"
  fi
  cd "${MC_REPO_ROOT}"
  mc_run_with_traps python -m mixed_cuts.main \
    --config-name "${config}" \
    "hydra.searchpath=[pkg://verl.trainer.config]" \
    "seed=${MC_SEED}" \
    "run_name=${MC_RUN_NAME}" \
    "paths.run_dir=${MC_RUN_DIR}" \
    "paths.model_dir=${MC_STAGED_MODEL_DIR}" \
    "paths.data_dir=${MC_STAGED_DATA_DIR}" \
    "paths.repo_dir=${MC_REPO_ROOT}" \
    "hydra.run.dir=${MC_JOB_DIR}/hydra" \
    "${extra[@]}" "$@"
  local rc=$?
  python "${MC_REPO_ROOT}/scripts/profile_memory.py" --report "${MC_RUN_DIR}" > "${MC_RUN_DIR}/memory_profile.md" 2>/dev/null || true
  return "${rc}"
}
