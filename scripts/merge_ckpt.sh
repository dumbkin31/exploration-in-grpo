#!/usr/bin/env bash
# Convert a verl FSDP checkpoint into a HuggingFace model directory for eval / vLLM.
#
#   scripts/merge_ckpt.sh <run_dir> [global_step]
#
# <run_dir> is the trainer's default_local_dir (contains global_step_N/actor/). Without a step
# the latest one is used. Output: <run_dir>/hf/global_step_N. Uses verl's own merger:
#   python -m verl.model_merger merge --backend fsdp --local_dir .../actor --target_dir ...
set -euo pipefail
RUN_DIR="${1:?usage: merge_ckpt.sh <run_dir> [global_step]}"
STEP="${2:-}"
if [ -z "$STEP" ]; then
  if [ -f "$RUN_DIR/latest_checkpointed_iteration.txt" ]; then
    STEP="$(cat "$RUN_DIR/latest_checkpointed_iteration.txt")"
  else
    STEP="$(ls -d "$RUN_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)"
  fi
fi
[ -n "$STEP" ] || { echo "no global_step_* under $RUN_DIR" >&2; exit 1; }
SRC="$RUN_DIR/global_step_${STEP}/actor"
DST="$RUN_DIR/hf/global_step_${STEP}"
[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
PY="${MC_VENV_DIR:-.venv}/bin/python"
echo "merging $SRC -> $DST"
"$PY" -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST"
# verl's merger writes weights + config; make sure the tokenizer travels with the model.
if [ -n "${MC_MODEL_DIR:-}" ] && [ ! -f "$DST/tokenizer_config.json" ]; then
  cp -n "$MC_MODEL_DIR"/tokenizer* "$MC_MODEL_DIR"/*.jinja "$MC_MODEL_DIR"/vocab.json "$MC_MODEL_DIR"/merges.txt "$DST"/ 2>/dev/null || true
fi
echo "done: $DST"
