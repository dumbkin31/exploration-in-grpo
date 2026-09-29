#!/usr/bin/env bash
# Convert a verl FSDP checkpoint (full FT or LoRA) into a HuggingFace model directory for eval / vLLM.
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
BASE="${MC_STAGED_MODEL_DIR:-${MC_MODEL_DIR:-}}"
if [ -f "$SRC/lora_train_meta.json" ] && "$PY" - "$SRC" <<'PY'
import glob, sys, torch
pts = sorted(glob.glob(sys.argv[1] + "/model_world_size_*_rank_0.pt"))
sd = torch.load(pts[0], map_location="cpu", weights_only=False) if pts else {}
sys.exit(0 if sd and all(("lora_" in k or ".adapter_" in k) for k in sd) else 1)
PY
then
  # save_lora_only checkpoint (the default since decision 012): the .pt holds adapters only, so verl's
  # merger would assert on the missing base keys. Rebuild the PEFT adapter and merge it into the base model.
  [ -d "$BASE" ] || { echo "LoRA-only checkpoint needs the base model: set MC_MODEL_DIR (source configs/ada.env.sh)" >&2; exit 1; }
  "$PY" "$(dirname "$0")/merge_lora.py" --verl-actor-dir "$SRC" --base "$BASE" --out "$DST" --dtype "${MC_MERGE_DTYPE:-float16}"
else
  "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST"
  # full-state LoRA checkpoints: the merger leaves base weights + lora_adapter/ side by side; fold the adapter in
  "$PY" "$(dirname "$0")/merge_lora.py" "$DST" --dtype "${MC_MERGE_DTYPE:-float16}"
fi
# verl's merger writes weights + config; make sure the tokenizer travels with the model.
if [ -n "${MC_MODEL_DIR:-}" ] && [ ! -f "$DST/tokenizer_config.json" ]; then
  cp -n "$MC_MODEL_DIR"/tokenizer* "$MC_MODEL_DIR"/*.jinja "$MC_MODEL_DIR"/vocab.json "$MC_MODEL_DIR"/merges.txt "$DST"/ 2>/dev/null || true
fi
echo "done: $DST"
