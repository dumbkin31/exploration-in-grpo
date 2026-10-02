#!/usr/bin/env bash
# Evaluation worker: pops targets from eval_todo.txt (one per line: "base" or "<run>:<step>") and evaluates
# each on GPU <gpu> with jarvis/eval.sh. Several workers (one per GPU) can share the list.
# A checkpoint whose weights verl pruned is first restored (hard links) from checkpoints_kept/.
#   nohup setsid bash /home/mixed-cuts-data/ops/eval_worker.sh <gpu> >> /home/mixed-cuts-data/ops/eval_worker.log 2>&1 &
set -u
GPU="${1:?usage: eval_worker.sh <gpu>}"
O=/home/mixed-cuts-data/ops; R=/home/mixed-cuts-data/runs
next() { flock "$O/eval_todo.lock" bash -c "l=\$(head -1 $O/eval_todo.txt 2>/dev/null); [ -n \"\$l\" ] && sed -i 1d $O/eval_todo.txt; echo \"\$l\""; }
cd /home/mixed-cuts
while t="$(next)"; [ -n "$t" ]; do
  log="$R/eval_logs/${t//:/_}.log"
  if [ "$t" != base ]; then
    run="${t%%:*}"; step="${t##*:}"; C="$R/$run/checkpoints"; K="$R/$run/checkpoints_kept/global_step_$step"
    if ! ls "$C/global_step_$step"/actor/*.pt >/dev/null 2>&1; then
      if [ -d "$K" ]; then mkdir -p "$C/global_step_$step" && cp -aln "$K/." "$C/global_step_$step/" && echo "$(date -u +%T) $t: weights restored from checkpoints_kept"
      else echo "$(date -u +%T) $t: no weights left, skipped"; continue; fi
    fi
  fi
  echo "$(date -u +%T) $t: evaluating on GPU $GPU (log $log)"
  if [ "$t" = base ]; then bash jarvis/eval.sh base "$GPU"; else bash jarvis/eval.sh "$run" "$GPU" "$step"; fi > "$log" 2>&1
  echo "$(date -u +%T) $t: done, exit $?"
done
echo "$(date -u +%T) GPU $GPU: queue empty"
