#!/usr/bin/env bash
# Keep hard-linked copies of checkpoints that verl prunes (trainer.max_actor_ckpt_to_keep: 2), so both
# arms can be evaluated at the same step if training has to stop before step 100. Every 10th step and
# every step from 85 on; ~0.84 GB each. Restart after an instance pause:
#   nohup setsid bash /home/mixed-cuts-data/ops/keep_ckpts.sh >> /home/mixed-cuts-data/ops/keep_ckpts.log 2>&1 &
set -u
R=/home/mixed-cuts-data/runs
keep() { [ $(( $1 % 10 )) -eq 0 ] || [ "$1" -ge 85 ]; }
while true; do
  for run in math_grpo-h200-s1 math_mixed_cuts-h200-s1; do
    C="$R/$run/checkpoints"; K="$R/$run/checkpoints_kept"
    latest="$(cat "$C/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)"
    for d in "$C"/global_step_*; do
      [ -d "$d" ] || continue
      s="${d##*_}"
      [ "$s" -le "$latest" ] && keep "$s" && [ ! -d "$K/global_step_$s" ] || continue
      ls "$d"/actor/*.pt >/dev/null 2>&1 || continue      # weights already pruned
      mkdir -p "$K" && cp -al "$d" "$K/global_step_$s" && echo "$(date -u +%T) kept $run step $s"
    done
  done
  sleep 60
done
