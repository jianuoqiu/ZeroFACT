#!/bin/bash
# Record PERFECT episodes (policy reached every subgoal and retracted) for the four play2perfect
# problems, ONE Kit process per episode with its own seed. One process per episode because the
# simulator is bit-deterministic only from a cold start: an episode recorded after an in-process
# auto-reset carries hidden PhysX state and does not replay exactly from its saved initial state
# (measured 2026-09-08: joint positions off by up to 0.28 rad, forces by tens of newtons). A seed
# whose episode is not perfect is discarded and the next seed is tried, until N perfect episodes
# exist (at most MAX_SEEDS tries per problem). Episodes land in
#   outputs/play2perfect/episodes[_<hand>]/<problem>/<stamp>_seed<k>/ep_0000/
# The hand follows ISAACSIMENVS_HAND (sharpa default, xhand); export it before running.
#   bash play2perfect_force_controller/collect_all.sh [N per problem] [first seed] [problems...]
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-/home/jianuoqiu/anaconda3/envs/env_isaaclab/bin/python}
N=${1:-3}; SEED0=${2:-0}; shift 2 2>/dev/null || shift $#
PROBLEMS=${@:-tight_insertion beam_assembly_step1 beam_assembly_step2 screwing}
MAX_SEEDS=${MAX_SEEDS:-$((N * 6))}
STAMP=$(date +%Y%m%d_%H%M%S)
HAND=${ISAACSIMENVS_HAND:-sharpa}
EP_ROOT=outputs/play2perfect/episodes
[ "$HAND" = sharpa ] || EP_ROOT=outputs/play2perfect/episodes_$HAND
for problem in $PROBLEMS; do
    got=0
    for ((k = 0; k < MAX_SEEDS && got < N; k++)); do
        seed=$((SEED0 + k))
        out=$EP_ROOT/$problem/${STAMP}_seed${seed}
        echo "===== $problem seed $seed ($got/$N perfect so far) ====="
        $PY play2perfect_force_controller/collect_episodes.py --problem $problem --episodes 1 \
            --seed $seed --only-perfect --max-attempts 1 --out-dir $out ${COLLECT_FLAGS:-} \
            2>&1 | grep -E "Traceback|RuntimeError|\[collect\] (ep [0-9]+:|attempt)"
        if [ -f "$out/ep_0000/summary.json" ]; then got=$((got + 1)); else rm -rf "$out"; fi
    done
    echo "===== $problem: $got perfect episodes ====="
done
