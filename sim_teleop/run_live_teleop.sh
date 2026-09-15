#!/bin/bash
# ---------------------------------------------------------------------------
# Live teleop, both halves at once:
#   1. sim_teleop/live_sim_view.py   (env_isaaclab)  Isaac Sim window, robot mirrors the hand
#   2. sim_teleop/live_teleop.py     (cam)           camera window, tracking + keyboard + recording
#
# Usage:
#   ./run_live_teleop.sh <episode name> [extra live_teleop.py args...]
#   ./run_live_teleop.sh mug_pick --objects-from run_2026-05-15_17-55-22
#   ./run_live_teleop.sh t1 --source ~/Video2Sim2Real_main/keyframe_detection_test/run_2026-05-15_17-52-54
#
# Keys in the camera window: t engage/release the clutch (delta control - the robot starts
# at a known pose and follows your hand's RELATIVE motion; no camera calibration needed),
# SPACE start/stop recording, q quit, m mirror.
# After a successful stop the run is packaged as data/runs/teleop_<name>; add --replay to
# also run the validated recorder afterwards (waits for the sim window to close first):
#   ./run_live_teleop.sh mug_pick --replay
#
# --objects-from is forwarded to BOTH halves: the sim window shows those objects (welded,
# as a placement reference) and the packaged run includes them for the contact replay.
# ---------------------------------------------------------------------------
set -uo pipefail

CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
ENV_ISAACLAB="${TELEOP_ENV_ISAACLAB:-env_isaaclab}"
ENV_CAM="${TELEOP_ENV_CAM:-cam}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

NAME="${1:?usage: run_live_teleop.sh <episode name> [live_teleop.py args...]}"
shift

# pull out the flags this launcher itself understands; forward the rest
REPLAY=0
OBJECTS_FROM=""
INIT_SET=0
GLOVE=""            # "", "live", or a .mcap path (replay debug)
SOURCE_SET=0
VIEW_ARGS=()
TELEOP_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --replay) REPLAY=1; shift ;;
        --floating) VIEW_ARGS+=("$1"); TELEOP_ARGS+=("$1"); shift ;;
        --dynamic-objects) VIEW_ARGS+=("$1"); shift ;;   # viewer-only: graspable objects
        --glove) GLOVE="${2:-live}"; shift 2 ;;
        --source) SOURCE_SET=1; TELEOP_ARGS+=("$1" "$2"); shift 2 ;;
        --objects-from) OBJECTS_FROM="$2"; TELEOP_ARGS+=("$1" "$2"); shift 2 ;;
        --init-from) INIT_SET=1; TELEOP_ARGS+=("$1" "$2"); shift 2 ;;
        --scene-from) VIEW_ARGS+=("$1" "$2"); TELEOP_ARGS+=("$1" "$2"); shift 2 ;;
        *) TELEOP_ARGS+=("$1"); shift ;;
    esac
done
# Wuji-glove mode: forward --glove; without an explicit camera, run glove-only
# (wrist parked; add "--source realsense" to also track wrist translation)
if [ -n "$GLOVE" ]; then
    TELEOP_ARGS+=(--glove)
    [ $SOURCE_SET -eq 0 ] && TELEOP_ARGS+=(--source none)
fi
# the sim window shows the objects the packaged run will replay against, and (unless
# overridden) the robot starts at that run's frame-0 pose - sane relative to those objects
if [ -n "$OBJECTS_FROM" ]; then
    VIEW_ARGS+=(--scene-from "$OBJECTS_FROM")
    [ $INIT_SET -eq 0 ] && TELEOP_ARGS+=(--init-from "$OBJECTS_FROM")
fi

echo "[teleop] episode: $NAME"

# `conda run` / `setsid` spawn the real python as a CHILD, so killing the job's PID
# leaves the python orphaned (streaming forever, ignoring your Ctrl-C). Kill by the
# whole process GROUP, and pattern-match our own scripts as a belt-and-braces fallback.
cleanup() {
    # kill tracked PIDs and their groups (covers conda-run/setsid wrappers) ...
    for pid in "${VIEW_PID:-}" "${BRIDGE_PID:-}"; do
        [ -n "$pid" ] || continue
        kill -- -"$pid" 2>/dev/null      # process group
        kill "$pid" 2>/dev/null          # the wrapper itself
    done
    # ... then pattern-match our scripts as the reliable fallback (the real python is a
    # grandchild of conda-run/setsid, so PID tracking alone misses it). Substring match
    # catches both absolute- and relative-path invocations.
    pkill -f "sim_teleop/wuji_bridge.py" 2>/dev/null
    pkill -f "sim_teleop/live_sim_view.py" 2>/dev/null
}

BRIDGE_PID=""
if [ -n "$GLOVE" ]; then
    if [ "$GLOVE" = "live" ]; then
        echo "[teleop] starting the Wuji glove bridge (wuji-sdk env, live SDK)..."
        setsid "$CONDA_BIN" run --no-capture-output -n "${TELEOP_ENV_WUJI:-wuji-sdk}" \
            python "$PROJECT_ROOT/sim_teleop/wuji_bridge.py" &
    else
        echo "[teleop] starting the Wuji glove bridge (mcap replay: $GLOVE)..."
        setsid python3 "$PROJECT_ROOT/sim_teleop/wuji_bridge.py" --mcap "$GLOVE" --loop &
    fi
    BRIDGE_PID=$!
fi

echo "[teleop] starting the Isaac Sim viewer (takes ~1 min to boot)..."
setsid "$CONDA_BIN" run --no-capture-output -n "$ENV_ISAACLAB" \
    python "$PROJECT_ROOT/sim_teleop/live_sim_view.py" "${VIEW_ARGS[@]}" &
VIEW_PID=$!
# fire cleanup on normal exit AND on Ctrl-C / kill of the launcher itself
trap 'cleanup' EXIT
trap 'cleanup; exit 130' INT TERM

# start the camera only when the sim window is ready (the viewer binds the UDP port then)
PORT="${TELEOP_LIVE_PORT:-5556}"
for _ in $(seq 1 120); do
    ss -uln 2>/dev/null | grep -q ":$PORT " && break
    kill -0 $VIEW_PID 2>/dev/null || { echo "[teleop][ERROR] viewer died during startup"; exit 1; }
    sleep 1
done

echo "[teleop] starting the hand tracker (camera window)..."
"$CONDA_BIN" run --no-capture-output -n "$ENV_CAM" \
    python "$PROJECT_ROOT/sim_teleop/live_teleop.py" --name "$NAME" "${TELEOP_ARGS[@]}"
TRACKER_STATUS=$?

# the tracker sends "bye" on exit; give the viewer a moment to close by itself
for _ in $(seq 1 50); do
    kill -0 $VIEW_PID 2>/dev/null || break
    sleep 0.2
done
cleanup
wait $VIEW_PID 2>/dev/null
trap - EXIT INT TERM

RUN_NAME="$NAME"; case "$RUN_NAME" in teleop_*) ;; *) RUN_NAME="teleop_$NAME" ;; esac
if [ $TRACKER_STATUS -eq 0 ] && [ -d "$PROJECT_ROOT/data/runs/$RUN_NAME" ]; then
    if [ $REPLAY -eq 1 ]; then
        echo "[teleop] replaying with the validated recorder..."
        "$CONDA_BIN" run --no-capture-output -n "$ENV_ISAACLAB" \
            python "$PROJECT_ROOT/scripts/replay_trajectory.py" --run "$RUN_NAME"
    else
        echo "[teleop] episode packaged. Collect the data (force_controller format) with:"
        echo "[teleop]   conda run -n $ENV_ISAACLAB python scripts/replay_trajectory.py --run $RUN_NAME"
    fi
fi
