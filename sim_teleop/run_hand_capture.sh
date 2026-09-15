#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 1+2 of the sim_teleop pipeline: video -> world-frame MANO hand motion.
#
#   1. VIPE      (conda env: vipe)        camera pose + monocular depth
#   2. Dyn-HaMR  (conda env: dynhamr5090) world-grounded MANO hand sequence
#
# These are the exact command lines proven to work on this machine - see
# ~/Dyn-HaMR/README_dynhamr_video2sim2real.md. This script only wraps them.
#
# Usage:
#   ./run_hand_capture.sh <video.mp4> [TRACK_ID]
#   ./run_hand_capture.sh ~/videos/mug_pick.mp4          # TRACK_ID defaults to 001
#
# TRACK_ID: HaMeR's id for the hand track. With a single right hand in view it
# is 001. If Dyn-HaMR complains, list the detected ids with:
#   find $CAPTURE_ROOT/dynhamr/track_preds/<seq> -maxdepth 2 -type f | sort
#
# Output (what the next stage consumes):
#   $DYNHAMR_ROOT/outputs/logs/video-custom/$EXP/<seq>-<track>-shot-0-0--1/
#       smooth_fit/<...>_world_results.npz     <- MANO params, world frame
# and a convenience symlink:  outputs/sim_teleop/<seq>/dynhamr_result
# ---------------------------------------------------------------------------
set -euo pipefail

DYNHAMR_ROOT="${DYNHAMR_ROOT:-$HOME/Dyn-HaMR}"
CAPTURE_ROOT="${TELEOP_CAPTURE_ROOT:-$DYNHAMR_ROOT/test_video2sim2real}"
VIPE_RESULTS="${TELEOP_VIPE_RESULTS:-$DYNHAMR_ROOT/third-party/vipe/vipe_results_video2sim2real}"
EXP="${TELEOP_DYNHAMR_EXP:-video2sim2real-test}"
ENV_VIPE="${TELEOP_ENV_VIPE:-vipe}"
ENV_DYNHAMR="${TELEOP_ENV_DYNHAMR:-dynhamr5090}"   # NOT "dynhamr": returns zeros on the 5090
CONDA_BIN="${CONDA_BIN:-$HOME/anaconda3/bin/conda}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

VIDEO="${1:?usage: run_hand_capture.sh <video.mp4> [TRACK_ID]}"
TRACK_ID="${2:-001}"
SEQ="$(basename "$VIDEO" .mp4)"

echo "[capture] seq=$SEQ  track=$TRACK_ID"
echo "[capture] video: $VIDEO"

# ---- put the video where Dyn-HaMR expects it: $CAPTURE_ROOT/videos/<seq>.mp4 ----
mkdir -p "$CAPTURE_ROOT/videos"
if [ ! -f "$CAPTURE_ROOT/videos/$SEQ.mp4" ]; then
    cp -v "$VIDEO" "$CAPTURE_ROOT/videos/$SEQ.mp4"
fi

# ---- 1. VIPE: camera pose + depth preprocessing ----
if [ -d "$VIPE_RESULTS/$SEQ" ]; then
    echo "[capture] VIPE output exists, skipping ($VIPE_RESULTS/$SEQ)"
else
    echo "[capture] running VIPE (env $ENV_VIPE) ..."
    ( cd "$DYNHAMR_ROOT/third-party/vipe" && \
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
      "$CONDA_BIN" run --no-capture-output -n "$ENV_VIPE" \
        vipe infer "$CAPTURE_ROOT/videos/$SEQ.mp4" \
        --output "$VIPE_RESULTS" ) 2>&1 | tee "$DYNHAMR_ROOT/vipe_${SEQ}.log"
fi

# ---- 2. Dyn-HaMR: world-grounded MANO optimization ----
OUT="$DYNHAMR_ROOT/outputs/logs/video-custom/$EXP/${SEQ}-${TRACK_ID}-shot-0-0--1"
if compgen -G "$OUT/smooth_fit/*_world_results.npz" > /dev/null; then
    echo "[capture] Dyn-HaMR output exists, skipping ($OUT)"
else
    echo "[capture] running Dyn-HaMR (env $ENV_DYNHAMR) ..."
    ( cd "$DYNHAMR_ROOT/dyn-hamr" && \
      HYDRA_FULL_ERROR=1 \
      "$CONDA_BIN" run --no-capture-output -n "$ENV_DYNHAMR" \
        python run_opt.py \
          data=video_vipe \
          run_opt=True \
          run_vis=True \
          data.seq="$SEQ" \
          data.track_ids="$TRACK_ID" \
          is_static=False \
          data.root="$CAPTURE_ROOT" \
          data.vipe_dir="$VIPE_RESULTS" \
          paths.DATA_DIR="$DYNHAMR_ROOT/_DATA/data" \
          exp_name="$EXP" ) 2>&1 | tee "$DYNHAMR_ROOT/run_opt_${SEQ}.log"
fi

if ! compgen -G "$OUT/smooth_fit/*_world_results.npz" > /dev/null; then
    echo "[capture][ERROR] no smooth_fit/*_world_results.npz under $OUT"
    echo "  - wrong TRACK_ID? list ids: find $CAPTURE_ROOT/dynhamr/track_preds/$SEQ -type f | sort"
    exit 1
fi

# ---- convenience symlink for the next stage ----
WORK="$PROJECT_ROOT/outputs/sim_teleop/$SEQ"
mkdir -p "$WORK"
ln -sfn "$OUT" "$WORK/dynhamr_result"
echo "[capture] DONE. Dyn-HaMR result: $OUT"
echo "[capture] next: python sim_teleop/make_hand_traj.py --seq $SEQ --camera-pose-json <calib>"
