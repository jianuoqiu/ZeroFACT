"""Central paths + settings for the sim_teleop pipeline.

Everything points at installs that already exist and work on this machine (the
Dyn-HaMR + VIPE + retargeting stack under ~/Dyn-HaMR and ~/Video2Sim2Real_main).
Override any of them with the environment variable of the same name.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- this repo ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]           # ~/v2s2r_isaaclab
TELEOP_OUT_ROOT = PROJECT_ROOT / "outputs" / "sim_teleop"    # per-sequence work dirs
RUNS_DIR = PROJECT_ROOT / "data" / "runs"                    # packaged runs land here

# ---- external installs (the proven hand-capture / retargeting stack) ----
DYNHAMR_ROOT = Path(os.environ.get("DYNHAMR_ROOT", os.path.expanduser("~/Dyn-HaMR")))
V2S2R_MAIN = Path(os.environ.get("V2S2R_MAIN", os.path.expanduser("~/Video2Sim2Real_main")))
CONDA_BIN = os.environ.get("CONDA_BIN", os.path.expanduser("~/anaconda3/bin/conda"))

# conda env names (see ~/Dyn-HaMR/README_dynhamr_video2sim2real.md)
ENV_VIPE = os.environ.get("TELEOP_ENV_VIPE", "vipe")             # camera/depth preprocessing
ENV_DYNHAMR = os.environ.get("TELEOP_ENV_DYNHAMR", "dynhamr5090")  # Dyn-HaMR + MANO forward
#   NOTE: on this machine (RTX 5090 / sm_120) the plain "dynhamr" env silently
#   returns zeros - always use dynhamr5090.
ENV_RETARGET = os.environ.get("TELEOP_ENV_RETARGET", "vid2sim2real")  # mink/mujoco IK

# where run_hand_capture.sh keeps videos + Dyn-HaMR inputs (the layout Dyn-HaMR expects)
CAPTURE_ROOT = Path(os.environ.get("TELEOP_CAPTURE_ROOT",
                                   str(DYNHAMR_ROOT / "test_video2sim2real")))
VIPE_RESULTS = Path(os.environ.get("TELEOP_VIPE_RESULTS",
                                   str(DYNHAMR_ROOT / "third-party" / "vipe"
                                       / "vipe_results_video2sim2real")))
DYNHAMR_EXP = os.environ.get("TELEOP_DYNHAMR_EXP", "video2sim2real-test")

# ---- camera / calibration defaults (match the recorded pipeline) ----
# RealSense intrinsics used by the recordings (same values as depth_anchor.py and
# this repo's cam docs): fx, fy, cx, cy.
REAL_INTRINSICS = [385.4, 385.4, 317.4, 244.0]
ROBOT_TAG_INDEX = 23           # AprilTag marking the robot base in camera_frame_pose.json
WRIST_OFFSET_XYZ = (0.145, 0.0, 0.0)  # tag -> robot-base offset used by the converter

# ---- packaging defaults ----
DEFAULT_SCENE_DONOR = "run_2026-05-15_17-55-22"   # ingested run whose calib/objects to borrow
TARGET_FPS = 10.0              # the replay convention: 1 frame = 0.1 s

# ---- live teleop (sim_teleop/live_teleop.py + live_sim_view.py) ----
ENV_CAM = os.environ.get("TELEOP_ENV_CAM", "cam")   # has pyrealsense2 + mediapipe + mink/mujoco
# proven components reused in place:
CAMERA_READER_DIR = V2S2R_MAIN / "Scene_reconstruction" / "camera"        # BGR_Reader (RealSense)
HAND_DETECTOR_DIR = V2S2R_MAIN / "human_retargeting" / "example" / "vector_retargeting"
KINOVA_LEAP_URDF = (V2S2R_MAIN / "retargeting_kinova" / "assets"
                    / "kinova_leap_description" / "v12_vision.urdf")      # IK model (MinkSolver)
LIVE_UDP_ADDR = ("127.0.0.1", int(os.environ.get("TELEOP_LIVE_PORT", "5556")))
# wuji_bridge.py (glove process) -> live_teleop.py glove stream
WUJI_UDP_ADDR = ("127.0.0.1", int(os.environ.get("TELEOP_WUJI_PORT", "5557")))
LIVE_IK_ITERS = 20             # mink iterations per camera frame (offline retarget used 100)
# One Euro filtering (speed-adaptive: heavy at rest, light in motion). Measured need: with a
# front-facing camera the PnP wrist ORIENTATION wobbles ~9 deg/frame on a static hand, and the
# bracelet target's 12.7 cm offset lever turns that into a visibly shaking robot.
LIVE_POS_CUTOFF = 0.6          # Hz, keypoint positions at rest
LIVE_POS_BETA = 9.0            # cutoff gain per m/s of hand speed
LIVE_ROT_CUTOFF = 0.25         # Hz, wrist orientation at rest
LIVE_ROT_BETA = 3.0            # cutoff gain per unit/s of quaternion rate
# joint-rate caps applied to the IK output (rad/s, scaled by the actual frame dt). Measured:
# the redundant arm sometimes swings 20-40 deg in one frame on a sub-cm hand move (wrist-branch
# flips); capping at hardware-plausible speeds turns those into smooth reconfigurations and the
# capped pose warm-starts the next solve (trust region). Kinova Gen3 spec is ~1.2-1.4 rad/s.
LIVE_ARM_RATE = 1.2            # joint_1..7
LIVE_FINGER_RATE = 6.0         # LEAP joints
DEPTH_ANCHOR_JOINTS = [0, 5, 9, 13, 17]   # wrist + MCP knuckles (same as estimate_depth_scale)
