#!/usr/bin/env python3
"""Live teleop operator console: camera -> hand tracking -> IK -> sim, with key-gated recording.

Runs in the ``cam`` conda env (pyrealsense2 + mediapipe + mink/mujoco).  Shows the camera
feed with the detected hand skeleton, streams the retargeted Kinova+LEAP joint targets over
UDP to ``sim_teleop/live_sim_view.py`` (the Isaac Lab window), and records a trajectory
between two key presses.

Default is **delta (clutch) control** - absolute hand placement does not matter: the robot
starts at a proven initial pose (frame 0 of ``--init-from``, default the scene donor) and,
once you engage, follows your hand's RELATIVE motion from wherever it stands.  No camera
calibration is needed and the camera can sit anywhere that sees your hand; you close the
loop by watching the sim window.  Engaging never makes the robot jump: the current hand
pose is bound to the robot's current wrist pose (``M = B @ H0^-1``, one rigid map).

    robot sits at its start pose     <- move your hand into the camera view
    [t]      engage the clutch       <- robot follows your hand's relative motion
    [t]      release (any time)      <- robot holds; reposition your hand, [t] again
                                        (workspace extension, like lifting a mouse)
    [SPACE]  start recording         <- episode frame 0 = the pose at this moment
    [SPACE]  stop                    <- trajectory saved + packaged as data/runs/teleop_<name>
    [q/ESC]  quit (recording, if any, is DISCARDED)
    [m]      mirror the preview horizontally (display only)

Tip: engage with your palm roughly matching the sim robot's palm direction (palm down,
fingers away from the robot base) and motion directions will feel natural immediately.
``--mode absolute`` restores the calibrated absolute mapping (requires the camera to match
``--calib``; useful when the hand must land exactly where a real-table recording was made).

The recorded trajectory is written in the exact stage-4 contract
(``{"traj": [{"robot_cfg": {...}}]}``) and packaged with the existing ``package_run.py``,
so the data-collection step stays the validated recorder:

    python scripts/replay_trajectory.py --run teleop_<name>     # -> replay_data.npz
    (the replay_data.npz episode is what force_controller consumes)

How a hand pose is measured (all proven pieces, reused):
  - MediaPipe hands: 21 landmarks (2D px + a metric wrist-local 3D shape).
  - Pose by ``cv2.solvePnP`` of that shape against its own 2D landmarks with the real
    intrinsics (all 21 correspondences, previous-frame warm start) - chirality-safe, so
    the palm cannot mirror-flip the way a coplanar-anchor fit could.
  - Metric anchoring from RealSense aligned depth at the wrist + 4 MCP knuckles (the
    anchor joints ``estimate_depth_scale.py`` also trusts): the hand's position is
    re-anchored to the sensor every frame, the hand-size factor is slow-locked - so
    depth comes from the sensor, never from PnP's noisy monocular z.
  - camera -> robot base: delta clutch map by default; in ``--mode absolute`` the
    AprilTag formula ``dynhamr_to_skeleton.world_to_robot_base_apriltag`` uses.
  - keypoints -> joints with the same MinkSolver + target frames as
    ``retargeting_kinova.retarget_leap`` (warm-started, fewer iterations per frame).

Typical use (same table + camera setup as the calibration donor):
    ./sim_teleop/run_live_teleop.sh my_episode                  # starts both processes
or by hand:
    conda run -n env_isaaclab python sim_teleop/live_sim_view.py &
    conda run -n cam python sim_teleop/live_teleop.py --name my_episode

Test without a camera (replays a recorded RGB-D folder as if it were live):
    conda run -n cam python sim_teleop/live_teleop.py --name t1 \
        --source ~/Video2Sim2Real_main/keyframe_detection_test/run_2026-05-15_17-52-54 \
        --auto-record --no-window
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sim_teleop.teleop_config import (  # noqa: E402
    CAMERA_READER_DIR,
    DEFAULT_SCENE_DONOR,
    DEPTH_ANCHOR_JOINTS,
    HAND_DETECTOR_DIR,
    KINOVA_LEAP_URDF,
    LIVE_ARM_RATE,
    LIVE_FINGER_RATE,
    LIVE_IK_ITERS,
    LIVE_POS_BETA,
    LIVE_POS_CUTOFF,
    LIVE_ROT_BETA,
    LIVE_ROT_CUTOFF,
    LIVE_UDP_ADDR,
    ROBOT_TAG_INDEX,
    RUNS_DIR,
    TARGET_FPS,
    TELEOP_OUT_ROOT,
    V2S2R_MAIN,
    WRIST_OFFSET_XYZ,
)

sys.path.insert(0, str(V2S2R_MAIN))            # retargeting_kinova package
sys.path.insert(0, str(HAND_DETECTOR_DIR))     # single_hand_detector
sys.path.insert(0, str(CAMERA_READER_DIR))     # camera_reader (BGR_Reader)


# --------------------------------------------------------------------------------------
# IK target frames - copied VERBATIM from retargeting_kinova/retarget_leap.py (importing
# that module would drag in pybullet + tyro, which the cam env does not have).  Any change
# there must be mirrored here, or live and offline retargeting drift apart.
# --------------------------------------------------------------------------------------
def _wrist_pose_from_pts21_world(pts: np.ndarray):
    if pts is None or pts.shape != (21, 3):
        return None

    def _norm(v, eps=1e-12):
        n = np.linalg.norm(v)
        return (v / (n + eps), n)

    w = pts[0]
    p9 = pts[9]
    p5 = pts[5]
    p13 = pts[13]

    z, nz = _norm(p9 - w)
    if nz < 1e-9:
        return None

    n_plane, nn = _norm(np.cross(p13 - p5, w - p5))
    if nn < 1e-9:
        return None

    x, nx = _norm(np.cross(n_plane, z))
    if nx < 1e-9:
        v = p13 - p5
        v = v - np.dot(v, n_plane) * n_plane
        v = v - np.dot(v, z) * z
        x, nx = _norm(v)
        if nx < 1e-9:
            return None

    y, ny = _norm(np.cross(z, x))
    if ny < 1e-9:
        return None

    if np.dot(y, n_plane) < 0.0:
        x = -x
        y = -y

    x = -x
    z = -z
    y = np.cross(z, x)
    y, _ = _norm(y)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = w.astype(np.float32)
    T[:3, 3] += -0.09 * T[:3, 1]
    T[:3, 3] += 0.09 * T[:3, 2]
    return T


def _finger_poses_from_mano_world(pts: np.ndarray) -> dict:
    assert pts.shape == (21, 3)
    wrist = pts[0]
    idx_tip, ring_tip = pts[8], pts[16]
    palm_n = np.cross(idx_tip - wrist, ring_tip - wrist)
    palm_n = palm_n / (np.linalg.norm(palm_n) + 1e-12)

    tips_prev = {
        "thumb": (4, 3),
        "index": (8, 7),
        "middle": (12, 11),
        "ring": (16, 15),
    }

    res = {}
    for name, (tip_i, prev_i) in tips_prev.items():
        tip = pts[tip_i]
        prev = pts[prev_i]

        z = tip - prev
        z = z / (np.linalg.norm(z) + 1e-12)
        x = np.cross(palm_n, z)
        x = x / (np.linalg.norm(x) + 1e-12)
        y = np.cross(x, z)
        y = y / (np.linalg.norm(y) + 1e-12)

        if name == "thumb":
            x, y, z = x, -z, -y
        else:
            x, y, z = x, z, -y

        T = np.eye(4, dtype=np.float32)
        T[:3, 0] = x
        T[:3, 1] = y
        T[:3, 2] = z
        T[:3, 3] = tip.astype(np.float32)
        res[name] = T
    return res


def prepare_x_display() -> None:
    """Point DISPLAY at an X server this user can open windows on (shared box: ~/.bashrc
    exports another user's display). Same probe the Isaac scripts use, minus the Vulkan
    requirement - a cv2 window only needs a plain X connection."""
    import getpass
    import os

    from zerofact.runtime import _user_displays, _x_display_ok

    override = os.environ.get("V2S2R_DISPLAY")
    if override:
        os.environ["DISPLAY"] = override
        return
    if _x_display_ok():
        return
    current = os.environ.get("DISPLAY")
    for cand in _user_displays(getpass.getuser()):
        if cand != current and _x_display_ok(cand):
            os.environ["DISPLAY"] = cand
            print(f"[live] DISPLAY={current!r} is not usable; switched to {cand!r} (yours).")
            return
    print(f"[live][WARN] no usable X display found (DISPLAY={current!r}) - "
          "the preview window will fail; consider --no-window or V2S2R_DISPLAY=:N.")


# --------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------
class RealSenseSource:
    """Live RGB-D via the recording pipeline's own BGR_Reader (aligned depth, real intrinsics)."""

    def __init__(self, width=640, height=480, fps=30, serial=None):
        from camera_reader import BGR_Reader

        self.reader = BGR_Reader(width=width, height=height, fps=fps,
                                 visualize=False, serial_number=serial, depth=True)
        self.reader.start()
        self.intrinsics = self.reader.get_intrinsics()          # fx, fy, cx, cy
        self.depth_to_m = float(self.reader.get_depth_scale())  # raw depth unit -> meters
        self.fps = fps

    def read(self):
        return self.reader.read()          # (bgr, depth_raw) or (None, None)

    def close(self):
        self.reader.end()


class FolderSource:
    """Replay a recorded session (``image/N.png`` + ``depth/N.png``) as if it were live."""

    def __init__(self, folder: Path, fps: float = 10.0, pace: bool = True):
        import cv2

        self.cv2 = cv2
        self.folder = Path(folder)
        img_dir = self.folder / "image"
        self.depth_dir = self.folder / "depth"
        if not img_dir.is_dir() or not self.depth_dir.is_dir():
            raise FileNotFoundError(f"{folder} needs image/ and depth/ sub-folders")
        self.frames = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
        if not self.frames:
            raise FileNotFoundError(f"no PNGs under {img_dir}")

        cam_params = self.folder / "cam_params.txt"
        if cam_params.is_file():
            from zerofact.scene_spec import load_camera_intrinsics
            self.intrinsics = list(load_camera_intrinsics(cam_params))
        else:
            from sim_teleop.teleop_config import REAL_INTRINSICS
            self.intrinsics = list(REAL_INTRINSICS)
            print(f"[live][WARN] {cam_params} missing - using default intrinsics")
        self.depth_to_m = 0.001            # recordings store uint16 millimetres
        self.fps = fps
        self.pace = pace
        self.idx = 0
        self._next_t = None

    def read(self):
        if self.idx >= len(self.frames):
            return None, None
        if self.pace:
            now = time.perf_counter()
            if self._next_t is None:
                self._next_t = now
            if self._next_t > now:
                time.sleep(self._next_t - now)
            self._next_t += 1.0 / self.fps
        img_path = self.frames[self.idx]
        bgr = self.cv2.imread(str(img_path), self.cv2.IMREAD_COLOR)
        depth = self.cv2.imread(str(self.depth_dir / img_path.name), self.cv2.IMREAD_UNCHANGED)
        self.idx += 1
        return bgr, depth

    def close(self):
        pass


# --------------------------------------------------------------------------------------
# Keypoint lifting: MediaPipe shape + sensor depth anchors -> metric camera-frame joints
# --------------------------------------------------------------------------------------
def sample_anchor_points(depth_raw: np.ndarray, px: np.ndarray, intr, depth_to_m: float,
                         joints=DEPTH_ANCHOR_JOINTS, patch: int = 2,
                         zmin: float = 0.15, zmax: float = 2.0):
    """Back-project the anchor joints using the aligned depth image.

    Same sampling recipe as estimate_depth_scale.py: a (2*patch+1)^2 window, median of the
    non-zero readings, gated to a plausible range.  Returns ``[K,3]`` camera-frame points
    and a validity mask over the anchor joints.
    """
    fx, fy, cx, cy = intr
    h, w = depth_raw.shape[:2]
    pts = np.zeros((len(joints), 3))
    valid = np.zeros(len(joints), dtype=bool)
    for k, j in enumerate(joints):
        u, v = int(round(px[j, 0])), int(round(px[j, 1]))
        if not (patch <= u < w - patch and patch <= v < h - patch):
            continue
        win = depth_raw[v - patch:v + patch + 1, u - patch:u + patch + 1].astype(np.float64)
        win = win[win > 0] * depth_to_m
        if win.size < patch:
            continue
        z = float(np.median(win))
        if not (zmin < z < zmax):
            continue
        pts[k] = ((u - cx) * z / fx, (v - cy) * z / fy, z)
        valid[k] = True
    return pts, valid


class OneEuro:
    """One Euro filter (Casiez et al. 2012): cutoff = min_cutoff + beta * |velocity|.

    At rest it smooths hard (estimator wobble vanishes); during real motion the cutoff
    opens and lag stays small. ``x`` may be a vector; for a point set pass ``speed`` as
    the mean per-point speed so the adaptation reflects physical hand speed.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float, speed: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / dt
        ad = self._alpha(self.d_cutoff, dt)
        self.dx_prev = ad * dx + (1.0 - ad) * self.dx_prev
        if speed is None:
            speed = float(np.linalg.norm(self.dx_prev))
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev

    def reset(self) -> None:
        self.x_prev = self.dx_prev = None


class HandLifter:
    """MediaPipe detection + PnP + depth ray-scaling -> (21,3) camera-frame joints.

    Pose: ``cv2.solvePnP`` of MediaPipe's metric hand shape (world landmarks) against its
    own 2D landmarks with the real intrinsics - all 21 correspondences, chirality-safe,
    warm-started from the previous frame so the planar (flat-hand) two-fold ambiguity
    cannot flip the palm.  (The previous 5-anchor similarity fit was under-constrained
    exactly there: wrist + MCP knuckles are nearly coplanar, so depth noise could mirror
    the palm - thumb landing on the pinky side, robot fingers crossing.)

    Metric: position is re-anchored to the SENSOR every frame - ``t = mean(anchor
    depth points) - s * R @ mean(anchor shape points)`` - so per-frame depth comes from
    the RealSense (stable), never from PnP's monocular z (noisy).  Only the hand-size
    factor ``s`` (person vs MediaPipe prior, a constant) is slow-EMA-locked, estimated by
    1D least squares against the anchors with R held fixed.  On a depth dropout the
    position holds while orientation and fingers keep tracking.

    The MediaPipe handedness label is NOT filtered on (it is pure noise: "Left" on every
    frame of a right-hand test video) - but the world-landmark GEOMETRY does flip to a
    mirrored left hand on some frames, so chirality is checked per frame from the
    landmark determinant and mirrored shapes are corrected. Use your right hand.
    """

    SCALE_EMA = 0.05          # hand size is constant: lock onto it slowly
    MAX_REPROJ_PX = 20.0      # mean 2D residual above this = unreliable detection, hold
    FLIP_REJECT_DEG = 60.0    # a single-frame rotation this large = planar mirror branch
    WRIST_FLIP_DEG = 45.0     # output palm-frame jump floor treated as an estimation flip
    WRIST_FLIP_RATE = 600.0   # deg/s: fastest believable deliberate wrist turn
    WRIST_FLIP_HOLD = 15      # consecutive flip frames before conceding it is real motion

    def __init__(self, intr, depth_to_m: float, smooth: float = 1.0, min_conf: float = 0.6):
        import mediapipe as mp
        from single_hand_detector import SingleHandDetector

        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                              min_detection_confidence=min_conf,
                                              min_tracking_confidence=min_conf)
        self.draw = SingleHandDetector.draw_skeleton_on_image
        self.intr = intr
        fx, fy, cx, cy = intr
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.depth_to_m = depth_to_m
        # One Euro filters: near-static wobble is crushed, real motion passes with little
        # lag. `smooth` scales the resting cutoffs (>1 = smoother). The point filter's
        # beta is normalised by sqrt(21) so its speed term equals physical hand speed for
        # rigid motion; measured need: 9 deg/frame wrist-orientation wobble on a STATIC
        # hand (front camera) shook the whole robot through the bracelet-offset lever.
        self._f_pos = OneEuro(LIVE_POS_CUTOFF / smooth, LIVE_POS_BETA / np.sqrt(21.0))
        self._f_rot = OneEuro(LIVE_ROT_CUTOFF / smooth, LIVE_ROT_BETA)
        self._scale = None                 # slow-EMA hand-size factor
        self._t = None                     # last sensor-anchored translation
        self._q_smooth = None              # filtered orientation (xyzw)
        self._last_t = None                # wall time of the previous lift (rotation dt)
        self._rvec = self._tvec = None     # previous PnP pose (branch continuity)
        self._lost = False
        self._H_acc = None                 # last ACCEPTED output palm frame (flip gate)
        self._t_acc = 0.0
        self._flip_run = 0
        self._fit_ema = None               # EMA of the anchor fit on accepted frames
        self._rp_ema = None                # EMA of the PnP reprojection error (px)
        self.anchor_residual = 0.0         # fit error vs the depth anchors (m): the honest
                                           # view-quality signal, > ~1.5 cm = unusable view
        self.reproj_px = 0.0
        self.status = "no hand"

    def quality(self) -> tuple[str, str | None]:
        """Live estimation-health verdict: is the 3D hand estimate trustworthy RIGHT NOW?

        Calibrated on the known-good rig recording (reproj median 4.3 px, fit 1.27 cm)
        vs real desk sessions that produced unusable data (reproj 7-10 px, fit 1.5-2.1):
        reprojection error is MediaPipe-internal (2D/3D self-consistency, no depth
        involved) - when IT is high the LANDMARKS are bad and the cure is lighting/
        background/motion, not camera pose; the anchor fit adds the depth cross-check.
        """
        rp = self._rp_ema if self._rp_ema is not None else 99.0
        ft = (self._fit_ema if self._fit_ema is not None else 0.099) * 100
        if rp < 6.0 and ft < 1.5:
            return "GOOD", None
        if rp < 10.0 and ft < 2.0:
            return "OK", None
        if rp >= 10.0:
            return "BAD", "estimate unstable: light the scene EVENLY (no blur/washout on the hand), slower moves"
        return "BAD", "depth disagrees: face palm/back to camera, hand 40-60 cm away"

    def lift(self, rgb: np.ndarray, depth_raw: np.ndarray):
        """Returns ((21,3) camera-frame joints or None, keypoint_2d for drawing, anchor px+mask)."""
        import cv2

        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self.status = "no hand"
            self._lost = True
            return None, None, None
        kp2d = res.multi_hand_landmarks[0]
        h, w = rgb.shape[:2]
        px = np.array([[p.x * w, p.y * h] for p in kp2d.landmark])
        shape = np.array([[p.x, p.y, p.z] for p in
                          res.multi_hand_world_landmarks[0].landmark])

        # NOTE on chirality: the handedness label is noise ("Left" on every frame of a
        # right-hand test video) and is ignored. A det-sign or depth-anchor chirality
        # guard was tried and MEASURED HARMFUL: MediaPipe flattens the thumb toward the
        # palm plane (normalized det +-0.1 vs 0.25..0.49 on MANO truth) so the sign is
        # ill-conditioned, and the wrist+MCP anchors are near-planar (mirror-blind).
        # Procrustes against Dyn-HaMR truth: raw geometry right-handed on 131/160
        # frames, mirrored ~18% with sub-cm effect - minor noise, not correctable live.

        # ---- pose: PnP of the hand shape against its own 2D landmarks ----
        if self._rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(shape, px, self.K, None,
                                          rvec=self._rvec.copy(), tvec=self._tvec.copy(),
                                          useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
            candidates = [(rvec, tvec)] if ok else []
        else:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(shape, px, self.K, None,
                                                     flags=cv2.SOLVEPNP_SQPNP)
            candidates = list(zip(rvecs, tvecs))
        best = None
        for rv, tv in candidates:
            proj, _ = cv2.projectPoints(shape, rv, tv, self.K, None)
            err = float(np.linalg.norm(proj.reshape(-1, 2) - px, axis=1).mean())
            if best is None or err < best[0]:
                best = (err, rv, tv)
        if best is None or best[0] > self.MAX_REPROJ_PX or float(np.ravel(best[2])[2]) <= 0.0:
            self.status = "pose unreliable" + (f" ({best[0]:.0f}px)" if best else "")
            self._rvec = None              # re-initialise from scratch next frame
            self._lost = True
            return None, kp2d, None
        reproj_err, self._rvec, self._tvec = best
        self.reproj_px = reproj_err
        self._rp_ema = reproj_err if self._rp_ema is None else (
            self._rp_ema + 0.2 * (reproj_err - self._rp_ema))
        R, _ = cv2.Rodrigues(self._rvec)

        # ---- rotation filtering (One Euro on the quaternion) + branch-flip rejection ----
        # A near-planar or foreshortened hand leaves PnP's orientation weakly constrained:
        # measured wobble up to ~9 deg/frame on a STATIC hand. The speed-adaptive filter
        # freezes that at rest yet follows deliberate turns; a >60 deg single-frame change
        # is the planar mirror branch - hold the previous orientation instead.
        from scipy.spatial.transform import Rotation as Rot

        now = time.perf_counter()
        dt = float(np.clip(now - self._last_t, 0.02, 0.25)) if self._last_t else 0.1
        self._last_t = now

        q_new = Rot.from_matrix(R).as_quat()
        if self._q_smooth is None or self._lost:
            self._f_rot.reset()
            self._q_smooth = self._f_rot(q_new, dt)
        else:
            if float(np.dot(q_new, self._q_smooth)) < 0.0:
                q_new = -q_new
            ang = 2.0 * np.degrees(np.arccos(
                np.clip(abs(float(np.dot(q_new, self._q_smooth))), 0.0, 1.0)))
            if ang <= self.FLIP_REJECT_DEG:
                q = self._f_rot(q_new, dt)
                self._q_smooth = q / np.linalg.norm(q)
            # else: keep the previous filtered rotation for this frame
        R = Rot.from_quat(self._q_smooth).as_matrix()

        # ---- metric anchoring: size (slow-locked) + per-frame position from the sensor ----
        anchors, valid = sample_anchor_points(depth_raw, px, self.intr, self.depth_to_m)
        n_ok = int(valid.sum())
        glitch = False
        if n_ok:
            sel = np.array(DEPTH_ANCHOR_JOINTS)[valid]
            w_bar = shape[sel].mean(axis=0)
            a_bar = anchors[valid].mean(axis=0)
            if n_ok >= 3:
                X = (shape[sel] - w_bar) @ R.T
                Y = anchors[valid] - a_bar
                s_raw = float((X * Y).sum() / max((X * X).sum(), 1e-9))
                if 0.5 < s_raw < 2.0:
                    self._scale = s_raw if self._scale is None else (
                        self._scale + self.SCALE_EMA * (s_raw - self._scale))
            if self._scale is not None:
                t_new = a_bar - self._scale * (R @ w_bar)
                # a depth glitch teleports the anchors; a real hand stays under ~2 m/s
                if (self._t is not None and not self._lost
                        and np.linalg.norm(t_new - self._t) > 2.0 * dt):
                    glitch = True
                else:
                    self._t = t_new
        if self._scale is None or self._t is None:
            self.status = "no depth yet"
            return None, kp2d, (px, valid)
        self.status = (f"depth ok ({n_ok}/5 anchors)" if n_ok else "depth HOLD") \
            + f"  reproj {reproj_err:.0f}px"
        if glitch:
            self.status += "  | depth glitch (position held)"

        cam = self._scale * shape @ R.T + self._t

        # view-quality check. MediaPipe HALLUCINATES plausible landmarks when the hand is
        # seen edge-on (a side view: fingers occlude each other), so no image-space metric
        # is trustworthy - but the hallucinated pixels then sample depth off inconsistent
        # surfaces, and the anchor fit error jumps from millimetres to centimetres. That
        # residual is the honest "is this view usable?" signal.
        if n_ok:
            self.anchor_residual = float(np.linalg.norm(
                cam[np.array(DEPTH_ANCHOR_JOINTS)[valid]] - anchors[valid], axis=1).mean())
            self.status += f"  fit {self.anchor_residual * 100:.1f}cm"
            if self.anchor_residual > 0.015:
                self.status += "  | POOR VIEW - face your palm or its back to the camera!"

        # ---- wrist-frame flip gate (the guard that actually matters downstream) ----
        # PnP's own flip-reject protects R, but MediaPipe can flip its 3D INTERPRETATION
        # of the hand (palm-toward vs palm-away) between frames: the world shape rotates
        # internally and the OUTPUT points flip even under a held R. Measured on a real
        # desk session: 85-118 deg single-frame jumps of the derived palm frame with the
        # wrist translating ~1 cm and ZERO dropouts - impossible motion (>850 deg/s),
        # and the finger targets ride that frame, so every flip slams the fingers into
        # their limits. Gate on the derived frame: hold the previous points for jumps
        # faster than a human wrist; accept a PERSISTENT new orientation only when the
        # depth anchors fit it clearly better (we were on a wrong branch) or it survives
        # WRIST_FLIP_HOLD frames (real, very fast turn).
        H_now = _wrist_pose_from_pts21_world(cam.astype(np.float32))
        if H_now is not None and self._H_acc is not None and not self._lost:
            from scipy.spatial.transform import Rotation as Rot

            gap = float(np.clip(now - self._t_acc, dt, 1.0))
            dang = float(np.degrees(np.linalg.norm(
                Rot.from_matrix(self._H_acc[:3, :3].T @ H_now[:3, :3]).as_rotvec())))
            if dang > max(self.WRIST_FLIP_DEG, self.WRIST_FLIP_RATE * gap):
                self._flip_run += 1
                fit_better = (n_ok and self._fit_ema is not None
                              and self.anchor_residual < self._fit_ema - 0.005)
                if not (fit_better or self._flip_run > self.WRIST_FLIP_HOLD):
                    self.status += f"  | WRIST FLIP rejected ({dang:.0f} deg) - holding"
                    return None, kp2d, (px, valid)
            else:
                self._flip_run = 0
        if H_now is not None:
            self._H_acc, self._t_acc, self._flip_run = H_now, now, 0
        if n_ok:
            self._fit_ema = (self.anchor_residual if self._fit_ema is None else
                             self._fit_ema + 0.1 * (self.anchor_residual - self._fit_ema))

        if self._lost:
            self._lost = False
            self._f_pos.reset()            # fresh start after a dropout (no stale lag)
        out = self._f_pos(cam, dt)
        return out.copy(), kp2d, (px, valid)


# --------------------------------------------------------------------------------------
# camera frame -> robot base frame (AprilTag calibration)
# --------------------------------------------------------------------------------------
class TagTransform:
    """The exact formula of dynhamr_to_skeleton.world_to_robot_base_apriltag (proven path):
    ``pts_robot = (pts_cam - t_tag) @ R_tag.T + offset`` with R_tag/t_tag the tag-23 entry.
    """

    def __init__(self, camera_pose_json: Path, tag_index: int = ROBOT_TAG_INDEX,
                 offset_xyz=WRIST_OFFSET_XYZ):
        from scipy.spatial.transform import Rotation as R

        tags = json.loads(Path(camera_pose_json).read_text())
        tag = next((e for e in tags if e.get("tag_index") == tag_index), None)
        if tag is None:
            raise ValueError(f"tag_index {tag_index} not in {camera_pose_json}")
        self.t = np.asarray(tag["position(m)"], dtype=np.float64)
        self.R = R.from_euler("XYZ", tag["orientation_deg_XYZ(deg)"], degrees=True).as_matrix()
        self.offset = np.asarray(offset_xyz, dtype=np.float64)
        self.source = str(camera_pose_json)

    def __call__(self, pts_cam: np.ndarray) -> np.ndarray:
        return (pts_cam - self.t) @ self.R.T + self.offset


# Camera axes (x = image right, y = image down, z = into the scene) -> robot base axes
# (x forward/away from base, y left, z up), applied to motion DELTAS in the view-preset
# modes. Both are proper rotations (det +1).
VIEW_AXIS_MAPS = {
    # camera FACING the operator: push away from yourself = robot forward, up = up,
    # your right (= image left) = robot right
    "front": np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0]]),
    # camera above/behind looking down at the workspace (the original rig): image up =
    # robot forward - same convention as dynhamr_to_skeleton's approximate CAM_TO_ROBOT
    "top": np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]]),
}


class DeltaMapper:
    """Delta (clutch) control: bind the hand at engage time to the robot's CURRENT
    bracelet pose ``B0``, then drive the robot with the hand's relative motion.

    view="front"/"top" (recommended): translation and rotation DELTAS are mapped through
    the fixed camera->robot axis preset above, so steering directions are predictable
    ("push away = robot forward") no matter how the palm was oriented at engage. The
    wrist rotates in place when your hand rotates in place; finger articulation rides
    along as the local hand shape re-attached at the target wrist frame.

    view="hand": the original single rigid map ``M = B0 @ H0^-1`` - works with ANY
    camera placement, but motion directions then depend on matching your palm to the
    robot's palm at engage (measured to be the main steering trap in practice).

    In every mode the map is exact at the engage instant, so engaging never jumps.
    """

    def __init__(self, view: str = "front"):
        if view != "hand" and view not in VIEW_AXIS_MAPS:
            raise ValueError(f"unknown view {view!r} (hand/front/top)")
        self.view = view
        self.F = VIEW_AXIS_MAPS.get(view)
        self._M = None                     # 'hand' mode: the constant rigid map
        self._H0 = None                    # preset modes: hand pose at engage
        self._B0 = None                    # robot bracelet pose at engage
        self._H_last = None

    @property
    def ready(self) -> bool:
        return self._M is not None or self._H0 is not None

    def anchor(self, pts_cam: np.ndarray, bracelet_T: np.ndarray) -> bool:
        H0 = _wrist_pose_from_pts21_world(pts_cam.astype(np.float32))
        if H0 is None:
            return False
        H0 = H0.astype(np.float64)
        B0 = bracelet_T.astype(np.float64)
        if self.view == "hand":
            self._M = B0 @ np.linalg.inv(H0)
        else:
            self._H0, self._B0, self._H_last = H0, B0, H0
            c0 = pts_cam[0].astype(np.float64)               # wrist keypoint at engage
            self._c0 = c0
            self._l_w = (c0 - H0[:3, 3]) @ H0[:3, :3]        # its (constant) local coords
            self._r0 = self._B0[:3, :3] @ self._l_w + self._B0[:3, 3]   # its mapped position
        return True

    def __call__(self, pts_cam: np.ndarray) -> np.ndarray:
        pts = pts_cam.astype(np.float64)
        if self.view == "hand":
            return pts @ self._M[:3, :3].T + self._M[:3, 3]
        H = _wrist_pose_from_pts21_world(pts.astype(np.float32))
        H = self._H_last if H is None else H.astype(np.float64)
        self._H_last = H
        # deltas are referenced to the WRIST KEYPOINT: translate your wrist -> the robot
        # wrist translates by the axis-mapped delta; rotate your hand in place -> the
        # robot hand rotates in place about its wrist (not orbiting the bracelet origin)
        dR = self.F @ (H[:3, :3] @ self._H0[:3, :3].T) @ self.F.T
        r = self._r0 + self.F @ (pts[0] - self._c0)          # mapped wrist-point position
        W_R = dR @ self._B0[:3, :3]
        W_t = r - W_R @ self._l_w
        local = (pts - H[:3, 3]) @ H[:3, :3]                 # hand shape in the wrist frame
        return local @ W_R.T + W_t


# --------------------------------------------------------------------------------------
# Wuji glove input (skeleton + palm IMU via wuji_bridge.py) - replaces vision entirely
# --------------------------------------------------------------------------------------
class GloveReceiver:
    """Non-blocking reader of wuji_bridge.py's UDP stream (see that file for the wire
    format). Keeps only the newest frame; tracks rate/staleness for the EST badge."""

    def __init__(self, addr=None):
        import socket

        from sim_teleop.teleop_config import WUJI_UDP_ADDR

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(addr or WUJI_UDP_ADDR)
        self.sock.setblocking(False)
        self.latest = None                  # {"skel": (21,3) f64, "quat": (4,) xyzw}
        self.alive = True                   # False once the bridge says bye
        self._t_last = None                 # wall time of the newest frame
        self._rate = 0.0                    # EMA of the arrival rate
        self._seq = None
        self.dropped = 0

    def poll(self):
        """Drain the socket; returns the newest frame dict or None if nothing new."""
        new, n_new = None, 0
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                break
            msg = json.loads(data.decode())
            if msg.get("bye"):
                self.alive = False
                continue
            if self._seq is not None:
                if msg["seq"] < self._seq:          # source restarted (mcap --loop)
                    pass
                elif msg["seq"] > self._seq + 1:
                    self.dropped += msg["seq"] - self._seq - 1
            self._seq = msg["seq"]
            new = msg
            n_new += 1
        if new is not None:
            now = time.perf_counter()
            if self._t_last is not None:
                # frames arrive in bursts (the source outruns this loop): rate = burst
                # size over the drain interval, not 1/interval
                dt = max(now - self._t_last, 1e-4)
                self._rate += 0.2 * (n_new / dt - self._rate)
            self._t_last = now
            self.latest = {"skel": np.asarray(new["skel"], dtype=np.float64),
                           "quat": np.asarray(new["quat"], dtype=np.float64)}
        return self.latest

    def stale_s(self) -> float:
        return 999.0 if self._t_last is None else time.perf_counter() - self._t_last

    def quality(self):
        """(grade, hint) for the EST badge - glove health instead of vision health."""
        if not self.alive:
            return "BAD", "glove bridge exited - restart wuji_bridge.py"
        st = self.stale_s()
        if st > 0.5:
            return "BAD", "no glove data - is wuji_bridge.py running / glove on?"
        if st < 0.15 and self._rate > 60:
            return "GOOD", None
        if st < 0.3 and self._rate > 25:
            return "OK", None
        return "BAD", f"glove stream unstable ({self._rate:.0f} Hz)"

    def status(self) -> str:
        if not self.alive:
            return "glove: bridge exited"
        if self.latest is None:
            return "glove: waiting for wuji_bridge.py ..."
        return f"glove {self._rate:5.1f} Hz" + (f"  drop {self.dropped}" if self.dropped else "")


class CamBlobTracker:
    """Wrist position from the DEPTH image alone (glove sessions: MediaPipe cannot see a
    gloved hand, but the depth blob is appearance- and lighting-independent). Tracks the
    largest / previously-tracked connected component in a near-depth band and returns
    its 3D centroid in the camera frame; deltas of this drive the robot's translation."""

    Z_BAND = (0.25, 1.10)      # m: where the operator's hand lives
    MIN_AREA = 1200            # px: reject speckle

    def __init__(self, intr, depth_to_m: float, smooth: float = 1.0):
        self.fx, self.fy, self.cx, self.cy = intr
        self.depth_to_m = depth_to_m
        self._f = OneEuro(LIVE_POS_CUTOFF / smooth, LIVE_POS_BETA)
        self._prev_px = None
        self._prev_p = None
        self._t_prev = None

    def track(self, depth_raw):
        import cv2

        z = depth_raw.astype(np.float32) * self.depth_to_m
        mask = ((z > self.Z_BAND[0]) & (z < self.Z_BAND[1])).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        best, best_score = None, None
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < self.MIN_AREA:
                continue
            c = cents[i]
            if self._prev_px is not None:
                # locked: stay with the blob nearest the previous pick
                score = -np.hypot(c[0] - self._prev_px[0], c[1] - self._prev_px[1])
            else:
                # initial lock-on: the operator's hand is the CLOSEST thing in the
                # band, not the biggest (a table plane inside the band dwarfs it)
                score = -float(np.median(z[labels == i]))
            if best is None or score > best_score:
                best, best_score = i, score
        if best is None:
            return None
        sel = labels == best
        zi = float(np.median(z[sel]))
        cu, cv_ = cents[best]
        p = np.array([(cu - self.cx) / self.fx * zi, (cv_ - self.cy) / self.fy * zi, zi])
        now = time.perf_counter()
        dt = float(np.clip(now - self._t_prev, 0.01, 0.25)) if self._t_prev else 0.05
        self._t_prev = now
        if self._prev_p is not None and np.linalg.norm(p - self._prev_p) > 2.0 * dt:
            return self._prev_p                      # depth glitch: hold
        self._prev_px, self._prev_p = (cu, cv_), p
        return self._f(p, dt)


class TagWristTracker:
    """Wrist position from an AprilTag on the back of the glove - the RELIABLE wrist source
    (the depth blob is a hand-region centroid: coarse and orientation-coupled). dt_apriltags
    (the same detector as the offline scene calibration) detects the tag and estimates its
    6-DoF pose; we use the POSITION (deltas drive the robot). The constant tag->wrist offset
    cancels in clutch control, so the tag can be mounted anywhere flat on the hand with no
    calibration. The position is One-Euro smoothed; a missed frame holds the last position
    (short) then parks. Default family tagStandard41h12 (OpenCV aruco cannot detect it)."""

    def __init__(self, intr, marker_m: float, family: str = "tagStandard41h12",
                 tag_id: int | None = None, smooth: float = 1.0):
        from dt_apriltags import Detector

        self.fx, self.fy, self.cx, self.cy = intr
        self.cam_params = (self.fx, self.fy, self.cx, self.cy)
        self.marker_m = float(marker_m)
        self.tag_id = tag_id
        # dt_apriltags (same lib the offline scene calibration uses) - detects the
        # tagStandard41h12 family that OpenCV's aruco cannot, and estimates pose directly
        # (pose_t = tag centre in the camera frame). refine_edges + no decimation help a
        # small/imperfect printed tag on a glove.
        self.detector = Detector(families=family, nthreads=2, quad_decimate=1.0,
                                 quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
        self._f = OneEuro(LIVE_POS_CUTOFF / smooth, LIVE_POS_BETA)
        self._prev_p = None
        self._t_prev = None
        self.seen = False
        self._px = None                         # last tag centre in pixels (for drawing)
        self.miss = 0

    def track(self, bgr):
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dets = self.detector.detect(gray, estimate_tag_pose=True,
                                    camera_params=self.cam_params, tag_size=self.marker_m)
        pick = None
        for d in dets:
            if self.tag_id is None or d.tag_id == self.tag_id:
                pick = d
                break
        if pick is None:
            self.seen = False
            self.miss += 1
            return self._prev_p if self.miss <= 5 else None   # brief hold, then park
        self.seen = True
        self.miss = 0
        p = np.asarray(pick.pose_t, dtype=np.float64).reshape(3)   # tag centre, cam frame
        self._px = np.asarray(pick.center, dtype=np.float64)
        now = time.perf_counter()
        dt = float(np.clip(now - self._t_prev, 0.01, 0.25)) if self._t_prev else 0.05
        self._t_prev = now
        if self._prev_p is not None and np.linalg.norm(p - self._prev_p) > 2.0 * dt:
            return self._prev_p                               # implausible jump: hold
        self._prev_p = p
        return self._f(p, dt)


class GloveMapper:
    """Delta (clutch) mapping for the glove: wrist ORIENTATION deltas from the palm IMU
    (world frame is gravity-aligned z-up, same as the robot base - and the constant
    IMU-mount rotation cancels in ``R(t) @ R(0)^T``, so no mount calibration needed),
    wrist POSITION deltas from the camera depth blob (optional; parked without it),
    fingers from the glove's wrist-local skeleton re-attached at the mapped wrist.

    The one free parameter is the YAW between the IMU's world and the robot base
    (gyro-initialised, arbitrary): start with --imu-yaw, then align live with the
    '[' / ']' keys (5 deg steps; each press re-anchors, so the robot never jumps)."""

    def __init__(self, yaw_deg: float = 0.0, view: str = "front"):
        self.yaw_deg = float(yaw_deg)
        self.F_cam = VIEW_AXIS_MAPS.get(view, VIEW_AXIS_MAPS["front"])
        self._q0 = None
        self._B0 = None
        self._H = None            # constant local palm frame of the glove skeleton
        self._l_w = None
        self._r0 = None
        self._p_cam0 = None

    @property
    def ready(self) -> bool:
        return self._q0 is not None

    @staticmethod
    def _Rz(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def anchor(self, skel: np.ndarray, quat: np.ndarray, bracelet_T: np.ndarray,
               p_cam=None) -> bool:
        from scipy.spatial.transform import Rotation as Rot

        H = _wrist_pose_from_pts21_world(skel.astype(np.float32))
        if H is None:
            return False
        self._H = H.astype(np.float64)
        self._q0 = Rot.from_quat(quat).as_matrix()
        self._B0 = bracelet_T.astype(np.float64)
        c0 = skel[0]                                       # wrist keypoint (local origin)
        self._l_w = (c0 - self._H[:3, 3]) @ self._H[:3, :3]
        self._r0 = self._B0[:3, :3] @ self._l_w + self._B0[:3, 3]
        self._p_cam0 = None if p_cam is None else np.asarray(p_cam, dtype=np.float64)
        return True

    def map(self, skel: np.ndarray, quat: np.ndarray, p_cam=None) -> np.ndarray:
        from scipy.spatial.transform import Rotation as Rot

        F = self._Rz(self.yaw_deg)
        dR_w = Rot.from_quat(quat).as_matrix() @ self._q0.T
        W_R = F @ dR_w @ F.T @ self._B0[:3, :3]
        r = self._r0.copy()
        if p_cam is not None and self._p_cam0 is not None:
            r = r + self.F_cam @ (np.asarray(p_cam, dtype=np.float64) - self._p_cam0)
        W_t = r - W_R @ self._l_w
        local = (skel.astype(np.float64) - self._H[:3, 3]) @ self._H[:3, :3]
        return local @ W_R.T + W_t


def load_init_pose(run_name: str, joint_names: list[str]) -> dict | None:
    """Frame 0 of an ingested run's trajectory = a proven robot start pose over the table."""
    path = RUNS_DIR / run_name / "retarget_kinova_leap_optimized.json"
    if not path.is_file():
        print(f"[live][WARN] no init pose: {path} missing - starting from the zero pose")
        return None
    cfg = json.loads(path.read_text())["traj"][0]["robot_cfg"]
    missing = [n for n in joint_names if n not in cfg]
    if missing:
        print(f"[live][WARN] init pose {path} lacks joint(s) {missing} - using zero pose")
        return None
    return {n: float(cfg[n]) for n in joint_names}


# --------------------------------------------------------------------------------------
# IK (MinkSolver, warm-started) and UDP publishing
# --------------------------------------------------------------------------------------
class LiveRetargeter:
    def __init__(self, urdf_path: Path = KINOVA_LEAP_URDF, num_iter: int = LIVE_IK_ITERS,
                 arm_rate: float = LIVE_ARM_RATE, finger_rate: float = LIVE_FINGER_RATE,
                 q_init: dict | None = None, hand_scale: float | None = None,
                 decoupled: bool = True, finger_map: str = "knuckle"):
        from retargeting_kinova.kinova_leap_kinematics import MinkSolver, Target

        self.Target = Target
        self.solver = MinkSolver(str(urdf_path))
        self.q = dict(q_init) if q_init else {name: 0.0 for name in self.solver.joint_names}
        self.finger_map = finger_map

        # Hand-size compensation: the LEAP hand is larger than a human hand (its open
        # wrist->middle-tip reach is ~22 cm vs ~16-18 cm), so raw human-scale fingertip
        # targets sit INSIDE its envelope and the fingers must curl even for a fully open
        # hand. Scaling the keypoints about the wrist by (robot reach / operator reach)
        # maps open->open, and a touching pinch still maps to a touching pinch (scaling
        # about a point preserves zero gaps). hand_scale=None -> auto from the operator's
        # measured hand; a number fixes it (1.0 = off, the offline pipeline's behaviour).
        zero_q = {n: 0.0 for n in self.solver.joint_names}
        self.links = {"thumb": "thumb_tip_head", "index": "index_tip_head",
                      "middle": "middle_tip_head", "ring": "ring_tip_head"}
        self._kn_links = {"thumb": "pip_4", "index": "mcp_joint",
                          "middle": "mcp_joint_2", "ring": "mcp_joint_3"}
        fk0 = self.solver.compute_fk(zero_q, ["bracelet_link"] + list(self.links.values())
                                     + list(self._kn_links.values()))
        Tb = fk0["bracelet_link"]
        wrist_equiv = Tb[:3, 3] + 0.09 * Tb[:3, 1] - 0.09 * Tb[:3, 2]  # undo the bracelet offsets
        self.robot_reach = float(np.linalg.norm(fk0["middle_tip_head"][:3, 3] - wrist_equiv))
        # "knuckle" finger map: each fingertip target = the LEAP finger's OWN knuckle
        # (constant in the wrist/bracelet frame) + the human tip-from-knuckle vector,
        # scaled by (LEAP finger length / human finger length). Neutral spread then IS
        # the LEAP palm's (the wrist-anchored uniform scale squeezes tips to human
        # spacing, ~3.2 cm vs the palm's 4.5 cm - constant inward abduction, fingers
        # collide; and an open hand solved ~20 deg curled). Also invariant to the hand
        # scale k (it cancels in the ratio), so a drifting auto-scale cannot breathe
        # the fingers. "v2s2r" keeps the offline pipeline's raw-tip targets.
        inv_B0 = np.linalg.inv(Tb.astype(np.float64))
        self._kn_b, self._leap_len = {}, {}
        for f in self.links:
            kn = (inv_B0 @ np.r_[fk0[self._kn_links[f]][:3, 3], 1.0])[:3]
            tip = (inv_B0 @ np.r_[fk0[self.links[f]][:3, 3], 1.0])[:3]
            self._kn_b[f] = kn
            self._leap_len[f] = float(np.linalg.norm(tip - kn))    # straight finger, q=0
        self._chains = {"thumb": (2, 3, 4), "index": (5, 6, 7, 8),
                        "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16)}
        self.hand_scale_fixed = hand_scale
        self.hand_scale = 1.0 if hand_scale is None else float(hand_scale)
        self._human_reach = None
        self._scale_announced = False
        self.num_iter = num_iter
        self._last_wrist_T = None
        # the first solve always runs to full convergence, uncapped: recording must never
        # contain the rate-capped catch-up ramp from the start pose to the hand (in delta
        # mode engaging is zero-error anyway, so this only matters for absolute mode)
        self._converged = False
        # decoupled (default): arm solves for the wrist target ONLY, then fingers solve
        # with the bracelet anchored (fixed_links cost 1000). In the coupled whole-body
        # solve the 4 fingertip tasks recruit the ARM whenever the fingers cannot keep up
        # - measured live: finger curling alone dragged the bracelet around at 3 cm and
        # 11 deg per frame (finger-activity/arm-motion correlation 0.62) with a STILL
        # wrist. Decoupling makes that impossible: fingers can never move the arm.
        self.decoupled = decoupled
        # the wrist-target frame is built from wrist+MCP keypoints, and finger curls
        # shift the MCP estimates - give the wrist orientation its own One Euro filter
        self._f_wrot = OneEuro(LIVE_ROT_CUTOFF, LIVE_ROT_BETA)
        self._wq = None
        self.wrist_T = None                # latest filtered wrist (bracelet-target) frame
        self._R_raw = None                 # raw palm axes of the newest frame (fingers)
        self._B_canon = None               # floating mode: fixed palm pose for finger IK
        self._b2root = None                # constant bracelet -> leap_mount offset
        # per-joint rate cap (rad/s): keeps redundant-arm branch flips from snapping the robot
        self.rate = {n: (arm_rate if n.startswith("joint_") else finger_rate)
                     for n in self.solver.joint_names}

    def fk_bracelet(self) -> np.ndarray:
        """Current bracelet pose (4x4) of the held configuration - the delta-mode anchor."""
        return self.solver.compute_fk(self.q, ["bracelet_link"])["bracelet_link"]

    def _build_targets(self, pts_robot: np.ndarray, dt: float):
        """Hand-scale + wrist filtering + IK target frames; sets ``self.wrist_T``."""
        pts_robot = pts_robot.astype(np.float64)
        # measured operator hand reach (wrist->MCP + middle-finger segments: pose-invariant)
        reach = (np.linalg.norm(pts_robot[9] - pts_robot[0])
                 + np.linalg.norm(pts_robot[10] - pts_robot[9])
                 + np.linalg.norm(pts_robot[11] - pts_robot[10])
                 + np.linalg.norm(pts_robot[12] - pts_robot[11]))
        self._human_reach = reach if self._human_reach is None else (
            self._human_reach + 0.05 * (reach - self._human_reach))
        if self.hand_scale_fixed is None:
            self.hand_scale = float(np.clip(self.robot_reach / max(self._human_reach, 1e-6),
                                            1.0, 2.2))
        if not self._scale_announced:
            self._scale_announced = True
            print(f"[live] hand scale x{self.hand_scale:.2f} "
                  f"(LEAP reach {self.robot_reach * 100:.1f} cm / "
                  f"operator {self._human_reach * 100:.1f} cm"
                  + (", fixed)" if self.hand_scale_fixed is not None else ", auto)")
                  + ("  [finger map: knuckle-anchored - finger shape is scale-invariant]"
                     if self.finger_map == "knuckle" else ""))
        if self.hand_scale != 1.0:
            pts_robot = pts_robot[0] + self.hand_scale * (pts_robot - pts_robot[0])

        wrist_T = _wrist_pose_from_pts21_world(pts_robot.astype(np.float32))
        if wrist_T is not None:
            self._R_raw = wrist_T[:3, :3].astype(np.float64)   # THIS frame's own palm axes
        if wrist_T is None:
            wrist_T = self._last_wrist_T
        if wrist_T is None:
            return None

        # dedicated wrist-orientation filter: curling fingers shifts the MCP landmarks the
        # frame is built from; the position (wrist keypoint + offsets) is rebuilt with the
        # filtered axes so the +-9 cm bracelet offsets cannot wobble either
        from scipy.spatial.transform import Rotation as Rot

        dtc = float(np.clip(dt, 0.02, 0.25))
        q_w = Rot.from_matrix(wrist_T[:3, :3].astype(np.float64)).as_quat()
        if self._wq is not None and float(np.dot(q_w, self._wq)) < 0.0:
            q_w = -q_w
        q_f = self._f_wrot(q_w, dtc)
        self._wq = q_f / np.linalg.norm(q_f)
        R_f = Rot.from_quat(self._wq).as_matrix()
        w_pt = wrist_T[:3, 3] + 0.09 * wrist_T[:3, 1] - 0.09 * wrist_T[:3, 2]
        wrist_T = np.eye(4, dtype=np.float32)
        wrist_T[:3, :3] = R_f
        wrist_T[:3, 3] = w_pt - 0.09 * R_f[:, 1] + 0.09 * R_f[:, 2]
        self._last_wrist_T = wrist_T
        self.wrist_T = wrist_T.astype(np.float64)

        if self.finger_map == "knuckle":
            # fingertip = LEAP knuckle + scaled human tip-from-knuckle vector, built in
            # the CURRENT frame's OWN raw palm axes and re-attached at the filtered wrist
            # frame. Because the vector and the palm frame come from the same points, a
            # whole-hand estimation flip cancels EXACTLY for the fingers (finger shape
            # depends only on the hand relative to its own palm) - measured live, world-
            # frame vectors + the lagging filtered frame turned each 90-118 deg wrist
            # flip into fingers slamming their limits. compute_ik weights finger
            # ORIENTATION at 0, so identity rotations suffice.
            wT = wrist_T.astype(np.float64)
            finger_Ts = {}
            for f, ch in self._chains.items():
                v_l = self._R_raw.T @ (pts_robot[ch[-1]] - pts_robot[ch[0]])
                l_hum = sum(np.linalg.norm(pts_robot[ch[i + 1]] - pts_robot[ch[i]])
                            for i in range(len(ch) - 1))
                p_local = self._kn_b[f] + (self._leap_len[f] / max(l_hum, 1e-6)) * v_l
                T = np.eye(4, dtype=np.float32)
                T[:3, 3] = (wT @ np.r_[p_local, 1.0])[:3]
                finger_Ts[f] = T
        else:
            finger_Ts = _finger_poses_from_mano_world(pts_robot.astype(np.float32))
        return wrist_T, finger_Ts

    def solve(self, pts_robot: np.ndarray, dt: float = 0.1) -> dict:
        built = self._build_targets(pts_robot, dt)
        if built is None:
            return dict(self.q)
        wrist_T, finger_Ts = built

        wrist_target = self.Target(task_name="bracelet_link", relative=False, root_name="",
                                   link_name="bracelet_link", link_pose=wrist_T)
        finger_targets = [
            self.Target(task_name=fname, relative=False, root_name="",
                        link_name=self.links[fname], link_pose=T_f)
            for fname, T_f in finger_Ts.items()
        ]
        # first solve: full convergence from the start pose (the offline retarget's 100
        # iters); afterwards the warm start makes a few iterations per frame enough
        iters = self.num_iter if self._converged else 100
        if self.decoupled:
            res_a = self.solver.compute_ik(q_dict=self.q, ik_targets=[wrist_target],
                                           fixed_links=[], solver="quadprog",
                                           num_iter=iters, return_err=True)
            res = self.solver.compute_ik(q_dict=res_a["q"], ik_targets=finger_targets,
                                         fixed_links=["bracelet_link"], solver="quadprog",
                                         num_iter=iters, return_err=True)
        else:
            res = self.solver.compute_ik(q_dict=self.q,
                                         ik_targets=finger_targets + [wrist_target],
                                         fixed_links=[], solver="quadprog",
                                         num_iter=iters, return_err=True)
        q_new = res["q"]
        if self._converged:
            dt = float(np.clip(dt, 0.02, 0.25))
            for n, v in q_new.items():
                cap = self.rate[n] * dt
                prev = self.q[n]
                q_new[n] = float(np.clip(v, prev - cap, prev + cap))
        self._converged = True
        self.q = q_new.copy()          # the capped pose warm-starts the next solve
        return {k: float(v) for k, v in self.q.items()}

    def solve_floating(self, pts_robot: np.ndarray, dt: float = 0.1) -> dict:
        """Floating-hand collection: fingers-only IK at a fixed canonical palm - no arm
        anywhere in the loop. ``self.wrist_T`` carries the raw wrist pose for recording
        and the viewer; the arm is fitted OFFLINE by sim_teleop/retarget_float.py."""
        built = self._build_targets(pts_robot, dt)
        if built is None:
            return dict(self.q)
        wrist_T, finger_Ts = built
        if self._B_canon is None:
            self._B_canon = self.fk_bracelet().astype(np.float64)
        # re-express the finger targets at the canonical palm: articulation only
        M = self._B_canon @ np.linalg.inv(wrist_T.astype(np.float64))
        finger_targets = [
            self.Target(task_name=fname, relative=False, root_name="",
                        link_name=self.links[fname],
                        link_pose=(M @ T_f.astype(np.float64)).astype(np.float32))
            for fname, T_f in finger_Ts.items()
        ]
        iters = self.num_iter if self._converged else 100
        res = self.solver.compute_ik(q_dict=self.q, ik_targets=finger_targets,
                                     fixed_links=["bracelet_link"], solver="quadprog",
                                     num_iter=iters, return_err=True)
        q_new = res["q"]
        for n in q_new:                          # the arm plays no role in floating mode
            if n.startswith("joint_"):
                q_new[n] = self.q[n]
        if self._converged:
            dtc = float(np.clip(dt, 0.02, 0.25))
            for n, v in q_new.items():
                cap = self.rate[n] * dtc
                q_new[n] = float(np.clip(v, self.q[n] - cap, self.q[n] + cap))
        self._converged = True
        self.q = q_new.copy()
        return {k: float(v) for k, v in self.q.items()}

    def root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Floating-hand root (leap_mount) pose in robot-base coords for the sim viewer:
        the current wrist (bracelet-target) frame composed with the constant
        bracelet->leap_mount fixed-joint offset. Returns ``(pos, quat_xyzw)``."""
        from scipy.spatial.transform import Rotation as Rot

        if self._b2root is None:
            fk = self.solver.compute_fk(self.q, ["bracelet_link", "leap_mount"])
            self._b2root = np.linalg.inv(fk["bracelet_link"]) @ fk["leap_mount"]
        T_w = (self.wrist_T if self.wrist_T is not None
               else self.fk_bracelet().astype(np.float64))   # parked at the start pose
        T = T_w @ self._b2root
        return T[:3, 3].copy(), Rot.from_matrix(T[:3, :3]).as_quat()


class TargetPublisher:
    def __init__(self, addr=LIVE_UDP_ADDR):
        self.addr = tuple(addr)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, state: str, q: dict | None, root: list | None = None):
        msg = {"seq": self.seq, "state": state}
        if q is not None:
            msg["q"] = q
        if root is not None:
            msg["root"] = root                 # floating hand: [x,y,z, qx,qy,qz,qw]
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except OSError:
            pass
        self.seq += 1

    def bye(self):
        self.send("bye", None)


# --------------------------------------------------------------------------------------
# Saving + packaging
# --------------------------------------------------------------------------------------
def save_and_package(name: str, traj: list, record_fps: float, meta: dict, args) -> Path | None:
    work = TELEOP_OUT_ROOT / name
    work.mkdir(parents=True, exist_ok=True)
    traj_path = work / "retarget_kinova_leap.json"
    traj_path.write_text(json.dumps({"traj": traj}, indent=2))
    (work / "live_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[live] {len(traj)} frames -> {traj_path}")

    cmd = [sys.executable, str(PROJECT_ROOT / "sim_teleop" / "package_run.py"),
           "--traj", str(traj_path), "--name", name,
           "--scene-from", args.scene_from, "--source-fps", str(record_fps), "--force"]
    if args.objects_from:
        cmd += ["--objects-from", args.objects_from]
    print(f"[live] $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        print(f"[live][WARN] packaging failed (exit {r.returncode}); the raw trajectory is at "
              f"{traj_path} - package it by hand with sim_teleop/package_run.py")
        return None
    run_name = name if name.startswith("teleop_") else f"teleop_{name}"
    print(f"[live] packaged run: {RUNS_DIR / run_name}")
    print(f"[live] collect the episode (force_controller format):")
    print(f"[live]   conda run -n env_isaaclab python scripts/replay_trajectory.py --run {run_name}")
    return RUNS_DIR / run_name


# --------------------------------------------------------------------------------------
# UI overlay
# --------------------------------------------------------------------------------------
GLOVE_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_glove_hand(cv2, frame, glove_latest, yaw_deg, full=False):
    """Live view of the GLOVE's own hand estimate: the wrist-local skeleton rotated by
    the palm IMU (top view, world x right / y up on screen) - fingers and orientation
    move exactly as the robot will. Full-canvas when there is no camera, corner inset
    otherwise."""
    if glove_latest is None:
        cv2.putText(frame, "waiting for wuji_bridge.py ...", (30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        return
    from scipy.spatial.transform import Rotation as Rot

    R = Rot.from_quat(glove_latest["quat"]).as_matrix()
    pts = glove_latest["skel"] @ R.T                       # world frame (z-up)
    h, w = frame.shape[:2]
    if full:
        cx, cy, scale = w // 2, h // 2 + 20, 1000.0
    else:
        cx, cy, scale = w - 110, h - 110, 420.0
        cv2.rectangle(frame, (w - 220, h - 220), (w - 4, h - 4), (45, 45, 45), -1)
        cv2.rectangle(frame, (w - 220, h - 220), (w - 4, h - 4), (110, 110, 110), 1)
    px = (cx + pts[:, 0] * scale).astype(int)
    py = (cy - pts[:, 1] * scale).astype(int)              # world y -> screen up
    for a, b in GLOVE_BONES:
        cv2.line(frame, (px[a], py[a]), (px[b], py[b]), (90, 200, 90), 2)
    for i in range(21):
        cv2.circle(frame, (px[i], py[i]), 3, (60, 230, 230) if i == 0 else (90, 200, 90), -1)
    cv2.putText(frame, f"glove top view  yaw {yaw_deg:+.0f}",
                (cx - 100 if full else w - 214, (cy + 160) if full else h - 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_overlay(cv2, frame, state, rec_frames, hz, lifter_status, wrist_robot, anchors,
                 est=None):
    h, w = frame.shape[:2]
    banner_color = {"WAIT": (80, 80, 80), "CLUTCH": (0, 150, 255), "TRACKING": (200, 150, 0),
                    "RECORDING": (0, 0, 230), "SAVED": (0, 180, 0)}[state]
    cv2.rectangle(frame, (0, 0), (w, 34), banner_color, -1)
    label = state
    if state == "CLUTCH":
        label = "CLUTCH OFF - press 't' to engage"
    if state == "RECORDING":
        label = f"RECORDING  {rec_frames} frames ({rec_frames / TARGET_FPS:.1f}s)"
        if int(time.time() * 2) % 2:
            cv2.circle(frame, (w - 22, 17), 9, (255, 255, 255), -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    hint = None
    if est is not None:
        grade, hint = est
        gc = {"GOOD": (0, 180, 0), "OK": (0, 200, 255), "BAD": (0, 0, 230)}[grade]
        x0 = w - 150
        cv2.rectangle(frame, (x0, 38), (w - 6, 66), gc, -1)
        cv2.putText(frame, f"EST {grade}", (x0 + 10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    lines = [f"{hz:4.1f} Hz   {lifter_status}"]
    if hint:
        lines.append(f">> {hint}")
    if wrist_robot is not None:
        lines.append("wrist@robot  x{:+.3f}  y{:+.3f}  z{:+.3f} m".format(*wrist_robot))
    lines.append("t engage/release    SPACE start/stop    q quit    m mirror")
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, h - 10 - 22 * (len(lines) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if anchors is not None:
        px, valid = anchors
        for k, j in enumerate(DEPTH_ANCHOR_JOINTS):
            u, v = int(px[j, 0]), int(px[j, 1])
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(frame, (u, v), 5, (0, 200, 0) if valid[k] else (0, 0, 255), 2)


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=None,
                    help="episode name -> data/runs/teleop_<name> (default live_<stamp>)")
    ap.add_argument("--source", default="realsense",
                    help="'realsense' or a recorded folder with image/ + depth/ (test mode)")
    ap.add_argument("--serial", default=None, help="RealSense serial number (default: first device)")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="camera fps (realsense: 30; folder pacing: 10)")
    ap.add_argument("--mode", choices=["delta", "absolute"], default="delta",
                    help="delta (default): robot starts at a known pose and follows relative "
                    "hand motion from the moment you engage ('t') - no camera calibration "
                    "needed. absolute: map the hand's absolute pose via the AprilTag calib "
                    "(camera must match the calibration's setup).")
    ap.add_argument("--init-from", default=None,
                    help="delta mode: run whose trajectory frame 0 is the robot start pose "
                    "(default: the --scene-from run)")
    ap.add_argument("--view", choices=["front", "top", "hand"], default="front",
                    help="delta-mode steering directions. front (default): camera faces you - "
                    "push away = robot forward, up = up, your right = robot right. top: camera "
                    "above/behind looking down (the original rig). hand: directions relative to "
                    "your palm orientation at engage (works for any camera, harder to steer).")
    ap.add_argument("--calib", default=None,
                    help="absolute mode: camera_frame_pose.json of the CURRENT camera setup "
                    "(default: the folder's own file in folder mode, else the donor run's)")
    ap.add_argument("--tag-index", type=int, default=ROBOT_TAG_INDEX)
    ap.add_argument("--scene-from", default=DEFAULT_SCENE_DONOR,
                    help="donor run for the packaged calibration files")
    ap.add_argument("--objects-from", default=None,
                    help="donor run whose objects go into the packaged run (contact episodes)")
    ap.add_argument("--record-fps", type=float, default=TARGET_FPS,
                    help="trajectory sampling rate (default 10 = the replay convention)")
    ap.add_argument("--hand-scale", default=None,
                    help="finger-target scaling about the wrist. 'auto' (delta-mode default): "
                    "LEAP reach / your measured hand reach, so a fully open hand maps to a "
                    "fully open LEAP. A number fixes it; 1.0 disables (absolute-mode default, "
                    "matching the offline pipeline's video-registered geometry).")
    ap.add_argument("--num-iter", type=int, default=LIVE_IK_ITERS, help="IK iterations per frame")
    ap.add_argument("--coupled-ik", action="store_true",
                    help="single whole-body IK solve (old behaviour: finger targets can "
                    "recruit the arm; default is the decoupled arm-then-fingers solve)")
    ap.add_argument("--floating", action="store_true",
                    help="floating-hand collection: no arm in the loop at all - the sim "
                    "shows a free-floating LEAP that mirrors your hand 1:1; the wrist pose "
                    "is recorded raw and the arm is fitted OFFLINE afterwards "
                    "(retarget_float.py, run automatically on save)")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="One Euro smoothing strength at rest (>1 = calmer static hand, "
                    "slightly laggier; motion responsiveness is speed-adaptive either way)")
    ap.add_argument("--finger-map", choices=["knuckle", "v2s2r"], default="knuckle",
                    help="fingertip target construction: 'knuckle' (default) anchors each "
                    "tip at the LEAP finger's own knuckle + the scaled human tip vector "
                    "(open->open, no finger convergence/crossing, hand-size invariant); "
                    "'v2s2r' is the offline pipeline's raw-tip targets")
    ap.add_argument("--min-conf", type=float, default=0.6, help="MediaPipe confidence thresholds")
    ap.add_argument("--glove", action="store_true",
                    help="Wuji-glove input instead of vision: fingers from the glove "
                    "skeleton (120 Hz, flip-free), wrist orientation from the palm IMU. "
                    "Start sim_teleop/wuji_bridge.py separately (the launcher does it). "
                    "Wrist TRANSLATION comes from the camera depth blob when --source is "
                    "a camera/folder; with --source none the wrist stays parked "
                    "(orientation + fingers only).")
    ap.add_argument("--imu-yaw", type=float, default=0.0,
                    help="glove: initial yaw (deg) between the IMU world and the robot "
                    "base; align live with the '[' / ']' keys (5 deg steps, re-anchors)")
    ap.add_argument("--wrist-tag", action="store_true",
                    help="glove: take wrist TRANSLATION from an AprilTag on the back of "
                    "the glove (reliable, sub-cm) instead of the depth blob (coarse). "
                    "Needs --source realsense and a printed tag; see --tag-size/--tag-id")
    ap.add_argument("--tag-size", type=float, default=0.053,
                    help="AprilTag BLACK-square side length in METERS (measure it; "
                    "default 0.053)")
    ap.add_argument("--tag-id", type=int, default=None,
                    help="only track this AprilTag id (default: any; use if other tags "
                    "are visible)")
    ap.add_argument("--tag-family", "--tag-dict", dest="tag_family",
                    default="tagStandard41h12",
                    help="dt_apriltags family for the wrist tag (default tagStandard41h12; "
                    "e.g. tag36h11). OpenCV aruco families are NOT used")
    ap.add_argument("--port", type=int, default=LIVE_UDP_ADDR[1])
    ap.add_argument("--mirror", action="store_true", help="mirror the preview (display only)")
    ap.add_argument("--no-window", action="store_true", help="headless (tests; use --auto-record)")
    ap.add_argument("--auto-record", action="store_true",
                    help="record from the first tracked frame until the source ends (tests)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N recorded frames")
    args = ap.parse_args()

    if not args.no_window:
        prepare_x_display()
    import cv2

    name = args.name or datetime.now().strftime("live_%Y%m%d_%H%M%S")

    # ---- source ----
    if args.source == "none":
        if not args.glove:
            raise SystemExit("[live] --source none is only valid with --glove")
        src = None
        print("[live] no camera source: glove-only session (wrist position parked)")
    elif args.source == "realsense":
        try:
            src = RealSenseSource(fps=int(args.source_fps or 30), serial=args.serial)
        except Exception as e:
            if not args.glove:
                raise
            src = None
            print(f"[live][WARN] RealSense unavailable ({e}) - glove-only session "
                  "(wrist position parked)")
    else:
        # always paced: the recorder samples wall-clock time, so a folder replayed at its
        # native fps yields the same trajectory a live session at that rate would
        src = FolderSource(Path(args.source), fps=args.source_fps or 10.0, pace=True)
    if src is not None:
        print(f"[live] source: {args.source}  intrinsics: {[round(v, 1) for v in src.intrinsics]}")

    # ---- hand -> robot mapping ----
    to_robot = None
    if args.mode == "absolute":
        calib = args.calib
        if calib is None and args.source != "realsense":
            cand = Path(args.source) / "camera_frame_pose.json"
            calib = cand if cand.is_file() else None
        if calib is None:
            calib = RUNS_DIR / args.scene_from / "scene" / "camera_frame_pose.json"
            print(f"[live][WARN] no --calib given: using the donor run's {calib}. "
                  "Only valid if the camera and table have NOT moved since that recording.")
        to_robot = TagTransform(calib, args.tag_index)
        print(f"[live] absolute mode: camera->robot from {to_robot.source} (tag {args.tag_index})")

    glove_rx = glove_mapper = blob = None
    glove_latest = None
    if args.glove:
        glove_rx = GloveReceiver()
        glove_mapper = GloveMapper(yaw_deg=args.imu_yaw, view=args.view)
        if src is not None:
            if args.wrist_tag:
                blob = TagWristTracker(src.intrinsics, args.tag_size, args.tag_family,
                                       args.tag_id, smooth=args.smooth)
                print(f"[live] wrist translation from AprilTag ({args.tag_family}, "
                      f"{args.tag_size * 100:.1f} cm"
                      + (f", id {args.tag_id}" if args.tag_id is not None else "")
                      + ") - keep the tag facing the camera")
            else:
                blob = CamBlobTracker(src.intrinsics, src.depth_to_m, smooth=args.smooth)
        if args.mode == "absolute":
            raise SystemExit("[live] --glove is delta/clutch only (the glove has no "
                             "absolute position); drop --mode absolute")
        wsrc = ("wrist parked (no camera)" if src is None
                else "wrist translation from the AprilTag" if args.wrist_tag
                else "wrist translation from the camera depth blob")
        print(f"[live] glove mode: fingers + wrist orientation from the Wuji glove; {wsrc}")
        lifter = None
    else:
        lifter = HandLifter(src.intrinsics, src.depth_to_m, smooth=args.smooth,
                            min_conf=args.min_conf)
    if args.hand_scale is None:
        hand_scale = None if args.mode == "delta" else 1.0     # auto only where no video geometry
    elif str(args.hand_scale).lower() == "auto":
        hand_scale = None
    else:
        hand_scale = float(args.hand_scale)
    retargeter = LiveRetargeter(num_iter=args.num_iter, hand_scale=hand_scale,
                                decoupled=not args.coupled_ik, finger_map=args.finger_map)
    q_init = load_init_pose(args.init_from or args.scene_from, retargeter.solver.joint_names)
    if q_init is not None:
        retargeter.q = dict(q_init)
    mapper = DeltaMapper(view=args.view)
    if args.mode == "delta":
        print(f"[live] delta mode ({args.view} view): robot starts at frame 0 of "
              f"{args.init_from or args.scene_from}; press 't' to engage the clutch.")
    pub = TargetPublisher(("127.0.0.1", args.port))

    # per-frame diagnostics, always on: this is what to look at when a session felt wrong
    work_dir = TELEOP_OUT_ROOT / name
    work_dir.mkdir(parents=True, exist_ok=True)
    track_log = open(work_dir / "track_log.jsonl", "w", buffering=1)
    session_t0 = time.time()

    engaged = args.mode == "absolute"    # absolute mode follows as soon as a hand appears
    recording = False
    ever_lifted = False
    traj: list[dict] = []
    rec_est = {"GOOD": 0, "OK": 0, "BAD": 0}   # estimation health per recorded frame
    last_q = dict(retargeter.q)          # published from frame one: the robot's start pose
    cam21_latest = None
    last_solve_t = None
    rec_period = 1.0 / args.record_fps
    next_rec_t = None
    last_key_t = 0.0
    t_hz, n_hz, hz = time.perf_counter(), 0, 0.0
    window = "hand capture  (t engage, SPACE record, q quit)"
    if args.mode == "delta":
        print("[live] move your hand into view, press 't' to engage, SPACE to record.")
    else:
        print("[live] tracking... move your hand; SPACE starts the recording.")

    def state_label() -> str:
        if recording:
            return "RECORDING"
        ready = glove_mapper.ready if args.glove else mapper.ready
        if engaged and (args.mode == "absolute" or ready):
            return "TRACKING"
        return "CLUTCH" if ever_lifted else "WAIT"

    VIEW_HINTS = {
        "front": "push away from you = robot FORWARD | up = UP | your right = robot RIGHT",
        "top": "image up (away from you) = robot FORWARD | image right = robot RIGHT",
        "hand": "directions follow your palm: along your fingers = along the ROBOT's fingers",
    }

    def current_bracelet() -> np.ndarray:
        """The pose re-clutching must bind to: in floating mode the hand has MOVED away
        from the arm's FK (arm joints stay parked), so anchoring to fk_bracelet would
        teleport it back to the start pose on every re-engage."""
        if args.floating and retargeter.wrist_T is not None:
            return retargeter.wrist_T
        return retargeter.fk_bracelet()

    def engage_clutch() -> bool:
        if args.glove:
            if glove_latest is None:
                print("[live] cannot engage: no glove data yet (is wuji_bridge.py running?).")
                return False
            if not glove_mapper.anchor(glove_latest["skel"], glove_latest["quat"],
                                       current_bracelet(), p_cam_latest):
                print("[live] cannot engage: degenerate glove skeleton, retry.")
                return False
            print(f"[live] clutch ENGAGED (glove, imu yaw {glove_mapper.yaw_deg:+.0f} deg) - "
                  "if 'forward' steers sideways, tap '[' / ']' to rotate the mapping.")
            return True
        if cam21_latest is None:
            print("[live] cannot engage: no tracked hand right now.")
            return False
        if not mapper.anchor(cam21_latest, current_bracelet()):
            print("[live] cannot engage: degenerate hand pose, adjust and retry.")
            return False
        print(f"[live] clutch ENGAGED - {VIEW_HINTS[args.view]}")
        return True

    p_cam_latest = None
    try:
        while True:
            if src is not None:
                bgr, depth = src.read()
                if bgr is None or depth is None:
                    print("[live] source ended.")
                    break
            else:
                time.sleep(1.0 / 60.0)               # glove-only: pace the loop ourselves
                bgr, depth = None, None

            cam21, kp2d, anchors = None, None, None
            if args.glove:
                if glove_rx.poll() is not None:
                    glove_latest = glove_rx.latest
                if not glove_rx.alive:
                    print("[live] glove bridge said bye - stopping.")
                    break
                if blob is not None:
                    # TagWristTracker reads the color image; CamBlobTracker the depth
                    if isinstance(blob, TagWristTracker):
                        p = blob.track(bgr) if bgr is not None else None
                    else:
                        p = blob.track(depth) if depth is not None else None
                    if p is not None:
                        p_cam_latest = p
                if glove_latest is not None and not ever_lifted:
                    ever_lifted = True
                    if not args.auto_record:
                        print("[live] glove acquired - press 't' to engage the clutch.")
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cam21, kp2d, anchors = lifter.lift(rgb, depth)
                if cam21 is not None:
                    cam21_latest = cam21
                    if not ever_lifted:
                        ever_lifted = True
                        if args.mode == "delta" and not args.auto_record:
                            print("[live] hand acquired - press 't' to engage the clutch.")
            tracked_now = (glove_latest is not None) if args.glove else (cam21 is not None)
            if tracked_now and args.auto_record and not recording:
                if args.mode == "delta" and not engaged:
                    engaged = engage_clutch()
                if engaged:
                    recording = True
                    rec_est = {"GOOD": 0, "OK": 0, "BAD": 0}
                    next_rec_t = time.perf_counter()
                    print("[live] --auto-record: recording started.")

            # ---- hand -> robot: absolute (calibrated) / delta (clutch) / glove ----
            wrist_robot = None
            robot21 = None
            if args.glove:
                if engaged and glove_mapper.ready and glove_latest is not None:
                    robot21 = glove_mapper.map(glove_latest["skel"], glove_latest["quat"],
                                               p_cam_latest)
            elif cam21 is not None and engaged and (args.mode == "absolute" or mapper.ready):
                robot21 = to_robot(cam21) if args.mode == "absolute" else mapper(cam21)
            if robot21 is not None:
                wrist_robot = robot21[0]
                now = time.perf_counter()
                solve_dt = now - last_solve_t if last_solve_t is not None else 0.1
                last_solve_t = now
                if args.floating:
                    last_q = retargeter.solve_floating(robot21, dt=solve_dt)
                else:
                    last_q = retargeter.solve(robot21, dt=solve_dt)
            msg_root = None
            if args.floating:
                rp, rq = retargeter.root_pose()
                msg_root = [round(float(v), 5) for v in rp] + [round(float(v), 6) for v in rq]
            pub.send(state_label(), last_q, root=msg_root)   # always a pose to show

            health = glove_rx if args.glove else lifter
            track_log.write(json.dumps({
                "t": round(time.time() - session_t0, 3),
                "state": state_label(),
                "status": glove_rx.status() if args.glove else lifter.status,
                "fit_cm": None if args.glove else round(lifter.anchor_residual * 100, 2),
                "reproj_px": None if args.glove else round(lifter.reproj_px, 1),
                "glove_hz": round(glove_rx._rate, 1) if args.glove else None,
                "wrist_cam": ([round(float(v), 4) for v in p_cam_latest]
                              if args.glove and p_cam_latest is not None else
                              None if (args.glove or cam21 is None) else
                              [round(float(v), 4) for v in cam21[0]]),
                "wrist_robot": None if wrist_robot is None else [round(float(v), 4) for v in wrist_robot],
                "k": round(retargeter.hand_scale, 3),
                "est": health.quality()[0],
                "rec": len(traj),
            }) + "\n")

            # ---- fixed-rate trajectory sampling (holds the last pose on dropouts) ----
            if recording:
                now = time.perf_counter()
                while next_rec_t is not None and now >= next_rec_t:
                    entry = {"robot_cfg": dict(last_q)}
                    if args.floating:
                        from scipy.spatial.transform import Rotation as Rot
                        T_w = (retargeter.wrist_T if retargeter.wrist_T is not None
                               else retargeter.fk_bracelet().astype(np.float64))
                        entry["wrist_pos"] = [float(v) for v in T_w[:3, 3]]
                        entry["wrist_quat_xyzw"] = [
                            float(v) for v in Rot.from_matrix(T_w[:3, :3]).as_quat()]
                    traj.append(entry)
                    rec_est[(glove_rx if args.glove else lifter).quality()[0]] += 1
                    next_rec_t += rec_period
                if args.max_frames and len(traj) >= args.max_frames:
                    print(f"[live] --max-frames {args.max_frames} reached.")
                    break

            # ---- fps meter ----
            n_hz += 1
            if time.perf_counter() - t_hz >= 1.0:
                hz = n_hz / (time.perf_counter() - t_hz)
                t_hz, n_hz = time.perf_counter(), 0
                if args.no_window:
                    print(f"[live] {state_label():9s} {hz:4.1f} Hz  "
                          f"{glove_rx.status() if args.glove else lifter.status}  "
                          f"rec {len(traj)} frames", flush=True)

            # ---- preview + keys ----
            if not args.no_window:
                if bgr is None:
                    bgr = np.full((480, 640, 3), 26, np.uint8)   # glove-only canvas
                if args.glove:
                    draw_glove_hand(cv2, bgr, glove_latest, glove_mapper.yaw_deg,
                                    full=(src is None))
                    tag_px = None                       # tag tracker uses _px, blob _prev_px
                    if blob is not None:
                        tag_px = getattr(blob, "_px", None)
                        if tag_px is None:
                            tag_px = getattr(blob, "_prev_px", None)
                    if tag_px is not None:
                        u, v = int(tag_px[0]), int(tag_px[1])
                        seen = getattr(blob, "seen", True)     # tag tracker: is it visible now?
                        col = (0, 220, 220) if seen else (0, 120, 200)
                        cv2.circle(bgr, (u, v), 9, col, 2)
                        label = "wrist" + ("" if seen else " (tag lost)")
                        cv2.putText(bgr, label, (u + 12, v + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
                elif kp2d is not None:
                    lifter.draw(bgr, kp2d)
                status = glove_rx.status() if args.glove else lifter.status
                if isinstance(blob, TagWristTracker):
                    status += "  |  wrist tag: " + ("SEEN" if blob.seen else "LOST")
                if recording and not engaged:
                    status += " | CLUTCH OFF (holding)"
                draw_overlay(cv2, bgr, state_label(), len(traj), hz, status, wrist_robot,
                             anchors, est=(glove_rx if args.glove else lifter).quality())
                cv2.imshow(window, bgr[:, ::-1].copy() if args.mirror else bgr)
                key = cv2.waitKey(1) & 0xFF
                now = time.perf_counter()
                if key in (ord("q"), 27):
                    if recording:
                        print(f"[live] quit during recording - {len(traj)} frames DISCARDED.")
                        traj = []
                    break
                if key == ord("m") and now - last_key_t > 0.4:
                    args.mirror = not args.mirror
                    last_key_t = now
                if args.glove and key in (ord("["), ord("]")) and now - last_key_t > 0.15:
                    last_key_t = now
                    glove_mapper.yaw_deg += 5.0 if key == ord("]") else -5.0
                    if glove_mapper.ready and glove_latest is not None:
                        glove_mapper.anchor(glove_latest["skel"], glove_latest["quat"],
                                            current_bracelet(), p_cam_latest)
                    print(f"[live] imu yaw -> {glove_mapper.yaw_deg:+.0f} deg (re-anchored)")
                if key == ord("t") and now - last_key_t > 0.4:
                    last_key_t = now
                    if args.mode == "absolute":
                        print("[live] 't' does nothing in absolute mode.")
                    elif engaged:
                        engaged = False
                        print("[live] clutch RELEASED - robot holds; reposition your hand, "
                              "'t' to re-engage.")
                    else:
                        engaged = engage_clutch()
                if key == 32 and now - last_key_t > 0.4:          # SPACE
                    last_key_t = now
                    if recording:
                        break                                     # stop -> save + package
                    elif engaged and (args.mode == "absolute"
                                      or (glove_mapper.ready if args.glove else mapper.ready)):
                        recording, traj = True, []
                        rec_est = {"GOOD": 0, "OK": 0, "BAD": 0}
                        next_rec_t = time.perf_counter()
                        print("[live] RECORDING - SPACE again to stop and save.")
                    elif args.mode == "delta":
                        print("[live] engage the clutch first ('t' with your hand in view).")
                    else:
                        print("[live] no hand tracked yet - recording not started.")
    finally:
        pub.bye()
        if src is not None:
            src.close()
        track_log.close()
        print(f"[live] per-frame diagnostics -> {work_dir / 'track_log.jsonl'}")
        if not args.no_window:
            cv2.destroyAllWindows()

    # ---- save ----
    if traj:
        meta = {
            "name": name,
            "source": str(args.source),
            "mode": args.mode,
            "view": args.view if args.mode == "delta" else None,
            "calib": str(to_robot.source) if to_robot is not None else None,
            "init_from": (args.init_from or args.scene_from) if args.mode == "delta" else None,
            "hand_scale": retargeter.hand_scale,
            "finger_map": retargeter.finger_map,
            "glove": bool(args.glove),
            "imu_yaw_deg": glove_mapper.yaw_deg if args.glove else None,
            "floating": args.floating,
            "record_fps": args.record_fps,
            "num_frames": len(traj),
            "ik_iters": args.num_iter,
            "smooth": args.smooth,
            "est_quality": dict(rec_est),
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        n_est = max(sum(rec_est.values()), 1)
        good_pct = 100.0 * rec_est["GOOD"] / n_est
        print(f"[live] take estimation health: {good_pct:.0f}% GOOD, "
              f"{100.0 * rec_est['OK'] / n_est:.0f}% OK, "
              f"{100.0 * rec_est['BAD'] / n_est:.0f}% BAD"
              + (("  -> consider re-recording ("
                  + ("check the glove stream/bridge" if args.glove
                     else "fix lighting/view until EST shows GOOD") + ")")
                 if good_pct < 60 else ""))
        if args.floating:
            float_path = work_dir / "float_traj.json"
            float_path.write_text(json.dumps({"meta": meta, "frames": traj}, indent=2))
            print(f"[live] floating recording (wrist poses + fingers) -> {float_path}")
            print("[live] fitting the arm to the recorded wrist poses (offline retarget)...")
            from sim_teleop.retarget_float import retarget_float_frames

            full_traj, report = retarget_float_frames(traj, q_init,
                                                      frame_dt=1.0 / args.record_fps)
            meta["float_retarget"] = report
            save_and_package(name, full_traj, args.record_fps, meta, args)
        else:
            save_and_package(name, traj, args.record_fps, meta, args)
    else:
        print("[live] nothing recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
