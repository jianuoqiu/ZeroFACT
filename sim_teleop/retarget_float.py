#!/usr/bin/env python3
"""Offline arm retargeting for floating-hand recordings (stage 4b of live teleop).

Floating-mode collection (``live_teleop.py --floating``) records, at 10 Hz, the wrist
(bracelet-target) pose in robot-base coordinates plus the 16 solved LEAP finger joints -
no arm is involved while the operator works. This script fits the 7-DoF Kinova arm to
that wrist-pose series afterwards (warm-started bracelet-only IK, hardware-rate capped),
merges the recorded fingers, and emits the standard stage-4 trajectory - so packaging,
the validated replay and force_controller all stay unchanged.

The per-frame wrist tracking error is reported: where the arm cannot reach the recorded
wrist pose (workspace/joint limits), the episode deviates from what the operator saw with
the floating hand - large errors mean the demo should be redone closer to the workspace.

Run in the ``cam`` env (mujoco/mink):
    conda run -n cam python sim_teleop/retarget_float.py \
        --float-traj outputs/sim_teleop/<name>/float_traj.json --name <name> \
        [--objects-from <run>] [--scene-from <run>] [--init-from <run>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sim_teleop.teleop_config import (  # noqa: E402
    DEFAULT_SCENE_DONOR,
    LIVE_ARM_RATE,
    TARGET_FPS,
)

ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
LEAP_JOINTS = [str(i) for i in range(16)]

# wrist-flip repair (see _clean_wrist_flips): a real wrist cannot exceed ~600 deg/s, so a
# 10 Hz sample jumping more than this in one step is an estimation flip, not motion
FLIP_JUMP_DEG = 45.0
FLIP_MAX_LEN = 20         # longest flipped stint (frames) the repair will bridge
FLIP_RETURN_RATE = 200.0  # deg/s of REAL motion tolerated across a bridged stint
FLIP_RETURN_CAP = 90.0    # the return tolerance never grows beyond this


def _clean_wrist_flips(frames: list[dict], frame_dt: float) -> int:
    """Repair MediaPipe interpretation flips in the recorded wrist-pose series.

    The live tracker gates impossible palm-frame jumps but concedes to PERSISTENT flips
    (they are kinematically indistinguishable from a fast turn while they last), so a
    recording can contain segments rotated ~90-120 deg that snap back afterwards. Here,
    with the whole series in hand, they are detectable: a segment bounded by two
    opposite impossible jumps whose bracketing poses ARE mutually consistent is a flip,
    not motion. Such segments are replaced by slerp between the bracketing poses
    (positions are left alone - measured flips moved the wrist ~1 cm while the
    orientation snapped 90-118 deg). Returns the number of repaired frames."""
    from scipy.spatial.transform import Rotation as Rot, Slerp

    qs = np.array([f["wrist_quat_xyzw"] for f in frames], dtype=np.float64)
    for i in range(1, len(qs)):                       # hemisphere-align for the angle test
        if np.dot(qs[i], qs[i - 1]) < 0.0:
            qs[i] = -qs[i]

    def ang(a, b):
        return 2.0 * np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))

    jump = max(FLIP_JUMP_DEG, 600.0 * frame_dt)
    repaired = 0
    i = 1
    while i < len(qs):
        if ang(qs[i], qs[i - 1]) <= jump:
            i += 1
            continue
        # impossible jump into frame i: the stint is flipped only if it later RETURNS
        # near the pre-entry pose (allowing modest real motion meanwhile); a fast real
        # turn never produces an impossible jump in the first place, and a stint that
        # never returns cannot be disambiguated - leave it and let the fit report flag it
        end = None
        for j in range(i, min(i + FLIP_MAX_LEN, len(qs) - 1)):
            tol = min(jump + FLIP_RETURN_RATE * frame_dt * (j + 1 - i), FLIP_RETURN_CAP)
            if ang(qs[j + 1], qs[i - 1]) <= tol:
                end = j                                # frames i..j are the flipped stint
                break
        if end is None:
            i += 1
            continue
        sl = Slerp([0.0, 1.0], Rot.from_quat([qs[i - 1], qs[end + 1]]))
        ts = [(k - (i - 1)) / (end + 2 - i) for k in range(i, end + 1)]
        for k, q in zip(range(i, end + 1), sl(ts).as_quat()):
            qs[k] = q
            frames[k]["wrist_quat_xyzw"] = [float(v) for v in q]
            repaired += 1
        i = end + 1
    return repaired


def retarget_float_frames(frames: list[dict], q_init: dict | None,
                          num_iter: int = 60, frame_dt: float = 1.0 / TARGET_FPS,
                          arm_rate: float = LIVE_ARM_RATE) -> tuple[list[dict], dict]:
    """Wrist-pose series + recorded fingers -> stage-4 trajectory + tracking report."""
    from scipy.spatial.transform import Rotation as Rot

    from sim_teleop.live_teleop import LiveRetargeter

    repaired = _clean_wrist_flips(frames, frame_dt)
    if repaired:
        print(f"[float-retarget] repaired {repaired} wrist-flip frames "
              f"(impossible >600 deg/s orientation snaps that returned; slerp-bridged)")

    rt = LiveRetargeter(num_iter=num_iter, q_init=q_init)
    Target = rt.Target
    cap = arm_rate * frame_dt
    traj, pos_err, ori_err = [], [], []
    for k, fr in enumerate(frames):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rot.from_quat(fr["wrist_quat_xyzw"]).as_matrix()
        T[:3, 3] = fr["wrist_pos"]
        target = Target(task_name="bracelet_link", relative=False, root_name="",
                        link_name="bracelet_link", link_pose=T)
        iters = 100 if k == 0 else num_iter          # frame 0: full convergence, no ramp
        res = rt.solver.compute_ik(q_dict=rt.q, ik_targets=[target], fixed_links=[],
                                   solver="quadprog", num_iter=iters, return_err=True)
        q_new = res["q"]
        if k > 0:
            for n in ARM_JOINTS:                     # hardware-plausible arm speed
                q_new[n] = float(np.clip(q_new[n], rt.q[n] - cap, rt.q[n] + cap))
        rt.q = q_new.copy()

        fk = rt.solver.compute_fk(rt.q, ["bracelet_link"])["bracelet_link"]
        pos_err.append(float(np.linalg.norm(fk[:3, 3] - T[:3, 3])))
        dR = fk[:3, :3].T @ T[:3, :3].astype(np.float64)
        ori_err.append(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))

        cfg = {n: float(rt.q[n]) for n in ARM_JOINTS}
        cfg.update({n: float(fr["robot_cfg"][n]) for n in LEAP_JOINTS})
        traj.append({"robot_cfg": cfg})

    pos_err, ori_err = np.array(pos_err), np.array(ori_err)
    report = {
        "frames": len(traj),
        "repaired_flip_frames": repaired,
        "wrist_pos_err_cm": {"mean": round(float(pos_err.mean()) * 100, 2),
                             "p95": round(float(np.percentile(pos_err, 95)) * 100, 2),
                             "max": round(float(pos_err.max()) * 100, 2)},
        "wrist_ori_err_deg": {"mean": round(float(ori_err.mean()), 1),
                              "p95": round(float(np.percentile(ori_err, 95)), 1),
                              "max": round(float(ori_err.max()), 1)},
        "worst_frame": int(pos_err.argmax()),
    }
    print(f"[float-retarget] {len(traj)} frames | wrist pos err "
          f"mean {report['wrist_pos_err_cm']['mean']} / max {report['wrist_pos_err_cm']['max']} cm"
          f" | ori err mean {report['wrist_ori_err_deg']['mean']} / "
          f"max {report['wrist_ori_err_deg']['max']} deg")
    if report["wrist_pos_err_cm"]["max"] > 3.0 or report["wrist_ori_err_deg"]["max"] > 15.0:
        print(f"[float-retarget][WARN] the arm could not follow the wrist everywhere "
              f"(worst frame {report['worst_frame']}) - the replayed episode deviates from "
              f"what the floating hand showed there. Keep demos inside the arm workspace.")
    return traj, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--float-traj", required=True, help="float_traj.json from --floating")
    ap.add_argument("--name", required=True, help="episode name -> data/runs/teleop_<name>")
    ap.add_argument("--init-from", default=None, help="run whose frame 0 warm-starts the arm")
    ap.add_argument("--scene-from", default=DEFAULT_SCENE_DONOR)
    ap.add_argument("--objects-from", default=None)
    ap.add_argument("--num-iter", type=int, default=60)
    args = ap.parse_args()

    from sim_teleop.live_teleop import load_init_pose, save_and_package

    data = json.loads(Path(args.float_traj).read_text())
    frames = data["frames"]
    from sim_teleop.live_teleop import LiveRetargeter
    q_init = load_init_pose(args.init_from or args.scene_from,
                            LiveRetargeter().solver.joint_names)
    traj, report = retarget_float_frames(frames, q_init, num_iter=args.num_iter)

    meta = dict(data.get("meta", {}))
    meta["float_retarget"] = report
    save_and_package(args.name, traj, TARGET_FPS, meta, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
