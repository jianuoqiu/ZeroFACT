#!/usr/bin/env python3
"""Stage 3+4 of the sim_teleop pipeline: Dyn-HaMR result -> robot joint trajectory.

Drives the two proven external tools (no math is re-implemented here):

  3. ``retargeting_kinova/dynhamr_to_skeleton.py``  (conda env dynhamr5090)
       MANO params -> (T,21,3) keypoints in the ROBOT BASE frame, applying
       - the AprilTag camera calibration (camera_frame_pose.json), and
       - the depth correction (``--depth-scale``, from estimate_depth_scale.py).
  4. ``retargeting_kinova.retarget_leap``           (conda env vid2sim2real)
       keypoints -> Kinova + LEAP joints via mink/mujoco IK
       -> ``retarget_kinova_leap.json``  ({"traj": [{"robot_cfg": {...}}, ...]})

Typical use:
    # a) export world-frame joints, so the depth scale can be measured
    python sim_teleop/make_hand_traj.py --seq mug_pick --export-world

    # b) measure the depth correction (needs the recording's depth PNGs)
    python sim_teleop/estimate_depth_scale.py --world-npy ... --npz ... --depth-dir ...

    # c) full conversion + retargeting
    python sim_teleop/make_hand_traj.py --seq mug_pick \
        --camera-pose-json data/runs/<run>/scene/camera_frame_pose.json \
        --depth-scale-json outputs/sim_teleop/mug_pick/depth_scale.json

Everything lands in outputs/sim_teleop/<seq>/. Next: sim_teleop/package_run.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_teleop.teleop_config import (  # noqa: E402
    CONDA_BIN,
    DYNHAMR_ROOT,
    ENV_DYNHAMR,
    ENV_RETARGET,
    ROBOT_TAG_INDEX,
    TELEOP_OUT_ROOT,
    V2S2R_MAIN,
)


def _run(cmd: list[str], cwd: Path, extra_env: dict | None = None) -> None:
    env = os.environ.copy()
    env.update(extra_env or {})
    print(f"[traj] $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def _conda(env_name: str, args: list) -> list:
    return [CONDA_BIN, "run", "--no-capture-output", "-n", env_name] + args


def find_seq_dir(seq: str, explicit: str | None) -> Path:
    """The Dyn-HaMR result folder (holds smooth_fit/*_world_results.npz)."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    work_link = TELEOP_OUT_ROOT / seq / "dynhamr_result"      # made by run_hand_capture.sh
    candidates.append(work_link)
    candidates += [Path(p).parent.parent for p in glob.glob(
        str(DYNHAMR_ROOT / "outputs" / "logs" / "video-custom" / "*" / f"{seq}-*"
            / "smooth_fit" / "*_world_results.npz"))]
    candidates.append(V2S2R_MAIN / "hand_estimation_dyn-hamr" / seq)
    for c in candidates:
        if c and Path(c).is_dir() and glob.glob(str(Path(c) / "smooth_fit" / "*_world_results.npz")):
            return Path(c).resolve()
    raise FileNotFoundError(
        f"no Dyn-HaMR result for seq {seq!r} (looked at {[str(c) for c in candidates]}); "
        f"run sim_teleop/run_hand_capture.sh first, or pass --seq-dir")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="sequence name (video basename)")
    ap.add_argument("--seq-dir", default=None,
                    help="Dyn-HaMR result dir (default: auto-discover from --seq)")
    ap.add_argument("--phase", default="smooth_fit")
    ap.add_argument("--camera-pose-json", default=None,
                    help="AprilTag calib (camera_frame_pose.json) of the RECORDING SETUP. "
                    "Strongly recommended - without it the camera->robot transform is a guess "
                    "(correct up-axis, yaw unknown).")
    ap.add_argument("--tag-index", type=int, default=ROBOT_TAG_INDEX)
    ap.add_argument("--depth-scale", type=float, default=None,
                    help="explicit depth correction (converter divides by it)")
    ap.add_argument("--depth-scale-json", default=None,
                    help="json from estimate_depth_scale.py (reads depth_scale_for_converter)")
    ap.add_argument("--export-world", action="store_true",
                    help="only export (T,21,3) WORLD-frame joints for depth-scale estimation")
    ap.add_argument("--out-dir", default=None,
                    help="work dir (default outputs/sim_teleop/<seq>)")
    args = ap.parse_args()

    seq_dir = find_seq_dir(args.seq, args.seq_dir)
    out_dir = Path(args.out_dir) if args.out_dir else TELEOP_OUT_ROOT / args.seq
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[traj] seq dir : {seq_dir}")
    print(f"[traj] work dir: {out_dir}")

    converter = V2S2R_MAIN / "retargeting_kinova" / "dynhamr_to_skeleton.py"
    if not converter.is_file():
        raise FileNotFoundError(f"{converter} missing - set V2S2R_MAIN")
    conv_env = {"DYNHAMR_ROOT": str(DYNHAMR_ROOT)}

    # ---- world-frame export only (for the depth-scale measurement) ----
    if args.export_world:
        world_npy = out_dir / "hand_world.npy"
        _run(_conda(ENV_DYNHAMR, ["python", str(converter),
                                  "--seq-dir", str(seq_dir), "--phase", args.phase,
                                  "--frame", "world", "--out", str(world_npy)]),
             cwd=V2S2R_MAIN, extra_env=conv_env)
        npz = sorted(glob.glob(str(seq_dir / args.phase / "*_world_results.npz")))[-1]
        print(f"[traj] world joints -> {world_npy}")
        print(f"[traj] next: python sim_teleop/estimate_depth_scale.py "
              f"--world-npy {world_npy} --npz {npz} --depth-dir <recording depth folder>")
        return 0

    # ---- resolve the depth correction ----
    depth_scale = args.depth_scale
    if depth_scale is None and args.depth_scale_json:
        depth_scale = float(json.loads(Path(args.depth_scale_json).read_text())
                            ["depth_scale_for_converter"])
    if depth_scale is None:
        auto = out_dir / "depth_scale.json"
        if auto.is_file():
            depth_scale = float(json.loads(auto.read_text())["depth_scale_for_converter"])
            print(f"[traj] using depth correction from {auto}")
    if depth_scale is None:
        depth_scale = 1.0
        print("[traj][WARN] no depth correction given (--depth-scale[-json]); using 1.0. "
              "The hand may sit at the wrong distance - run estimate_depth_scale.py.")
    else:
        print(f"[traj] depth correction: divide camera-frame points by {depth_scale:.4f}")

    # ---- 3. MANO -> robot-base keypoints ----
    skeleton_npy = out_dir / "skeleton_motion_robot_base.npy"
    conv_cmd = ["python", str(converter),
                "--seq-dir", str(seq_dir), "--phase", args.phase,
                "--frame", "robot_base", "--out", str(skeleton_npy),
                "--depth-scale", str(depth_scale)]
    if args.camera_pose_json:
        conv_cmd += ["--camera-pose-json", str(Path(args.camera_pose_json).resolve()),
                     "--tag-index", str(args.tag_index)]
    else:
        print("[traj][WARN] no --camera-pose-json: approximate camera->robot transform "
              "(up-axis right, yaw is a guess). Pass the recording's camera_frame_pose.json.")
    _run(_conda(ENV_DYNHAMR, conv_cmd), cwd=V2S2R_MAIN, extra_env=conv_env)
    if not skeleton_npy.is_file():
        raise RuntimeError(f"converter did not produce {skeleton_npy}")

    # ---- 4. keypoints -> Kinova+LEAP joints (IK) ----
    # retarget_leap writes retarget_kinova_leap.json NEXT TO the input npy.
    _run(_conda(ENV_RETARGET, ["python", "-m", "retargeting_kinova.retarget_leap",
                               "--cfg.hand-world-npy", str(skeleton_npy)]),
         cwd=V2S2R_MAIN)
    traj_json = skeleton_npy.parent / "retarget_kinova_leap.json"
    if not traj_json.is_file():
        raise RuntimeError(f"retargeting did not produce {traj_json}")

    n = len(json.loads(traj_json.read_text())["traj"])
    print(f"[traj] DONE: {n} frames -> {traj_json}")
    print(f"[traj] next: python sim_teleop/package_run.py --traj {traj_json} --name {args.seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
