#!/usr/bin/env python3
"""Stage 5: package a retargeted hand trajectory as a normal ingested run.

Builds ``data/runs/teleop_<name>/`` in exactly the layout every existing tool
already understands, so nothing downstream is new code:

    scripts/replay_trajectory.py --run teleop_<name>      # replay + record npz/video
    force_controller/... --episode outputs/teleop_<name>  # new training episodes

What goes in:
  --traj         retarget_kinova_leap.json from make_hand_traj.py
  --scene-from   an ingested donor run: its AprilTag calibration files
                 (table_frame_pose.json / camera_frame_pose.json / cam_params.txt)
                 are copied. Valid whenever the teleop video was recorded on the
                 SAME table + camera setup as the donor. Pass explicit files
                 with --table-pose/--camera-pose/--cam-params otherwise.
  --objects-from a donor run whose scene objects (URDF/meshes + poses) to place
                 in the scene. Default: no objects (robot + table only).
                 With objects, replayed episodes contain real contact forces -
                 which is what force_controller needs.

Frame rate: the replay convention is 1 frame = 0.1 s (10 Hz). If the video was
processed at a different rate, pass --source-fps and the trajectory is
resampled by nearest-frame picking.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sim_teleop.teleop_config import DEFAULT_SCENE_DONOR, RUNS_DIR, TARGET_FPS  # noqa: E402

KEY_FRAME_NAMES = ["hand_pose_frame", "pregrasp_frame", "contact_frame",
                   "pre_interaction_frame", "interaction_frame", "drop_frame"]


def resample(traj: list, source_fps: float, target_fps: float = TARGET_FPS) -> list:
    if abs(source_fps - target_fps) < 1e-6:
        return traj
    n_out = max(1, int(round(len(traj) * target_fps / source_fps)))
    idx = np.clip(np.round(np.arange(n_out) * source_fps / target_fps).astype(int),
                  0, len(traj) - 1)
    print(f"[package] resampling {len(traj)} frames @ {source_fps:g} fps "
          f"-> {n_out} frames @ {target_fps:g} fps")
    return [traj[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", required=True, help="retarget_kinova_leap.json")
    ap.add_argument("--name", required=True, help="run name -> data/runs/teleop_<name>")
    ap.add_argument("--scene-from", default=DEFAULT_SCENE_DONOR,
                    help=f"donor run for the calibration files (default {DEFAULT_SCENE_DONOR})")
    ap.add_argument("--table-pose", default=None, help="explicit table_frame_pose.json")
    ap.add_argument("--camera-pose", default=None, help="explicit camera_frame_pose.json")
    ap.add_argument("--cam-params", default=None, help="explicit cam_params.txt")
    ap.add_argument("--objects-from", default=None,
                    help="donor run whose scene objects to include (default: none)")
    ap.add_argument("--source-fps", type=float, default=TARGET_FPS,
                    help="fps of the trajectory (default 10 = no resampling)")
    for k in KEY_FRAME_NAMES:
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None,
                        help=f"optional {k} annotation")
    ap.add_argument("--force", action="store_true", help="overwrite an existing packaged run")
    args = ap.parse_args()

    run_name = args.name if args.name.startswith("teleop_") else f"teleop_{args.name}"
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        if not args.force:
            raise SystemExit(f"{run_dir} exists - pass --force to overwrite")
        shutil.rmtree(run_dir)
    (run_dir / "scene").mkdir(parents=True)

    # ---- trajectory ----
    src = json.loads(Path(args.traj).read_text())
    traj = resample(src["traj"], args.source_fps)
    n = len(traj)
    key_frames = {k: getattr(args, k) for k in KEY_FRAME_NAMES}
    out_traj = {
        "traj_id": run_name,
        "total_frames": n,
        **key_frames,
        "traj": traj,
    }
    (run_dir / "retarget_kinova_leap_optimized.json").write_text(json.dumps(out_traj, indent=2))

    # ---- calibration files (the robot base + camera pose come from these) ----
    donor_scene = RUNS_DIR / args.scene_from / "scene"
    picks = {
        "table_frame_pose.json": args.table_pose,
        "camera_frame_pose.json": args.camera_pose,
        "cam_params.txt": args.cam_params,
    }
    for fname, explicit in picks.items():
        src_path = Path(explicit) if explicit else donor_scene / fname
        if not src_path.is_file():
            raise FileNotFoundError(
                f"missing calibration file {src_path} "
                f"(donor {args.scene_from!r}; or pass --{fname.split('.')[0].replace('_', '-')})")
        shutil.copy2(src_path, run_dir / "scene" / fname)
    calib_src = "explicit files" if any(picks.values()) else f"donor {args.scene_from}"
    print(f"[package] calibration from {calib_src}")

    # ---- objects (optional) ----
    objects, static_keys, manipulated_key, scene_run = [], [], None, run_name
    mesh_scale = None
    hand_effort = None
    if args.objects_from:
        donor_dir = RUNS_DIR / args.objects_from
        donor_meta = json.loads((donor_dir / "run_meta.json").read_text())
        scene_run = donor_meta["scene_run"]        # existing USDs under assets/usd/scenes/<...>
        static_keys = donor_meta.get("static_keys", [])
        manipulated_key = donor_meta.get("manipulated_key")
        mesh_scale = donor_meta.get("object_mesh_scale")
        hand_effort = donor_meta.get("hand_effort_limit")     # authored scenes: real LEAP motor torque
        objects = donor_meta["objects"]
        for entry in objects:
            for key in ("urdf_file", "mesh_file"):
                src_f = donor_dir / entry[key]
                dst_f = run_dir / entry[key]
                dst_f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_f, dst_f)
        print(f"[package] {len(objects)} objects from {args.objects_from} "
              f"(scene_run={scene_run}, manipulated={manipulated_key})")
    else:
        print("[package] no objects: robot + table only "
              "(pass --objects-from <run> for contact-rich episodes)")

    # ---- run_meta ----
    meta = {
        "traj_run": run_name,
        "scene_run": scene_run,
        "source": "sim_teleop",
        "teleop_traj_source": str(Path(args.traj).resolve()),
        "total_frames": n,
        "num_traj_frames": n,
        "key_frames": key_frames,
        "object_mesh_scale": mesh_scale,
        "hand_effort_limit": hand_effort,
        "traj_id": run_name,
        "objects": objects,
        "manipulated_key": manipulated_key,
        "static_keys": static_keys,
        "flow_files": [],
        "mesh_transfer": [],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- validate: the run must load through the standard loader ----
    from v2s2r_isaaclab.scene_spec import describe, load_run_spec
    spec = load_run_spec(run_dir)
    print("=" * 78)
    print(describe(spec))
    print("=" * 78)
    print(f"[package] OK -> {run_dir}  ({n} frames = {n / TARGET_FPS:.1f} s)")
    print(f"[package] replay it:  python scripts/replay_trajectory.py --run {run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
