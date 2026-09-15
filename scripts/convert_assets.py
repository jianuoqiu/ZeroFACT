#!/usr/bin/env python3
"""Convert the Kinova+LEAP robot and every ingested scene's object URDFs into USD for Isaac Lab.

Outputs (all inside this project, so nothing depends on the Video2Sim2Real tree at run time)::

    assets/usd/robot/kinova_leap.usd
    assets/usd/robot/robot_info.json          # joint / body names as Isaac Lab reports them
    assets/usd/scenes/<scene_run>/obj_XXXX.usd
    assets/usd/index.json                     # what was converted, from which URDF, with which settings

Collision settings mirror the Isaac Gym asset options used by the reference replay:

* robot  -> convex hull per collision mesh   (Isaac Gym: no VHACD on the robot asset)
* object -> convex decomposition             (Isaac Gym: ``vhacd_enabled = True``)

Objects are converted **dynamic** (``fix_base=False``); a run that needs a welded object gets it at
spawn time via ``kinematic_enabled=True``, so one USD serves every run that shares a scene.

Usage::

    python scripts/convert_assets.py                 # convert everything that is missing
    python scripts/convert_assets.py --force         # re-convert
    python scripts/convert_assets.py --runs run_2026-08-11_17-29-12
    python scripts/convert_assets.py --robot-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from zerofact.runtime import prepare_display  # noqa: E402  (must run before AppLauncher)

# ---------------------------------------------------------------------------------------------
# CLI + Isaac Sim startup
# ---------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="ingested data folder")
parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd", help="USD output folder")
parser.add_argument("--runs", nargs="*", default=None, help="only convert objects for these traj runs")
parser.add_argument("--force", action="store_true", help="re-convert even if the USD already exists")
parser.add_argument("--robot-only", action="store_true", help="only convert the robot")
parser.add_argument("--objects-only", action="store_true", help="only convert scene objects")
parser.add_argument(
    "--object-collider",
    choices=["convex_decomposition", "convex_hull"],
    default="convex_decomposition",
    help="collision approximation for scene objects (Isaac Gym used VHACD -> convex_decomposition)",
)
parser.add_argument("--skip-inspect", action="store_true", help="do not load the robot USD to dump joint names")

prepare_display()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True                      # conversion never needs a window

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------------------
# Everything below needs the app running
# ---------------------------------------------------------------------------------------------
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from zerofact.naming import format_mapping, match_joint_names  # noqa: E402
from zerofact.scene_spec import ARM_JOINT_NAMES, LEAP_JOINT_NAMES, LEAP_JOINT_RENAME  # noqa: E402

# PD gains baked into the USD drives. The replay's ImplicitActuatorCfg overwrites these at runtime
# (see zerofact/replay.py), but keeping them consistent avoids a surprising first step if the
# USD is ever opened directly in Isaac Sim.
ARM_STIFFNESS, ARM_DAMPING = 400.0, 40.0
HAND_STIFFNESS, HAND_DAMPING = 350.0, 12.0


def convert_robot(urdf_path: Path, out_dir: Path, force: bool) -> Path:
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(out_dir),
        usd_file_name="kinova_leap.usd",
        force_usd_conversion=force,
        make_instanceable=False,          # single robot; keeps per-prim edits simple and reliable
        fix_base=True,                    # Isaac Gym: robot_asset_options.fix_base_link = True
        # Isaac Gym keeps every link (AssetOptions.collapse_fixed_joints defaults to False), and the
        # replay reads palm_lower / leap_mount / *fingertip poses, so do not merge them away.
        merge_fixed_joints=False,
        self_collision=False,             # Isaac Gym: create_actor(..., collision_group=1) w/o self-collision
        collider_type="convex_hull",      # Isaac Gym: no VHACD on the robot asset
        link_density=0.0,                 # every link in this URDF carries inertial data
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={r"^joint_[1-7]$": ARM_STIFFNESS, r"^leap_j\d+$": HAND_STIFFNESS},
                damping={r"^joint_[1-7]$": ARM_DAMPING, r"^leap_j\d+$": HAND_DAMPING},
            ),
        ),
    )
    converter = UrdfConverter(cfg)
    return Path(converter.usd_path)


def convert_object(urdf_path: Path, out_dir: Path, collider: str, force: bool) -> Path:
    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(out_dir),
        usd_file_name=f"{urdf_path.stem}.usd",
        force_usd_conversion=force,
        make_instanceable=False,
        fix_base=False,                   # static objects are pinned at spawn time instead
        merge_fixed_joints=True,
        self_collision=False,
        collider_type=collider,
        link_density=0.0,                 # honour the URDF mass/inertia (Isaac Gym: override_*=False)
        joint_drive=None,                 # single-link object, no joints
    )
    converter = UrdfConverter(cfg)
    return Path(converter.usd_path)


def inspect_robot(usd_path: Path) -> dict:
    """Spawn the converted robot once and record the names/limits Isaac Lab actually reports."""
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device))

    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={},
    )
    robot = Articulation(cfg)
    sim.reset()

    info = {
        "usd": str(usd_path),
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "num_joints": int(robot.num_joints),
        "num_bodies": int(robot.num_bodies),
        "joint_pos_limits": robot.data.joint_pos_limits[0].cpu().numpy().tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().numpy().tolist(),
    }

    urdf_names = ARM_JOINT_NAMES + [LEAP_JOINT_RENAME[n] for n in LEAP_JOINT_NAMES]
    mapping = match_joint_names(urdf_names, info["joint_names"])
    info["urdf_to_sim_joint"] = mapping

    print("\n[convert_assets] robot articulation:")
    print(f"  bodies ({info['num_bodies']}): {info['body_names']}")
    print(f"  joints ({info['num_joints']}):")
    print(format_mapping(mapping, info["joint_names"]))

    del robot
    sim.clear_all_callbacks()
    sim.clear_instance()
    return info


def main() -> int:
    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.json"
    index = {"robot": None, "scenes": {}}
    if index_path.is_file():
        with open(index_path) as f:
            index = json.load(f)

    # ---------------- robot ----------------
    if not args.objects_only:
        robot_urdf = data_dir / "robot" / "kinova_leap_description" / "v12_vision.urdf"
        if not robot_urdf.is_file():
            print(f"[ERR] robot URDF missing: {robot_urdf} (run scripts/ingest_data.py first)")
            return 1
        robot_out = out_dir / "robot"
        print(f"[convert_assets] robot: {robot_urdf}")
        robot_usd = convert_robot(robot_urdf, robot_out, args.force)
        print(f"[convert_assets] robot USD -> {robot_usd}")

        info = None
        if not args.skip_inspect:
            info = inspect_robot(robot_usd)
            with open(robot_out / "robot_info.json", "w") as f:
                json.dump(info, f, indent=2)
            print(f"[convert_assets] robot info -> {robot_out / 'robot_info.json'}")

        index["robot"] = {
            "urdf": str(robot_urdf),
            "usd": str(robot_usd),
            "collider_type": "convex_hull",
            "fix_base": True,
            "num_joints": (info or {}).get("num_joints"),
        }

    # ---------------- scene objects ----------------
    if not args.robot_only:
        runs_dir = data_dir / "runs"
        scene_to_run: dict[str, Path] = {}
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            if args.runs and run_dir.name not in args.runs:
                continue
            meta_path = run_dir / "run_meta.json"
            if not meta_path.is_file():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            scene_to_run.setdefault(meta["scene_run"], run_dir)   # one conversion per unique scene

        print(f"\n[convert_assets] {len(scene_to_run)} unique scene(s) to convert")
        for scene_run, run_dir in sorted(scene_to_run.items()):
            with open(run_dir / "run_meta.json") as f:
                meta = json.load(f)
            scene_out = out_dir / "scenes" / scene_run
            entries = []
            for obj in meta["objects"]:
                urdf_path = run_dir / obj["urdf_file"]
                print(f"[convert_assets] {scene_run}/{obj['urdf_name']} ({args.object_collider}) ...")
                # one directory per object: the converter stores its lazy-conversion hash in
                # `usd_dir/.asset_hash`, so objects sharing a directory overwrite each other's hash
                # and every object is re-converted on every invocation
                usd_path = convert_object(
                    urdf_path, scene_out / Path(obj["urdf_name"]).stem, args.object_collider, args.force
                )
                size_mb = usd_path.stat().st_size / 1e6 if usd_path.is_file() else 0.0
                print(f"[convert_assets]   -> {usd_path}  ({size_mb:.1f} MB)")
                entries.append(
                    {
                        "key": obj["key"],
                        "name": Path(obj["urdf_name"]).stem,
                        "urdf": str(urdf_path),
                        "usd": str(usd_path),
                        "is_manipulated": obj["is_manipulated"],
                    }
                )
            index["scenes"][scene_run] = {
                "source_run": run_dir.name,
                "collider_type": args.object_collider,
                "objects": entries,
            }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n[convert_assets] index -> {index_path}")
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    from zerofact.runtime import hard_exit

    hard_exit(simulation_app, status)
