#!/usr/bin/env python3
"""Build the floating-LEAP-hand asset for arm-free teleop data collection.

Derives ``leap_float.urdf`` (the hand subtree rooted at ``leap_mount``) from THIS repo's
processed robot URDF - so joint names (``leap_j*``), armature-baked inertias and meshes
are identical to the validated full robot - then converts it to
``assets/usd/robot/leap_float.usd`` with the same converter settings as the full robot,
except ``fix_base=False`` (the viewer writes the root pose every physics step).

Run once (env_isaaclab, headless):
    conda run -n env_isaaclab python sim_teleop/make_float_hand.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SRC_URDF = PROJECT_ROOT / "data" / "robot" / "kinova_leap_description" / "v12_vision.urdf"
FLOAT_URDF = SRC_URDF.parent / "leap_float.urdf"
OUT_DIR = PROJECT_ROOT / "assets" / "usd" / "robot"
ROOT_LINK = "leap_mount"


def derive_float_urdf(src: Path = SRC_URDF, dst: Path = FLOAT_URDF,
                      root_link: str = ROOT_LINK) -> Path:
    tree = ET.parse(src)
    robot = tree.getroot()
    joints = robot.findall("joint")
    children = {}                                   # parent link -> [(joint, child link)]
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(
            (j, j.find("child").get("link")))

    keep_links, keep_joints = {root_link}, []
    stack = [root_link]
    while stack:
        for joint, child in children.get(stack.pop(), []):
            keep_links.add(child)
            keep_joints.append(joint)
            stack.append(child)

    for link in robot.findall("link"):
        if link.get("name") not in keep_links:
            robot.remove(link)
    for joint in joints:
        if joint not in keep_joints:
            robot.remove(joint)
    robot.set("name", "leap_float")
    tree.write(dst)
    n_j = len(keep_joints)
    print(f"[float] {dst.name}: root {root_link!r}, {len(keep_links)} links, {n_j} joints "
          f"({sum(1 for j in keep_joints if j.get('type') != 'fixed')} actuated)")
    return dst


def main() -> int:
    derive_float_urdf()

    import argparse

    from zerofact.runtime import prepare_display

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="reconvert even if the USD exists")
    prepare_display(require_window=False)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app

    status = 1
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        from zerofact.replay import HAND_DAMPING, HAND_STIFFNESS

        usd_path = OUT_DIR / "leap_float.usd"
        if usd_path.is_file() and not args.force:
            print(f"[float] {usd_path} exists (pass --force to reconvert)")
        else:
            cfg = UrdfConverterCfg(
                asset_path=str(FLOAT_URDF),
                usd_dir=str(OUT_DIR),
                usd_file_name="leap_float.usd",
                force_usd_conversion=True,
                make_instanceable=False,
                fix_base=False,                    # floating: the viewer writes the root pose
                merge_fixed_joints=False,
                self_collision=False,
                collider_type="convex_hull",
                link_density=0.0,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    drive_type="force",
                    target_type="position",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness={r"^leap_j\d+$": HAND_STIFFNESS},
                        damping={r"^leap_j\d+$": HAND_DAMPING},
                    ),
                ),
            )
            print(f"[float] converting -> {usd_path}")
            UrdfConverter(cfg)

        # sanity: load it once and report what Isaac Lab sees
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device=args.device))
        robot = Articulation(ArticulationCfg(
            prim_path="/World/FloatHand",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path)),
            actuators={"hand": ImplicitActuatorCfg(
                joint_names_expr=[r"leap_j\d+"], stiffness=HAND_STIFFNESS, damping=HAND_DAMPING)},
        ))
        sim.reset()
        print(f"[float] loaded: {robot.num_joints} joints, {robot.num_bodies} bodies")
        print(f"[float] joints: {list(robot.joint_names)}")
        assert robot.num_joints == 16, "expected the 16 LEAP joints"
        status = 0
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        from zerofact.runtime import hard_exit

        hard_exit(app, status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
