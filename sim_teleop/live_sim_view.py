#!/usr/bin/env python3
"""Live sim window for teleop: the robot mirrors the hand streamed by live_teleop.py.

Runs in the ``env_isaaclab`` conda env.  Loads the validated scene (robot + table, plus a
donor run's objects as a WELDED visual reference by default), opens the Isaac Sim window,
and continuously PD-tracks the newest joint target received over UDP from
``sim_teleop/live_teleop.py``.  Pure visualization - nothing is recorded here; the episode
data comes from replaying the packaged trajectory with scripts/replay_trajectory.py.

    conda run -n env_isaaclab python sim_teleop/live_sim_view.py                 # GUI
    conda run -n env_isaaclab python sim_teleop/live_sim_view.py --no-window \
        --max-seconds 30                                                        # smoke test

The donor objects are welded so a jittery pre-recording hand cannot scatter them - they
mark WHERE the real objects sit on the real table (grasp there!).  Pass --dynamic-objects
to let them react, or --no-objects for robot + table only.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _make_noncolliding(prim_path: str) -> tuple[int, int]:
    """Disable every collider under ``prim_path``, de-instancing first so the collision
    prims (authored in a referenced, instanced layer) become editable. Isaac Lab spawns
    object/robot USDs instanceable, so a plain CollisionPropertiesCfg or a naive USD sweep
    silently no-ops on them. Call before ``sim.reset()``. Returns (found, disabled)."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    # de-instance by PATH (re-fetching each time) so we never hold a proxy handle that a
    # SetInstanceable() call invalidates; repeat because instances can nest.
    for _ in range(6):
        paths = [p.GetPath() for p in Usd.PrimRange(root) if p.IsInstance()]
        if not paths:
            break
        for path in paths:
            stage.GetPrimAtPath(path).SetInstanceable(False)
    found = disabled = 0
    for prim in Usd.PrimRange(root):                   # now descends into real children
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            found += 1
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False):
                disabled += 1
    return found, disabled


def _cli() -> int:
    import argparse

    from sim_teleop.teleop_config import DEFAULT_SCENE_DONOR, LIVE_UDP_ADDR
    from zerofact.runtime import check_memory, prepare_display  # before AppLauncher

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-from", default=DEFAULT_SCENE_DONOR,
                        help="ingested run whose calibration (robot base) + objects to show")
    parser.add_argument("--floating", action="store_true",
                        help="show the free-floating LEAP hand (live_teleop.py --floating) "
                        "instead of the full arm+hand robot")
    parser.add_argument("--no-objects", action="store_true", help="robot + table only")
    parser.add_argument("--dynamic-objects", action="store_true",
                        help="objects react to contact instead of being welded in place")
    parser.add_argument("--port", type=int, default=LIVE_UDP_ADDR[1])
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
    parser.add_argument("--no-window", action="store_true",
                        help="no window (smoke test); default is the GUI")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="auto-exit after this long (tests)")
    parser.add_argument("--gui-render-interval", type=int, default=2)
    parser.add_argument("--no-compat-rendering", action="store_true")

    prepare_display(require_window="--no-window" not in sys.argv)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    gui = not args.no_window

    args.headless = not gui
    if gui and not args.no_compat_rendering:
        # same broken-install workaround as teleop_player.py / replay_trajectory.py
        args.experience = "isaaclab.python.headless.kit"
        exts = ["omni.replicator.core", "omni.kit.viewport.rtx", "omni.kit.material.library",
                "omni.kit.mainwindow", "omni.kit.viewport.window", "omni.kit.manipulator.camera",
                "omni.kit.window.toolbar", "omni.kit.window.status_bar"]
        kit_args = " ".join(f"--enable {e}" for e in exts) + " --/isaaclab/cameras_enabled=true"
        args.kit_args = f"{args.kit_args} {kit_args}".strip() if getattr(args, "kit_args", "") else kit_args
        args.enable_cameras = True

    check_memory(required_gb=14.0 if gui else 8.0)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    status = 1
    try:
        import torch

        from isaaclab.sim import SimulationContext

        from zerofact.naming import format_mapping, match_joint_names
        from zerofact.replay import ReplayConfig, build_scene, check_scene, make_simulation_cfg
        from zerofact.scene_spec import describe, load_joint_name_map, load_run_spec

        run_dir = args.data_dir / "runs" / args.scene_from
        spec = load_run_spec(run_dir)
        if args.no_objects:
            spec.objects = []
        elif not args.dynamic_objects:
            for obj in spec.objects:
                obj.is_static = True
        print(describe(spec))

        cfg = ReplayConfig(device=args.device, render=False, save_images=False, video=False,
                           hand_cam=False, contact_sensors=False, force_vis=False,
                           flow_points=0, gui=gui, gui_render_interval=args.gui_render_interval,
                           realtime=False,
                           output_dir=PROJECT_ROOT / "outputs" / "sim_teleop" / "live_view")

        robot_usd = args.usd_dir / "robot" / "kinova_leap.usd"
        object_usds = {}
        for obj in spec.objects:
            scene_dir = args.usd_dir / "scenes" / spec.scene_run
            cand = [scene_dir / obj.name / f"{obj.name}.usd", scene_dir / f"{obj.name}.usd"]
            usd = next((c for c in cand if c.is_file()), None)
            if usd is None:
                raise SystemExit(f"object USD missing: {cand[0]} (run scripts/convert_assets.py)")
            object_usds[obj.key] = usd

        sim = SimulationContext(make_simulation_cfg(cfg, spec))
        if args.floating:
            # minimal scene: lights, ground, table, welded objects, free-floating LEAP hand
            import isaaclab.sim as sim_utils
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

            from zerofact.replay import (
                ENV_PRIM,
                GROUND_FRICTION,
                HAND_DAMPING,
                HAND_STIFFNESS,
                TABLE_FRICTION,
                TABLE_RESTITUTION,
                _spawn_static_box,
                hand_effort_limit,
                object_collision_props,
                object_rigid_props,
                prepare_object_colliders,
            )
            from zerofact.scene_spec import quat_xyzw_to_wxyz

            hand_usd = args.usd_dir / "robot" / "leap_float.usd"
            if not hand_usd.is_file():
                raise SystemExit(f"{hand_usd} missing - run: conda run -n env_isaaclab "
                                 "python sim_teleop/make_float_hand.py")
            dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.9, 0.95))
            dome.func("/World/DomeLight", dome)
            distant = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0), angle=1.0)
            distant.func("/World/DistantLight", distant, orientation=(0.9238795, 0.0, 0.3826834, 0.0))
            _spawn_static_box("/World/GroundPlane", size=(40.0, 40.0, 0.2),
                              position=np.array([0.0, 0.0, -0.1]), friction=GROUND_FRICTION,
                              color=(0.25, 0.25, 0.27))
            # --dynamic-objects: objects are DYNAMIC (gravity, collision) resting on a
            # COLLIDABLE table, so you can actually grasp/move them and get live contact
            # feedback (screwing, picking). Default: objects welded + collision stripped =
            # a stable pose preview the hand passes through (no contact feedback, but the
            # teleported hand can't blow up on an immovable object).
            graspable = args.dynamic_objects
            if graspable:
                _spawn_static_box(f"{ENV_PRIM}/Table",
                                  size=tuple(float(v) for v in spec.table_size),
                                  position=spec.table_position, friction=TABLE_FRICTION,
                                  restitution=TABLE_RESTITUTION)
            else:
                _tbl = sim_utils.CuboidCfg(
                    size=tuple(float(v) for v in spec.table_size),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.55, 0.42, 0.30), roughness=0.8))
                _tbl.func(f"{ENV_PRIM}/Table", _tbl,
                          translation=tuple(float(v) for v in spec.table_position))
            objects = {}
            for obj in spec.objects:
                objects[obj.key] = RigidObject(RigidObjectCfg(
                    prim_path=f"{ENV_PRIM}/{obj.name}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=str(object_usds[obj.key]),
                        rigid_props=object_rigid_props(obj, kinematic=not graspable),
                        collision_props=object_collision_props(obj),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=tuple(float(v) for v in obj.position),
                        rot=tuple(float(v) for v in quat_xyzw_to_wxyz(obj.quat_xyzw)),
                    ),
                ))
            robot = Articulation(ArticulationCfg(
                prim_path=f"{ENV_PRIM}/FloatHand",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(hand_usd),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False, fix_root_link=False),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2)),
                actuators={"hand": ImplicitActuatorCfg(joint_names_expr=[r"leap_j\d+"],
                                                       stiffness=HAND_STIFFNESS,
                                                       damping=HAND_DAMPING,
                                                       effort_limit_sim=hand_effort_limit(spec))},
            ))
            if graspable:
                # same URDF friction material + collider approximation (convex decomposition or
                # SDF for threaded parts) as the validated scene, otherwise PhysX cooks the nut
                # into a convex hull with no hole and it sinks into the screw
                for obj in spec.objects:
                    prepare_object_colliders(obj, f"{ENV_PRIM}/{obj.name}", cfg)
                print("[view] floating GRASP mode: objects are dynamic on a solid table - "
                      "grasp/move them for live contact feedback (--dynamic-objects)",
                      flush=True)
            else:
                # welded preview: strip object collision so the teleported hand can't ram
                # its fingers into an immovable object and fling the links apart
                tot_found = tot_off = 0
                for obj in spec.objects:
                    f, d = _make_noncolliding(f"{ENV_PRIM}/{obj.name}")
                    tot_found += f; tot_off += d
                print(f"[view] floating preview: disabled {tot_off}/{tot_found} object colliders "
                      "(pose-only preview, hand passes through; add --dynamic-objects to "
                      "grasp). Contact is validated in the replay.", flush=True)
            scene_objects = objects
        else:
            scene = build_scene(spec, cfg, robot_usd, object_usds)
            robot = scene.robot
            scene_objects = scene.objects
        sim.reset()
        if not args.floating:
            check_scene(scene, spec)
        if gui:
            sim.set_camera_view(eye=(1.2, 0.9, 1.9), target=(-0.4, -0.4, 0.85))

        device = robot.device
        physics_dt = cfg.physics_dt

        # floating: hand poses arrive in ROBOT-BASE coordinates; compose with the base pose
        from scipy.spatial.transform import Rotation as Rsc

        R_base = Rsc.from_quat(spec.robot_quat_xyzw).as_matrix()
        t_base = np.asarray(spec.robot_position, dtype=np.float64)
        pending_root = None
        # floating + dynamic objects: the palm must be able to STOP at an object, so the root is
        # pulled by a capped spring-damper wrench instead of being teleported (see root_drive.py)
        root_drive = None
        if args.floating and args.dynamic_objects:
            from sim_teleop.root_drive import RootPoseDrive

            root_drive = RootPoseDrive(robot)
            print(f"[view] {root_drive.describe()} - the hand lags behind the tracker when an object blocks it",
                  flush=True)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", args.port))
        sock.setblocking(False)
        print(f"[view] listening on udp://127.0.0.1:{args.port} - start sim_teleop/live_teleop.py",
              flush=True)

        # hold the spawn pose until the first target arrives
        target = robot.data.joint_pos[0:1].clone()
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()

        msg_keys: list[str] = []                 # message keys arranged in sim joint order
        root_drive_started = False
        last_state = None
        last_seq = -1
        got_first = False
        t0 = time.time()
        n_msgs = n_steps = 0
        last_report = time.time()

        while simulation_app.is_running():
            step_t0 = time.perf_counter()

            # ---- drain the socket, keep the newest ----
            newest = None
            bye = False
            while True:
                try:
                    data, _ = sock.recvfrom(8192)
                except BlockingIOError:
                    break
                except OSError:
                    break
                try:
                    msg = json.loads(data.decode())
                except json.JSONDecodeError:
                    continue
                if msg.get("state") == "bye":
                    bye = True
                    continue
                if msg.get("seq", 0) >= last_seq and "q" in msg:
                    newest, last_seq = msg, msg.get("seq", 0)
                elif "q" not in msg and msg.get("state") != last_state:
                    last_state = msg.get("state")
                    print(f"[view] tracker state: {last_state}", flush=True)
            if bye:
                print("[view] tracker said bye - closing.", flush=True)
                break

            if newest is not None:
                n_msgs += 1
                if newest.get("state") != last_state:
                    last_state = newest.get("state")
                    print(f"[view] tracker state: {last_state}", flush=True)
                q = newest["q"]
                if not msg_keys:
                    # first target: build the name mapping exactly like the validated replay
                    keys = ([k for k in q if not k.startswith("joint_")] if args.floating
                            else list(q.keys()))
                    traj_to_urdf = load_joint_name_map(args.data_dir)
                    urdf_to_sim = match_joint_names(
                        [traj_to_urdf.get(n, n) for n in keys], list(robot.joint_names))
                    key_to_sim = {n: urdf_to_sim[traj_to_urdf.get(n, n)] for n in keys}
                    sim_to_key = {v: k for k, v in key_to_sim.items()}
                    missing = [n for n in robot.joint_names if n not in sim_to_key]
                    if missing:
                        raise KeyError(f"stream provides no command for sim joint(s) {missing}")
                    msg_keys = [sim_to_key[n] for n in robot.joint_names]
                    print("[view] joint mapping (simulation order):")
                    print(format_mapping(key_to_sim, list(robot.joint_names)))
                vec = torch.tensor([[q[k] for k in msg_keys]], dtype=torch.float32, device=device)
                target = vec
                if args.floating and newest.get("root") is not None:
                    r = newest["root"]                       # robot-base frame, quat xyzw
                    R_w = R_base @ Rsc.from_quat(r[3:]).as_matrix()
                    p_w = R_base @ np.asarray(r[:3], dtype=np.float64) + t_base
                    qx, qy, qz, qw = Rsc.from_matrix(R_w).as_quat()
                    pending_root = torch.tensor(
                        [[p_w[0], p_w[1], p_w[2], qw, qx, qy, qz]],
                        dtype=torch.float32, device=device)
                if not got_first:
                    # teleport to the first tracked pose instead of swinging across the table
                    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
                    got_first = True
                    print("[view] first target received - robot teleported to the hand pose.",
                          flush=True)

            if args.floating and pending_root is not None:
                if root_drive is not None:
                    if not root_drive_started:
                        # first tracked pose: teleport once, then let the drive take over
                        robot.write_root_pose_to_sim(pending_root)
                        robot.write_root_velocity_to_sim(
                            torch.zeros((1, 6), dtype=torch.float32, device=device))
                        root_drive_started = True
                    root_drive.set_target(pending_root)
                    root_drive.apply()
                else:
                    robot.write_root_pose_to_sim(pending_root)
                    robot.write_root_velocity_to_sim(
                        torch.zeros((1, 6), dtype=torch.float32, device=device))
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            for obj in scene_objects.values():
                obj.write_data_to_sim()
            sim.step(render=False)
            robot.update(physics_dt)
            for obj in scene_objects.values():
                obj.update(physics_dt)
            n_steps += 1
            if gui and n_steps % max(1, args.gui_render_interval) == 0:
                sim.render()

            if time.time() - last_report >= 5.0:
                err = float((robot.data.joint_pos[0:1] - target).abs().max())
                lag = f"   root lag {root_drive.position_error() * 1000:.0f} mm" if root_drive is not None else ""
                print(f"[view] {n_msgs / (time.time() - last_report):5.1f} msg/s   "
                      f"|q - target|max {err:.3f} rad   state {last_state}{lag}", flush=True)
                n_msgs, last_report = 0, time.time()
            if args.max_seconds is not None and time.time() - t0 > args.max_seconds:
                print("[view] --max-seconds reached.", flush=True)
                break

            # real-time pacing (viewer runs at 1x speed like the GUI replay)
            rest = physics_dt - (time.perf_counter() - step_t0)
            if rest > 0:
                time.sleep(rest)

        status = 0
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        from zerofact.runtime import hard_exit

        hard_exit(simulation_app, status)
    return status


if __name__ == "__main__":
    raise SystemExit(_cli())
