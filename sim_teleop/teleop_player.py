#!/usr/bin/env python3
"""Stage 6: play a teleop hand trajectory in the simulator ("update the hand in sim").

Two ways to use this file:

**As a module** (the env is already loaded - the case the pipeline assumes):
    from sim_teleop.teleop_player import resolve_sim_targets, play_trajectory
    targets = resolve_sim_targets(spec, scene.robot)      # traj json -> sim joint order
    log = play_trajectory(scene, sim, targets, cfg)       # stream it, frame by frame

**As a CLI** (quick look at a packaged run, GUI or headless):
    python sim_teleop/teleop_player.py --run teleop_mug_pick --gui
    python sim_teleop/teleop_player.py --run teleop_mug_pick --no-render --max-frames 40

The player reproduces the validated replay semantics exactly: 12 physics steps
of 1/120 s per trajectory frame, and the first 2 steps of each frame still hold
the previous frame's command (the recorded pipeline's stale-target behaviour).

For full data recording (contact forces, videos, plots, replay_data.npz that
force_controller can consume) do NOT use this player - use the validated
recorder on the packaged run:
    python scripts/replay_trajectory.py --run teleop_<name>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------------------------------
# Module API (import these when the env is already loaded)
# ----------------------------------------------------------------------------------------------
def resolve_sim_targets(spec, robot) -> np.ndarray:
    """Trajectory JSON joints -> ``[T, J]`` float array in the live articulation's joint order.

    Same recipe as the validated replay (name-based, never positional): trajectory
    name -> URDF name (data/robot/joint_name_map.json) -> sim name (conservative matcher).
    """
    from v2s2r_isaaclab.naming import match_joint_names
    from v2s2r_isaaclab.scene_spec import load_joint_name_map

    traj_to_urdf = load_joint_name_map(spec.root.parents[1])
    urdf_to_sim = match_joint_names(
        [traj_to_urdf.get(n, n) for n in spec.trajectory.joint_names], list(robot.joint_names)
    )
    traj_to_sim = {n: urdf_to_sim[traj_to_urdf.get(n, n)] for n in spec.trajectory.joint_names}
    sim_to_traj = {v: k for k, v in traj_to_sim.items()}
    missing = [n for n in robot.joint_names if n not in sim_to_traj]
    if missing:
        raise KeyError(f"trajectory provides no command for sim joint(s) {missing}")
    return spec.trajectory.reorder([sim_to_traj[n] for n in robot.joint_names])


def play_trajectory(
    scene,
    sim,
    targets: np.ndarray,           # [T, J] in sim joint order (resolve_sim_targets)
    physics_dt: float = 1.0 / 120.0,
    steps_per_frame: int = 12,
    stale_target_steps: int = 2,   # replay convention: first 2 steps hold the previous command
    max_frames: int | None = None,
    realtime: bool = False,
    gui: bool = False,
    gui_render_interval: int = 2,
    on_frame=None,                 # optional callback(frame_idx, joint_pos_np)
) -> dict:
    """Stream the joint targets into the loaded scene; return what was measured.

    Assumes ``sim.reset()`` has already run (the env is loaded). Resets the robot to
    the trajectory's first pose, then per frame sends the command for 12 physics steps.
    """
    import torch

    robot = scene.robot
    device = robot.device
    T = targets.shape[0] if max_frames is None else min(max_frames, targets.shape[0])
    tgt = torch.as_tensor(np.asarray(targets[:T]), dtype=torch.float32, device=device)

    # ---- reset to frame 0 (same as the replay: initial state = first command) ----
    first = tgt[0:1]
    robot.reset()
    robot.write_joint_state_to_sim(first, torch.zeros_like(first))
    robot.set_joint_position_target(first)
    robot.write_data_to_sim()
    for obj in scene.objects.values():
        obj.reset()
        obj.write_data_to_sim()

    log_pos, log_target = [], []
    t_start = time.time()
    prev = first
    for frame in range(T):
        cur = tgt[frame:frame + 1]
        for step in range(steps_per_frame):
            step_t0 = time.perf_counter()
            cmd = prev if (step < stale_target_steps and frame > 0) else cur
            robot.set_joint_position_target(cmd)
            robot.write_data_to_sim()
            for obj in scene.objects.values():
                obj.write_data_to_sim()
            sim.step(render=False)
            robot.update(physics_dt)
            for obj in scene.objects.values():
                obj.update(physics_dt)
            if gui and step % gui_render_interval == 0:
                sim.render()
            if realtime:
                rest = physics_dt - (time.perf_counter() - step_t0)
                if rest > 0:
                    time.sleep(rest)
        prev = cur
        qp = robot.data.joint_pos[0].cpu().numpy().copy()
        log_pos.append(qp)
        log_target.append(cur[0].cpu().numpy().copy())
        if on_frame is not None:
            on_frame(frame, qp)
        if frame % 20 == 0 or frame == T - 1:
            print(f"[teleop] frame {frame + 1:4d}/{T}  ({time.time() - t_start:5.1f}s)", flush=True)

    return {
        "num_frames": T,
        "joint_names": list(robot.joint_names),
        "joint_pos": np.stack(log_pos),
        "joint_target": np.stack(log_target),
        "wall_time_s": time.time() - t_start,
    }


# ----------------------------------------------------------------------------------------------
# CLI (loads the env itself, then uses the module API above)
# ----------------------------------------------------------------------------------------------
def _cli() -> int:
    import argparse

    from v2s2r_isaaclab.runtime import check_memory, prepare_display  # before AppLauncher

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="packaged run name, e.g. teleop_mug_pick")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gui", action="store_true", help="live window, paced to real time")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="(headless smoke test)")
    parser.add_argument("--save", type=Path, default=None,
                        help="save the measured joints to this .npz")
    parser.add_argument("--no-compat-rendering", action="store_true")

    prepare_display(require_window="--gui" in sys.argv)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = not args.gui
    if args.gui and not args.no_compat_rendering:
        # same broken-install workaround as run_tracking.py / replay_trajectory.py
        args.experience = "isaaclab.python.headless.kit"
        exts = ["omni.replicator.core", "omni.kit.viewport.rtx", "omni.kit.material.library",
                "omni.kit.mainwindow", "omni.kit.viewport.window", "omni.kit.manipulator.camera",
                "omni.kit.window.toolbar", "omni.kit.window.status_bar"]
        kit_args = " ".join(f"--enable {e}" for e in exts) + " --/isaaclab/cameras_enabled=true"
        args.kit_args = f"{args.kit_args} {kit_args}".strip() if getattr(args, "kit_args", "") else kit_args
        args.enable_cameras = True

    check_memory(required_gb=14.0 if args.gui else 8.0)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    status = 1
    try:
        from isaaclab.sim import SimulationContext

        from v2s2r_isaaclab.replay import ReplayConfig, build_scene, check_scene, make_simulation_cfg
        from v2s2r_isaaclab.scene_spec import describe, load_run_spec

        run_dir = args.data_dir / "runs" / args.run
        spec = load_run_spec(run_dir)
        print(describe(spec))

        cfg = ReplayConfig(device=args.device, render=False, save_images=False, video=False,
                           hand_cam=False, contact_sensors=False, force_vis=args.gui,
                           flow_points=0, gui=args.gui, gui_render_interval=2,
                           realtime=args.gui and not args.no_realtime,
                           output_dir=PROJECT_ROOT / "outputs" / "sim_teleop" / "player")

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
        scene = build_scene(spec, cfg, robot_usd, object_usds)
        sim.reset()
        check_scene(scene, spec)
        if args.gui:
            sim.set_camera_view(eye=(1.2, 0.9, 1.9), target=(-0.4, -0.4, 0.85))

        targets = resolve_sim_targets(spec, scene.robot)
        result = play_trajectory(scene, sim, targets, physics_dt=cfg.physics_dt,
                                 steps_per_frame=cfg.steps_per_frame,
                                 max_frames=args.max_frames,
                                 realtime=cfg.realtime, gui=args.gui)

        err = np.abs(result["joint_pos"] - result["joint_target"])
        print(f"[teleop] played {result['num_frames']} frames in {result['wall_time_s']:.1f}s | "
              f"PD tracking |q - target|: mean {err.mean():.4f} rad, max {err.max():.4f} rad")
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save, **{k: v for k, v in result.items()
                                              if isinstance(v, np.ndarray)},
                                joint_names=np.array(result["joint_names"]))
            print(f"[teleop] joints -> {args.save}")
        print("[teleop] full recording (forces/videos/npz): "
              f"python scripts/replay_trajectory.py --run {args.run}")
        status = 0
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        from v2s2r_isaaclab.runtime import hard_exit

        hard_exit(simulation_app, status)
    return status


if __name__ == "__main__":
    raise SystemExit(_cli())
