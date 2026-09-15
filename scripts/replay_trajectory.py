#!/usr/bin/env python3
"""Replay a Video2Sim2Real optimized trajectory (Kinova Gen3 + LEAP hand) in Isaac Lab.

Port of ``Video2Sim2Real/contact_opt/optimized_replay.py`` (Isaac Gym) onto Isaac Lab 2.3 /
Isaac Sim 5.1. The scene is rebuilt from the reconstructed demo: table, ground, the scanned scene
objects at their measured poses, the robot at its AprilTag-measured base pose and a camera at the
real camera pose.

Examples::

    # headless replay with images + video + plots
    python scripts/replay_trajectory.py --run run_2026-08-11_17-29-12

    # watch it in the Isaac Sim window
    python scripts/replay_trajectory.py --run run_2026-08-11_17-29-12 --gui

    # kinematic preview (joints teleported, no PD tracking) - Isaac Gym's --visualize
    python scripts/replay_trajectory.py --run run_2026-08-11_17-29-12 --kinematic

    # physics only, fastest
    python scripts/replay_trajectory.py --run run_2026-08-11_17-29-12 --no-render

Outputs land in ``outputs/<run>/<timestamp>/``:

    rgb_images/*.png        camera frames (one per trajectory frame, demo camera pose)
    replay.mp4              the same frames as a video
    replay_data.npz         joint / body / object / contact / flow arrays
    summary.json            configuration + tracking, object-motion, contact and flow metrics
    joint_tracking.png      commanded vs simulated joint angles
    object_tracking.png     object world position over time
    contact_forces.png      fingertip contact forces at every physics step
    contact_forces.mp4      the same plot with a cursor sweeping in step with the replay
    replay_with_forces.mp4  annotated side-by-side video: camera view (with force arrows) + plot
    flow_comparison.png     simulated object motion vs the video demo's 3D flow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from v2s2r_isaaclab.runtime import check_memory, prepare_display  # noqa: E402  (before AppLauncher)

# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", default=None, help="trajectory run name under data/runs (or a path to one)")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
parser.add_argument("--out-dir", type=Path, default=None, help="output folder (default outputs/<run>/<stamp>)")
parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window (default: headless)")
parser.add_argument("--keep-open", action="store_true", help="with --gui, keep the window open after the replay")
parser.add_argument("--no-realtime", action="store_true", help="with --gui, replay as fast as possible")
parser.add_argument(
    "--gui-render-interval",
    type=int,
    default=2,
    help="with --gui, render every Nth physics step (default 2 = 60 Hz view; 1 is smoothest, 12 fastest)",
)
parser.add_argument("--kinematic", action="store_true", help="teleport joints instead of PD tracking")
parser.add_argument("--no-render", action="store_true", help="skip all rendering (no images/video)")
parser.add_argument("--no-images", action="store_true", help="render for the video but do not save PNGs")
parser.add_argument("--no-video", action="store_true", help="do not write replay.mp4")
parser.add_argument("--save-depth", action="store_true", help="also save per-frame depth (.npy)")
parser.add_argument("--no-contact-sensors", action="store_true", help="disable fingertip contact sensors")
parser.add_argument("--no-force-vis", action="store_true", help="do not draw fingertip force arrows in the scene")
parser.add_argument("--force-vis-scale", type=float, default=0.005, help="force-arrow length in m per N (default 0.005)")
parser.add_argument("--no-force-videos", action="store_true",
                    help="skip contact_forces.mp4 and the combined replay_with_forces.mp4")
parser.add_argument("--no-hand-cam", action="store_true",
                    help="disable the close-up camera that tracks the fingertips (handcam.mp4)")
parser.add_argument("--max-frames", type=int, default=None, help="stop after N trajectory frames")
parser.add_argument("--physics-dt", type=float, default=None, help="physics step (default 1/120 = Isaac Gym dt/substeps)")
parser.add_argument(
    "--steps-per-frame",
    type=int,
    default=None,
    help="physics steps per trajectory frame (default 12 = Isaac Gym's 6 steps x 2 substeps)",
)
parser.add_argument("--settle-steps", type=int, default=0, help="physics steps before the trajectory starts")
parser.add_argument("--object-mesh-scale", type=float, default=None, help="override the run's object_mesh_scale")
parser.add_argument("--object-decomposition-error", type=float, default=None,
                    help="convex-decomposition error tolerance in percent for object colliders "
                    "(default 1.0; PhysX's own default of 10 bloats small objects by ~1.5 mm and "
                    "makes marginal grasps slip - see docs/validation.md)")
parser.add_argument("--flow-points", type=int, default=256, help="surface samples for the flow metric (0 disables)")
parser.add_argument("--video-fps", type=int, default=20)
parser.add_argument("--save-ratio", type=int, default=1, help="save every Nth camera frame")
parser.add_argument("--cam-far", type=float, default=20.0, help="camera far clip (Isaac Gym used 1.0)")
parser.add_argument(
    "--use-real-intrinsics",
    action="store_true",
    help="use cam_params.txt instead of the 90-deg-FOV intrinsics Isaac Gym forced",
)
parser.add_argument(
    "--stale-target-steps",
    type=int,
    default=None,
    help="physics steps at the start of each frame that still hold the previous command "
    "(default 2 = Isaac Gym's simulate() before the target is set)",
)
parser.add_argument("--joint-armature", type=float, default=0.0,
                    help="PhysX joint armature; keep 0 because Isaac Gym's armature is baked into the link inertias")
parser.add_argument("--static-object-ids", type=str, default=None,
                    help="comma/space separated object ids (e.g. '0000') to weld in place, overriding run_meta")
parser.add_argument("--no-compat-rendering", action="store_true",
                    help="use Isaac Lab's stock rendering experience instead of the compatibility launch")
parser.add_argument("--seed", type=int, default=0)

# --gui needs a display we can actually open a window on, which is a stricter test than the one a
# headless (offscreen-rendering) run needs; argparse has not run yet, so peek at argv.
prepare_display(require_window="--gui" in sys.argv)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# fail fast: resolving the run needs nothing from Isaac Sim, and starting Kit takes ~30 s
_runs_dir = args.data_dir / "runs"
if not args.run or not (
    (Path(args.run) / "run_meta.json").is_file() or (_runs_dir / args.run / "run_meta.json").is_file()
):
    _available = sorted(p.name for p in _runs_dir.iterdir() if p.is_dir()) if _runs_dir.is_dir() else []
    raise SystemExit(
        (f"unknown run {args.run!r}" if args.run else "--run is required")
        + ". Available runs:\n  "
        + "\n  ".join(_available)
    )

render_enabled = not args.no_render
args.headless = not args.gui
if render_enabled:
    # cameras only produce data when Kit is started with rendering enabled
    args.enable_cameras = True
    if not args.no_compat_rendering:
        # Isaac Sim 5.1's rendering experience files (isaaclab.python[.headless].rendering.kit, and
        # the windowed isaaclab.python.kit) crash this installation during extension startup:
        # omni.usd's bindings fail to register ("module 'omni.usd' has no attribute 'UsdContext'")
        # and Kit segfaults. The plain headless experience loads cleanly, so start from that and
        # switch the extensions we need on by hand - including the UI stack for --gui, which brings
        # up a normal Omniverse Kit window. Pass --no-compat-rendering to use the stock experience
        # once the install is repaired.
        args.experience = "isaaclab.python.headless.kit"
        compat_exts = [
            "omni.replicator.core",          # camera annotators
            "omni.kit.viewport.rtx",         # RTX viewport renderer
            "omni.kit.material.library",
        ]
        if args.gui:
            compat_exts += [
                "omni.kit.mainwindow",       # the window itself
                "omni.kit.viewport.window",  # the 3D viewport inside it
                "omni.kit.manipulator.camera",  # orbit/pan/zoom with the mouse
                "omni.kit.window.toolbar",   # AppLauncher._hide_stop_button() imports this
                "omni.kit.window.status_bar",
            ]
        compat_kit_args = " ".join(f"--enable {ext}" for ext in compat_exts)
        compat_kit_args += " --/isaaclab/cameras_enabled=true"
        args.kit_args = f"{args.kit_args} {compat_kit_args}".strip() if getattr(args, "kit_args", "") else compat_kit_args
elif args.gui:
    raise SystemExit("--gui needs rendering; drop --no-render.")

check_memory(required_gb=8.0 if args.no_render else (14.0 if args.gui else 12.0))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------------------
# Isaac-dependent imports
# ---------------------------------------------------------------------------------------------
from isaaclab.sim import SimulationContext  # noqa: E402

from v2s2r_isaaclab import analysis  # noqa: E402
from v2s2r_isaaclab.replay import (  # noqa: E402
    ReplayConfig,
    build_scene,
    check_scene,
    make_simulation_cfg,
    run_replay,
    write_video,
)
from v2s2r_isaaclab.scene_spec import describe, load_run_spec  # noqa: E402


def resolve_run_dir(run: str | None, data_dir: Path) -> Path:
    available = sorted(p.name for p in (data_dir / "runs").iterdir() if p.is_dir())
    if not run:
        raise SystemExit("--run is required. Available runs:\n  " + "\n  ".join(available))
    candidate = Path(run)
    if candidate.is_dir() and (candidate / "run_meta.json").is_file():
        return candidate.resolve()
    candidate = data_dir / "runs" / run
    if candidate.is_dir():
        return candidate.resolve()
    raise SystemExit(f"Run {run!r} not found. Available runs:\n  " + "\n  ".join(available))


def resolve_assets(spec, usd_dir: Path) -> tuple[Path, dict[str, Path]]:
    robot_usd = usd_dir / "robot" / "kinova_leap.usd"
    if not robot_usd.is_file():
        raise SystemExit(f"Robot USD missing: {robot_usd}\nRun: python scripts/convert_assets.py")
    object_usds = {}
    for obj in spec.objects:
        scene_dir = usd_dir / "scenes" / spec.scene_run
        candidates = [scene_dir / obj.name / f"{obj.name}.usd", scene_dir / f"{obj.name}.usd"]
        usd_path = next((c for c in candidates if c.is_file()), None)
        if usd_path is None:
            raise SystemExit(
                f"Object USD missing: {candidates[0]}\n"
                f"Run: python scripts/convert_assets.py --runs {spec.traj_run}"
            )
        object_usds[obj.key] = usd_path
    return robot_usd, object_usds


def apply_static_override(spec, tokens: str) -> None:
    """--static-object-ids '0000,0001' -> weld those scene objects (never the manipulated one)."""
    wanted = {tok.strip().lower() for tok in tokens.replace(",", " ").split() if tok.strip()}
    for obj in spec.objects:
        aliases = {obj.key.lower(), obj.name.lower(), obj.name.split("_")[-1].lower()}
        if obj.name.split("_")[-1].isdigit():
            aliases.add(str(int(obj.name.split("_")[-1])))
        should_be_static = bool(aliases & wanted) and not obj.is_manipulated
        if should_be_static != obj.is_static:
            print(f"[replay] --static-object-ids: {obj.name} static -> {should_be_static}")
        obj.is_static = should_be_static


def main() -> int:
    run_dir = resolve_run_dir(args.run, args.data_dir)
    spec = load_run_spec(run_dir, object_mesh_scale=args.object_mesh_scale)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / spec.traj_run / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReplayConfig(
        device=args.device,
        kinematic=args.kinematic,
        render=render_enabled,
        save_images=render_enabled and not args.no_images,
        save_depth=render_enabled and args.save_depth,
        video=render_enabled and not args.no_video,
        video_fps=args.video_fps,
        hand_cam=not args.no_hand_cam,
        contact_sensors=not args.no_contact_sensors,
        force_vis=not args.no_force_vis,
        force_vis_scale=args.force_vis_scale,
        flow_points=args.flow_points,
        max_frames=args.max_frames,
        settle_steps=args.settle_steps,
        save_ratio=args.save_ratio,
        cam_far=args.cam_far,
        cam_use_real_intrinsics=args.use_real_intrinsics,
        joint_armature=args.joint_armature,
        gui=args.gui,
        gui_render_interval=max(1, args.gui_render_interval),
        realtime=args.gui and not args.no_realtime,
        output_dir=out_dir,
        seed=args.seed,
    )
    if args.stale_target_steps is not None:
        cfg.stale_target_steps = args.stale_target_steps
    if args.object_decomposition_error is not None:
        cfg.object_decomposition_error = args.object_decomposition_error
    if args.static_object_ids is not None:
        apply_static_override(spec, args.static_object_ids)
    if args.physics_dt is not None:
        cfg.physics_dt = args.physics_dt
    if args.steps_per_frame is not None:
        cfg.steps_per_frame = args.steps_per_frame

    print("=" * 90)
    print(describe(spec))
    print(
        f"physics dt {cfg.physics_dt:.6f}s x {cfg.steps_per_frame} steps/frame "
        f"= {cfg.physics_dt * cfg.steps_per_frame:.4f}s per trajectory frame "
        f"({'kinematic' if cfg.kinematic else 'PD position control'})"
    )
    print(f"output -> {out_dir}")
    print("=" * 90, flush=True)

    robot_usd, object_usds = resolve_assets(spec, args.usd_dir)

    sim = SimulationContext(make_simulation_cfg(cfg, spec))
    scene = build_scene(spec, cfg, robot_usd, object_usds)

    sim.reset()

    checks = check_scene(scene, spec)

    if args.gui:
        # look at the table from the operator's side
        sim.set_camera_view(eye=(1.2, 0.9, 1.9), target=(-0.4, -0.4, 0.85))
        print(
            "[replay] GUI window open. "
            f"{'real-time' if not args.no_realtime else 'as fast as possible'}; "
            "orbit with the left mouse button, pan with the middle, zoom with the wheel.",
            flush=True,
        )

    result = run_replay(spec, cfg, scene, sim)
    records = result["records"]

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "replay_data.npz",
        joint_names=np.array(result["joint_names"]),
        traj_joint_names=np.array(result["traj_joint_names"]),
        tracked_links=np.array(result["tracked_links"]),
        object_keys=np.array(result["object_keys"]),
        object_names=np.array(result["object_names"]),
        fingertip_bodies=np.array(result["fingertip_bodies"]),
        contact_object_keys=np.array(result["contact_object_keys"]),
        handcam_poses=result["handcam_poses"],
        handcam_intrinsics=(
            result["handcam_intrinsics"] if result["handcam_intrinsics"] is not None else np.empty(0)
        ),
        **{k: v for k, v in records.items()},
    )

    if cfg.video and result["frames_rgb"]:
        if write_video(result["frames_rgb"], out_dir / "replay.mp4", cfg.video_fps):
            print(f"[replay] video -> {out_dir / 'replay.mp4'}")
    if cfg.video and result["frames_handcam"]:
        if result["handcam_intrinsics"] is not None and records["contact_points_w"].size:
            result["frames_handcam"] = analysis.overlay_contact_points(
                result["frames_handcam"],
                result["handcam_poses"],
                result["handcam_intrinsics"],
                records["contact_points_w"],
                records["contact_points_force_N"],
                result["fingertip_bodies"],
            )
        if write_video(result["frames_handcam"], out_dir / "handcam.mp4", cfg.video_fps):
            print(f"[replay] hand-cam video -> {out_dir / 'handcam.mp4'}")

    key_frames = spec.trajectory.key_frames
    joint_pos, joint_target = records["joint_pos"], records["joint_target"]
    analysis.plot_joint_tracking(
        joint_pos, joint_target, result["joint_names"], out_dir / "joint_tracking.png", key_frames
    )
    manipulated_idx = (
        result["object_keys"].index(result["manipulated_key"]) if result["manipulated_key"] else None
    )
    analysis.plot_object_tracking(
        records["object_pos"], result["object_names"], out_dir / "object_tracking.png", key_frames, manipulated_idx
    )

    # fingertip force reading: names of the objects the sensors were filtered against, in the
    # force-matrix column order, and the manipulated object's column among them
    contact_object_names = [
        result["object_names"][result["object_keys"].index(key)] for key in result["contact_object_keys"]
    ]
    contact_manip_idx = (
        result["contact_object_keys"].index(result["manipulated_key"])
        if result["manipulated_key"] in result["contact_object_keys"]
        else None
    )
    if records["contact_force_steps"].size:
        analysis.plot_contact_forces(
            records["contact_force_steps"],
            result["fingertip_bodies"],
            out_dir / "contact_forces.png",
            key_frames=key_frames,
            object_force_steps=records["contact_object_force_steps"],
            object_names=contact_object_names,
            manipulated_idx=contact_manip_idx,
        )

        # animated force plot + combined annotated video, paced like replay.mp4
        if cfg.video and result["frames_rgb"] and not args.no_force_videos:
            plot_frames = analysis.render_contact_force_video(
                records["contact_force_steps"],
                result["fingertip_bodies"],
                out_dir / "contact_forces.mp4",
                fps=cfg.video_fps,
                key_frames=key_frames,
                object_force_steps=records["contact_object_force_steps"],
                object_names=contact_object_names,
                manipulated_idx=contact_manip_idx,
                sim_time_per_frame=cfg.physics_dt * cfg.steps_per_frame,
                return_frames=True,
            )
            if plot_frames:
                print(f"[replay] force-plot video -> {out_dir / 'contact_forces.mp4'}")
                if analysis.render_composite_video(
                    result["frames_rgb"],
                    plot_frames,
                    out_dir / "replay_with_forces.mp4",
                    fps=cfg.video_fps,
                    run_name=spec.traj_run,
                    fingertip_bodies=result["fingertip_bodies"],
                    contact_force=records["contact_force"],
                    key_frames=key_frames,
                    sim_time_per_frame=cfg.physics_dt * cfg.steps_per_frame,
                    force_vis_scale=cfg.force_vis_scale if cfg.force_vis else None,
                    manipulated_name=(
                        contact_object_names[contact_manip_idx] if contact_manip_idx is not None else None
                    ),
                    inset_frames=result["frames_handcam"] or None,
                ):
                    print(f"[replay] combined video   -> {out_dir / 'replay_with_forces.mp4'}")

    hand_traj = analysis.hand_pose_trajectory(
        records["body_pos"],
        records["body_quat"],
        result["tracked_links"],
        records["object_pos"],
        records["object_quat"],
        result["object_keys"],
        result["manipulated_key"],
        contact_step=analysis.read_contact_step(run_dir),
    )
    if hand_traj is not None:
        with open(out_dir / "hand_pose_traj.json", "w") as f:
            json.dump(hand_traj, f, indent=2)

    demo_flow = run_dir / "flow_data" / "3d_flow_point.pkl"
    flow_ok = analysis.plot_flow_comparison(
        records["flow_points_3d"], demo_flow, out_dir / "flow_comparison.png", key_frames
    )

    summary = {
        "run": spec.traj_run,
        "scene_run": spec.scene_run,
        "timestamp": stamp,
        "num_frames": result["num_frames"],
        "wall_time_s": round(result["wall_time_s"], 2),
        "config": {
            "device": cfg.device,
            "physics_dt": cfg.physics_dt,
            "steps_per_frame": cfg.steps_per_frame,
            "sim_time_per_frame_s": cfg.physics_dt * cfg.steps_per_frame,
            "kinematic": cfg.kinematic,
            "object_mesh_scale": spec.object_mesh_scale,
            "render": cfg.render,
            "contact_sensors": cfg.contact_sensors,
            "force_vis": cfg.force_vis,
            "force_vis_scale": cfg.force_vis_scale if cfg.force_vis else None,
            "video_fps": cfg.video_fps,
            "flow_points": cfg.flow_points,
            "stale_target_steps": cfg.stale_target_steps,
            "joint_armature": cfg.joint_armature,
            "object_mesh_scale_source": spec.object_mesh_scale_source,
            "object_decomposition_error_pct": cfg.object_decomposition_error,
            "cam_far": cfg.cam_far,
            "cam_use_real_intrinsics": cfg.cam_use_real_intrinsics,
            "compat_rendering": bool(render_enabled and not args.no_compat_rendering and not args.gui),
        },
        "checks": checks,
        "joint_map": result["joint_map"],
        "static_objects": result["static_keys"],
        "assets": {"robot_usd": str(robot_usd), "object_usds": {k: str(v) for k, v in object_usds.items()}},
        "key_frames": key_frames,
        "joint_names": result["joint_names"],
        "tracking": analysis.tracking_error(joint_pos, joint_target),
        "objects": {
            name: analysis.object_motion(records["object_pos"], i)
            for i, name in enumerate(result["object_names"])
        },
        "manipulated_object": result["manipulated_key"],
        "flow": analysis.flow_metrics(records["flow_points_3d"], demo_flow if flow_ok else None),
        "robot_base_pose": {
            "position": spec.robot_position.tolist(),
            "quat_xyzw": spec.robot_quat_xyzw.tolist(),
        },
        "camera_pose": {
            "position": spec.camera.position.tolist(),
            "quat_xyzw": spec.camera.quat_xyzw.tolist(),
            "intrinsics_fx_fy_cx_cy": [spec.camera.fx, spec.camera.fy, spec.camera.cx, spec.camera.cy],
        },
    }
    contact_summary = analysis.contact_metrics(
        records["contact_force"],
        records["contact_force_steps"],
        result["fingertip_bodies"],
        object_force_steps=records["contact_object_force_steps"],
        object_names=contact_object_names,
    )
    if contact_summary is not None:
        summary["contact"] = contact_summary

    analysis.write_summary(summary, out_dir / "summary.json")

    if args.gui and args.keep_open:
        print("[replay] replay finished - close the window to exit.", flush=True)
        while simulation_app.is_running():
            sim.render()

    print("\n" + "=" * 90)
    print(f"[replay] frames            : {summary['num_frames']}")
    print(f"[replay] wall time         : {summary['wall_time_s']}s")
    print(
        f"[replay] joint tracking    : mean |err| {summary['tracking']['mean_abs_rad']:.4f} rad, "
        f"max {summary['tracking']['max_abs_rad']:.4f} rad"
    )
    for name, stats in summary["objects"].items():
        print(
            f"[replay] object {name:<10s} : moved {stats['total_displacement_m']:.4f} m, "
            f"max lift {stats['max_lift_m']:.4f} m"
        )
    if contact_summary is not None:
        manip_name = contact_object_names[contact_manip_idx] if contact_manip_idx is not None else None
        for i, body in enumerate(contact_summary["bodies"]):
            peak = contact_summary.get("peak_force_N", contact_summary["max_force_N"])[i]
            line = f"[replay] fingertip {body:<16s}: peak |F| {peak:7.2f} N"
            grasp = contact_summary.get("per_object", {}).get(body, {}).get(manip_name) if manip_name else None
            if grasp is not None:
                line += (
                    f"  (on {manip_name}: {grasp['peak_force_N']:.2f} N, "
                    f"first contact frame {grasp['first_contact_frame']})"
                )
            print(line)
    if summary["flow"]:
        flow = summary["flow"]
        if "centroid_delta_error_mean_m" in flow:
            print(
                f"[replay] flow vs demo      : mean Δ-error {flow['centroid_delta_error_mean_m']:.4f} m, "
                f"final {flow['centroid_delta_error_final_m']:.4f} m "
                f"(abs mean {flow['centroid_abs_error_mean_m']:.4f} m)"
            )
    print(f"[replay] outputs           : {out_dir}")
    print("=" * 90, flush=True)
    return 0


if __name__ == "__main__":
    status = 0
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        from v2s2r_isaaclab.runtime import hard_exit

        hard_exit(simulation_app, status)
