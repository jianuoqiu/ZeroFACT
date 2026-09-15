#!/usr/bin/env python3
"""Run the hybrid force controller closed-loop in Isaac Lab.

The recorded episode plays the BC policy's part (chunked predictions of robot states + fingertip
forces); the middle layer turns them plus the live force readings into PD joint targets; the
validated replay scene executes them. Outputs land in ``outputs/force_controller/<run>/<stamp>/``.

Examples::

    conda activate env_isaaclab

    # framework regression: reproduce the replay bit-exactly through the whole pipeline
    python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
        --exact-replay --no-render

    # baseline: predicted states, no force feedback (expect under-squeezing during the grasp)
    python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
        --force-law null --no-render

    # the actual controller: predicted states + force tracking, with videos (the default)
    python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22

    # same, physics only - faster and lighter, for gain sweeps
    python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 --no-render
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from v2s2r_isaaclab.runtime import check_memory, prepare_display  # noqa: E402  (before AppLauncher)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episode", required=True,
                    help="recorded episode: outputs/<run> (latest stamp) or outputs/<run>/<stamp>")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
parser.add_argument("--out-dir", type=Path, default=None)
# ---- controller ----
parser.add_argument("--chunk", type=int, default=8, help="frames per policy query (C)")
parser.add_argument("--horizon", type=int, default=16, help="predicted frames per chunk (H)")
parser.add_argument("--state-source", choices=["joint_pos", "joint_target"], default="joint_pos")
parser.add_argument("--interp", choices=["linear", "hold"], default="linear")
parser.add_argument("--latency-steps", type=int, default=0)
parser.add_argument("--force-law", choices=["null", "task_space"], default="task_space")
# NOTE: no --force-source. The loop always reads the NET fingertip force, the only thing a tactile
# pad can measure; the per-object breakdown is recorded for analysis but never fed back.
# task_space force channel (direction-free: feedforward on the predicted vector + PID on the error)
parser.add_argument("--kff", type=float, default=None, help="task_space: feedforward gain on f_ref")
parser.add_argument("--task-kp", type=float, default=None, help="task_space: P gain on the force error")
parser.add_argument("--task-ki", type=float, default=None, help="task_space: I rate (1/s)")
parser.add_argument("--task-kd", type=float, default=None, help="task_space: D gain (s)")
parser.add_argument("--no-adapt-kff", action="store_true",
                    help="task_space: disable the adaptive feedforward gain (fixed kff)")
parser.add_argument("--ff-gain-max", type=float, default=None,
                    help="task_space: clamp on the adaptive feedforward gain (default 8)")
parser.add_argument("--adapt-tau", type=float, default=None,
                    help="task_space: EMA time constant of the achieved/commanded estimate (s)")
parser.add_argument("--cone-deg", type=float, default=None,
                    help="task_space: cap on the angle between f_cmd and the predicted force "
                    "direction (deg; <=0 disables). Default 20 - see config.py")
parser.add_argument("--allow-arm-offset", action="store_true",
                    help="task_space: let the offset recruit arm joints too (per-tip "
                    "superposition ignores their coupling, which drifts the EE in multi-finger "
                    "grasps - see config.py; default is hand-only)")
# task_space contact-point channel (tangential P term through the contact-Jacobian pseudo-inverse)
parser.add_argument("--point-kp", type=float, default=None,
                    help="task_space: P gain on the tangential contact-point error (0 disables "
                    "the contact-point channel)")
parser.add_argument("--point-clip-m", type=float, default=None,
                    help="task_space: clamp on |p_ref - p_meas| fed to the law (m)")
parser.add_argument("--no-predict-point", action="store_true",
                    help="task_space: do not anchor the statics map at the predicted contact "
                    "point before the pad touches (fall straight back to the fingertip origin)")
parser.add_argument("--exact-replay", action="store_true",
                    help="preset: joint_target + hold + latency 2 + null law; must bit-reproduce the replay")
parser.add_argument("--diff-against", type=Path, default=None,
                    help="replay_data.npz to diff the rollout against (default with --exact-replay: "
                    "the episode's own)")
# ---- simulation ----
parser.add_argument("--render", action="store_true",
                    help="(default) demo camera + force arrows + rollout videos")
parser.add_argument("--no-render", action="store_true",
                    help="physics only: no camera, no videos. ~3.3 GB instead of ~6.6 GB and a "
                    "few seconds instead of a few minutes - use it for regression runs and gain "
                    "sweeps, where the videos are not worth the wall time")
parser.add_argument("--gui", action="store_true", help="live Isaac Sim window")
parser.add_argument("--no-realtime", action="store_true", help="with --gui, run as fast as possible")
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--settle-steps", type=int, default=0)
parser.add_argument("--object-mesh-scale", type=float, default=None)
parser.add_argument("--object-decomposition-error", type=float, default=None)
parser.add_argument("--video-fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-compat-rendering", action="store_true")

prepare_display(require_window="--gui" in sys.argv)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# resolve the episode before Kit starts (fail fast; Kit takes ~30 s)
from force_controller.episode import ReplayEpisode  # noqa: E402

episode_dir = ReplayEpisode.resolve_dir(args.episode, PROJECT_ROOT / "outputs")
episode = ReplayEpisode.load(episode_dir)
run_dir = args.data_dir / "runs" / episode.run_name
if not (run_dir / "run_meta.json").is_file():
    raise SystemExit(f"episode {episode_dir} belongs to run {episode.run_name!r}, "
                     f"but {run_dir} is not an ingested run folder")

# rendering is the default: a rollout is easier to judge with the video than without.
# --no-render opts out for regression runs and sweeps (args.render is then redundant
# but harmless, so existing command lines keep working).
render_enabled = not args.no_render
args.headless = not args.gui
if render_enabled or args.gui:
    args.enable_cameras = True
    if not args.no_compat_rendering:
        # same broken-install workaround as scripts/replay_trajectory.py (see README)
        args.experience = "isaaclab.python.headless.kit"
        compat_exts = ["omni.replicator.core", "omni.kit.viewport.rtx", "omni.kit.material.library"]
        if args.gui:
            compat_exts += [
                "omni.kit.mainwindow", "omni.kit.viewport.window", "omni.kit.manipulator.camera",
                "omni.kit.window.toolbar", "omni.kit.window.status_bar",
            ]
        compat_kit_args = " ".join(f"--enable {ext}" for ext in compat_exts)
        compat_kit_args += " --/isaaclab/cameras_enabled=true"
        args.kit_args = (
            f"{args.kit_args} {compat_kit_args}".strip() if getattr(args, "kit_args", "") else compat_kit_args
        )

check_memory(required_gb=8.0 if not (render_enabled or args.gui) else (14.0 if args.gui else 12.0))

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
    write_video,
)
from v2s2r_isaaclab.scene_spec import describe, load_run_spec  # noqa: E402

from force_controller.config import (  # noqa: E402
    ControllerConfig,
    ForceLawConfig,
    PolicyConfig,
    ReferenceConfig,
)
from force_controller.metrics import (  # noqa: E402
    command_fidelity_metrics,
    contact_point_metrics,
    format_command_fidelity,
    format_contact_point_metrics,
    format_grasp_slip,
    format_vector_metrics,
    grasp_slip_metrics,
    vector_force_metrics,
)
from force_controller.plots import (  # noqa: E402
    force_tracking_metrics,
    plot_action_comparison,
    plot_contact_point_error,
    plot_force_components,
    plot_force_error_components,
    plot_force_tracking,
    plot_vector_error,
)
from force_controller.sim_runner import diff_against_replay, run_force_tracking  # noqa: E402


def build_controller_config() -> ControllerConfig:
    law_cfg = ForceLawConfig(law=args.force_law)
    if args.kff is not None:
        law_cfg.kff = args.kff
    if args.task_kp is not None:
        law_cfg.task_kp = args.task_kp
    if args.task_ki is not None:
        law_cfg.task_ki = args.task_ki
    if args.task_kd is not None:
        law_cfg.task_kd = args.task_kd
    if args.no_adapt_kff:
        law_cfg.adapt_kff = False
    if args.ff_gain_max is not None:
        law_cfg.ff_gain_max = args.ff_gain_max
    if args.adapt_tau is not None:
        law_cfg.adapt_tau_s = args.adapt_tau
    if args.cone_deg is not None:
        law_cfg.cone_half_angle_deg = args.cone_deg
    if args.allow_arm_offset:
        law_cfg.allow_arm_offset = True
    if args.point_kp is not None:
        law_cfg.point_kp = args.point_kp
    if args.point_clip_m is not None:
        law_cfg.point_clip_m = args.point_clip_m
    if args.no_predict_point:
        law_cfg.predict_point_before_contact = False
    cfg = ControllerConfig(
        policy=PolicyConfig(horizon=args.horizon, chunk=args.chunk,
                            state_source=args.state_source),
        reference=ReferenceConfig(interp=args.interp, latency_steps=args.latency_steps),
        force_law=law_cfg,
    )
    if args.exact_replay:
        cfg = cfg.exact_replay()
        print("[force] --exact-replay: joint_target + hold + latency 2 + null law (regression mode)")
    return cfg


def main() -> int:
    spec = load_run_spec(run_dir, object_mesh_scale=args.object_mesh_scale)
    ctrl_cfg = build_controller_config()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "force_controller" / episode.run_name / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReplayConfig(
        device=args.device,
        render=render_enabled,
        save_images=False,
        video=render_enabled,
        video_fps=args.video_fps,
        hand_cam=False,
        contact_sensors=True,                      # the whole point: live force readings
        force_vis=render_enabled or args.gui,
        flow_points=0,
        max_frames=args.max_frames,
        settle_steps=args.settle_steps,
        gui=args.gui,
        gui_render_interval=2,
        realtime=args.gui and not args.no_realtime,
        output_dir=out_dir,
        seed=args.seed,
    )
    if args.object_decomposition_error is not None:
        cfg.object_decomposition_error = args.object_decomposition_error

    print("=" * 90)
    print(describe(spec))
    print(f"episode      : {episode_dir}  ({episode.num_frames} frames)")
    print(f"controller   : chunk {ctrl_cfg.policy.chunk} / horizon {ctrl_cfg.policy.horizon}, "
          f"states from {ctrl_cfg.policy.state_source!r}, force = NET fingertip pad reading, "
          f"interp {ctrl_cfg.reference.interp} (latency {ctrl_cfg.reference.latency_steps} steps), "
          f"law {ctrl_cfg.force_law.law!r} (point_kp {ctrl_cfg.force_law.point_kp})")
    print(f"output -> {out_dir}")
    print("=" * 90, flush=True)

    robot_usd = args.usd_dir / "robot" / "kinova_leap.usd"
    object_usds = {}
    for obj in spec.objects:
        scene_dir = args.usd_dir / "scenes" / spec.scene_run
        candidates = [scene_dir / obj.name / f"{obj.name}.usd", scene_dir / f"{obj.name}.usd"]
        usd_path = next((c for c in candidates if c.is_file()), None)
        if usd_path is None:
            raise SystemExit(f"Object USD missing: {candidates[0]} (run scripts/convert_assets.py)")
        object_usds[obj.key] = usd_path

    sim = SimulationContext(make_simulation_cfg(cfg, spec))
    scene = build_scene(spec, cfg, robot_usd, object_usds)
    sim.reset()
    check_scene(scene, spec)

    if args.gui:
        sim.set_camera_view(eye=(1.2, 0.9, 1.9), target=(-0.4, -0.4, 0.85))

    aligned = episode.reordered(list(scene.robot.joint_names))
    result = run_force_tracking(spec, cfg, ctrl_cfg, aligned, scene, sim)
    records = result["records"]
    T = result["num_frames"]

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "controller_data.npz",
        joint_names=np.array(result["joint_names"]),
        fingertip_bodies=np.array(result["fingertip_bodies"]),
        tracked_links=np.array(result["tracked_links"]),
        object_keys=np.array(result["object_keys"]),
        object_names=np.array(result["object_names"]),
        contact_object_keys=np.array(result["contact_object_keys"]),
        episode_path=np.array(str(episode_dir)),
        **records,
    )

    contact_object_names = [
        result["object_names"][result["object_keys"].index(key)] for key in result["contact_object_keys"]
    ]
    contact_manip_idx = (
        result["contact_object_keys"].index(result["manipulated_key"])
        if result["manipulated_key"] in result["contact_object_keys"] else None
    )

    if cfg.video and result["frames_rgb"]:
        if write_video(result["frames_rgb"], out_dir / "rollout.mp4", cfg.video_fps):
            print(f"[force] video -> {out_dir / 'rollout.mp4'}")
        # annotated side-by-side: camera view (with force arrows) + sweeping measured-force plot,
        # same tooling as the replay's replay_with_forces.mp4
        if records["contact_force_steps"].size:
            plot_frames = analysis.render_contact_force_video(
                records["contact_force_steps"], result["fingertip_bodies"],
                out_dir / "contact_forces.mp4", fps=cfg.video_fps, key_frames=episode.key_frames,
                object_force_steps=records["contact_object_force_steps"],
                object_names=contact_object_names, manipulated_idx=contact_manip_idx,
                sim_time_per_frame=cfg.physics_dt * cfg.steps_per_frame, return_frames=True,
            )
            if plot_frames:
                print(f"[force] force-plot video -> {out_dir / 'contact_forces.mp4'}")
                if analysis.render_composite_video(
                    result["frames_rgb"], plot_frames, out_dir / "rollout_with_forces.mp4",
                    fps=cfg.video_fps, run_name=f"{episode.run_name} [{ctrl_cfg.force_law.law}]",
                    fingertip_bodies=result["fingertip_bodies"],
                    contact_force=records["contact_force"], key_frames=episode.key_frames,
                    sim_time_per_frame=cfg.physics_dt * cfg.steps_per_frame,
                    force_vis_scale=cfg.force_vis_scale if cfg.force_vis else None,
                    manipulated_name=(
                        contact_object_names[contact_manip_idx] if contact_manip_idx is not None else None
                    ),
                ):
                    print(f"[force] combined video   -> {out_dir / 'rollout_with_forces.mp4'}")

    analysis.plot_joint_tracking(
        records["joint_pos"], records["joint_target"], result["joint_names"],
        out_dir / "joint_tracking.png", episode.key_frames,
    )
    manipulated_idx = (
        result["object_keys"].index(result["manipulated_key"]) if result["manipulated_key"] else None
    )
    analysis.plot_object_tracking(
        records["object_pos"], result["object_names"], out_dir / "object_tracking.png",
        episode.key_frames, manipulated_idx,
    )
    if records["contact_force_steps"].size:
        analysis.plot_contact_forces(
            records["contact_force_steps"], result["fingertip_bodies"],
            out_dir / "contact_forces.png", key_frames=episode.key_frames,
            object_force_steps=records["contact_object_force_steps"],
            object_names=contact_object_names, manipulated_idx=contact_manip_idx,
        )
    plot_force_tracking(
        records["f_ref_steps"], records["f_meas_steps"], result["fingertip_bodies"],
        out_dir / "force_tracking.png", cfg.steps_per_frame, episode.key_frames,
        u=records["u_steps"] if ctrl_cfg.force_law.law != "null" else None,
        title=f"{episode.run_name}: closed-loop force tracking (law={ctrl_cfg.force_law.law})",
        u_label="|integral| [N]" if ctrl_cfg.force_law.law == "task_space" else "u [rad]",
        engage_threshold_n=ctrl_cfg.force_law.engage_threshold_n,
    )
    plot_action_comparison(
        records["joint_target"], aligned.joint_target[:T], result["joint_names"],
        out_dir / "commands_vs_predicted.png", episode.key_frames,
        title=f"{episode.run_name}: recorded vs rollout commands vs predicted states "
              f"(law={ctrl_cfg.force_law.law})",
        predicted=records["q_ref_steps"][:, -1, :],
    )
    plot_vector_error(
        records["f_ref_vec_steps"], records["f_meas_vec_steps"], result["fingertip_bodies"],
        out_dir / "force_vector_error.png", cfg.steps_per_frame, episode.key_frames,
        ctrl_cfg.force_law.engage_threshold_n,
        title=f"{episode.run_name}: vector force-tracking error |f_ref − f_meas| "
              f"(law={ctrl_cfg.force_law.law})",
    )
    plot_force_components(
        records["f_ref_vec_steps"], records["f_meas_vec_steps"], result["fingertip_bodies"],
        out_dir / "force_components.png", cfg.steps_per_frame, episode.key_frames,
        ctrl_cfg.force_law.engage_threshold_n,
        title=f"{episode.run_name}: force vector components, world frame "
              f"(law={ctrl_cfg.force_law.law})",
    )

    plot_force_error_components(
        records["f_ref_vec_steps"], records["f_meas_vec_steps"], result["fingertip_bodies"],
        out_dir / "force_error_components.png", cfg.steps_per_frame, episode.key_frames,
        ctrl_cfg.force_law.engage_threshold_n,
        title=f"{episode.run_name}: force-tracking error per world axis "
              f"(law={ctrl_cfg.force_law.law})",
    )
    plot_contact_point_error(
        records["p_ref_steps"], records["p_meas_steps"], records["f_ref_vec_steps"],
        result["fingertip_bodies"], out_dir / "contact_point_error.png",
        cfg.steps_per_frame, episode.key_frames, ctrl_cfg.force_law.engage_threshold_n,
        title=f"{episode.run_name}: contact-point tracking error |p_ref − p_meas| "
              f"(law={ctrl_cfg.force_law.law})",
    )

    metrics = force_tracking_metrics(
        records["f_ref_steps"], records["f_meas_steps"], result["fingertip_bodies"],
        ctrl_cfg.force_law.engage_threshold_n,
    )
    vec_metrics = vector_force_metrics(
        records["f_ref_vec_steps"], records["f_meas_vec_steps"], result["fingertip_bodies"],
        ctrl_cfg.force_law.engage_threshold_n,
    )
    point_metrics = contact_point_metrics(
        records["p_ref_steps"], records["p_meas_steps"], records["f_ref_vec_steps"],
        result["fingertip_bodies"], ctrl_cfg.force_law.engage_threshold_n,
    )
    # the episode stands in for an ideal policy, so its recorded joint_target is a control input
    # KNOWN to achieve the reference exactly -> |action - ideal| is a controller score with a
    # true zero, and its per-joint split shows whether the right force came from the right joints
    contact_frames = records["f_ref_steps"][:, -1].max(axis=1) >= ctrl_cfg.force_law.engage_threshold_n
    cmd_metrics = command_fidelity_metrics(
        records["joint_target"], aligned.joint_target[:T], records["q_ref_steps"][:, -1],
        result["joint_names"], contact_frames,
    )

    # in-hand slip of the manipulated object relative to the palm (the drop detector)
    slip_metrics = None
    ep_manip = (aligned.object_keys.index(aligned.manipulated_key)
                if aligned.manipulated_key in aligned.object_keys else None)
    if (manipulated_idx is not None and ep_manip is not None
            and "palm_lower" in result["tracked_links"] and "palm_lower" in aligned.tracked_links):
        palm_i = result["tracked_links"].index("palm_lower")
        ep_palm = aligned.tracked_links.index("palm_lower")
        slip_metrics = grasp_slip_metrics(
            records["object_pos"][:, manipulated_idx],
            records["body_pos"][:, palm_i], records["body_quat"][:, palm_i],
            aligned.object_pos[:T, ep_manip],
            aligned.body_pos[:T, ep_palm], aligned.body_quat[:T, ep_palm],
            records["f_ref_steps"][:, -1], records["f_meas_steps"][:, -1],
        )

    # how the rollout's outcome compares with the recorded episode
    def max_lift(pos: np.ndarray, idx: int | None) -> float | None:
        if idx is None:
            return None
        return float((pos[:, idx, 2] - pos[0, idx, 2]).max())

    ep_manip_idx = (
        episode.object_keys.index(episode.manipulated_key)
        if episode.manipulated_key in episode.object_keys else None
    )
    outcome = {
        "manipulated_max_lift_m": max_lift(records["object_pos"], manipulated_idx),
        "episode_manipulated_max_lift_m": max_lift(episode.object_pos[:T], ep_manip_idx),
        "objects": {
            name: analysis.object_motion(records["object_pos"], i)
            for i, name in enumerate(result["object_names"])
        },
    }

    summary = {
        "episode": str(episode_dir),
        "run": episode.run_name,
        "timestamp": stamp,
        "num_frames": T,
        "wall_time_s": round(result["wall_time_s"], 2),
        "controller": asdict(ctrl_cfg),
        "sim_config": {
            "device": cfg.device, "physics_dt": cfg.physics_dt,
            "steps_per_frame": cfg.steps_per_frame, "render": cfg.render,
            "settle_steps": cfg.settle_steps,
            "object_mesh_scale": spec.object_mesh_scale,
            "object_decomposition_error_pct": cfg.object_decomposition_error,
        },
        "tracking": analysis.tracking_error(records["joint_pos"], records["joint_target"]),
        "force_tracking": metrics,
        "force_tracking_vector": vec_metrics,
        "contact_point_tracking": point_metrics,
        "command_fidelity": cmd_metrics,
        "grasp_slip": slip_metrics,
        "outcome": outcome,
    }

    diff_path = args.diff_against
    if diff_path is None and args.exact_replay:
        diff_path = episode_dir / "replay_data.npz"
    if diff_path is not None:
        diff = diff_against_replay(records, diff_path, T)
        summary["diff_against"] = {"path": str(diff_path), "max_abs_diff": diff}

    analysis.write_summary(summary, out_dir / "summary.json")

    # ---------------- report ----------------
    print("\n" + "=" * 90)
    print(f"[force] frames    : {T}   wall time {summary['wall_time_s']}s")
    for i, body in enumerate(metrics["bodies"]):
        print(f"[force] {body:<16s} engaged {metrics['engaged_fraction'][i] * 100:5.1f}%  "
              f"force rmse {metrics['rmse_engaged_N'][i]:7.3f} N  "
              f"bias {metrics['bias_engaged_N'][i]:+7.3f} N")
    print("[force] vector error |f_ref_vec - f_meas_vec| (engaged steps):")
    for line in format_vector_metrics(vec_metrics, prefix="[force]   "):
        print(line)
    print("[force] contact-point error |p_ref - p_meas| (engaged + touching steps):")
    for line in format_contact_point_metrics(point_metrics, prefix="[force]   "):
        print(line)
    for line in format_command_fidelity(cmd_metrics, prefix="[force] "):
        print(line)
    if slip_metrics is not None:
        print(format_grasp_slip(slip_metrics, prefix="[force] "))
    if ctrl_cfg.force_law.law == "task_space" and records["ff_gain_steps"].size:
        eng = records["f_ref_steps"] >= ctrl_cfg.force_law.engage_threshold_n     # [T,S,4]
        g = records["ff_gain_steps"]
        stats = []
        for i, b in enumerate(result["fingertip_bodies"]):
            m = eng[:, :, i]
            if m.any():
                stats.append(f"{b[:5]}:{float(g[:, :, i][m].mean()):.2f}(end {float(g[-1, -1, i]):.2f})")
        u_sat = float((records["u_steps"] >= 0.99 * ctrl_cfg.force_law.task_i_max_n)[eng].mean()) if eng.any() else 0.0
        print(f"[force] adaptive ff gain (engaged mean): {'  '.join(stats)}   |   "
              f"integral at clamp: {100 * u_sat:.1f}% of engaged steps")
    lift, ep_lift = outcome["manipulated_max_lift_m"], outcome["episode_manipulated_max_lift_m"]
    if lift is not None:
        line = f"[force] manipulated object max lift: {lift * 1000:7.1f} mm"
        if ep_lift is not None:
            line += f"  (episode: {ep_lift * 1000:7.1f} mm)"
        print(line)
    if diff_path is not None:
        print(f"[force] diff vs {diff_path}:")
        for key, entry in summary["diff_against"]["max_abs_diff"].items():
            if "error" in entry:
                print(f"[force]   {key:<28s} {entry['error']}  <- MISMATCH")
                continue
            value, ulp = entry["max_abs_diff"], entry["max_ulp"]
            if value == 0.0:
                flag = "OK (bit-exact)"
            elif ulp <= 1.0:
                flag = f"OK ({value:.3g}, <= 1 float32 ulp: sensor readback jitter)"
            else:
                flag = f"{value:.6g} ({ulp:.1f} ulp)" + ("  <- NONZERO" if args.exact_replay else "")
            print(f"[force]   {key:<28s} {flag}")
    print(f"[force] outputs -> {out_dir}")
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
