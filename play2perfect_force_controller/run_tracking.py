#!/usr/bin/env python3
"""Run the hybrid force controller closed-loop in the play2perfect scene.

A recorded episode (``collect_episodes.py``) plays the BC policy's part - chunked predictions of
robot states + fingertip force targets; the middle layer turns them plus the live tactile readings
into PD joint targets; play2perfect's own scene executes them. Outputs land in
``outputs/play2perfect/force_controller/<episode run>/<stamp>/``.

Validation ladder (run in this order)::

    conda activate env_isaaclab
    EP=outputs/play2perfect/episodes/tight_insertion/<stamp>/ep_0000

    # 1. no sim: the chunking/interp plumbing must rebuild the recorded commands bit-exactly
    python play2perfect_force_controller/offline_check.py --episode $EP
    # 2. in sim: the whole runner must reproduce the recording (replays are deterministic here)
    python play2perfect_force_controller/run_tracking.py --episode $EP --exact-replay --no-render
    # 3. baseline: predicted states, no force feedback
    python play2perfect_force_controller/run_tracking.py --episode $EP --force-law null --no-render
    # 4. the controller: predicted states + task-space force tracking, with videos
    python play2perfect_force_controller/run_tracking.py --episode $EP
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

from play2perfect_force_controller.launch import (  # noqa: E402
    add_sim_args, finalize_launcher_args, hard_exit, prepare_display,
)
from play2perfect_force_controller.robot_spec import (  # noqa: E402
    FRAME_DT, PALM_LINK, register_with_zerofact_analysis,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episode", required=True, help="recorded episode folder (contains replay_data.npz)")
parser.add_argument("--out-dir", type=Path, default=None)
# ---- controller (identical flags to force_controller/run_tracking.py) ----
parser.add_argument("--chunk", type=int, default=None, help="frames per policy query (C); frames are 60 Hz")
parser.add_argument("--horizon", type=int, default=None, help="predicted frames per chunk (H)")
parser.add_argument("--state-source", choices=["joint_pos", "joint_target"], default="joint_pos")
parser.add_argument("--interp", choices=["linear", "hold"], default="linear")
parser.add_argument("--latency-steps", type=int, default=0)
parser.add_argument("--force-law", choices=["null", "task_space"], default="task_space")
parser.add_argument("--kff", type=float, default=None, help="task_space: feedforward gain on f_ref")
parser.add_argument("--task-kp", type=float, default=None, help="task_space: P gain on the force error")
parser.add_argument("--task-ki", type=float, default=None, help="task_space: I rate (1/s)")
parser.add_argument("--task-kd", type=float, default=None, help="task_space: D gain (s)")
parser.add_argument("--no-adapt-kff", action="store_true", help="task_space: fixed feedforward gain")
parser.add_argument("--ff-gain-max", type=float, default=None, help="task_space: adaptive ff gain clamp")
parser.add_argument("--adapt-tau", type=float, default=None, help="task_space: EMA time constant (s)")
parser.add_argument("--cone-deg", type=float, default=None, help="task_space: friction-cone cap (deg; <=0 off)")
parser.add_argument("--allow-arm-offset", action="store_true", help="task_space: offset may use arm joints")
parser.add_argument("--point-kp", type=float, default=None, help="task_space: contact-point P gain (0 = off)")
parser.add_argument("--point-clip-m", type=float, default=None, help="task_space: contact-point error clamp (m)")
parser.add_argument("--offset-clip-rad", type=float, default=None, help="uniform per-joint |offset| clamp (rad)")
parser.add_argument("--effort-clip", action="store_true",
                    help="clip each joint at tau_max_j / K_j instead of the uniform clamp (evaluated worse)")
parser.add_argument("--limit-margin-rad", type=float, default=None,
                    help="'at the limit' band for the joint-limit masking (<=0 disables)")
parser.add_argument("--map", choices=["statics", "constrained"], default=None,
                    help="force->offset map: closed-form statics + per-joint clip (default) or the bounded "
                         "least squares that keeps the force direction under the clip/effort bounds")
parser.add_argument("--map-null-arm-m", type=float, default=None, help="constrained map: lever arm weighing null-space torque (m)")
parser.add_argument("--point-engage-n", type=float, default=None,
                    help="point channel acts only above this predicted |force| (0 = engage threshold)")
parser.add_argument("--point-steady-rate", type=float, default=None,
                    help="point channel acts only while the predicted |force| changes slower than this (1/s; 0 = off)")
parser.add_argument("--engage-threshold-n", type=float, default=None,
                    help="predicted |force| below which a fingertip is not meant to press (both channels off)")
parser.add_argument("--no-predict-point", action="store_true",
                    help="task_space: do not anchor the statics map at the predicted contact point")
parser.add_argument("--filter-signals", choices=["none", "traj", "force", "all"], default="none",
                    help="smooth the recorded episode before replaying it (signal_filtering.py): "
                         "traj = the joint trajectory a BC policy would emit, force = the contact "
                         "channels, all = both")
parser.add_argument("--filter-window", type=int, default=None, help="Savitzky-Golay window in frames")
parser.add_argument("--filter-order", type=int, default=None, help="Savitzky-Golay polynomial order")
parser.add_argument("--filter-causal", action="store_true",
                    help="one-sided filter instead of the centered default (see REPORT.md: causal "
                         "Savitzky-Golay amplifies this data's noise band)")
parser.add_argument("--exact-replay", action="store_true",
                    help="preset: joint_target + hold + latency 0 + null law; must reproduce the recording")
parser.add_argument("--policy", choices=["replay", "oracle"], default="replay",
                    help="who serves the chunks: slices of the recorded episode (replay, open loop) or the "
                         "RL policy rolled forward in the simulator from the current state every --chunk "
                         "frames (oracle: closed loop at chunk rate, see oracle_policy.py)")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="--policy oracle: model.pth (default: the problem's released checkpoint)")
parser.add_argument("--stochastic", action="store_true",
                    help="--policy oracle: sample the policy instead of taking its mean action")
parser.add_argument("--grace-frames", type=int, default=600,
                    help="--policy oracle: frames the rollout may run past the recording's length "
                         "(the env's own timeout still ends it)")
parser.add_argument("--diff-against", type=Path, default=None,
                    help="replay_data.npz to diff the rollout against (default with --exact-replay: the episode's)")
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--no-realtime", action="store_true", help="with --gui, run as fast as possible")
parser.add_argument("--goal-marker-opacity", type=float, default=0.18,
                    help="translucency of the target-pose marker in renders (1.0 = opaque as in play2perfect)")
add_sim_args(parser)

prepare_display(require_window="--gui" in sys.argv)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# resolve the episode before Kit starts (fail fast; Kit takes ~1 min)
from play2perfect_force_controller.episode import ReplayEpisode  # noqa: E402

episode_dir = ReplayEpisode.resolve_dir(args.episode, PROJECT_ROOT / "outputs")
episode = ReplayEpisode.load(episode_dir)
if args.filter_signals != "none":
    episode = episode.filtered(args.filter_signals, window=args.filter_window,
                               polyorder=args.filter_order,
                               causal=True if args.filter_causal else False)
    print(f"[force] episode filtered: signals={args.filter_signals} window={args.filter_window or 'spec'} "
          f"order={args.filter_order or 'spec'} {'causal' if args.filter_causal else 'centered'}", flush=True)
problem = episode.summary.get("problem")
if not problem or "init_state" not in episode.summary:
    raise SystemExit(f"{episode_dir} was not recorded by collect_episodes.py (no problem/init_state in summary)")
render_enabled = finalize_launcher_args(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------------------
# Isaac-dependent imports
# ---------------------------------------------------------------------------------------------
from zerofact import analysis  # noqa: E402
from zerofact.replay import write_video  # noqa: E402

from play2perfect_force_controller.config import (  # noqa: E402
    ControllerConfig, ForceLawConfig, PolicyConfig, ReferenceConfig,
)
from play2perfect_force_controller.metrics import (  # noqa: E402
    command_fidelity_metrics, contact_point_metrics, format_command_fidelity,
    format_contact_point_metrics, format_grasp_slip, format_task_success, format_vector_metrics,
    grasp_slip_metrics, task_success_metrics, vector_force_metrics,
)
from play2perfect_force_controller.oracle_policy import OracleChunkPolicy  # noqa: E402
from play2perfect_force_controller.p2p_env import (  # noqa: E402
    AssemblyBench, SceneOptions, load_policy, make_env_cfg,
)
from play2perfect_force_controller.p2p_paths import checkpoint_path  # noqa: E402
from play2perfect_force_controller.plots import (  # noqa: E402
    force_tracking_metrics, plot_action_comparison, plot_contact_point_error, plot_force_components,
    plot_force_error_components, plot_force_tracking, plot_vector_error,
)
from play2perfect_force_controller.sim_runner import (  # noqa: E402
    RolloutConfig, diff_against_replay, run_force_tracking,
)

register_with_zerofact_analysis()


def build_controller_config() -> ControllerConfig:
    law_cfg = ForceLawConfig(law=args.force_law)
    for name, value in [("kff", args.kff), ("task_kp", args.task_kp), ("task_ki", args.task_ki),
                        ("task_kd", args.task_kd), ("ff_gain_max", args.ff_gain_max),
                        ("adapt_tau_s", args.adapt_tau), ("cone_half_angle_deg", args.cone_deg),
                        ("point_kp", args.point_kp), ("point_clip_m", args.point_clip_m),
                        ("offset_clip_rad", args.offset_clip_rad),
                        ("engage_threshold_n", args.engage_threshold_n), ("map", args.map),
                        ("map_null_arm_m", args.map_null_arm_m), ("point_engage_n", args.point_engage_n),
                        ("point_steady_rate", args.point_steady_rate)]:
        if value is not None:
            setattr(law_cfg, name, value)
    if args.effort_clip:
        law_cfg.offset_clip_from_effort = True
    if args.limit_margin_rad is not None:
        law_cfg.limit_margin_rad = args.limit_margin_rad
    if args.no_adapt_kff:
        law_cfg.adapt_kff = False
    if args.allow_arm_offset:
        law_cfg.allow_arm_offset = True
    if args.no_predict_point:
        law_cfg.predict_point_before_contact = False
    policy_cfg = PolicyConfig(state_source=args.state_source, source=args.policy)
    if args.horizon is not None:
        policy_cfg.horizon = args.horizon
    if args.chunk is not None:
        policy_cfg.chunk = args.chunk
    cfg = ControllerConfig(
        policy=policy_cfg,
        reference=ReferenceConfig(interp=args.interp, latency_steps=args.latency_steps),
        force_law=law_cfg,
    )
    if args.exact_replay:
        cfg = cfg.exact_replay()
        cfg.policy.source = args.policy
        print("[force] --exact-replay: joint_target + hold + latency 0 + null law (regression mode)")
    return cfg


def main() -> int:
    ctrl_cfg = build_controller_config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "play2perfect" / "force_controller" / episode.run_name / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_every = max(1, round(1.0 / (FRAME_DT * args.video_fps)))
    fps = round(1.0 / (FRAME_DT * video_every))

    print("=" * 90)
    print(f"episode      : {episode_dir}  ({episode.num_frames} frames, problem {problem})")
    print(f"controller   : chunk {ctrl_cfg.policy.chunk} / horizon {ctrl_cfg.policy.horizon} frames "
          f"(60 Hz), states from {ctrl_cfg.policy.state_source!r}, force = NET fingertip pad reading, "
          f"interp {ctrl_cfg.reference.interp} (latency {ctrl_cfg.reference.latency_steps} steps), "
          f"law {ctrl_cfg.force_law.law!r} (point_kp {ctrl_cfg.force_law.point_kp})")
    print(f"policy       : {'oracle - the RL policy re-planned in the simulator at every chunk boundary' if args.policy == 'oracle' else 'replay - slices of the recorded episode (open loop)'}")
    print(f"output -> {out_dir}")
    print("=" * 90, flush=True)

    ep_seed = int(episode.summary.get("config", {}).get("seed", args.seed))
    env_seed = ep_seed if args.policy == "oracle" else args.seed
    cfg = make_env_cfg(problem, seed=env_seed, sim_device=args.device, render=render_enabled or args.gui)
    env = AssemblyBench(cfg, SceneOptions(render=render_enabled or args.gui, max_contact_points=args.max_contact_points,
                                    goal_marker_opacity=args.goal_marker_opacity))
    if args.gui:
        env.sim.set_camera_view(eye=(1.4, -1.0, 1.3), target=(0.0, 0.1, 0.6))

    roll = RolloutConfig(
        max_frames=args.max_frames, render=render_enabled, video_every=video_every,
        gui=args.gui, realtime=args.gui and not args.no_realtime, grace_frames=args.grace_frames,
    )
    policy = None
    if args.policy == "oracle":
        ckpt = args.checkpoint or checkpoint_path(problem)
        player, wrapped = load_policy(env, ckpt, deterministic=not args.stochastic, rl_device=args.device,
                                      seed=ep_seed)
        policy = OracleChunkPolicy(env, player, wrapped, ctrl_cfg.policy, deterministic=not args.stochastic)
        roll.reset_seed = ep_seed
        print(f"[force] oracle chunk policy: {ckpt}, re-planned every {ctrl_cfg.policy.chunk} frames "
              f"(horizon {ctrl_cfg.policy.horizon}), env reset with the recording's seed {ep_seed}", flush=True)
    aligned = episode.reordered(list(env.robot.joint_names))
    result = run_force_tracking(env, ctrl_cfg, aligned, roll, policy=policy)
    records = result["records"]
    T = result["num_frames"]
    aligned = aligned.padded(T)          # a closed-loop rollout may outlast the recording
    fingertips = result["fingertip_bodies"]

    # ---------------- save ----------------
    np.savez_compressed(
        out_dir / "controller_data.npz",
        joint_names=np.array(result["joint_names"]),
        fingertip_bodies=np.array(fingertips),
        tracked_links=np.array(result["tracked_links"]),
        object_keys=np.array(result["object_keys"]),
        object_names=np.array(result["object_names"]),
        contact_object_keys=np.array(result["contact_object_keys"]),
        episode_path=np.array(str(episode_dir)),
        joint_limits=np.asarray(result["joint_limits"], dtype=np.float32),
        **records,
    )
    contact_keys = result["contact_object_keys"]
    contact_manip_idx = contact_keys.index(result["manipulated_key"]) if result["manipulated_key"] in contact_keys else None
    manipulated_idx = result["object_keys"].index(result["manipulated_key"]) if result["manipulated_key"] else None
    key_frames = episode.key_frames
    law = ctrl_cfg.force_law.law

    if render_enabled and result["frames_rgb"]:
        every = result["video_every"]
        if write_video(result["frames_rgb"], out_dir / "rollout.mp4", fps):
            print(f"[force] video -> {out_dir / 'rollout.mp4'}")
            # camera + live force-tracking plot (target vs measured per fingertip); needs the
            # saved controller_data.npz, so it is rendered after the save below
        plot_frames = analysis.render_contact_force_video(
            records["contact_force_steps"], fingertips, out_dir / "contact_forces.mp4", fps=fps,
            key_frames=key_frames, object_force_steps=records["contact_object_force_steps"],
            object_names=contact_keys, manipulated_idx=contact_manip_idx,
            sim_time_per_frame=FRAME_DT, return_frames=True,
        )
        if plot_frames and analysis.render_composite_video(
            result["frames_rgb"], plot_frames[::every], out_dir / "rollout_with_forces.mp4", fps=fps,
            run_name=f"{episode.run_name} [{law}]", fingertip_bodies=fingertips,
            contact_force=records["contact_force"][::every], key_frames=key_frames,
            sim_time_per_frame=FRAME_DT * every, force_vis_scale=env.options.force_vis_scale,
            manipulated_name=result["manipulated_key"],
        ):
            print(f"[force] combined video   -> {out_dir / 'rollout_with_forces.mp4'}")

    analysis.plot_joint_tracking(records["joint_pos"], records["joint_target"], result["joint_names"],
                                 out_dir / "joint_tracking.png", key_frames)
    analysis.plot_object_tracking(records["object_pos"], result["object_names"], out_dir / "object_tracking.png",
                                  key_frames, manipulated_idx)
    analysis.plot_contact_forces(
        records["contact_force_steps"], fingertips, out_dir / "contact_forces.png", key_frames=key_frames,
        object_force_steps=records["contact_object_force_steps"], object_names=contact_keys,
        manipulated_idx=contact_manip_idx,
    )
    S = episode.steps_per_frame
    thr = ctrl_cfg.force_law.engage_threshold_n
    plot_force_tracking(
        records["f_ref_steps"], records["f_meas_steps"], fingertips, out_dir / "force_tracking.png", S, key_frames,
        u=records["u_steps"] if law != "null" else None,
        title=f"{episode.run_name}: closed-loop force tracking (law={law})",
        u_label="|integral| [N]" if law == "task_space" else "u [rad]", engage_threshold_n=thr,
    )
    plot_action_comparison(
        records["joint_target"], aligned.joint_target[:T], result["joint_names"],
        out_dir / "commands_vs_predicted.png", key_frames,
        title=f"{episode.run_name}: recorded vs rollout commands vs predicted states (law={law})",
        predicted=records["q_ref_steps"][:, -1, :],
    )
    plot_vector_error(records["f_ref_vec_steps"], records["f_meas_vec_steps"], fingertips,
                      out_dir / "force_vector_error.png", S, key_frames, thr,
                      title=f"{episode.run_name}: vector force-tracking error |f_ref − f_meas| (law={law})")
    plot_force_components(records["f_ref_vec_steps"], records["f_meas_vec_steps"], fingertips,
                          out_dir / "force_components.png", S, key_frames, thr,
                          title=f"{episode.run_name}: force vector components, world frame (law={law})")
    plot_force_error_components(records["f_ref_vec_steps"], records["f_meas_vec_steps"], fingertips,
                                out_dir / "force_error_components.png", S, key_frames, thr,
                                title=f"{episode.run_name}: force-tracking error per world axis (law={law})")
    plot_contact_point_error(records["p_ref_steps"], records["p_meas_steps"], records["f_ref_vec_steps"],
                             fingertips, out_dir / "contact_point_error.png", S, key_frames, thr,
                             title=f"{episode.run_name}: contact-point tracking error |p_ref − p_meas| (law={law})")

    metrics = force_tracking_metrics(records["f_ref_steps"], records["f_meas_steps"], fingertips, thr)
    vec_metrics = vector_force_metrics(records["f_ref_vec_steps"], records["f_meas_vec_steps"], fingertips, thr)
    point_metrics = contact_point_metrics(records["p_ref_steps"], records["p_meas_steps"],
                                          records["f_ref_vec_steps"], fingertips, thr)
    contact_frames = records["f_ref_steps"][:, -1].max(axis=1) >= thr
    cmd_metrics = command_fidelity_metrics(records["joint_target"], aligned.joint_target[:T],
                                           records["q_ref_steps"][:, -1], result["joint_names"], contact_frames)

    slip_metrics = None
    ep_manip = aligned.object_keys.index(aligned.manipulated_key) if aligned.manipulated_key in aligned.object_keys else None
    if manipulated_idx is not None and ep_manip is not None and PALM_LINK in result["tracked_links"] \
            and PALM_LINK in aligned.tracked_links:
        palm_i = result["tracked_links"].index(PALM_LINK)
        ep_palm = aligned.tracked_links.index(PALM_LINK)
        slip_metrics = grasp_slip_metrics(
            records["object_pos"][:, manipulated_idx], records["body_pos"][:, palm_i], records["body_quat"][:, palm_i],
            aligned.object_pos[:T, ep_manip], aligned.body_pos[:T, ep_palm], aligned.body_quat[:T, ep_palm],
            records["f_ref_steps"][:, -1], records["f_meas_steps"][:, -1],
        )

    # task outcome: where the part ended relative to where the recording left it
    final_pos_err = final_lift = ep_lift = None
    if manipulated_idx is not None and ep_manip is not None:
        final_pos_err = float(np.linalg.norm(records["object_pos"][-1, manipulated_idx] - aligned.object_pos[T - 1, ep_manip]))
        final_lift = float((records["object_pos"][:, manipulated_idx, 2] - records["object_pos"][0, manipulated_idx, 2]).max())
        ep_lift = float((aligned.object_pos[:T, ep_manip, 2] - aligned.object_pos[0, ep_manip, 2]).max())
    # the task's own verdict: keypoints of the part within the env's tolerance of the FINAL goal
    task = None
    goal_i = aligned.object_keys.index("goal_viz") if "goal_viz" in aligned.object_keys else None
    if manipulated_idx is not None and goal_i is not None:
        tip_ids = [result["tracked_links"].index(b) for b in fingertips if b in result["tracked_links"]]
        task = task_success_metrics(
            records["object_pos"][:, manipulated_idx], records["object_quat"][:, manipulated_idx],
            aligned.object_pos[-1, goal_i], aligned.object_quat[-1, goal_i],
            fingertip_pos=records["body_pos"][:, tip_ids] if tip_ids else None,
        )
    outcome = {
        "task": task,
        "manipulated_final_position_error_vs_episode_m": final_pos_err,
        "manipulated_max_lift_m": final_lift,
        "episode_manipulated_max_lift_m": ep_lift,
        "objects": {name: analysis.object_motion(records["object_pos"], i) for i, name in enumerate(result["object_names"])},
    }
    if result["closed_loop"]:
        # the env's own verdict (success counter, retract flag, termination) - the same test the
        # recordings were selected with
        outcome["env"] = {"end": result["env_end"], "last": result["env_last"]}

    summary = {
        "episode": str(episode_dir), "run": episode.run_name, "problem": problem, "timestamp": stamp,
        "num_frames": T, "wall_time_s": round(result["wall_time_s"], 2),
        "controller": asdict(ctrl_cfg),
        "sim_config": {"device": args.device, "physics_dt": env.physics_dt, "steps_per_frame": S,
                       "render": render_enabled},
        "episode_filter": {"signals": args.filter_signals, "window": args.filter_window,
                           "order": args.filter_order,
                           "kind": "causal" if args.filter_causal else "centered"},
        "tracking": analysis.tracking_error(records["joint_pos"], records["joint_target"]),
        "force_tracking": metrics, "force_tracking_vector": vec_metrics,
        "contact_point_tracking": point_metrics, "command_fidelity": cmd_metrics,
        "grasp_slip": slip_metrics, "outcome": outcome,
        "policy_stats": result["policy_stats"],
    }
    diff_path = args.diff_against
    if diff_path is None and args.exact_replay:
        diff_path = episode_dir / "replay_data.npz"
    if diff_path is not None:
        summary["diff_against"] = {"path": str(diff_path), "max_abs_diff": diff_against_replay(records, diff_path, T)}
    analysis.write_summary(summary, out_dir / "summary.json")

    # ---------------- report ----------------
    print("\n" + "=" * 90)
    print(f"[force] frames    : {T}   wall time {summary['wall_time_s']}s")
    for i, body in enumerate(metrics["bodies"]):
        print(f"[force] {body:<16s} engaged {metrics['engaged_fraction'][i] * 100:5.1f}%  "
              f"force rmse {metrics['rmse_engaged_N'][i]:7.3f} N  bias {metrics['bias_engaged_N'][i]:+7.3f} N")
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
    if law == "task_space" and records["ff_gain_steps"].size:
        eng = records["f_ref_steps"] >= thr
        g = records["ff_gain_steps"]
        stats = [f"{fingertips[i].replace('left_', '')[:6]}:{float(g[:, :, i][eng[:, :, i]].mean()):.2f}"
                 for i in range(len(fingertips)) if eng[:, :, i].any()]
        u_sat = float((records["u_steps"] >= 0.99 * ctrl_cfg.force_law.task_i_max_n)[eng].mean()) if eng.any() else 0.0
        print(f"[force] adaptive ff gain (engaged mean): {'  '.join(stats)}   |   "
              f"integral at clamp: {100 * u_sat:.1f}% of engaged steps")
    if law == "task_space" and records.get("limit_masked_steps") is not None and records["limit_masked_steps"].size:
        m = records["limit_masked_steps"].reshape(-1, records["limit_masked_steps"].shape[-1]).mean(axis=0)
        top = np.argsort(-m)[:4]
        print("[force] joint-limit masking (share of steps): " + ", ".join(
            f"{result['joint_names'][i].replace('left_', '')} {100 * m[i]:.0f}%" for i in top if m[i] > 0))
    if result["closed_loop"] and result["env_last"]:
        v, end = result["env_last"], result["env_end"]
        print(f"[force] env verdict: goals {v['successes']}/{v['max_goals']}, retract "
              f"{'ok' if v['retract_succeeded'] else 'no'}, "
              + (f"finished at frame {end['frame'] + 1} by {[k for k, f in end['termination'].items() if f]}"
                 if end else f"not finished within {T} frames"))
        st = result["policy_stats"] or {}
        if st:
            print(f"[force] oracle: {st['plans']} plans, {st['mean_plan_ms']} ms each, "
                  f"{len(st['planned_done_frames'])} plans saw the env finish")
    if task is not None:
        print(format_task_success(task, prefix="[force] "))
    if final_pos_err is not None:
        print(f"[force] manipulated object: final position {final_pos_err * 1000:6.1f} mm from the recording's, "
              f"max lift {final_lift * 1000:6.1f} mm (episode {ep_lift * 1000:6.1f} mm)")
    if diff_path is not None:
        print(f"[force] diff vs {diff_path}:")
        for key, entry in summary["diff_against"]["max_abs_diff"].items():
            if "error" in entry:
                print(f"[force]   {key:<28s} {entry['error']}  <- MISMATCH")
                continue
            value, ulp = entry["max_abs_diff"], entry["max_ulp"]
            # the goal marker is re-written every frame from the float32 recording, which can
            # leave ~1e-9 on a near-zero quaternion component (ulp counts explode there)
            flag = ("OK (bit-exact)" if value == 0.0 else
                    f"OK ({value:.3g}, <= 1 float32 ulp: sensor readback jitter)" if ulp <= 1.0 else
                    f"OK ({value:.3g}: float32 rounding of a kinematic pose write)" if value < 1e-7 else
                    f"{value:.6g} ({ulp:.1f} ulp)" + ("  <- NONZERO" if args.exact_replay else ""))
            print(f"[force]   {key:<28s} {flag}")
    if render_enabled and result["frames_rgb"] and (out_dir / "rollout.mp4").is_file():
        try:
            from play2perfect_force_controller.render_force_video import render as render_force_video
            print(f"[force] force-tracking video -> {render_force_video(out_dir, fps=fps)}")
        except Exception as exc:  # noqa: BLE001 - the video is a convenience, never fail the rollout
            print(f"[force] force-tracking video skipped: {exc}")
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
        hard_exit(simulation_app, status)
