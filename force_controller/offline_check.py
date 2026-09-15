#!/usr/bin/env python3
"""Validate the force-controller pipeline offline - no Isaac Sim needed.

Feeds the recorded per-step tactile readings back into the middle layer as if they were live
sensor data and checks the plumbing end to end:

1. **Exact-replay self-test**: with ``state_source=joint_target``, ``interp=hold``,
   ``latency_steps=2`` and the null force law, the middle layer must reconstruct the recorded
   ``joint_target`` sequence *bit-exactly* (that is literally what the replay commanded). Any
   mismatch is a framework bug, not a tuning problem.
2. **Requested configuration**: runs the configured policy/middle-layer combination open-loop
   against the recorded measurements and writes plots + metrics. The loop is NOT closed here -
   closing it needs the sim runner - and the ``task_space`` law needs contact-point Jacobians the
   sim runner alone can provide, so its offset degrades to zero offline. What this rung *does*
   validate without a simulator: chunking/interpolation/latency timing, reference continuity, the
   command lead the force law has to recover, and how well the predicted force target and the
   predicted **contact point** match the recorded ones.

Usage::

    python force_controller/offline_check.py --episode outputs/run_2026-05-15_17-55-22
    python force_controller/offline_check.py --episode outputs/run_2026-05-16_01-27-32/20260827_143709 \
        --chunk 8 --horizon 16
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

from force_controller.config import ControllerConfig, ForceLawConfig, PolicyConfig, ReferenceConfig  # noqa: E402
from force_controller.episode import ReplayEpisode  # noqa: E402
from force_controller.metrics import (  # noqa: E402
    contact_point_metrics,
    format_contact_point_metrics,
    vector_force_metrics,
)
from force_controller.middle_layer import HybridForceMiddleLayer, Measured  # noqa: E402
from force_controller.plots import (  # noqa: E402
    force_tracking_metrics,
    plot_action_comparison,
    plot_contact_point_error,
    plot_force_tracking,
)
from force_controller.policy import ChunkedReplayPolicy  # noqa: E402


def offline_rollout(episode: ReplayEpisode, cfg: ControllerConfig, num_frames: int | None = None) -> dict:
    """Run policy + middle layer against recorded measurements; return per-step arrays."""
    T = episode.num_frames if num_frames is None else min(num_frames, episode.num_frames)
    S = episode.steps_per_frame
    J = episode.num_joints
    dt = episode.frame_dt / S
    tips = len(episode.fingertip_bodies)

    policy = ChunkedReplayPolicy(episode, cfg.policy)
    middle = HybridForceMiddleLayer(cfg, episode.joint_names, episode.fingertip_bodies)
    q0 = episode.joint_target[0]
    middle.reset(q0)

    # recorded tactile readings flattened to global steps: reading at the end of step g. Only the
    # NET pad force is used - the same signal the closed loop is allowed to see.
    f_net_steps = episode.contact_force_steps[:T].reshape(T * S, tips, 3)
    # the pad contact centroid is recorded per frame, not per step: hold it across the frame
    p_meas_frames = episode.fingertip_contact_point()[:T]                # [T, 4, 3]

    out = {
        "action_steps": np.zeros((T, S, J)),
        "q_ref_steps": np.zeros((T, S, J)),
        "dq_steps": np.zeros((T, S, J)),
        "f_ref_steps": np.zeros((T, S, tips)),
        "f_meas_steps": np.zeros((T, S, tips)),
        "u_steps": np.zeros((T, S, tips)),
        "f_ref_vec_steps": np.zeros((T, S, tips, 3)),
        "f_meas_vec_steps": np.zeros((T, S, tips, 3)),
        "p_ref_steps": np.full((T, S, tips, 3), np.nan),
        "p_meas_steps": np.full((T, S, tips, 3), np.nan),
    }

    for frame in range(T):
        if frame % cfg.policy.chunk == 0:
            middle.on_new_chunk(policy.predict(frame))
        for step in range(S):
            g = frame * S + step
            # the controller acts on the previous step's sensor reading (1-step feedback delay,
            # like the live loop); before the first step there is no reading yet
            if g == 0:
                f_net = np.zeros((tips, 3))
                p_meas = np.full((tips, 3), np.nan)
                q_meas, qd_meas = q0.copy(), np.zeros(J)
            else:
                f_net = f_net_steps[g - 1]
                prev_frame = (g - 1) // S
                p_meas = p_meas_frames[prev_frame]
                q_meas = episode.joint_pos[prev_frame]
                qd_meas = episode.joint_vel[prev_frame]
            meas = Measured(q=q_meas, qd=qd_meas, f_net=f_net, tip_contact_pos=p_meas)
            t_cmd = middle.command_time(frame, step, S)
            ctrl = middle.compute_action(t_cmd, meas, dt)
            out["action_steps"][frame, step] = ctrl.action
            out["q_ref_steps"][frame, step] = ctrl.q_ref
            out["dq_steps"][frame, step] = ctrl.dq
            out["f_ref_steps"][frame, step] = ctrl.f_ref
            out["f_meas_steps"][frame, step] = ctrl.f_meas
            out["f_ref_vec_steps"][frame, step] = ctrl.f_ref_vec
            out["f_meas_vec_steps"][frame, step] = ctrl.f_meas_vec
            out["p_ref_steps"][frame, step] = ctrl.p_ref
            out["p_meas_steps"][frame, step] = ctrl.p_meas
            if "u" in ctrl.law_info:
                out["u_steps"][frame, step] = ctrl.law_info["u"]

    out["action"] = out["action_steps"][:, -1, :]        # command in effect at each frame end
    out["num_frames"] = T
    return out


def exact_replay_self_test(episode: ReplayEpisode, chunk: int, horizon: int) -> float:
    """Reconstruct the recorded commands through the full pipeline; return the max abs error."""
    cfg = ControllerConfig(
        policy=PolicyConfig(horizon=horizon, chunk=chunk, state_source="joint_target"),
        reference=ReferenceConfig(interp="hold", latency_steps=2),
        force_law=ForceLawConfig(law="null"),
    )
    result = offline_rollout(episode, cfg)
    T, S = result["action_steps"].shape[:2]
    # what the replay commanded at (frame, step): the previous frame's target for the first
    # `stale_target_steps` steps, the current frame's target afterwards
    expected = np.empty_like(result["action_steps"])
    for f in range(T):
        prev = episode.joint_target[max(f - 1, 0)]
        expected[f, :2] = prev if f > 0 else episode.joint_target[0]
        expected[f, 2:] = episode.joint_target[f]
    return float(np.abs(result["action_steps"] - expected).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", required=True, help="outputs/<run>[/<stamp>] with replay_data.npz")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--state-source", choices=["joint_pos", "joint_target"], default="joint_pos")
    parser.add_argument("--interp", choices=["linear", "hold"], default="linear")
    parser.add_argument("--latency-steps", type=int, default=0)
    parser.add_argument("--force-law", choices=["null", "task_space"], default="task_space",
                        help="task_space needs contact-point Jacobians, which only the sim runner "
                        "provides - offline its offset degrades to zero (timing, reference and "
                        "target checks still run)")
    parser.add_argument("--kff", type=float, default=None, help="task_space: feedforward on f_ref")
    parser.add_argument("--task-kp", type=float, default=None, help="task_space: P on the force error")
    parser.add_argument("--task-ki", type=float, default=None, help="task_space: I rate (1/s)")
    parser.add_argument("--task-kd", type=float, default=None, help="task_space: D gain (s)")
    parser.add_argument("--point-kp", type=float, default=None,
                        help="task_space: P on the tangential contact-point error")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    episode_dir = ReplayEpisode.resolve_dir(args.episode, PROJECT_ROOT / "outputs")
    episode = ReplayEpisode.load(episode_dir)
    print(f"[offline] episode: {episode_dir}  ({episode.run_name}, {episode.num_frames} frames, "
          f"{episode.steps_per_frame} steps/frame, manipulated={episode.manipulated_key})")

    # ---- 1. framework self-test -------------------------------------------------------------
    err = exact_replay_self_test(episode, args.chunk, args.horizon)
    status = "OK" if err == 0.0 else "FAIL"
    print(f"[offline] exact-replay self-test: max |action - recorded command| = {err:.3e}  [{status}]")
    if err > 0.0:
        print("[offline] the chunking/interpolation/latency plumbing does not reproduce the "
              "recorded commands - fix this before trusting anything downstream.")
        return 1

    # ---- 2. requested configuration ---------------------------------------------------------
    law_cfg = ForceLawConfig(law=args.force_law)
    for attr, value in [("kff", args.kff), ("task_kp", args.task_kp), ("task_ki", args.task_ki),
                        ("task_kd", args.task_kd), ("point_kp", args.point_kp)]:
        if value is not None:
            setattr(law_cfg, attr, value)
    cfg = ControllerConfig(
        policy=PolicyConfig(horizon=args.horizon, chunk=args.chunk, state_source=args.state_source),
        reference=ReferenceConfig(interp=args.interp, latency_steps=args.latency_steps),
        force_law=law_cfg,
    )
    result = offline_rollout(episode, cfg, args.max_frames)
    T = result["num_frames"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / "force_controller" / "offline"
                               / f"{episode.run_name}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # continuity: the action must never jump, chunk switches included
    step_jump = np.abs(np.diff(result["action_steps"].reshape(T * episode.steps_per_frame, -1), axis=0))
    print(f"[offline] action continuity: max per-step |delta| = {step_jump.max():.5f} rad "
          f"(interp={cfg.reference.interp})")

    # how far the action is from the command that actually produced the episode
    gap = np.abs(result["action"] - episode.joint_target[:T])
    hand = [i for i, n in enumerate(episode.joint_names) if n.startswith("leap_")]
    print(f"[offline] |action - recorded command|: hand mean {gap[:, hand].mean():.5f} rad, "
          f"hand max {gap[:, hand].max():.5f} rad")

    metrics = force_tracking_metrics(
        result["f_ref_steps"], result["f_meas_steps"], episode.fingertip_bodies,
        cfg.force_law.engage_threshold_n,
    )
    vec_metrics = vector_force_metrics(
        result["f_ref_vec_steps"], result["f_meas_vec_steps"], episode.fingertip_bodies,
        cfg.force_law.engage_threshold_n,
    )
    point_metrics = contact_point_metrics(
        result["p_ref_steps"], result["p_meas_steps"], result["f_ref_vec_steps"],
        episode.fingertip_bodies, cfg.force_law.engage_threshold_n,
    )
    for i, body in enumerate(metrics["bodies"]):
        print(f"[offline] {body:<16s} engaged {metrics['engaged_fraction'][i] * 100:5.1f}%  "
              f"rmse {metrics['rmse_engaged_N'][i]:7.3f} N  bias {metrics['bias_engaged_N'][i]:+7.3f} N"
              "   (vs recorded readings, open loop)")
    print("[offline] contact-point target vs recorded (open loop):")
    for line in format_contact_point_metrics(point_metrics, prefix="[offline]   "):
        print(line)

    plot_force_tracking(
        result["f_ref_steps"], result["f_meas_steps"], episode.fingertip_bodies,
        out_dir / "force_tracking.png", episode.steps_per_frame, episode.key_frames,
        u=result["u_steps"] if cfg.force_law.law != "null" else None,
        title=f"{episode.run_name}: predicted vs recorded force (open loop)",
        u_label="|integral| [N]",
        engage_threshold_n=cfg.force_law.engage_threshold_n,
    )
    plot_action_comparison(
        result["action"], episode.joint_target[:T], episode.joint_names,
        out_dir / "commands_vs_predicted.png", episode.key_frames,
        title=f"{episode.run_name}: recorded vs rollout commands vs predicted states "
              f"({cfg.policy.state_source}, law={cfg.force_law.law}, open loop)",
        predicted=result["q_ref_steps"][:, -1, :],
    )
    plot_contact_point_error(
        result["p_ref_steps"], result["p_meas_steps"], result["f_ref_vec_steps"],
        episode.fingertip_bodies, out_dir / "contact_point_error.png",
        episode.steps_per_frame, episode.key_frames, cfg.force_law.engage_threshold_n,
        title=f"{episode.run_name}: contact-point target vs recorded (open loop)",
    )

    summary = {
        "episode": str(episode_dir),
        "run": episode.run_name,
        "self_test_max_err": err,
        "config": asdict(cfg),
        "action_continuity_max_rad": float(step_jump.max()),
        "action_vs_recorded_hand_mean_rad": float(gap[:, hand].mean()),
        "action_vs_recorded_hand_max_rad": float(gap[:, hand].max()),
        "force_tracking_open_loop": metrics,
        "force_tracking_vector_open_loop": vec_metrics,
        "contact_point_open_loop": point_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[offline] outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
