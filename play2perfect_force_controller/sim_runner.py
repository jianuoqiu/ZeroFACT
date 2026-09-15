"""Closed-loop force-tracking rollout in the play2perfect scene (import after ``AppLauncher``).

The port of ``force_controller/sim_runner.py`` to the ``AssemblyBench`` world: the scene, physics
configuration and step/update ordering are play2perfect's own (``AssemblyBench.physics_step`` is
the env's decimation loop body); the only change is who computes the joint-position target:

    recording :  target[frame]                      (the RL policy's command, held for the frame)
    here      :  middle_layer(t, measured forces)   (policy chunk + force feedback, every step)

With ``ControllerConfig.exact_replay()`` the middle layer replays the recorded commands, so the
whole runner can be regression-tested against the episode's ``replay_data.npz``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import ControllerConfig
from .episode import ReplayEpisode, pad_contact_centroid
from .middle_layer import HybridForceMiddleLayer, Measured
from .p2p_env import RECORDED_OBJECTS, AssemblyBench
from .policy import BasePolicy, ChunkedReplayPolicy
from .robot_spec import FINGERTIP_PAD_OFFSETS


@dataclass
class RolloutConfig:
    max_frames: int | None = None
    render: bool = True                 # capture the demo camera (+ force arrows)
    video_every: int = 2                # capture every Nth frame (2 -> 30 fps from 60 Hz frames)
    # show the (collision-free) goal marker at the recording's FINAL goal pose for the whole
    # rollout. Replaying the recorded subgoal schedule instead (pre-insert -> final -> ... at the
    # frames where the RECORDED policy reached them) reads as a phantom part sinking on its own
    # while the real part is somewhere else.
    show_goal_marker: bool = True
    gui: bool = False
    gui_render_interval: int = 2
    realtime: bool = False
    # closed-loop (oracle) runs: reset the env with the recording's seed so the start - and
    # everything the env derives from it at reset - is the recording's, and let the rollout run
    # past the recording's length by this many frames (the env's own timeout still applies)
    reset_seed: int | None = None
    grace_frames: int = 600


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate *v* by the wxyz quaternion *q* (rows)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    qv = np.stack([x, y, z], axis=-1)
    t = 2.0 * np.cross(qv, v)
    return v + w[..., None] * t + np.cross(qv, t)


def _tip_point_jacobians(
    robot,
    tip_body_ids: list[int],
    tip_pos: np.ndarray,
    contact_pos: np.ndarray | None,
    predicted_pos: np.ndarray | None,
    fallback_pos: np.ndarray,
) -> np.ndarray:
    """Positional Jacobian of each fingertip's contact point, world frame, ``[tips, 3, J]``.

    The body Jacobian is shifted from the body origin to the contact point p via
    ``J_p = J_v - skew(p - p_body) J_w``. Anchor order: measured pad centroid, predicted contact
    point, then the pad centre (the DP body origin sits back at the joint, 2 cm from the pad).
    """
    jac = robot.root_physx_view.get_jacobians()[0].cpu().numpy()      # [B or B-1, 6, J]
    base_offset = 1 if jac.shape[0] == robot.num_bodies - 1 else 0
    out = np.zeros((len(tip_body_ids), 3, jac.shape[-1]))
    for t, bid in enumerate(tip_body_ids):
        body_jac = jac[bid - base_offset]
        j_v, j_w = body_jac[:3], body_jac[3:]
        p = fallback_pos[t]
        if contact_pos is not None and np.isfinite(contact_pos[t]).all():
            p = contact_pos[t]
        elif predicted_pos is not None and np.isfinite(predicted_pos[t]).all():
            p = predicted_pos[t]
        out[t] = j_v - _skew(p - tip_pos[t]) @ j_w
    return out


def run_force_tracking(
    env: AssemblyBench,
    ctrl_cfg: ControllerConfig,
    episode: ReplayEpisode,
    roll: RolloutConfig,
    policy: BasePolicy | None = None,
) -> dict:
    """Roll the policy + middle layer out in simulation; return everything recorded."""
    robot = env.robot
    device = robot.device
    physics_dt = env.physics_dt
    steps = int(env.cfg.decimation)

    # ---- align the episode with the live articulation ----
    episode = episode.reordered(list(robot.joint_names))
    if env.contact_object_keys != episode.contact_object_keys:
        raise RuntimeError(f"contact-sensor filter order differs from the episode "
                           f"({env.contact_object_keys} vs {episode.contact_object_keys})")
    if env.fingertip_bodies != episode.fingertip_bodies:
        raise RuntimeError("fingertip sensor order differs from the episode")
    if episode.steps_per_frame != steps:
        raise RuntimeError(f"episode has {episode.steps_per_frame} physics steps per frame, env {steps}")
    init = episode.summary.get("init_state")
    if not init:
        raise RuntimeError("episode summary.json has no init_state (recorded by collect_episodes.py)")

    policy = policy or ChunkedReplayPolicy(episode, ctrl_cfg.policy)
    # a closed-loop policy (oracle_policy.OracleChunkPolicy) needs the env's task logic to run
    # around the externally driven substeps, and updates its recurrent state per executed frame
    advance = getattr(policy, "advance", None)
    after_frame = getattr(policy, "after_frame", None)
    closed_loop = after_frame is not None
    # the task-space law maps forces to offsets through the PD stiffness: use the live values
    joint_stiffness = None
    stiffness_attr = getattr(robot.data, "joint_stiffness", None)
    if stiffness_attr is not None:
        stiffness_np = stiffness_attr[0].cpu().numpy().astype(np.float64)
        if np.all(stiffness_np > 0):
            joint_stiffness = stiffness_np
    effort_limits = None
    effort_attr = getattr(robot.data, "joint_effort_limits", None)
    if effort_attr is not None:
        effort_limits = effort_attr[0].cpu().numpy().astype(np.float64)
    middle = HybridForceMiddleLayer(ctrl_cfg, episode.joint_names, episode.fingertip_bodies, joint_stiffness,
                                    joint_effort_limits=effort_limits)
    if ctrl_cfg.force_law.law == "task_space":
        clip = middle.offset_clip
        hand = [i for i, n in enumerate(episode.joint_names) if clip[i] > 0]
        print("[force] per-joint offset clip (rad): " + ", ".join(
            f"{episode.joint_names[i].replace('left_', '')}={clip[i]:.2f}" for i in hand), flush=True)
    needs_jacobians = ctrl_cfg.force_law.law == "task_space"
    predict_point = ctrl_cfg.force_law.predict_point_before_contact

    if closed_loop:
        budget = episode.num_frames + int(roll.grace_frames)
        num_frames = budget if roll.max_frames is None else min(roll.max_frames, budget)
    else:
        num_frames = episode.num_frames if roll.max_frames is None else min(roll.max_frames, episode.num_frames)
    n_tips = len(episode.fingertip_bodies)
    joint_limits = robot.data.joint_pos_limits[0].cpu().numpy()          # [J, 2]
    tip_body_ids = env.tip_body_ids
    tracked_ids = env.tracked_body_ids
    pad_offsets = np.asarray(FINGERTIP_PAD_OFFSETS, dtype=np.float64)   # [tips, 3]

    # ---- initial state: exactly where the recorded episode started ----
    env.frame_hook = None
    env.reset(seed=roll.reset_seed)
    if roll.reset_seed is not None:
        # with the recording's seed the reset itself should reproduce the recorded start; report
        # any key that does not (the restore below fixes the poses either way)
        now = env.snapshot_state()
        worst = {k: float(np.abs(np.asarray(now[k], dtype=np.float64) - np.asarray(init[k], dtype=np.float64)).max())
                 for k in init if k in now and np.shape(now[k]) == np.shape(init[k])}
        bad = {k: v for k, v in worst.items() if v > 0.0}
        print("[force] same-seed reset vs recorded start: " + ("identical on every key" if not bad else
              "differs on " + ", ".join(f"{k} ({v:.3g})" for k, v in bad.items())), flush=True)
    env.restore_state(init)
    q0 = robot.data.joint_pos[0].cpu().numpy().astype(np.float64)
    middle.reset(q0)
    policy.reset()
    if roll.render and env.camera is not None:
        for _ in range(5):
            env.capture_frame()

    goal_idx = RECORDED_OBJECTS.index("goal_viz") if "goal_viz" in RECORDED_OBJECTS else None
    object_keys = list(RECORDED_OBJECTS)
    rec: dict[str, list] = {k: [] for k in [
        "joint_pos", "joint_vel", "joint_target",
        "action_steps", "q_ref_steps", "dq_steps", "f_ref_steps", "f_meas_steps", "u_steps",
        "f_ref_vec_steps", "f_meas_vec_steps", "p_ref_steps", "p_meas_steps", "point_err_steps",
        "ff_gain_steps", "limit_masked_steps",
        "body_pos", "body_quat",
        "object_pos", "object_quat", "object_lin_vel",
        "contact_force", "contact_force_steps", "contact_object_force_steps", "contact_point_w",
    ]}
    if closed_loop:
        rec.update({k: [] for k in ["successes", "retract_phase", "keypoints_max_dist"]})
    env_end = None                   # the env's own end-of-episode verdict, if it finished
    last_verdict = None
    chunk = None                     # the active policy chunk (its plan is recorded per frame)
    frames_rgb: list[np.ndarray] = []
    t_start = time.time()

    for frame in range(num_frames):
        if frame % policy.chunk == 0:
            obs = {"frame": frame, "joint_pos": robot.data.joint_pos[0].cpu().numpy()}
            chunk = policy.predict(frame, obs)
            middle.on_new_chunk(chunk)
        if advance is not None:
            advance()
        # (closed loop: the env moves its own goal marker through the sub-goals, as recorded)
        if roll.show_goal_marker and goal_idx is not None and frame == 0 and not closed_loop:
            pose = np.concatenate([episode.object_pos[-1, goal_idx], episode.object_quat[-1, goal_idx]])
            env.goal_viz.write_root_pose_to_sim(torch.as_tensor(pose, dtype=torch.float32, device=device).unsqueeze(0))

        step_action = np.zeros((steps, episode.num_joints))
        step_q_ref = np.zeros_like(step_action)
        step_dq = np.zeros_like(step_action)
        step_f_ref = np.zeros((steps, n_tips))
        step_f_meas = np.zeros((steps, n_tips))
        step_u = np.zeros((steps, n_tips))
        step_f_ref_vec = np.zeros((steps, n_tips, 3))
        step_f_meas_vec = np.zeros((steps, n_tips, 3))
        step_p_ref = np.full((steps, n_tips, 3), np.nan)
        step_p_meas = np.full((steps, n_tips, 3), np.nan)
        step_point_err = np.full((steps, n_tips), np.nan)
        step_ff_gain = np.ones((steps, n_tips))
        step_masked = np.zeros((steps, episode.num_joints), dtype=bool)

        for step in range(steps):
            step_start = time.perf_counter()
            # ---- tactile reading: what the pads reported at the end of the previous step ----
            # NET force + object-agnostic pad centroid only (the per-object breakdown is
            # privileged simulator information, recorded below for analysis, never fed back)
            f_net, pair_f, pair_p = env.tactile_latest()
            tip_contact_pos = pad_contact_centroid(pair_f, pair_p)
            tip_pos = env.tip_positions()
            tip_quat = robot.data.body_quat_w[0, tip_body_ids].cpu().numpy().astype(np.float64)
            pad_pos = tip_pos + _quat_apply(tip_quat, pad_offsets)

            # ---- middle layer -> PD target ----
            t_cmd = middle.command_time(frame, step, steps)
            tip_jacobians = None
            if needs_jacobians:
                predicted_pt = middle.peek_reference(t_cmd).contact_point if predict_point else None
                tip_jacobians = _tip_point_jacobians(
                    robot, tip_body_ids, tip_pos, tip_contact_pos, predicted_pt, pad_pos
                )
            meas = Measured(
                q=robot.data.joint_pos[0].cpu().numpy().astype(np.float64),
                qd=robot.data.joint_vel[0].cpu().numpy().astype(np.float64),
                f_net=f_net,
                tip_pos=tip_pos,
                tip_contact_pos=tip_contact_pos,
                tip_jacobians=tip_jacobians,
                joint_limits=joint_limits,
            )
            ctrl = middle.compute_action(t_cmd, meas, physics_dt)
            action = ctrl.action
            if ctrl_cfg.clamp_action_to_limits:
                action = np.clip(action, joint_limits[:, 0], joint_limits[:, 1])

            env.physics_step(torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0))

            step_action[step] = action
            step_q_ref[step] = ctrl.q_ref
            step_dq[step] = ctrl.dq
            step_f_ref[step] = ctrl.f_ref
            step_f_meas[step] = ctrl.f_meas
            step_f_ref_vec[step] = ctrl.f_ref_vec
            step_f_meas_vec[step] = ctrl.f_meas_vec
            step_p_ref[step] = ctrl.p_ref
            step_p_meas[step] = ctrl.p_meas
            if "u" in ctrl.law_info:
                step_u[step] = ctrl.law_info["u"]
            if "point_error" in ctrl.law_info:
                step_point_err[step] = ctrl.law_info["point_error"]
            if "ff_gain" in ctrl.law_info:
                step_ff_gain[step] = ctrl.law_info["ff_gain"]
            if "limit_masked" in ctrl.law_info:
                step_masked[step] = ctrl.law_info["limit_masked"]

            if roll.gui and step % roll.gui_render_interval == 0:
                env.sim.render()
            if roll.realtime:
                remaining = physics_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

        # ---- record (same shapes/conventions as the recorder) ----
        rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        rec["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        rec["joint_target"].append(step_action[-1].astype(np.float32).copy())
        rec["action_steps"].append(step_action.astype(np.float32))
        rec["q_ref_steps"].append(step_q_ref.astype(np.float32))
        rec["dq_steps"].append(step_dq.astype(np.float32))
        rec["f_ref_steps"].append(step_f_ref.astype(np.float32))
        rec["f_meas_steps"].append(step_f_meas.astype(np.float32))
        rec["u_steps"].append(step_u.astype(np.float32))
        rec["f_ref_vec_steps"].append(step_f_ref_vec.astype(np.float32))
        rec["f_meas_vec_steps"].append(step_f_meas_vec.astype(np.float32))
        rec["p_ref_steps"].append(step_p_ref.astype(np.float32))
        rec["p_meas_steps"].append(step_p_meas.astype(np.float32))
        rec["point_err_steps"].append(step_point_err.astype(np.float32))
        rec["ff_gain_steps"].append(step_ff_gain.astype(np.float32))
        rec["limit_masked_steps"].append(step_masked)
        rec["body_pos"].append(robot.data.body_pos_w[0, tracked_ids].cpu().numpy().copy())
        rec["body_quat"].append(robot.data.body_quat_w[0, tracked_ids].cpu().numpy().copy())
        # closed loop: the plan knot for this frame (the policy's own command and reached state)
        if chunk is not None and "plan_joint_target" in chunk.info:
            j = min(frame - chunk.start_frame, chunk.horizon - 1)
            rec.setdefault("plan_joint_target", []).append(chunk.info["plan_joint_target"][j].astype(np.float32))
            rec.setdefault("plan_joint_pos", []).append(chunk.info["plan_joint_pos"][j].astype(np.float32))
            rec.setdefault("plan_force", []).append(chunk.info["plan_force"][j].astype(np.float32))

        obj_pos, obj_quat, obj_vel = [], [], []
        for key in object_keys:
            data = env.scene_objects[key].data
            obj_pos.append(data.root_pos_w[0].cpu().numpy().copy())
            obj_quat.append(data.root_quat_w[0].cpu().numpy().copy())
            obj_vel.append(data.root_lin_vel_w[0].cpu().numpy().copy())
        rec["object_pos"].append(np.stack(obj_pos))
        rec["object_quat"].append(np.stack(obj_quat))
        rec["object_lin_vel"].append(np.stack(obj_vel))

        net_steps, obj_steps = env.tactile_history()
        rec["contact_force"].append(net_steps[-1].copy())
        rec["contact_force_steps"].append(net_steps)
        rec["contact_object_force_steps"].append(obj_steps)
        _, _, pair_p = env.tactile_latest()
        rec["contact_point_w"].append(pair_p.astype(np.float32))

        if closed_loop:
            last_verdict = after_frame(step_action[-1])
            rec["successes"].append(last_verdict["successes"])
            rec["retract_phase"].append(last_verdict["retract_phase"])
            rec["keypoints_max_dist"].append(last_verdict["keypoints_max_dist"])

        if roll.render and env.camera is not None and frame % roll.video_every == 0:
            env.draw_forces()
            frames_rgb.append(env.capture_frame())

        if closed_loop and last_verdict["done"]:
            env_end = {"frame": frame, **last_verdict}
            reasons = [k for k, v in last_verdict["termination"].items() if v]
            print(f"[force] env finished the episode at frame {frame + 1}: goals "
                  f"{last_verdict['successes']}/{last_verdict['max_goals']}, retract "
                  f"{'ok' if last_verdict['retract_succeeded'] else 'no'}, ended by {reasons}", flush=True)
            num_frames = frame + 1
            break

        if frame % 120 == 0 or frame == num_frames - 1:
            print(f"[force] frame {frame + 1:4d}/{num_frames}  ({time.time() - t_start:6.1f}s elapsed)",
                  flush=True)

    return {
        "num_frames": num_frames,
        "joint_names": list(robot.joint_names),
        "fingertip_bodies": list(episode.fingertip_bodies),
        "tracked_links": list(episode.tracked_links),
        "object_keys": object_keys,
        "object_names": object_keys,
        "manipulated_key": episode.manipulated_key,
        "contact_object_keys": list(env.contact_object_keys),
        "wall_time_s": time.time() - t_start,
        "records": {k: (np.stack(v) if v else np.empty(0)) for k, v in rec.items()},
        "frames_rgb": frames_rgb,
        "video_every": roll.video_every,
        "closed_loop": closed_loop,
        "joint_limits": joint_limits,
        "env_end": env_end,
        "env_last": last_verdict,
        "policy_stats": getattr(policy, "stats", None),
    }


def diff_against_replay(records: dict, replay_npz: str | Path, num_frames: int) -> dict:
    """Max |difference| vs the recorded episode for every comparable key (exact-replay check),
    also in float32-ulp units of the reference (the GPU contact-*report* readback shows isolated
    1-ulp jitter that never feeds back into the dynamics; ``max_ulp <= 1`` is still a pass)."""
    ref = np.load(replay_npz, allow_pickle=True)
    keys = [
        "joint_target", "joint_pos", "joint_vel",
        "object_pos", "object_quat", "object_lin_vel",
        "body_pos", "body_quat",
        "contact_force", "contact_force_steps", "contact_object_force_steps",
    ]
    out = {}
    for key in keys:
        if key not in ref.files or records.get(key) is None or not len(records[key]):
            continue
        a = np.asarray(records[key])[:num_frames]
        b = np.asarray(ref[key])[:num_frames]
        if a.shape != b.shape:
            out[key] = {"error": f"shape {a.shape} vs {b.shape}"}
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        entry = {"max_abs_diff": float(diff.max()), "max_ulp": 0.0}
        if entry["max_abs_diff"] > 0.0:
            ulp = np.spacing(np.abs(b).astype(np.float32)).astype(np.float64)
            entry["max_ulp"] = float((diff / ulp).max())
        out[key] = entry
    return out
