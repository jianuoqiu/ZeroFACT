"""Closed-loop force-tracking rollout in Isaac Lab.

Import this module **after** ``AppLauncher`` has started Isaac Sim (same rule as
``v2s2r_isaaclab.replay``). The scene, physics configuration and the step/update ordering are the
validated replay ones - the only change is who computes the joint-position target:

    replay :  target[frame]                      (recorded command, stale for the first 2 steps)
    here   :  middle_layer(t, measured forces)   (policy chunk + force feedback, every step)

With ``ControllerConfig.exact_replay()`` the middle layer reproduces the replay commands
bit-exactly, so the whole runner can be regression-tested against a recorded ``replay_data.npz``
(replays are bit-deterministic on this machine).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from v2s2r_isaaclab.replay import (
    ReplayConfig,
    SceneHandles,
    _update_force_arrows,
)
from v2s2r_isaaclab.scene_spec import FINGERTIP_LINKS, RunSpec

from .config import ControllerConfig
from .episode import ReplayEpisode, pad_contact_centroid
from .middle_layer import HybridForceMiddleLayer, Measured
from .policy import BasePolicy, ChunkedReplayPolicy


def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _tip_point_jacobians(
    robot,
    tip_body_ids: list[int],
    tip_pos: np.ndarray,
    contact_pos: np.ndarray | None,
    predicted_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Positional Jacobian of each fingertip's contact point, world frame, ``[tips, 3, J]``.

    PhysX reports per-body Jacobians (fixed-base articulations exclude the root link, hence the
    index offset probe). The body Jacobian is shifted from the body origin to the contact point p
    via ``J_p = J_v - skew(p - p_body) J_w``.

    The anchor point falls back in order: the **measured** pad centroid, then the **predicted**
    contact point (so the feedforward is already anchored where the touch is about to happen
    while the pad still reads nothing), then the fingertip body origin.
    """
    jac = robot.root_physx_view.get_jacobians()[0].cpu().numpy()      # [B or B-1, 6, J]
    base_offset = 1 if jac.shape[0] == robot.num_bodies - 1 else 0
    out = np.zeros((len(tip_body_ids), 3, jac.shape[-1]))
    for t, bid in enumerate(tip_body_ids):
        body_jac = jac[bid - base_offset]
        j_v, j_w = body_jac[:3], body_jac[3:]
        p = tip_pos[t]
        if contact_pos is not None and np.isfinite(contact_pos[t]).all():
            p = contact_pos[t]
        elif predicted_pos is not None and np.isfinite(predicted_pos[t]).all():
            p = predicted_pos[t]
        out[t] = j_v - _skew(p - tip_pos[t]) @ j_w
    return out


def run_force_tracking(
    spec: RunSpec,
    cfg: ReplayConfig,
    ctrl_cfg: ControllerConfig,
    episode: ReplayEpisode,
    scene: SceneHandles,
    sim,
    policy: BasePolicy | None = None,
) -> dict:
    """Roll the policy + middle layer out in simulation; return everything recorded."""
    robot = scene.robot
    device = robot.device

    # ---- align the episode with the live articulation ----
    episode = episode.reordered(list(robot.joint_names))
    if scene.contact_object_keys != episode.contact_object_keys:
        raise RuntimeError(
            "contact-sensor filter order differs from the episode "
            f"(scene {scene.contact_object_keys} vs episode {episode.contact_object_keys}); "
            "per-object force attribution would be misaligned"
        )
    if [s.body_names[0] for s in scene.contacts] != episode.fingertip_bodies:
        raise RuntimeError("fingertip sensor order differs from the episode")

    policy = policy or ChunkedReplayPolicy(episode, ctrl_cfg.policy)
    # the task-space law maps forces to offsets through the PD stiffness: use the articulation's
    # live values rather than assuming the configured constants
    joint_stiffness = None
    stiffness_attr = getattr(robot.data, "joint_stiffness", None)
    if stiffness_attr is not None:
        stiffness_np = stiffness_attr[0].cpu().numpy().astype(np.float64)
        if np.all(stiffness_np > 0):
            joint_stiffness = stiffness_np
    middle = HybridForceMiddleLayer(
        ctrl_cfg, episode.joint_names, episode.fingertip_bodies, joint_stiffness
    )
    needs_jacobians = ctrl_cfg.force_law.law == "task_space"
    predict_point = ctrl_cfg.force_law.predict_point_before_contact
    has_pairs = bool(scene.contact_object_keys)

    num_frames = episode.num_frames if cfg.max_frames is None else min(cfg.max_frames, episode.num_frames)
    steps = cfg.steps_per_frame
    n_tips = len(episode.fingertip_bodies)
    joint_limits = robot.data.joint_pos_limits[0].cpu().numpy()          # [J, 2]

    # ---- initial state: the episode's first command (== trajectory frame 0) ----
    q0 = episode.joint_target[0]
    q0_t = torch.as_tensor(q0, dtype=torch.float32, device=device).unsqueeze(0)
    robot.reset()
    robot.write_joint_state_to_sim(q0_t, torch.zeros_like(q0_t))
    robot.set_joint_position_target(q0_t)
    robot.write_data_to_sim()
    for obj in scene.objects.values():
        obj.reset()
        obj.write_data_to_sim()
    middle.reset(q0)
    policy.reset()

    for _ in range(cfg.settle_steps):
        sim.step(render=False)
        robot.update(cfg.physics_dt)
        for obj in scene.objects.values():
            obj.update(cfg.physics_dt)

    if cfg.render and scene.camera is not None:
        for _ in range(cfg.render_warmup_steps):
            sim.render()
            scene.camera.update(cfg.physics_dt, force_recompute=True)

    tip_body_ids = []
    for link in FINGERTIP_LINKS:
        ids, _ = robot.find_bodies(link, preserve_order=True)
        tip_body_ids.append(int(ids[0]))
    tracked_ids = []
    tracked_links = []
    for link in episode.tracked_links:
        ids, _ = robot.find_bodies(link, preserve_order=True)
        if ids:
            tracked_ids.append(int(ids[0]))
            tracked_links.append(link)

    object_keys = list(scene.objects.keys())
    rec: dict[str, list] = {k: [] for k in [
        "joint_pos", "joint_vel", "joint_target",
        "action_steps", "q_ref_steps", "dq_steps", "f_ref_steps", "f_meas_steps", "u_steps",
        "f_ref_vec_steps", "f_meas_vec_steps", "p_ref_steps", "p_meas_steps", "point_err_steps",
        "ff_gain_steps",
        "body_pos", "body_quat",
        "object_pos", "object_quat", "object_lin_vel",
        "contact_force", "contact_force_steps", "contact_object_force_steps", "contact_point_w",
    ]}
    frames_rgb: list[np.ndarray] = []
    t_start = time.time()

    for frame in range(num_frames):
        if frame % policy.chunk == 0:
            obs = {"frame": frame, "joint_pos": robot.data.joint_pos[0].cpu().numpy()}
            middle.on_new_chunk(policy.predict(frame, obs))

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

        for step in range(steps):
            step_start = time.perf_counter()
            # ---- tactile reading: what the pads reported at the end of the previous step ----
            # Only what a fingertip tactile sensor can produce: the NET force on the pad and the
            # pad's contact centroid. PhysX's per-object force breakdown is NOT read here - it is
            # privileged information no real sensor has - it is only recorded for analysis below.
            f_net = np.stack([s.data.net_forces_w[0, 0].cpu().numpy() for s in scene.contacts])
            tip_contact_pos = None
            if has_pairs:
                pair_f = np.stack(
                    [s.data.force_matrix_w[0, 0].cpu().numpy() for s in scene.contacts]
                )                                                     # [tips, M, 3]
                pair_p = np.stack(
                    [s.data.contact_pos_w[0, 0].cpu().numpy() for s in scene.contacts]
                )                                                     # [tips, M, 3]
                # fuse the per-pair patches into one centroid per pad, object-agnostic
                tip_contact_pos = pad_contact_centroid(
                    pair_f.astype(np.float64), pair_p.astype(np.float64)
                )
            tip_pos = robot.data.body_pos_w[0, tip_body_ids].cpu().numpy().astype(np.float64)

            # ---- middle layer -> PD target ----
            t_cmd = middle.command_time(frame, step, steps)
            tip_jacobians = None
            if needs_jacobians:
                # peek at the reference first: the predicted contact point anchors the statics map
                # while the pad is not touching yet (peek does not advance the chunk anchor)
                predicted_pt = middle.peek_reference(t_cmd).contact_point if predict_point else None
                tip_jacobians = _tip_point_jacobians(
                    robot, tip_body_ids, tip_pos, tip_contact_pos, predicted_pt
                )
            meas = Measured(
                q=robot.data.joint_pos[0].cpu().numpy().astype(np.float64),
                qd=robot.data.joint_vel[0].cpu().numpy().astype(np.float64),
                f_net=f_net.astype(np.float64),
                tip_pos=tip_pos,
                tip_contact_pos=tip_contact_pos,
                tip_jacobians=tip_jacobians,
            )
            ctrl = middle.compute_action(t_cmd, meas, cfg.physics_dt)
            action = ctrl.action
            if ctrl_cfg.clamp_action_to_limits:
                action = np.clip(action, joint_limits[:, 0], joint_limits[:, 1])

            command = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            robot.set_joint_position_target(command)
            robot.write_data_to_sim()
            for obj in scene.objects.values():
                obj.write_data_to_sim()
            sim.step(render=False)
            robot.update(cfg.physics_dt)
            for obj in scene.objects.values():
                obj.update(cfg.physics_dt)
            for sensor in scene.contacts:
                sensor.update(cfg.physics_dt, force_recompute=True)

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

            if cfg.gui and step % cfg.gui_render_interval == 0:
                sim.render()
            if cfg.realtime:
                remaining = cfg.physics_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)

        # ---- record (same shapes/conventions as the replay recorder) ----
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

        rec["body_pos"].append(robot.data.body_pos_w[0, tracked_ids].cpu().numpy().copy())
        rec["body_quat"].append(robot.data.body_quat_w[0, tracked_ids].cpu().numpy().copy())

        obj_pos, obj_quat, obj_vel = [], [], []
        for key in object_keys:
            data = scene.objects[key].data
            obj_pos.append(data.root_pos_w[0].cpu().numpy().copy())
            obj_quat.append(data.root_quat_w[0].cpu().numpy().copy())
            obj_vel.append(data.root_lin_vel_w[0].cpu().numpy().copy())
        rec["object_pos"].append(np.stack(obj_pos))
        rec["object_quat"].append(np.stack(obj_quat))
        rec["object_lin_vel"].append(np.stack(obj_vel))

        pair_forces_np = contact_points_np = None
        if scene.contacts:
            step_net = [
                torch.flip(sensor.data.net_forces_w_history[0, :, 0], dims=[0])
                for sensor in scene.contacts
            ]
            net_steps = torch.stack(step_net, dim=1)
            rec["contact_force"].append(net_steps[-1].cpu().numpy().copy())
            rec["contact_force_steps"].append(net_steps.cpu().numpy().copy())
            if scene.contact_object_keys:
                step_obj = [
                    torch.flip(sensor.data.force_matrix_w_history[0, :, 0], dims=[0])
                    for sensor in scene.contacts
                ]
                obj_steps = torch.stack(step_obj, dim=1)
                rec["contact_object_force_steps"].append(obj_steps.cpu().numpy().copy())
                pair_forces_np = rec["contact_object_force_steps"][-1][-1]
                contact_points_np = np.stack(
                    [sensor.data.contact_pos_w[0, 0].cpu().numpy() for sensor in scene.contacts]
                )
                rec["contact_point_w"].append(contact_points_np.copy())

            if scene.force_markers is not None:
                _update_force_arrows(
                    scene.force_markers,
                    robot.data.body_pos_w[0, tip_body_ids].cpu().numpy(),
                    rec["contact_force"][-1],
                    pair_forces_np,
                    contact_points_np,
                    cfg.force_vis_scale,
                    cfg.force_vis_max_len,
                )

        if cfg.render and scene.camera is not None:
            sim.render()
            scene.camera.update(cfg.physics_dt, force_recompute=True)
            rgb = scene.camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            if cfg.video:
                frames_rgb.append(rgb)

        if frame % 20 == 0 or frame == num_frames - 1:
            print(
                f"[force] frame {frame + 1:4d}/{num_frames}  ({time.time() - t_start:6.1f}s elapsed)",
                flush=True,
            )

    return {
        "num_frames": num_frames,
        "joint_names": list(robot.joint_names),
        "fingertip_bodies": list(episode.fingertip_bodies),
        "tracked_links": tracked_links,
        "object_keys": object_keys,
        "object_names": scene.object_names,
        "manipulated_key": scene.manipulated_key,
        "contact_object_keys": list(scene.contact_object_keys),
        "wall_time_s": time.time() - t_start,
        "records": {k: (np.stack(v) if v else np.empty(0)) for k, v in rec.items()},
        "frames_rgb": frames_rgb,
    }


def diff_against_replay(records: dict, replay_npz: str | Path, num_frames: int) -> dict:
    """Max |difference| vs a recorded replay for every comparable key (exact-replay validation).

    Also reports the difference in float32-ulp units of the reference values: the physics rollout
    itself is bit-deterministic, but the GPU contact-*report* readback (``*_steps`` force keys)
    shows isolated 1-ulp reduction-order jitter that does not feed back into the dynamics. A key
    with ``max_ulp <= 1`` is therefore still a pass.
    """
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
