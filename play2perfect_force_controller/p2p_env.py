"""The play2perfect assembly scene as a force-controller test bench (import after ``AppLauncher``).

play2perfect's ``PreciseAssemblyEnv`` (KUKA iiwa14 + Sharpa hand, table, insertion part, receptacle,
goal marker) is reused **verbatim** as the world; this module only adds what the force-controller
pipeline needs on top of it:

* one ``ContactSensor`` per fingertip pad (net force at every physics substep, per-object force
  matrix, contact-patch centres) - the same sensing the Kinova/LEAP replays were recorded with;
* an optional demo camera and per-fingertip force arrows for the videos;
* a per-frame recording hook inside ``step()`` (the state at the end of a policy step, *before*
  the env auto-resets a finished episode, which the stock ``DirectRLEnv.step`` never exposes);
* snapshot / restore of an episode's initial state, so the force-controller rollout can start from
  exactly the recorded configuration;
* the released checkpoint loaded through play2perfect's own eval plumbing.

Timing: 120 Hz physics, ``decimation = 2`` -> one recorded frame is one 60 Hz policy step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from .p2p_paths import AGENT_ENTRY, TASK_ID, bootstrap

bootstrap()

import isaacsimenvs  # noqa: E402, F401  (side effect: registers the gym task id)
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _configure_agent,
    _load_env_cfg,
)
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim.schemas import activate_contact_sensors  # noqa: E402
from isaaclab.sim.utils import get_current_stage  # noqa: E402

from .p2p_vis import (  # noqa: E402
    hide_goal_marker, make_demo_camera, make_force_arrow_markers, make_goal_marker_translucent,
    update_force_arrows,
)
from .robot_spec import (  # noqa: E402
    CONTACT_OBJECT_KEYS,
    FINGERTIP_BODIES,
    PALM_LINK,
    STEPS_PER_FRAME,
    TRACKED_LINKS,
)

# scene asset name -> prim name under /World/envs/env_0 (play2perfect scene_utils.setup_scene)
ASSET_PRIMS = {"object": "Object", "hole": "Hole", "table": "Table", "goal_viz": "GoalViz"}
RECORDED_OBJECTS = ["object", "hole", "table", "goal_viz"]     # poses recorded every frame
# contact-point READBACK capacity per sensor (not physics): PhysX SDF meshes (beam parts, screw
# thread) report many contact points per pair; overflowing the buffer trips a CUDA assert
MAX_CONTACT_POINTS = 256


@dataclass
class SceneOptions:
    """What to add on top of the play2perfect env."""

    render: bool = True                              # demo camera + force arrows
    camera_eye: tuple[float, float, float] = (0.95, -0.62, 0.92)
    camera_target: tuple[float, float, float] = (0.0, 0.12, 0.60)
    camera_width: int = 640
    camera_height: int = 480
    force_vis_scale: float = 0.02                    # arrow length per newton (m/N)
    force_vis_max_len: float = 0.30                  # cap on the arrow shaft length (m)
    max_contact_points: int = MAX_CONTACT_POINTS     # sensor readback capacity per fingertip
    goal_marker_opacity: float = 0.18               # translucent target-pose marker (1.0 = opaque, as in play2perfect)
    goal_marker_visible: bool = True                # False: hide the marker entirely (clean images for a BC dataset)
    force_arrows: bool = True                        # in-scene fingertip force arrows (they show up in captured frames)
    render_in_step: bool = True                      # False: render ONLY at capture_frame() (the renderer is the
                                                     # bottleneck on a shared GPU; physics never depends on it)


def make_env_cfg(problem: str, seed: int = 0, sim_device: str = "cuda:0",
                 goal_mode: str = "preInsertAndFinal", render: bool = False, antialiasing: str = "TAA"):
    """The env config play2perfect's own evaluation uses (DR off), for one env, sensors eager."""
    cfg = _load_env_cfg(TASK_ID)
    _apply_env_overrides(
        cfg,
        problem=problem,
        goal_mode=goal_mode,
        random_goal_fraction=0.0,
        insertion_success_tolerance=0.01,
        retract_success_tolerance=0.005,
        num_envs=1,
        sim_device=sim_device,
        sdf=False,
        keep_dr=False,               # no obs/action delays, no random wrenches, no joint noise
        extra_overrides={},
    )
    # the fingertip sensors must refresh at every physics substep (their history buffer is the
    # per-step force record); with lazy updates they would only refresh once per policy step
    cfg.scene.lazy_sensor_update = False
    cfg.seed = int(seed)
    # play2perfect sizes the PhysX GPU work buffers for 8192 envs (2 GB collision stack alone);
    # one env needs a fraction of that. Capacities only bound the GPU workspaces (PhysX reports an
    # overflow instead of silently dropping contacts), so the physics is unchanged - the exact
    # replay check (--exact-replay must reproduce the recording bit for bit) is the proof; verified
    # on the heaviest scene (screwing) for both sets below. Default "tiny": physics-only process
    # 3.7 GB instead of 4.9 GB, which matters when another Kit process shares the GPU.
    buffers = os.environ.get("P2P_PHYSX_BUFFERS", "tiny")                # "bench" / "play2perfect" for A/B tests
    if buffers == "bench":
        cfg.sim.physx.gpu_collision_stack_size = 2**28
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**21
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**21
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**21
    elif buffers == "tiny":
        cfg.sim.physx.gpu_collision_stack_size = 2**26
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**19
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**17
        cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**19
        cfg.sim.physx.gpu_heap_capacity = 2**25
        cfg.sim.physx.gpu_temp_buffer_capacity = 2**23
        cfg.sim.physx.gpu_max_soft_body_contacts = 2**10
        cfg.sim.physx.gpu_max_particle_contacts = 2**10
    if render:
        # play2perfect trains with the cheapest RTX settings ("performance", no AA), which look
        # grainy in a demo video; the physics does not depend on rendering settings
        cfg.sim.render.rendering_mode = os.environ.get("P2P_RENDER_MODE", "balanced")
        # TAA (not FXAA): the translucent goal marker is a stochastic fractional cutout that only
        # temporal accumulation smooths; translucency must be on for any fractional opacity
        cfg.sim.render.antialiasing_mode = antialiasing      # FXAA when frames are rendered one at a time
        cfg.sim.render.enable_translucency = True
    return cfg


def _rigid_body_prim_paths(prim_path: str) -> list[str]:
    """Prims under *prim_path* carrying ``UsdPhysics.RigidBodyAPI`` (stage-context aware)."""
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return []
    return [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


class AssemblyBench(PreciseAssemblyEnv):
    """``PreciseAssemblyEnv`` + fingertip tactile sensors + recording hook + state restore."""

    def __init__(self, cfg, options: SceneOptions | None = None, **kwargs):
        self.options = options or SceneOptions()
        self.contacts: list[ContactSensor] = []
        self.contact_object_keys: list[str] = []
        self.camera = None
        self.force_markers = None
        self.frame_hook = None            # callable(env, done: bool), see step()
        self.substep_hook = None          # callable(env, substep: int) after every physics substep, see step()
        self._measured_torque_ok = None   # None = untested, then True/False (PhysX projected joint forces)
        self.last_action: torch.Tensor | None = None
        # set while the oracle chunk policy rolls the RL policy forward to PLAN: no auto-reset on
        # done, no rendering (the world is put back by restore_full() afterwards)
        self.planning = False
        super().__init__(cfg, **kwargs)
        # body ids the pipeline indexes with (fixed order from robot_spec, NOT regex order)
        self.tip_body_ids = [int(self.robot.find_bodies(b, preserve_order=True)[0][0]) for b in FINGERTIP_BODIES]
        self.palm_body_id = int(self.robot.find_bodies(PALM_LINK, preserve_order=True)[0][0])
        self.tracked_body_ids = [int(self.robot.find_bodies(b, preserve_order=True)[0][0]) for b in TRACKED_LINKS]
        assert self.num_envs == 1, "the recording/rollout tools drive a single environment"

    # ------------------------------------------------------------------ scene additions
    def _setup_scene(self) -> None:
        super()._setup_scene()
        # play2perfect spawns its assets without the PhysX contact-reporter API (its policy never
        # reads forces); the sensors need it on the fingertip bodies, and the pair filters work
        # best with it on both sides, so switch it on for the robot and the filtered bodies
        activate_contact_sensors("/World/envs/env_0/Robot", threshold=0.0)
        for key in CONTACT_OBJECT_KEYS:
            activate_contact_sensors(f"/World/envs/env_0/{ASSET_PRIMS[key]}", threshold=0.0)
        filter_paths = []
        for key in CONTACT_OBJECT_KEYS:
            paths = _rigid_body_prim_paths(f"/World/envs/env_0/{ASSET_PRIMS[key]}")
            if len(paths) != 1:
                raise RuntimeError(f"expected one rigid body under {ASSET_PRIMS[key]}, found {paths}")
            filter_paths.append(paths[0])
        self.contact_object_keys = list(CONTACT_OBJECT_KEYS)
        for link in FINGERTIP_BODIES:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link}",
                update_period=0.0,
                history_length=STEPS_PER_FRAME,
                track_air_time=False,
                filter_prim_paths_expr=list(filter_paths),
                track_contact_points=True,
                max_contact_data_count_per_prim=self.options.max_contact_points,
            ))
            self.scene.sensors[f"contact_{link}"] = sensor      # updated inside scene.update()
            self.contacts.append(sensor)
        print(f"[bench] fingertip contact sensors on {FINGERTIP_BODIES}; force matrix vs "
              f"{self.contact_object_keys} ({filter_paths})", flush=True)
        if self.options.render:
            self.camera = make_demo_camera(
                self.options.camera_eye, self.options.camera_target,
                self.options.camera_width, self.options.camera_height,
            )
            if self.options.force_arrows and not os.environ.get("P2P_NO_ARROWS"):   # P2P_NO_ARROWS: diagnostic switch
                self.force_markers = make_force_arrow_markers()
            # the marker's shader is edited (or the marker hidden) at the FIRST capture (after the sim
            # has started): doing it here, before the first render, made PhysX's first GPU kernels fail
            # to launch on some scenes (screwing seed 1 hung every time) when the force-arrow instancer
            # is also present
            self._goal_marker_pending = (not self.options.goal_marker_visible) or self.options.goal_marker_opacity < 1.0

    # ------------------------------------------------------------------ stepping
    def step(self, action: torch.Tensor):
        """``DirectRLEnv.step`` with one addition: ``frame_hook(env, done)`` runs after the physics
        substeps and the done/reward computation, but BEFORE finished episodes are reset, so the
        recorder sees the true end-of-frame state (the stock step returns post-reset buffers)."""
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)
        self.last_action = action.clone()
        self._pre_physics_step(action)
        is_rendering = ((self.sim.has_gui() or self.sim.has_rtx_sensors()) and not self.planning
                        and self.options.render_in_step)
        for substep in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            if self.substep_hook is not None and not self.planning:
                self.substep_hook(self, substep)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        if self.frame_hook is not None:
            self.frame_hook(self, bool(self.reset_buf[0].item()))
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0 and not self.planning:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def physics_step(self, joint_target: torch.Tensor) -> None:
        """One raw 1/120 s physics step under a joint-position target (the force-controller loop):
        the same write/step/update sequence as the env's decimation loop, nothing else."""
        self._sim_step_counter += 1
        self.robot.set_joint_position_target(joint_target)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    # ------------------------------------------------------------------ readers
    @property
    def fingertip_bodies(self) -> list[str]:
        return list(FINGERTIP_BODIES)

    @property
    def scene_objects(self) -> dict:
        return {"object": self.object, "hole": self.hole, "table": self.table, "goal_viz": self.goal_viz}

    def tactile_latest(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What the pads report at the latest physics step: net force ``[tips, 3]``, per-object
        force ``[tips, M, 3]`` and contact-patch centres ``[tips, M, 3]`` (NaN off-contact)."""
        f_net = np.stack([s.data.net_forces_w[0, 0].cpu().numpy() for s in self.contacts])
        pair_f = np.stack([s.data.force_matrix_w[0, 0].cpu().numpy() for s in self.contacts])
        pair_p = np.stack([s.data.contact_pos_w[0, 0].cpu().numpy() for s in self.contacts])
        return f_net.astype(np.float64), pair_f.astype(np.float64), pair_p.astype(np.float64)

    def tactile_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-physics-step forces of the last frame, chronological: net ``[S, tips, 3]`` and
        per-object ``[S, tips, M, 3]``."""
        net = torch.stack(
            [torch.flip(s.data.net_forces_w_history[0, :, 0], dims=[0]) for s in self.contacts], dim=1
        )
        obj = torch.stack(
            [torch.flip(s.data.force_matrix_w_history[0, :, 0], dims=[0]) for s in self.contacts], dim=1
        )
        return net.cpu().numpy().copy(), obj.cpu().numpy().copy()

    def tip_positions(self) -> np.ndarray:
        return self.robot.data.body_pos_w[0, self.tip_body_ids].cpu().numpy().astype(np.float64)

    def tactile_frame(self) -> dict[str, np.ndarray]:
        """:meth:`tactile_history` + the latest contact points in ONE device sync (the recorder's
        per-frame read): ``net_steps`` [S, tips, 3], ``obj_steps`` [S, tips, M, 3] (chronological),
        ``contact_point_w`` [tips, M, 3] float32 (NaN off-contact)."""
        net = torch.stack([torch.flip(s.data.net_forces_w_history[0, :, 0], dims=[0]) for s in self.contacts], dim=1)
        obj = torch.stack([torch.flip(s.data.force_matrix_w_history[0, :, 0], dims=[0]) for s in self.contacts], dim=1)
        pts = torch.stack([s.data.contact_pos_w[0, 0] for s in self.contacts])
        flat = torch.cat([net.reshape(-1), obj.reshape(-1), pts.reshape(-1)]).cpu().numpy()
        n_net, n_obj = net.numel(), obj.numel()
        return {
            "net_steps": flat[:n_net].reshape(net.shape).copy(),
            "obj_steps": flat[n_net:n_net + n_obj].reshape(obj.shape).copy(),
            "contact_point_w": flat[n_net + n_obj:].reshape(pts.shape).astype(np.float32).copy(),
        }

    def scene_frame(self) -> dict[str, np.ndarray]:
        """Robot + object state of the frame in ONE device sync: joint_pos/vel/target [J], body_pos
        [K, 3] / body_quat [K, 4] of the tracked links, object_pos [N, 3] / object_quat [N, 4] /
        object_lin_vel [N, 3] of RECORDED_OBJECTS, successes, retract_phase, keypoints_max_dist."""
        d = self.robot.data
        objs = [self.scene_objects[k].data for k in RECORDED_OBJECTS]
        parts = [
            d.joint_pos[0], d.joint_vel[0], d.joint_pos_target[0],
            d.body_pos_w[0, self.tracked_body_ids].reshape(-1), d.body_quat_w[0, self.tracked_body_ids].reshape(-1),
            torch.stack([o.root_pos_w[0] for o in objs]).reshape(-1),
            torch.stack([o.root_quat_w[0] for o in objs]).reshape(-1),
            torch.stack([o.root_lin_vel_w[0] for o in objs]).reshape(-1),
            torch.stack([self._successes[0].float(), self.retract_phase[0].float(), self._keypoints_max_dist[0].float()]),
        ]
        sizes = [p.numel() for p in parts]
        flat = torch.cat([p.reshape(-1).float() for p in parts]).cpu().numpy()
        chunks, i = [], 0
        for n in sizes:
            chunks.append(flat[i:i + n])
            i += n
        n_j, n_k, n_o = d.joint_pos.shape[1], len(self.tracked_body_ids), len(RECORDED_OBJECTS)
        return {
            "joint_pos": chunks[0].copy(), "joint_vel": chunks[1].copy(), "joint_target": chunks[2].copy(),
            "body_pos": chunks[3].reshape(n_k, 3).copy(), "body_quat": chunks[4].reshape(n_k, 4).copy(),
            "object_pos": chunks[5].reshape(n_o, 3).copy(), "object_quat": chunks[6].reshape(n_o, 4).copy(),
            "object_lin_vel": chunks[7].reshape(n_o, 3).copy(),
            "successes": int(round(float(chunks[8][0]))), "retract_phase": bool(chunks[8][1] > 0.5),
            "keypoints_max_dist": float(chunks[8][2]),
        }

    # ------------------------------------------------------------------ joint-level "tactile" readers
    def measured_joint_torque(self) -> torch.Tensor:
        """PhysX's projected joint force per DOF ``[J]``: the joint reaction projected on the joint
        axis (active component), i.e. the torque the joint actually transmits under gravity and
        contact loads. NaN if the tensor API cannot provide it on this articulation."""
        if self._measured_torque_ok is not False:
            try:
                t = self.robot.root_physx_view.get_dof_projected_joint_forces()[0]
                self._measured_torque_ok = True
                return t
            except Exception as exc:                     # noqa: BLE001
                if self._measured_torque_ok is None:
                    print(f"[bench] get_dof_projected_joint_forces unavailable ({exc}); measured torque = NaN", flush=True)
                self._measured_torque_ok = False
        return torch.full_like(self.robot.data.joint_pos[0], float("nan"))

    def joint_drive_signals(self) -> dict[str, np.ndarray]:
        """Per-joint signals at the latest physics substep (articulation joint order, float32):
        ``joint_pos`` / ``joint_vel`` / ``joint_target``; ``applied_torque`` = the implicit-PD drive
        torque K (q_t - q) + D (0 - qd) clipped to the joint's effort limit (what a motor-side
        torque or current sensor would report; Isaac Lab computes it from the pre-step state at
        write_data_to_sim); ``computed_torque`` = the same before clipping; ``measured_torque`` =
        :meth:`measured_joint_torque`."""
        d = self.robot.data
        names = ("joint_pos", "joint_vel", "joint_target", "applied_torque", "computed_torque", "measured_torque")
        stacked = torch.stack([d.joint_pos[0], d.joint_vel[0], d.joint_pos_target[0], d.applied_torque[0],
                               d.computed_torque[0], self.measured_joint_torque()]).detach().cpu().numpy()
        return {k: stacked[i].astype(np.float32).copy() for i, k in enumerate(names)}

    def joint_wrench_b(self) -> np.ndarray:
        """Incoming joint wrench of every body ``[B, 6]`` (force xyz, torque xyz) in the joint's
        child frame (PhysX ``get_link_incoming_joint_force``); NaN if unavailable."""
        try:
            return self.robot.data.body_incoming_joint_wrench_b[0].detach().cpu().numpy().astype(np.float32).copy()
        except Exception as exc:                         # noqa: BLE001
            if not getattr(self, "_wrench_warned", False):
                self._wrench_warned = True
                print(f"[bench] body_incoming_joint_wrench_b unavailable ({exc}); joint wrench = NaN", flush=True)
            return np.full((self.robot.num_bodies, 6), np.nan, dtype=np.float32)

    def joint_properties(self) -> dict[str, np.ndarray]:
        """Static per-joint properties of the live articulation (effort limits, PD gains, armature,
        position limits) - recorded with every episode so the fake motor signals are reproducible."""
        d = self.robot.data
        props = {}
        for key, attr in (("joint_effort_limits", "joint_effort_limits"), ("joint_stiffness", "joint_stiffness"),
                          ("joint_damping", "joint_damping"), ("joint_armature", "joint_armature"),
                          ("joint_pos_limits", "joint_pos_limits"), ("joint_vel_limits", "joint_vel_limits")):
            val = getattr(d, attr, None)
            if val is not None:
                props[key] = val[0].detach().cpu().numpy().astype(np.float64).copy()
        return props

    def draw_forces(self) -> None:
        if self.force_markers is None:
            return
        f_net, pair_f, pair_p = self.tactile_latest()
        update_force_arrows(self.force_markers, self.tip_positions(), f_net, pair_f, pair_p,
                            self.options.force_vis_scale, self.options.force_vis_max_len)

    def capture_frame(self) -> np.ndarray | None:
        """Render the demo camera now and return the RGB image ``[H, W, 3]`` (uint8)."""
        if self.camera is None:
            return None
        if getattr(self, "_goal_marker_pending", False):
            self._goal_marker_pending = False
            marker = f"/World/envs/env_0/{ASSET_PRIMS['goal_viz']}"
            if not self.options.goal_marker_visible:
                hide_goal_marker(marker)
            else:
                make_goal_marker_translucent(marker, self.options.goal_marker_opacity)
        self.sim.render()
        self.camera.update(self.physics_dt, force_recompute=True)
        return self.camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------ initial-state transfer
    def snapshot_state(self) -> dict[str, np.ndarray]:
        """Everything needed to put the scene back where an episode started (world frame)."""
        st = {
            "joint_pos": self.robot.data.joint_pos[0],
            "joint_vel": self.robot.data.joint_vel[0],
            "joint_target": self._cur_targets[0],
            "object_root_state": self.object.data.root_state_w[0],      # pos, quat wxyz, lin, ang
            "hole_root_state": self.hole.data.root_state_w[0],
            "table_root_state": self.table.data.root_state_w[0],
            "goal_root_state": self.goal_viz.data.root_state_w[0],
            "hole_pos_local": self.hole_pos[0],
            "hole_quat_wxyz": self.hole_quat_wxyz[0],
            "table_z": self._table_z_per_env[0:1],
            "object_init_z": self._object_init_z[0:1],
            "env_origin": self.scene.env_origins[0],
        }
        return {k: v.detach().cpu().numpy().astype(np.float64).copy() for k, v in st.items()}

    def restore_state(self, state: dict) -> None:
        """Overwrite the scene with a :meth:`snapshot_state` dict (call after ``reset()``)."""
        dev = self.device

        def t(a):
            return torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev).reshape(1, -1)

        self.robot.write_joint_state_to_sim(t(state["joint_pos"]), t(state["joint_vel"]))
        target = t(state["joint_target"])
        self._cur_targets[:] = target
        self._prev_targets[:] = target
        self.robot.set_joint_position_target(target)
        for name, key in (("object", "object_root_state"), ("hole", "hole_root_state"),
                          ("table", "table_root_state"), ("goal_viz", "goal_root_state")):
            root = t(state[key])
            body = getattr(self, name)
            body.write_root_pose_to_sim(root[:, :7])
            body.write_root_velocity_to_sim(root[:, 7:])
        self.hole_pos[0] = t(state["hole_pos_local"])[0]
        self.hole_quat_wxyz[0] = t(state["hole_quat_wxyz"])[0]
        self._table_z_per_env[0] = float(np.asarray(state["table_z"]).reshape(-1)[0])
        self._object_init_z[0] = float(np.asarray(state["object_init_z"]).reshape(-1)[0])
        self.scene.write_data_to_sim()
        self.sim.forward()


    # ------------------------------------------------------------------ closed-loop support
    # Used when the position layer is the RL policy itself (``oracle_policy.OracleChunkPolicy``):
    # the middle layer drives the substeps through ``physics_step`` and the env's task logic has
    # to keep running around it, and the planner has to be able to roll the world forward and
    # put it back.
    def observe(self) -> dict:
        """The policy observation of the CURRENT state (what ``step()`` would have returned at the
        end of the previous frame), without stepping anything."""
        obs = self._get_observations()
        if self.cfg.observation_noise_model:
            obs["policy"] = self._observation_noise_model(obs["policy"])
        return obs

    def post_frame_bookkeeping(self, joint_target: torch.Tensor) -> dict:
        """The non-physics tail of ``step()`` for a frame whose substeps were driven externally:
        the applied command becomes the policy's "previous target" (the base of its action
        smoothing and part of its observation), the episode clock advances and the task logic
        runs - success counting, sub-goal advance, retract phase, terminations - but the finished
        episode is NOT reset. Returns the env's own verdict for this frame."""
        self._cur_targets[:] = joint_target
        self._prev_targets = self._cur_targets.clone()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        reasons = {k: bool(v[0].item()) for k, v in getattr(self, "_termination_reasons", {}).items()}
        return {
            "done": bool(self.reset_buf[0].item()),
            "terminated": bool(self.reset_terminated[0].item()),
            "time_out": bool(self.reset_time_outs[0].item()),
            "termination": reasons,
            "successes": int(self._successes[0].item()),
            "max_goals": int(self.env_max_goals[0].item()),
            "retract_phase": bool(self.retract_phase[0].item()),
            "retract_succeeded": bool(self.retract_succeeded[0].item()),
            "keypoints_max_dist": float(self._keypoints_max_dist[0].item()),
        }

    # everything step() touches besides the physics state: the env's bookkeeping tensors
    # (targets, success/retract flags, goal trackers, episode clock), a few counters, the
    # tactile-sensor buffers and the torch RNG
    _SCALAR_STATE = ("common_step_counter", "_sim_step_counter", "_frame_counter",
                     "_last_curriculum_update", "_current_success_tolerance")

    def snapshot_full(self) -> dict:
        """Mid-episode snapshot: :meth:`snapshot_state` plus the env's own bookkeeping, the
        sensor buffers and the RNG, so that a planning rollout can be undone exactly (up to the
        PhysX contact cache, which cannot be read back)."""
        tensors = {k: v.detach().clone() for k, v in vars(self).items() if torch.is_tensor(v)}
        scalars = {k: getattr(self, k) for k in self._SCALAR_STATE if hasattr(self, k)}
        sensors = [{k: v.detach().clone() for k, v in vars(s.data).items() if torch.is_tensor(v)}
                   for s in self.contacts]
        return {
            "sim": self.snapshot_state(),
            "tensors": tensors, "scalars": scalars, "sensors": sensors,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None,
        }

    def restore_full(self, snap: dict) -> None:
        """Undo everything since :meth:`snapshot_full`."""
        self.restore_state(snap["sim"])
        for k, v in snap["tensors"].items():
            cur = getattr(self, k, None)
            if torch.is_tensor(cur) and cur.shape == v.shape and cur.dtype == v.dtype:
                try:
                    cur.copy_(v)
                    continue
                except RuntimeError:      # expanded / read-only views: re-bind instead
                    pass
            setattr(self, k, v.clone())
        for k, v in snap["scalars"].items():
            setattr(self, k, v)
        # the physics state was written directly: drop every lazily refreshed asset buffer so
        # the next read fetches the restored poses instead of the planner's last ones
        for asset in (self.robot, self.object, self.hole, self.table, self.goal_viz):
            for buf in vars(asset.data).values():
                if hasattr(buf, "timestamp"):
                    buf.timestamp = -1.0
        for sensor, saved in zip(self.contacts, snap["sensors"]):
            for k, v in saved.items():
                getattr(sensor.data, k).copy_(v)
        torch.set_rng_state(snap["rng_cpu"])
        if snap["rng_cuda"] is not None:
            torch.cuda.set_rng_state(snap["rng_cuda"], self.device)


# ----------------------------------------------------------------------------------------------
# the released RL policy
# ----------------------------------------------------------------------------------------------
def load_policy(env: AssemblyBench, checkpoint, deterministic: bool = True, rl_device: str = "cuda:0",
                seed: int = 0):
    """rl_games player for a play2perfect checkpoint, wired to *env* exactly as
    ``evaluation/eval_isaacsim.py`` does. Returns ``(player, wrapped_env)``; drive it with
    ``obs = player.env_reset(wrapped)`` / ``player.get_action(obs, is_deterministic=...)`` /
    ``player.env_step(wrapped, action)``."""
    import math

    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights

    agent_cfg = _configure_agent(
        TASK_ID, AGENT_ENTRY, rl_device=rl_device, num_envs=env.num_envs,
        deterministic=deterministic, games=10**9, extra_overrides={},
    )
    # rl_games re-seeds torch/numpy from its own config when the runner loads, which would make
    # every collection start from the same reset pose regardless of --seed
    agent_cfg["params"]["seed"] = int(seed)
    # the SAPG network has one extra_params / sigma row per exploration block, and the player
    # cannot infer that count from its single env: read it off the checkpoint itself so a policy
    # trained with any num_envs / expl_coef_block_size loads (released Sharpa 6, XHand here 4)
    import torch as _torch

    # a SAPG checkpoint is keyed by policy index ({0: {"model": state_dict, ...}})
    _ckpt = _torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    _inner = _ckpt[0] if isinstance(_ckpt, dict) and 0 in _ckpt else _ckpt
    _extra = (_inner or {}).get("model", {}).get("a2c_network.extra_params")
    if _extra is None:
        raise RuntimeError(
            f"{checkpoint}: no a2c_network.extra_params in the checkpoint; cannot tell how many "
            "SAPG exploration blocks the player must build"
        )
    agent_cfg["params"]["config"]["expl_num_blocks"] = int(_extra.shape[0])
    # condition the policy on the EXPLOIT block (intrinsic-exploration coefficient 0), not
    # the most exploratory one: the block ids are linspace(50, 0, blocks) and rl_games
    # spreads them over the envs, so a single-env player would otherwise always get id 50.
    agent_cfg["params"]["config"]["expl_eval_coef_id"] = 0.0
    print(f"[bench] SAPG exploration blocks in the checkpoint: {int(_extra.shape[0])}", flush=True)
    del _ckpt, _inner, _extra
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions)
    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.set_weights(_load_checkpoint_weights(player, str(checkpoint)))
    player.has_batch_dimension = True
    env.seed(int(seed))              # the reset randomisation (start poses) follows --seed again
    print(f"[bench] policy weights <- {checkpoint}  (seed {seed})", flush=True)
    return player, wrapped
