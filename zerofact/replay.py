"""Isaac Lab scene construction + trajectory replay for a Video2Sim2Real run.

This module must be imported **after** ``AppLauncher`` has started Isaac Sim.

The scene and the control loop are a port of the Isaac Gym replay in
``Video2Sim2Real/contact_opt/optimized_replay.py``. Where the two engines differ, the Isaac Gym
behaviour was measured (in the ``vid2sim2real`` env) and reproduced here rather than guessed:

======================================  ==========================================================
Isaac Gym                               Isaac Lab (here)
======================================  ==========================================================
``sim_params.dt = 1/60, substeps = 2``  ``SimulationCfg.dt = 1/120`` with twice the steps
6 ``simulate()`` per trajectory frame   ``STEPS_PER_FRAME = 12`` (0.1 s -> 10 Hz replay)
target set inside the inner loop        first ``STALE_TARGET_STEPS = 2`` steps hold frame k-1
``physx.solver_type = 1``               ``PhysxCfg.solver_type = 1`` (TGS)
``num_position/velocity_iterations``    ``Articulation/RigidBodyPropertiesCfg.solver_*_iteration``
``physx.contact_offset / rest_offset``  ``CollisionPropertiesCfg.contact_offset / rest_offset``
``AssetOptions.armature = 0.01``        baked into the URDF link inertias at ingest time, because
                                        Isaac Gym adds it to the *inertia diagonal*, not the joint
``create_box`` table + ``add_ground``   static ``CuboidCfg`` colliders (no rigid body)
loaded-asset default friction 1.0       ``SimulationCfg.physics_material`` = 1.0/1.0/0.0
``set_actor_rigid_shape_properties``    ``RigidBodyMaterialCfg`` + ``bind_physics_material``
``set_actor_scale(s)``                  ``UsdFileCfg.scale`` **plus** mass x s^3, com x s, I x s^5
``DOF_MODE_POS`` + stiffness/damping    ``ImplicitActuatorCfg(stiffness=..., damping=...)``
``set_dof_position_target_tensor``      ``Articulation.set_joint_position_target``
``set_dof_state_tensor`` (--visualize)  ``Articulation.write_joint_state_to_sim`` (--kinematic)
camera sensor 640x480 @ hfov 90         ``CameraCfg`` + ``PinholeCameraCfg.from_intrinsic_matrix``
fingertip force sensors (6-axis wrench  one ``ContactSensor`` per fingertip body: net contact
in world frame, read once per frame)    force at **every physics step** plus the per-object force
                                        matrix (fingertip <-> each scene object), world frame
======================================  ==========================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from .naming import format_mapping, match_joint_names
from .scene_spec import (
    BRACELET_LINK,
    FINGERTIP_COLORS,
    FINGERTIP_LINKS,
    HAND_MOUNT_LINK,
    PALM_LINK,
    RunSpec,
    load_joint_name_map,
    quat_xyzw_to_wxyz,
)
from .video import VideoWriter

# --------------------------------------------------------------------------------------
# Physics / control constants ported (and where noted, measured) from the Isaac Gym replay
# --------------------------------------------------------------------------------------
GYM_DT = 1.0 / 60.0            # sim_params.dt
GYM_SUBSTEPS = 2               # sim_params.substeps
GYM_STEPS_PER_FRAME = 6        # 1 simulate() before the inner loop + plan_every_n_steps (5)

PHYSICS_DT = GYM_DT / GYM_SUBSTEPS                        # 1/120 s == Isaac Gym's effective substep
STEPS_PER_FRAME = GYM_STEPS_PER_FRAME * GYM_SUBSTEPS      # 12 -> 0.1 s of sim per trajectory frame
STALE_TARGET_STEPS = GYM_SUBSTEPS                         # the first simulate() still holds traj[k-1]

GRAVITY = (0.0, 0.0, -9.8)                                # Isaac Gym value (Isaac Lab defaults to -9.81)
SOLVER_TYPE = 1                                           # TGS
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 1
CONTACT_OFFSET = 0.001
REST_OFFSET = 0.0
FRICTION_OFFSET_THRESHOLD = 0.001
FRICTION_CORRELATION_DISTANCE = 0.0005
BOUNCE_THRESHOLD_VELOCITY = 0.2                           # Isaac Gym default (Isaac Lab: 0.5)
MAX_DEPENETRATION_VELOCITY = 100.0                        # Isaac Gym default

# Isaac Gym gives every shape of a *loaded* asset friction 1.0 unless the script overrides it, and
# the replay only overrides the table (0.45) and the scene objects (URDF mu1).
DEFAULT_SHAPE_FRICTION = 1.0
ROBOT_FRICTION = 1.0
GROUND_FRICTION = 1.0
TABLE_FRICTION = 0.45
TABLE_RESTITUTION = 0.0

ARM_STIFFNESS, ARM_DAMPING = 400.0, 40.0
HAND_STIFFNESS, HAND_DAMPING, HAND_EFFORT = 350.0, 12.0, 50.0
JOINT_ARMATURE = 0.0          # Isaac Gym's armature lives in the link inertias (see ingest_data.py)

# Isaac Gym: object_asset_options.vhacd_params.resolution = 100000
VHACD_RESOLUTION = 100000
VHACD_MAX_HULLS = 64
VHACD_HULL_VERTICES = 64
# PhysX 5's decomposer replaced VHACD and, in this Isaac Sim build, its cooked result reacts to
# errorPercentage only - voxelResolution, shrinkWrap and even maxConvexHulls changes produced
# bit-identical replays. At the 10% default the hulls of a hand-sized scanned object are ~1.5 mm
# fatter than Isaac Gym's VHACD hulls (objects rest 1.5 mm above the table and marginal grasps close
# on a bloated surface and slip). 1% reproduces Isaac Gym's effective geometry: resting height
# matches to 0.1 mm and the run_2026-05-15_17-55-22 grasp lifts 139 mm vs gym's 134 mm.
VHACD_SHRINK_WRAP = True
VHACD_ERROR_PERCENTAGE = 1.0                              # PhysX default: 10 (%)
# SDF triangle-mesh colliders (run_meta object "collision": "sdf"): PhysX samples a signed distance
# field over the mesh's AABB, so contacts follow the real surface - threads, teeth, holes. The
# resolution is the number of samples along the longest AABB edge (PhysX's nut-and-bolt demo uses
# 256); a 126 mm screw at 256 -> 0.5 mm cells, comfortably below the 3.7 mm thread depth.
SDF_RESOLUTION = 256
SDF_SUBGRID_RESOLUTION = 6                                # sparse SDF (PhysX default); 0 = dense
# Interlocking SDF contacts are stiff and shallow (3.7 mm thread depth): with the grasping defaults
# (8 position iterations, 1 mm contact offset, 100 m/s depenetration) a 200 N push - which a fixed-
# base PD arm produces effortlessly against a welded screw - drives the nut 14 mm through the
# thread in 0.5 s and the depenetration response then flings it away (scripts/test_thread.py
# --push-force 200). These apply to SDF objects only, so every convex-decomposition scene keeps its
# Isaac-Gym-matched, bit-reproducible settings. (PhysX's Factory nut-and-bolt task runs 192
# position iterations; the pair takes the max of the two bodies' counts.)
SDF_SOLVER_POSITION_ITERATIONS = 64
SDF_SOLVER_VELOCITY_ITERATIONS = 2
SDF_CONTACT_OFFSET = 0.005                                # contacts form before the threads touch
SDF_MAX_DEPENETRATION_VELOCITY = 2.0                      # m/s: recover from penetration gently

ENV_PRIM = "/World/envs/env_0"


@dataclass
class ReplayConfig:
    """Everything the replay loop can be told to do."""

    device: str = "cuda:0"
    physics_dt: float = PHYSICS_DT
    steps_per_frame: int = STEPS_PER_FRAME
    stale_target_steps: int = STALE_TARGET_STEPS
    kinematic: bool = False                            # Isaac Gym's --visualize (teleport the joints)
    settle_steps: int = 0                              # physics steps before the trajectory starts
    render: bool = True
    render_warmup_steps: int = 5                       # let the RTX textures load before frame 0
    save_images: bool = True
    save_depth: bool = False
    save_ratio: int = 1
    video: bool = True
    video_fps: int = 20
    hand_cam: bool = True                              # close-up camera tracking the fingertips
    contact_sensors: bool = True
    object_voxel_resolution: int = VHACD_RESOLUTION    # convex-decomposition voxel count per object
    object_decomposition_error: float = VHACD_ERROR_PERCENTAGE  # cooker error tolerance (%)
    force_vis: bool = True                             # draw fingertip force arrows in the scene
    force_vis_scale: float = 0.005                     # arrow length per newton (m/N)
    force_vis_max_len: float = 0.35                    # cap on the arrow shaft length (m)
    flow_points: int = 256                             # surface samples used for the flow metric
    joint_armature: float = JOINT_ARMATURE
    gui: bool = False                                  # a Kit window is open: animate it live
    gui_render_interval: int = 2                       # render every Nth physics step in the window
    realtime: bool = False                             # pace the replay to wall-clock time
    cam_far: float = 20.0                              # Isaac Gym used 1.0, which clips the scene
    cam_use_real_intrinsics: bool = False
    max_frames: int | None = None
    output_dir: Path | None = None
    seed: int = 0


@dataclass
class SceneHandles:
    robot: Articulation
    objects: dict[str, RigidObject]
    camera: Camera | None
    contacts: list[ContactSensor]              # one sensor per fingertip (empty when disabled)
    contact_object_keys: list[str]             # object keys in force_matrix_w filter order
    object_names: list[str]
    manipulated_key: str | None
    static_keys: list[str] = field(default_factory=list)
    force_markers: VisualizationMarkers | None = None  # per-fingertip force arrows (rendered runs)
    contact_point_markers: VisualizationMarkers | None = None  # spheres at individual contact points
    hand_cam: Camera | None = None                     # close-up camera re-aimed at the grip each frame


# --------------------------------------------------------------------------------------
# Simulation configuration
# --------------------------------------------------------------------------------------
def hand_effort_limit(spec: RunSpec | None) -> float:
    """LEAP finger effort limit for this run: run_meta ``hand_effort_limit`` or the replay's 50 N m."""
    v = getattr(spec, "hand_effort_limit", None) if spec is not None else None
    return float(v) if v is not None else HAND_EFFORT


def scene_has_sdf(spec: RunSpec | None) -> bool:
    return spec is not None and any(getattr(o, "collision", "") == "sdf" for o in spec.objects)


def make_simulation_cfg(cfg: ReplayConfig, spec: RunSpec | None = None) -> SimulationCfg:
    """PhysX scene settings. Pass the run *spec* so a scene with SDF objects may raise the
    iteration ceiling for those bodies; every other body still requests (and gets) the pinned
    Isaac Gym counts, so runs without SDF objects are unaffected."""
    max_pos_iters = SOLVER_POSITION_ITERATIONS
    max_vel_iters = SOLVER_VELOCITY_ITERATIONS
    if scene_has_sdf(spec):
        max_pos_iters = max(max_pos_iters, SDF_SOLVER_POSITION_ITERATIONS)
        max_vel_iters = max(max_vel_iters, SDF_SOLVER_VELOCITY_ITERATIONS)
    return SimulationCfg(
        dt=cfg.physics_dt,
        render_interval=1,
        gravity=GRAVITY,
        device=cfg.device,
        physx=sim_utils.PhysxCfg(
            solver_type=SOLVER_TYPE,
            # pin the iteration counts so the values baked into the USD cannot leak through
            min_position_iteration_count=SOLVER_POSITION_ITERATIONS,
            max_position_iteration_count=max_pos_iters,
            min_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            max_velocity_iteration_count=max_vel_iters,
            bounce_threshold_velocity=BOUNCE_THRESHOLD_VELOCITY,
            friction_offset_threshold=FRICTION_OFFSET_THRESHOLD,
            friction_correlation_distance=FRICTION_CORRELATION_DISTANCE,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**18,
            gpu_collision_stack_size=2**27,
        ),
        # Isaac Gym's default shape friction for a *loaded* asset is 1.0; anything not explicitly
        # overridden below (i.e. the robot) must see the same number.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=DEFAULT_SHAPE_FRICTION,
            dynamic_friction=DEFAULT_SHAPE_FRICTION,
            restitution=0.0,
        ),
    )


# --------------------------------------------------------------------------------------
# Scene construction
# --------------------------------------------------------------------------------------
def _spawn_static_box(
    prim_path: str,
    size: tuple[float, float, float],
    position: np.ndarray,
    friction: float,
    restitution: float = 0.0,
    color: tuple[float, float, float] = (0.55, 0.42, 0.30),
) -> None:
    """A collider-only box (no rigid body) == Isaac Gym's ``fix_base_link`` box actor."""
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=friction, dynamic_friction=friction, restitution=restitution
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
    )
    cfg.func(prim_path, cfg, translation=tuple(float(v) for v in position))


def _apply_scaled_mass_properties(prim_path: str, obj, scale: float) -> dict | None:
    """Reproduce Isaac Gym's ``set_actor_scale``: mass x s^3, com x s, inertia x s^5.

    ``UsdFileCfg.scale`` only scales geometry, so without this a 0.9-scaled object keeps the full
    mass of the unscaled mesh (a 37% error) and its full inertia (a 69% error).
    """
    if scale == 1.0:
        return None

    import omni.usd
    from pxr import Gf, UsdPhysics

    from .scene_spec import load_urdf_inertial

    inertial = load_urdf_inertial(obj.urdf_path)
    if inertial is None:
        print(f"[replay][WARN] no <inertial> in {obj.urdf_path}; mass not rescaled for scale {scale}")
        return None

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return None

    # Measured against PhysX (root_physx_view.get_masses/get_coms/get_inertias): the authored mass
    # and inertia are used verbatim, but `physics:centerOfMass` is a point in the body's local space
    # and therefore picks up the prim's xformOp:scale a second time. So author mass and inertia
    # pre-scaled, and the centre of mass *unscaled* - the effective values then match Isaac Gym's
    # set_actor_scale (mass x s^3, com x s, inertia x s^5).
    scaled = {
        "mass": inertial["mass"] * scale**3,
        "com": list(inertial["com"]),
        "effective_com": [c * scale for c in inertial["com"]],
        "inertia": {k: v * scale**5 for k, v in inertial["inertia"].items()},
    }

    applied = 0
    for prim in [root] + list(root.GetChildren()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(float(scaled["mass"]))
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*[float(c) for c in scaled["com"]]))
        mass_api.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(
                float(scaled["inertia"]["ixx"]),
                float(scaled["inertia"]["iyy"]),
                float(scaled["inertia"]["izz"]),
            )
        )
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        applied += 1

    if applied == 0:
        print(f"[replay][WARN] no rigid body found under {prim_path}; mass not rescaled")
        return None
    return scaled



# fingertip force arrows: a unit-height shaft (Z-scaled to the force magnitude) plus a fixed head
FORCE_ARROW_SHAFT_RADIUS = 0.0035
FORCE_ARROW_HEAD_RADIUS = 0.009
FORCE_ARROW_HEAD_LENGTH = 0.025
FORCE_ARROW_MIN_FORCE = 0.05               # N below which the arrow is hidden
MAX_CONTACT_POINTS = 64                    # per-sensor cap == max_contact_data_count_per_prim
# 64 (was 32, raised 2026-08-31): a READBACK buffer size, not physics - it bounds how many
# of PhysX's already-resolved contact points the sensor can report. At 32, firm grasps from
# the force controller's adaptive feedforward overflowed it (PhysX logs 'Incomplete contact
# data', then Isaac Lab's ContactSensor.update trips a CUDA assert and kills the run).
# Verified after the change: --exact-replay is still bit-exact on every state key.
CONTACT_POINT_MIN_FORCE = 0.01             # N below which an individual contact point is not drawn
CONTACT_POINT_RADIUS = 0.004               # base marker radius (m); grows mildly with force


def _make_force_arrow_markers() -> VisualizationMarkers:
    """One arrow (cylinder shaft + cone head) prototype pair per fingertip, in its colour.

    Prototype order is ``[shaft_0, head_0, shaft_1, head_1, ...]`` following ``FINGERTIP_LINKS``.
    These are plain ``UsdGeom`` prims, so they show up in the offscreen demo camera (and the GUI
    viewport), unlike debug-draw overlays.
    """
    markers: dict[str, sim_utils.SpawnerCfg] = {}
    for link in FINGERTIP_LINKS:
        color = FINGERTIP_COLORS.get(link, (1.0, 1.0, 1.0))
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4)
        markers[f"{link}_shaft"] = sim_utils.CylinderCfg(
            radius=FORCE_ARROW_SHAFT_RADIUS, height=1.0, visual_material=material
        )
        markers[f"{link}_head"] = sim_utils.ConeCfg(
            radius=FORCE_ARROW_HEAD_RADIUS, height=FORCE_ARROW_HEAD_LENGTH, visual_material=material
        )
    return VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/FingertipForces", markers=markers)
    )


def _make_contact_point_markers() -> VisualizationMarkers:
    """One emissive sphere prototype per fingertip for its individual contact points.

    The spheres sit exactly at PhysX's reported contact positions, half-buried in the
    finger/object interface, so they are emissive to stay visible there.
    """
    markers: dict[str, sim_utils.SpawnerCfg] = {}
    for link in FINGERTIP_LINKS:
        color = FINGERTIP_COLORS.get(link, (1.0, 1.0, 1.0))
        glow = tuple(min(2.0, c * 2.5) for c in color)   # over-unity emissive so they read on any surface
        markers[f"{link}_point"] = sim_utils.SphereCfg(
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, emissive_color=glow, roughness=0.3
            ),
        )
    return VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/FingertipContactPoints", markers=markers)
    )


def _update_contact_point_markers(
    markers: VisualizationMarkers,
    points: np.ndarray | None,           # [tips, objects, MAX_CONTACT_POINTS, 3], NaN-padded
    point_forces: np.ndarray | None,     # [tips, objects, MAX_CONTACT_POINTS] normal force (N)
) -> None:
    """A sphere at every individual PhysX contact point, radius scaled mildly by its force."""
    if points is None or points.size == 0:
        return
    n_tips, n_objects, max_points = points.shape[:3]
    capacity = n_tips * n_objects * max_points
    translations = np.tile(np.array([0.0, 0.0, -10.0]), (capacity, 1))
    scales = np.full((capacity, 3), 1e-4)
    marker_indices = np.repeat(np.arange(n_tips), n_objects * max_points)
    flat_points = points.reshape(capacity, 3)
    flat_forces = point_forces.reshape(capacity)
    visible = np.isfinite(flat_points).all(axis=1) & (np.nan_to_num(flat_forces) >= CONTACT_POINT_MIN_FORCE)
    if visible.any():
        translations[visible] = flat_points[visible]
        radius = CONTACT_POINT_RADIUS * (
            1.0 + np.sqrt(np.clip(np.abs(flat_forces[visible]), 0.0, 50.0) / 50.0)
        )
        scales[visible] = radius[:, None]
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (capacity, 1))
    markers.visualize(
        translations=translations, orientations=orientations, scales=scales, marker_indices=marker_indices
    )


def _quat_z_to(direction: np.ndarray) -> np.ndarray:
    """wxyz quaternion rotating +Z onto the (unit) *direction*."""
    c = float(direction[2])                          # dot([0,0,1], d)
    if c > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])       # 180 deg about X
    axis = np.array([-direction[1], direction[0], 0.0])  # cross([0,0,1], d)
    axis /= np.linalg.norm(axis)
    half = 0.5 * math.acos(max(-1.0, min(1.0, c)))
    return np.concatenate([[math.cos(half)], axis * math.sin(half)])


def _update_force_arrows(
    markers: VisualizationMarkers,
    tip_positions: np.ndarray,           # [num_fingertips, 3] world (fallback anchor only)
    net_forces: np.ndarray,              # [num_fingertips, 3] net contact force, world
    pair_forces: np.ndarray | None,      # [num_fingertips, num_objects, 3] fingertip<->object force
    contact_points: np.ndarray | None,   # [num_fingertips, num_objects, 3], NaN when not touching
    scale: float,
    max_len: float,
) -> None:
    """Draw one arrow per fingertip<->object contact, anchored at the measured contact point.

    The contact sensors report, per fingertip and per scene object, the total contact force and the
    centre of the actual contact patch on the fingertip surface (``contact_pos_w``), so the arrows
    sit where the finger really touches - not at the link's frame origin, which is back at the
    joint. When a fingertip carries force with no reported point (contact with the table or another
    robot link, which are outside the pair filters), the residual is drawn at the fingertip body
    origin as a fallback.
    """
    n_tips = tip_positions.shape[0]
    n_objects = pair_forces.shape[1] if (pair_forces is not None and pair_forces.size) else 0
    slots = n_tips * (n_objects + 1)                    # per-object arrows + one residual per tip
    translations = np.tile(np.array([0.0, 0.0, -10.0]), (2 * slots, 1))  # hidden: underground
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2 * slots, 1))
    scales = np.full((2 * slots, 3), 1e-4)
    marker_indices = np.zeros(2 * slots, dtype=np.int64)

    def draw(slot: int, tip: int, origin: np.ndarray, force: np.ndarray) -> None:
        magnitude = float(np.linalg.norm(force))
        if magnitude < FORCE_ARROW_MIN_FORCE or not np.isfinite(origin).all():
            return
        direction = force / magnitude
        length = min(magnitude * scale, max_len)
        quat = _quat_z_to(direction)
        translations[2 * slot] = origin + direction * (length / 2)
        orientations[2 * slot] = quat
        scales[2 * slot] = (1.0, 1.0, length)
        translations[2 * slot + 1] = origin + direction * (length + FORCE_ARROW_HEAD_LENGTH / 2)
        orientations[2 * slot + 1] = quat
        scales[2 * slot + 1] = (1.0, 1.0, 1.0)

    slot = 0
    for tip in range(n_tips):
        accounted = np.zeros(3)
        for obj in range(n_objects):
            marker_indices[2 * slot] = 2 * tip
            marker_indices[2 * slot + 1] = 2 * tip + 1
            force = pair_forces[tip, obj]
            accounted += force
            point = contact_points[tip, obj] if contact_points is not None else tip_positions[tip]
            if not np.isfinite(point).all():            # force without a reported patch centre
                point = tip_positions[tip]
            draw(slot, tip, point, force)
            slot += 1
        # residual = contact with anything outside the filtered objects (table, robot links)
        marker_indices[2 * slot] = 2 * tip
        marker_indices[2 * slot + 1] = 2 * tip + 1
        draw(slot, tip, tip_positions[tip], net_forces[tip] - accounted)
        slot += 1

    markers.visualize(
        translations=translations,
        orientations=orientations,
        scales=scales,
        marker_indices=marker_indices,
    )


def _find_rigid_body_prim_paths(prim_path: str) -> list[str]:
    """Paths of the prims under *prim_path* that carry ``UsdPhysics.RigidBodyAPI``.

    The fingertip contact sensors filter their force reports against these prims (PhysX pair
    filters match rigid-body prims, not asset roots, and the URDF importer is free to put the
    body on the root or on a child link).
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return []
    return [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _disable_instancing(prim_path: str, only_collisions: bool = True) -> int:
    """Flatten instanced sub-trees under *prim_path* so per-prim edits can reach the colliders.

    Isaac Sim's URDF importer marks each link's ``visuals`` and ``collisions`` scope as
    *instanceable*. Isaac Lab's ``apply_nested`` helpers (``bind_physics_material``,
    ``modify_mesh_collision_properties``, ...) skip instanced prims, so a physics material bound to
    the asset root silently never reaches the collision meshes and the collider keeps the *scene
    default* material. Turning instancing off first makes those edits land where they are meant to.

    Un-instancing costs memory (each flattened scope stops sharing its prototype's mesh data), so by
    default only the ``collisions`` scopes are flattened - the ``visuals`` ones are never edited.
    On a 5-object scene that roughly halves the extra footprint.

    Returns the number of prims that were un-instanced.
    """
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return 0

    count = 0
    # nested instances only become visible once their parent is flattened, hence the loop
    for _ in range(8):
        instances = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.IsInstance() and (not only_collisions or "collisions" in str(prim.GetPath()))
        ]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
            count += 1
    return count


def _prepare_colliders(
    prim_path: str, material_path: str, material_cfg, contact_offset: float = CONTACT_OFFSET
) -> dict:
    """Flatten instancing, then apply the contact/rest offsets and bind the physics material.

    The spawner's ``collision_props`` and any ``bind_physics_material`` call made before this point
    are no-ops on an importer-produced USD (the ``collisions`` scopes are instanced), so both are
    re-applied here and the result is verified rather than assumed.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics

    _disable_instancing(prim_path)

    sim_utils.schemas.modify_collision_properties(
        prim_path,
        sim_utils.schemas.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=contact_offset, rest_offset=REST_OFFSET
        ),
    )

    material_cfg.func(material_path, material_cfg)
    sim_utils.bind_physics_material(prim_path, material_path)

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    collider_paths: list[str] = []
    colliders = bound = offsets = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        colliders += 1
        collider_paths.append(str(prim.GetPath()))
        rel = prim.GetRelationship("material:binding:physics")
        if rel and rel.GetTargets():
            bound += 1
        attr = prim.GetAttribute("physxCollision:contactOffset")
        if attr and attr.Get() is not None and abs(float(attr.Get()) - contact_offset) < 1e-9:
            offsets += 1

    if colliders == 0 or bound < colliders or offsets < colliders:
        print(
            f"[replay][WARN] {prim_path}: {colliders} collider(s), {bound} with the physics material, "
            f"{offsets} with contact_offset={contact_offset}"
        )
    return {
        "colliders": colliders,
        "material_bound": bound,
        "contact_offset_set": offsets,
        "contact_offset": contact_offset,
        "collider_paths": collider_paths,
    }


def object_contact_offset(obj) -> float:
    return SDF_CONTACT_OFFSET if getattr(obj, "collision", "") == "sdf" else CONTACT_OFFSET


def object_rigid_props(obj, kinematic: bool | None = None) -> sim_utils.RigidBodyPropertiesCfg:
    """Rigid-body settings for a scene object; *kinematic* defaults to ``obj.is_static``.

    A "static" scene object is welded in place: Isaac Gym used fix_base_link=True plus
    disable_gravity=True, which a kinematic rigid body reproduces exactly. SDF objects get the
    stiffer-contact settings documented at ``SDF_SOLVER_POSITION_ITERATIONS``.
    """
    if kinematic is None:
        kinematic = bool(obj.is_static)
    sdf = getattr(obj, "collision", "") == "sdf"
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic,
        disable_gravity=kinematic,
        retain_accelerations=False,
        max_depenetration_velocity=SDF_MAX_DEPENETRATION_VELOCITY if sdf else MAX_DEPENETRATION_VELOCITY,
        solver_position_iteration_count=SDF_SOLVER_POSITION_ITERATIONS if sdf else SOLVER_POSITION_ITERATIONS,
        solver_velocity_iteration_count=SDF_SOLVER_VELOCITY_ITERATIONS if sdf else SOLVER_VELOCITY_ITERATIONS,
    )


def object_collision_props(obj) -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=object_contact_offset(obj), rest_offset=REST_OFFSET)


def prepare_object_colliders(obj, prim_path: str, cfg: ReplayConfig) -> dict:
    """Bind the object's URDF surface material and configure its collider approximation.

    Shared by :func:`build_scene` and the live-teleop viewer's floating-hand scene, so every
    path that spawns a scene object gets identical contact behaviour. Isaac Gym applied the URDF's
    gazebo <mu1>/<mu2>/restitution to every shape of the actor.
    """
    collider_report = _prepare_colliders(
        prim_path,
        f"/World/Materials/{obj.name}",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=obj.friction,
            dynamic_friction=obj.friction,  # Isaac Gym has a single friction coefficient (mu1)
            restitution=obj.restitution,
        ),
        contact_offset=object_contact_offset(obj),
    )

    # Collider approximation settings. Isaac Gym's VHACD ran with resolution = 100000, but PhysX
    # 5's decomposition quantises the hulls to the voxel grid: at 100000 voxels a hand-sized object
    # gets ~1.4 mm of outward hull bloat (measurable as the object resting ~1.5 mm above the table,
    # unlike Isaac Gym). cfg.object_voxel_resolution raises the voxel count to shrink that error -
    # matching Isaac Gym's *effective* collision geometry matters more than matching its
    # resolution number.
    # The importer authors CollisionAPI + the approximation token on the link's `World` *Xform*
    # while the geometry lives on a child Mesh prim with no collision APIs of its own. PhysX
    # ignores PhysxConvexDecompositionCollisionAPI attributes authored on the Xform, so the
    # settings must land on the Mesh gprims themselves - otherwise the cooker silently uses its
    # defaults regardless of what is configured here.
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if obj.collision == "sdf":
        mesh_cfg = sim_utils.schemas.SDFMeshPropertiesCfg(
            sdf_resolution=SDF_RESOLUTION,
            sdf_subgrid_resolution=SDF_SUBGRID_RESOLUTION,
        )
        print(
            f"[replay] {obj.name}: SDF triangle-mesh collider (resolution {SDF_RESOLUTION}, "
            f"{SDF_SOLVER_POSITION_ITERATIONS} position iterations, contact offset {SDF_CONTACT_OFFSET})"
        )
    elif obj.collision == "convex_decomposition":
        mesh_cfg = sim_utils.schemas.ConvexDecompositionPropertiesCfg(
            voxel_resolution=cfg.object_voxel_resolution,
            max_convex_hulls=VHACD_MAX_HULLS,
            hull_vertex_limit=VHACD_HULL_VERTICES,
            shrink_wrap=VHACD_SHRINK_WRAP,
            error_percentage=cfg.object_decomposition_error,
        )
    else:
        raise ValueError(
            f"{obj.name}: unknown collision approximation {obj.collision!r} "
            "(run_meta object 'collision' must be 'convex_decomposition' or 'sdf')"
        )
    for collider_path in collider_report["collider_paths"]:
        targets = [collider_path]
        root = stage.GetPrimAtPath(collider_path)
        if root.IsValid():
            targets += [str(p.GetPath()) for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
        for target in targets:
            try:
                if obj.collision == "sdf":
                    # Isaac Lab's helper writes the sdf* attributes but never *applies* the
                    # PhysxSDFMeshCollisionAPI schema; omni.physx only honours the "sdf"
                    # approximation token on a prim that carries the schema and otherwise parses
                    # the shape as a plain triangle mesh ("approximation None ... falling back to
                    # convexHull" for a dynamic body - the nut then has no hole at all).
                    from pxr import PhysxSchema

                    target_prim = stage.GetPrimAtPath(target)
                    if target_prim.IsValid() and not target_prim.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                        PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(target_prim)
                sim_utils.schemas.modify_mesh_collision_properties(target, mesh_cfg)
            except Exception as exc:  # pragma: no cover - depends on how the USD was cooked
                print(f"[replay][WARN] {obj.collision} collider params not set on {target}: {exc}")

    print(
        f"[replay] {obj.name}: friction {obj.friction} restitution {obj.restitution}, "
        f"{collider_report['colliders']} collider(s) "
        f"({collider_report['material_bound']} with material, "
        f"{collider_report['contact_offset_set']} with contact_offset {collider_report['contact_offset']})"
    )
    return collider_report


def build_scene(
    spec: RunSpec,
    cfg: ReplayConfig,
    robot_usd: Path,
    object_usds: dict[str, Path],
) -> SceneHandles:
    """Spawn ground, table, robot, scene objects and the demo camera; return live handles."""
    # ---- lights (Kit renders a black image without them) ----
    dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.9, 0.95))
    dome.func("/World/DomeLight", dome)
    distant = sim_utils.DistantLightCfg(intensity=1800.0, color=(1.0, 1.0, 1.0), angle=1.0)
    distant.func("/World/DistantLight", distant, orientation=(0.9238795, 0.0, 0.3826834, 0.0))

    # ---- ground plane (Isaac Gym: gym.add_ground, +Z normal, friction 1.0) ----
    _spawn_static_box(
        "/World/GroundPlane",
        size=(40.0, 40.0, 0.2),
        position=np.array([0.0, 0.0, -0.1]),
        friction=GROUND_FRICTION,
        color=(0.25, 0.25, 0.27),
    )

    # ---- table ----
    _spawn_static_box(
        f"{ENV_PRIM}/Table",
        size=tuple(float(v) for v in spec.table_size),
        position=spec.table_position,
        friction=TABLE_FRICTION,
        restitution=TABLE_RESTITUTION,
    )

    # ---- robot ----
    traj_to_urdf = load_joint_name_map(spec.root.parents[1])
    init_joint_pos = {
        traj_to_urdf.get(name, name): float(value)
        for name, value in zip(
            spec.trajectory.joint_names, spec.trajectory.positions[0].tolist(), strict=True
        )
    }
    robot_cfg = ArticulationCfg(
        prim_path=f"{ENV_PRIM}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            activate_contact_sensors=cfg.contact_sensors,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,                     # Isaac Gym: robot_asset_options.disable_gravity
                retain_accelerations=False,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,            # Isaac Gym: self-collision off
                fix_root_link=True,                       # Isaac Gym: fix_base_link
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=CONTACT_OFFSET, rest_offset=REST_OFFSET
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in spec.robot_position),
            rot=tuple(float(v) for v in quat_xyzw_to_wxyz(spec.robot_quat_xyzw)),
            joint_pos=init_joint_pos,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[r"joint_[1-7]"],
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
                armature=cfg.joint_armature,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[r"leap_j\d+"],
                stiffness=HAND_STIFFNESS,
                damping=HAND_DAMPING,
                armature=cfg.joint_armature,
                # Isaac Gym: dof_props["effort"][7:] = 50; authored scenes may set the URDF's 0.95
                effort_limit_sim=hand_effort_limit(spec),
            ),
        },
    )
    robot = Articulation(robot_cfg)

    robot_colliders = _prepare_colliders(
        f"{ENV_PRIM}/Robot",
        "/World/Materials/robot",
        sim_utils.RigidBodyMaterialCfg(
            static_friction=ROBOT_FRICTION, dynamic_friction=ROBOT_FRICTION, restitution=0.0
        ),
    )
    print(
        f"[replay] robot: friction {ROBOT_FRICTION}, {robot_colliders['colliders']} colliders "
        f"({robot_colliders['material_bound']} with material, "
        f"{robot_colliders['contact_offset_set']} with contact_offset {CONTACT_OFFSET})"
    )

    # ---- scene objects ----
    objects: dict[str, RigidObject] = {}
    object_names: list[str] = []
    static_keys: list[str] = []
    contact_filter_paths: list[str] = []       # rigid-body prims the fingertip sensors filter against
    contact_object_keys: list[str] = []
    scale = float(spec.object_mesh_scale)

    for obj in spec.objects:
        usd_path = object_usds[obj.key]
        prim_path = f"{ENV_PRIM}/{obj.name}"
        obj_cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                scale=(scale, scale, scale) if scale != 1.0 else None,
                activate_contact_sensors=cfg.contact_sensors,
                rigid_props=object_rigid_props(obj),
                collision_props=object_collision_props(obj),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(v) for v in obj.position),
                rot=tuple(float(v) for v in quat_xyzw_to_wxyz(obj.quat_xyzw)),
            ),
        )
        objects[obj.key] = RigidObject(obj_cfg)
        object_names.append(obj.name)
        if obj.is_static:
            static_keys.append(obj.key)

        prepare_object_colliders(obj, prim_path, cfg)

        if cfg.contact_sensors:
            rb_paths = _find_rigid_body_prim_paths(prim_path)
            if len(rb_paths) == 1:
                contact_filter_paths.append(rb_paths[0])
                contact_object_keys.append(obj.key)
            else:
                print(
                    f"[replay][WARN] {obj.name}: expected exactly 1 rigid-body prim, found "
                    f"{len(rb_paths)}; fingertip forces will not be attributed to this object"
                )

        scaled = _apply_scaled_mass_properties(prim_path, obj, scale)
        if scaled is not None:
            print(
                f"[replay] {obj.name}: scale {scale} -> mass {scaled['mass']:.8f} kg, "
                f"com {[round(c, 6) for c in scaled['effective_com']]} (effective), "
                f"Ixx {scaled['inertia']['ixx']:.6e}"
            )

    # ---- demo camera ----
    camera = None
    if cfg.render:
        cam = spec.camera
        fx, fy, cx, cy = (
            cam.real_intrinsics if (cfg.cam_use_real_intrinsics and cam.real_intrinsics) else (cam.fx, cam.fy, cam.cx, cam.cy)
        )
        data_types = ["rgb"] + (["distance_to_image_plane"] if cfg.save_depth else [])
        camera_cfg = CameraCfg(
            prim_path=f"{ENV_PRIM}/DemoCamera",
            update_period=0.0,
            height=cam.height,
            width=cam.width,
            data_types=data_types,
            depth_clipping_behavior="none",
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                width=cam.width,
                height=cam.height,
                clipping_range=(0.01, cfg.cam_far),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in cam.position),
                rot=tuple(float(v) for v in quat_xyzw_to_wxyz(cam.quat_xyzw)),
                convention="ros",              # +Z forward / +Y down, same as the measured pose
            ),
        )
        camera = Camera(camera_cfg)

    # ---- close-up hand camera (pose is re-aimed at the fingertips every frame) ----
    hand_cam = None
    if cfg.render and cfg.hand_cam:
        hand_cam_cfg = CameraCfg(
            prim_path=f"{ENV_PRIM}/HandCam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 20.0)),
        )
        hand_cam = Camera(hand_cam_cfg)
        print("[replay] hand camera: close-up view tracking the fingertips -> handcam.mp4")

    # ---- fingertip contact sensors ----
    # One sensor per fingertip: Isaac Lab only supports filtered (per-object) force reporting when
    # the sensor matches a single body. The per-step history buffer captures the force at every
    # physics step of a frame, not just the last one: contact forces are impulsive and a 10 Hz
    # sample of the final step misses the peaks.
    contacts: list[ContactSensor] = []
    if cfg.contact_sensors:
        for link in FINGERTIP_LINKS:
            contact_cfg = ContactSensorCfg(
                prim_path=f"{ENV_PRIM}/Robot/{link}",
                update_period=0.0,
                history_length=max(1, cfg.steps_per_frame),
                track_air_time=False,
                filter_prim_paths_expr=list(contact_filter_paths),
                # centre of the actual contact patch per fingertip<->object pair (contact_pos_w),
                # so forces can be recorded and drawn at the real touch location instead of the
                # link origin. Needs headroom for multi-point patches on the decomposed hulls.
                track_contact_points=bool(contact_filter_paths),
                max_contact_data_count_per_prim=MAX_CONTACT_POINTS,
            )
            contacts.append(ContactSensor(contact_cfg))
        print(
            f"[replay] fingertip contact sensors on {FINGERTIP_LINKS}: net force every physics "
            f"step, per-object forces + contact-patch positions against "
            f"{len(contact_filter_paths)} object(s)"
        )

    # ---- fingertip force arrows (visible in the demo camera and the GUI viewport) ----
    force_markers = None
    if contacts and cfg.force_vis and (cfg.render or cfg.gui):
        force_markers = _make_force_arrow_markers()
        # the instancer starts with every prototype at the origin; park them out of sight until
        # the first frame's forces are known
        _update_force_arrows(
            force_markers,
            np.zeros((len(FINGERTIP_LINKS), 3)),
            np.zeros((len(FINGERTIP_LINKS), 3)),
            np.zeros((len(FINGERTIP_LINKS), len(contact_object_keys), 3)),
            np.full((len(FINGERTIP_LINKS), len(contact_object_keys), 3), np.nan),
            cfg.force_vis_scale,
            cfg.force_vis_max_len,
        )
        print(
            f"[replay] fingertip force arrows: 1 N = {cfg.force_vis_scale * 100:.2f} cm "
            f"(shaft capped at {cfg.force_vis_max_len:.2f} m, hidden below {FORCE_ARROW_MIN_FORCE} N)"
        )

    contact_point_markers = None
    if contacts and cfg.force_vis and (cfg.render or cfg.gui) and contact_object_keys:
        contact_point_markers = _make_contact_point_markers()
        _update_contact_point_markers(
            contact_point_markers,
            np.full((len(FINGERTIP_LINKS), len(contact_object_keys), MAX_CONTACT_POINTS, 3), np.nan),
            np.full((len(FINGERTIP_LINKS), len(contact_object_keys), MAX_CONTACT_POINTS), np.nan),
        )
        print(
            f"[replay] individual contact points: emissive spheres at every PhysX contact "
            f"(up to {MAX_CONTACT_POINTS}/fingertip, hidden below {CONTACT_POINT_MIN_FORCE} N)"
        )

    return SceneHandles(
        robot=robot,
        objects=objects,
        camera=camera,
        contacts=contacts,
        contact_object_keys=contact_object_keys,
        object_names=object_names,
        manipulated_key=spec.manipulated.key if spec.manipulated else None,
        static_keys=static_keys,
        force_markers=force_markers,
        contact_point_markers=contact_point_markers,
        hand_cam=hand_cam,
    )


# --------------------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------------------
def check_scene(scene: SceneHandles, spec: RunSpec) -> dict:
    """Assert the things that silently ruin a replay, and return what was checked."""
    robot = scene.robot
    report: dict = {}

    report["num_joints"] = int(robot.num_joints)
    report["num_bodies"] = int(robot.num_bodies)
    if robot.num_joints != 23:
        raise RuntimeError(f"expected 23 actuated joints, got {robot.num_joints}: {robot.joint_names}")
    if robot.num_bodies != 31:
        print(
            f"[replay][WARN] expected 31 rigid bodies (Isaac Gym keeps every URDF link), got "
            f"{robot.num_bodies}. Was the robot converted with merge_fixed_joints=True?"
        )

    limits = robot.data.joint_pos_limits[0]
    continuous = ["joint_1", "joint_3", "joint_5", "joint_7"]
    unlimited = {}
    for name in continuous:
        ids, _ = robot.find_joints(name, preserve_order=True)
        lo, hi = (float(v) for v in limits[ids[0]])
        unlimited[name] = [lo, hi]
        if abs(lo) < 1e30 or abs(hi) < 1e30:
            print(
                f"[replay][WARN] continuous joint {name} is limited in the USD ({lo:.4f}, {hi:.4f}). "
                "Re-run scripts/ingest_data.py and scripts/convert_assets.py --force: the URDF's "
                "stray <limit lower/upper> on a continuous joint clamps the commanded trajectory."
            )
    report["continuous_joint_limits"] = unlimited

    # the LEAP fingertips must carry Isaac Gym's armature bump in their inertia
    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(f"{ENV_PRIM}/Robot/fingertip")
        if prim.IsValid() and prim.HasAPI(UsdPhysics.MassAPI):
            inertia = UsdPhysics.MassAPI(prim).GetDiagonalInertiaAttr().Get()
            if inertia is not None:
                report["fingertip_diagonal_inertia"] = [float(v) for v in inertia]
                if float(inertia[0]) < 1e-3:
                    print(
                        "[replay][WARN] fingertip inertia looks un-bumped "
                        f"({float(inertia[0]):.3e}); Isaac Gym ran with +0.01 on the diagonal. "
                        "Re-run scripts/ingest_data.py (--armature 0.01) and reconvert."
                    )
    except Exception as exc:  # pragma: no cover
        report["fingertip_inertia_error"] = str(exc)

    # the fingertip contact sensors must have resolved exactly one body each, and their per-object
    # filter must have matched every scene object we asked for
    if scene.contacts:
        sensors = []
        for sensor in scene.contacts:
            entry = {
                "body": sensor.body_names[0] if sensor.num_bodies else None,
                "num_bodies": int(sensor.num_bodies),
                "filter_count": int(sensor.contact_physx_view.filter_count),
            }
            sensors.append(entry)
            if entry["num_bodies"] != 1:
                print(f"[replay][WARN] contact sensor {sensor.cfg.prim_path} matched {entry['num_bodies']} bodies, expected 1")
            if entry["filter_count"] != len(scene.contact_object_keys):
                print(
                    f"[replay][WARN] contact sensor {sensor.cfg.prim_path}: filter matched "
                    f"{entry['filter_count']} prim(s), expected {len(scene.contact_object_keys)}; "
                    "per-object force attribution will be misaligned"
                )
        report["contact_sensors"] = sensors

    return report


# --------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------
def _sample_surface_points(mesh_path: Path, num_points: int, scale: float, seed: int) -> np.ndarray | None:
    """Sample points on the manipulated object's mesh (Isaac Gym did the same for the flow metric)."""
    try:
        import trimesh
    except ImportError:
        print("[replay][WARN] trimesh not available -> skipping the 3D flow-point metric")
        return None
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if scale != 1.0:
            mesh.apply_scale(scale)
        points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
        return np.asarray(points, dtype=np.float64)
    except Exception as exc:
        print(f"[replay][WARN] could not sample {mesh_path}: {exc}")
        return None


def run_replay(spec: RunSpec, cfg: ReplayConfig, scene: SceneHandles, sim: SimulationContext) -> dict:
    """Execute the trajectory and return everything that was recorded."""
    robot = scene.robot
    device = robot.device

    # ---- joint-name mapping: trajectory json -> simulation joint order ----
    traj_to_urdf = load_joint_name_map(spec.root.parents[1])
    urdf_to_sim = match_joint_names(
        [traj_to_urdf.get(n, n) for n in spec.trajectory.joint_names], list(robot.joint_names)
    )
    traj_to_sim = {n: urdf_to_sim[traj_to_urdf.get(n, n)] for n in spec.trajectory.joint_names}
    sim_to_traj = {v: k for k, v in traj_to_sim.items()}
    missing_sim = [n for n in robot.joint_names if n not in sim_to_traj]
    if missing_sim:
        raise KeyError(
            f"The trajectory does not provide a command for simulation joint(s) {missing_sim}. "
            f"Trajectory joints: {spec.trajectory.joint_names}"
        )
    print("[replay] joint mapping (simulation order):")
    print(format_mapping(traj_to_sim, list(robot.joint_names)))

    traj_order = [sim_to_traj[name] for name in robot.joint_names]
    targets_np = spec.trajectory.reorder(traj_order)            # [T, num_joints] in sim joint order
    num_frames = targets_np.shape[0] if cfg.max_frames is None else min(cfg.max_frames, targets_np.shape[0])
    targets = torch.as_tensor(targets_np[:num_frames], dtype=torch.float32, device=device)

    # ---- reset to the trajectory's first pose (Isaac Gym: default_dof_state = traj[0]) ----
    robot.reset()
    joint_pos = targets[0:1].clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    for obj in scene.objects.values():
        obj.reset()
        obj.write_data_to_sim()

    for _ in range(cfg.settle_steps):
        sim.step(render=False)
        robot.update(cfg.physics_dt)
        for obj in scene.objects.values():
            obj.update(cfg.physics_dt)

    # RTX needs a few rendered frames before the textures are resident, otherwise frame 0 is blank
    if cfg.render and scene.camera is not None:
        for _ in range(cfg.render_warmup_steps):
            sim.render()
            scene.camera.update(cfg.physics_dt, force_recompute=True)

    # ---- bodies we track ----
    tracked_links = [PALM_LINK, HAND_MOUNT_LINK, BRACELET_LINK] + FINGERTIP_LINKS
    body_ids: dict[str, int] = {}
    for link in tracked_links:
        ids, _ = robot.find_bodies(link, preserve_order=True)
        if ids:
            body_ids[link] = int(ids[0])
    missing = [l for l in tracked_links if l not in body_ids]
    if missing:
        print(f"[replay][WARN] links not found in the articulation, not recorded: {missing}")

    manipulated = spec.manipulated
    surface_points = None
    if manipulated is not None and cfg.flow_points > 0:
        surface_points = _sample_surface_points(
            manipulated.mesh_path, cfg.flow_points, spec.object_mesh_scale, cfg.seed
        )

    rec: dict[str, list] = {
        "joint_pos": [],
        "joint_target": [],
        "joint_vel": [],
        "joint_pos_steps": [],
        "joint_vel_steps": [],
        "applied_torque": [],
        "applied_torque_steps": [],
        "body_pos": [],
        "body_quat": [],
        "object_pos": [],
        "object_quat": [],
        "object_lin_vel": [],
        "contact_force": [],
        "contact_force_steps": [],
        "contact_object_force_steps": [],
        "contact_point_w": [],
        "contact_points_w": [],
        "contact_points_force_N": [],
        "contact_points_normal_w": [],
        "flow_points_3d": [],
    }

    rgb_dir = depth_dir = None
    if cfg.output_dir is not None:
        if cfg.save_images:
            rgb_dir = cfg.output_dir / "rgb_images"
            rgb_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_depth:                     # independent of --no-images
            depth_dir = cfg.output_dir / "depth_images"
            depth_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb: list[np.ndarray] = []
    frames_handcam: list[np.ndarray] = []
    handcam_poses: list[np.ndarray] = []
    handcam_intrinsics = None
    object_keys = list(scene.objects.keys())
    prev_target = targets[0:1]
    t_start = time.time()

    for frame in range(num_frames):
        target = targets[frame : frame + 1]
        step_joint_pos: list[torch.Tensor] = []
        step_joint_vel: list[torch.Tensor] = []
        step_torque: list[torch.Tensor] = []

        for step in range(cfg.steps_per_frame):
            step_start = time.perf_counter()
            # Isaac Gym issues one simulate() before setting the new targets, so the first
            # GYM_SUBSTEPS physics steps of a frame still track the previous command.
            command = prev_target if step < cfg.stale_target_steps else target
            if cfg.kinematic:
                # Isaac Gym's --visualize: teleport the joints instead of tracking them with the PD
                robot.write_joint_state_to_sim(command, torch.zeros_like(command))
            robot.set_joint_position_target(command)
            robot.write_data_to_sim()
            for obj in scene.objects.values():
                obj.write_data_to_sim()
            sim.step(render=False)
            robot.update(cfg.physics_dt)
            for obj in scene.objects.values():
                obj.update(cfg.physics_dt)
            # refresh the fingertip force reading every physics step so the history buffer holds
            # the whole frame (reading .data later does not refresh again)
            for sensor in scene.contacts:
                sensor.update(cfg.physics_dt, force_recompute=True)
            # measured joint state + actuator output at every physics step: the *simulated* robot
            # deviates from the commanded trajectory whenever contact resists it, and the frame-rate
            # sample alone (10 Hz) under-resolves that. Kept on the GPU here; copied once per frame.
            step_joint_pos.append(robot.data.joint_pos[0].clone())
            step_joint_vel.append(robot.data.joint_vel[0].clone())
            step_torque.append(robot.data.applied_torque[0].clone())

            # keep the on-screen viewport animating during the frame, not only between frames
            if cfg.gui and step % cfg.gui_render_interval == 0:
                sim.render()
            if cfg.realtime:
                remaining = cfg.physics_dt - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        prev_target = target

        # ---- record ----
        rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        rec["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        rec["joint_target"].append(target[0].cpu().numpy().copy())
        rec["joint_pos_steps"].append(torch.stack(step_joint_pos).cpu().numpy())
        rec["joint_vel_steps"].append(torch.stack(step_joint_vel).cpu().numpy())
        torque_steps = torch.stack(step_torque).cpu().numpy()
        rec["applied_torque_steps"].append(torque_steps)
        rec["applied_torque"].append(torque_steps[-1].copy())

        if body_ids:
            idx = list(body_ids.values())
            rec["body_pos"].append(robot.data.body_pos_w[0, idx].cpu().numpy().copy())
            rec["body_quat"].append(robot.data.body_quat_w[0, idx].cpu().numpy().copy())

        obj_pos, obj_quat, obj_vel = [], [], []
        for key in object_keys:
            data = scene.objects[key].data
            obj_pos.append(data.root_pos_w[0].cpu().numpy().copy())
            obj_quat.append(data.root_quat_w[0].cpu().numpy().copy())
            obj_vel.append(data.root_lin_vel_w[0].cpu().numpy().copy())
        rec["object_pos"].append(np.stack(obj_pos))
        rec["object_quat"].append(np.stack(obj_quat))
        rec["object_lin_vel"].append(np.stack(obj_vel))

        if scene.contacts:
            # history index 0 is the newest physics step; flip to chronological order
            step_net = [
                torch.flip(sensor.data.net_forces_w_history[0, :, 0], dims=[0])
                for sensor in scene.contacts
            ]
            net_steps = torch.stack(step_net, dim=1)                    # [steps, fingertips, 3]
            rec["contact_force"].append(net_steps[-1].cpu().numpy().copy())
            rec["contact_force_steps"].append(net_steps.cpu().numpy().copy())
            pair_forces_np = contact_points_np = None
            if scene.contact_object_keys:
                step_obj = [
                    torch.flip(sensor.data.force_matrix_w_history[0, :, 0], dims=[0])
                    for sensor in scene.contacts
                ]
                obj_steps = torch.stack(step_obj, dim=1)                # [steps, fingertips, objects, 3]
                rec["contact_object_force_steps"].append(obj_steps.cpu().numpy().copy())
                pair_forces_np = rec["contact_object_force_steps"][-1][-1]   # last step [tips, objs, 3]
                # centre of the contact patch on each fingertip per touched object (NaN otherwise)
                contact_points_np = np.stack(
                    [sensor.data.contact_pos_w[0, 0].cpu().numpy() for sensor in scene.contacts]
                )
                rec["contact_point_w"].append(contact_points_np.copy())

                # every individual contact point PhysX resolved this step: position, normal and
                # per-point normal force, addressed per fingertip<->object pair
                n_tips, n_objs = len(scene.contacts), len(scene.contact_object_keys)
                raw_pts = np.full((n_tips, n_objs, MAX_CONTACT_POINTS, 3), np.nan, dtype=np.float32)
                raw_frc = np.full((n_tips, n_objs, MAX_CONTACT_POINTS), np.nan, dtype=np.float32)
                raw_nrm = np.full((n_tips, n_objs, MAX_CONTACT_POINTS, 3), np.nan, dtype=np.float32)
                for t, sensor in enumerate(scene.contacts):
                    forces_b, points_b, normals_b, _sep, counts_b, starts_b = (
                        sensor.contact_physx_view.get_contact_data(dt=cfg.physics_dt)
                    )
                    counts = counts_b.view(-1, n_objs)[0].cpu().numpy()
                    starts = starts_b.view(-1, n_objs)[0].cpu().numpy()
                    for m in range(n_objs):
                        count = min(int(counts[m]), MAX_CONTACT_POINTS)
                        if count > 0:
                            begin = int(starts[m])
                            raw_pts[t, m, :count] = points_b[begin : begin + count].cpu().numpy()
                            raw_frc[t, m, :count] = forces_b[begin : begin + count, 0].cpu().numpy()
                            raw_nrm[t, m, :count] = normals_b[begin : begin + count].cpu().numpy()
                rec["contact_points_w"].append(raw_pts)
                rec["contact_points_force_N"].append(raw_frc)
                rec["contact_points_normal_w"].append(raw_nrm)
                if scene.contact_point_markers is not None:
                    _update_contact_point_markers(scene.contact_point_markers, raw_pts, raw_frc)

            # draw each contact's force arrow at its measured contact point before rendering
            if scene.force_markers is not None:
                tip_ids = [body_ids[l] for l in FINGERTIP_LINKS if l in body_ids]
                if len(tip_ids) == len(scene.contacts):
                    _update_force_arrows(
                        scene.force_markers,
                        robot.data.body_pos_w[0, tip_ids].cpu().numpy(),
                        rec["contact_force"][-1],
                        pair_forces_np,
                        contact_points_np,
                        cfg.force_vis_scale,
                        cfg.force_vis_max_len,
                    )

        if surface_points is not None and manipulated is not None:
            m_idx = object_keys.index(manipulated.key)
            pos = rec["object_pos"][-1][m_idx]
            rot = _quat_wxyz_to_matrix(rec["object_quat"][-1][m_idx])
            rec["flow_points_3d"].append(surface_points @ rot.T + pos)

        # ---- render (once per trajectory frame, like the Isaac Gym script's saved image) ----
        if cfg.render and scene.camera is not None:
            if scene.hand_cam is not None and body_ids:
                # re-aim the close-up camera: hover 30 cm from the fingertip centroid, on the demo
                # camera's side so the view matches the main video's perspective, held above the grip
                tip_ids = [body_ids[l] for l in FINGERTIP_LINKS if l in body_ids]
                centroid = robot.data.body_pos_w[0, tip_ids].mean(dim=0).cpu().numpy().astype(np.float64)
                toward_demo = np.asarray(spec.camera.position, dtype=np.float64) - centroid
                toward_demo /= max(np.linalg.norm(toward_demo), 1e-6)
                eye = centroid + toward_demo * 0.30
                eye[2] = max(eye[2], centroid[2] + 0.12)
                hand_cam_eye, hand_cam_target = eye, centroid
                scene.hand_cam.set_world_poses_from_view(
                    torch.tensor(eye, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(centroid, dtype=torch.float32, device=device).unsqueeze(0),
                )
            sim.render()
            scene.camera.update(cfg.physics_dt, force_recompute=True)
            rgb = scene.camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            if cfg.video:
                frames_rgb.append(rgb)
            if scene.hand_cam is not None and cfg.video:
                scene.hand_cam.update(cfg.physics_dt, force_recompute=True)
                frames_handcam.append(
                    scene.hand_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                )
                # record the exact look-at used this frame (the camera data buffers are not
                # reliably populated for poses set via set_world_poses_from_view)
                handcam_poses.append(np.concatenate([hand_cam_eye, hand_cam_target]))
                if handcam_intrinsics is None:
                    handcam_intrinsics = scene.hand_cam.data.intrinsic_matrices[0].cpu().numpy()
            if rgb_dir is not None and frame % max(1, cfg.save_ratio) == 0:
                _write_png(rgb_dir / f"frame_{frame:04d}.png", rgb)
            if depth_dir is not None and frame % max(1, cfg.save_ratio) == 0:
                depth = scene.camera.data.output.get("distance_to_image_plane")
                if depth is not None:
                    np.save(depth_dir / f"frame_{frame:04d}.npy", depth[0].cpu().numpy())

        if frame % 20 == 0 or frame == num_frames - 1:
            print(
                f"[replay] frame {frame + 1:4d}/{num_frames}  ({time.time() - t_start:6.1f}s elapsed)",
                flush=True,
            )

    return {
        "num_frames": num_frames,
        "joint_names": list(robot.joint_names),
        "traj_joint_names": traj_order,
        "joint_map": traj_to_sim,
        "tracked_links": list(body_ids.keys()),
        "object_keys": object_keys,
        "object_names": scene.object_names,
        "manipulated_key": scene.manipulated_key,
        "static_keys": scene.static_keys,
        "wall_time_s": time.time() - t_start,
        "records": {k: (np.stack(v) if v else np.empty(0)) for k, v in rec.items()},
        "frames_rgb": frames_rgb,
        "frames_handcam": frames_handcam,
        "handcam_poses": (np.stack(handcam_poses) if handcam_poses else np.empty(0)),
        "handcam_intrinsics": handcam_intrinsics,
        "fingertip_bodies": [s.body_names[0] for s in scene.contacts],
        "contact_object_keys": list(scene.contact_object_keys),
    }


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _write_png(path: Path, rgb: np.ndarray) -> None:
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return
    except ImportError:
        pass
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        np.save(path.with_suffix(".npy"), rgb)


def write_video(frames: list[np.ndarray], path: Path, fps: int) -> bool:
    if not frames:
        return False
    try:
        import cv2

        height, width = frames[0].shape[:2]
        writer = VideoWriter(path, fps, (width, height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return writer.release()
    except Exception as exc:
        print(f"[replay][WARN] could not write {path}: {exc}")
        return False
