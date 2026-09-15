"""Visualisation helpers for the play2perfect scene: fingertip force arrows + a demo camera.

Import **after** ``AppLauncher`` (Isaac Lab imports). The arrow code is the one from
``zerofact.replay`` with the fingertip list made explicit, so both robots draw their forces
the same way; it lives here rather than being imported so this package does not depend on the
Kinova/LEAP scene module.
"""

from __future__ import annotations

import math

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, CameraCfg

from .robot_spec import FINGERTIP_BODIES, FINGERTIP_COLORS

FORCE_ARROW_SHAFT_RADIUS = 0.0035
FORCE_ARROW_HEAD_RADIUS = 0.009
FORCE_ARROW_HEAD_LENGTH = 0.025
FORCE_ARROW_MIN_FORCE = 0.05               # N below which the arrow is hidden


# ----------------------------------------------------------------------------------------------
# force arrows
# ----------------------------------------------------------------------------------------------
def make_force_arrow_markers(
    fingertip_bodies: list[str] = FINGERTIP_BODIES, prim_path: str = "/Visuals/FingertipForces"
) -> VisualizationMarkers:
    """One arrow (cylinder shaft + cone head) prototype pair per fingertip, in its colour.

    Prototype order is ``[shaft_0, head_0, shaft_1, head_1, ...]`` following *fingertip_bodies*.
    Plain ``UsdGeom`` prims, so they show up in the offscreen demo camera and the GUI viewport.
    """
    markers: dict[str, sim_utils.SpawnerCfg] = {}
    for link in fingertip_bodies:
        color = FINGERTIP_COLORS.get(link, (1.0, 1.0, 1.0))
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.4)
        markers[f"{link}_shaft"] = sim_utils.CylinderCfg(
            radius=FORCE_ARROW_SHAFT_RADIUS, height=1.0, visual_material=material
        )
        markers[f"{link}_head"] = sim_utils.ConeCfg(
            radius=FORCE_ARROW_HEAD_RADIUS, height=FORCE_ARROW_HEAD_LENGTH, visual_material=material
        )
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers=markers))


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


def update_force_arrows(
    markers: VisualizationMarkers,
    tip_positions: np.ndarray,           # [num_fingertips, 3] world (fallback anchor only)
    net_forces: np.ndarray,              # [num_fingertips, 3] net contact force, world
    pair_forces: np.ndarray | None,      # [num_fingertips, num_objects, 3] fingertip<->object force
    contact_points: np.ndarray | None,   # [num_fingertips, num_objects, 3], NaN when not touching
    scale: float,
    max_len: float,
) -> None:
    """Draw one arrow per fingertip<->object contact, anchored at the measured contact point.

    A fingertip carrying force with no reported patch centre (contact with something outside the
    pair filters, e.g. another robot link) gets the residual drawn at the fingertip body origin.
    """
    n_tips = tip_positions.shape[0]
    n_objects = pair_forces.shape[1] if (pair_forces is not None and pair_forces.size) else 0
    slots = n_tips * (n_objects + 1)                    # per-object arrows + one residual per tip
    translations = np.tile(np.array([0.0, 0.0, -10.0]), (2 * slots, 1))  # hidden: underground
    orientations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2 * slots, 1))
    scales = np.full((2 * slots, 3), 1e-4)
    marker_indices = np.zeros(2 * slots, dtype=np.int64)

    def draw(slot: int, origin: np.ndarray, force: np.ndarray) -> None:
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
            draw(slot, point, force)
            slot += 1
        marker_indices[2 * slot] = 2 * tip
        marker_indices[2 * slot + 1] = 2 * tip + 1
        draw(slot, tip_positions[tip], net_forces[tip] - accounted)   # residual (unfiltered contacts)
        slot += 1

    markers.visualize(
        translations=translations, orientations=orientations, scales=scales, marker_indices=marker_indices
    )


# ----------------------------------------------------------------------------------------------
# translucent goal marker
def make_goal_marker_translucent(prim_path: str, opacity: float = 0.18) -> None:
    """Make the GoalViz copy of the part (the target pose) translucent so it is told apart from
    the real part even when the two coincide.  Visual only: the marker's OWN material shaders
    (its baked USD is a separate copy of the part's, so the part keeps its look) get the opacity
    inputs; no physics prim is touched.

    Rendering notes (checked on this machine, RTX real-time, Isaac Sim 5.1): binding a NEW
    material to the already-referenced meshes never reaches the renderer, editing the existing
    shader does.  Fractional opacity needs ``/rtx/translucency/enabled``
    (RenderCfg.enable_translucency in make_env_cfg) and ``/rtx/raytracing/fractionalCutoutOpacity``
    at renderer start-up (kit arg in launch.finalize_launcher_args); it is a stochastic cutout,
    so the render config uses TAA (FXAA leaves it grainy)."""
    import carb
    from pxr import Sdf, Usd, UsdShade

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        print(f"[vis] goal marker {prim_path} not found; left opaque", flush=True)
        return
    edited = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() == "UsdPreviewSurface":
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        else:                                       # OmniPBR (URDF importer default) and kin
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(float(opacity))
        edited.append(str(prim.GetPath()))
    st = carb.settings.get_settings()
    st.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
    print(f"[vis] goal marker {prim_path}: opacity {opacity} on {len(edited)} shader(s) {edited}; "
          f"translucency {st.get('/rtx/translucency/enabled')}, fractional cutout "
          f"{st.get('/rtx/raytracing/fractionalCutoutOpacity')}", flush=True)


def hide_goal_marker(prim_path: str) -> None:
    """Hide the GoalViz marker from every render product (USD visibility, no shader edit) - the
    same thing play2perfect does for its distillation-student camera
    (``scene_utils.hide_goal_viz_for_student_camera``). Visual only: the marker never had
    collision, and its pose is still simulated and recorded."""
    from pxr import UsdGeom

    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[vis] goal marker {prim_path} not found; nothing hidden", flush=True)
        return
    UsdGeom.Imageable(prim).MakeInvisible()
    print(f"[vis] goal marker {prim_path}: hidden (USD visibility)", flush=True)


# demo camera
# ----------------------------------------------------------------------------------------------
def look_at_quat_wxyz(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Orientation of a camera at *eye* looking at *target* in Isaac Lab's ``world`` convention
    (+X forward, +Y left, +Z up), as a wxyz quaternion."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross(np.asarray(up, dtype=np.float64), fwd)
    left /= np.linalg.norm(left)
    up_c = np.cross(fwd, left)
    rot = np.stack([fwd, left, up_c], axis=1)          # columns = camera axes in world
    from scipy.spatial.transform import Rotation as R

    x, y, z, w = R.from_matrix(rot).as_quat()
    return (float(w), float(x), float(y), float(z))


def make_demo_camera(
    eye, target, width: int = 640, height: int = 480, prim_path: str = "/World/DemoCam"
) -> Camera:
    """Fixed RGB camera framing the table; read it with ``sim.render()`` + ``camera.update``."""
    cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in eye), rot=look_at_quat_wxyz(eye, target), convention="world"
        ),
    )
    return Camera(cfg)
