"""Scene / trajectory description for a Video2Sim2Real run — pure numpy, no Isaac imports.

Everything in here is a direct port of the world layout used by the Isaac Gym scripts
(``contact_opt/optimized_replay.py`` and ``contact_opt/grasp_test/kinova_replay_grasping_test_interaction.py``)
so that the Isaac Lab scene is geometrically identical:

* the world origin sits at the table corner, +Z up, the table top at ``TABLE_HEIGHT``;
* the robot base pose comes from the AprilTag (tag 23) measurement in ``table_frame_pose.json``;
* object poses come from ``Scene_reconstruction/scene_output_final.json`` with the
  ``z -> TABLE_HEIGHT - z`` convention used by the reconstruction pipeline;
* the camera pose comes from ``camera_frame_pose.json`` (tag 0 = table/world reference).

Quaternions are carried as ``xyzw`` (scipy / Isaac Gym order). Isaac Lab wants ``wxyz``;
use :func:`quat_xyzw_to_wxyz` at the boundary.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------------------------------------------
# Constants measured from the real-world table (ported verbatim from the Isaac Gym scripts)
# --------------------------------------------------------------------------------------
TABLE_HEIGHT = 0.80
TABLE_WIDTH = 0.78          # along +X
TABLE_LENGTH = 1.0          # along +Y
TAG_ROBOT_OFFSET = 0.145
TABLE_X_MARGIN = 0.08
TABLE_Y_MARGIN = 0.15

ROBOT_BASE_HEIGHT_OFFSET = 0.0
OBJECT_X_OFFSET = 0.0
OBJECT_Y_OFFSET = 0.0

# 0.9 is the shrink factor grasp_retry_loop.sh:33 passes to grasp-pose *generation* (lightning-
# grasp) only; the pipeline's Isaac Gym replays/tests all ran at 1.0 (see load_run_spec).
GRASP_GENERATION_MESH_SCALE = 0.9

ROBOT_TAG_INDEX = 23
REFERENCE_TAG_INDEX = 0

# Isaac Gym camera sensor settings used to render the "sim view" of the demo
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_HFOV_DEG = 90.0
CAM_NEAR = 1e-2
CAM_FAR = 1.0               # Isaac Gym value; the Isaac Lab replay overrides this (see replay script)

ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
LEAP_JOINT_NAMES = [str(i) for i in range(16)]

# The LEAP joints are named "0".."15" in the original URDF and in every trajectory json. USD prim
# names may not start with a digit, so ``scripts/ingest_data.py`` renames them in this project's copy
# of the URDF (and therefore in the USD and in Isaac Lab). See data/robot/joint_name_map.json.
LEAP_JOINT_RENAME = {str(i): f"leap_j{i}" for i in range(16)}
TRAJ_TO_SIM_JOINT = {**{name: name for name in ARM_JOINT_NAMES}, **LEAP_JOINT_RENAME}


def load_joint_name_map(data_dir: str | os.PathLike) -> dict[str, str]:
    """``{trajectory joint name: joint name in the USD}``, from data/robot/joint_name_map.json."""
    path = Path(data_dir) / "robot" / "joint_name_map.json"
    if not path.is_file():
        return dict(TRAJ_TO_SIM_JOINT)
    with open(path) as f:
        return dict(json.load(f)["map"])


FINGERTIP_LINKS = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
# which finger each LEAP fingertip link belongs to (kinematic chains leap_j0-3, 4-7, 8-11, 12-15)
FINGERTIP_LABELS = {
    "fingertip": "index",
    "fingertip_2": "middle",
    "fingertip_3": "ring",
    "thumb_fingertip": "thumb",
}
# one colour per fingertip (matplotlib tab10 as RGB in [0, 1]), shared between the in-sim force
# arrows and every plot/video so the reading is traceable across them
FINGERTIP_COLORS = {
    "fingertip": (0.121, 0.466, 0.705),        # tab:blue
    "fingertip_2": (1.000, 0.498, 0.054),      # tab:orange
    "fingertip_3": (0.172, 0.627, 0.172),      # tab:green
    "thumb_fingertip": (0.839, 0.152, 0.156),  # tab:red
}
PALM_LINK = "palm_lower"
HAND_MOUNT_LINK = "leap_mount"
BRACELET_LINK = "bracelet_link"


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    out = np.concatenate([q[:, 3:4], q[:, 0:3]], axis=1)
    return out.reshape(-1) if out.shape[0] == 1 else out


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    out = np.concatenate([q[:, 1:4], q[:, 0:1]], axis=1)
    return out.reshape(-1) if out.shape[0] == 1 else out


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------
@dataclass
class Trajectory:
    """A ``retarget_kinova_leap_optimized.json`` trajectory."""

    joint_names: list[str]              # order as stored in the json
    positions: np.ndarray               # [T, num_joints] float32, columns follow joint_names
    key_frames: dict[str, int | None]
    total_frames: int
    object_mesh_scale: float
    traj_id: str | None
    source: Path

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    def reorder(self, target_names: list[str]) -> np.ndarray:
        """Return the trajectory reordered so column *i* corresponds to ``target_names[i]``."""
        name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
        missing = [n for n in target_names if n not in name_to_idx]
        if missing:
            raise KeyError(
                f"trajectory {self.source} has no joint(s) {missing}; it provides {self.joint_names}"
            )
        return self.positions[:, [name_to_idx[n] for n in target_names]]


def load_trajectory(json_path: str | os.PathLike) -> Trajectory:
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    frames = data["traj"]
    joint_names = list(frames[0]["robot_cfg"].keys())
    positions = np.array(
        [[frame["robot_cfg"][name] for name in joint_names] for frame in frames],
        dtype=np.float32,
    )
    key_frames = {
        k: data.get(k)
        for k in [
            "hand_pose_frame",
            "pregrasp_frame",
            "contact_frame",
            "pre_interaction_frame",
            "interaction_frame",
            "drop_frame",
        ]
    }
    return Trajectory(
        joint_names=joint_names,
        positions=positions,
        key_frames=key_frames,
        total_frames=int(data.get("total_frames", len(frames))),
        object_mesh_scale=float(data.get("object_mesh_scale", 1.0)),
        traj_id=data.get("traj_id"),
        source=json_path,
    )


# --------------------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------------------
@dataclass
class SceneObject:
    key: str                    # scene_obj_<i>
    name: str                   # obj_0000
    urdf_path: Path
    mesh_path: Path
    position: np.ndarray        # world frame, meters
    quat_xyzw: np.ndarray
    is_manipulated: bool
    is_static: bool
    friction: float             # PhysX friction (mu1, matching the Isaac Gym "friction" field)
    dynamic_friction: float     # mu2 from the URDF (Isaac Gym only had one coefficient)
    rolling_friction: float
    torsion_friction: float
    restitution: float
    prompt: str = ""
    # collider approximation for this object's mesh: "convex_decomposition" (default, Isaac Gym's
    # VHACD) or "sdf" (PhysX signed-distance-field triangle mesh - exact geometry, needed for
    # interlocking parts such as a nut on a threaded screw; a convex decomposition has no thread).
    collision: str = "convex_decomposition"


@dataclass
class CameraSpec:
    position: np.ndarray        # camera origin in world frame
    quat_xyzw: np.ndarray       # camera orientation in world frame, OpenCV/ROS convention (+Z forward, +Y down)
    width: int = CAM_WIDTH
    height: int = CAM_HEIGHT
    hfov_deg: float = CAM_HFOV_DEG
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    real_intrinsics: tuple[float, float, float, float] | None = None

    def world_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rot, pos) mapping world points to camera frame: ``p_cam = rot @ p_world + pos``."""
        rot_cam_in_world = R.from_quat(self.quat_xyzw).as_matrix()
        rot = rot_cam_in_world.T
        pos = -rot @ self.position
        return rot, pos

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project ``[..., 3]`` world points to ``[..., 2]`` pixel coordinates."""
        rot, pos = self.world_to_camera()
        pts_cam = points_world @ rot.T + pos
        z = pts_cam[..., 2] + 1e-6
        u = self.fx * (pts_cam[..., 0] / z) + self.cx
        v = self.fy * (pts_cam[..., 1] / z) + self.cy
        return np.stack([u, v], axis=-1)


@dataclass
class RunSpec:
    """Everything needed to build the Isaac Lab scene for one trajectory run."""

    traj_run: str
    scene_run: str
    root: Path
    trajectory: Trajectory
    robot_position: np.ndarray
    robot_quat_xyzw: np.ndarray
    objects: list[SceneObject]
    camera: CameraSpec
    object_mesh_scale: float
    object_mesh_scale_source: str = "run_meta"
    # LEAP finger drive effort limit (N m). None = the replay's Isaac-Gym-matched 50 N m. The real
    # LEAP motors (and the URDF) give 0.95 N m; authored teleop scenes set that so the fingers
    # stall against a rigid object instead of pressing with ~1 kN and sinking into it.
    hand_effort_limit: float | None = None
    table_position: np.ndarray = field(default_factory=lambda: np.array([
        -0.5 * TABLE_WIDTH + TABLE_X_MARGIN,
        -0.5 * TABLE_LENGTH + TABLE_Y_MARGIN,
        0.5 * TABLE_HEIGHT,
    ]))
    table_size: np.ndarray = field(default_factory=lambda: np.array([TABLE_WIDTH, TABLE_LENGTH, TABLE_HEIGHT]))

    @property
    def manipulated(self) -> SceneObject | None:
        return next((o for o in self.objects if o.is_manipulated), None)


def _get_float_xml(root, tag_name, default=None):
    elem = root.find(f".//{tag_name}")
    if elem is None or elem.text is None:
        return default
    try:
        return float(elem.text.strip())
    except ValueError:
        return default


def load_urdf_surface_params(
    urdf_path: str | os.PathLike,
    default_friction: float = 0.5,
    default_rolling_friction: float = 0.0,
    default_torsion_friction: float = 0.0,
    default_restitution: float = 0.0,
) -> dict:
    """Port of ``load_urdf_surface_params`` from the Isaac Gym scripts (friction_mode='mu1').

    Reads Gazebo-style ``<mu1>/<mu2>`` plus the ``rolling_friction=/restitution=`` values that the
    reconstruction pipeline leaves in a trailing XML comment.
    """
    import re
    import xml.etree.ElementTree as ET

    params = {
        "friction": default_friction,
        "dynamic_friction": default_friction,
        "rolling_friction": default_rolling_friction,
        "torsion_friction": default_torsion_friction,
        "restitution": default_restitution,
    }
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        print(f"[WARN] URDF not found, using default surface params: {urdf_path}")
        return params

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        mu1 = _get_float_xml(root, "mu1", None)
        mu2 = _get_float_xml(root, "mu2", None)
        restitution = _get_float_xml(root, "restitution", None)
        rolling_friction = _get_float_xml(root, "rolling_friction", None)
        torsion_friction = _get_float_xml(root, "torsion_friction", None)

        if mu1 is not None:
            params["friction"] = mu1                      # Isaac Gym used mu1 as *the* friction
        elif mu2 is not None:
            params["friction"] = mu2
        params["dynamic_friction"] = mu2 if mu2 is not None else params["friction"]

        if restitution is not None:
            params["restitution"] = restitution
        if rolling_friction is not None:
            params["rolling_friction"] = rolling_friction
        if torsion_friction is not None:
            params["torsion_friction"] = torsion_friction

        text = urdf_path.read_text()
        for key in ["rolling_friction", "torsion_friction", "restitution"]:
            m = re.search(rf"{key}\s*=\s*([-+0-9.eE]+)", text)
            if m is not None:
                params[key] = float(m.group(1))
    except Exception as exc:  # pragma: no cover - mirrors the reference's defensive behaviour
        print(f"[WARN] Failed to parse URDF surface params from {urdf_path}: {exc}")

    return params


def load_urdf_inertial(urdf_path: str | os.PathLike) -> dict | None:
    """Read ``<inertial>`` (mass, com, inertia) from a single-link object URDF."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        print(f"[WARN] cannot parse {urdf_path}: {exc}")
        return None

    inertial = root.find(".//inertial")
    if inertial is None:
        return None

    mass_el = inertial.find("mass")
    origin_el = inertial.find("origin")
    inertia_el = inertial.find("inertia")
    if mass_el is None or inertia_el is None:
        return None

    com = [0.0, 0.0, 0.0]
    if origin_el is not None and origin_el.get("xyz"):
        com = [float(v) for v in origin_el.get("xyz").split()]

    return {
        "mass": float(mass_el.get("value", 0.0)),
        "com": com,
        "inertia": {k: float(inertia_el.get(k, 0.0)) for k in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]},
    }


def load_camera_intrinsics(path: str | os.PathLike) -> tuple[float, float, float, float]:
    with open(path) as f:
        line = f.readline().strip().strip("[]")
    values = [float(x.strip()) for x in line.split(",")]
    return tuple(values)  # type: ignore[return-value]


def compute_robot_pose(table_frame_pose_json: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Robot base pose in the world frame, from the tag-23 measurement."""
    with open(table_frame_pose_json) as f:
        poses = json.load(f)

    poses = [
        p
        for p in poses
        if not (
            p.get("tag_index") == REFERENCE_TAG_INDEX
            and all(v == 0.0 for v in p.get("position(m)", []))
            and all(v == 0.0 for v in p.get("orientation_deg_XYZ(deg)", []))
        )
    ]
    entry = next((p for p in poses if p.get("tag_index") == ROBOT_TAG_INDEX), None)
    if entry is None:
        raise ValueError(
            f"{table_frame_pose_json}: no entry for robot tag_index {ROBOT_TAG_INDEX} "
            f"(found {[p.get('tag_index') for p in poses]})"
        )

    pos = np.asarray(entry["position(m)"], dtype=np.float64)
    rot_deg = np.asarray(entry["orientation_deg_XYZ(deg)"], dtype=np.float64)

    position = np.array(
        [
            pos[0] - TAG_ROBOT_OFFSET,
            pos[1],
            TABLE_HEIGHT + pos[2] + ROBOT_BASE_HEIGHT_OFFSET,
        ]
    )
    quat_xyzw = R.from_euler("XYZ", rot_deg, degrees=True).as_quat()
    return position, quat_xyzw


def compute_camera_spec(
    camera_frame_pose_json: str | os.PathLike,
    cam_params_txt: str | os.PathLike | None = None,
) -> CameraSpec:
    """Camera pose in the world frame, from the tag-0 (table/world reference) measurement."""
    with open(camera_frame_pose_json) as f:
        poses = json.load(f)

    entry = next((p for p in poses if p["tag_index"] == REFERENCE_TAG_INDEX), None)
    if entry is None:
        raise ValueError(f"{camera_frame_pose_json}: no entry for reference tag {REFERENCE_TAG_INDEX}")

    world_in_cam_pos = np.asarray(entry["position(m)"], dtype=np.float64)
    world_in_cam_rot_deg = np.asarray(entry["orientation_deg_XYZ(deg)"], dtype=np.float64)

    T_cam_from_world = np.eye(4)
    T_cam_from_world[:3, :3] = R.from_euler("XYZ", world_in_cam_rot_deg, degrees=True).as_matrix()
    T_cam_from_world[:3, 3] = world_in_cam_pos

    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    cam_pos = T_world_from_cam[:3, 3].copy()
    cam_pos[2] += TABLE_HEIGHT           # world origin sits at the table top, tags were measured on it
    cam_rot = R.from_matrix(T_world_from_cam[:3, :3])

    # Isaac Gym cannot honour real intrinsics: it derives them from the sensor's hfov. Keep the same
    # numbers so pixel-space comparisons against the Isaac Gym replay stay valid.
    fx = fy = (CAM_WIDTH / 2) / math.tan(math.radians(CAM_HFOV_DEG) * 0.5)
    cx, cy = CAM_WIDTH / 2, CAM_HEIGHT / 2

    real = None
    if cam_params_txt is not None and Path(cam_params_txt).is_file():
        real = load_camera_intrinsics(cam_params_txt)

    return CameraSpec(
        position=cam_pos,
        quat_xyzw=cam_rot.as_quat(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        real_intrinsics=real,
    )


def load_run_spec(run_dir: str | os.PathLike, object_mesh_scale: float | None = None) -> RunSpec:
    """Load one ingested run folder (``data/runs/<traj_run>``) into a :class:`RunSpec`."""
    run_dir = Path(run_dir).resolve()
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{run_dir} is not an ingested run folder (missing run_meta.json)")
    with open(meta_path) as f:
        meta = json.load(f)

    traj = load_trajectory(run_dir / "retarget_kinova_leap_optimized.json")
    scene_dir = run_dir / "scene"

    robot_pos, robot_quat = compute_robot_pose(scene_dir / "table_frame_pose.json")
    camera = compute_camera_spec(scene_dir / "camera_frame_pose.json", scene_dir / "cam_params.txt")

    static_keys = set(meta.get("static_keys", []))
    objects: list[SceneObject] = []
    for entry in meta["objects"]:
        urdf_path = run_dir / entry["urdf_file"]
        surface = load_urdf_surface_params(urdf_path)
        pos = np.asarray(entry["pos"], dtype=np.float64)
        objects.append(
            SceneObject(
                key=entry["key"],
                name=os.path.splitext(entry["urdf_name"])[0],
                urdf_path=urdf_path,
                mesh_path=run_dir / entry["mesh_file"],
                position=np.array(
                    [pos[0] + OBJECT_X_OFFSET, pos[1] + OBJECT_Y_OFFSET, TABLE_HEIGHT - pos[2]]
                ),
                quat_xyzw=np.asarray(entry["quat_xyzw"], dtype=np.float64),
                is_manipulated=bool(entry["is_manipulated"]),
                is_static=entry["key"] in static_keys,
                friction=float(surface["friction"]),
                dynamic_friction=float(surface["dynamic_friction"]),
                rolling_friction=float(surface["rolling_friction"]),
                torsion_friction=float(surface["torsion_friction"]),
                restitution=float(surface["restitution"]),
                prompt=entry.get("prompt", ""),
                collision=str(entry.get("collision", "convex_decomposition")),
            )
        )

    # Scale provenance, measured against the pipeline's actual invocations (2026-08-26):
    # 0.9 was only ever passed to the *grasp-pose generation* scripts (grasp_retry_loop.sh:33 ->
    # lightning-grasp), and the candidate/optimized JSONs merely record it as metadata. Every Isaac
    # Gym replay and test - kinova_replay_grasping_test_interaction (run.sh:199), the disturbance
    # test (grasp_retry_loop.sh stage 4) and optimized_replay itself - was invoked WITHOUT
    # --object_mesh_scale, i.e. at the scripts' default of 1.0, and none of them read the JSON
    # field. Grasps that "worked in Isaac Gym" therefore closed on full-size objects; replaying at
    # 0.9 shrinks the object out of the grasp (verified: interaction candidate 0027 of
    # run_2026-05-15_17-52-54 lifts at 1.0 and whiffs at 0.9 in both engines). So: honour an
    # explicit field, otherwise fall back to 1.0 like every replay the pipeline actually ran.
    if object_mesh_scale is not None:
        scale = float(object_mesh_scale)
        scale_source = "cli"
    elif meta.get("object_mesh_scale") is not None:
        scale = float(meta["object_mesh_scale"])
        scale_source = "run_meta"
    else:
        scale = 1.0
        scale_source = "isaac gym replay default"

    return RunSpec(
        traj_run=meta["traj_run"],
        scene_run=meta["scene_run"],
        root=run_dir,
        trajectory=traj,
        robot_position=robot_pos,
        robot_quat_xyzw=robot_quat,
        objects=objects,
        camera=camera,
        object_mesh_scale=scale,
        object_mesh_scale_source=scale_source,
        hand_effort_limit=(float(meta["hand_effort_limit"]) if meta.get("hand_effort_limit") is not None else None),
    )


def describe(spec: RunSpec) -> str:
    lines = [
        f"run              : {spec.traj_run}  (scene: {spec.scene_run})",
        f"frames           : {spec.trajectory.num_frames} (total_frames={spec.trajectory.total_frames})",
        f"key frames       : {spec.trajectory.key_frames}",
        f"object scale     : {spec.object_mesh_scale}  (from {spec.object_mesh_scale_source})",
        f"hand effort limit: {spec.hand_effort_limit if spec.hand_effort_limit is not None else 'replay default'} N m",
        f"robot base pos   : {np.round(spec.robot_position, 5).tolist()}",
        f"robot base quat  : {np.round(spec.robot_quat_xyzw, 5).tolist()} (xyzw)",
        f"table centre     : {np.round(spec.table_position, 5).tolist()}  size {spec.table_size.tolist()}",
        f"camera pos       : {np.round(spec.camera.position, 5).tolist()}",
        f"camera quat      : {np.round(spec.camera.quat_xyzw, 5).tolist()} (xyzw, +Z forward)",
        f"camera intrinsics: fx={spec.camera.fx:.2f} fy={spec.camera.fy:.2f} "
        f"cx={spec.camera.cx:.1f} cy={spec.camera.cy:.1f} (real: {spec.camera.real_intrinsics})",
    ]
    for obj in spec.objects:
        lines.append(
            f"  object {obj.key:<14s} {obj.name:<10s} pos={np.round(obj.position, 4).tolist()} "
            f"quat={np.round(obj.quat_xyzw, 4).tolist()} "
            f"{'[manipulated]' if obj.is_manipulated else ''}{'[static]' if obj.is_static else ''} "
            f"mu={obj.friction}/{obj.dynamic_friction} restitution={obj.restitution} "
            f"{'collision=' + obj.collision + ' ' if obj.collision != 'convex_decomposition' else ''}"
            f"({obj.prompt})"
        )
    return "\n".join(lines)
