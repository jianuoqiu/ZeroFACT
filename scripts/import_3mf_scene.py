#!/usr/bin/env python3
"""Turn a 3MF print project into an ingested *scene run* under ``data/runs/<name>/``.

The result has exactly the layout every existing tool understands (the same one
``scripts/ingest_data.py`` and ``sim_teleop/package_run.py`` produce), so nothing downstream is
new code::

    data/runs/<name>/
        run_meta.json                        objects, poses, manipulated/static keys, provenance
        retarget_kinova_leap_optimized.json  a HELD start pose (donor frame 0) so the run loads,
                                             replays and serves as --init-from for live teleop
        scene/table_frame_pose.json          copied from the donor run (robot base calibration)
        scene/camera_frame_pose.json         copied from the donor run (demo camera)
        scene/cam_params.txt                 copied from the donor run
        scene/meshes/obj_XXXX.obj            one mesh per object, metres, XY-centred, z_min = 0
        scene/urdfs/obj_XXXX.urdf            single-link URDF: mass/COM/inertia + gazebo friction

Afterwards::

    python scripts/convert_assets.py --objects-only --runs <name>      # -> assets/usd/scenes/<name>/
    python scripts/replay_trajectory.py --run <name> --no-render       # settle test / smoke test
    ./sim_teleop/run_live_teleop.sh my_ep --scene-from <name> --objects-from <name>
    python force_controller/run_tracking.py --episode outputs/teleop_my_ep

3MF specifics handled here (Bambu Studio / PrusaSlicer "production extension" layout):

* the root ``3D/3dmodel.model`` holds only ``<component p:path=... objectid=...>`` references and
  the ``<build><item>`` placements; the triangles live in ``3D/Objects/*.model``;
* units are millimetres (``unit=`` is honoured), Z is up (print bed) - same as the sim world;
* every ``<item>`` is one rigid body. Items sharing a mesh share it on disk too (one OBJ/URDF per
  unique mesh would break the one-USD-per-object convention, so each item gets its own files);
* per-plate print weights from ``Metadata/slice_info.config`` become the object masses when the
  slicer stored them (one object per plate); otherwise mass = volume x PLA density x fill
  fraction (``--fill-fraction``), with ``--mass`` overriding either.

Placement: ``--place <idx>=x,y[,yaw_deg[,z]]`` in the world frame (table corner origin, metres,
+Z up; ``z`` is the height of the object's base above the table top, default 0 = standing on the
table; the existing scenes' objects sit at x in [-0.41, -0.12], y in [-0.31, -0.08]). Objects not
placed explicitly are lined up along +Y at ``--row-x``. Object indices follow build order;
``--inspect`` prints them (and mesh sizes / masses) without writing anything.

Interlocking parts (a nut already on its screw): give both ``--collision <idx>=sdf`` so PhysX
collides the exact triangle meshes through signed distance fields - a convex decomposition has no
thread and the nut would just fall or sit on the hulls. Place the nut at the screw's x,y with a
``z`` on the thread and the yaw where the helices mate (see docs / run_meta ``import_command``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sim_teleop.teleop_config import DEFAULT_SCENE_DONOR, RUNS_DIR, TARGET_FPS  # noqa: E402

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
UNIT_TO_M = {"micron": 1e-6, "millimeter": 1e-3, "centimeter": 1e-2, "inch": 0.0254, "foot": 0.3048, "meter": 1.0}

PLA_DENSITY_G_CM3 = 1.24
# printed parts are hollow: walls + sparse infill. Measured on M110-6-Kant-Schraube-Stifthalter
# (10 % infill; slicer-reported 218.82 g / 73.76 g against closed-mesh volumes of 814 / 227 cm3):
# 0.22-0.26 of solid PLA. Small thin parts are mostly wall and land far higher - pass --mass for
# those (a 20 mm printed hex nut is ~2-3 g, not 1 g).
DEFAULT_FILL_FRACTION = 0.3

# gazebo-style surface parameters written into the URDF, in the same fields the reconstruction
# pipeline used (scene_spec.load_urdf_surface_params reads mu1/mu2 + the trailing comment)
DEFAULT_MU1 = 0.5
DEFAULT_MU2 = 0.4
DEFAULT_ROLLING = 0.003
DEFAULT_RESTITUTION = 0.3
LEAP_URDF_EFFORT = 0.95      # N m, <limit effort=> of every LEAP joint in v12_vision.urdf

KEY_FRAME_NAMES = ["hand_pose_frame", "pregrasp_frame", "contact_frame",
                   "pre_interaction_frame", "interaction_frame", "drop_frame"]


# --------------------------------------------------------------------------------------
# 3MF parsing (stdlib only; trimesh's 3MF loader needs lxml, which env_isaaclab lacks)
# --------------------------------------------------------------------------------------
def _tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def parse_transform(text: str | None) -> np.ndarray:
    """3MF ``transform`` = 12 numbers, row-vector convention: ``p' = [p 1] @ M(4x3)``."""
    if not text:
        return np.eye(4)
    v = [float(t) for t in text.split()]
    if len(v) != 12:
        raise ValueError(f"bad 3MF transform {text!r}")
    m = np.eye(4)
    m[:3, :3] = np.array(v[:9]).reshape(3, 3).T     # column-vector form
    m[:3, 3] = v[9:12]
    return m


@dataclass
class MeshData:
    vertices: np.ndarray        # [N, 3] in model units
    faces: np.ndarray           # [M, 3] int


@dataclass
class BuildItem:
    index: int
    object_id: str
    name: str
    transform: np.ndarray       # item -> bed, model units
    parts: list[tuple[MeshData, np.ndarray]] = field(default_factory=list)  # (mesh, mesh -> item)
    mesh_key: str = ""          # identifies items sharing the same geometry


class ThreeMF:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path)
        self.names = set(self.zf.namelist())
        self._mesh_cache: dict[tuple[str, str], MeshData] = {}
        self._model_cache: dict[str, ET.Element] = {}
        root = self._model("3D/3dmodel.model")
        self.unit = root.get("unit", "millimeter")
        self.scale_to_m = UNIT_TO_M[self.unit]
        self.object_names = self._read_object_names()
        self.plate_weights = self._read_plate_weights()

    # ---- package plumbing ----
    def _model(self, name: str) -> ET.Element:
        name = name.lstrip("/")
        if name not in self._model_cache:
            with self.zf.open(name) as f:
                self._model_cache[name] = ET.parse(f).getroot()
        return self._model_cache[name]

    def _read_object_names(self) -> dict[str, str]:
        """``Metadata/model_settings.config``: build object id -> user-visible name."""
        names: dict[str, str] = {}
        if "Metadata/model_settings.config" not in self.names:
            return names
        root = self._model("Metadata/model_settings.config")
        for obj in root.iter("object"):
            for md in obj.findall("metadata"):
                if md.get("key") == "name" and md.get("value"):
                    names[obj.get("id")] = md.get("value")
                    break
        return names

    def _read_plate_weights(self) -> dict[str, float]:
        """Slicer-reported print weight (g) per build object, when a plate holds exactly one."""
        weights: dict[str, float] = {}
        if not {"Metadata/model_settings.config", "Metadata/slice_info.config"} <= self.names:
            return weights
        settings = self._model("Metadata/model_settings.config")
        identify_to_object: dict[str, str] = {}
        for plate in settings.iter("plate"):
            for inst in plate.findall("model_instance"):
                md = {m.get("key"): m.get("value") for m in inst.findall("metadata")}
                if "identify_id" in md and "object_id" in md:
                    identify_to_object[md["identify_id"]] = md["object_id"]
        info = self._model("Metadata/slice_info.config")
        for plate in info.iter("plate"):
            objs = plate.findall("object")
            grams = sum(float(f.get("used_g", 0.0)) for f in plate.findall("filament"))
            if len(objs) == 1 and grams > 0 and objs[0].get("identify_id") in identify_to_object:
                weights[identify_to_object[objs[0].get("identify_id")]] = grams
        return weights

    # ---- geometry ----
    def _mesh(self, model_path: str, object_id: str) -> MeshData:
        key = (model_path.lstrip("/"), object_id)
        if key in self._mesh_cache:
            return self._mesh_cache[key]
        root = self._model(model_path)
        obj = next((o for o in root.iter(_tag(NS_CORE, "object")) if o.get("id") == object_id), None)
        if obj is None:
            raise KeyError(f"{self.path.name}: object {object_id} not in {model_path}")
        mesh = obj.find(_tag(NS_CORE, "mesh"))
        if mesh is None:
            raise ValueError(f"{self.path.name}: object {object_id} in {model_path} has no <mesh>")
        verts = np.array(
            [[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
             for v in mesh.find(_tag(NS_CORE, "vertices"))],
            dtype=np.float64,
        )
        faces = np.array(
            [[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))]
             for t in mesh.find(_tag(NS_CORE, "triangles"))],
            dtype=np.int64,
        )
        self._mesh_cache[key] = MeshData(verts, faces)
        return self._mesh_cache[key]

    def _collect(self, model_path: str, object_id: str, xf: np.ndarray,
                 out: list[tuple[MeshData, np.ndarray]], keys: list[str]) -> None:
        root = self._model(model_path)
        obj = next((o for o in root.iter(_tag(NS_CORE, "object")) if o.get("id") == object_id), None)
        if obj is None:
            raise KeyError(f"{self.path.name}: object {object_id} not in {model_path}")
        if obj.find(_tag(NS_CORE, "mesh")) is not None:
            out.append((self._mesh(model_path, object_id), xf))
            keys.append(f"{model_path.lstrip('/')}#{object_id}")
            return
        comps = obj.find(_tag(NS_CORE, "components"))
        if comps is None:
            return
        for comp in comps.findall(_tag(NS_CORE, "component")):
            sub_path = comp.get(_tag(NS_PROD, "path")) or model_path
            self._collect(sub_path, comp.get("objectid"), xf @ parse_transform(comp.get("transform")), out, keys)

    def build_items(self) -> list[BuildItem]:
        root = self._model("3D/3dmodel.model")
        build = root.find(_tag(NS_CORE, "build"))
        items: list[BuildItem] = []
        for i, it in enumerate(build.findall(_tag(NS_CORE, "item"))):
            oid = it.get("objectid")
            item = BuildItem(index=i, object_id=oid, name=self.object_names.get(oid, f"object_{oid}"),
                             transform=parse_transform(it.get("transform")))
            keys: list[str] = []
            self._collect("3D/3dmodel.model", oid, np.eye(4), item.parts, keys)
            item.mesh_key = "+".join(keys)
            items.append(item)
        return items


# --------------------------------------------------------------------------------------
# One rigid body per build item
# --------------------------------------------------------------------------------------
@dataclass
class SceneBody:
    index: int
    name: str                   # obj_0000
    source_name: str            # slicer object name
    mesh: "object"              # trimesh.Trimesh, metres, canonical frame
    size: np.ndarray            # bbox extents (m)
    volume_m3: float
    watertight: bool
    mass_kg: float
    mass_source: str
    com: np.ndarray
    inertia: dict[str, float]
    pos: np.ndarray             # table-frame placement, z=0 -> object base on the table top
    quat_xyzw: np.ndarray
    is_manipulated: bool = False
    is_static: bool = False
    prompt: str = ""
    collision: str = "convex_decomposition"


def canonical_mesh(item: BuildItem, scale_to_m: float):
    """Merge the item's parts, apply the item's rotation, convert to metres, centre XY, floor at z=0.

    Same convention as the reconstruction pipeline's ``*_transformed.obj``: the URDF origin is
    on the table top under the bbox centre, so ``pos = [x, y, 0]`` in run_meta puts the object
    standing on the table at (x, y).
    """
    import trimesh

    parts = []
    for mesh, xf in item.parts:
        rot_only = item.transform.copy()
        rot_only[:3, 3] = 0.0                       # bed placement is irrelevant; keep orientation
        full = rot_only @ xf
        v = (np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ full.T)[:, :3]
        parts.append(trimesh.Trimesh(vertices=v * scale_to_m, faces=mesh.faces, process=False))
    tm = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
    tm.merge_vertices()
    tm.remove_unreferenced_vertices()
    lo, hi = tm.bounds
    tm.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return tm


def mass_properties(tm, mass_kg: float) -> tuple[float, bool, np.ndarray, dict[str, float]]:
    """Volume, watertightness, COM and inertia (about the COM, kg m^2) for a uniform solid of
    the given total mass. Non-watertight meshes fall back to their convex hull for the solid."""
    solid = tm
    watertight = bool(tm.is_watertight)
    if not watertight:
        solid = tm.convex_hull
    volume = float(abs(solid.volume))
    # trimesh: density-weighted tensor; rescale to the requested mass
    props = solid.mass_properties
    ratio = mass_kg / float(props["mass"]) if props["mass"] > 0 else 0.0
    inertia = np.asarray(props["inertia"], dtype=np.float64) * ratio
    com = np.asarray(props["center_mass"], dtype=np.float64)
    return volume, watertight, com, {
        "ixx": float(inertia[0, 0]), "iyy": float(inertia[1, 1]), "izz": float(inertia[2, 2]),
        "ixy": float(inertia[0, 1]), "ixz": float(inertia[0, 2]), "iyz": float(inertia[1, 2]),
    }


def urdf_text(body: SceneBody, mesh_rel: str, mu1: float, mu2: float, rolling: float, restitution: float) -> str:
    i = body.inertia
    return f"""<?xml version="1.0"?>
<robot name="{body.name}">
  <link name="base_link">
    <inertial>
      <origin xyz="{body.com[0]:.9f} {body.com[1]:.9f} {body.com[2]:.9f}" rpy="0 0 0"/>
      <mass value="{body.mass_kg:.9f}"/>
      <inertia
        ixx="{i['ixx']:.15g}" ixy="{i['ixy']:.15g}" ixz="{i['ixz']:.15g}"
        iyy="{i['iyy']:.15g}" iyz="{i['iyz']:.15g}"
        izz="{i['izz']:.15g}"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel}"/>
      </geometry>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel}"/>
      </geometry>
    </collision>
  </link>

  <gazebo reference="base_link">
    <mu1>{mu1:.6f}</mu1>
    <mu2>{mu2:.6f}</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
    <minDepth>0.001</minDepth>
    <maxVel>0.1</maxVel>
    <material>Gazebo/Grey</material>
  </gazebo>

  <!-- reference only: rolling_friction={rolling:.6f}, restitution={restitution:.6f} -->
  <!-- source: {body.source_name} from the 3MF; mass {body.mass_source} -->
</robot>
"""


def yaw_quat_xyzw(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([0.0, 0.0, math.sin(h), math.cos(h)])


def parse_place(tokens: list[str] | None) -> dict[int, tuple[float, float, float, float]]:
    """``<idx>=x,y[,yaw_deg[,z]]`` -> {idx: (x, y, yaw_deg, z_above_table)}."""
    out: dict[int, tuple[float, float, float, float]] = {}
    num = r"([-+0-9.eE]+)"
    for tok in tokens or []:
        m = re.fullmatch(rf"\s*(\d+)\s*=\s*{num}\s*,\s*{num}(?:\s*,\s*{num})?(?:\s*,\s*{num})?\s*", tok)
        if not m:
            raise SystemExit(f"--place expects <idx>=x,y[,yaw_deg[,z]], got {tok!r}")
        out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)),
                                float(m.group(4) or 0.0), float(m.group(5) or 0.0))
    return out


def parse_indexed(tokens: list[str] | None, kind: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for tok in tokens or []:
        if "=" not in tok:
            raise SystemExit(f"--{kind} expects <idx>=<value>, got {tok!r}")
        k, v = tok.split("=", 1)
        out[int(k)] = v
    return out


def parse_indices(text: str | None) -> set[int]:
    if not text:
        return set()
    return {int(t) for t in text.replace(",", " ").split()}


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--3mf", dest="threemf", required=True, help="the .3mf print project")
    ap.add_argument("--name", required=True, help="run name -> data/runs/<name> and assets/usd/scenes/<name>")
    ap.add_argument("--inspect", action="store_true", help="print the build items + derived masses, write nothing")
    ap.add_argument("--place", nargs="*", default=None, metavar="IDX=X,Y[,YAW[,Z]]",
                    help="world-frame placement per build item (metres, degrees; Z = base height above the table)")
    ap.add_argument("--collision", nargs="*", default=None, metavar="IDX=sdf|convex_decomposition",
                    help="collider approximation per item (default convex_decomposition; sdf for threads/interlocks)")
    ap.add_argument("--row-x", type=float, default=-0.30, help="x of the default row for unplaced items")
    ap.add_argument("--row-y0", type=float, default=-0.30, help="y of the first unplaced item")
    ap.add_argument("--row-gap", type=float, default=0.03, help="clearance between unplaced items along +Y")
    ap.add_argument("--manipulated", type=int, default=None, help="build item index of the manipulated object")
    ap.add_argument("--static", default=None, help="build item indices welded in place (fixtures), e.g. '0' or '0,2'")
    ap.add_argument("--prompt", nargs="*", default=None, metavar="IDX=TEXT", help="object description per item")
    ap.add_argument("--mass", nargs="*", default=None, metavar="IDX=GRAMS", help="mass override per item (g)")
    ap.add_argument("--fill-fraction", type=float, default=DEFAULT_FILL_FRACTION,
                    help="printed-part solid fraction for the volume-based mass estimate")
    ap.add_argument("--mu1", type=float, default=DEFAULT_MU1)
    ap.add_argument("--mu2", type=float, default=DEFAULT_MU2)
    ap.add_argument("--rolling-friction", type=float, default=DEFAULT_ROLLING)
    ap.add_argument("--restitution", type=float, default=DEFAULT_RESTITUTION)
    ap.add_argument("--hand-effort", type=float, default=LEAP_URDF_EFFORT,
                    help="LEAP finger effort limit for this scene, N m (default: the URDF's real-motor value; "
                         "the Isaac-Gym-matched replay uses 50, which lets fingers press ~1 kN into a rigid part)")
    ap.add_argument("--scene-from", default=DEFAULT_SCENE_DONOR,
                    help=f"donor run for calibration + start pose (default {DEFAULT_SCENE_DONOR})")
    ap.add_argument("--hold-frames", type=int, default=30,
                    help="length of the held start-pose trajectory (frames of 0.1 s)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run folder")
    args = ap.parse_args()

    import trimesh  # noqa: F401  (fail early, before any parsing)

    src = Path(args.threemf).resolve()
    if not src.is_file():
        raise SystemExit(f"{src} not found")
    pkg = ThreeMF(src)
    items = pkg.build_items()
    if not items:
        raise SystemExit(f"{src.name}: no <build><item> entries")
    print(f"[3mf] {src.name}: unit={pkg.unit}, {len(items)} build item(s), "
          f"slicer weights for object ids {sorted(pkg.plate_weights) or '-'}")

    places = parse_place(args.place)
    prompts = parse_indexed(args.prompt, "prompt")
    masses = {k: float(v) for k, v in parse_indexed(args.mass, "mass").items()}
    collisions = parse_indexed(args.collision, "collision")
    bad = {k: v for k, v in collisions.items() if v not in ("sdf", "convex_decomposition")}
    if bad:
        raise SystemExit(f"--collision values must be sdf or convex_decomposition, got {bad}")
    static_idx = parse_indices(args.static)
    for idx in set(places) | set(prompts) | set(masses) | set(collisions) | static_idx | ({args.manipulated} - {None}):
        if idx < 0 or idx >= len(items):
            raise SystemExit(f"build item index {idx} out of range 0..{len(items) - 1}")

    # ---- bodies ----
    bodies: list[SceneBody] = []
    for item in items:
        tm = canonical_mesh(item, pkg.scale_to_m)
        vol_solid = float(abs((tm if tm.is_watertight else tm.convex_hull).volume))
        if item.index in masses:
            mass_kg, mass_source = masses[item.index] / 1000.0, "cli override"
        elif item.object_id in pkg.plate_weights:
            mass_kg, mass_source = pkg.plate_weights[item.object_id] / 1000.0, "slicer print weight"
        else:
            mass_kg = vol_solid * 1e6 * PLA_DENSITY_G_CM3 * args.fill_fraction / 1000.0
            mass_source = f"volume x {PLA_DENSITY_G_CM3} g/cm3 x fill {args.fill_fraction:g}"
        volume, watertight, com, inertia = mass_properties(tm, mass_kg)
        bodies.append(SceneBody(
            index=item.index, name=f"obj_{item.index:04d}", source_name=item.name, mesh=tm,
            size=tm.extents.copy(), volume_m3=volume, watertight=watertight,
            mass_kg=mass_kg, mass_source=mass_source, com=com, inertia=inertia,
            pos=np.zeros(3), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            prompt=prompts.get(item.index, Path(item.name).stem),
        ))

    # ---- placement ----
    y_cursor = args.row_y0
    for b in bodies:
        if b.index in places:
            x, y, yaw, z = places[b.index]
        else:
            x, yaw, z = args.row_x, 0.0, 0.0
            y = y_cursor + b.size[1] / 2
            y_cursor = y + b.size[1] / 2 + args.row_gap
        # run_meta "pos" follows the reconstruction convention: world z = TABLE_HEIGHT - pos[2]
        b.pos = np.array([x, y, -z])
        b.quat_xyzw = yaw_quat_xyzw(yaw)
        b.is_static = b.index in static_idx
        b.collision = collisions.get(b.index, "convex_decomposition")
    if args.manipulated is not None:
        bodies[args.manipulated].is_manipulated = True
        bodies[args.manipulated].is_static = False        # never freeze the manipulated object

    print(f"{'idx':>3} {'name':<9} {'source':<36} {'size (mm)':<20} {'faces':>7} {'vol cm3':>8} "
          f"{'mass g':>7}  {'pos (m)':<22} yaw  flags")
    for b in bodies:
        yaw = math.degrees(2 * math.atan2(b.quat_xyzw[2], b.quat_xyzw[3]))
        flags = ("manipulated " if b.is_manipulated else "") + ("static " if b.is_static else "") + \
                (f"{b.collision} " if b.collision != "convex_decomposition" else "") + \
                ("" if b.watertight else "NOT-WATERTIGHT(hull) ")
        print(f"{b.index:>3} {b.name:<9} {b.source_name[:36]:<36} "
              f"{'x'.join(f'{v * 1000:.1f}' for v in b.size):<20} {len(b.mesh.faces):>7} "
              f"{b.volume_m3 * 1e6:>8.1f} {b.mass_kg * 1000:>7.1f}  "
              f"{str([round(float(v), 3) for v in b.pos]):<22} {yaw:>4.0f} {flags}[{b.mass_source}]")
    if args.inspect:
        return 0

    # ---- write the run ----
    run_dir = RUNS_DIR / args.name
    if run_dir.exists():
        if not args.force:
            raise SystemExit(f"{run_dir} exists - pass --force to overwrite")
        shutil.rmtree(run_dir)
    (run_dir / "scene" / "meshes").mkdir(parents=True)
    (run_dir / "scene" / "urdfs").mkdir(parents=True)

    donor_scene = RUNS_DIR / args.scene_from / "scene"
    for fname in ("table_frame_pose.json", "camera_frame_pose.json", "cam_params.txt"):
        src_f = donor_scene / fname
        if not src_f.is_file():
            raise FileNotFoundError(f"donor calibration file missing: {src_f}")
        shutil.copy2(src_f, run_dir / "scene" / fname)

    objects = []
    for b in bodies:
        mesh_file = f"scene/meshes/{b.name}.obj"
        urdf_file = f"scene/urdfs/{b.name}.urdf"
        b.mesh.export(run_dir / mesh_file, file_type="obj", include_normals=False, include_texture=False)
        (run_dir / urdf_file).write_text(
            urdf_text(b, f"../meshes/{b.name}.obj", args.mu1, args.mu2, args.rolling_friction, args.restitution))
        objects.append({
            "key": f"scene_obj_{b.index}",
            "urdf_name": f"{b.name}.urdf",
            "mesh_name": f"meshes/{b.name}.obj",
            "pos": [float(v) for v in b.pos],
            "quat_xyzw": [float(v) for v in b.quat_xyzw],
            "is_manipulated": bool(b.is_manipulated),
            "prompt": b.prompt,
            "collision": b.collision,
            "mesh_file": mesh_file,
            "urdf_file": urdf_file,
            "source_3mf_object": b.source_name,
            "size_m": [float(v) for v in b.size],
            "mass_kg": float(b.mass_kg),
            "mass_source": b.mass_source,
        })

    # held start pose: the donor's frame 0 (a proven pose over the table), repeated
    donor_traj = json.loads((RUNS_DIR / args.scene_from / "retarget_kinova_leap_optimized.json").read_text())
    frame0 = donor_traj["traj"][0]
    n = max(1, int(args.hold_frames))
    traj = {
        "traj_id": args.name,
        "total_frames": n,
        **{k: None for k in KEY_FRAME_NAMES},
        "traj": [{"robot_cfg": dict(frame0["robot_cfg"])} for _ in range(n)],
    }
    (run_dir / "retarget_kinova_leap_optimized.json").write_text(json.dumps(traj, indent=2))

    manipulated = next((o["key"] for o in objects if o["is_manipulated"]), None)
    meta = {
        "traj_run": args.name,
        "scene_run": args.name,
        "source": "import_3mf_scene",
        "source_3mf": str(src),
        "import_command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "calibration_from": args.scene_from,
        "total_frames": n,
        "num_traj_frames": n,
        "key_frames": {k: None for k in KEY_FRAME_NAMES},
        "object_mesh_scale": None,
        "hand_effort_limit": args.hand_effort,
        "traj_id": args.name,
        "objects": objects,
        "manipulated_key": manipulated,
        "static_keys": [o["key"] for o, b in zip(objects, bodies) if b.is_static],
        "flow_files": [],
        "mesh_transfer": ["3mf"],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- validate through the standard loader ----
    from zerofact.scene_spec import describe, load_run_spec
    spec = load_run_spec(run_dir)
    print("=" * 78)
    print(describe(spec))
    print("=" * 78)
    print(f"[3mf] OK -> {run_dir}  ({n} held frames = {n / TARGET_FPS:.1f} s)")
    print(f"[3mf] next: python scripts/convert_assets.py --objects-only --runs {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
