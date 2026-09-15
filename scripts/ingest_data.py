#!/usr/bin/env python3
"""Copy the Video2Sim2Real optimized trajectories + their scene assets into this project.

The source layout (Video2Sim2Real repo) is::

    <V2S2R>/video_tests/Manioulation_data/five_objects/
        contact_opt/leap/retarget_kinova_leap_optimized/<traj_run>/retarget_kinova_leap_optimized.json
        <scene_run>/Scene_reconstruction/{scene_output_final.json,urdfs/*.urdf,transformed_meshes/*.obj}
        <scene_run>/{table_frame_pose.json,camera_frame_pose.json,cam_params.txt}

``<traj_run>`` sometimes carries a bookkeeping suffix (``_record_time_cost``, ``_time_cost``,
``" (Copy)"``) that the matching ``<scene_run>`` does not have; those are stripped to find the scene.

The destination layout (this project, self-contained) is::

    data/runs/<traj_run>/
        retarget_kinova_leap_optimized.json
        run_meta.json                       <- provenance + resolved paths
        scene/scene_output_final.json
        scene/table_frame_pose.json
        scene/camera_frame_pose.json
        scene/cam_params.txt
        scene/urdfs/obj_XXXX.urdf           <- mesh filename rewritten to ../meshes/<file>
        scene/meshes/obj_XXXX_transformed.obj
    data/robot/kinova_leap_description/     <- robot URDF + meshes (shared by every run)

Object meshes are hard-linked when possible (same filesystem, zero extra disk, still a real file
if the source tree moves) and copied otherwise. Pass --copy-meshes to force real copies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_V2S2R = Path("/home/jianuoqiu/Video2Sim2Real")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAJ_SUFFIXES = ["_record_time_cost", "_time_cost", " (Copy)"]
TRAJ_FILE = "retarget_kinova_leap_optimized.json"

SCENE_FILES = [
    ("Scene_reconstruction/scene_output_final.json", "scene_output_final.json"),
    ("table_frame_pose.json", "table_frame_pose.json"),
    ("camera_frame_pose.json", "camera_frame_pose.json"),
    ("cam_params.txt", "cam_params.txt"),
]

# Small per-run signals from the video pipeline: the demo's 3D/2D object flow (used to score how well
# the replay reproduces the demonstrated object motion) and the detected contact/interaction steps.
FLOW_FILES = [
    "3d_flow_point.pkl",
    "obj_flow_traj.npy",
    "flow_lifting_height.npy",
    "contact_step.txt",
    "interaction_step.txt",
    "is_grasping.txt",
]

# Scene objects that must be welded to the world (fixed base, no gravity), keyed by traj run name.
# Ported verbatim from contact_opt/grasp_test/kinova_replay_grasping_test_interaction.py.
STATIC_OBJECTS_BY_RUN = {
    "run_2026-08-09_15-37-32": ["0000"],
}


def scene_run_name(traj_run: str) -> str:
    """Strip the bookkeeping suffixes that only exist on the trajectory folder."""
    name = traj_run
    for suffix in TRAJ_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def link_or_copy(src: Path, dst: Path, force_copy: bool = False) -> str:
    """Hard-link src -> dst when possible, else copy. Returns the method used."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not force_copy:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


MESH_FILENAME_RE = re.compile(r'(<mesh\b[^>]*?\bfilename\s*=\s*)(["\'])(.*?)\2', re.DOTALL)


def _rewrite_mesh_filenames(src_urdf: Path, dst_urdf: Path, resolver) -> list[Path]:
    """Rewrite every ``<mesh filename=...>`` in the URDF *text* (comments and formatting preserved).

    ``resolver(filename) -> (resolved_source_path, new_filename_value)``.
    Returns the resolved source mesh paths in document order.
    """
    text = src_urdf.read_text()
    src_meshes: list[Path] = []

    def _sub(match: re.Match) -> str:
        prefix, quote, filename = match.group(1), match.group(2), match.group(3)
        resolved, new_value = resolver(filename)
        src_meshes.append(resolved)
        return f"{prefix}{quote}{new_value}{quote}"

    new_text = MESH_FILENAME_RE.sub(_sub, text)
    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    dst_urdf.write_text(new_text)
    return src_meshes


def rewrite_object_urdf(src_urdf: Path, dst_urdf: Path, mesh_dir_rel: str = "../meshes") -> list[Path]:
    """Copy an object URDF, rewriting every <mesh filename=...> to <mesh_dir_rel>/<basename>.

    Returns the list of source mesh files the URDF referenced (absolute, resolved). The URDF text is
    edited in place rather than re-serialised so the trailing ``<!-- ... restitution=... -->`` comment
    (which the replay reads for restitution / rolling friction) survives the copy.
    """

    def resolver(filename: str):
        resolved = Path(filename.replace("package://", ""))
        if not resolved.is_absolute():
            resolved = (src_urdf.parent / resolved).resolve()
        return resolved, f"{mesh_dir_rel}/{resolved.name}"

    return _rewrite_mesh_filenames(src_urdf, dst_urdf, resolver)


JOINT_NAME_RE = re.compile(r'(<joint\b[^>]*?\bname\s*=\s*)(["\'])(.*?)\2', re.DOTALL)


def rename_numeric_joints(urdf_text: str, prefix: str = "leap_j") -> tuple[str, dict[str, str]]:
    """Give the LEAP hand's numeric joints (``"0" ... "15"``) valid USD identifier names.

    Isaac Sim's URDF importer cannot keep a joint named ``"7"``: prim names may not start with a
    digit, so it rewrites them ("The path 7 is not a valid usd path, modifying to a_") and, worse,
    every single-digit joint collapses onto the *same* name ``a_`` while ``10..15`` become
    ``a_0..a_5``. Renaming here keeps a 1:1, readable mapping all the way into Isaac Lab.

    Returns the patched URDF text and the ``{original: new}`` mapping.
    """
    mapping: dict[str, str] = {}

    def _sub(match: re.Match) -> str:
        head, quote, name = match.group(1), match.group(2), match.group(3)
        if name.isdigit():
            new_name = f"{prefix}{int(name)}"
            mapping[name] = new_name
            return f"{head}{quote}{new_name}{quote}"
        return match.group(0)

    return JOINT_NAME_RE.sub(_sub, urdf_text), mapping



LIMIT_RE = re.compile(r"(<limit\b[^>]*?/>)", re.DOTALL)
CONT_JOINT_RE = re.compile(r'(<joint\b[^>]*?type\s*=\s*"continuous".*?</joint>)', re.DOTALL)


def strip_continuous_joint_limits(urdf_text: str) -> tuple[str, list[str]]:
    """Remove ``lower=``/``upper=`` from ``<limit>`` of *continuous* joints.

    ``joint_1`` is declared ``type="continuous"`` yet still carries
    ``<limit lower="-1.5708" upper="0.0" .../>`` left over from an older revolute definition.
    Isaac Gym ignores position limits on continuous joints (``hasLimits=False``), but the Isaac Sim
    URDF importer honours the tag and produces ``physics:lowerLimit=-90 deg, upperLimit=0 deg``.
    Five of the optimized trajectories command ``joint_1 > 0``, so leaving it in silently clamps the
    arm. Effort/velocity limits are kept.
    """
    touched: list[str] = []

    def _fix_joint(match: re.Match) -> str:
        block = match.group(0)
        name = re.search(r'name\s*=\s*"([^"]+)"', block)
        new_block = block
        for limit in LIMIT_RE.findall(block):
            cleaned = re.sub(r'\s*(lower|upper)\s*=\s*"[^"]*"', "", limit)
            if cleaned != limit:
                new_block = new_block.replace(limit, cleaned)
                if name:
                    touched.append(name.group(1))
        return new_block

    return CONT_JOINT_RE.sub(_fix_joint, urdf_text), touched


INERTIA_RE = re.compile(r"<inertia\b[^>]*/>", re.DOTALL)
XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def bake_link_inertia_armature(urdf_text: str, armature: float) -> tuple[str, int]:
    """Add *armature* to every link's ``ixx/iyy/izz``.

    Isaac Gym's ``AssetOptions.armature`` (0.01 here) is documented as "the value added to the
    diagonal elements of rigid body inertia tensors" - it is **not** PhysX joint armature. Probing
    the Isaac Gym asset confirms it: the LEAP ``fingertip`` link's URDF ``ixx=3.368e-6`` is reported
    as ``0.010003368``. Since the replay's PD gains (hand: 350/12) were tuned against those inflated
    inertias, the same bump has to exist in the USD or the fingers behave completely differently.
    """
    if armature == 0.0:
        return urdf_text, 0
    count = 0

    # protect XML comments (this URDF has commented-out <inertia> blocks) from the rewrite
    comments: list[str] = []

    def _stash(match: re.Match) -> str:
        comments.append(match.group(0))
        return f"<!--__V2S2R_COMMENT_{len(comments) - 1}__-->"

    urdf_text = XML_COMMENT_RE.sub(_stash, urdf_text)

    def _fix(match: re.Match) -> str:
        nonlocal count
        block = match.group(0)
        for key in ("ixx", "iyy", "izz"):
            m = re.search(rf'{key}\s*=\s*"([^"]+)"', block)
            if m is None:
                continue
            block = block.replace(m.group(0), f'{key}="{float(m.group(1)) + armature!r}"')
        count += 1
        return block

    patched = INERTIA_RE.sub(_fix, urdf_text)
    for i, comment in enumerate(comments):
        patched = patched.replace(f"<!--__V2S2R_COMMENT_{i}__-->", comment)
    return patched, count


def rewrite_robot_urdf(src_urdf: Path, dst_urdf: Path, src_asset_root: Path) -> list[Path]:
    """Copy the robot URDF, rewriting mesh paths (which are relative to <V2S2R>/assets) to be
    relative to the URDF itself so the copied description is self-contained."""

    def resolver(filename: str):
        raw = filename.replace("package://", "")
        candidates = [
            src_asset_root / raw,               # <V2S2R>/assets/kinova_leap_description/meshes/x.STL
            src_urdf.parent / raw,              # relative to the URDF itself
            Path(raw),                          # absolute
        ]
        resolved = next((c.resolve() for c in candidates if c.exists()), None)
        if resolved is None:
            raise FileNotFoundError(f"Cannot resolve robot mesh {filename!r} referenced by {src_urdf}")
        try:
            rel = resolved.relative_to(src_urdf.parent)
        except ValueError:
            rel = Path(resolved.name)
        return resolved, str(rel)

    return _rewrite_mesh_filenames(src_urdf, dst_urdf, resolver)


def load_scene_objects(scene_json: Path) -> list[dict]:
    with open(scene_json) as f:
        scene = json.load(f)
    objects = []
    for i, obj in enumerate(scene.get("objects", [])):
        urdf_path_field = obj.get("urdf_path")
        if not urdf_path_field:
            continue
        objects.append(
            {
                "key": f"scene_obj_{i}",
                "urdf_name": os.path.basename(urdf_path_field),
                "mesh_name": obj.get("mesh_path", ""),
                "pos": obj["pose"]["position_m"],
                "quat_xyzw": obj["pose"]["orientation_quat_xyzw"],
                "is_manipulated": bool(obj.get("is_manipulated", False)),
                "prompt": obj.get("prompt", ""),
            }
        )
    return objects


def resolve_static_keys(objects: list[dict], traj_run: str, scene_run: str | None = None) -> list[str]:
    # the Isaac Gym script keys this table by the *data folder* (scene run); the trajectory folder can
    # carry a bookkeeping suffix, so accept either spelling
    tokens = {str(t).strip().lower() for t in STATIC_OBJECTS_BY_RUN.get(traj_run, [])}
    if scene_run and scene_run != traj_run:
        tokens |= {str(t).strip().lower() for t in STATIC_OBJECTS_BY_RUN.get(scene_run, [])}
    if not tokens:
        return []
    alias_to_key = {}
    for obj in objects:
        stem = os.path.splitext(obj["urdf_name"])[0]     # obj_0000
        suffix = stem.split("_")[-1]                      # 0000
        aliases = {obj["key"], obj["urdf_name"], stem, suffix}
        if suffix.isdigit():
            aliases.add(str(int(suffix)))
        for alias in aliases:
            alias_to_key[alias.lower()] = obj["key"]
    keys = [alias_to_key[t] for t in tokens if t in alias_to_key]
    # never freeze the manipulated object
    manipulated = {o["key"] for o in objects if o["is_manipulated"]}
    return sorted(k for k in keys if k not in manipulated)


def ingest_run(traj_dir: Path, five_objects: Path, dest_root: Path, force_copy: bool) -> dict | None:
    traj_run = traj_dir.name
    traj_json = traj_dir / TRAJ_FILE
    if not traj_json.is_file():
        print(f"[SKIP] {traj_run}: no {TRAJ_FILE}")
        return None

    scene_run = scene_run_name(traj_run)
    scene_src = five_objects / scene_run
    if not scene_src.is_dir():
        print(f"[SKIP] {traj_run}: source scene folder missing ({scene_src})")
        return None

    dest = dest_root / traj_run
    (dest / "scene" / "urdfs").mkdir(parents=True, exist_ok=True)
    (dest / "scene" / "meshes").mkdir(parents=True, exist_ok=True)

    shutil.copy2(traj_json, dest / TRAJ_FILE)

    missing = []
    for rel_src, rel_dst in SCENE_FILES:
        src = scene_src / rel_src
        if not src.is_file():
            missing.append(rel_src)
            continue
        shutil.copy2(src, dest / "scene" / rel_dst)
    if missing:
        print(f"[SKIP] {traj_run}: missing scene files {missing}")
        return None

    flow_src = scene_src / "flow_data"
    flow_present = []
    if flow_src.is_dir():
        (dest / "flow_data").mkdir(parents=True, exist_ok=True)
        for name in FLOW_FILES:
            src = flow_src / name
            if src.is_file():
                shutil.copy2(src, dest / "flow_data" / name)
                flow_present.append(name)

    objects = load_scene_objects(dest / "scene" / "scene_output_final.json")
    urdf_src_dir = scene_src / "Scene_reconstruction" / "urdfs"

    mesh_methods = set()
    for obj in objects:
        src_urdf = urdf_src_dir / obj["urdf_name"]
        if not src_urdf.is_file():
            print(f"[SKIP] {traj_run}: missing object URDF {src_urdf}")
            return None
        dst_urdf = dest / "scene" / "urdfs" / obj["urdf_name"]
        src_meshes = rewrite_object_urdf(src_urdf, dst_urdf)
        for mesh in src_meshes:
            if not mesh.is_file():
                print(f"[SKIP] {traj_run}: missing mesh {mesh}")
                return None
            mesh_methods.add(link_or_copy(mesh, dest / "scene" / "meshes" / mesh.name, force_copy))
        obj["mesh_file"] = f"scene/meshes/{Path(src_meshes[0]).name}" if src_meshes else ""
        obj["urdf_file"] = f"scene/urdfs/{obj['urdf_name']}"

    with open(traj_json) as f:
        traj_data = json.load(f)

    meta = {
        "traj_run": traj_run,
        "scene_run": scene_run,
        "source": {
            "traj_json": str(traj_json),
            "scene_folder": str(scene_src),
        },
        "total_frames": traj_data.get("total_frames"),
        "num_traj_frames": len(traj_data.get("traj", [])),
        "key_frames": {
            k: traj_data.get(k)
            for k in [
                "hand_pose_frame",
                "pregrasp_frame",
                "contact_frame",
                "pre_interaction_frame",
                "interaction_frame",
                "drop_frame",
            ]
        },
        "object_mesh_scale": traj_data.get("object_mesh_scale"),  # None -> replay falls back to the pipeline default (0.9)
        "traj_id": traj_data.get("traj_id"),
        "objects": objects,
        "manipulated_key": next((o["key"] for o in objects if o["is_manipulated"]), None),
        "static_keys": resolve_static_keys(objects, traj_run, scene_run),
        "flow_files": flow_present,
        "mesh_transfer": sorted(mesh_methods),
    }
    with open(dest / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[OK]   {traj_run:<45s} scene={scene_run:<26s} objs={len(objects)} "
        f"frames={meta['num_traj_frames']} "
        f"scale={meta['object_mesh_scale'] if meta['object_mesh_scale'] is not None else '1.0 (gym replay default)'} "
        f"static={meta['static_keys'] or '-'} meshes={'/'.join(sorted(mesh_methods)) or '-'}"
    )
    return meta


def ingest_robot(v2s2r: Path, dest_root: Path, force_copy: bool, armature: float = 0.01) -> dict:
    src_urdf = v2s2r / "assets" / "kinova_leap_description" / "v12_vision.urdf"
    if not src_urdf.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {src_urdf}")
    dst_dir = dest_root / "kinova_leap_description"
    dst_urdf = dst_dir / "v12_vision.urdf"
    src_meshes = rewrite_robot_urdf(src_urdf, dst_urdf, v2s2r / "assets")

    patched, joint_map = rename_numeric_joints(dst_urdf.read_text())
    patched, limit_fixed = strip_continuous_joint_limits(patched)
    patched, inertia_links = bake_link_inertia_armature(patched, armature)
    dst_urdf.write_text(patched)

    # the rewrites are regex-based; if the upstream URDF is ever re-exported with different
    # formatting they would silently become no-ops, so check what they actually did
    problems = []
    if len(joint_map) != 16:
        problems.append(f"renamed {len(joint_map)} numeric joints, expected 16")
    if "joint_1" not in limit_fixed:
        problems.append("joint_1 still carries position limits on a continuous joint")
    if armature and inertia_links == 0:
        problems.append("no link inertia was armature-bumped")
    if problems:
        print(
            "[WARN] robot URDF preparation did not do what it should: "
            + "; ".join(problems)
            + f"\n       check {dst_urdf} against the upstream file before trusting a replay."
        )

    methods = set()
    for mesh in sorted(set(src_meshes)):
        try:
            rel = mesh.relative_to(src_urdf.parent)
        except ValueError:
            rel = Path(mesh.name)          # same fallback the path resolver uses
        methods.add(link_or_copy(mesh, dst_dir / rel, force_copy))

    joint_map_path = dest_root / "joint_name_map.json"
    joint_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(joint_map_path, "w") as f:
        json.dump(
            {
                "comment": "trajectory/URDF joint name -> joint name in the project's URDF and USD",
                "source_urdf": str(src_urdf),
                "map": {**{f"joint_{i}": f"joint_{i}" for i in range(1, 8)}, **joint_map},
            },
            f,
            indent=2,
        )

    print(
        f"[OK]   robot description -> {dst_urdf} ({len(set(src_meshes))} meshes, {'/'.join(sorted(methods))}, "
        f"{len(joint_map)} numeric joints renamed, position limits stripped from {limit_fixed or '-'}, "
        f"armature {armature} baked into {inertia_links} link inertias)"
    )
    return {
        "urdf": str(dst_urdf.relative_to(dest_root.parent)),
        "num_meshes": len(set(src_meshes)),
        "joint_name_map": str(joint_map_path.relative_to(dest_root.parent)),
        "renamed_joints": joint_map,
        "continuous_joints_limit_stripped": limit_fixed,
        "armature_added_to_link_inertia": armature,
        "num_links_inertia_bumped": inertia_links,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2s2r-root", type=Path, default=DEFAULT_V2S2R, help="Video2Sim2Real repo root")
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=None,
        help="folder holding the run_*/retarget_kinova_leap_optimized.json trajectories "
        "(default: <v2s2r>/video_tests/Manioulation_data/five_objects/contact_opt/leap/retarget_kinova_leap_optimized)",
    )
    parser.add_argument("--dest", type=Path, default=PROJECT_ROOT / "data", help="destination data folder")
    parser.add_argument("--runs", nargs="*", default=None, help="only ingest these traj run names")
    parser.add_argument("--copy-meshes", action="store_true", help="copy meshes instead of hard-linking")
    parser.add_argument("--skip-robot", action="store_true", help="do not (re)copy the robot description")
    parser.add_argument(
        "--armature",
        type=float,
        default=0.01,
        help="value added to every link's inertia diagonal, reproducing Isaac Gym's "
        "AssetOptions.armature=0.01 (pass 0 to keep the raw URDF inertias)",
    )
    args = parser.parse_args()

    v2s2r = args.v2s2r_root.resolve()
    five_objects = v2s2r / "video_tests" / "Manioulation_data" / "five_objects"
    traj_root = args.traj_root or (five_objects / "contact_opt" / "leap" / "retarget_kinova_leap_optimized")
    if not traj_root.is_dir():
        print(f"[ERR] trajectory root not found: {traj_root}", file=sys.stderr)
        return 1

    dest_runs = args.dest / "runs"
    dest_runs.mkdir(parents=True, exist_ok=True)

    robot_meta = None
    if not args.skip_robot:
        robot_meta = ingest_robot(v2s2r, args.dest / "robot", args.copy_meshes, args.armature)

    metas = []
    for traj_dir in sorted(p for p in traj_root.iterdir() if p.is_dir()):
        if args.runs and traj_dir.name not in args.runs:
            continue
        meta = ingest_run(traj_dir, five_objects, dest_runs, args.copy_meshes)
        if meta:
            metas.append(meta)

    manifest = {
        "source_repo": str(v2s2r),
        "traj_root": str(traj_root),
        "robot": robot_meta,
        "runs": [
            {
                "traj_run": m["traj_run"],
                "scene_run": m["scene_run"],
                "num_frames": m["num_traj_frames"],
                "num_objects": len(m["objects"]),
                "manipulated_key": m["manipulated_key"],
                "static_keys": m["static_keys"],
                "object_mesh_scale": m["object_mesh_scale"],
            }
            for m in metas
        ],
    }
    manifest_path = args.dest / "manifest.json"
    if args.runs or args.skip_robot:
        # keep previously ingested entries when doing a partial run
        if manifest_path.is_file():
            with open(manifest_path) as f:
                old = json.load(f)
            known = {r["traj_run"] for r in manifest["runs"]}
            manifest["runs"] = manifest["runs"] + [r for r in old.get("runs", []) if r["traj_run"] not in known]
            manifest["runs"].sort(key=lambda r: r["traj_run"])
            manifest["robot"] = manifest["robot"] or old.get("robot")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nIngested {len(metas)} trajectory run(s) -> {dest_runs}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
