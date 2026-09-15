#!/usr/bin/env python3
"""Validate the Isaac Lab replay against the original Isaac Gym replay, run for run.

It patches ``Video2Sim2Real/contact_opt/optimized_replay.py`` **into a temporary copy** (never in
place) so it runs headless and dumps the per-frame object poses and joint states, executes that copy
with an Isaac Gym interpreter, then compares the result against this project's ``replay_data.npz``.

Isaac Gym needs a PyTorch built for this GPU. On this machine ``env_isaaclab``'s torch has no sm_120
kernels for the RTX 5090, but ``video2real`` does - hence ``--gym-python``.

Example::

    python scripts/compare_with_isaacgym.py --run run_2026-08-11_17-29-12

Outputs (under ``outputs/_comparisons/<run>/<stamp>/``): ``gym_reference.npz``,
``comparison.json``, ``comparison.png`` and the patched script that produced the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GYM_PYTHON = "/home/jianuoqiu/anaconda3/envs/video2real/bin/python"
DEFAULT_V2S2R = Path("/home/jianuoqiu/Video2Sim2Real")


def patch_reference(src_path: Path, out_npz: Path, traj_json: Path) -> str:
    """Return a headless, non-destructive variant of ``optimized_replay.py``."""
    src = src_path.read_text()
    replacements = [
        (
            'optimized_traj_path = os.path.join(args.data_folder.parent, "contact_opt", args.hand_type , "retarget_kinova_leap_optimized", args.data_folder.name, "retarget_kinova_leap_optimized.json")',
            f'optimized_traj_path = r"{traj_json}"  # PATCHED: replay exactly the trajectory the port replayed',
        ),
        (
            """viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")""",
            "viewer = None  # PATCHED: headless comparison run",
        ),
        (
            "    if i == 0:\n        gym.clear_lines(viewer)\n    gym.add_lines(viewer, env, 3, vertices, colors)",
            "    pass  # PATCHED: no viewer lines",
        ),
        ("gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)", "pass  # PATCHED"),
        ("while not gym.query_viewer_has_closed(viewer):", "while True:"),
        (
            """        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

        gym.render_all_camera_sensors(sim)""",
            "        pass  # PATCHED: no rendering",
        ),
        (
            """record_robot_joints = []
record_target_joints = []
record_hand_poses = []""",
            """record_robot_joints = []
record_target_joints = []
record_hand_poses = []
record_object_pos = []
record_object_quat = []
_num_object_actors = len(object_assets)
_object_body_indices = [
    gym.get_actor_rigid_body_index(envs[0], gym.find_actor_handle(envs[0], f"object_{i}"), 0, gymapi.DOMAIN_ENV)
    for i in range(_num_object_actors)
]
print("PATCHED object body indices:", _object_body_indices)""",
        ),
        (
            """    retargeted_index += 1
    if retargeted_index >= retargeted_traj.shape[0]:
        break""",
            """    gym.refresh_rigid_body_state_tensor(sim)
    record_object_pos.append(rb_states[0, _object_body_indices, :3].cpu().numpy().copy())
    record_object_quat.append(rb_states[0, _object_body_indices, 3:7].cpu().numpy().copy())

    retargeted_index += 1
    if retargeted_index >= retargeted_traj.shape[0]:
        break""",
        ),
        (
            'hand_pose_traj_path = data_folder / "hand_pose_traj.json"',
            f'''import numpy as _np
_np.savez_compressed(
    r"{out_npz}",
    object_pos=_np.stack(record_object_pos),
    object_quat=_np.stack(record_object_quat),
    joint_pos=_np.stack(record_robot_joints),
    joint_target=_np.stack(record_target_joints),
    dof_names=_np.array(dof_names),
)
print("PATCHED saved reference trajectory ->", r"{out_npz}")
raise SystemExit(0)

hand_pose_traj_path = data_folder / "hand_pose_traj.json"''',
        ),
    ]

    for old, new in replacements:
        if old not in src:
            raise RuntimeError(
                "compare_with_isaacgym.py could not patch optimized_replay.py - the reference script "
                f"changed. Missing snippet:\n{old[:200]}"
            )
        src = src.replace(old, new)
    return src


def compare(gym_npz: Path, lab_npz: Path, joint_map: dict[str, str]) -> dict:
    gym = np.load(gym_npz, allow_pickle=True)
    lab = np.load(lab_npz, allow_pickle=True)

    gym_names = [str(x) for x in gym["dof_names"]]
    lab_names = [str(x) for x in lab["joint_names"]]
    # the replay stores, per simulation joint, the trajectory joint name it was driven from
    if "traj_joint_names" in lab.files:
        traj_order = [str(x) for x in lab["traj_joint_names"]]
    else:
        sim_to_traj = {v: k for k, v in joint_map.items()}
        traj_order = [sim_to_traj[n] for n in lab_names]
    cols = [gym_names.index(n) for n in traj_order]

    T = min(len(gym["object_pos"]), len(lab["object_pos"]))
    gp, lp = gym["object_pos"][:T], lab["object_pos"][:T]
    gj, lj = gym["joint_pos"][:T][:, cols], lab["joint_pos"][:T]
    gt, lt = gym["joint_target"][:T][:, cols], lab["joint_target"][:T]

    # Isaac Gym stores object quaternions as xyzw, this project as wxyz
    gq = gym["object_quat"][:T]
    lq = lab["object_quat"][:T]
    gq_wxyz = np.concatenate([gq[..., 3:4], gq[..., 0:3]], axis=-1)

    def _angle_between(q1, q2):
        dot = np.abs(np.sum(q1 * q2, axis=-1)).clip(-1.0, 1.0)
        return 2.0 * np.arccos(dot)

    object_names = [str(x) for x in lab["object_names"]]
    objects = {}
    for i, name in enumerate(object_names[: min(gp.shape[1], lp.shape[1])]):
        d = np.linalg.norm(gp[:, i] - lp[:, i], axis=1)
        ang = np.degrees(_angle_between(gq_wxyz[:, i], lq[:, i]))
        objects[name] = {
            "position_error_mean_m": float(d.mean()),
            "position_error_final_m": float(d[-1]),
            "position_error_max_m": float(d.max()),
            "gym_displacement_m": float(np.linalg.norm(gp[-1, i] - gp[0, i])),
            "lab_displacement_m": float(np.linalg.norm(lp[-1, i] - lp[0, i])),
            "gym_max_lift_m": float((gp[:, i, 2] - gp[0, i, 2]).max()),
            "lab_max_lift_m": float((lp[:, i, 2] - lp[0, i, 2]).max()),
            "orientation_error_mean_deg": float(ang.mean()),
            "orientation_error_max_deg": float(ang.max()),
        }

    joint_err = np.abs(gj - lj)
    return {
        "frames_compared": int(T),
        "objects": objects,
        "joint_position_error_mean_rad": float(joint_err.mean()),
        "joint_position_error_max_rad": float(joint_err.max()),
        "joint_position_error_per_joint_max_rad": dict(
            zip(lab_names, [float(v) for v in joint_err.max(axis=0)])
        ),
        "commanded_target_max_difference_rad": float(np.abs(gt - lt).max()),
        "gym_reference": str(gym_npz),
        "isaaclab_run": str(lab_npz),
    }


def plot(gym_npz: Path, lab_npz: Path, out_png: Path, object_names: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gym = np.load(gym_npz, allow_pickle=True)
    lab = np.load(lab_npz, allow_pickle=True)
    gp, lp = gym["object_pos"], lab["object_pos"]
    T = min(len(gp), len(lp))
    n_obj = min(gp.shape[1], lp.shape[1])

    fig, axes = plt.subplots(n_obj, 3, figsize=(15, 3.4 * n_obj), squeeze=False, constrained_layout=True)
    for i in range(n_obj):
        for k, axis in enumerate("xyz"):
            ax = axes[i][k]
            ax.plot(np.arange(T), gp[:T, i, k], color="tab:red", label="Isaac Gym (reference)")
            ax.plot(np.arange(T), lp[:T, i, k], color="tab:blue", linestyle="--", label="Isaac Lab (this port)")
            ax.set_title(f"{object_names[i] if i < len(object_names) else i} {axis} (m)")
            ax.set_xlabel("trajectory frame")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=9)
    fig.suptitle("Object motion: Isaac Gym reference vs Isaac Lab port", fontsize=14)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="trajectory run name (must exist under data/runs)")
    parser.add_argument("--v2s2r-root", type=Path, default=DEFAULT_V2S2R)
    parser.add_argument("--gym-python", default=DEFAULT_GYM_PYTHON, help="interpreter with Isaac Gym + sm_120 torch")
    parser.add_argument("--lab-npz", type=Path, default=None, help="replay_data.npz (default: newest for the run)")
    parser.add_argument("--object-mesh-scale", type=float, default=None,
                        help="scale passed to the Isaac Gym reference (default: the run's own value)")
    parser.add_argument(
        "--display",
        default=None,
        help="DISPLAY for the Isaac Gym process (default: the one this user can actually reach)",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    from zerofact.runtime import prepare_display
    from zerofact.scene_spec import load_run_spec

    # Isaac Gym needs a usable X display for its graphics device, and ~/.bashrc points DISPLAY at
    # another user's session on this box.
    display = args.display or prepare_display() or ":1"

    run_dir = PROJECT_ROOT / "data" / "runs" / args.run
    if not (run_dir / "run_meta.json").is_file():
        raise SystemExit(f"unknown run: {args.run}")
    with open(run_dir / "run_meta.json") as f:
        meta = json.load(f)
    spec = load_run_spec(run_dir, object_mesh_scale=args.object_mesh_scale)

    lab_npz = args.lab_npz
    if lab_npz is None:
        candidates = sorted((PROJECT_ROOT / "outputs" / args.run).glob("*/replay_data.npz"))
        candidates += sorted((PROJECT_ROOT / "outputs" / "_batches").glob(f"*/{args.run}/replay_data.npz"))
        if not candidates:
            raise SystemExit(
                f"no Isaac Lab result for {args.run}. Run: python scripts/replay_trajectory.py --run {args.run}"
            )
        lab_npz = max(candidates, key=lambda p: p.stat().st_mtime)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "_comparisons" / args.run / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    gym_npz = out_dir / "gym_reference.npz"
    script_path = out_dir / "gym_reference_replay.py"
    reference = args.v2s2r_root / "contact_opt" / "optimized_replay.py"
    traj_json = run_dir / "retarget_kinova_leap_optimized.json"
    script_path.write_text(patch_reference(reference, gym_npz, traj_json))
    if meta["scene_run"] != args.run:
        print(
            f"[compare] note: trajectory folder {args.run!r} maps to scene {meta['scene_run']!r}; "
            f"the reference is pointed at {traj_json} explicitly."
        )

    data_folder = args.v2s2r_root / "video_tests" / "Manioulation_data" / "five_objects" / meta["scene_run"]
    cmd = [
        args.gym_python,
        "-u",
        str(script_path),
        "--data_folder",
        str(data_folder),
        "--object_mesh_scale",
        str(spec.object_mesh_scale),
    ]
    env = dict(os.environ, DISPLAY=display)
    print(f"[compare] running the Isaac Gym reference: {' '.join(cmd)}", flush=True)
    log_path = out_dir / "gym_reference.log"
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(args.v2s2r_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n[compare] Isaac Gym reference timed out after {args.timeout}s\n")
            returncode = -1
    if returncode != 0 or not gym_npz.is_file():
        print(f"[compare] Isaac Gym reference failed (exit {returncode}); see {log_path}")
        return 1

    joint_map = json.load(open(PROJECT_ROOT / "data" / "robot" / "joint_name_map.json"))["map"]
    result = compare(gym_npz, lab_npz, joint_map)
    result["run"] = args.run
    result["object_mesh_scale"] = spec.object_mesh_scale
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    lab = np.load(lab_npz, allow_pickle=True)
    plot(gym_npz, lab_npz, out_dir / "comparison.png", [str(x) for x in lab["object_names"]])

    print("\n" + "=" * 90)
    print(f"[compare] {args.run}  ({result['frames_compared']} frames, object scale {spec.object_mesh_scale})")
    print(
        f"[compare] commanded targets identical: "
        f"{'yes' if result['commanded_target_max_difference_rad'] == 0 else result['commanded_target_max_difference_rad']}"
    )
    print(
        f"[compare] joint position error   : mean {result['joint_position_error_mean_rad']:.5f} rad, "
        f"max {result['joint_position_error_max_rad']:.5f} rad"
    )
    for name, stats in result["objects"].items():
        print(
            f"[compare] {name:<10s} pos error : mean {stats['position_error_mean_m'] * 1000:.1f} mm, "
            f"max {stats['position_error_max_m'] * 1000:.1f} mm   "
            f"(moved: gym {stats['gym_displacement_m'] * 1000:.1f} mm vs lab {stats['lab_displacement_m'] * 1000:.1f} mm; "
            f"lift: gym {stats['gym_max_lift_m'] * 1000:.1f} mm vs lab {stats['lab_max_lift_m'] * 1000:.1f} mm; "
            f"rot err mean {stats['orientation_error_mean_deg']:.2f} deg)"
        )
    print(f"[compare] outputs -> {out_dir}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
