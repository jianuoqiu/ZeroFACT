#!/usr/bin/env python3
"""Collect "perfect" episodes: roll the released play2perfect checkpoint out in its own Isaac Lab
scene and record everything the force-controller pipeline AND a behaviour-cloning dataset need, in
the ``replay_data.npz`` layout that ``episode.ReplayEpisode`` loads (plus the BC additions
documented in ``bc_dataset.py``: clean RGB frames, the policy observation, per-substep joint
signals and the fake tactile channels - PD drive torque, measured joint torque, motor current,
state-command difference, joint wrenches).

One Kit process per problem (Isaac Sim cannot rebuild a different scene in one process). Each
finished episode lands in::

    <out-dir>/ep_XXXX/            (or <out-dir>/ itself with --flat)
        replay_data.npz      joint states + commands, per-step fingertip forces, contact points,
                             object/goal poses, tracked links, initial state, policy actions +
                             obs, torques / currents / command errors (frame + substep), wrenches
        summary.json         outcome, validation report, key frames, config, initial state,
                             BC conventions
        rgb_images/          <state index>.png demo-camera frames (every --image-every states)
        rollout.mp4          the captured frames (review)
        rollout_with_forces.mp4   the same next to the fingertip-force plot (review)
        contact_forces.png / joint_tracking.png / object_tracking.png

Examples::

    conda activate env_isaaclab
    python play2perfect_force_controller/collect_episodes.py --problem tight_insertion --episodes 5
    python play2perfect_force_controller/collect_episodes.py --problem screwing --episodes 3 --no-render
    # one verified BC episode, cold start, seed 1000 (what collect_bc_dataset.py runs per seed):
    python play2perfect_force_controller/collect_episodes.py --problem tight_insertion --episodes 1 \\
        --seed 1000 --strict --max-attempts 1 --flat --out-dir outputs/play2perfect/bc_dataset/tight_insertion/seed_1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from play2perfect_force_controller.bc_dataset import (  # noqa: E402
    MOTOR_RATED_CURRENT_A, effort_limits, image_stats_from_frames, motor_torque_constants,
    validate_records,
)
from play2perfect_force_controller.p2p_paths import (  # noqa: E402
    EPISODE_ROOT,
    PROBLEMS,
    checkpoint_path,
)
from play2perfect_force_controller.launch import (  # noqa: E402
    add_sim_args, finalize_launcher_args, hard_exit, prepare_display,
)
from play2perfect_force_controller.robot_spec import (  # noqa: E402
    FRAME_DT, MANIPULATED_KEY, PHYSICS_DT, STEPS_PER_FRAME, TRACKED_LINKS,
    register_with_v2s2r_analysis,
)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--problem", choices=PROBLEMS, default="tight_insertion")
parser.add_argument("--episodes", type=int, default=3, help="episodes to record")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="model.pth (default: play2perfect/pretrained_assembly/<problem>/model.pth)")
parser.add_argument("--stochastic", action="store_true",
                    help="sample the policy instead of taking its mean action")
parser.add_argument("--max-frames", type=int, default=1800,
                    help="hard cap per episode in policy steps (30 s); the env's own timeout is "
                         "10 s without progress")
parser.add_argument("--only-perfect", action="store_true",
                    help="keep only episodes in which the policy reached every insertion subgoal "
                         "(and retracted); keep rolling until --episodes of them exist")
parser.add_argument("--strict", action="store_true",
                    help="--only-perfect plus the full bc_dataset.validate_records acceptance test "
                         "(clean termination, object at goal, offline insertion test, physics and "
                         "image sanity); the verdict is stored in summary.json either way")
parser.add_argument("--max-attempts", type=int, default=20,
                    help="give up after this many episodes when --only-perfect finds too few")
parser.add_argument("--min-frames", type=int, default=60,
                    help="discard episodes shorter than this (unstable start, e.g. object fell)")
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--flat", action="store_true",
                    help="write the (single) episode directly into --out-dir instead of ep_0000/")
# --- BC dataset options
parser.add_argument("--image-every", type=int, default=None,
                    help="capture a camera frame every N states (default: 60 / --video-fps, i.e. 2 = 30 Hz)")
parser.add_argument("--no-images", action="store_true", help="render for the videos only, save no PNGs")
parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
parser.add_argument("--goal-marker", choices=["hidden", "translucent", "opaque"], default="hidden",
                    help="the GoalViz target-pose marker in the images (hidden = clean BC images)")
parser.add_argument("--goal-marker-opacity", type=float, default=0.18, help="for --goal-marker translucent")
parser.add_argument("--force-arrows", action="store_true",
                    help="draw the in-scene fingertip force arrows (they appear in the saved images)")
parser.add_argument("--videos", choices=["none", "review", "all"], default="review",
                    help="review = rollout.mp4 + rollout_with_forces.mp4; all = + contact_forces.mp4")
parser.add_argument("--render-mode", choices=["capture", "step"], default="capture",
                    help="capture = render only when a frame is captured (FXAA, fast on a shared GPU); "
                         "step = Isaac Lab's per-substep rendering (TAA, the old video look)")
add_sim_args(parser)

prepare_display(require_window="--gui" in sys.argv)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
render = finalize_launcher_args(args)
checkpoint = args.checkpoint or checkpoint_path(args.problem)
if not checkpoint.is_file():
    raise SystemExit(f"checkpoint not found: {checkpoint} (run play2perfect/download_checkpoints.py)")
if args.strict:
    args.only_perfect = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------------------------
# Isaac-dependent imports
# ---------------------------------------------------------------------------------------------
import torch  # noqa: E402

from v2s2r_isaaclab import analysis  # noqa: E402
from v2s2r_isaaclab.replay import write_video  # noqa: E402

from play2perfect_force_controller.p2p_env import (  # noqa: E402
    RECORDED_OBJECTS, AssemblyBench, SceneOptions, load_policy, make_env_cfg,
)

register_with_v2s2r_analysis()


def obs_to_numpy(obs) -> np.ndarray:
    """The policy observation vector of env 0 as float32 (rl_games hands a dict when the env also
    has critic states)."""
    o = obs["obs"] if isinstance(obs, dict) else obs
    return o[0].detach().cpu().numpy().astype(np.float32).copy()


class EpisodeRecorder:
    """Accumulates one episode frame by frame through ``AssemblyBench.frame_hook`` (state at the end
    of every policy step) and ``substep_hook`` (joint signals after every physics substep)."""

    FRAME_KEYS = [
        "joint_pos", "joint_vel", "joint_target", "action",
        "contact_force", "contact_force_steps", "contact_object_force_steps", "contact_point_w",
        "object_pos", "object_quat", "object_lin_vel", "body_pos", "body_quat",
        "successes", "retract_phase", "keypoints_max_dist",
        # BC additions (see bc_dataset.py)
        "obs_policy", "applied_torque", "computed_torque", "joint_torque_measured", "joint_cmd_err",
        "joint_wrench_b",
        "joint_pos_steps", "joint_vel_steps", "joint_target_steps", "applied_torque_steps",
        "computed_torque_steps", "joint_torque_measured_steps",
    ]
    SUBSTEP_MAP = {          # joint_drive_signals() key -> per-substep record key
        "joint_pos": "joint_pos_steps", "joint_vel": "joint_vel_steps", "joint_target": "joint_target_steps",
        "applied_torque": "applied_torque_steps", "computed_torque": "computed_torque_steps",
        "measured_torque": "joint_torque_measured_steps",
    }

    def __init__(self, env: AssemblyBench, image_every: int, capture_images: bool):
        self.env = env
        self.image_every = max(1, image_every)
        self.capture_images = capture_images and env.camera is not None
        self.pending_obs: np.ndarray | None = None      # the obs the policy acts on in the next frame
        self.obs_dim: int | None = None
        self.begin(env.snapshot_state())

    def begin(self, init_state: dict) -> None:
        self.init_state = init_state
        self.rec: dict[str, list] = {k: [] for k in self.FRAME_KEYS}
        self._sub: list[dict[str, np.ndarray]] = []
        self.frames_rgb: list[np.ndarray] = []
        self.image_state_index: list[int] = []
        self.done = False
        self.termination: dict = {}
        self.t_start = time.time()
        self.timing = {"capture": 0.0, "substep_hook": 0.0, "frame_hook": 0.0}
        if self.capture_images:
            self._capture(0)                               # the reset state

    @property
    def num_frames(self) -> int:
        return len(self.rec["joint_pos"])

    def _capture(self, state_index: int) -> None:
        t0 = time.perf_counter()
        self.env.draw_forces()                             # no-op unless --force-arrows
        frame = self.env.capture_frame()
        if frame is not None:
            self.frames_rgb.append(frame)
            self.image_state_index.append(int(state_index))
        self.timing["capture"] += time.perf_counter() - t0

    def on_substep(self, env: AssemblyBench, substep: int) -> None:
        t0 = time.perf_counter()
        self._sub.append(env.joint_drive_signals())
        self.timing["substep_hook"] += time.perf_counter() - t0

    def on_frame(self, env: AssemblyBench, done: bool) -> None:
        t0 = time.perf_counter()
        self._on_frame(env, done)
        self.timing["frame_hook"] += time.perf_counter() - t0

    def _on_frame(self, env: AssemblyBench, done: bool) -> None:
        r = self.rec
        # three device syncs per frame (scene, tactile, action) instead of ~20: on a GPU shared with a
        # training run every sync waits for a time slice
        scene = env.scene_frame()
        for key in ("joint_pos", "joint_vel", "joint_target", "object_pos", "object_quat", "object_lin_vel",
                    "body_pos", "body_quat", "successes", "retract_phase", "keypoints_max_dist"):
            r[key].append(scene[key])
        r["action"].append(env.last_action[0].cpu().numpy().copy())
        tact = env.tactile_frame()
        net_steps, obj_steps = tact["net_steps"], tact["obj_steps"]
        r["contact_force"].append(net_steps[-1])
        r["contact_force_steps"].append(net_steps)
        r["contact_object_force_steps"].append(obj_steps)
        r["contact_point_w"].append(tact["contact_point_w"])
        # joint-level signals: every substep of this frame (chronological) + the last one per frame
        sub = self._sub if self._sub else [env.joint_drive_signals()]
        self._sub = []
        for src, dst in self.SUBSTEP_MAP.items():
            r[dst].append(np.stack([s[src] for s in sub]))
        last = sub[-1]
        r["applied_torque"].append(last["applied_torque"])
        r["computed_torque"].append(last["computed_torque"])
        r["joint_torque_measured"].append(last["measured_torque"])
        r["joint_cmd_err"].append(last["joint_target"] - last["joint_pos"])
        r["joint_wrench_b"].append(env.joint_wrench_b())
        if self.pending_obs is not None:
            self.obs_dim = int(self.pending_obs.shape[0])
            r["obs_policy"].append(self.pending_obs)
        else:
            r["obs_policy"].append(np.full(self.obs_dim or 1, np.nan, dtype=np.float32))
        # state index s = t + 1 = num_frames (this frame is already appended)
        if self.capture_images and self.num_frames % self.image_every == 0:
            self._capture(self.num_frames)
        if done:
            self.done = True
            self.termination = {k: bool(v[0].item()) for k, v in env._termination_reasons.items()}
            self.termination["env_max_goals"] = int(env.env_max_goals[0].item())
            self.termination["retract_succeeded"] = bool(env.retract_succeeded[0].item())

    def finish(self) -> dict:
        rec = {k: np.stack(v) for k, v in self.rec.items()}
        succ = rec["successes"]
        max_goals = self.termination.get("env_max_goals", int(self.env.env_max_goals[0].item()))
        manip = RECORDED_OBJECTS.index(MANIPULATED_KEY)
        obj_force = np.linalg.norm(rec["contact_object_force_steps"][:, :, :, manip, :], axis=-1).max(axis=(1, 2))
        lift = rec["object_pos"][:, manip, 2] - rec["object_pos"][0, manip, 2]
        key_frames = {
            "first_contact_frame": _first(obj_force > 0.5),
            "lift_frame": _first(lift > 0.02),
        }
        for g in range(1, max_goals + 1):
            key_frames[f"subgoal_{g}_frame"] = _first(succ >= g)
        key_frames["retract_frame"] = _first(rec["retract_phase"])
        outcome = {
            "num_frames": int(self.num_frames),
            "sim_time_s": float(self.num_frames * FRAME_DT),
            "insertion_goals_reached": int(succ.max()) if len(succ) else 0,
            "insertion_goals_total": int(max_goals),
            "insertion_complete": bool(len(succ) and succ.max() >= max_goals),
            "retract_success": bool(self.termination.get("retract_succeeded", False)),
            "termination": {k: v for k, v in self.termination.items()
                            if k not in ("env_max_goals", "retract_succeeded")},
            "max_fingertip_force_on_object_N": float(obj_force.max()) if len(obj_force) else 0.0,
            "manipulated_max_lift_m": float(lift.max()) if len(lift) else 0.0,
        }
        outcome["perfect"] = outcome["insertion_complete"]     # all insertion subgoals reached
        env = self.env
        pa = env.cfg.precise_assembly
        validation = validate_records(
            rec, outcome, manipulated_idx=manip, goal_idx=RECORDED_OBJECTS.index("goal_viz"),
            manipulated_contact_idx=env.contact_object_keys.index(MANIPULATED_KEY),
            insertion_tolerance_m=float(pa.insertion_success_tolerance),
            retract_tolerance_m=float(pa.retract_success_tolerance),
            success_steps=int(getattr(env.cfg.termination, "success_steps", 10)),
            image_stats=image_stats_from_frames(self.frames_rgb) if self.capture_images else None,
        )
        wall = time.time() - self.t_start
        timing = {k: round(v, 2) for k, v in self.timing.items()}
        timing["physics_and_render"] = round(wall - sum(self.timing.values()), 2)
        timing["frames_per_s"] = round(self.num_frames / max(wall, 1e-6), 1)
        return {"records": rec, "key_frames": key_frames, "outcome": outcome, "validation": validation,
                "init_state": self.init_state, "frames_rgb": self.frames_rgb,
                "image_state_index": list(self.image_state_index),
                "wall_time_s": wall, "timing": timing}


def _first(mask) -> int | None:
    idx = np.flatnonzero(np.asarray(mask))
    return int(idx[0]) if idx.size else None


def _write_image(path: Path, rgb: np.ndarray) -> None:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if path.suffix == ".jpg":
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def obs_fields(env: AssemblyBench) -> list[list]:
    """``[[field, size], ...]`` in the order the policy observation is stacked."""
    try:
        from isaacsimenvs.tasks.play.utils.obs_utils import OBS_FIELD_SIZES

        return [[str(f), int(OBS_FIELD_SIZES[f])] for f in env.cfg.obs.obs_list]
    except Exception as exc:                                  # noqa: BLE001
        print(f"[collect] obs field layout unavailable: {exc}", flush=True)
        return []


def save_episode(ep_dir: Path, env: AssemblyBench, ep: dict, run_name: str, meta: dict, fps: int) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    rec = ep["records"]
    joint_names = list(env.robot.joint_names)
    props = env.joint_properties()
    effort = effort_limits(joint_names, props.get("joint_effort_limits"))
    kt = motor_torque_constants(joint_names, effort)
    rec["motor_current"] = (rec["applied_torque"] / kt).astype(np.float32)
    rec["motor_current_steps"] = (rec["applied_torque_steps"] / kt).astype(np.float32)
    rec["joint_cmd_err_steps"] = (rec["joint_target_steps"] - rec["joint_pos_steps"]).astype(np.float32)
    num_frames = int(rec["joint_pos"].shape[0])

    image_format = meta["image_format"]
    if ep["frames_rgb"] and meta["save_images"]:
        img_dir = ep_dir / "rgb_images"
        img_dir.mkdir(exist_ok=True)
        t0 = time.time()
        for s, frame in zip(ep["image_state_index"], ep["frames_rgb"]):
            _write_image(img_dir / f"{s:06d}.{image_format}", frame)
        print(f"[collect] {len(ep['frames_rgb'])} images -> {img_dir} ({time.time() - t0:.1f}s)", flush=True)
    image_state_index = np.array(ep["image_state_index"] if meta["save_images"] else [], dtype=np.int64)

    init = {f"init_{k}": v for k, v in ep["init_state"].items()}
    np.savez_compressed(
        ep_dir / "replay_data.npz",
        joint_names=np.array(joint_names),
        body_names=np.array(list(env.robot.body_names)),
        fingertip_bodies=np.array(env.fingertip_bodies),
        contact_object_keys=np.array(env.contact_object_keys),
        object_keys=np.array(RECORDED_OBJECTS),
        object_names=np.array(RECORDED_OBJECTS),
        tracked_links=np.array(TRACKED_LINKS),
        frame_time_s=(np.arange(1, num_frames + 1) * FRAME_DT).astype(np.float64),
        image_state_index=image_state_index,
        joint_effort_limit_nm=effort,
        motor_kt=kt,
        motor_rated_current_A=np.array([MOTOR_RATED_CURRENT_A["arm" if n.startswith("iiwa14_") else "hand"]
                                        for n in joint_names]),
        **props, **rec, **init,
    )
    validation = ep["validation"]
    summary = {
        "run": run_name,
        "problem": meta["problem"],
        "checkpoint": str(meta["checkpoint"]),
        "episode_index": meta["episode_index"],
        "timestamp": meta["stamp"],
        "config": {
            "sim_time_per_frame_s": FRAME_DT, "physics_dt": PHYSICS_DT,
            "steps_per_frame": STEPS_PER_FRAME, "deterministic_policy": meta["deterministic"],
            "seed": meta["seed"], "render": meta["render"], "cold_start": meta["cold_start"],
            "insertion_success_tolerance": float(env.cfg.precise_assembly.insertion_success_tolerance),
            "retract_success_tolerance": float(env.cfg.precise_assembly.retract_success_tolerance),
            "success_steps": int(getattr(env.cfg.termination, "success_steps", 10)),
        },
        "manipulated_object": MANIPULATED_KEY,
        "key_frames": ep["key_frames"],
        "num_frames": ep["outcome"]["num_frames"],
        "wall_time_s": round(ep["wall_time_s"], 2),
        "timing": ep["timing"],
        "outcome": ep["outcome"],
        "validation": validation,
        "bc": {
            "conventions": "play2perfect_force_controller/bc_dataset.py (module docstring)",
            "image_every": meta["image_every"], "image_fps": fps, "image_format": image_format,
            "image_size_wh": [env.options.camera_width, env.options.camera_height],
            "image_state_index": [int(s) for s in image_state_index],
            "camera": {"eye": list(env.options.camera_eye), "target": list(env.options.camera_target),
                       "prim": "/World/DemoCam"},
            "goal_marker": meta["goal_marker"], "force_arrows_in_images": bool(meta["force_arrows"]),
            "obs_fields": obs_fields(env),
            "fake_tactile": {
                "applied_torque": "implicit-PD drive torque K(q_target-q)-D*qd clipped to the URDF effort limit [N.m]",
                "computed_torque": "the same before clipping [N.m]",
                "joint_torque_measured": "PhysX projected joint force along the joint axis (drive + gravity + contact loads) [N.m]",
                "motor_current": "applied_torque / motor_kt [A]; motor_kt = effort limit / rated current",
                "motor_rated_current_A": MOTOR_RATED_CURRENT_A,
                "joint_cmd_err": "joint_target - joint_pos [rad] (state-command difference)",
                "joint_wrench_b": "[bodies, 6] incoming joint wrench (force xyz, torque xyz), joint child frame",
                "substeps": "*_steps arrays are [T, 2, ...] at 120 Hz; per-frame arrays are the last substep",
            },
        },
        "init_state": {k: np.asarray(v).tolist() for k, v in ep["init_state"].items()},
    }
    analysis.write_summary(summary, ep_dir / "summary.json")

    key_frames = ep["key_frames"]
    analysis.plot_joint_tracking(rec["joint_pos"], rec["joint_target"], joint_names,
                                 ep_dir / "joint_tracking.png", key_frames)
    manip = RECORDED_OBJECTS.index(MANIPULATED_KEY)
    analysis.plot_object_tracking(rec["object_pos"], RECORDED_OBJECTS, ep_dir / "object_tracking.png",
                                  key_frames, manip)
    contact_manip = env.contact_object_keys.index(MANIPULATED_KEY)
    analysis.plot_contact_forces(
        rec["contact_force_steps"], env.fingertip_bodies, ep_dir / "contact_forces.png",
        key_frames=key_frames, object_force_steps=rec["contact_object_force_steps"],
        object_names=env.contact_object_keys, manipulated_idx=contact_manip,
    )
    videos = meta["videos"]
    if ep["frames_rgb"] and videos != "none":
        if write_video(ep["frames_rgb"], ep_dir / "rollout.mp4", fps):
            print(f"[collect] video -> {ep_dir / 'rollout.mp4'}", flush=True)
        every = meta["image_every"]
        # the force plot at the image rate: plot frame j <-> frame j*every; image k <-> state
        # k*every = the end of frame k*every - 1 (image 0 = reset state -> plot frame 0)
        kf_scaled = {k: (None if v is None else int(v) // every) for k, v in key_frames.items()}
        plot_frames = analysis.render_contact_force_video(
            rec["contact_force_steps"][::every], env.fingertip_bodies,
            ep_dir / "contact_forces.mp4" if videos == "all" else None,
            fps=fps, key_frames=kf_scaled, object_force_steps=rec["contact_object_force_steps"][::every],
            object_names=env.contact_object_keys, manipulated_idx=contact_manip,
            sim_time_per_frame=FRAME_DT * every, return_frames=True,
        )
        if plot_frames:
            n_img = len(ep["frames_rgb"])
            sel = [min(max(k - 1, 0), len(plot_frames) - 1) for k in range(n_img)]
            force_rows = [min(max(k * every - 1, 0), num_frames - 1) for k in range(n_img)]
            analysis.render_composite_video(
                ep["frames_rgb"], [plot_frames[j] for j in sel], ep_dir / "rollout_with_forces.mp4", fps=fps,
                run_name=run_name, fingertip_bodies=env.fingertip_bodies,
                contact_force=rec["contact_force"][force_rows], key_frames=kf_scaled,
                sim_time_per_frame=FRAME_DT * every,
                force_vis_scale=env.options.force_vis_scale if env.force_markers is not None else None,
                manipulated_name=MANIPULATED_KEY,
            )


def main() -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir or (EPISODE_ROOT / args.problem / stamp)
    out_root.mkdir(parents=True, exist_ok=True)
    deterministic = not args.stochastic
    image_every = args.image_every or max(1, round(1.0 / (FRAME_DT * args.video_fps)))
    fps = round(1.0 / (FRAME_DT * image_every))
    if args.flat and args.episodes != 1:
        raise SystemExit("--flat records exactly one episode (--episodes 1)")

    capture_only = args.render_mode == "capture"
    cfg = make_env_cfg(args.problem, seed=args.seed, sim_device=args.device, render=render,
                       antialiasing="FXAA" if capture_only else "TAA")
    options = SceneOptions(
        render=render, max_contact_points=args.max_contact_points,
        goal_marker_visible=args.goal_marker != "hidden",
        goal_marker_opacity=1.0 if args.goal_marker == "opaque" else args.goal_marker_opacity,
        force_arrows=args.force_arrows, render_in_step=not capture_only,
    )
    env = AssemblyBench(cfg, options)
    player, wrapped = load_policy(env, checkpoint, deterministic=deterministic, rl_device=args.device,
                                  seed=args.seed)

    print("=" * 90)
    print(f"problem      : {args.problem}   checkpoint {checkpoint}")
    print(f"policy       : {'deterministic (mean action)' if deterministic else 'stochastic'}, "
          f"60 Hz, {env.cfg.decimation} physics steps per frame")
    print(f"joints       : {len(env.robot.joint_names)}  fingertips {env.fingertip_bodies}")
    print(f"contacts vs  : {env.contact_object_keys}")
    print(f"images       : {'off' if (not render or args.no_images) else f'every {image_every} states ({fps} Hz), {args.image_format}, goal marker {args.goal_marker}, arrows {args.force_arrows}, render {args.render_mode}'}")
    print(f"acceptance   : {'strict' if args.strict else ('perfect' if args.only_perfect else 'all')}")
    print(f"output -> {out_root}")
    print("=" * 90, flush=True)

    player.reset()
    obs = player.env_reset(wrapped)
    if env.camera is not None:
        for _ in range(5):
            env.capture_frame()                  # let the RTX textures load before frame 0
    recorder = EpisodeRecorder(env, image_every, capture_images=render)
    recorder.pending_obs = obs_to_numpy(obs)
    env.frame_hook = recorder.on_frame
    env.substep_hook = recorder.on_substep

    results = []
    rejected = []
    ep_index = 0
    attempts = 0
    t0 = time.time()
    while ep_index < args.episodes and attempts < args.max_attempts:
        action = player.get_action(obs, is_deterministic=deterministic)
        obs, _rew, dones, _infos = player.env_step(wrapped, action)
        recorder.pending_obs = obs_to_numpy(obs)          # what the policy sees for the next frame
        if recorder.num_frames % 120 == 0:
            print(f"[collect] ep {ep_index} frame {recorder.num_frames:5d}  goals "
                  f"{int(env._successes[0].item())}  ({time.time() - t0:6.1f}s)", flush=True)
        forced = recorder.num_frames >= args.max_frames and not recorder.done
        if not (recorder.done or forced):
            continue
        attempts += 1
        if forced:
            # the env did not finish on its own: reset it by hand and mark the episode
            recorder.termination = {"forced_max_frames": True, "env_max_goals": int(env.env_max_goals[0].item()),
                                    "retract_succeeded": bool(env.retract_succeeded[0].item())}
            env.frame_hook = None
            env.substep_hook = None
            obs = player.env_reset(wrapped)
            recorder.pending_obs = obs_to_numpy(obs)
            env.frame_hook = recorder.on_frame
            env.substep_hook = recorder.on_substep
        ep = recorder.finish()
        outcome = ep["outcome"]
        val = ep["validation"]
        perfect = outcome["perfect"] and outcome["retract_success"]
        reason = None
        if outcome["num_frames"] < args.min_frames:
            reason = f"only {outcome['num_frames']} frames ({outcome['termination']}) - unstable start"
        elif args.only_perfect and not perfect:
            reason = (f"goals {outcome['insertion_goals_reached']}/{outcome['insertion_goals_total']}, retract "
                      f"{'ok' if outcome['retract_success'] else 'no'}, ended by "
                      f"{[k for k, v in outcome['termination'].items() if v]} - not perfect")
        elif args.strict and not val["ok"]:
            reason = f"perfect but failed validation {val['failed']}: " + "; ".join(
                f"{k}={ {kk: vv for kk, vv in val['checks'][k].items() if kk != 'ok'} }" for k in val["failed"])
        if reason is not None:
            print(f"[collect] attempt {attempts}: {reason} - discarded", flush=True)
            rejected.append({"attempt": attempts, "reason": reason, "outcome": outcome, "validation_failed": val["failed"]})
            if perfect and args.flat and not val["ok"]:
                # a success the acceptance test refused (force spike, reset kick, ...): keep the arrays
                # and plots next to the dataset so the verdict can be reviewed without re-running
                diag_dir = out_root.parent / "rejected_perfect" / out_root.name
                try:
                    save_episode(diag_dir, env, {**ep, "frames_rgb": [], "image_state_index": []},
                                 f"{args.problem}_{stamp}_seed{args.seed}_rejected", {
                                     "problem": args.problem, "checkpoint": checkpoint, "episode_index": ep_index,
                                     "stamp": stamp, "deterministic": deterministic, "seed": args.seed, "render": render,
                                     "cold_start": attempts == 1, "image_every": image_every,
                                     "image_format": args.image_format, "save_images": False,
                                     "goal_marker": args.goal_marker, "force_arrows": args.force_arrows, "videos": "none",
                                 }, fps)
                    print(f"[collect] rejected-but-perfect diagnostics -> {diag_dir}", flush=True)
                except Exception as exc:                  # noqa: BLE001
                    print(f"[collect] could not save diagnostics: {exc}", flush=True)
        else:
            run_name = (f"{args.problem}_{stamp}_seed{args.seed}" if args.flat
                        else f"{args.problem}_{stamp}_ep{ep_index:04d}")
            ep_dir = out_root if args.flat else out_root / f"ep_{ep_index:04d}"
            save_episode(ep_dir, env, ep, run_name, {
                "problem": args.problem, "checkpoint": checkpoint, "episode_index": ep_index,
                "stamp": stamp, "deterministic": deterministic, "seed": args.seed, "render": render,
                "cold_start": attempts == 1, "image_every": image_every, "image_format": args.image_format,
                "save_images": not args.no_images, "goal_marker": args.goal_marker,
                "force_arrows": args.force_arrows, "videos": args.videos,
            }, fps)
            term = [k for k, v in outcome["termination"].items() if v]
            print(f"[collect] ep {ep_index}: {outcome['num_frames']} frames, goals "
                  f"{outcome['insertion_goals_reached']}/{outcome['insertion_goals_total']}, "
                  f"retract {'ok' if outcome['retract_success'] else 'no'}, ended by {term}, "
                  f"max |F| on object {outcome['max_fingertip_force_on_object_N']:.1f} N, validation "
                  f"{'OK' if val['ok'] else 'FAILED ' + str(val['failed'])}, {len(ep['frames_rgb'])} images, "
                  f"timing {ep['timing']} -> {ep_dir}", flush=True)
            results.append({"episode": run_name, "dir": str(ep_dir), "validation_ok": val["ok"],
                            "validation_failed": val["failed"], **outcome})
            ep_index += 1
        # the env auto-reset inside step(); the policy's recurrent state starts fresh too
        player.reset()
        recorder.begin(env.snapshot_state())

    collection = {
        "problem": args.problem, "checkpoint": str(checkpoint), "timestamp": stamp,
        "deterministic_policy": deterministic, "seed": args.seed, "attempts": attempts,
        "acceptance": "strict" if args.strict else ("perfect" if args.only_perfect else "all"),
        "episodes": results, "rejected": rejected, "wall_time_s": round(time.time() - t0, 1),
    }
    (out_root / "collection_summary.json").write_text(json.dumps(collection, indent=2))
    print("\n" + "=" * 90)
    print(f"[collect] {len(results)} episodes in {attempts} attempts, {collection['wall_time_s']}s")
    for r in results:
        print(f"[collect]   {r['episode']:<48s} frames {r['num_frames']:5d}  goals "
              f"{r['insertion_goals_reached']}/{r['insertion_goals_total']}  "
              f"perfect={'yes' if r['perfect'] else 'no '}  retract={'yes' if r['retract_success'] else 'no'}  "
              f"valid={'yes' if r['validation_ok'] else 'no'}")
    print(f"[collect] outputs -> {out_root}")
    print("=" * 90, flush=True)
    return 0


if __name__ == "__main__":
    status = 0
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        hard_exit(simulation_app, status)
