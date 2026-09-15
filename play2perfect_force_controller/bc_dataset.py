"""Behaviour-cloning dataset conventions for the play2perfect (KUKA iiwa14 + Sharpa) episodes.

Pure numpy (no Isaac, no torch) so the validator, the index builder and the loader run anywhere.
``collect_episodes.py`` runs :func:`validate_records` inside Kit on the freshly recorded arrays and
``collect_bc_dataset.py`` runs :func:`validate_episode_dir` on the saved files afterwards: an
episode enters the dataset only when both agree.

Episode directory (one cold-start Kit process per episode, distinct seed; see
``collect_bc_dataset.py``)::

    <root>/<problem>/seed_<k>/
        replay_data.npz            all arrays below (+ the force-controller replay layout, so
                                   run_tracking.py / offline_check.py work on these episodes too)
        summary.json               outcome, validation report, key frames, conventions
        rgb_images/<s>.png         demo camera at state index s (clean: no force arrows, goal marker hidden)
        rollout.mp4                the captured frames (review)
        rollout_with_forces.mp4    the same next to the fingertip-force plot (review)
        contact_forces.png / joint_tracking.png / object_tracking.png

Time convention - T frames of 1/60 s (one policy step), S = 2 physics substeps of 1/120 s::

    frame t:  obs_policy[t] -> action[t] -> joint_target[t] (held for both substeps)
              -> joint_pos[t], contact_force[t], object_pos[t], ... = the state at the END of frame t

    state index s:  s = 0 is the reset state (init_* arrays; obs_policy[0] is built from it),
                    s = t + 1 is the state at the end of frame t (joint_pos[t]), i.e. what the
                    policy sees when it chooses action[t + 1].
    rgb_images/<s>.png is the camera at state s; ``image_state_index`` lists the captured s
    (every image_every-th state, s = 0 included).
    ``*_steps`` arrays are [T, S, ...] with the substeps in chronological order; the per-frame
    array of the same name is the last substep.

Fake tactile signals (per frame and per substep, articulation joint order ``joint_names``)::

    applied_torque         N.m  implicit-PD drive torque K (q_target - q) - D qd, clipped to the
                                URDF effort limit: the motor-side "joint torque sensor"
    computed_torque        N.m  the same before clipping
    joint_torque_measured  N.m  PhysX projected joint force: the reaction transmitted through the
                                joint along its axis (drive + gravity + contact loads)
    motor_current          A    applied_torque / motor_kt; motor_kt = effort limit / rated current,
                                rated current arm 10 A, hand 1 A (MOTOR_RATED_CURRENT_A) - FAKE
                                constants (the real motor constants are not published), rescale freely
    joint_cmd_err          rad  joint_target - joint_pos, the state-command difference (PD lead)
    joint_wrench_b         -    [B, 6] incoming joint wrench (force xyz, torque xyz) of every body
                                (``body_names``) in its joint's child frame

Everything else is the force-controller layout: joint_pos/vel/target, action, contact_force
[T, tips, 3] (net fingertip force, world), contact_force_steps [T, S, tips, 3],
contact_object_force_steps [T, S, tips, M, 3] (per filtered object: object, hole, table),
contact_point_w [T, tips, M, 3] (NaN off-contact), object_pos/quat/lin_vel [T, 4 objects, ...]
(object, hole, table, goal_viz), body_pos/quat [T, 6, ...] (palm + fingertips), obs_policy [T, D].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---- fake motor constants ------------------------------------------------------------------------
MOTOR_RATED_CURRENT_A = {"arm": 10.0, "hand": 1.0}     # |current| == rated at the effort limit

# URDF effort limits (iiwa14_left_sharpa_adjusted_restricted.urdf), the fallback if the live
# articulation does not report finite limits
SHARPA_URDF_EFFORT_NM = {
    **{f"iiwa14_joint_{i}": 300.0 for i in range(1, 8)},
    "left_1_thumb_CMC_FE": 3.3, "left_thumb_CMC_AA": 3.3, "left_thumb_MCP_FE": 1.864,
    "left_thumb_MCP_AA": 1.864, "left_thumb_IP": 0.638,
    "left_5_pinky_CMC": 0.5285,
    **{f"left_{n}_{f}_MCP_FE": 1.864 for n, f in ((2, "index"), (3, "middle"), (4, "ring"))},
    "left_pinky_MCP_FE": 1.864,
    **{f"left_{f}_MCP_AA": 1.864 for f in ("index", "middle", "ring", "pinky")},
    **{f"left_{f}_PIP": 0.638 for f in ("index", "middle", "ring", "pinky")},
    **{f"left_{f}_DIP": 0.189369 for f in ("index", "middle", "ring", "pinky")},
}


def is_arm_joint(name: str) -> bool:
    return name.startswith("iiwa14_")


def effort_limits(joint_names: list[str], live: np.ndarray | None) -> np.ndarray:
    """Finite effort limits per joint: the articulation's own where finite and < 1e5, else the URDF table."""
    out = np.zeros(len(joint_names), dtype=np.float64)
    for j, name in enumerate(joint_names):
        val = float(live[j]) if live is not None and j < len(live) else np.nan
        if not np.isfinite(val) or val <= 0.0 or val > 1e5:
            val = SHARPA_URDF_EFFORT_NM.get(name, np.nan)
        out[j] = val
    return out


def motor_torque_constants(joint_names: list[str], effort_limit_nm: np.ndarray) -> np.ndarray:
    """kt [N.m/A] per joint so that |motor_current| reaches the rated current at the effort limit."""
    rated = np.array([MOTOR_RATED_CURRENT_A["arm" if is_arm_joint(n) else "hand"] for n in joint_names])
    return np.asarray(effort_limit_nm, dtype=np.float64) / rated


# ---- acceptance test ---------------------------------------------------------------------------------
SANITY = {
    "min_frames": 60,                  # shorter = unstable start (object fell at reset)
    "max_joint_vel_rad_s": 40.0,       # the 12 reference perfect episodes peak at 13 rad/s
    "max_fingertip_force_n": 400.0,    # ... 142 N
    "max_object_speed_m_s": 4.0,       # ... 1.3 m/s
    "max_free_object_speed_m_s": 2.0,  # part moving with NO fingertip force on it = reset bounce
                                       # (a spawn pose intersecting the table kicks it at 3-6 m/s
                                       # in the first frames; seed 1012) or a flung part; a topple from
                                       # the random spawn orientation peaks at ~1.2 m/s (accepted)
    "min_displacement_m": 0.02,        # the part must have been manipulated: moved >= 2 cm ...
    "min_rotation_deg": 20.0,          # ... or turned >= 20 deg (a screwing leg spawned tilted ABOVE the
                                       # goal height goes down into the hole, so "lifted" is the wrong test)
    "min_image_mean": 5.0,             # a black frame = renderer never produced the camera
    "max_image_mean": 250.0,
    "min_image_change": 1.0,           # mean |first - last| in 8-bit units; 0 = frozen render
}

# play2perfect keypoint test constants (mirrors metrics.task_success_metrics, kept local so this
# module stays torch/controller free)
_KEYPOINT_CORNERS = np.array([(1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)], dtype=np.float64)
_FIXED_SIZE = np.array([0.141, 0.03025, 0.0271])
KEYPOINT_SCALE = 1.5


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = (q[..., i] for i in range(4))
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def keypoint_distance(obj_pos: np.ndarray, obj_quat: np.ndarray, goal_pos: np.ndarray, goal_quat: np.ndarray) -> np.ndarray:
    """The env's insertion measure per frame: max distance between the part's fixed-size box-corner
    keypoints and the goal's ([T])."""
    kp = _KEYPOINT_CORNERS * (0.5 * KEYPOINT_SCALE * _FIXED_SIZE)
    r_obj = _quat_wxyz_to_mat(obj_quat)
    r_goal = _quat_wxyz_to_mat(goal_quat)
    obj_kp = np.asarray(obj_pos, np.float64)[:, None, :] + np.einsum("tij,kj->tki", r_obj, kp)
    goal_kp = np.asarray(goal_pos, np.float64)[None, :] + kp @ r_goal.T
    return np.linalg.norm(obj_kp - goal_kp[None], axis=-1).max(axis=1)


def validate_records(rec: dict, outcome: dict, *, manipulated_idx: int, goal_idx: int,
                     manipulated_contact_idx: int = 0,
                     insertion_tolerance_m: float = 0.01, retract_tolerance_m: float = 0.005,
                     success_steps: int = 10, image_stats: dict | None = None,
                     sanity: dict = SANITY) -> dict:
    """The acceptance test. ``rec`` holds the recorded arrays (numpy), ``outcome`` the collector's
    outcome dict, ``manipulated_idx`` / ``goal_idx`` index ``object_keys`` (recorded poses),
    ``manipulated_contact_idx`` indexes ``contact_object_keys`` (the force matrix), ``image_stats``
    = {"count", "mean_first", "mean_last", "change"} or None when no images were captured.
    Returns ``{"ok", "failed": [...], "checks": {name: {"ok", "value", ...}}}``."""
    checks: dict[str, dict] = {}

    def check(name: str, ok: bool, **info) -> None:
        checks[name] = {"ok": bool(ok), **info}

    term = dict(outcome.get("termination", {}))
    forced = bool(term.pop("forced_max_frames", False))
    others = {k: bool(v) for k, v in term.items() if k != "max_successes" and v}
    check("env_success", outcome.get("insertion_complete", False) and outcome.get("retract_success", False),
          goals=f"{outcome.get('insertion_goals_reached')}/{outcome.get('insertion_goals_total')}",
          retract=bool(outcome.get("retract_success", False)))
    check("termination_clean", bool(term.get("max_successes", False)) and not others and not forced,
          termination=[k for k, v in term.items() if v] + (["forced_max_frames"] if forced else []))
    n = int(rec["joint_pos"].shape[0])
    check("min_frames", n >= sanity["min_frames"], frames=n)

    core = ["joint_pos", "joint_vel", "joint_target", "action", "contact_force_steps", "object_pos", "object_quat", "body_pos"]
    bad = [k for k in core if k in rec and not np.isfinite(np.asarray(rec[k], dtype=np.float64)).all()]
    check("finite", not bad, non_finite=bad)

    kp = np.asarray(rec["keypoints_max_dist"], dtype=np.float64)
    tol_end = retract_tolerance_m * KEYPOINT_SCALE
    check("object_at_goal_end", n > 0 and kp[-1] <= tol_end, final_keypoint_mm=float(kp[-1] * 1000) if n else None,
          tolerance_mm=float(tol_end * 1000))

    # the env's insertion test re-run offline on the recording against the FINAL goal pose
    obj_pos = np.asarray(rec["object_pos"])[:, manipulated_idx]
    obj_quat = np.asarray(rec["object_quat"])[:, manipulated_idx]
    goal_pos = np.asarray(rec["object_pos"])[-1, goal_idx]
    goal_quat = np.asarray(rec["object_quat"])[-1, goal_idx]
    dist = keypoint_distance(obj_pos, obj_quat, goal_pos, goal_quat)
    near = dist <= insertion_tolerance_m * KEYPOINT_SCALE
    run, inserted_frame = 0, None
    for t, ok in enumerate(near):
        run = run + 1 if ok else 0
        if run >= success_steps:
            inserted_frame = int(t)
            break
    check("offline_insertion", inserted_frame is not None and bool(near[-1]),
          inserted_frame=inserted_frame, min_keypoint_mm=float(dist.min() * 1000) if n else None,
          final_keypoint_mm=float(dist[-1] * 1000) if n else None)

    qd = float(np.abs(rec["joint_vel"]).max()) if n else 0.0
    fmax = float(np.linalg.norm(np.asarray(rec["contact_force_steps"]), axis=-1).max()) if n else 0.0
    vmax = float(np.linalg.norm(np.asarray(rec["object_lin_vel"])[:, manipulated_idx], axis=-1).max()) if n else 0.0
    check("physics_sane", qd <= sanity["max_joint_vel_rad_s"] and fmax <= sanity["max_fingertip_force_n"]
          and vmax <= sanity["max_object_speed_m_s"], max_joint_vel_rad_s=qd, max_fingertip_force_n=fmax,
          max_object_speed_m_s=vmax)
    # a part nobody touches must rest (or sit in the hole): fast free motion is a reset bounce
    # (spawn pose intersecting the table / hole) or a part flung out of the hand
    tip_force_on_part = np.linalg.norm(np.asarray(rec["contact_object_force_steps"])[:, :, :, manipulated_contact_idx, :],
                                       axis=-1).max(axis=(1, 2)) if n else np.zeros(0)
    speed = np.linalg.norm(np.asarray(rec["object_lin_vel"])[:, manipulated_idx], axis=-1) if n else np.zeros(0)
    free = tip_force_on_part <= 0.5
    if free.any():
        idx = np.flatnonzero(free)
        v_free = float(speed[idx].max())
        t_free = int(idx[speed[idx].argmax()])
    else:
        v_free, t_free = 0.0, None
    check("object_free_motion", v_free <= sanity["max_free_object_speed_m_s"],
          max_speed_without_fingertip_contact_m_s=v_free, frame=t_free,
          limit_m_s=sanity["max_free_object_speed_m_s"])
    lift = float((obj_pos[:, 2] - obj_pos[0, 2]).max()) if n else 0.0
    disp = float(np.linalg.norm(obj_pos - obj_pos[0], axis=-1).max()) if n else 0.0
    dot = abs(float(np.dot(obj_quat[0], obj_quat[-1]))) if n else 1.0
    rot_deg = float(2.0 * np.degrees(np.arccos(min(1.0, dot))))
    check("object_moved", disp >= sanity["min_displacement_m"] or rot_deg >= sanity["min_rotation_deg"],
          max_displacement_m=disp, rotation_first_to_last_deg=rot_deg, max_lift_m=lift)
    moved = float(np.asarray(rec["joint_pos"]).std(axis=0).max()) if n else 0.0
    check("robot_moved", moved > 1e-3, joint_pos_std_max=moved)

    if image_stats is not None:
        ok_img = (image_stats["count"] >= 1 and sanity["min_image_mean"] <= image_stats["mean_first"] <= sanity["max_image_mean"]
                  and sanity["min_image_mean"] <= image_stats["mean_last"] <= sanity["max_image_mean"]
                  and image_stats["change"] >= sanity["min_image_change"])
        check("images_ok", ok_img, **image_stats)

    failed = [k for k, v in checks.items() if not v["ok"]]
    return {"ok": not failed, "failed": failed, "checks": checks}


def image_stats_from_frames(frames: list[np.ndarray]) -> dict | None:
    if not frames:
        return {"count": 0, "mean_first": 0.0, "mean_last": 0.0, "change": 0.0}
    first, last = frames[0].astype(np.float64), frames[-1].astype(np.float64)
    return {"count": len(frames), "mean_first": float(first.mean()), "mean_last": float(last.mean()),
            "change": float(np.abs(first - last).mean())}


def _read_rgb(path: Path) -> np.ndarray:
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))


def validate_episode_dir(ep_dir: str | Path) -> dict:
    """Re-run the acceptance test on a SAVED episode (independent of the collector's own verdict),
    plus file-level checks: the images on disk match ``image_state_index``."""
    ep_dir = Path(ep_dir)
    npz, summary_path = ep_dir / "replay_data.npz", ep_dir / "summary.json"
    if not npz.is_file() or not summary_path.is_file():
        return {"ok": False, "failed": ["files_missing"], "checks": {"files_missing": {"ok": False}}}
    d = np.load(npz, allow_pickle=True)
    summary = json.loads(summary_path.read_text())
    rec = {k: d[k] for k in d.files}
    keys = [str(k) for k in d["object_keys"]]
    image_stats = None
    img_dir = ep_dir / "rgb_images"
    if "image_state_index" in d.files:
        idx = [int(s) for s in d["image_state_index"]]
        files = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")) if img_dir.is_dir() else []
        on_disk = sorted(int(p.stem) for p in files)
        if on_disk != sorted(idx):
            image_stats = {"count": 0, "mean_first": 0.0, "mean_last": 0.0, "change": 0.0,
                           "mismatch": f"{len(on_disk)} files vs {len(idx)} indices"}
        elif files:
            frames = [_read_rgb(files[0]), _read_rgb(files[-1])]
            image_stats = image_stats_from_frames(frames)
            image_stats["count"] = len(files)
    cfg = summary.get("config", {})
    manip_key = summary.get("manipulated_object", "object")
    contact_keys = [str(k) for k in d["contact_object_keys"]]
    report = validate_records(
        rec, summary["outcome"], manipulated_idx=keys.index(manip_key), goal_idx=keys.index("goal_viz"),
        manipulated_contact_idx=contact_keys.index(manip_key) if manip_key in contact_keys else 0,
        insertion_tolerance_m=float(cfg.get("insertion_success_tolerance", 0.01)),
        retract_tolerance_m=float(cfg.get("retract_success_tolerance", 0.005)),
        success_steps=int(cfg.get("success_steps", 10)), image_stats=image_stats,
    )
    report["episode"] = str(ep_dir)
    return report


# ---- dataset index / loader ----------------------------------------------------------------------
def scan_episodes(root: str | Path, problems: list[str] | None = None) -> list[dict]:
    """Every ``<root>/<problem>/seed_*/summary.json`` with its stored validation verdict."""
    root = Path(root)
    rows = []
    for summary in sorted(root.glob("*/seed_*/summary.json")):
        s = json.loads(summary.read_text())
        problem = summary.parent.parent.name
        if problems and problem not in problems:
            continue
        val = s.get("validation", {})
        o = s["outcome"]
        rows.append({
            "problem": problem, "seed": s.get("config", {}).get("seed"), "dir": str(summary.parent.relative_to(root)),
            "num_frames": o["num_frames"], "sim_time_s": round(o["sim_time_s"], 3),
            "images": len(s.get("bc", {}).get("image_state_index", [])) if "bc" in s else None,
            "max_fingertip_force_on_object_N": round(o["max_fingertip_force_on_object_N"], 2),
            "insertion_goals": f"{o['insertion_goals_reached']}/{o['insertion_goals_total']}",
            "retract_success": o["retract_success"], "validation_ok": bool(val.get("ok", False)),
            "validation_failed": val.get("failed", []),
        })
    return rows


def build_index(root: str | Path, problems: list[str] | None = None, revalidate: bool = False) -> dict:
    """Write ``<root>/dataset_index.json`` (accepted episodes only) and return it."""
    root = Path(root)
    rows = scan_episodes(root, problems)
    if revalidate:
        for r in rows:
            rep = validate_episode_dir(root / r["dir"])
            r["validation_ok"], r["validation_failed"] = rep["ok"], rep["failed"]
    accepted = [r for r in rows if r["validation_ok"]]
    counts = {}
    for r in accepted:
        counts[r["problem"]] = counts.get(r["problem"], 0) + 1
    index = {"root": str(root), "counts": counts, "episodes": accepted,
             "rejected_in_tree": [r for r in rows if not r["validation_ok"]]}
    (root / "dataset_index.json").write_text(json.dumps(index, indent=2))
    return index


def load_episode(ep_dir: str | Path, load_images: bool = False) -> dict:
    """One episode as BC-ready arrays. ``states`` are [T+1, ...] indexed by state s (s = 0 = reset
    state): joint_pos, joint_vel, contact_force (zeros at s=0), object_pos/quat, applied_torque,
    motor_current, joint_cmd_err (zeros at s=0: the drive is not yet loaded at reset); ``actions``
    [T, A]; ``joint_targets`` [T, J]; ``obs_policy`` [T, D] (the policy obs of state s = t);
    ``image_state_index`` and ``image_paths`` (and ``images`` [K, H, W, 3] when load_images)."""
    ep_dir = Path(ep_dir)
    d = np.load(ep_dir / "replay_data.npz", allow_pickle=True)
    summary = json.loads((ep_dir / "summary.json").read_text())
    keys = [str(k) for k in d["object_keys"]]
    manip = keys.index(summary.get("manipulated_object", "object"))
    T, J = d["joint_pos"].shape

    def with_init(per_frame: np.ndarray, init: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(init)[None], np.asarray(per_frame)], axis=0)

    zeros_j = np.zeros((J,), dtype=np.float32)
    init_obj = np.asarray(d["init_object_root_state"])
    states = {
        "joint_pos": with_init(d["joint_pos"], d["init_joint_pos"]),
        "joint_vel": with_init(d["joint_vel"], d["init_joint_vel"]),
        "contact_force": with_init(d["contact_force"], np.zeros_like(d["contact_force"][0])),
        "object_pos": with_init(d["object_pos"][:, manip], init_obj[:3]),
        "object_quat": with_init(d["object_quat"][:, manip], init_obj[3:7]),
        "body_pos": with_init(d["body_pos"], d["body_pos"][0]),      # no reset snapshot of the links: frame-0 value
    }
    for key in ("applied_torque", "computed_torque", "joint_torque_measured", "motor_current", "joint_cmd_err"):
        if key in d.files:
            states[key] = with_init(d[key], zeros_j)
    out = {
        "problem": summary["problem"], "seed": summary["config"]["seed"], "num_frames": T,
        "frame_dt": summary["config"]["sim_time_per_frame_s"],
        "joint_names": [str(n) for n in d["joint_names"]],
        "states": states, "actions": np.asarray(d["action"]), "joint_targets": np.asarray(d["joint_target"]),
        "obs_policy": np.asarray(d["obs_policy"]) if "obs_policy" in d.files else None,
        "image_state_index": [int(s) for s in d["image_state_index"]] if "image_state_index" in d.files else [],
        "key_frames": summary.get("key_frames", {}), "summary": summary,
    }
    ext = summary.get("bc", {}).get("image_format", "png")
    out["image_paths"] = [ep_dir / "rgb_images" / f"{s:06d}.{ext}" for s in out["image_state_index"]]
    if load_images and out["image_paths"]:
        out["images"] = np.stack([_read_rgb(p) for p in out["image_paths"]])
    return out


# ---- review sheet -------------------------------------------------------------------------------
def write_review_sheet(problem_dir: str | Path, out_path: str | Path | None = None, thumb_w: int = 224) -> Path | None:
    """One PNG per problem: a row per accepted episode (seed, frames, peak force) with thumbnails
    at the reset state, first contact, lift, first sub-goal and the final state - a 30-second
    visual skim instead of 50 videos."""
    try:
        import cv2
    except ImportError:
        return None
    problem_dir = Path(problem_dir)
    out_path = Path(out_path) if out_path else problem_dir / "review_sheet.png"
    rows = []
    for summary in sorted(problem_dir.glob("seed_*/summary.json")):
        s = json.loads(summary.read_text())
        if not s.get("validation", {}).get("ok"):
            continue
        ep = summary.parent
        ext = s.get("bc", {}).get("image_format", "png")
        idx = s.get("bc", {}).get("image_state_index", [])
        if not idx:
            continue
        kf = s.get("key_frames", {})
        wanted = {"reset": 0, "contact": kf.get("first_contact_frame"), "lift": kf.get("lift_frame"),
                  "goal 1": kf.get("subgoal_1_frame"), "final": idx[-1] - 1}
        tiles = []
        for label, frame in wanted.items():
            s_idx = 0 if frame is None else min(idx, key=lambda v: abs(v - (int(frame) + 1)))
            img = cv2.imread(str(ep / "rgb_images" / f"{s_idx:06d}.{ext}"))
            if img is None:
                img = np.zeros((thumb_w * 3 // 4, thumb_w, 3), np.uint8)
            h = int(round(img.shape[0] * thumb_w / img.shape[1]))
            img = cv2.resize(img, (thumb_w, h), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{label} s={s_idx}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        strip = np.concatenate(tiles, axis=1)
        o = s["outcome"]
        text = (f"seed {s['config']['seed']}  {o['num_frames']} frames ({o['sim_time_s']:.1f} s)  goals "
                f"{o['insertion_goals_reached']}/{o['insertion_goals_total']}  retract {'ok' if o['retract_success'] else 'NO'}  "
                f"peak |F| on part {o['max_fingertip_force_on_object_N']:.0f} N")
        bar = np.full((22, strip.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        rows.append(np.concatenate([bar, strip], axis=0))
    if not rows:
        return None
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def write_final_frame_grid(problem_dir: str | Path, out_path: str | Path | None = None, thumb_w: int = 320,
                           columns: int = 5, state: str = "final") -> Path | None:
    """One PNG per problem with the FINAL captured frame of every accepted episode (or the reset
    frame with ``state="reset"``): 50 successes fit on one screen, each tile labelled with its seed."""
    try:
        import cv2
    except ImportError:
        return None
    problem_dir = Path(problem_dir)
    out_path = Path(out_path) if out_path else problem_dir / f"{state}_frames_grid.png"
    tiles = []
    for summary in sorted(problem_dir.glob("seed_*/summary.json")):
        s = json.loads(summary.read_text())
        idx = s.get("bc", {}).get("image_state_index", [])
        if not s.get("validation", {}).get("ok") or not idx:
            continue
        ext = s.get("bc", {}).get("image_format", "png")
        s_idx = idx[-1] if state == "final" else idx[0]
        img = cv2.imread(str(summary.parent / "rgb_images" / f"{s_idx:06d}.{ext}"))
        if img is None:
            continue
        h = int(round(img.shape[0] * thumb_w / img.shape[1]))
        img = cv2.resize(img, (thumb_w, h), interpolation=cv2.INTER_AREA)
        o = s["outcome"]
        label = f"seed {s['config']['seed']}  {o['num_frames']}f  {o['insertion_goals_reached']}/{o['insertion_goals_total']}"
        cv2.rectangle(img, (0, 0), (thumb_w, 18), (20, 20, 20), -1)
        cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    grid = np.zeros((rows * h, columns * w, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    cv2.imwrite(str(out_path), grid)
    return out_path
