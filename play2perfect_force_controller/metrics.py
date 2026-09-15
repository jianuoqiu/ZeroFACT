"""Force-tracking metrics on full vectors, |f_ref_vec - f_meas_vec| (numpy only).

New rollouts record the per-step reference and filtered measured force vectors directly
(``f_ref_vec_steps`` / ``f_meas_vec_steps`` in ``controller_data.npz``). For rollouts recorded
before that, :func:`load_force_vectors` reconstructs both deterministically:

* the reference vectors by re-running the pretend policy + ChunkTracker with the run's own
  controller config (pure interpolation of episode data - bit-reproducible), and
* the filtered measured vectors by re-applying the middle layer's EMA + one-step sensor delay to
  the raw per-step contact forces the run recorded.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .config import ARM_JOINT_NAMES, ControllerConfig, ForceLawConfig, PolicyConfig, ReferenceConfig
from .episode import ReplayEpisode
from .middle_layer import HybridForceMiddleLayer
from .policy import ChunkedReplayPolicy


def vector_force_metrics(
    f_ref_vec: np.ndarray,           # [T, S, 4, 3] (or [N, 4, 3]) predicted force vectors
    f_meas_vec: np.ndarray,          # same shape: measured (filtered) force vectors
    fingertip_bodies: list[str],
    engage_threshold_n: float = 0.2,
    direction_min_n: float = 1.0,    # angle error only counted when both vectors exceed this
) -> dict:
    """Per-fingertip vector tracking error |f_ref - f_meas|, split into magnitude and direction.

    ``rmse_vec_N`` is the headline number: the RMS of the full vector error norm over the steps
    where contact is commanded. ``bias_mag_N`` (signed, negative = too weak) and
    ``angle_mean_deg`` show how much of it is wrong *strength* vs wrong *direction*.
    """
    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]).astype(np.float64)   # [N, 4, 3]
    fm = f_meas_vec.reshape(-1, *f_meas_vec.shape[-2:]).astype(np.float64)
    err_norm = np.linalg.norm(fm - fr, axis=-1)                            # [N, 4]
    ref_mag = np.linalg.norm(fr, axis=-1)
    meas_mag = np.linalg.norm(fm, axis=-1)
    engaged = ref_mag >= engage_threshold_n
    mag_err = meas_mag - ref_mag

    out = {
        "bodies": list(fingertip_bodies),
        "engage_threshold_n": engage_threshold_n,
        "direction_min_n": direction_min_n,
        "engaged_fraction": [],
        "rmse_vec_N": [], "mean_vec_N": [], "max_vec_N": [],
        "rmse_mag_N": [], "bias_mag_N": [],
        "angle_mean_deg": [], "angle_p95_deg": [],
        # signed per-world-axis error f_meas - f_ref over engaged steps, [tips][x, y, z]
        "bias_xyz_N": [], "rmse_xyz_N": [],
    }
    for i in range(err_norm.shape[1]):
        mask = engaged[:, i]
        out["engaged_fraction"].append(float(mask.mean()))
        if mask.any():
            e = err_norm[mask, i]
            out["rmse_vec_N"].append(float(np.sqrt((e**2).mean())))
            out["mean_vec_N"].append(float(e.mean()))
            out["max_vec_N"].append(float(e.max()))
            out["rmse_mag_N"].append(float(np.sqrt((mag_err[mask, i] ** 2).mean())))
            out["bias_mag_N"].append(float(mag_err[mask, i].mean()))
            axis_err = (fm - fr)[mask, i, :]                       # [n, 3]
            out["bias_xyz_N"].append([float(v) for v in axis_err.mean(axis=0)])
            out["rmse_xyz_N"].append([float(v) for v in np.sqrt((axis_err ** 2).mean(axis=0))])
        else:
            for key in ["rmse_vec_N", "mean_vec_N", "max_vec_N", "rmse_mag_N", "bias_mag_N"]:
                out[key].append(0.0)
            out["bias_xyz_N"].append([0.0, 0.0, 0.0])
            out["rmse_xyz_N"].append([0.0, 0.0, 0.0])
        dir_mask = mask & (ref_mag[:, i] >= direction_min_n) & (meas_mag[:, i] >= direction_min_n)
        if dir_mask.any():
            cos = np.sum(fr[dir_mask, i] * fm[dir_mask, i], axis=-1) / (
                ref_mag[dir_mask, i] * meas_mag[dir_mask, i]
            )
            angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            out["angle_mean_deg"].append(float(angle.mean()))
            out["angle_p95_deg"].append(float(np.percentile(angle, 95)))
        else:
            out["angle_mean_deg"].append(0.0)
            out["angle_p95_deg"].append(0.0)
    return out


def contact_point_metrics(
    p_ref: np.ndarray,               # [T, S, 4, 3] (or [N, 4, 3]) predicted contact points
    p_meas: np.ndarray,              # same shape: measured (filtered) pad centroids
    f_ref_vec: np.ndarray,           # same shape: predicted force vectors (for engagement + split)
    fingertip_bodies: list[str],
    engage_threshold_n: float = 0.2,
) -> dict:
    """Per-fingertip contact-point tracking error |p_ref - p_meas|, in millimetres.

    Counted only over steps where contact is commanded (``|f_ref| >= engage_threshold_n``) AND
    both points exist — a pad that is not touching reports no centroid, so those steps say
    nothing about tracking. ``measured_fraction`` is how often a commanded contact actually
    produced a measurable one, which is itself a tracking result.

    The error is split the same way the law treats it: ``normal`` is the component along
    ``f_ref`` (depth of press, regulated by the force channel) and ``tangential`` is the rest
    (where on the surface the finger sits, regulated by the contact-point channel).
    """
    pr = p_ref.reshape(-1, *p_ref.shape[-2:]).astype(np.float64)
    pm = p_meas.reshape(-1, *p_meas.shape[-2:]).astype(np.float64)
    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]).astype(np.float64)
    ref_mag = np.linalg.norm(fr, axis=-1)
    engaged = ref_mag >= engage_threshold_n
    both = np.isfinite(pr).all(axis=-1) & np.isfinite(pm).all(axis=-1)
    delta = pr - pm

    out = {
        "bodies": list(fingertip_bodies),
        "engage_threshold_n": engage_threshold_n,
        "engaged_fraction": [], "measured_fraction": [],
        "mean_mm": [], "rmse_mm": [], "p95_mm": [], "max_mm": [],
        "tangential_mean_mm": [], "normal_mean_mm": [],
    }
    for i in range(pr.shape[1]):
        eng = engaged[:, i]
        mask = eng & both[:, i]
        out["engaged_fraction"].append(float(eng.mean()))
        out["measured_fraction"].append(float(mask.sum() / max(int(eng.sum()), 1)))
        if not mask.any():
            for key in ["mean_mm", "rmse_mm", "p95_mm", "max_mm",
                        "tangential_mean_mm", "normal_mean_mm"]:
                out[key].append(0.0)
            continue
        dv = delta[mask, i]
        dist = np.linalg.norm(dv, axis=-1) * 1000.0
        unit = fr[mask, i] / np.maximum(ref_mag[mask, i], 1e-9)[:, None]
        along = np.sum(dv * unit, axis=-1)
        out["mean_mm"].append(float(dist.mean()))
        out["rmse_mm"].append(float(np.sqrt((dist**2).mean())))
        out["p95_mm"].append(float(np.percentile(dist, 95)))
        out["max_mm"].append(float(dist.max()))
        out["tangential_mean_mm"].append(
            float(np.linalg.norm(dv - along[:, None] * unit, axis=-1).mean() * 1000.0))
        out["normal_mean_mm"].append(float(np.abs(along).mean() * 1000.0))
    return out


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """[..., 4] wxyz quaternions -> [..., 3, 3] rotation matrices (numpy only)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - w * z); m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y); m[..., 2, 1] = 2 * (y * z + w * x); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def grasp_slip_metrics(
    obj_pos: np.ndarray,             # [T, 3] rollout: manipulated-object position (world)
    palm_pos: np.ndarray,            # [T, 3] rollout: palm position (world)
    palm_quat: np.ndarray,           # [T, 4] rollout: palm orientation (wxyz)
    ep_obj_pos: np.ndarray,          # [T, 3] episode (ground truth) same quantities
    ep_palm_pos: np.ndarray,
    ep_palm_quat: np.ndarray,
    f_ref: np.ndarray,               # [T, tips] predicted force magnitude (frame-end)
    f_meas: np.ndarray,              # [T, tips] measured force magnitude (frame-end)
    hold_force_n: float = 5.0,       # "a strong grasp is commanded" threshold on |f_ref|
    slip_mm: float = 20.0,           # deviation that counts as real slip
) -> dict:
    """Unintended in-hand slip, measured against the episode's own intended motion.

    The object's position is expressed in the PALM frame (so hand translation and rotation drop
    out), for both the rollout and the recorded episode. Their difference is the *unintended*
    part: commanded pushing, pouring and carrying match the episode and cancel; a grasp that
    creeps or lets go diverges. The exact-replay run scores ~0 by construction.

    Reported over the frames where the episode commands a strong grasp (``|f_ref| >= 5 N``).
    ``dropped``: after the grasp had formed, the measured force collapsed (< 0.5 N) while a
    strong grasp was still commanded - the object left the hand (a commanded release lowers
    ``f_ref`` too, so it does not trigger this).
    """
    T = min(len(obj_pos), len(ep_obj_pos))
    rel = np.einsum("tji,tj->ti", _quat_wxyz_to_mat(palm_quat[:T]),
                    np.asarray(obj_pos[:T], np.float64) - np.asarray(palm_pos[:T], np.float64))
    ep_rel = np.einsum("tji,tj->ti", _quat_wxyz_to_mat(ep_palm_quat[:T]),
                       np.asarray(ep_obj_pos[:T], np.float64) - np.asarray(ep_palm_pos[:T], np.float64))
    dev = np.linalg.norm(rel - ep_rel, axis=1) * 1000.0              # mm, unintended motion

    strong = np.asarray(f_ref[:T]).max(axis=1) >= hold_force_n       # commanded strong grasp
    held = strong & (np.asarray(f_meas[:T]).max(axis=1) >= hold_force_n)
    out = {"hold_force_n": hold_force_n, "slip_mm": slip_mm,
           "window_frames": int(strong.sum()),
           "dev_mean_mm": 0.0, "dev_max_mm": 0.0, "dev_end_mm": 0.0,
           "slip_onset_frame": None, "dropped": False, "drop_frame": None}
    if not strong.any():
        return out
    w = np.nonzero(strong)[0]
    out["dev_mean_mm"] = float(dev[w].mean())
    out["dev_max_mm"] = float(dev[w].max())
    out["dev_end_mm"] = float(dev[w[-1]])
    slipping = strong & (dev > slip_mm)
    if slipping.any():
        out["slip_onset_frame"] = int(np.argmax(slipping))
    if held.any():
        first_held = int(np.argmax(held))
        lost = strong & (np.asarray(f_meas[:T]).max(axis=1) < 0.5)
        lost[:first_held + 3] = False
        # two consecutive lost frames = the object is gone, not a one-step sensor blink
        lost2 = lost[:-1] & lost[1:]
        if lost2.any():
            out["dropped"] = True
            out["drop_frame"] = int(np.argmax(lost2))
    return out


def format_grasp_slip(m: dict, prefix: str = "") -> str:
    if m["window_frames"] == 0:
        return f"{prefix}grasp slip: no strong-grasp phase (|f_ref| never >= {m['hold_force_n']:g} N)"
    line = (f"{prefix}grasp slip vs episode ({m['window_frames']} strong-grasp frames): "
            f"unintended obj-in-palm motion mean {m['dev_mean_mm']:5.1f} mm, "
            f"max {m['dev_max_mm']:6.1f}, end {m['dev_end_mm']:6.1f}")
    if m["dropped"]:
        line += f"   [DROPPED at f{m['drop_frame']}]"
    elif m["slip_onset_frame"] is not None:
        line += f"   [slip > {m['slip_mm']:g} mm from f{m['slip_onset_frame']}]"
    return line


def command_fidelity_metrics(
    action: np.ndarray,              # [T, J] the command the middle layer actually sent (frame end)
    ideal_target: np.ndarray,        # [T, J] the recorded joint_target - a control input KNOWN to
    #                                  achieve the reference exactly (--exact-replay reproduces it)
    q_ref: np.ndarray,               # [T, J] the predicted state the offset was added to
    joint_names: list[str],
    contact_mask: np.ndarray | None = None,   # [T] frames where contact is commanded
) -> dict:
    """Two scores for the offset the force law produced. One is deployable, one is not.

    **``offset_*`` (DEPLOYABLE, use this to tune).** ``|dq| = |action - q_ref|``: how far the
    controller had to depart from the predicted state to satisfy the predicted force. The middle
    layer holds two references - the policy's state and its force target - and this is the price
    it paid in the first to satisfy the second. Needs nothing a real robot lacks.

    **``hand_*`` / ``per_joint`` (GROUND TRUTH, development diagnostic only).**
    ``|action - ideal_target|`` against the recorded command, which reproduces the reference
    bit-exactly. It does not exist on a real robot, so it must never be a control signal or a
    tuning target - it is computed post-hoc here, never inside the loop. Its value is the
    per-joint split: ``needed_lead = ideal_target - q_ref`` is the offset the law was *supposed*
    to synthesise and ``produced_dq = action - q_ref`` is what it produced, so the residual shows
    *which joints* the offset went to. Aggregate force error cannot show that.

    The two agree closely (rank correlation 0.98 over the 26 dev rollouts of 2026-08-29), which is
    what licenses tuning on ``offset_mean_mrad`` alone and keeping the ground-truth score as an
    occasional honesty check on the surrogate.
    """
    action = np.asarray(action, dtype=np.float64)
    ideal_target = np.asarray(ideal_target, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    mask = np.ones(action.shape[0], bool) if contact_mask is None else np.asarray(contact_mask, bool)
    if not mask.any():
        mask = np.ones(action.shape[0], bool)
    hand = [i for i, n in enumerate(joint_names) if n not in ARM_JOINT_NAMES]
    arm = [i for i, n in enumerate(joint_names) if i not in hand]

    gap = np.abs(action - ideal_target)
    needed = (ideal_target - q_ref)[mask]
    produced = (action - q_ref)[mask]
    offset = np.abs(action - q_ref)
    out = {
        "contact_frames": int(mask.sum()),
        # ---- deployable: no ground truth involved ----
        "offset_mean_mrad": float(offset[np.ix_(mask, hand)].mean() * 1000.0),
        "offset_max_mrad": float(offset[np.ix_(mask, hand)].max() * 1000.0),
        "offset_rms_mrad": float(np.sqrt((offset[np.ix_(mask, hand)] ** 2).mean()) * 1000.0),
        # ---- ground-truth diagnostic: never a control signal or tuning target ----
        "hand_mean_mrad": float(gap[np.ix_(mask, hand)].mean() * 1000.0),
        "hand_max_mrad": float(gap[np.ix_(mask, hand)].max() * 1000.0),
        "hand_rms_mrad": float(np.sqrt((gap[np.ix_(mask, hand)] ** 2).mean()) * 1000.0),
        "hand_mean_all_frames_mrad": float(gap[:, hand].mean() * 1000.0),
        "arm_mean_mrad": float(gap[np.ix_(mask, arm)].mean() * 1000.0) if arm else 0.0,
        "per_joint": {},
    }
    for k, j in enumerate(hand):
        out["per_joint"][joint_names[j]] = {
            "needed_lead_mrad": float(needed[:, j].mean() * 1000.0),
            "produced_dq_mrad": float(produced[:, j].mean() * 1000.0),
            "residual_mrad": float((needed[:, j] - produced[:, j]).mean() * 1000.0),
        }
    return out


def format_command_fidelity(metrics: dict, prefix: str = "", top: int = 5) -> list[str]:
    """Summary line plus the joints whose offset deviates most from the ideal lead."""
    lines = [
        f"{prefix}offset cost |dq| over {metrics['contact_frames']} contact frames: "
        f"hand mean {metrics['offset_mean_mrad']:6.2f} mrad, max {metrics['offset_max_mrad']:6.1f}"
        f"   [deployable]",
        f"{prefix}|action - ideal command|: hand mean {metrics['hand_mean_mrad']:6.2f} mrad, "
        f"max {metrics['hand_max_mrad']:6.1f}   [ground truth, diagnostic only]",
    ]
    worst = sorted(metrics["per_joint"].items(), key=lambda kv: -abs(kv[1]["residual_mrad"]))[:top]
    if worst:
        lines.append(f"{prefix}{'joint':<12}{'needed lead':>13}{'produced dq':>13}{'residual':>11}  (mrad)")
        for name, v in worst:
            lines.append(f"{prefix}{name:<12}{v['needed_lead_mrad']:>13.2f}"
                         f"{v['produced_dq_mrad']:>13.2f}{v['residual_mrad']:>11.2f}")
    return lines


def format_contact_point_metrics(metrics: dict, prefix: str = "") -> list[str]:
    """Human-readable per-fingertip lines for a :func:`contact_point_metrics` result."""
    lines = []
    for i, body in enumerate(metrics["bodies"]):
        lines.append(
            f"{prefix}{body:<16s} engaged {metrics['engaged_fraction'][i] * 100:5.1f}%  "
            f"touching {metrics['measured_fraction'][i] * 100:5.1f}%  "
            f"|Δp| mean {metrics['mean_mm'][i]:6.2f} mm (p95 {metrics['p95_mm'][i]:6.2f}, "
            f"max {metrics['max_mm'][i]:6.2f})  "
            f"tang {metrics['tangential_mean_mm'][i]:5.2f} / norm {metrics['normal_mean_mm'][i]:5.2f} mm"
        )
    return lines


def format_vector_metrics(metrics: dict, prefix: str = "") -> list[str]:
    """Human-readable per-fingertip lines for a :func:`vector_force_metrics` result."""
    lines = []
    for i, body in enumerate(metrics["bodies"]):
        lines.append(
            f"{prefix}{body:<16s} engaged {metrics['engaged_fraction'][i] * 100:5.1f}%  "
            f"|Δf⃗| rmse {metrics['rmse_vec_N'][i]:7.3f} N (max {metrics['max_vec_N'][i]:6.1f})  "
            f"mag bias {metrics['bias_mag_N'][i]:+7.3f} N  "
            f"angle {metrics['angle_mean_deg'][i]:5.1f}° (p95 {metrics['angle_p95_deg'][i]:5.1f}°)"
        )
        if metrics.get("bias_xyz_N"):
            bx, by, bz = metrics["bias_xyz_N"][i]
            rx, ry, rz = metrics["rmse_xyz_N"][i]
            lines.append(
                f"{prefix}{'':<16s}  per-axis bias X {bx:+6.2f} Y {by:+6.2f} Z {bz:+6.2f} N  |  "
                f"rms X {rx:5.2f} Y {ry:5.2f} Z {rz:5.2f} N"
            )
    return lines


# --------------------------------------------------------------------------------------
# Vector recovery for rollouts recorded before the vectors were saved
# --------------------------------------------------------------------------------------
def _only_known(cls, values: dict):
    """Build *cls* from *values*, dropping keys the dataclass no longer has.

    Rollouts recorded before 2026-08-29 carry fields that have since been removed (the squeeze
    law's ``squeeze_weights``/``kp``/``ki``/``u_max_rad``, ``policy.force_source``). They must
    still replot.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in known})


def _controller_config_from_summary(summary: dict) -> ControllerConfig:
    ctrl = summary["controller"]
    return ControllerConfig(
        policy=_only_known(PolicyConfig, ctrl["policy"]),
        reference=_only_known(ReferenceConfig, ctrl["reference"]),
        force_law=_only_known(ForceLawConfig, ctrl["force_law"]),
        clamp_action_to_limits=ctrl.get("clamp_action_to_limits", True),
    )


def reconstruct_reference_vectors(
    episode: ReplayEpisode, cfg: ControllerConfig, T: int, S: int, force_source: str = "net"
) -> np.ndarray:
    """Re-run policy + ChunkTracker (no sim, no law) and return ``f_ref_vec`` at every step.

    *force_source* reproduces what the run being analysed actually tracked: rollouts before
    2026-08-29 used the privileged per-object force ("manipulated"), current ones use the pad's
    net force. Analysis only — the control path is always net.
    """
    policy = ChunkedReplayPolicy(episode, cfg.policy, force_source=force_source)
    # only the reference is wanted here, so the law is forced to null: a legacy run may name a law
    # this build no longer has (the squeeze law, removed 2026-08-29) and it would not be used anyway
    cfg = dataclasses.replace(cfg, force_law=dataclasses.replace(cfg.force_law, law="null"))
    middle = HybridForceMiddleLayer(cfg, episode.joint_names, episode.fingertip_bodies)
    middle.reset(episode.joint_target[0])
    n = len(episode.fingertip_bodies)
    out = np.zeros((T, S, n, 3))
    for frame in range(T):
        if frame % cfg.policy.chunk == 0:
            middle.on_new_chunk(policy.predict(frame))
        for step in range(S):
            ref = middle.evaluate_reference(middle.command_time(frame, step, S))
            if ref.f_vec is not None:
                out[frame, step] = ref.f_vec
    return out


def reconstruct_filtered_measured(
    raw_steps: np.ndarray,           # [T, S, 4, 3] recorded per-step force vectors (sensor frame k)
    ema_alpha: float,
) -> np.ndarray:
    """Re-apply the middle layer's sensing model: one-step delay + per-axis EMA."""
    T, S, n, _ = raw_steps.shape
    raw = raw_steps.reshape(T * S, n, 3).astype(np.float64)
    filt = np.zeros_like(raw)
    alpha = float(np.clip(ema_alpha, 0.0, 1.0))
    state = np.zeros((n, 3))                     # reading at the very first step is zero (post-reset)
    filt[0] = state
    for g in range(1, T * S):
        reading = raw[g - 1]                     # the controller acts on the previous step's reading
        state = reading if alpha >= 1.0 else alpha * reading + (1.0 - alpha) * state
        filt[g] = state
    return filt.reshape(T, S, n, 3)


def load_force_vectors(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(f_ref_vec_steps, f_meas_vec_steps, summary)`` for a rollout folder.

    Uses the recorded arrays when the run saved them; otherwise reconstructs both exactly (see
    module docstring).
    """
    run_dir = Path(run_dir)
    data = np.load(run_dir / "controller_data.npz", allow_pickle=True)
    summary = json.loads((run_dir / "summary.json").read_text())

    if "f_ref_vec_steps" in data.files:
        return (np.asarray(data["f_ref_vec_steps"], dtype=np.float64),
                np.asarray(data["f_meas_vec_steps"], dtype=np.float64), summary)

    cfg = _controller_config_from_summary(summary)
    legacy_source = summary["controller"]["policy"].get("force_source", "net")
    joint_names = [str(n) for n in data["joint_names"]]
    episode = ReplayEpisode.load(Path(str(data["episode_path"]))).reordered(joint_names)
    T, S = data["f_ref_steps"].shape[:2]

    f_ref_vec = reconstruct_reference_vectors(episode, cfg, T, S, force_source=legacy_source)

    if legacy_source == "manipulated" and episode.manipulated_contact_idx is not None:
        keys = [str(k) for k in data["contact_object_keys"]]
        midx = keys.index(episode.manipulated_key)
        raw = np.asarray(data["contact_object_force_steps"], dtype=np.float64)[:, :, :, midx, :]
    else:
        raw = np.asarray(data["contact_force_steps"], dtype=np.float64)
    f_meas_vec = reconstruct_filtered_measured(raw, cfg.force_law.meas_ema_alpha)
    return f_ref_vec, f_meas_vec, summary


# ----------------------------------------------------------------------------------------------
# [play2perfect] task outcome, scored the way the play2perfect env scores it
# ----------------------------------------------------------------------------------------------
# fixed-size keypoints (play2perfect obs_utils.KEYPOINT_CORNERS x reward.fixed_size x
# keypoint_scale / 2) - the same box corners the env's success test measures, for every problem
P2P_KEYPOINT_CORNERS = np.array([(1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)], dtype=np.float64)
P2P_FIXED_SIZE = np.array([0.141, 0.03025, 0.0271])
P2P_KEYPOINT_SCALE = 1.5


def task_success_metrics(
    obj_pos: np.ndarray,             # [T, 3] manipulated part position (world), rollout
    obj_quat: np.ndarray,            # [T, 4] wxyz
    goal_pos: np.ndarray,            # [3]    FINAL goal pose of the recording (the assembled pose)
    goal_quat: np.ndarray,           # [4]    wxyz
    fingertip_pos: np.ndarray | None = None,   # [T, tips, 3] for the retract test
    insertion_tolerance_m: float = 0.01,        # env: precise_assembly.insertion_success_tolerance
    retract_tolerance_m: float = 0.005,         # env: precise_assembly.retract_success_tolerance
    success_steps: int = 10,                    # env: termination.success_steps (consecutive)
    retract_distance_m: float = 0.1,            # env: precise_assembly.retract_distance_threshold
) -> dict:
    """Did the rollout actually assemble the part? The play2perfect env calls a subgoal reached
    when the max distance between the part's fixed-size box-corner keypoints and the goal's is
    within ``insertion_tolerance * keypoint_scale`` for ``success_steps`` consecutive policy
    steps; the episode is a success when the FINAL pose is reached, and the retract succeeds when
    the fingertips then move ``retract_distance`` away while the part stays within
    ``retract_tolerance * keypoint_scale``. This reproduces that test offline against the
    recording's final goal pose. ``keypoint_dist_mm`` is the same number the env uses."""
    kp = P2P_KEYPOINT_CORNERS * (0.5 * P2P_KEYPOINT_SCALE * P2P_FIXED_SIZE)          # [4, 3]
    r_obj = _quat_wxyz_to_mat(obj_quat)                                             # [T, 3, 3]
    r_goal = _quat_wxyz_to_mat(np.asarray(goal_quat, np.float64))                   # [3, 3]
    obj_kp = np.asarray(obj_pos, np.float64)[:, None, :] + np.einsum("tij,kj->tki", r_obj, kp)
    goal_kp = np.asarray(goal_pos, np.float64)[None, :] + kp @ r_goal.T
    dist = np.linalg.norm(obj_kp - goal_kp[None], axis=-1).max(axis=1)              # [T]
    tol = insertion_tolerance_m * P2P_KEYPOINT_SCALE
    near = dist <= tol
    run, inserted_frame = 0, None
    for t, ok in enumerate(near):
        run = run + 1 if ok else 0
        if run >= success_steps:
            inserted_frame = int(t)
            break
    out = {
        "inserted": inserted_frame is not None,
        "inserted_frame": inserted_frame,
        "keypoint_tolerance_mm": float(tol * 1000),
        "min_keypoint_dist_mm": float(dist.min() * 1000),
        "final_keypoint_dist_mm": float(dist[-1] * 1000),
        "frames_within_tolerance": int(near.sum()),
        "retract_success": False,
        "held_at_goal_until_end": bool(inserted_frame is not None and near[inserted_frame:].all()),
    }
    if inserted_frame is not None and fingertip_pos is not None:
        ft_dist = np.linalg.norm(np.asarray(fingertip_pos, np.float64) - np.asarray(obj_pos)[:, None, :], axis=-1).mean(axis=1)
        at_goal = dist <= retract_tolerance_m * P2P_KEYPOINT_SCALE
        out["retract_success"] = bool(((ft_dist > retract_distance_m) & at_goal)[inserted_frame:].any())
    return out


def format_task_success(m: dict, prefix: str = "") -> str:
    if m["inserted"]:
        s = (f"{prefix}task: INSERTED at frame {m['inserted_frame']} "
             f"(keypoint dist <= {m['keypoint_tolerance_mm']:.0f} mm), retract "
             f"{'ok' if m['retract_success'] else 'no'}, held to the end: "
             f"{'yes' if m['held_at_goal_until_end'] else 'no'}")
    else:
        s = (f"{prefix}task: NOT inserted (closest keypoint distance {m['min_keypoint_dist_mm']:.1f} mm "
             f"vs tolerance {m['keypoint_tolerance_mm']:.0f} mm, final {m['final_keypoint_dist_mm']:.1f} mm)")
    return s
