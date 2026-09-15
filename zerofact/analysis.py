"""Post-processing for a replay: tracking plots, object-motion plots and the 3D flow metric.

Nothing in here imports Isaac Sim, so the plots can be regenerated offline from ``replay_data.npz``.

The flow metric is the one the Video2Sim2Real Isaac Gym scripts use to score a replay: sample points
on the manipulated object's surface, follow them through the simulated object pose, and compare the
centroid trajectory against the 3D object flow estimated from the demonstration video
(``flow_data/3d_flow_point.pkl``).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .scene_spec import FINGERTIP_COLORS, FINGERTIP_LABELS
from .video import VideoWriter


# --------------------------------------------------------------------------------------
# Demo flow (video side)
# --------------------------------------------------------------------------------------
class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    """Pickles written under numpy >= 2 reference ``numpy._core``; map it back for numpy 1.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_3d_flow_points(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        with open(path, "rb") as f:
            return _NumpyCoreCompatUnpickler(f).load()
    if path.suffix.lower() == ".npy":
        return [frame.copy() for frame in np.load(path, allow_pickle=True)]
    raise ValueError(f"Unsupported flow file: {path}")


def _extract_pts_and_ids(frame):
    if (
        isinstance(frame, (list, tuple))
        and len(frame) == 2
        and not (hasattr(frame[0], "shape") and getattr(frame[0], "ndim", 0) == 0)
    ):
        pts_raw, ids_raw = frame
        pts = np.asarray(pts_raw, dtype=float).reshape(-1, 3)
        ids = np.asarray(ids_raw).reshape(-1).astype(np.int64)
        if ids.shape[0] != pts.shape[0]:
            raise ValueError(f"points/ids length mismatch: {pts.shape[0]} vs {ids.shape[0]}")
        return pts, ids
    pts = np.asarray(frame, dtype=float).reshape(-1, 3)
    return pts, np.arange(pts.shape[0], dtype=np.int64)


def compute_flow_traj_centroid_and_delta_3d(demo_flow_point_3d):
    """Port of the Isaac Gym helper: per-frame centroid and centroid shift w.r.t. frame 0.

    When point ids are available the shift is computed over the ids visible in both frames, which
    avoids spurious drift when the tracked point set changes.
    """
    if not isinstance(demo_flow_point_3d, (list, tuple)) or len(demo_flow_point_3d) == 0:
        raise ValueError("demo_flow_point_3d must be a non-empty list/tuple")

    T = len(demo_flow_point_3d)
    centroids = np.full((T, 3), np.nan)
    delta = np.full((T, 3), np.nan)

    pts_list, ids_list = [], []
    for t, frame in enumerate(demo_flow_point_3d):
        pts, ids = _extract_pts_and_ids(frame)
        pts_list.append(pts)
        ids_list.append(ids)
        if pts.shape[0] > 0:
            centroids[t] = pts.mean(axis=0)

    pts0, ids0 = pts_list[0], ids_list[0]
    if pts0.shape[0] == 0:
        return centroids, delta

    id0_to_idx = {int(i): k for k, i in enumerate(ids0)}
    for t in range(T):
        pts_t, ids_t = pts_list[t], ids_list[t]
        if pts_t.shape[0] == 0:
            continue
        common_t, common_0 = [], []
        for k, i in enumerate(ids_t):
            j = id0_to_idx.get(int(i))
            if j is not None:
                common_t.append(k)
                common_0.append(j)
        if common_0:
            delta[t] = pts_t[common_t].mean(axis=0) - pts0[common_0].mean(axis=0)
        else:
            delta[t] = centroids[t] - centroids[0]

    return centroids, delta


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def tracking_error(joint_pos: np.ndarray, joint_target: np.ndarray) -> dict:
    """Per-joint and overall PD tracking error (rad)."""
    err = joint_pos - joint_target
    return {
        "mean_abs_rad": float(np.abs(err).mean()),
        "max_abs_rad": float(np.abs(err).max()),
        "rms_rad": float(np.sqrt((err**2).mean())),
        "per_joint_mean_abs_rad": np.abs(err).mean(axis=0).tolist(),
        "per_joint_max_abs_rad": np.abs(err).max(axis=0).tolist(),
    }


def object_motion(object_pos: np.ndarray, key: int) -> dict:
    """Displacement statistics of one object across the replay."""
    pos = object_pos[:, key]
    delta = pos - pos[0]
    return {
        "start_xyz": pos[0].tolist(),
        "end_xyz": pos[-1].tolist(),
        "total_displacement_m": float(np.linalg.norm(delta[-1])),
        "max_displacement_m": float(np.linalg.norm(delta, axis=1).max()),
        "max_lift_m": float(delta[:, 2].max()),
        "path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()),
    }


def flow_metrics(sim_flow_points: np.ndarray, demo_flow_path: Path | None) -> dict:
    """Compare the simulated object-surface flow with the demo's 3D flow.

    ``sim_flow_points``: [T, N, 3] world-frame positions of the sampled surface points.
    """
    out: dict = {}
    if sim_flow_points.size == 0:
        return out

    sim_centroid = sim_flow_points.mean(axis=1)                    # [T, 3]
    sim_delta = sim_centroid - sim_centroid[0]
    out["sim_centroid_start"] = sim_centroid[0].tolist()
    out["sim_centroid_end"] = sim_centroid[-1].tolist()
    out["sim_delta_end_m"] = sim_delta[-1].tolist()
    out["sim_max_lift_m"] = float(sim_delta[:, 2].max())

    if demo_flow_path is None or not Path(demo_flow_path).is_file():
        return out

    try:
        demo_frames = load_3d_flow_points(demo_flow_path)
        demo_centroid, demo_delta = compute_flow_traj_centroid_and_delta_3d(demo_frames)
    except Exception as exc:
        out["demo_flow_error"] = str(exc)
        return out

    T = min(len(demo_centroid), len(sim_centroid))
    out["demo_frames"] = int(len(demo_centroid))
    out["compared_frames"] = int(T)
    out["demo_delta_end_m"] = demo_delta[T - 1].tolist()
    out["demo_max_lift_m"] = float(np.nanmax(demo_delta[:T, 2]))

    # absolute-position error (the metric the Isaac Gym script minimises when picking a grasp)
    abs_err = np.linalg.norm(sim_centroid[:T] - demo_centroid[:T], axis=1)
    out["centroid_abs_error_mean_m"] = float(np.nanmean(abs_err))
    out["centroid_abs_error_final_m"] = float(abs_err[-1])
    # delta-motion error: robust to a constant offset between the reconstructed and simulated frames
    delta_err = np.linalg.norm(sim_delta[:T] - demo_delta[:T], axis=1)
    out["centroid_delta_error_mean_m"] = float(np.nanmean(delta_err))
    out["centroid_delta_error_final_m"] = float(delta_err[-1])
    return out


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
KEY_FRAME_COLORS = {
    "hand_pose_frame": "tab:gray",
    "pregrasp_frame": "tab:olive",
    "contact_frame": "tab:red",
    "pre_interaction_frame": "tab:purple",
    "interaction_frame": "tab:green",
    "drop_frame": "tab:brown",
}


def _key_frame_lines(ax, key_frames: dict, num_frames: int, label: bool = False):
    for name, value in (key_frames or {}).items():
        if value is None or not (0 <= int(value) < num_frames):
            continue
        color = KEY_FRAME_COLORS.get(name, "k")
        ax.axvline(int(value), color=color, linestyle=":", linewidth=1.0, alpha=0.7)
        if label:
            ax.annotate(
                name.replace("_frame", ""),
                xy=(int(value), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=7,
                color=color,
                alpha=0.9,
            )


def plot_joint_tracking(
    joint_pos: np.ndarray,
    joint_target: np.ndarray,
    joint_names: list[str],
    save_path: Path,
    key_frames: dict | None = None,
) -> None:
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = joint_pos.shape
    n_cols = 4
    n_rows = math.ceil(N / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.4 * n_rows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    time = np.arange(T)

    for i in range(N):
        ax = axes[i]
        ax.plot(time, joint_target[:, i], color="orange", linestyle="--", label="command")
        ax.plot(time, joint_pos[:, i], color="tab:blue", label="actual")
        _key_frame_lines(ax, key_frames or {}, T)
        ax.set_title(joint_names[i], fontsize=9)
        ax.grid(True, alpha=0.3)
    for i in range(N, len(axes)):
        fig.delaxes(axes[i])
    if N:
        axes[0].legend(fontsize=8)
    fig.suptitle("Joint tracking: commanded vs simulated (rad)", fontsize=14)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def plot_object_tracking(
    object_pos: np.ndarray,
    object_names: list[str],
    save_path: Path,
    key_frames: dict | None = None,
    manipulated_idx: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = object_pos.shape[0]
    time = np.arange(T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, ax in zip("xyz", axes):
        k = "xyz".index(axis)
        for j, name in enumerate(object_names):
            style = "-" if (manipulated_idx is None or j == manipulated_idx) else "--"
            ax.plot(time, object_pos[:, j, k], style, label=name)
        _key_frame_lines(ax, key_frames or {}, T)
        ax.set_title(f"object {axis} (m)")
        ax.set_xlabel("trajectory frame")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Object world position during the replay", fontsize=14)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def plot_flow_comparison(
    sim_flow_points: np.ndarray,
    demo_flow_path: Path | None,
    save_path: Path,
    key_frames: dict | None = None,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sim_flow_points.size == 0:
        return False
    sim_centroid = sim_flow_points.mean(axis=1)
    sim_delta = sim_centroid - sim_centroid[0]

    demo_delta = None
    if demo_flow_path is not None and Path(demo_flow_path).is_file():
        try:
            _, demo_delta = compute_flow_traj_centroid_and_delta_3d(load_3d_flow_points(demo_flow_path))
        except Exception:
            demo_delta = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for k, (axis, ax) in enumerate(zip("xyz", axes)):
        ax.plot(np.arange(len(sim_delta)), sim_delta[:, k], color="tab:blue", label="sim (Isaac Lab)")
        if demo_delta is not None:
            ax.plot(np.arange(len(demo_delta)), demo_delta[:, k], color="tab:green", label="demo (video flow)")
        _key_frame_lines(ax, key_frames or {}, len(sim_delta))
        ax.set_title(f"object flow centroid Δ{axis} (m)")
        ax.set_xlabel("frame")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Object motion: simulated replay vs video demonstration", fontsize=14)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return True


def _draw_contact_force_axes(
    axes,
    contact_force_steps: np.ndarray,
    fingertip_bodies: list[str],
    key_frames: dict | None,
    object_force_steps: np.ndarray | None,
    object_names: list[str] | None,
    manipulated_idx: int | None,
    label_key_frames: bool = True,
) -> tuple[int, int]:
    """Draw the per-fingertip |F| curves onto pre-made *axes*; returns (T, S).

    Each fingertip's main curve uses that fingertip's colour (the same colour as its in-scene force
    arrow): the force from the manipulated object when the per-object breakdown is available,
    otherwise the net force. The net force is then a thin grey line and every remaining contact
    (other objects, table, ground, self-collision) a thin dashed one.
    """
    T, S, B = contact_force_steps.shape[:3]
    net = np.linalg.norm(contact_force_steps, axis=-1).reshape(T * S, B)
    # step s of frame k is drawn at x = k - 1 + (s + 1) / S, so each frame's *last* step lands
    # exactly on x = k - the same abscissa the per-frame samples use in the other plots
    x = (np.arange(T * S) + 1.0) / S - 1.0

    manip = other = None
    if (
        object_force_steps is not None
        and object_force_steps.size
        and manipulated_idx is not None
        and 0 <= manipulated_idx < object_force_steps.shape[3]
    ):
        manip_vec = object_force_steps[:, :, :, manipulated_idx, :]
        manip = np.linalg.norm(manip_vec, axis=-1).reshape(T * S, B)
        other = np.linalg.norm(contact_force_steps - manip_vec, axis=-1).reshape(T * S, B)

    manip_name = (
        object_names[manipulated_idx] if (manip is not None and object_names) else "manipulated object"
    )
    for i, body in enumerate(fingertip_bodies):
        ax = axes[i]
        color = FINGERTIP_COLORS.get(body, (0.2, 0.2, 0.2))
        if manip is not None:
            ax.plot(x, net[:, i], color="0.6", linewidth=0.8, label="net contact force")
            ax.plot(x, manip[:, i], color=color, linewidth=1.4, label=f"from {manip_name} (manipulated)")
            ax.plot(x, other[:, i], color="0.35", linewidth=0.8, linestyle="--", label="other contacts")
        else:
            ax.plot(x, net[:, i], color=color, linewidth=1.2, label="net contact force")
        _key_frame_lines(ax, key_frames or {}, T, label=(label_key_frames and i == 0))
        label = FINGERTIP_LABELS.get(body)
        ax.set_title(f"{body} ({label})" if label else body, fontsize=10, color=color)
        ax.set_ylabel("|F| (N)")
        ax.set_xlim(-1.0, T - 1)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("trajectory frame")
    return T, S


def plot_contact_forces(
    contact_force_steps: np.ndarray,
    fingertip_bodies: list[str],
    save_path: Path,
    key_frames: dict | None = None,
    object_force_steps: np.ndarray | None = None,
    object_names: list[str] | None = None,
    manipulated_idx: int | None = None,
) -> bool:
    """Per-fingertip contact force magnitude at every physics step.

    ``contact_force_steps``: [T, S, B, 3] net world-frame force on each fingertip, S physics steps
    per trajectory frame. ``object_force_steps``: [T, S, B, M, 3] force from each of the M filtered
    scene objects; when given (with ``manipulated_idx``) the grasp force on the manipulated object
    is drawn separately from everything else (other objects, table, ground, self-collision).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if contact_force_steps.size == 0 or not fingertip_bodies:
        return False
    B = contact_force_steps.shape[2]
    fig, axes = plt.subplots(B, 1, figsize=(14, 2.3 * B), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    _, S = _draw_contact_force_axes(
        axes, contact_force_steps, fingertip_bodies, key_frames, object_force_steps, object_names, manipulated_idx
    )
    fig.suptitle(f"Fingertip contact forces, world frame ({S} physics steps per frame)", fontsize=14)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    return True


def render_contact_force_video(
    contact_force_steps: np.ndarray,
    fingertip_bodies: list[str],
    save_path: Path | None,
    fps: int,
    key_frames: dict | None = None,
    object_force_steps: np.ndarray | None = None,
    object_names: list[str] | None = None,
    manipulated_idx: int | None = None,
    sim_time_per_frame: float = 0.1,
    figsize: tuple[float, float] = (8.5, 6.0),
    dpi: int = 120,
    return_frames: bool = False,
) -> list[np.ndarray] | bool:
    """The contact-force plot as a video: a cursor sweeps the trajectory frame axis.

    One video frame per trajectory frame at *fps*, i.e. the same pace as ``replay.mp4``. The static
    plot is drawn once and only the cursor is re-blitted, so this takes seconds, not minutes.
    Returns the RGB frames when ``return_frames`` (for compositing), else True on success.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fail():
        return [] if return_frames else False

    if contact_force_steps.size == 0 or not fingertip_bodies:
        return _fail()

    B = contact_force_steps.shape[2]
    fig, axes = plt.subplots(B, 1, figsize=figsize, dpi=dpi, sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    T, S = _draw_contact_force_axes(
        axes, contact_force_steps, fingertip_bodies, key_frames, object_force_steps, object_names, manipulated_idx
    )
    fig.suptitle(f"Fingertip contact forces, world frame ({S} physics steps per frame)", fontsize=12)

    # draw the static plot once, then blit only the moving cursor over it
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    cursors = [ax.axvline(0, color="crimson", linewidth=1.2, alpha=0.9, animated=True) for ax in axes]
    frame_text = fig.text(0.005, 0.962, "", ha="left", va="top", fontsize=8, color="crimson", animated=True)

    frames: list[np.ndarray] = []
    for k in range(T):
        fig.canvas.restore_region(background)
        for cursor in cursors:
            cursor.set_xdata([k, k])
            cursor.axes.draw_artist(cursor)
        frame_text.set_text(f"frame {k + 1}/{T}   t = {k * sim_time_per_frame:5.1f} s")
        fig.draw_artist(frame_text)
        fig.canvas.blit(fig.bbox)
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    if save_path is not None and not _write_mp4(frames, save_path, fps):
        return _fail()
    return frames if return_frames else True


def _write_mp4(frames_rgb: list[np.ndarray], path: Path, fps: int) -> bool:
    if not frames_rgb:
        return False
    try:
        import cv2
    except ImportError:
        print(f"[analysis][WARN] cv2 not available -> {path} not written")
        return False
    height, width = frames_rgb[0].shape[:2]
    writer = VideoWriter(path, fps, (width, height))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return writer.release()


def _bgr255(rgb01: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in rgb01[::-1])


def overlay_contact_points(
    frames: list[np.ndarray],
    cam_poses: np.ndarray,          # [T, 6] camera eye + look-at target, world frame
    intrinsics: np.ndarray,         # [3, 3]
    contact_points: np.ndarray,     # [T, tips, objects, P, 3] world, NaN-padded
    contact_forces: np.ndarray,     # [T, tips, objects, P] per-point normal force (N)
    fingertip_bodies: list[str],
) -> list[np.ndarray]:
    """Draw each individual contact point into the camera frames as a ring.

    The 3D contact points sit exactly at the finger/object interface, so in a rendered image they
    are hidden inside the geometry; projecting them into the image and drawing rings keeps them
    visible while staying honest about where they are. Ring radius grows mildly with the
    per-point normal force; colours follow the fingertip colour scheme.
    """
    import cv2

    if not frames or cam_poses.size == 0 or contact_points.size == 0:
        return frames
    colors = [_bgr255(FINGERTIP_COLORS.get(b, (0.9, 0.9, 0.9))) for b in fingertip_bodies]
    T = min(len(frames), len(cam_poses), len(contact_points))
    out = []
    for k in range(len(frames)):
        frame = frames[k].copy()
        if k < T:
            c = cam_poses[k, :3]
            forward = cam_poses[k, 3:6] - c
            forward = forward / max(np.linalg.norm(forward), 1e-9)
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
            right = right / max(np.linalg.norm(right), 1e-9)
            down = np.cross(forward, right)
            rot = np.stack([right, down, forward], axis=1)   # camera-to-world (+Z forward, +Y down)
            for t, color in enumerate(colors):
                pts = contact_points[k, t].reshape(-1, 3)
                frc = contact_forces[k, t].reshape(-1)
                ok = np.isfinite(pts).all(axis=1) & (np.nan_to_num(frc) >= 0.01)
                for point, force in zip(pts[ok], frc[ok]):
                    p_cam = rot.T @ (point - c)
                    if p_cam[2] <= 0.01:
                        continue
                    u = intrinsics[0, 0] * p_cam[0] / p_cam[2] + intrinsics[0, 2]
                    v = intrinsics[1, 1] * p_cam[1] / p_cam[2] + intrinsics[1, 2]
                    if not (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
                        continue
                    radius = int(round(3 + 4 * min(abs(force), 50.0) / 50.0))
                    center = (int(round(u)), int(round(v)))
                    # frames are RGB here; convert the BGR colour tuple back
                    rgb = color[::-1]
                    cv2.circle(frame, center, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, center, radius, rgb, 2, cv2.LINE_AA)
                    cv2.circle(frame, center, 1, (255, 255, 255), -1, cv2.LINE_AA)
        out.append(frame)
    return out


def render_composite_video(
    sim_frames: list[np.ndarray],
    plot_frames: list[np.ndarray],
    save_path: Path,
    fps: int,
    run_name: str,
    fingertip_bodies: list[str],
    contact_force: np.ndarray,
    key_frames: dict | None = None,
    sim_time_per_frame: float = 0.1,
    force_vis_scale: float | None = None,
    manipulated_name: str | None = None,
    inset_frames: list[np.ndarray] | None = None,
) -> bool:
    """Side-by-side annotated video: demo-camera replay (left) + sweeping force plot (right).

    ``inset_frames`` (e.g. the fingertip-tracking hand camera) are drawn picture-in-picture in the
    sim panel's bottom-right corner.

    Annotations: a title bar with the run name, frame counter and sim clock; key-frame names
    flashed as they pass; a live per-fingertip force readout (in the fingertip/arrow colours) and
    the arrow scale on the sim panel.
    """
    try:
        import cv2
    except ImportError:
        print(f"[analysis][WARN] cv2 not available -> {save_path} not written")
        return False
    from matplotlib import colors as mcolors

    if not sim_frames or not plot_frames:
        return False
    num = min(len(sim_frames), len(plot_frames))
    forces = np.linalg.norm(contact_force, axis=-1) if contact_force.size else None   # [T, B]

    panel_h = plot_frames[0].shape[0]
    plot_w = plot_frames[0].shape[1]
    sim_h, sim_w = sim_frames[0].shape[:2]
    sim_scale = panel_h / sim_h
    sim_w_scaled = int(round(sim_w * sim_scale / 2) * 2)
    bar_h = 64
    width, height = sim_w_scaled + plot_w, bar_h + panel_h

    font = cv2.FONT_HERSHEY_SIMPLEX
    white, grey = (235, 235, 235), (170, 170, 170)
    tip_bgr = {b: _bgr255(FINGERTIP_COLORS.get(b, (0.8, 0.8, 0.8))) for b in fingertip_bodies}

    writer = VideoWriter(save_path, fps, (width, height))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for k in range(num):
        canvas[:] = (28, 24, 24)
        canvas[bar_h:, :sim_w_scaled] = cv2.resize(
            cv2.cvtColor(sim_frames[k], cv2.COLOR_RGB2BGR), (sim_w_scaled, panel_h), interpolation=cv2.INTER_LINEAR
        )
        canvas[bar_h:, sim_w_scaled:] = cv2.cvtColor(plot_frames[k], cv2.COLOR_RGB2BGR)

        # ---- title bar ----
        cv2.putText(canvas, f"V2S2R replay - {run_name}", (14, 26), font, 0.62, white, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, "fingertip contact forces (world frame)", (14, 50), font, 0.48, grey, 1, cv2.LINE_AA
        )
        clock = f"frame {k + 1:3d}/{num}   t = {k * sim_time_per_frame:5.1f} s"
        (tw, _), _ = cv2.getTextSize(clock, font, 0.55, 1)
        cv2.putText(canvas, clock, (width - tw - 14, 40), font, 0.55, white, 1, cv2.LINE_AA)
        # key-frame names flash for ~half a second as the replay passes them
        flashes = [
            (name.replace("_frame", ""), KEY_FRAME_COLORS.get(name, "w"))
            for name, value in (key_frames or {}).items()
            if value is not None and 0 <= k - int(value) < max(1, fps // 2)
        ]
        if flashes:
            text = "  |  ".join(name for name, _ in flashes)
            color = _bgr255(mcolors.to_rgb(flashes[0][1]))
            (tw, _), _ = cv2.getTextSize(text, font, 0.6, 2)
            cv2.putText(canvas, text, ((width - tw) // 2, 40), font, 0.6, color, 2, cv2.LINE_AA)

        # ---- hand-camera picture-in-picture, bottom-right of the sim panel ----
        if inset_frames is not None and k < len(inset_frames):
            inset_w = int(round(sim_w_scaled * 0.36 / 2) * 2)
            inset_h = int(round(inset_w * inset_frames[k].shape[0] / inset_frames[k].shape[1] / 2) * 2)
            x1, y1 = sim_w_scaled - inset_w - 8, height - inset_h - 8
            inset = cv2.resize(
                cv2.cvtColor(inset_frames[k], cv2.COLOR_RGB2BGR), (inset_w, inset_h),
                interpolation=cv2.INTER_AREA,
            )
            cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x1 + inset_w + 1, y1 + inset_h + 1), white, 2)
            canvas[y1 : y1 + inset_h, x1 : x1 + inset_w] = inset
            cv2.putText(canvas, "hand cam", (x1 + 6, y1 + 18), font, 0.42, white, 1, cv2.LINE_AA)

        # ---- sim-panel force readout (dark backdrop so it reads over any scene colour) ----
        if forces is not None and k < len(forces):
            captions = []
            if force_vis_scale:
                captions.append(f"arrows: 1 N = {force_vis_scale * 100:.1f} cm")
            if manipulated_name:
                captions.append(f"manipulated: {manipulated_name}")
            box_w, line_h, caption_h = 210, 22, 18
            box_h = line_h * (len(fingertip_bodies) + 1) + 8 + (len(captions) * caption_h + 8 if captions else 0)
            x0, y0 = 10, height - box_h - 10
            roi = canvas[y0 : y0 + box_h, x0 : x0 + box_w]
            roi[:] = (roi * 0.3).astype(np.uint8)                       # translucent backdrop
            cv2.putText(canvas, "net |F| per fingertip", (x0 + 8, y0 + 20), font, 0.42, white, 1, cv2.LINE_AA)
            for i, body in enumerate(fingertip_bodies):
                label = FINGERTIP_LABELS.get(body, body)
                text = f"{label:<6s} {forces[k, i]:6.1f} N"
                cv2.putText(
                    canvas, text, (x0 + 8, y0 + 20 + (i + 1) * line_h), font, 0.46, tip_bgr[body], 1, cv2.LINE_AA
                )
            y_caption = y0 + 20 + (len(fingertip_bodies) + 1) * line_h - 6
            for j, caption in enumerate(captions):
                cv2.putText(canvas, caption, (x0 + 8, y_caption + j * caption_h), font, 0.38, grey, 1, cv2.LINE_AA)
        writer.write(canvas)
    return writer.release()


def contact_metrics(
    contact_force: np.ndarray,
    contact_force_steps: np.ndarray,
    fingertip_bodies: list[str],
    object_force_steps: np.ndarray | None = None,
    object_names: list[str] | None = None,
    force_threshold: float = 1.0,
) -> dict | None:
    """Summary statistics of the fingertip force reading.

    ``max/mean_force_N`` keep their historical meaning (per-frame samples, i.e. the last physics
    step of each frame); ``peak_force_N`` is the true peak over every physics step. Per-object
    entries only list objects a fingertip actually touched.
    """
    if contact_force.size == 0 or not fingertip_bodies:
        return None
    forces = np.linalg.norm(contact_force, axis=-1)                    # [T, B] per-frame samples
    out: dict = {
        "bodies": list(fingertip_bodies),
        "force_threshold_N": force_threshold,
        "max_force_N": forces.max(axis=0).tolist(),
        "mean_force_N": forces.mean(axis=0).tolist(),
        "first_contact_frame": {
            body: (
                int(np.argmax(forces[:, i] > force_threshold))
                if (forces[:, i] > force_threshold).any()
                else None
            )
            for i, body in enumerate(fingertip_bodies)
        },
    }
    if contact_force_steps.size:
        step_mag = np.linalg.norm(contact_force_steps, axis=-1)        # [T, S, B]
        out["peak_force_N"] = step_mag.max(axis=(0, 1)).tolist()
    if object_force_steps is not None and object_force_steps.size and object_names:
        obj_mag = np.linalg.norm(object_force_steps, axis=-1)          # [T, S, B, M]
        per_frame = obj_mag.max(axis=1)                                # [T, B, M] peak within frame
        per_object: dict = {}
        for i, body in enumerate(fingertip_bodies):
            entries = {}
            for m, obj_name in enumerate(object_names):
                peak = float(obj_mag[:, :, i, m].max())
                if peak <= 0.0:
                    continue
                touched = per_frame[:, i, m] > force_threshold
                entries[obj_name] = {
                    "peak_force_N": peak,
                    "first_contact_frame": int(np.argmax(touched)) if touched.any() else None,
                }
            per_object[body] = entries
        out["per_object"] = per_object
    return out


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def hand_pose_trajectory(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    tracked_links: list[str],
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    object_keys: list[str],
    manipulated_key: str | None,
    hand_link: str = "palm_lower",
    contact_step: int | None = None,
) -> dict | None:
    """Recreate the Isaac Gym replay's ``hand_pose_traj.json``.

    Per frame: the hand (palm) pose in the world frame and in the manipulated object's frame, with
    quaternions in **xyzw** order, matching what the downstream Video2Sim2Real code expects.
    """
    if hand_link not in tracked_links or manipulated_key is None or manipulated_key not in object_keys:
        return None

    hand_idx = tracked_links.index(hand_link)
    obj_idx = object_keys.index(manipulated_key)
    frames = []

    for k in range(len(body_pos)):
        p_hand, q_hand = body_pos[k, hand_idx], body_quat[k, hand_idx]        # wxyz
        p_obj, q_obj = object_pos[k, obj_idx], object_quat[k, obj_idx]
        rot_obj = _quat_wxyz_to_matrix(q_obj)
        p_rel = rot_obj.T @ (p_hand - p_obj)
        q_obj_inv = np.array([q_obj[0], -q_obj[1], -q_obj[2], -q_obj[3]])
        q_rel = _quat_mul(q_obj_inv, q_hand)
        frames.append(
            {
                "frame": int(k),
                "world": {
                    "pos": [float(v) for v in p_hand],
                    "quat_xyzw": [float(q_hand[1]), float(q_hand[2]), float(q_hand[3]), float(q_hand[0])],
                },
                "obj_T_hand_rel": {
                    "pos": [float(v) for v in p_rel],
                    "quat_xyzw": [float(q_rel[1]), float(q_rel[2]), float(q_rel[3]), float(q_rel[0])],
                },
            }
        )

    return {"contact_step": contact_step, "hand_link": hand_link, "frames": frames}


def read_contact_step(run_dir: Path) -> int | None:
    path = Path(run_dir) / "flow_data" / "contact_step.txt"
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _json_safe(value):
    """Replace NaN / Inf with None so the summary stays valid JSON for strict parsers."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def write_summary(summary: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)
