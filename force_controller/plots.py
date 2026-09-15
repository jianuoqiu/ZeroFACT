"""Plots for the force-tracking pipeline (numpy + matplotlib only, no Isaac imports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# one colour per fingertip, same palette as v2s2r_isaaclab (tab10) so plots stay comparable
FINGERTIP_COLORS = {
    "fingertip": (0.121, 0.466, 0.705),
    "fingertip_2": (1.000, 0.498, 0.054),
    "fingertip_3": (0.172, 0.627, 0.172),
    "thumb_fingertip": (0.839, 0.152, 0.156),
}
FINGERTIP_LABELS = {
    "fingertip": "index",
    "fingertip_2": "middle",
    "fingertip_3": "ring",
    "thumb_fingertip": "thumb",
}


def _mark_key_frames(ax, key_frames: dict | None):
    if not key_frames:
        return
    for name, frame in key_frames.items():
        if frame is None:
            continue
        ax.axvline(frame, color="0.75", lw=0.8, zorder=0)
        ax.text(
            frame, ax.get_ylim()[1], name.replace("_frame", ""),
            rotation=90, va="top", ha="right", fontsize=6, color="0.5",
        )


def plot_force_tracking(
    f_ref: np.ndarray,               # [T, S, 4] or [T*S, 4] predicted force magnitude (N)
    f_meas: np.ndarray,              # same shape: measured (filtered) magnitude
    fingertip_bodies: list[str],
    out_path: str | Path,
    steps_per_frame: int,
    key_frames: dict | None = None,
    u: np.ndarray | None = None,     # [T, S, 4] per-tip law integrator state, optional
    title: str = "force tracking",
    u_label: str = "controller integral",
    engage_threshold_n: float | None = None,
) -> None:
    """Per-fingertip predicted-vs-measured force, one row per fingertip (x axis in frames).

    Colours are semantic and identical in every row (the row label carries the finger identity):
    blue = what the policy says the fingertip should feel, red = what the simulation sensor
    actually measured, grey fill = the gap between them (the tracking error), purple (right axis)
    = the force law's integral state, green shading = the law is engaged (predicted force above
    the engage threshold).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    f_ref = f_ref.reshape(-1, f_ref.shape[-1])
    f_meas = f_meas.reshape(-1, f_meas.shape[-1])
    t = np.arange(f_ref.shape[0]) / steps_per_frame
    n_tips = len(fingertip_bodies)

    C_REF, C_MEAS, C_ERR, C_U, C_ENG = "tab:blue", "tab:red", "0.55", "tab:purple", "tab:green"

    fig, axes = plt.subplots(n_tips, 1, figsize=(12, 2.3 * n_tips + 0.7), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, body) in enumerate(zip(axes, fingertip_bodies)):
        if engage_threshold_n is not None:
            engaged = f_ref[:, i] >= engage_threshold_n
            if engaged.any():
                ax.fill_between(t, 0, 1, where=engaged, transform=ax.get_xaxis_transform(),
                                color=C_ENG, alpha=0.07, lw=0)
        ax.fill_between(t, f_meas[:, i], f_ref[:, i], color=C_ERR, alpha=0.30, lw=0)
        ax.plot(t, f_ref[:, i], color=C_REF, lw=1.8)
        ax.plot(t, f_meas[:, i], color=C_MEAS, lw=0.9, alpha=0.9)
        ax.set_ylabel(f"{FINGERTIP_LABELS.get(body, body)}\n|F| [N]")
        ax.grid(alpha=0.25)
        if u is not None:
            ax2 = ax.twinx()
            ax2.plot(t, u.reshape(-1, u.shape[-1])[:, i], color=C_U, lw=0.9, alpha=0.75)
            ax2.set_ylabel(u_label, color=C_U, fontsize=7)
            ax2.tick_params(axis="y", labelsize=6, colors=C_U)
        _mark_key_frames(ax, key_frames)
    axes[-1].set_xlabel("trajectory frame")

    handles = [
        Line2D([], [], color=C_REF, lw=1.8, label="predicted force (policy target)"),
        Line2D([], [], color=C_MEAS, lw=1.2, label="measured force (sim sensor, filtered)"),
        Patch(color=C_ERR, alpha=0.30, label="tracking error (gap)"),
    ]
    if u is not None:
        handles.append(Line2D([], [], color=C_U, lw=1.2, label=f"{u_label} (right axis)"))
    if engage_threshold_n is not None:
        handles.append(Patch(color=C_ENG, alpha=0.15,
                             label=f"law engaged (predicted ≥ {engage_threshold_n:g} N)"))
    fig.suptitle(title, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=min(len(handles), 3), fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_action_comparison(
    action: np.ndarray,              # [T, J] rollout command (middle-layer action at frame end)
    reference_target: np.ndarray,    # [T, J] recorded joint_target (what actually worked)
    joint_names: list[str],
    out_path: str | Path,
    key_frames: dict | None = None,
    highlight: list[str] | None = None,   # joints to show (default: the hand joints)
    title: str = "commands vs predicted states",
    predicted: np.ndarray | None = None,  # [T, J] predicted state (policy reference at frame end)
) -> None:
    """Per joint: recorded command (ground truth) vs rollout command vs predicted state.

    Reading the plot: the gap between the *predicted state* (blue dashed, the policy's output and
    the middle layer's input) and the *recorded command* (black, what the replay actually sent to
    the PD) is the command lead the force law has to synthesise; the *rollout command* (red, what
    the middle layer actually sent) shows how much of it was recovered. With the null law red
    lies on blue; with a perfect law red lies on black wherever contact forces matter.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if highlight is None:
        highlight = [n for n in joint_names if n.startswith("leap_")]
    cols = 4
    rows = int(np.ceil(len(highlight) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.2 * rows + 0.6), sharex=True)
    axes = np.atleast_2d(axes)
    t = np.arange(action.shape[0])
    for k, name in enumerate(highlight):
        ax = axes[k // cols, k % cols]
        j = joint_names.index(name)
        ax.plot(t, reference_target[:, j], color="0.15", lw=1.5)
        if predicted is not None:
            ax.plot(t, predicted[:, j], color="tab:blue", lw=1.0, ls="--", alpha=0.9)
        ax.plot(t, action[:, j], color="tab:red", lw=1.0, alpha=0.9)
        ax.set_title(name, fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        _mark_key_frames(ax, key_frames)
    for k in range(len(highlight), rows * cols):
        axes[k // cols, k % cols].axis("off")

    handles = [
        Line2D([], [], color="0.15", lw=1.5, label="recorded command (episode ground truth)"),
        Line2D([], [], color="tab:red", lw=1.2, label="rollout command (middle-layer output)"),
    ]
    if predicted is not None:
        handles.insert(1, Line2D([], [], color="tab:blue", lw=1.2, ls="--",
                                 label="predicted state (policy reference)"))
    fig.suptitle(title, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=len(handles), fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _vector_error_split(f_ref_vec: np.ndarray, f_meas_vec: np.ndarray, eps: float = 1e-9):
    """Split e = f_meas - f_ref into components parallel/perpendicular to the reference.

    Returns ``(total, par_signed, perp, angle_deg)`` with shapes ``[N, tips]``;
    ``total^2 == par^2 + perp^2`` exactly. The angle is NaN where either vector is tiny.
    """
    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]).astype(np.float64)
    fm = f_meas_vec.reshape(-1, *f_meas_vec.shape[-2:]).astype(np.float64)
    ref_mag = np.linalg.norm(fr, axis=-1)
    meas_mag = np.linalg.norm(fm, axis=-1)
    unit = fr / np.maximum(ref_mag, eps)[..., None]
    proj = np.sum(fm * unit, axis=-1)
    par_signed = proj - ref_mag                       # + = stronger than predicted (along f_ref)
    perp = np.linalg.norm(fm - proj[..., None] * unit, axis=-1)
    total = np.linalg.norm(fm - fr, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(fr * fm, axis=-1) / np.maximum(ref_mag * meas_mag, eps)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    angle[(ref_mag < 1.0) | (meas_mag < 1.0)] = np.nan
    return total, par_signed, perp, angle


def plot_vector_error(
    f_ref_vec: np.ndarray,           # [T, S, 4, 3] predicted force vectors (world)
    f_meas_vec: np.ndarray,          # same shape: measured (filtered) vectors
    fingertip_bodies: list[str],
    out_path: str | Path,
    steps_per_frame: int,
    key_frames: dict | None = None,
    engage_threshold_n: float | None = None,
    title: str = "vector force-tracking error",
) -> None:
    """|f_ref_vec - f_meas_vec| over time, split into strength vs direction components.

    black = total vector error norm; orange = strength part (|e_par|: right direction, wrong
    magnitude); purple = direction part (e_perp: force pointing the wrong way); grey dashed
    (right axis) = angle between the vectors in degrees. total^2 = strength^2 + direction^2.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    total, par, perp, angle = _vector_error_split(f_ref_vec, f_meas_vec)
    ref_mag = np.linalg.norm(f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]), axis=-1)
    t = np.arange(total.shape[0]) / steps_per_frame
    n_tips = len(fingertip_bodies)

    C_TOT, C_PAR, C_PERP, C_ANG, C_ENG = "0.15", "tab:orange", "tab:purple", "0.45", "tab:green"

    fig, axes = plt.subplots(n_tips, 1, figsize=(12, 2.3 * n_tips + 0.7), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, body) in enumerate(zip(axes, fingertip_bodies)):
        if engage_threshold_n is not None:
            engaged = ref_mag[:, i] >= engage_threshold_n
            if engaged.any():
                ax.fill_between(t, 0, 1, where=engaged, transform=ax.get_xaxis_transform(),
                                color=C_ENG, alpha=0.07, lw=0)
        ax.plot(t, total[:, i], color=C_TOT, lw=1.5)
        ax.plot(t, np.abs(par[:, i]), color=C_PAR, lw=1.0, alpha=0.9)
        ax.plot(t, perp[:, i], color=C_PERP, lw=1.0, alpha=0.9)
        ax.set_ylabel(f"{FINGERTIP_LABELS.get(body, body)}\nerror [N]")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(t, angle[:, i], color=C_ANG, lw=0.9, ls="--", alpha=0.8)
        ax2.set_ylabel("angle [deg]", color=C_ANG, fontsize=7)
        ax2.tick_params(axis="y", labelsize=6, colors=C_ANG)
        ax2.set_ylim(bottom=0)
        _mark_key_frames(ax, key_frames)
    axes[-1].set_xlabel("trajectory frame")

    handles = [
        Line2D([], [], color=C_TOT, lw=1.5, label="total |f_ref − f_meas|"),
        Line2D([], [], color=C_PAR, lw=1.2, label="strength part |e∥| (wrong magnitude)"),
        Line2D([], [], color=C_PERP, lw=1.2, label="direction part |e⊥| (wrong direction)"),
        Line2D([], [], color=C_ANG, lw=1.2, ls="--", label="angle between vectors (right axis)"),
    ]
    if engage_threshold_n is not None:
        handles.append(Patch(color=C_ENG, alpha=0.15,
                             label=f"law engaged (predicted ≥ {engage_threshold_n:g} N)"))
    fig.suptitle(title, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_force_components(
    f_ref_vec: np.ndarray,           # [T, S, 4, 3] predicted force vectors (world)
    f_meas_vec: np.ndarray,          # same shape: measured (filtered) vectors
    fingertip_bodies: list[str],
    out_path: str | Path,
    steps_per_frame: int,
    key_frames: dict | None = None,
    engage_threshold_n: float = 0.2,
    title: str = "force vector components (world frame)",
) -> None:
    """Predicted vs measured force per world axis - shows *where* the direction deviates.

    Same colour semantics as force_tracking.png: blue = predicted, red = measured, grey fill =
    the gap. One row per fingertip that is ever engaged; columns are world X / Y / Z.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:])
    fm = f_meas_vec.reshape(-1, *f_meas_vec.shape[-2:])
    ref_mag = np.linalg.norm(fr, axis=-1)
    t = np.arange(fr.shape[0]) / steps_per_frame

    shown = [i for i in range(len(fingertip_bodies)) if (ref_mag[:, i] >= engage_threshold_n).any()]
    if not shown:
        shown = list(range(len(fingertip_bodies)))
    skipped = [fingertip_bodies[i] for i in range(len(fingertip_bodies)) if i not in shown]

    C_REF, C_MEAS, C_ERR = "tab:blue", "tab:red", "0.55"
    fig, axes = plt.subplots(len(shown), 3, figsize=(13, 2.2 * len(shown) + 0.8),
                             sharex=True, squeeze=False)
    for row, i in enumerate(shown):
        for col, axis_name in enumerate(["X", "Y", "Z"]):
            ax = axes[row, col]
            ax.fill_between(t, fm[:, i, col], fr[:, i, col], color=C_ERR, alpha=0.30, lw=0)
            ax.plot(t, fr[:, i, col], color=C_REF, lw=1.4)
            ax.plot(t, fm[:, i, col], color=C_MEAS, lw=0.8, alpha=0.9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"world {axis_name}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{FINGERTIP_LABELS.get(fingertip_bodies[i], fingertip_bodies[i])}\n[N]")
            _mark_key_frames(ax, key_frames)
    for col in range(3):
        axes[-1, col].set_xlabel("trajectory frame", fontsize=8)

    handles = [
        Line2D([], [], color=C_REF, lw=1.4, label="predicted force (policy target)"),
        Line2D([], [], color=C_MEAS, lw=1.2, label="measured force (sim sensor, filtered)"),
        Patch(color=C_ERR, alpha=0.30, label="gap"),
    ]
    subtitle = f" — fingertips never engaged not shown: {', '.join(skipped)}" if skipped else ""
    fig.suptitle(title + subtitle, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_contact_point_error(
    p_ref: np.ndarray,               # [T, S, 4, 3] predicted contact points (world), NaN-padded
    p_meas: np.ndarray,              # same shape: measured (filtered) pad centroids
    f_ref_vec: np.ndarray,           # same shape: predicted force vectors (engagement + split)
    fingertip_bodies: list[str],
    out_path: str | Path,
    steps_per_frame: int,
    key_frames: dict | None = None,
    engage_threshold_n: float = 0.2,
    title: str = "contact-point tracking error",
) -> None:
    """|p_ref - p_meas| over time in mm, split into tangential vs normal components.

    black = total contact-point error; purple = tangential part (where on the surface the finger
    sits — what the contact-point channel regulates); orange = normal part (depth of press, which
    the force channel owns). Green shading = contact commanded; red shading = commanded but the
    pad reports no contact at all, so there is no point to track.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pr = p_ref.reshape(-1, *p_ref.shape[-2:]).astype(np.float64)
    pm = p_meas.reshape(-1, *p_meas.shape[-2:]).astype(np.float64)
    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]).astype(np.float64)
    ref_mag = np.linalg.norm(fr, axis=-1)
    engaged = ref_mag >= engage_threshold_n
    both = np.isfinite(pr).all(axis=-1) & np.isfinite(pm).all(axis=-1)

    delta = pr - pm
    unit = fr / np.maximum(ref_mag, 1e-9)[..., None]
    along = np.sum(delta * unit, axis=-1)
    total = np.linalg.norm(delta, axis=-1) * 1000.0
    normal = np.abs(along) * 1000.0
    tangential = np.linalg.norm(delta - along[..., None] * unit, axis=-1) * 1000.0
    for arr in (total, normal, tangential):
        arr[~both] = np.nan

    t = np.arange(pr.shape[0]) / steps_per_frame
    n_tips = len(fingertip_bodies)
    C_TOT, C_TAN, C_NRM, C_ENG, C_MISS = "0.15", "tab:purple", "tab:orange", "tab:green", "tab:red"

    fig, axes = plt.subplots(n_tips, 1, figsize=(12, 2.3 * n_tips + 0.7), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, body) in enumerate(zip(axes, fingertip_bodies)):
        eng = engaged[:, i]
        if eng.any():
            ax.fill_between(t, 0, 1, where=eng, transform=ax.get_xaxis_transform(),
                            color=C_ENG, alpha=0.07, lw=0)
        miss = eng & ~both[:, i]
        if miss.any():
            ax.fill_between(t, 0, 1, where=miss, transform=ax.get_xaxis_transform(),
                            color=C_MISS, alpha=0.10, lw=0)
        ax.plot(t, total[:, i], color=C_TOT, lw=1.5)
        ax.plot(t, tangential[:, i], color=C_TAN, lw=1.0, alpha=0.9)
        ax.plot(t, normal[:, i], color=C_NRM, lw=1.0, alpha=0.9)
        ax.set_ylabel(f"{FINGERTIP_LABELS.get(body, body)}\n|Δp| [mm]")
        ax.grid(alpha=0.25)
        _mark_key_frames(ax, key_frames)
    axes[-1].set_xlabel("trajectory frame")

    handles = [
        Line2D([], [], color=C_TOT, lw=1.5, label="total |p_ref − p_meas|"),
        Line2D([], [], color=C_TAN, lw=1.2, label="tangential (contact-point channel)"),
        Line2D([], [], color=C_NRM, lw=1.2, label="normal (depth, force channel)"),
        Patch(color=C_ENG, alpha=0.15, label=f"contact commanded (≥ {engage_threshold_n:g} N)"),
        Patch(color=C_MISS, alpha=0.20, label="commanded but pad not touching"),
    ]
    fig.suptitle(title, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_force_error_components(
    f_ref_vec: np.ndarray,           # [T, S, 4, 3] predicted force vectors (world)
    f_meas_vec: np.ndarray,          # same shape: measured (filtered) vectors
    fingertip_bodies: list[str],
    out_path: str | Path,
    steps_per_frame: int,
    key_frames: dict | None = None,
    engage_threshold_n: float = 0.2,
    title: str = "force-tracking error per world axis",
) -> None:
    """Signed force-tracking error ``f_meas - f_ref`` on each world axis, one row per fingertip.

    Where :func:`plot_force_components` shows the two *signals* per axis, this shows the *error*
    directly, so a persistent bias on one axis is immediately visible instead of having to be read
    off as a gap between two large curves. Below zero = the measured force is weaker than
    predicted along that axis; above = stronger. Rows share a y-scale so the three axes are
    directly comparable, and each panel is annotated with its bias and RMS over engaged steps.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fr = f_ref_vec.reshape(-1, *f_ref_vec.shape[-2:]).astype(np.float64)
    fm = f_meas_vec.reshape(-1, *f_meas_vec.shape[-2:]).astype(np.float64)
    err = fm - fr
    ref_mag = np.linalg.norm(fr, axis=-1)
    t = np.arange(fr.shape[0]) / steps_per_frame

    shown = [i for i in range(len(fingertip_bodies)) if (ref_mag[:, i] >= engage_threshold_n).any()]
    if not shown:
        shown = list(range(len(fingertip_bodies)))
    skipped = [fingertip_bodies[i] for i in range(len(fingertip_bodies)) if i not in shown]

    C_ERR, C_POS, C_NEG, C_ENG = "0.15", "tab:red", "tab:blue", "tab:green"
    fig, axes = plt.subplots(len(shown), 3, figsize=(13, 2.3 * len(shown) + 0.9),
                             sharex=True, sharey="row", squeeze=False)
    for row, i in enumerate(shown):
        engaged = ref_mag[:, i] >= engage_threshold_n
        for col, axis_name in enumerate(["X", "Y", "Z"]):
            ax = axes[row, col]
            e = err[:, i, col]
            if engaged.any():
                ax.fill_between(t, 0, 1, where=engaged, transform=ax.get_xaxis_transform(),
                                color=C_ENG, alpha=0.07, lw=0)
            ax.fill_between(t, 0, e, where=e >= 0, color=C_POS, alpha=0.35, lw=0, interpolate=True)
            ax.fill_between(t, 0, e, where=e < 0, color=C_NEG, alpha=0.35, lw=0, interpolate=True)
            ax.plot(t, e, color=C_ERR, lw=0.9)
            ax.axhline(0.0, color="0.4", lw=0.8, zorder=1)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if engaged.any():
                ee = e[engaged]
                ax.text(0.015, 0.04, f"bias {ee.mean():+.2f} N   rms {np.sqrt((ee**2).mean()):.2f} N",
                        transform=ax.transAxes, fontsize=7, color="0.25",
                        bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.5))
            if row == 0:
                ax.set_title(f"world {axis_name}", fontsize=9)
            if col == 0:
                label = FINGERTIP_LABELS.get(fingertip_bodies[i], fingertip_bodies[i])
                ax.set_ylabel(f"{label}\nf_meas − f_ref [N]")
            _mark_key_frames(ax, key_frames)
    for col in range(3):
        axes[-1, col].set_xlabel("trajectory frame", fontsize=8)

    handles = [
        Line2D([], [], color=C_ERR, lw=1.2, label="error  f_meas − f_ref"),
        Patch(color=C_POS, alpha=0.35, label="measured stronger than predicted (+)"),
        Patch(color=C_NEG, alpha=0.35, label="measured weaker than predicted (−)"),
        Patch(color=C_ENG, alpha=0.15, label=f"contact commanded (≥ {engage_threshold_n:g} N)"),
    ]
    subtitle = f" — fingertips never engaged not shown: {', '.join(skipped)}" if skipped else ""
    fig.suptitle(title + subtitle, fontsize=11)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=4, fontsize=8, framealpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def force_tracking_metrics(
    f_ref: np.ndarray,               # [T, S, 4]
    f_meas: np.ndarray,              # [T, S, 4]
    fingertip_bodies: list[str],
    engage_threshold_n: float = 0.2,
) -> dict:
    """RMSE / bias of measured vs predicted force, overall and while contact is commanded."""
    f_ref2 = f_ref.reshape(-1, f_ref.shape[-1])
    f_meas2 = f_meas.reshape(-1, f_meas.shape[-1])
    err = f_meas2 - f_ref2
    engaged = f_ref2 >= engage_threshold_n
    out = {"bodies": list(fingertip_bodies), "engage_threshold_n": engage_threshold_n}
    out["rmse_N"] = [float(v) for v in np.sqrt((err**2).mean(axis=0))]
    rmse_engaged, bias_engaged, frac = [], [], []
    for i in range(err.shape[1]):
        mask = engaged[:, i]
        frac.append(float(mask.mean()))
        if mask.any():
            rmse_engaged.append(float(np.sqrt((err[mask, i] ** 2).mean())))
            bias_engaged.append(float(err[mask, i].mean()))
        else:
            rmse_engaged.append(0.0)
            bias_engaged.append(0.0)
    out["rmse_engaged_N"] = rmse_engaged
    out["bias_engaged_N"] = bias_engaged                # negative = under-squeezing
    out["engaged_fraction"] = frac
    return out
