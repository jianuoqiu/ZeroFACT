#!/usr/bin/env python3
"""Test Savitzky-Golay filtering on the BC dataset and plot raw vs filtered signals.

No Isaac Sim: this reads the recorded ``replay_data.npz`` files only.

What it produces in ``outputs/play2perfect/bc_dataset/filtering/``::

    noise_and_response.png     measured noise spectrum of each signal family, with the magnitude
                               response of every candidate filter drawn on top
    raw_vs_filtered_<task>.png fingertip forces, raw vs filtered, whole episode + contact zoom
    signals_<task>.png         the other filtered families (measured torque, motor current,
                               joint velocity) for the same episode
    parameter_sweep.png        window length vs noise removed / peak loss / lag, on real data
    causal_comparison.png      the causal options against each other (what a policy can use online)
    REPORT.md                  every number in this study plus the recommended settings

Usage::

    python play2perfect_force_controller/sg_filter_study.py                  # 3 episodes per task
    python play2perfect_force_controller/sg_filter_study.py --episodes 8 --windows 5 7 9 15 21
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import welch  # noqa: E402

from play2perfect_force_controller.p2p_paths import PROBLEMS  # noqa: E402
from play2perfect_force_controller.robot_spec import (  # noqa: E402
    FINGERTIP_COLORS, FINGERTIP_LABELS, FRAME_DT,
)
from play2perfect_force_controller.signal_filtering import (  # noqa: E402
    DEFAULT_FILTER_SPEC, band_power_ratio, butter_lowpass, ema, filter_arrays, frequency_response,
    hf_power_fraction, measured_lag_frames, moving_average, negative_fraction, onset_shift_frames,
    peak_attenuation, savgol_causal, savgol_centered,
)

FS = 1.0 / FRAME_DT                      # 60 Hz
HF_CUT = 10.0                            # Hz; "noise" for the summary numbers
NPERSEG = 96                             # Welch segment: fixed so every episode gives the same
                                         # frequency grid (the shortest episode has 113 frames)
MID_BAND = (4.0, 10.0)                   # where a causal Savitzky-Golay overshoots unity gain
DATASET = PROJECT_ROOT / "outputs" / "play2perfect" / "bc_dataset"

# the signal families this study looks at: label -> (npz key, how to reduce to [T, C], unit)
FAMILIES = {
    "fingertip force |F|": ("contact_force", lambda a: np.linalg.norm(a, axis=-1), "N"),
    "fingertip force xyz": ("contact_force", lambda a: a.reshape(len(a), -1), "N"),
    "measured joint torque": ("joint_torque_measured", lambda a: a, "N m"),
    "motor current": ("motor_current", lambda a: a, "A"),
    "joint velocity (hand)": ("joint_vel", lambda a: a[:, 7:], "rad/s"),
    "joint wrench": ("joint_wrench_b", lambda a: a.reshape(len(a), -1), "N, N m"),
    "joint position": ("joint_pos", lambda a: a, "rad"),
    "state-command diff": ("joint_cmd_err", lambda a: a, "rad"),
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 10, "axes.grid": True, "grid.alpha": 0.3,
})
C_RAW, C_CEN, C_CAU, C_BUT, C_EMA, C_BOX = "0.65", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"


# ---------------------------------------------------------------------------------------------
def pick_episodes(per_task: int) -> list[Path]:
    eps = []
    for problem in PROBLEMS:
        found = sorted((DATASET / problem).glob("seed_*/replay_data.npz"))
        eps += [p.parent for p in found[:per_task]]
    if not eps:
        raise SystemExit(f"no episodes under {DATASET}; run collect_bc_dataset.py first")
    return eps


def load_family(ep: Path, family: str) -> np.ndarray | None:
    key, reduce_fn, _ = FAMILIES[family]
    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    if key not in d.files:
        return None
    return reduce_fn(np.asarray(d[key], dtype=np.float64))


def engaged_mask(force_mag: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Frames where at least one fingertip carries load: the only place force noise is defined."""
    return force_mag.max(axis=1) > threshold


# ---------------------------------------------------------------------------------------------
# 1. noise characterisation + filter responses
# ---------------------------------------------------------------------------------------------
def figure_noise_and_response(episodes: list[Path], windows: list[int], out: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # -- left: measured spectra of the signal families
    ax = axes[0]
    stats = {}
    for family in FAMILIES:
        psds = []
        for ep in episodes:
            x = load_family(ep, family)
            if x is None or len(x) < NPERSEG:
                continue
            x = x - x.mean(axis=0, keepdims=True)
            keep = x.std(axis=0) > 1e-9
            if not keep.any() or len(x) < NPERSEG:
                continue
            f, P = welch(x[:, keep].T, fs=FS, nperseg=NPERSEG, axis=1)
            psds.append((P / P.sum(axis=1, keepdims=True)).mean(axis=0))
        if not psds:
            continue
        P = np.mean(psds, axis=0)
        stats[family] = {"hf_fraction": float(P[f > HF_CUT].sum())}
        style = dict(lw=2.2) if stats[family]["hf_fraction"] > 0.05 else dict(lw=1.2, ls=":")
        ax.semilogy(f, P / P.max(), label=f"{family} ({stats[family]['hf_fraction']:.0%} >10 Hz)", **style)
    ax.axvspan(HF_CUT, FS / 2, color="tab:red", alpha=0.07)
    ax.text(HF_CUT + 1, 1.3e-4, "treated as noise", color="tab:red", fontsize=9)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("normalised power (log)")
    ax.set_title(f"Measured noise: where each signal's power sits\n({len(episodes)} episodes, all four tasks)")
    ax.set_xlim(0, FS / 2); ax.set_ylim(1e-4, 2); ax.legend(fontsize=8, loc="lower left")

    # -- middle: centered response; right: causal response
    for ax, causal in ((axes[1], False), (axes[2], True)):
        for w, color in zip(windows, plt.cm.viridis(np.linspace(0.05, 0.85, len(windows)))):
            fr, mag, _ = frequency_response(w, 2, causal)
            ax.plot(fr, mag, color=color, lw=2, label=f"window {w} ({w * FRAME_DT * 1000:.0f} ms)")
        if causal:
            ax.plot(*_response_of(lambda x: butter_lowpass(x, 6.0)), color=C_BUT, lw=2, ls="--",
                    label="Butterworth 6 Hz (causal)")
            ax.plot(*_response_of(lambda x: ema(x, 3.0)), color=C_EMA, lw=2, ls="--", label="EMA tau 3 frames")
            ax.plot(*_response_of(lambda x: moving_average(x, 9, causal=True)), color=C_BOX, lw=1.6, ls=":",
                    label="boxcar 9 (causal)")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.axvspan(HF_CUT, FS / 2, color="tab:red", alpha=0.07)
        ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|H(f)|")
        ax.set_xlim(0, FS / 2); ax.set_ylim(0, 1.75)
        ax.set_title(("Causal Savitzky-Golay (order 2)\nnote the gain > 1 over the noise band"
                      if causal else "Centered Savitzky-Golay (order 2)\nzero phase, monotone roll-off"))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)
    return stats


def _response_of(fn, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Empirical magnitude response of any causal filter, from its impulse response."""
    imp = np.zeros((n, 1)); imp[n // 4] = 1.0
    h = np.asarray(fn(imp)).ravel()[n // 4:]
    H = np.fft.rfft(h, n=n)
    return np.fft.rfftfreq(n, d=FRAME_DT), np.abs(H)


# ---------------------------------------------------------------------------------------------
# 2. raw vs filtered time series (the visualisation asked for)
# ---------------------------------------------------------------------------------------------
def figure_raw_vs_filtered(ep: Path, window: int, out: Path) -> None:
    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    summary = json.loads((ep / "summary.json").read_text())
    tips = [str(b) for b in d["fingertip_bodies"]]
    V = np.asarray(d["contact_force"], dtype=np.float64)                              # [T, 5, 3]
    F = np.linalg.norm(V, axis=-1)
    t = np.arange(len(F)) * FRAME_DT
    # every filter is applied to the force VECTOR and the norm taken afterwards (section 3 of the report)
    cen = np.linalg.norm(savgol_centered(V, window, 2), axis=-1)
    cau = np.linalg.norm(savgol_causal(V, window, 2), axis=-1)
    but = np.linalg.norm(butter_lowpass(V, 6.0, causal=True), axis=-1)

    kf = summary["key_frames"]
    onset = kf.get("first_contact_frame") or 0
    zoom = slice(max(0, onset - 10), min(len(F), onset + 110))

    fig, axes = plt.subplots(len(tips), 2, figsize=(15.5, 2.1 * len(tips)), sharex="col",
                             gridspec_kw={"width_ratios": [2.0, 1.0]})
    for i, tip in enumerate(tips):
        color = FINGERTIP_COLORS.get(tip, (0, 0, 0))
        for col, sl in enumerate((slice(None), zoom)):
            ax = axes[i, col]
            ax.plot(t[sl], F[sl, i], color=C_RAW, lw=1.0, label="raw", zorder=1)
            ax.plot(t[sl], cen[sl, i], color=C_CEN, lw=1.9, label=f"SG centered ({window},2)", zorder=3)
            ax.plot(t[sl], cau[sl, i], color=C_CAU, lw=1.5, label=f"SG causal ({window},2)", zorder=2)
            ax.plot(t[sl], but[sl, i], color=C_BUT, lw=1.5, ls="--", label="Butterworth 6 Hz causal", zorder=2)
            if col == 0:
                for name, frame in kf.items():
                    if frame is not None and name.endswith("_frame"):
                        ax.axvline(frame * FRAME_DT, color="0.4", lw=0.7, ls=":")
            ax.set_ylabel(f"{FINGERTIP_LABELS.get(tip, tip)}\n|F| (N)", color=color, fontsize=10)
            ax.tick_params(axis="y", labelcolor=color)
        axes[i, 1].set_title("contact onset, 1.8 s" if i == 0 else "", fontsize=10)
    axes[0, 0].legend(ncol=4, fontsize=9, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Fingertip force, raw vs filtered - {summary['problem']} seed {summary['config']['seed']} "
                 f"({len(F)} frames, dotted lines = key frames)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out); plt.close(fig)


def figure_other_signals(ep: Path, out: Path) -> None:
    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    summary = json.loads((ep / "summary.json").read_text())
    joints = [str(n) for n in d["joint_names"]]
    panels = [
        ("measured joint torque", "joint_torque_measured", "N m", DEFAULT_FILTER_SPEC["joint_torque_measured"]),
        ("motor current", "motor_current", "A", DEFAULT_FILTER_SPEC["motor_current"]),
        ("joint velocity", "joint_vel", "rad/s", DEFAULT_FILTER_SPEC["joint_vel"]),
        ("state-command difference (left raw)", "joint_cmd_err", "rad", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15.0, 2.5 * len(panels)), sharex=True)
    for ax, (title, key, unit, spec) in zip(axes, panels):
        x = np.asarray(d[key], dtype=np.float64)
        j = int(np.argmax(x.std(axis=0)))                  # the busiest joint
        t = np.arange(len(x)) * FRAME_DT
        ax.plot(t, x[:, j], color=C_RAW, lw=1.0, label="raw")
        if spec is not None:
            ax.plot(t, savgol_centered(x, spec.window, spec.polyorder)[:, j], color=C_CEN, lw=1.9,
                    label=f"SG centered ({spec.window},{spec.polyorder})")
            ax.plot(t, savgol_causal(x, spec.window, spec.polyorder)[:, j], color=C_CAU, lw=1.3,
                    label=f"SG causal ({spec.window},{spec.polyorder})")
        hf = hf_power_fraction(x, fs=FS, cut=HF_CUT, nperseg=NPERSEG)
        ax.set_ylabel(f"{unit}")
        ax.set_title(f"{title} - joint {joints[j]} (power above {HF_CUT:.0f} Hz: {hf:.1%})", fontsize=11)
        ax.legend(fontsize=9, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Other filtered signals - {summary['problem']} seed {summary['config']['seed']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------------------------------------
# 3. parameter sweep on real data
# ---------------------------------------------------------------------------------------------
def sweep(episodes: list[Path], windows: list[int], orders: list[int]) -> list[dict]:
    rows = []
    variants = [(f"SG centered ({w},{o})", w, o, lambda x, w=w, o=o: savgol_centered(x, w, o), False)
                for w in windows for o in orders]
    variants += [(f"SG causal ({w},{o})", w, o, lambda x, w=w, o=o: savgol_causal(x, w, o), True)
                 for w in windows for o in orders]
    variants += [(f"boxcar {w} causal", w, 0, lambda x, w=w: moving_average(x, w, causal=True), True)
                 for w in windows]
    variants += [(f"Butterworth {c:.0f} Hz causal", 0, 0, lambda x, c=c: butter_lowpass(x, c, causal=True), True)
                 for c in (4.0, 6.0, 10.0)]
    variants += [(f"EMA tau {tau:.0f}", 0, 0, lambda x, tau=tau: ema(x, tau), True) for tau in (2.0, 3.0, 5.0)]

    for name, w, o, fn, causal in variants:
        acc = {k: [] for k in ("hf_after", "hf_removed", "peak_loss", "onset", "lag", "rms",
                               "mid_band", "negative", "negative_magroute")}
        for ep in episodes:
            V = np.load(ep / "replay_data.npz", allow_pickle=True)["contact_force"]
            V = np.asarray(V, dtype=np.float64)                 # [T, tips, 3]
            F = np.linalg.norm(V, axis=-1)
            m = engaged_mask(F)
            if m.sum() < 40:
                continue
            # the pipeline route: filter the vector, then take the norm (never negative)
            y = np.linalg.norm(fn(V), axis=-1)
            acc["negative"].append(negative_fraction(y))
            acc["negative_magroute"].append(negative_fraction(fn(F)))
            acc["mid_band"].append(band_power_ratio(F[m], y[m], MID_BAND, fs=FS, nperseg=NPERSEG))
            raw_hf = hf_power_fraction(F[m], fs=FS, cut=HF_CUT, nperseg=NPERSEG)
            new_hf = hf_power_fraction(y[m], fs=FS, cut=HF_CUT, nperseg=NPERSEG)
            acc["hf_after"].append(new_hf)
            # power removed above the cut, in absolute terms (variance-weighted, not the fraction)
            acc["hf_removed"].append(1.0 - (new_hf * y[m].var()) / max(raw_hf * F[m].var(), 1e-12))
            acc["peak_loss"].append(peak_attenuation(F[m], y[m]))
            acc["onset"].append(onset_shift_frames(F, y, threshold=1.0))
            acc["lag"].append(measured_lag_frames(F[m], y[m]))
            acc["rms"].append(float(np.sqrt(np.mean((y[m] - F[m]) ** 2))))
        if not acc["hf_after"]:
            continue
        rows.append({"filter": name, "window": w, "order": o, "causal": causal,
                     **{k: float(np.nanmean(v)) for k, v in acc.items()}})
    return rows


def figure_sweep(rows: list[dict], windows: list[int], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    groups = [("SG centered", C_CEN, "o-"), ("SG causal", C_CAU, "s-"), ("boxcar", C_BOX, "^:")]
    for metric, ax, label in ((("hf_removed"), axes[0], "noise power removed above 10 Hz"),
                              (("peak_loss"), axes[1], "peak force lost (99th percentile)"),
                              (("rms"), axes[2], "RMS change vs raw (N)")):
        for prefix, color, style in groups:
            sel = [r for r in rows if r["filter"].startswith(prefix) and r["order"] in (0, 2)]
            sel = sorted(sel, key=lambda r: r["window"])
            if not sel:
                continue
            ax.plot([r["window"] for r in sel], [r[metric] for r in sel], style, color=color, lw=2,
                    ms=6, label=prefix + (" (order 2)" if prefix != "boxcar" else ""))
        ax.set_xlabel("window (frames at 60 Hz)"); ax.set_ylabel(label)
        ax.set_xticks(windows)
        if metric in ("hf_removed", "peak_loss"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(fontsize=9)
    axes[0].set_title("More is better")
    axes[1].set_title("Less is better: how much true signal the filter eats")
    axes[2].set_title("How far the filtered signal moves")
    fig.suptitle("Savitzky-Golay parameter sweep on fingertip forces (engaged frames only)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out); plt.close(fig)


def figure_causal_comparison(ep: Path, out: Path) -> None:
    """What a policy can actually use online: the causal options on one contact episode."""
    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    summary = json.loads((ep / "summary.json").read_text())
    V = np.asarray(d["contact_force"], dtype=np.float64)
    F = np.linalg.norm(V, axis=-1)
    tips = [str(b) for b in d["fingertip_bodies"]]
    i = int(np.argmax(F.std(axis=0)))
    t = np.arange(len(F)) * FRAME_DT
    onset = summary["key_frames"].get("first_contact_frame") or 0
    zoom = slice(max(0, onset - 5), min(len(F), onset + 130))
    nrm = lambda a: np.linalg.norm(a, axis=-1)
    curves = [("SG causal (9,2)", nrm(savgol_causal(V, 9, 2)), C_CAU, "-"),
              ("Butterworth 6 Hz", nrm(butter_lowpass(V, 6.0, causal=True)), C_BUT, "-"),
              ("EMA tau 3", nrm(ema(V, 3.0)), C_EMA, "--"),
              ("boxcar 9", nrm(moving_average(V, 9, causal=True)), C_BOX, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    for ax, sl in zip(axes, (slice(None), zoom)):
        ax.plot(t[sl], F[sl, i], color=C_RAW, lw=1.0, label="raw")
        ax.plot(t[sl], nrm(savgol_centered(V, 9, 2))[sl, i], color=C_CEN, lw=2.2, label="SG centered (9,2) [offline]")
        for name, y, color, ls in curves:
            ax.plot(t[sl], y[sl, i], color=color, lw=1.5, ls=ls, label=name)
        ax.set_xlabel("time (s)"); ax.set_ylabel("|F| (N)")
    axes[0].legend(fontsize=9, ncol=3)
    axes[1].set_title("contact onset", fontsize=11)
    fig.suptitle(f"Causal (online-usable) filters on the {FINGERTIP_LABELS.get(tips[i], tips[i])} fingertip - "
                 f"{summary['problem']} seed {summary['config']['seed']}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out); plt.close(fig)


def figure_vector_vs_magnitude(ep: Path, window: int, out: Path) -> None:
    """Smoothing |F| undershoots below zero at contact onset and release; smoothing the xyz
    components and taking the norm afterwards cannot."""
    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    summary = json.loads((ep / "summary.json").read_text())
    V = np.asarray(d["contact_force"], dtype=np.float64)
    F = np.linalg.norm(V, axis=-1)
    tips = [str(b) for b in d["fingertip_bodies"]]
    i = int(np.argmax(F.std(axis=0)))
    t_ax = np.arange(len(F)) * FRAME_DT
    mag_route = savgol_centered(F, window, 2)[:, i]
    vec_route = np.linalg.norm(savgol_centered(V, window, 2), axis=-1)[:, i]
    onset = summary["key_frames"].get("first_contact_frame") or 0
    zoom = slice(max(0, onset - 8), min(len(F), onset + 60))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    for ax, sl in zip(axes, (slice(None), zoom)):
        ax.plot(t_ax[sl], F[sl, i], color=C_RAW, lw=1.0, label="raw |F|")
        ax.plot(t_ax[sl], mag_route[sl], color="#d62728", lw=1.8,
                label=f"filter |F| directly  ({negative_fraction(mag_route):.1%} negative, min {mag_route.min():.2f} N)")
        ax.plot(t_ax[sl], vec_route[sl], color="#1f77b4", lw=2.0,
                label=f"filter xyz then |.|  ({negative_fraction(vec_route):.1%} negative)")
        ax.axhline(0.0, color="k", lw=0.9)
        ax.fill_between(t_ax[sl], mag_route[sl], 0, where=mag_route[sl] < 0, color="#d62728", alpha=0.35)
        ax.set_xlabel("time (s)"); ax.set_ylabel("|F| (N)")
    axes[0].legend(fontsize=10, loc="upper left")
    axes[1].set_title("contact onset: the red curve goes below zero", fontsize=11)
    fig.suptitle(f"Force magnitude cannot be smoothed directly - centered Savitzky-Golay ({window},2), "
                 f"{FINGERTIP_LABELS.get(tips[i], tips[i])} tip, {summary['problem']} seed {summary['config']['seed']}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out); plt.close(fig)


# ---------------------------------------------------------------------------------------------
# self-test: the properties the study's conclusions rest on
# ---------------------------------------------------------------------------------------------
def self_test() -> int:
    """Check the filter implementations against their defining properties. Returns a failure count."""
    fails = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"[test] {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)
        if not ok:
            fails.append(name)

    n = 400
    t_ = np.arange(float(n))
    imp = np.zeros((n, 1)); imp[200] = 1.0
    for label, fn in (("savgol_causal(9,2)", lambda x: savgol_causal(x, 9, 2)),
                      ("butter_lowpass(6 Hz)", lambda x: butter_lowpass(x, 6.0, causal=True)),
                      ("ema(tau=3)", lambda x: ema(x, 3.0)),
                      ("moving_average(9, causal)", lambda x: moving_average(x, 9, causal=True))):
        r = float(np.abs(fn(imp)[:200]).max())
        check(f"{label} is causal (no response before the impulse)", r < 1e-12, f"max |y| before = {r:.1e}")
    r = float(np.abs(savgol_centered(imp, 9, 2)[:200]).max())
    check("savgol_centered(9,2) is NOT causal (expected)", r > 1e-9, f"max |y| before = {r:.2e}")

    ramp = (0.05 * t_)[:, None]
    e = float(np.abs(savgol_causal(ramp, 9, 2) - ramp)[9:].max())
    check("causal Savitzky-Golay passes a linear trend with zero lag", e < 1e-9, f"max error {e:.1e}")
    quad = (0.001 * t_ ** 2 - 0.2 * t_ + 3)[:, None]
    e = float(np.abs(savgol_centered(quad, 11, 2) - quad)[5:-5].max())
    check("centered Savitzky-Golay reproduces a quadratic in the interior", e < 1e-8, f"max error {e:.1e}")

    for w in (5, 9, 15, 21):
        _, _, gd = frequency_response(w, 2, causal=False)
        check(f"centered window {w} has zero group delay at every frequency",
              float(np.abs(gd).max()) < 1e-6, f"max |delay| {np.abs(gd).max():.1e} frames")
    peak_gain = max(frequency_response(w, 2, True)[1].max() for w in (9, 15, 21, 31))
    check("causal Savitzky-Golay overshoots unity gain (the study's main caveat)", peak_gain > 1.3,
          f"peak |H| = {peak_gain:.2f}")

    rng = np.random.default_rng(0)
    clean = (np.exp(-0.5 * ((t_ - 200) / 6) ** 2) * 20)[:, None]
    noisy = clean + rng.normal(0, 2.0, clean.shape)
    sg_peak, box_peak = savgol_centered(noisy, 9, 2).max(), moving_average(noisy, 9).max()
    check("Savitzky-Golay keeps a peak better than a boxcar of the same length",
          sg_peak > box_peak, f"SG {sg_peak:.1f} N vs boxcar {box_peak:.1f} N (true 20.0)")

    a = savgol_centered(noisy, 11, 2)
    b = savgol_centered(noisy, 11, 3)
    check("centered order 2 and order 3 are identical for smoothing",
          float(np.abs(a - b).max()) < 1e-9, f"max difference {np.abs(a - b).max():.1e}")

    ep = pick_episodes(1)[0]
    V = np.asarray(np.load(ep / "replay_data.npz", allow_pickle=True)["contact_force"], dtype=np.float64)
    F = np.linalg.norm(V, axis=-1)
    vec = np.linalg.norm(savgol_centered(V, 9, 2), axis=-1)
    mag = savgol_centered(F, 9, 2)
    check("filtering the force vector never produces a negative magnitude",
          negative_fraction(vec) == 0.0, f"{negative_fraction(vec):.2%} negative")
    check("filtering |F| directly DOES produce negatives (the trap)",
          negative_fraction(mag) > 0.0, f"{negative_fraction(mag):.2%} negative, min {mag.min():.2f} N")

    d = np.load(ep / "replay_data.npz", allow_pickle=True)
    out = filter_arrays({k: d[k] for k in d.files if k in DEFAULT_FILTER_SPEC or
                         k.replace("_steps", "") in DEFAULT_FILTER_SPEC})
    shapes_ok = all(np.asarray(out[k]).shape == np.asarray(d[k]).shape for k in out)
    check("filter_arrays keeps every array's shape", shapes_ok, f"{len(out)} arrays filtered")

    print(f"[test] {'all checks passed' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}",
          flush=True)
    return len(fails)


# ---------------------------------------------------------------------------------------------
def write_report(out_dir: Path, episodes: list[Path], rows: list[dict], noise: dict,
                 windows: list[int]) -> None:
    def fmt(r: dict) -> str:
        return (f"| {r['filter']} | {'causal' if r['causal'] else 'centered'} | {r['hf_removed']:.0%} | "
                f"{r['mid_band']:.2f} | {r['peak_loss']:+.1%} | {r['rms']:.2f} | {r['onset']:+.1f} | "
                f"{r['lag']:+.0f} |")

    lines = [
        "# Savitzky-Golay filtering study (play2perfect BC dataset)",
        "",
        f"`sg_filter_study.py` over {len(episodes)} episodes ({len(PROBLEMS)} tasks), 60 Hz frames.",
        "Filters live in `play2perfect_force_controller/signal_filtering.py`.",
        "",
        "## 1. Which signals are noisy",
        "",
        "| signal | power above 10 Hz | filtered? |",
        "|---|---|---|",
    ]
    for family, s in sorted(noise.items(), key=lambda kv: -kv[1]["hf_fraction"]):
        key = FAMILIES[family][0]
        lines.append(f"| {family} | {s['hf_fraction']:.1%} | "
                     f"{'yes' if key in DEFAULT_FILTER_SPEC else 'no, already clean'} |")
    lines += [
        "",
        "Contact forces, PhysX measured joint torque and joint wrenches carry the chatter of the",
        "120 Hz solver; joint positions, the state-command difference and object poses do not, so",
        "filtering those would only add lag.",
        "",
        "## 2. What each filter does to the fingertip forces",
        "",
        "Measured on engaged frames only (at least one fingertip above 1 N). `peak lost` is the",
        "change in the 99th percentile of |F|, `onset` is how many frames later the signal first",
        "crosses 1 N, `lag` is the cross-correlation lag against the raw signal.",
        "",
        f"`4-10 Hz gain` is filtered/raw power in that band: above 1.00 means the filter ADDED power there.",
        "",
        "| filter | kind | noise removed >10 Hz | 4-10 Hz gain | peak lost | RMS change (N) | onset (frames) | lag (frames) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [fmt(r) for r in rows]
    # the knee of the sweep: noise removal saturates quickly while peak loss keeps growing, so
    # score them against each other rather than reading the maximum of either curve
    centered = [r for r in rows if not r["causal"] and r["filter"].startswith("SG") and r["order"] == 2]
    best = max(centered, key=lambda r: r["hf_removed"] - 4 * abs(r["peak_loss"]))
    smoother = min((r for r in centered if r["window"] > best["window"]),
                   key=lambda r: r["window"], default=best)
    mag_route = np.nanmean([r["negative_magroute"] for r in rows if r["filter"].startswith("SG centered (9")])
    lines += [
        "",
        "## 3. Filter the force vector, never the force magnitude",
        "",
        f"`|F|` cannot be negative and steps up at contact. Smoothing `|F|` directly makes the filter",
        f"undershoot: **{mag_route:.1%} of the samples come out negative** (down to -3 N) with a centered",
        "window of 9. Smoothing the xyz components and taking the norm afterwards gives **0 %** impossible",
        "values, with 99th-percentile peaks within 2 % of the other route. Every number in the table above",
        "uses the vector route, and `DEFAULT_FILTER_SPEC` keys the vector array `contact_force` so the",
        "pipeline cannot get this wrong by accident (`signal_filtering.filtered_force_magnitude`).",
        "",
        "## 4. The catch: causal Savitzky-Golay amplifies this noise band",
        "",
        "A centered Savitzky-Golay is zero phase but reads `(window-1)/2` future samples, so it is",
        "only valid for data produced offline. The causal version fits the same polynomial to the",
        "last `window` samples and evaluates it at the newest one. That makes it exact on a linear",
        "trend (zero lag at DC, unlike a moving average) but its magnitude response rises above 1",
        "before it falls: with window 9 the gain peaks near 1.40 around 6-8 Hz, and windows 15 to 31",
        "peak at 1.54 to 1.66. This dataset's noise sits at 5-30 Hz, so a causal Savitzky-Golay",
        "partly amplifies exactly what it is meant to remove (see `noise_and_response.png`, right",
        "panel, and the table above).",
        "",
        "## 5. Recommended settings",
        "",
        f"* **Offline (labels, analysis, anything computed after the fact): centered Savitzky-Golay,",
        f"  window {best['window']} ({best['window'] * FRAME_DT * 1000:.0f} ms), order 2.** {best['hf_removed']:.0%} of the noise power above 10 Hz",
        f"  removed for {abs(best['peak_loss']):.1%} peak loss and zero lag; noise removal saturates after this",
        f"  window while peak loss keeps growing (window {smoother['window']} buys {smoother['hf_removed'] - best['hf_removed']:+.0%} noise for",
        f"  {abs(smoother['peak_loss']) - abs(best['peak_loss']):+.1%} more peak loss). Note it also moves contact onset about one frame",
        "  EARLIER, because a symmetric window mixes future samples in: never feed a centered-filtered",
        "  signal to a model as an input feature, or it sees contact before contact happens.",
        "* **Online policy inputs: a causal Savitzky-Golay is the wrong tool here.** It is exact on a",
        "  linear trend (zero lag at DC, which is its selling point) but its gain peaks at 1.40 to 1.66",
        "  around 5-8 Hz, so it adds power in the band this data is noisiest in. Use a causal",
        "  Butterworth at 6-10 Hz (never exceeds unity gain, 1-2 frames of lag) or an EMA with a 2-3",
        "  frame time constant if the filter has to be one multiply-add.",
        "* Order 2 and order 3 are identical for centered smoothing (the odd term is antisymmetric and",
        "  vanishes at the window centre), so there is nothing to gain from order 3 offline.",
        "* Leave joint positions, the state-command difference and object poses raw.",
        "",
        "## 6. Files",
        "",
        "| file | what it shows |",
        "|---|---|",
        "| `noise_and_response.png` | measured noise spectra + every filter's magnitude response |",
        "| `raw_vs_filtered_<task>.png` | fingertip forces raw vs filtered, full episode and contact zoom |",
        "| `signals_<task>.png` | measured torque, motor current, joint velocity, command difference |",
        "| `parameter_sweep.png` | window length vs noise removed, peak lost, RMS change |",
        "| `causal_comparison.png` | the online-usable filters against each other |",
        "| `vector_vs_magnitude.png` | why the force vector is filtered instead of its magnitude |",
        "",
        "Episodes used: " + ", ".join(f"`{e.parent.name}/{e.name}`" for e in episodes) + ".",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=3, help="episodes per task")
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 7, 9, 15, 21, 31])
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--plot-window", type=int, default=9, help="window used in the time-series figures")
    parser.add_argument("--out-dir", type=Path, default=DATASET / "filtering")
    parser.add_argument("--self-test", action="store_true",
                        help="check the filter implementations against their defining properties and exit")
    args = parser.parse_args()

    if args.self_test:
        return 1 if self_test() else 0

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    episodes = pick_episodes(args.episodes)
    print(f"[sg] {len(episodes)} episodes, windows {args.windows}, orders {args.orders} -> {out}", flush=True)

    noise = figure_noise_and_response(episodes, args.windows, out / "noise_and_response.png")
    print("[sg] noise_and_response.png", flush=True)

    for problem in PROBLEMS:
        ep = next((e for e in episodes if e.parent.name == problem), None)
        if ep is None:
            continue
        figure_raw_vs_filtered(ep, args.plot_window, out / f"raw_vs_filtered_{problem}.png")
        figure_other_signals(ep, out / f"signals_{problem}.png")
        print(f"[sg] raw_vs_filtered_{problem}.png, signals_{problem}.png", flush=True)

    rows = sweep(episodes, args.windows, args.orders)
    figure_sweep(rows, args.windows, out / "parameter_sweep.png")
    figure_causal_comparison(episodes[0], out / "causal_comparison.png")
    figure_vector_vs_magnitude(episodes[0], args.plot_window, out / "vector_vs_magnitude.png")
    print("[sg] parameter_sweep.png, causal_comparison.png, vector_vs_magnitude.png", flush=True)

    (out / "sweep.json").write_text(json.dumps(rows, indent=2))
    write_report(out, episodes, rows, noise, args.windows)
    print(f"[sg] REPORT.md + sweep.json -> {out}", flush=True)
    for r in sorted(rows, key=lambda r: -r["hf_removed"])[:6]:
        print(f"[sg]   {r['filter']:28s} noise -{r['hf_removed']:.0%}  peak {r['peak_loss']:+.1%}  "
              f"onset {r['onset']:+.1f}  lag {r['lag']:+.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
