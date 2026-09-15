"""Savitzky-Golay filtering for the BC dataset signals (pure numpy/scipy, no Isaac).

A Savitzky-Golay filter fits a degree-``polyorder`` polynomial to a sliding window of
``window`` samples by least squares and replaces the sample by the fitted value. Because the fit
is a linear operation it is a FIR filter: a fixed coefficient vector per output position. That is
why it keeps peak height far better than a moving average of the same length, which is the degree-0
special case.

Two evaluation positions matter here and they are NOT interchangeable:

``centered``  the window is symmetric around the sample (``scipy.signal.savgol_filter``). Zero
              phase, no lag, but it uses ``(window - 1) / 2`` FUTURE samples. Legitimate for data
              a model PREDICTS (targets, labels, anything computed offline), never for a signal the
              policy READS at run time.
``causal``    the window ends at the sample, so only past and present samples are used
              (``savgol_coeffs(window, polyorder, pos=window - 1)``). Deployable online, at the
              cost of a small lag and more high-frequency leak-through for the same window.

Both are provided and the study script (:mod:`sg_filter_study`) measures the difference. Filtering
happens along axis 0 (time) and every other axis is a channel.

ALWAYS FILTER THE FORCE VECTOR, NEVER ITS MAGNITUDE. ``|F|`` is non-negative and jumps at
contact onset; smoothing it makes the filter undershoot below zero on 6 % of the samples (down to
-3 N, measured over the dataset). Smoothing the xyz components and taking the norm afterwards
(:func:`filtered_force_magnitude`) gives 0 % impossible values and peaks within 2 % of the other
route. ``DEFAULT_FILTER_SPEC`` keys the vector array ``contact_force`` for exactly this reason.

Why these signals: in this dataset the fingertip forces, the PhysX measured joint torques and the
body joint wrenches carry 10-15 % of their power above 10 Hz (solver-step contact chatter), the
hand joint velocities and the drive torque / motor current 6-10 %, while joint positions, the
state-command difference and object poses are below 1 % and are left alone. See
``DEFAULT_FILTER_SPEC``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import freqz, savgol_coeffs, savgol_filter

# --------------------------------------------------------------------------------------------
# core filters
# --------------------------------------------------------------------------------------------
def _check(window: int, polyorder: int) -> None:
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder {polyorder} must be < window {window}")


def savgol_centered(x: np.ndarray, window: int, polyorder: int, deriv: int = 0,
                    delta: float = 1.0, mode: str = "nearest") -> np.ndarray:
    """Zero-phase Savitzky-Golay along axis 0. Uses future samples: offline use only.

    ``mode="nearest"`` pads the edges with the edge value instead of extrapolating the polynomial
    (scipy's ``interp`` default), which cannot overshoot at a contact onset sitting in the first
    or last half-window.
    """
    _check(window, polyorder)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < window:
        return x.copy()
    return savgol_filter(x, window, polyorder, deriv=deriv, delta=delta, axis=0, mode=mode)


def savgol_causal(x: np.ndarray, window: int, polyorder: int, deriv: int = 0,
                  delta: float = 1.0) -> np.ndarray:
    """One-sided Savitzky-Golay: output *t* uses samples ``t - window + 1 .. t`` only.

    Online-deployable. The first ``window - 1`` samples cannot fill the window; they are fitted
    with the longest window that fits and a polynomial order reduced to match, so the start of an
    episode is still causal and never extrapolated from nothing.
    """
    _check(window, polyorder)
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    out = np.empty_like(flat)
    coeffs = savgol_coeffs(window, polyorder, deriv=deriv, delta=delta, pos=window - 1, use="dot")
    for t in range(flat.shape[0]):
        if t >= window - 1:
            out[t] = coeffs @ flat[t - window + 1:t + 1]
        else:
            n = t + 1                                   # samples available so far
            if n == 1:
                out[t] = flat[0] if deriv == 0 else 0.0
                continue
            order = min(polyorder, n - 1)
            c = savgol_coeffs(n, order, deriv=deriv, delta=delta, pos=n - 1, use="dot")
            out[t] = c @ flat[:n]
    return out.reshape(x.shape)


def moving_average(x: np.ndarray, window: int, causal: bool = False) -> np.ndarray:
    """Boxcar of the same length, the degree-0 Savitzky-Golay, as the baseline to beat."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    pad = window - 1 if causal else (window - 1) // 2
    pad_after = 0 if causal else window - 1 - pad
    padded = np.concatenate([np.repeat(flat[:1], pad, axis=0), flat,
                             np.repeat(flat[-1:], pad_after, axis=0)], axis=0)
    csum = np.cumsum(padded, axis=0)
    out = np.empty_like(flat)
    out[0] = padded[:window].mean(axis=0)
    out[1:] = (csum[window:] - csum[:-window]) / window
    return out.reshape(x.shape)


def butter_lowpass(x: np.ndarray, cutoff_hz_: float, fs: float = 60.0, order: int = 2,
                   causal: bool = True) -> np.ndarray:
    """Butterworth low-pass along axis 0; ``causal`` uses ``lfilter`` (has lag), otherwise
    ``filtfilt`` (zero phase, offline). The honest causal baseline: unlike a causal
    Savitzky-Golay it never exceeds unity gain, so it cannot amplify the noise band.
    """
    from scipy.signal import butter, filtfilt, lfilter

    x = np.asarray(x, dtype=np.float64)
    b, a = butter(order, cutoff_hz_ / (fs / 2), btype="low")
    if causal:
        zi_shape = (max(len(a), len(b)) - 1,)
        flat = x.reshape(x.shape[0], -1)
        out = np.empty_like(flat)
        for c in range(flat.shape[1]):
            # start the filter settled at the first sample instead of at zero, so an episode that
            # begins mid-contact does not start with a ramp from 0
            from scipy.signal import lfilter_zi

            zi = lfilter_zi(b, a) * flat[0, c] if zi_shape[0] == len(lfilter_zi(b, a)) else None
            out[:, c] = lfilter(b, a, flat[:, c], zi=zi)[0] if zi is not None else lfilter(b, a, flat[:, c])
        return out.reshape(x.shape)
    if x.shape[0] <= 3 * max(len(a), len(b)):
        return x.copy()
    return filtfilt(b, a, x, axis=0)


def ema(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """First-order causal exponential moving average with time constant *tau_frames*.

    One state per channel, one multiply-add per sample: the cheapest thing that can run inside a
    60 Hz control loop, and the usual baseline for an online tactile filter.
    """
    x = np.asarray(x, dtype=np.float64)
    alpha = 1.0 - np.exp(-1.0 / max(tau_frames, 1e-6))
    flat = x.reshape(x.shape[0], -1)
    out = np.empty_like(flat)
    acc = flat[0].copy()
    for t in range(flat.shape[0]):
        acc += alpha * (flat[t] - acc)
        out[t] = acc
    return out.reshape(x.shape)


def filtered_force_magnitude(force_xyz: np.ndarray, window: int, polyorder: int,
                             causal: bool = True) -> np.ndarray:
    """``|F|`` from a filtered force VECTOR ``[T, tips, 3]`` -> ``[T, tips]``.

    The only correct way to get a smooth force magnitude: filtering ``|F|`` itself undershoots
    below zero at every contact onset and release (see the module docstring).
    """
    fn = savgol_causal if causal else savgol_centered
    return np.linalg.norm(fn(np.asarray(force_xyz, dtype=np.float64), window, polyorder), axis=-1)


def savgol_derivative(x: np.ndarray, window: int, polyorder: int, dt: float,
                      causal: bool = False) -> np.ndarray:
    """Smoothed time derivative from the same polynomial fit (units: x per second).

    The point of doing it this way: differentiating a noisy signal amplifies exactly the
    high-frequency content, while the Savitzky-Golay fit differentiates the fitted polynomial.
    """
    fn = savgol_causal if causal else savgol_centered
    return fn(x, window, polyorder, deriv=1, delta=dt)


# --------------------------------------------------------------------------------------------
# what each dataset signal gets
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterSpec:
    """How one recorded array is smoothed. ``window`` is in 60 Hz frames."""

    window: int
    polyorder: int
    causal: bool = True
    note: str = ""

    def lag_frames(self, fs: float = 60.0, at_hz: float = 5.0) -> float:
        """Group delay in frames at *at_hz*. A centered filter is 0 everywhere; a causal
        Savitzky-Golay is 0 at DC (it passes linear trends untouched) and grows with frequency,
        which is why it costs far less lag than a boxcar of the same length."""
        w, _, gd = frequency_response(self.window, self.polyorder, self.causal, fs=fs)
        return float(np.interp(at_hz, w, gd))


# Measured by sg_filter_study.py over 12 episodes of all four tasks (see the REPORT.md it writes).
# Window 7 (117 ms at 60 Hz) is the knee of the sweep on the contact channels: 89 % of the power
# above 10 Hz removed for 4.5 % loss of the 99th-percentile force peak. Going to 9 buys 6 more
# points of noise for 2 more points of peak, and past that noise removal saturates while peak loss
# keeps growing. Order 2, so a peak is fitted by a parabola instead of flattened by a line; order 3
# is mathematically identical for centered smoothing.
#
# causal=False is the DEFAULT because the study's other finding is that a causal Savitzky-Golay
# amplifies 5-8 Hz by up to 1.66x, which is where this data is noisiest. Centered filtering is
# correct for everything computed offline (labels, targets, analysis) and WRONG for a feature the
# policy reads online, both because it needs future samples and because it shifts contact onset
# about one frame earlier. For online use pick butter_lowpass(6-10 Hz) or ema(tau 2-3) instead.
DEFAULT_FILTER_SPEC: dict[str, FilterSpec] = {
    "contact_force": FilterSpec(7, 2, causal=False, note="fingertip net force VECTOR (never |F|)"),
    "joint_torque_measured": FilterSpec(7, 2, causal=False, note="PhysX projected joint force"),
    "joint_wrench_b": FilterSpec(7, 2, causal=False, note="per-body incoming joint wrench"),
    "applied_torque": FilterSpec(5, 2, causal=False, note="PD drive torque, already smoother"),
    "computed_torque": FilterSpec(5, 2, causal=False, note="unclipped drive torque"),
    "motor_current": FilterSpec(5, 2, causal=False, note="applied_torque / kt, same noise"),
    "joint_vel": FilterSpec(5, 2, causal=False, note="hand joints chatter, arm joints are clean"),
}
# The trajectory channels are NOT in DEFAULT_FILTER_SPEC (they are already clean: 0.2 % of their
# power is above 10 Hz). This second spec exists for one purpose: a behaviour-cloning policy trained
# on smoothed data emits a smoothed trajectory, so replaying a smoothed trajectory in the simulator
# is the test of whether smoothing destroyed anything the task needs (replay_filtered_study.py).
TRAJECTORY_FILTER_SPEC: dict[str, FilterSpec] = {
    "joint_target": FilterSpec(7, 2, causal=False, note="the command a BC policy would predict"),
    "joint_pos": FilterSpec(7, 2, causal=False, note="reached state, the other BC premise"),
    "joint_vel": FilterSpec(7, 2, causal=False, note="kept consistent with joint_pos"),
}
# every contact channel an episode carries, including the per-object decomposition the law reads
FORCE_FILTER_SPEC: dict[str, FilterSpec] = {
    **{k: v for k, v in DEFAULT_FILTER_SPEC.items() if k in ("contact_force", "joint_torque_measured",
                                                             "joint_wrench_b")},
    "contact_object_force_steps": FilterSpec(7, 2, causal=False, note="per-object force matrix"),
}

# the online-usable alternative for anything the policy reads at run time (see REPORT.md section 5)
ONLINE_FILTER_HZ = 8.0            # butter_lowpass(x, ONLINE_FILTER_HZ, causal=True)
ONLINE_EMA_TAU_FRAMES = 2.0       # ema(x, ONLINE_EMA_TAU_FRAMES), one multiply-add per sample
# left raw on purpose: joint_pos, joint_target, joint_cmd_err, object_pos/quat/lin_vel, action,
# obs_policy - all below 1 % power above 10 Hz, so filtering them only adds lag.
UNFILTERED = ("joint_pos", "joint_target", "joint_cmd_err", "object_pos", "object_quat",
              "object_lin_vel", "action", "obs_policy", "body_pos", "body_quat")


def filter_arrays(data: dict[str, np.ndarray], spec: dict[str, FilterSpec] | None = None,
                  causal: bool | None = None, suffix: str = "") -> dict[str, np.ndarray]:
    """Filter every key of *data* that has a spec. Returns ``{key + suffix: filtered}``.

    ``causal`` overrides every spec's own setting (used by the study to run both ways).
    Arrays with a substep axis (``*_steps``, shape ``[T, S, ...]``) are filtered along the flattened
    120 Hz time axis with the window doubled, so the smoothing covers the same wall-clock span.
    """
    spec = spec or DEFAULT_FILTER_SPEC
    out: dict[str, np.ndarray] = {}
    for key, s in spec.items():
        for name, sub in ((key, False), (f"{key}_steps", True)):
            if name not in data:
                continue
            x = np.asarray(data[name], dtype=np.float64)
            use_causal = s.causal if causal is None else causal
            if sub:
                steps = x.shape[1]
                window = s.window * steps + (1 - (s.window * steps) % 2)   # keep it odd
                flat = x.reshape(-1, *x.shape[2:])
                fn = savgol_causal if use_causal else savgol_centered
                out[name + suffix] = fn(flat, window, s.polyorder).reshape(x.shape).astype(np.float32)
            else:
                fn = savgol_causal if use_causal else savgol_centered
                out[name + suffix] = fn(x, s.window, s.polyorder).astype(np.float32)
    return out


def scaled_spec(spec: dict[str, FilterSpec], window: int, polyorder: int | None = None,
                causal: bool | None = None) -> dict[str, FilterSpec]:
    """The same spec with one window / polyorder / causality for every signal (sweeps)."""
    return {k: replace(v, window=window,
                       polyorder=polyorder if polyorder is not None else v.polyorder,
                       causal=causal if causal is not None else v.causal)
            for k, v in spec.items()}


# --------------------------------------------------------------------------------------------
# measuring what a filter did
# --------------------------------------------------------------------------------------------
def frequency_response(window: int, polyorder: int, causal: bool, fs: float = 60.0,
                       n: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(freq_hz, |H|, group_delay_frames)`` of the filter as it is actually applied.

    ``savgol_coeffs(..., use="conv")`` is the FIR impulse response of the causal application, so
    its group delay is the real delay. The centered application shifts the output back by
    ``(window - 1) / 2`` samples, which is subtracted here: a symmetric filter then correctly
    reports 0 at every frequency. Note the causal Savitzky-Golay is 0 at DC for any
    ``polyorder >= 1``, because it reproduces a linear trend exactly; the delay grows with
    frequency instead.
    """
    from scipy.signal import group_delay

    pos = window - 1 if causal else (window - 1) // 2
    b = savgol_coeffs(window, polyorder, pos=pos, use="conv")
    w, h = freqz(b, worN=n, fs=fs)
    _, gd = group_delay((b, 1.0), w=n, fs=fs)
    if not causal:
        gd = gd - (window - 1) / 2.0
    return w, np.abs(h), gd


def cutoff_hz(window: int, polyorder: int, causal: bool, fs: float = 60.0, db: float = -3.0) -> float:
    """First frequency where the magnitude response drops below *db* (and stays there)."""
    w, mag, _ = frequency_response(window, polyorder, causal, fs=fs)
    thr = 10 ** (db / 20)
    below = np.flatnonzero(mag < thr)
    return float(w[below[0]]) if below.size else float(fs / 2)


def hf_power_fraction(x: np.ndarray, fs: float = 60.0, cut: float = 10.0,
                      nperseg: int = 256) -> float:
    """Share of spectral power above *cut* Hz, averaged over channels (constant channels skipped)."""
    from scipy.signal import welch

    a = np.asarray(x, dtype=np.float64).reshape(np.shape(x)[0], -1).T      # [C, T]
    if a.shape[1] < 32:
        return float("nan")
    f, P = welch(a - a.mean(axis=1, keepdims=True), fs=fs, nperseg=min(nperseg, a.shape[1]), axis=1)
    tot = P.sum(axis=1)
    keep = tot > 1e-12
    if not keep.any():
        return float("nan")
    return float((P[keep][:, f > cut].sum(axis=1) / tot[keep]).mean())


def band_power_ratio(raw: np.ndarray, filt: np.ndarray, band: tuple[float, float],
                     fs: float = 60.0, nperseg: int = 96) -> float:
    """Filtered / raw power inside *band* (Hz). Above 1 means the filter AMPLIFIED that band.

    This is the number that exposes a causal Savitzky-Golay: its magnitude response overshoots
    unity before rolling off, so it can add power in the 4-10 Hz range while still looking good
    on a "power above 10 Hz" summary.
    """
    from scipy.signal import welch

    def band_power(x):
        a = np.asarray(x, dtype=np.float64).reshape(np.shape(x)[0], -1).T
        if a.shape[1] < 32:
            return np.nan
        f, P = welch(a - a.mean(axis=1, keepdims=True), fs=fs, nperseg=min(nperseg, a.shape[1]), axis=1)
        sel = (f >= band[0]) & (f <= band[1])
        return float(P[:, sel].sum())

    pr = band_power(raw)
    return float("nan") if not np.isfinite(pr) or pr < 1e-12 else band_power(filt) / pr


def negative_fraction(x: np.ndarray, tol: float = 1e-6) -> float:
    """Share of samples below zero. Meaningful only for quantities that cannot be negative
    (a force magnitude); the check that catches magnitude smoothing done the wrong way."""
    a = np.asarray(x, dtype=np.float64)
    return float((a < -tol).mean())


def measured_lag_frames(raw: np.ndarray, filt: np.ndarray, max_lag: int = 20) -> float:
    """Lag that best aligns *filt* with *raw*, by cross-correlation of the strongest channel.

    Reported in frames (60 Hz). A centered filter should land on 0.
    """
    a = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)
    b = np.asarray(filt, dtype=np.float64).reshape(len(filt), -1)
    ch = int(np.argmax(a.std(axis=0)))
    x, y = a[:, ch] - a[:, ch].mean(), b[:, ch] - b[:, ch].mean()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    lags = np.arange(-max_lag, max_lag + 1)
    scores = [np.corrcoef(x[max_lag:-max_lag], np.roll(y, -k)[max_lag:-max_lag])[0, 1] for k in lags]
    return float(lags[int(np.nanargmax(scores))])


def peak_attenuation(raw: np.ndarray, filt: np.ndarray, quantile: float = 0.99) -> float:
    """Relative loss of the signal's large values: ``1 - q(filt) / q(raw)`` at *quantile* of |x|.

    Uses a high quantile rather than the single maximum so one solver spike does not decide it.
    """
    a = np.abs(np.asarray(raw, dtype=np.float64)).ravel()
    b = np.abs(np.asarray(filt, dtype=np.float64)).ravel()
    qa = np.quantile(a, quantile)
    return float("nan") if qa < 1e-9 else float(1.0 - np.quantile(b, quantile) / qa)


def onset_shift_frames(raw: np.ndarray, filt: np.ndarray, threshold: float = 1.0) -> float:
    """How much later the filtered signal first crosses *threshold* (frames; negative = earlier).

    For contact forces this is the practically important number: a filter that delays the detection
    of first contact delays every downstream reaction.
    """
    a = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)
    b = np.asarray(filt, dtype=np.float64).reshape(len(filt), -1)
    shifts = []
    for c in range(a.shape[1]):
        ia = np.flatnonzero(np.abs(a[:, c]) > threshold)
        ib = np.flatnonzero(np.abs(b[:, c]) > threshold)
        if ia.size and ib.size:
            shifts.append(ib[0] - ia[0])
    return float(np.mean(shifts)) if shifts else float("nan")
