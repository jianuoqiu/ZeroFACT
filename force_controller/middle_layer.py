"""The middle layer: predicted (state, force target) chunks + live tactile readings -> PD targets.

Runs at physics rate (120 Hz) between the 10 Hz policy predictions:

    policy chunk (10 Hz) --> ChunkTracker -- interpolated q_ref(t), f_ref(t), p_ref(t) --+
                                                                                         v
    fingertip pads (120 Hz) ----------------------------> ForceFeedbackLaw --> dq --> action

The action is a joint-position target for the existing implicit PD actuators (arm K=400/D=40,
hand K=350/D=12), i.e. the same interface the replay drives. The force law only ever *adds an
offset* to the predicted state - with the law disabled the middle layer degrades to pure
trajectory playback, which is the framework's validation mode.

Why an offset is needed at all: the BC policy predicts where the robot should BE (``joint_pos``)
and what it should FEEL (fingertip force target). Under contact a PD controller must command
*past* the measured position to generate force (in the recordings the commanded target leads the
measured hand joints by up to ~0.04 rad during the 95 N grasp). Feeding predicted positions
straight to the PD therefore under-squeezes; closing the loop on the predicted-vs-measured force
restores the missing lead. That gap is exactly what the force law has to produce.

Sensing contract: the only exteroceptive input is the fingertip tactile pad - its NET contact
force and its contact centroid (:class:`Measured`). No per-object force decomposition, no object
poses, no knowledge of what is being touched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .config import ARM_JOINT_NAMES, ControllerConfig, ForceLawConfig, default_joint_stiffness
from .policy import PolicyOutput


# ----------------------------------------------------------------------------------------------
# Data carried across the interfaces
# ----------------------------------------------------------------------------------------------
@dataclass
class Reference:
    """The interpolated prediction at one control instant."""

    t: float                       # time in frame units
    q: np.ndarray                  # [J] predicted joint state
    f_mag: np.ndarray              # [4] predicted per-fingertip force magnitude (N) == |f_vec|
    f_vec: np.ndarray | None = None  # [4, 3] predicted force vectors (world), when available
    contact_point: np.ndarray | None = None  # [4, 3] predicted contact points (world), NaN-padded


@dataclass
class Measured:
    """What the fingertip tactile pads (plus proprioception) report at one control instant.

    Everything here is available on hardware: joint states from the encoders, fingertip poses and
    Jacobians from forward kinematics, and per-pad net force + contact centroid from the tactile
    sensor. There is deliberately no per-object force breakdown.
    """

    q: np.ndarray                  # [J] measured joint positions
    qd: np.ndarray                 # [J] measured joint velocities
    f_net: np.ndarray              # [4, 3] NET contact force on each fingertip pad, world frame
    tip_pos: np.ndarray | None = None   # [4, 3] fingertip body positions
    tip_contact_pos: np.ndarray | None = None   # [4, 3] pad contact centroid, NaN off-contact
    tip_jacobians: np.ndarray | None = None     # [4, 3, J] positional Jacobian of each fingertip's
    #                                             contact point (world frame; sim runner provides)


@dataclass
class FilteredForce:
    """EMA-filtered tactile reading handed to the law."""

    vec: np.ndarray                # [4, 3] filtered net force
    mag: np.ndarray                # [4] == |vec|
    point: np.ndarray | None = None  # [4, 3] filtered contact centroid, NaN off-contact


# ----------------------------------------------------------------------------------------------
# Chunk tracking / interpolation
# ----------------------------------------------------------------------------------------------
class ChunkTracker:
    """Holds the active prediction chunk and evaluates a continuous reference at any time.

    Knot ``j`` of a chunk starting at frame ``k`` lives at time ``k + j + 1`` (the end of frame
    ``k+j``). The segment before the first knot is anchored at the reference that was being
    tracked when the chunk arrived, so chunk switches never jump.

    interp="hold":   reference(t) = the next knot at or after t (zero-order hold-right). This is
                     the replay's constant-per-frame command shape.
    interp="linear": piecewise linear through (anchor, knot_0, ..., knot_{H-1}).
    Beyond the last knot the reference holds (relevant only at the episode end).

    The magnitude channel is DERIVED from the interpolated vector rather than interpolated
    separately: lerping norms and lerping vectors disagree between knots, which used to give the
    law's engagement gate and the vector metrics two different notions of "engaged".
    """

    def __init__(self, interp: str = "linear"):
        if interp not in ("hold", "linear"):
            raise ValueError(f"unknown interp mode {interp!r}")
        self.interp = interp
        self._times: np.ndarray | None = None      # [1 + H] anchor + knot times
        self._q: np.ndarray | None = None          # [1 + H, J]
        self._f: np.ndarray | None = None          # [1 + H, 4]
        self._f_vec: np.ndarray | None = None      # [1 + H, 4, 3] or None
        self._points: np.ndarray | None = None     # [1 + H, 4, 3] or None (NaN off-contact)

    @property
    def active(self) -> bool:
        return self._times is not None

    def set_chunk(self, chunk: PolicyOutput, anchor: Reference) -> None:
        knot_times = chunk.start_frame + 1.0 + np.arange(chunk.horizon, dtype=np.float64)
        # the anchor value is the reference being tracked when the chunk arrived; it represents
        # the state at the end of frame start-1, i.e. time == start_frame (latency-shifted command
        # times at the start of the chunk must still resolve to it)
        anchor_t = min(float(chunk.start_frame), float(knot_times[0]) - 1e-9)
        self._times = np.concatenate([[anchor_t], knot_times])
        self._q = np.concatenate([anchor.q[None], chunk.joint_pos], axis=0)
        self._f = np.concatenate([anchor.f_mag[None], chunk.fingertip_force], axis=0)
        if chunk.fingertip_force_vec is not None:
            anchor_vec = anchor.f_vec if anchor.f_vec is not None else np.zeros_like(chunk.fingertip_force_vec[0])
            self._f_vec = np.concatenate([anchor_vec[None], chunk.fingertip_force_vec], axis=0)
        else:
            self._f_vec = None
        if chunk.contact_point is not None:
            anchor_pt = (
                anchor.contact_point if anchor.contact_point is not None
                else np.full_like(chunk.contact_point[0], np.nan)
            )
            self._points = np.concatenate([anchor_pt[None], chunk.contact_point], axis=0)
        else:
            self._points = None

    @staticmethod
    def _blend_points(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
        """Lerp contact points, NaN-aware: off-contact knots defer to the touching endpoint."""
        out = (1.0 - w) * a + w * b
        a_ok = np.isfinite(a).all(axis=-1)
        b_ok = np.isfinite(b).all(axis=-1)
        out[~a_ok & b_ok] = b[~a_ok & b_ok]
        out[a_ok & ~b_ok] = a[a_ok & ~b_ok]
        return out

    def eval(self, t: float) -> Reference:
        if not self.active:
            raise RuntimeError("ChunkTracker.eval called before the first chunk was set")
        times = self._times
        if t <= times[0]:
            i0 = i1 = 0
            w = 0.0
        elif t >= times[-1]:
            i0 = i1 = len(times) - 1
            w = 0.0
        else:
            i1 = int(np.searchsorted(times, t, side="left"))   # first knot at or after t
            i0 = i1 - 1
            if self.interp == "hold":
                i0 = i1
                w = 0.0
            else:
                w = (t - times[i0]) / (times[i1] - times[i0])
        q = (1.0 - w) * self._q[i0] + w * self._q[i1]
        f_vec = None
        if self._f_vec is not None:
            f_vec = (1.0 - w) * self._f_vec[i0] + w * self._f_vec[i1]
            f = np.linalg.norm(f_vec, axis=-1)              # derived, never separately lerped
        else:
            f = (1.0 - w) * self._f[i0] + w * self._f[i1]
        point = None
        if self._points is not None:
            point = self._blend_points(self._points[i0], self._points[i1], w)
        return Reference(t=float(t), q=q, f_mag=f, f_vec=f_vec, contact_point=point)


# ----------------------------------------------------------------------------------------------
# Force feedback laws
# ----------------------------------------------------------------------------------------------
class ForceFeedbackLaw(ABC):
    """Maps the force-target tracking error to a joint-position offset added to the reference."""

    def __init__(self, cfg: ForceLawConfig, joint_names: list[str], fingertip_bodies: list[str],
                 joint_stiffness: np.ndarray | None = None):
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.fingertip_bodies = list(fingertip_bodies)
        self.joint_stiffness = (
            np.asarray(joint_stiffness, dtype=np.float64)
            if joint_stiffness is not None
            else np.asarray(default_joint_stiffness(self.joint_names))
        )
        # which joints the offset may recruit. Arm joints are opt-in (see ForceLawConfig): the
        # per-tip superposition ignores that they are shared between fingertips.
        self.joint_mask = np.ones(len(self.joint_names))
        if not cfg.allow_arm_offset:
            for i, name in enumerate(self.joint_names):
                if name in ARM_JOINT_NAMES:
                    self.joint_mask[i] = 0.0

    def reset(self) -> None:  # noqa: B027
        pass

    @abstractmethod
    def compute(self, ref: Reference, filt: FilteredForce, meas: Measured, dt: float
                ) -> tuple[np.ndarray, dict]:
        """Return ``(dq [J], info)`` for one control instant. *dt* is the physics step in s."""


class NullForceLaw(ForceFeedbackLaw):
    """No feedback: action = predicted state. The playback / validation baseline."""

    def compute(self, ref, filt, meas, dt):
        return np.zeros(len(self.joint_names)), {}


class TaskSpaceForceLaw(ForceFeedbackLaw):
    """Direction-free hybrid force/position tracking of the full contact target.

    The target is a contact description: contact point ``p_ref``, world force vector ``f_ref`` ON
    the fingertip FROM the environment, magnitude ``|f_ref|``. Per fingertip the offset is the sum
    of two channels living in orthogonal subspaces of the contact frame.

    **Force channel.** The joint offset that makes the implicit PD exert ``f`` follows from
    statics::

        tau = J_c^T F_finger_on_env = J_c^T (-f)      J_c: positional Jacobian at p_c
        tau = K dq                                    K: PD stiffness diagonal
        =>  dq_f = -K^-1 J_c^T f

    applied to ``f = kff*f_ref + task_kp*e + I + task_kd*de/dt`` with ``e = f_ref - f_meas`` and
    ``I`` accumulating ``task_ki*e*dt`` (clamped at ``task_i_max_n``).

    **Contact-point channel.** ``e_p = p_ref - p_meas`` is the contact-point tracking error. Its
    component along ``f_ref`` is depth-of-press, which the force channel already regulates, so it
    is projected out; the remaining tangential part is *where on the surface* the finger sits,
    which the force channel cannot observe. It is mapped back through the damped pseudo-inverse of
    the same contact Jacobian::

        e_p_perp = (I - n n^T) e_p,   n = f_ref / |f_ref|
        dq_p     = point_kp * J_c^T (J_c J_c^T + lambda^2 I)^-1 e_p_perp

    No direction is assumed anywhere: push, pull and squeeze all emerge from the predicted vector,
    and the force/position subspace split is *derived* from ``f_ref`` rather than fixed in
    advance. Engagement gates both channels: while ``|f_ref|`` is ~0 the integral leaks out
    (release) and the contact-point term is off, and the feedforward vanishes with ``f_ref`` by
    construction.

    Needs ``Measured.tip_jacobians`` (the sim runner computes them at the measured contact point,
    falling back to the predicted point, then the fingertip origin). Without them the law degrades
    to zero offset and warns once - the offline checker has no kinematics.
    """

    def __init__(self, cfg: ForceLawConfig, joint_names: list[str], fingertip_bodies: list[str],
                 joint_stiffness: np.ndarray | None = None):
        super().__init__(cfg, joint_names, fingertip_bodies, joint_stiffness)
        n = len(fingertip_bodies)
        self.integral = np.zeros((n, 3))           # N, per-tip integral of the error vector
        self._prev_err = np.zeros((n, 3))
        self.alpha = np.ones(n)                    # plant estimate: achieved/commanded force ratio
        self._prev_cmd_along = None                # [n] last commanded force along the prediction
        self._gain = np.ones(n)                    # slew-limited feedforward gain actually applied
        self._warned_no_jac = False

    def reset(self) -> None:
        self.integral[:] = 0.0
        self._prev_err[:] = 0.0
        self.alpha[:] = 1.0
        self._prev_cmd_along = None
        self._gain[:] = 1.0

    def _point_offset(self, ref: Reference, filt: FilteredForce, meas: Measured,
                      engaged: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Contact-point channel: ``(dq [J], per-tip tangential error magnitude [n])``."""
        cfg = self.cfg
        n = len(self.fingertip_bodies)
        dq = np.zeros(len(self.joint_names))
        err_mag = np.full(n, np.nan)
        if cfg.point_kp <= 0.0 or ref.contact_point is None or filt.point is None:
            return dq, err_mag
        # floored: a rank-deficient contact Jacobian (masked arm columns, singular pose)
        # makes J J^T singular, and only the damping term keeps the solve well posed
        lam2 = max(float(cfg.point_damping), 1e-6) ** 2
        eye3 = np.eye(3)
        for tip in range(n):
            p_ref, p_meas = ref.contact_point[tip], filt.point[tip]
            if not (np.isfinite(p_ref).all() and np.isfinite(p_meas).all()):
                continue
            e_p = p_ref - p_meas
            # split off the depth-of-press component along the commanded force direction; the
            # basis comes from the prediction, so no contact direction is assumed a priori
            f_norm = float(np.linalg.norm(ref.f_vec[tip])) if ref.f_vec is not None else 0.0
            if f_norm > 1e-9:
                n_hat = ref.f_vec[tip] / f_norm
                e_p = e_p - float(e_p @ n_hat) * n_hat
            mag = float(np.linalg.norm(e_p))
            err_mag[tip] = mag
            if not engaged[tip] or mag < 1e-9:
                continue
            e_p = e_p * min(1.0, cfg.point_clip_m / mag)
            # solve within the joints the offset is actually allowed to use, so the damped
            # least-squares redistributes onto them instead of producing motion that gets clipped
            jac = meas.tip_jacobians[tip] * self.joint_mask[None, :]
            dq_tip = cfg.point_kp * (jac.T @ np.linalg.solve(jac @ jac.T + lam2 * eye3, e_p))
            dq += np.clip(dq_tip, -cfg.point_offset_clip_rad, cfg.point_offset_clip_rad)
        return dq, err_mag

    def compute(self, ref, filt, meas, dt):
        cfg = self.cfg
        n = len(self.fingertip_bodies)
        if meas.tip_jacobians is None:
            if not self._warned_no_jac:
                print("[middle_layer][WARN] task_space law needs Measured.tip_jacobians "
                      "(sim runner provides them); returning zero offsets")
                self._warned_no_jac = True
            return np.zeros(len(self.joint_names)), {}
        if ref.f_vec is None:
            raise RuntimeError("task_space law needs the policy's predicted force vectors")

        err = ref.f_vec - filt.vec                                   # [n, 3]
        norm = np.linalg.norm(err, axis=-1, keepdims=True)           # [n, 1]
        err = np.where(norm < cfg.deadband_n, 0.0, err)
        scale = np.minimum(1.0, cfg.error_clip_n / np.maximum(norm, 1e-9))
        err = err * scale

        engaged = ref.f_mag >= cfg.engage_threshold_n                # [n]
        decay = np.exp(-dt / max(cfg.release_tau_s, 1e-6))
        self.integral = np.where(
            engaged[:, None], self.integral + cfg.task_ki * err * dt, self.integral * decay
        )
        i_norm = np.linalg.norm(self.integral, axis=-1, keepdims=True)
        self.integral *= np.minimum(1.0, cfg.task_i_max_n / np.maximum(i_norm, 1e-9))

        de = (err - self._prev_err) / max(dt, 1e-9)
        self._prev_err = err

        # ---- adaptive feedforward gain (the integral-saturation cure, see config.py) ----
        # r = (measured force along the predicted direction) / (what was commanded along it last
        # step). alpha = EMA of r = how much the plant actually delivers per newton commanded;
        # the feedforward is multiplied by 1/alpha so IT supplies the missing scale in the safe
        # direction, instead of the integral supplying it in a geometry-locked one.
        ref_mag_a = np.linalg.norm(ref.f_vec, axis=-1)               # [n]
        n_hat = ref.f_vec / np.maximum(ref_mag_a, 1e-9)[:, None]
        gain = np.ones(n)
        if cfg.adapt_kff:
            achieved = np.maximum(np.sum(filt.vec * n_hat, axis=-1), 0.0)
            if self._prev_cmd_along is not None:
                # adapt from FIRST touch (0.5 N): waiting for firm contact lets the integral
                # wind to its clamp before the gain rises, and the two then stack into an
                # over-press (measured 2026-08-31: 4/4 overflow crashes with a 2 N gate,
                # 3/4 clean with 0.5 N). Early, light-contact estimates are noisy but the EMA
                # and the slew limit absorb that.
                upd = engaged & (achieved > 0.5) & (self._prev_cmd_along > 2.0)
                r = np.clip(achieved / np.maximum(self._prev_cmd_along, 1e-9), 0.02, 3.0)
                beta = 1.0 - np.exp(-dt / max(cfg.adapt_tau_s, 1e-3))
                self.alpha = np.where(upd, (1.0 - beta) * self.alpha + beta * r, self.alpha)
            target = np.clip(1.0 / np.maximum(self.alpha, 1e-3), 0.5, cfg.ff_gain_max)
            # rise is slew-limited (over-press safety at first touch); fall is immediate
            self._gain = np.minimum(target, self._gain + cfg.ff_gain_slew * dt)
            self._gain = np.minimum(self._gain, np.maximum(target, 1.0))
            gain = self._gain.copy()

        f_cmd = (
            gain[:, None] * cfg.kff * ref.f_vec
            + (cfg.task_kp * err + self.integral + cfg.task_kd * de) * engaged[:, None]
        )                                                            # [n, 3] on-fingertip convention

        # friction-cone style cap: keep f_cmd within cone_half_angle_deg of the predicted force
        # direction (the prediction was measured on the real surface, so it lies inside the true
        # cone). Split f_cmd into along-prediction + sideways, forbid a net pull (along < 0), and
        # shrink the sideways part so the angle stays inside the cap. Magnitude correction (the
        # integral's real job) passes through untouched; only excess rotation is removed.
        cone_scale = np.ones(n)
        if cfg.cone_half_angle_deg > 0.0:
            tan_max = np.tan(np.radians(cfg.cone_half_angle_deg))
            ref_mag = np.linalg.norm(ref.f_vec, axis=-1)             # [n]
            for tip in range(n):
                if not engaged[tip] or ref_mag[tip] < 1e-9:
                    continue
                n_hat = ref.f_vec[tip] / ref_mag[tip]
                par = float(f_cmd[tip] @ n_hat)
                perp = f_cmd[tip] - par * n_hat
                par = max(par, 0.0)                                  # never command a pull
                perp_max = tan_max * par
                perp_norm = float(np.linalg.norm(perp))
                if perp_norm > perp_max:
                    perp *= perp_max / max(perp_norm, 1e-12)
                    cone_scale[tip] = perp_max / max(perp_norm, 1e-12)
                f_cmd[tip] = par * n_hat + perp

        self._prev_cmd_along = np.maximum(np.sum(f_cmd * n_hat, axis=-1), 0.0)

        dq = np.zeros(len(self.joint_names))
        inv_k = 1.0 / self.joint_stiffness
        for tip in range(n):
            # dq += -K^-1 J^T f  (the finger pushes on the environment with -f)
            dq -= self.joint_mask * inv_k * (meas.tip_jacobians[tip].T @ f_cmd[tip])

        dq_point, point_err = self._point_offset(ref, filt, meas, engaged)
        return dq + dq_point, {
            "force_error": np.linalg.norm(err, axis=-1),
            "u": np.linalg.norm(self.integral, axis=-1),             # N (integral magnitude)
            "engaged": engaged,
            "f_cmd": f_cmd,
            "cone_scale": cone_scale,                                # 1.0 = cap not active
            "ff_gain": gain,                                         # adaptive feedforward gain
            "point_error": point_err,                                # m, tangential, NaN = no pair
            "dq_point": dq_point,
        }


def make_force_law(cfg: ForceLawConfig, joint_names: list[str], fingertip_bodies: list[str],
                   joint_stiffness: np.ndarray | None = None) -> ForceFeedbackLaw:
    laws = {"null": NullForceLaw, "task_space": TaskSpaceForceLaw}
    if cfg.law not in laws:
        raise ValueError(f"unknown force law {cfg.law!r}; available: {sorted(laws)}")
    return laws[cfg.law](cfg, joint_names, fingertip_bodies, joint_stiffness)


# ----------------------------------------------------------------------------------------------
# The middle layer itself
# ----------------------------------------------------------------------------------------------
@dataclass
class ControlStep:
    """Everything the middle layer decided at one physics step (for recording / debugging)."""

    action: np.ndarray             # [J] joint-position target sent to the PD
    q_ref: np.ndarray              # [J] interpolated predicted state
    dq: np.ndarray                 # [J] force-feedback offset (clipped)
    f_ref: np.ndarray              # [4] predicted force magnitude
    f_meas: np.ndarray             # [4] filtered measured force magnitude
    f_ref_vec: np.ndarray          # [4, 3] predicted force vector (world)
    f_meas_vec: np.ndarray         # [4, 3] filtered measured force vector (world)
    p_ref: np.ndarray              # [4, 3] predicted contact point (world), NaN off-contact
    p_meas: np.ndarray             # [4, 3] filtered measured contact centroid, NaN off-contact
    law_info: dict = field(default_factory=dict)


class HybridForceMiddleLayer:
    """Chunk tracking + tactile filtering + hybrid force/contact-point feedback -> PD joint target."""

    def __init__(self, cfg: ControllerConfig, joint_names: list[str], fingertip_bodies: list[str],
                 joint_stiffness: np.ndarray | None = None):
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.fingertip_bodies = list(fingertip_bodies)
        self.tracker = ChunkTracker(cfg.reference.interp)
        self.law = make_force_law(cfg.force_law, self.joint_names, self.fingertip_bodies,
                                  joint_stiffness)
        # per-joint clamp on the force offset; arm joints are excluded when the law is configured
        # hand-only (allow_arm_offset=False)
        self._offset_clip = np.full(len(self.joint_names), cfg.force_law.offset_clip_rad)
        if not cfg.force_law.allow_arm_offset:
            for i, name in enumerate(self.joint_names):
                if name in ARM_JOINT_NAMES:
                    self._offset_clip[i] = 0.0
        self._f_filt: np.ndarray | None = None      # [4, 3] EMA of the measured force vector
        self._p_filt: np.ndarray | None = None      # [4, 3] EMA of the measured contact centroid
        self._last_ref: Reference | None = None

    def reset(self, q0: np.ndarray) -> None:
        """Start (or restart) tracking from initial joint state *q0* at time 0."""
        self.law.reset()
        self._f_filt = None
        self._p_filt = None
        self._last_ref = Reference(
            t=0.0, q=np.asarray(q0, dtype=np.float64).copy(),
            f_mag=np.zeros(len(self.fingertip_bodies)),
            f_vec=np.zeros((len(self.fingertip_bodies), 3)),
            contact_point=np.full((len(self.fingertip_bodies), 3), np.nan),
        )
        self.tracker = ChunkTracker(self.cfg.reference.interp)

    def on_new_chunk(self, chunk: PolicyOutput) -> None:
        """Install a fresh policy prediction; the reference stays continuous across the switch."""
        if self._last_ref is None:
            raise RuntimeError("call reset(q0) before the first chunk")
        self.tracker.set_chunk(chunk, anchor=self._last_ref)

    def command_time(self, frame: int, step: int, steps_per_frame: int) -> float:
        """The reference time (frame units) for the command applied at (frame, step).

        The command held during a step targets the state at the END of that step, shifted back by
        the configured latency: ``t = frame + (step + 1 - latency) / steps_per_frame``. With
        interp="hold" and latency 2 this reproduces the replay's stale-target behaviour exactly.
        """
        return frame + (step + 1 - self.cfg.reference.latency_steps) / steps_per_frame

    def peek_reference(self, t: float) -> Reference:
        """Evaluate the reference at *t* WITHOUT advancing the chunk anchor.

        The sim runner needs the predicted contact point before it can build the contact-point
        Jacobians that :meth:`compute_action` then consumes, so it peeks first.
        """
        return self.tracker.eval(t)

    def evaluate_reference(self, t: float) -> Reference:
        """Evaluate (and remember, for chunk anchoring) the reference at time *t*."""
        ref = self.tracker.eval(t)
        self._last_ref = ref
        return ref

    def _filter(self, meas: Measured) -> FilteredForce:
        """EMA the tactile reading: force vector per axis, contact centroid NaN-aware."""
        alpha = float(np.clip(self.cfg.force_law.meas_ema_alpha, 0.0, 1.0))
        if self._f_filt is None or alpha >= 1.0:
            self._f_filt = meas.f_net.copy()
        else:
            self._f_filt = alpha * meas.f_net + (1.0 - alpha) * self._f_filt

        if meas.tip_contact_pos is None:
            self._p_filt = None
        else:
            alpha_p = float(np.clip(self.cfg.force_law.point_ema_alpha, 0.0, 1.0))
            raw = np.asarray(meas.tip_contact_pos, dtype=np.float64)
            if self._p_filt is None or alpha_p >= 1.0:
                self._p_filt = raw.copy()
            else:
                # blend only where the pad was and still is touching; a pad that just made or
                # lost contact restarts from the raw reading (or NaN) instead of dragging a
                # stale centroid across the gap
                both = np.isfinite(raw).all(axis=-1) & np.isfinite(self._p_filt).all(axis=-1)
                blended = alpha_p * raw + (1.0 - alpha_p) * self._p_filt
                self._p_filt = np.where(both[:, None], blended, raw)
        return FilteredForce(
            vec=self._f_filt,
            mag=np.linalg.norm(self._f_filt, axis=-1),
            point=self._p_filt,
        )

    def compute_action(self, t: float, meas: Measured, dt: float) -> ControlStep:
        ref = self.evaluate_reference(t)
        filt = self._filter(meas)

        dq, law_info = self.law.compute(ref, filt, meas, dt)
        dq = np.clip(dq, -self._offset_clip, self._offset_clip)
        n = len(self.fingertip_bodies)
        return ControlStep(
            action=ref.q + dq,
            q_ref=ref.q,
            dq=dq,
            f_ref=ref.f_mag.copy(),
            f_meas=filt.mag.copy(),
            f_ref_vec=(ref.f_vec.copy() if ref.f_vec is not None else np.zeros((n, 3))),
            f_meas_vec=filt.vec.copy(),
            p_ref=(ref.contact_point.copy() if ref.contact_point is not None
                   else np.full((n, 3), np.nan)),
            p_meas=(filt.point.copy() if filt.point is not None else np.full((n, 3), np.nan)),
            law_info=law_info,
        )
