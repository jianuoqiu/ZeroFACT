"""Configuration dataclasses for the hybrid force controller pipeline.

Pure-python / numpy-free so every module (offline or in-sim) can import it. All timing is expressed
in *trajectory frames* (0.1 s each, ``steps_per_frame`` physics steps of 1/120 s), matching the
replay convention: the state recorded at index ``k`` is the state at the END of frame ``k``.

Sensing contract (2026-08-29): the controller may read **fingertip tactile sensors only** — the
net contact force on each fingertip pad and the pad's contact centroid — plus proprioception
(joint states) and the robot model (FK / Jacobians / PD gains). The simulator's per-object force
breakdown is privileged information no tactile sensor can produce; it stays available for
*analysis* but is unreachable from the control path (there is deliberately no ``force_source``
knob).
"""

from __future__ import annotations

from dataclasses import dataclass, field

ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]

# implicit-PD joint stiffness, mirroring v2s2r_isaaclab.replay (ARM_STIFFNESS / HAND_STIFFNESS).
# The task-space law maps desired contact force to a command offset through dq = K^-1 J^T f, so it
# needs K; the sim runner overrides these with the live articulation's actual values.
ARM_PD_STIFFNESS = 400.0
HAND_PD_STIFFNESS = 350.0


def default_joint_stiffness(joint_names: list[str]) -> list[float]:
    """Per-joint implicit-PD stiffness by name (arm joints 400, LEAP joints 350)."""
    return [ARM_PD_STIFFNESS if n in ARM_JOINT_NAMES else HAND_PD_STIFFNESS for n in joint_names]


@dataclass
class PolicyConfig:
    """How the recorded episode is served as if it came from a BC policy."""

    horizon: int = 16          # predicted frames per query (H)
    chunk: int = 8             # frames executed before the next query (C <= H)
    # what plays the role of the policy's "predicted robot state":
    #   "joint_pos"    - the recorded *measured* states (true BC semantics: the policy predicts
    #                    where the robot should BE; the middle layer must recover the command)
    #   "joint_target" - the recorded *commands* (upper bound / exact-replay validation mode)
    state_source: str = "joint_pos"
    # NOTE: there is no force_source. The force target is always the NET fingertip force, the only
    # thing a tactile pad can report, so prediction and measurement are the same quantity.


@dataclass
class ReferenceConfig:
    """How the 10 Hz chunk knots become a continuous reference at physics rate."""

    # "linear": piecewise-linear between knots (smooth upsampling, the deployment default)
    # "hold"  : zero-order hold-right (reference == next knot), which reproduces the replay's
    #           constant-per-frame commands exactly
    interp: str = "linear"
    # command latency in physics steps: the reference is evaluated at t - latency*dt. The replay
    # holds the previous frame's command for the first 2 steps of each frame (Isaac Gym's stale
    # target); latency_steps=2 + interp="hold" reproduces that exactly.
    latency_steps: int = 0


@dataclass
class ForceLawConfig:
    """Force-feedback law turning the force target tracking error into a joint offset.

    ``task_space`` (the only law): direction-free, and hybrid in the force/position sense. The
    policy's target is a full contact description — contact point ``p_ref``, world-frame force
    vector ``f_ref`` (ON the fingertip FROM the environment), magnitude ``|f_ref|`` — and the
    offset is the sum of two channels that live in orthogonal subspaces:

    * **force channel** — ``tau = J_c^T f`` at the contact point and ``tau = K dq`` for the
      implicit PD give ``dq = -K^-1 J_c^T f``, applied to a feedforward on ``f_ref`` plus a PID on
      the vector force error. Needs contact-point Jacobians (the sim runner provides them).
    * **contact-point channel** — a P term on the contact-point error ``p_ref - p_meas``, mapped
      through the damped pseudo-inverse of the same Jacobian, with the component ALONG ``f_ref``
      projected out. That component is depth-of-press, which the force channel already regulates;
      correcting it positionally would fight it. The remaining tangential component is *where on
      the surface* the finger sits, which the force channel cannot see.

    No contact direction is assumed anywhere: push, pull and squeeze all emerge from the predicted
    vector, and the force/position subspace split is derived from ``f_ref`` rather than fixed.
    """

    law: str = "task_space"                # "null" | "task_space"

    # ---- force channel (units: forces N; gains are relative to the model-based mapping) ----
    # Gain sizing, measured on the dev episodes (2026-08-28): the statics map K^-1 J^T
    # underestimates the needed offset because contact compliance adds displacement on top of
    # torque balance; the effective loop gain is only ~0.16 N of achieved force per N of integral,
    # so the integral needs ~100+ N of headroom (task_i_max_n) and a rate (task_ki) that reaches
    # it within a second. The real safety bound is offset_clip_rad, not the integral clamp.
    kff: float = 1.0                       # feedforward on the predicted force vector
    task_kp: float = 0.3                   # -           proportional on the force error vector
    task_ki: float = 10.0                  # 1/s         integral rate on the force error vector
    task_kd: float = 0.0                   # s           derivative (default off: contact is impulsive)
    task_i_max_n: float = 300.0            # N           per-tip integral clamp (anti-windup)
    # Cap on the angle between f_cmd and the predicted force direction (deg; <=0 disables).
    # The predicted force was measured on the real surface, so its direction lies inside the
    # contact's friction cone; letting the PID/integral rotate the command far away from it asks
    # for a force friction cannot return and grinds the contact sideways. Evidence (2026-08-31,
    # run_2026-05-16_01-27-32, mu=0.35 -> cone half-angle 19.3 deg): the thumb's measured force
    # ran 29-35 deg off the prediction through the hold and the jar crept out of the grasp in 1
    # of 3 identical runs. The cap bounds the command's deviation, not the direction itself -
    # no direction is assumed beyond what the policy already predicted.
    cone_half_angle_deg: float = 20.0
    # Adaptive feedforward (2026-08-31, the integral-saturation cure): the statics map
    # under-delivers (~0.16 N measured per N commanded - contact compliance, object motion), so a
    # fixed kff leaves the integral to supply ~6x the target force, saturating at task_i_max_n
    # with a geometry-locked direction. Instead, estimate the plant's achieved/commanded ratio
    # per tip online (EMA over adapt_tau_s, along the predicted direction, only while engaged and
    # touching) and multiply the feedforward by its inverse (clamped to [0.5, ff_gain_max]).
    # The PID gains are NOT scaled - loop gain stays as tuned; only the feedforward grows.
    adapt_kff: bool = True
    adapt_tau_s: float = 0.3               # s    EMA time constant of the achieved/commanded estimate
    ff_gain_max: float = 6.0               # -    clamp on the adaptive feedforward gain
    # the gain may RISE at most this fast (fall is free). Without it the gain can jump to the
    # clamp within ~tau at first touch - before the estimate is trustworthy - and the transient
    # over-press can blow past the contact sensor's 32-point buffer (the known CUDA-assert crash;
    # it killed 1 of 3 validation runs on 2026-08-31 before this limit existed).
    ff_gain_slew: float = 3.0              # 1/s  max upward gain rate
    # Arm participation is opt-in: the per-tip superposition ignores that arm joints are shared
    # between fingertips, and the residual net wrench of a grasp drifts the end effector
    # (measured: lift overshoots of 30-80 mm). Enable once the coupled multi-contact solve exists
    # (roadmap), or for single-contact tasks where the wrench is genuinely carried by the arm.
    allow_arm_offset: bool = False

    # ---- contact-point channel ----
    point_kp: float = 0.5                  # -    P gain on the tangential contact-point error
    point_clip_m: float = 0.02             # m    per-tip clamp on |p_ref - p_meas| fed to the law
    point_damping: float = 0.02            # m/rad  DLS lambda for the contact-Jacobian pseudo-inverse
    point_offset_clip_rad: float = 0.05    # rad  per-tip clamp on the contact-point offset
    # anchor the statics map at the *predicted* contact point while the fingertip is not yet
    # touching (the measured point does not exist then); falls back to the fingertip origin.
    predict_point_before_contact: bool = True

    # ---- shared ----
    deadband_n: float = 0.1                # N           |error| below this is ignored
    error_clip_n: float = 30.0             # N           per-tip error clamp fed to the law
    # when the *predicted* force drops below this the finger is not meant to press: the
    # integrator leaks back to zero with time constant release_tau_s instead of integrating
    engage_threshold_n: float = 0.2        # N
    release_tau_s: float = 0.15            # s
    meas_ema_alpha: float = 0.4            # EMA on the measured force (per axis, 1.0 = no filter);
    #                                        contact forces are impulsive at 120 Hz
    point_ema_alpha: float = 0.4           # EMA on the measured contact centroid (per axis)
    offset_clip_rad: float = 0.12          # rad  final per-joint |offset| clamp


@dataclass
class ControllerConfig:
    """Top-level bundle: pretend-policy + reference tracking + force feedback."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    force_law: ForceLawConfig = field(default_factory=ForceLawConfig)
    # safety clamp of the final action to the articulation's joint limits. Off in exact-replay
    # mode: the recorded commands must reach the PD verbatim to preserve bit-parity.
    clamp_action_to_limits: bool = True

    def exact_replay(self) -> "ControllerConfig":
        """Preset that must bit-reproduce the recorded replay (framework regression test)."""
        return ControllerConfig(
            policy=PolicyConfig(
                horizon=self.policy.horizon,
                chunk=self.policy.chunk,
                state_source="joint_target",
            ),
            reference=ReferenceConfig(interp="hold", latency_steps=2),
            force_law=ForceLawConfig(law="null"),
            clamp_action_to_limits=False,
        )
