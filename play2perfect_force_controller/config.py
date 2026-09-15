"""Configuration dataclasses for the hybrid force controller pipeline (play2perfect edition).

Pure-python / numpy-free so every module (offline or in-sim) can import it. All timing is expressed
in *trajectory frames*: here one frame is one 60 Hz play2perfect policy step, i.e.
``steps_per_frame = 2`` physics steps of 1/120 s (the Isaac Gym replays this package descends from
used 0.1 s frames of 12 steps; nothing in the controller depends on the frame length). The state
recorded at index ``k`` is the state at the END of frame ``k``.

Sensing contract (2026-08-29): the controller may read **fingertip tactile sensors only** — the
net contact force on each fingertip pad and the pad's contact centroid — plus proprioception
(joint states) and the robot model (FK / Jacobians / PD gains). The simulator's per-object force
breakdown is privileged information no tactile sensor can produce; it stays available for
*analysis* but is unreachable from the control path (there is deliberately no ``force_source``
knob).

[play2perfect] The only robot-specific edits versus ``force_controller/config.py``: the arm joint
names and PD stiffness table come from :mod:`robot_spec` (iiwa14 + Sharpa instead of Kinova +
LEAP), the exact-replay preset uses latency 0 (the env applies the target at both substeps of a
frame), and the chunk/horizon defaults are in 60 Hz frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# [play2perfect] robot-specific names/gains live in robot_spec so this file stays generic
from .robot_spec import (  # noqa: F401  (re-exported for middle_layer / offline tools)
    ARM_JOINT_NAMES,
    STALE_TARGET_STEPS,
    default_joint_stiffness,
)


@dataclass
class PolicyConfig:
    """How the recorded episode is served as if it came from a BC policy."""

    # [play2perfect] frames are 60 Hz policy steps: H=24 / C=12 predict 0.4 s and re-plan every
    # 0.2 s. (force_controller/ used 16 / 8 at 10 Hz, i.e. 1.6 s / 0.8 s; pass --horizon/--chunk
    # 96 / 48 for the same wall-clock chunking.)
    horizon: int = 24          # predicted frames per query (H)
    chunk: int = 12            # frames executed before the next query (C <= H)
    # what plays the role of the policy's "predicted robot state":
    #   "joint_pos"    - the recorded *measured* states (true BC semantics: the policy predicts
    #                    where the robot should BE; the middle layer must recover the command)
    #   "joint_target" - the recorded *commands* (upper bound / exact-replay validation mode)
    state_source: str = "joint_pos"
    # who serves the chunks: "replay" (slices of the recorded episode, open loop) or "oracle"
    # (the RL policy rolled forward in the simulator from the CURRENT state at every chunk
    # boundary - closed loop at chunk rate; see oracle_policy.py)
    source: str = "replay"
    # NOTE: there is no force_source. The force target is always the NET fingertip force, the only
    # thing a tactile pad can report, so prediction and measurement are the same quantity.


@dataclass
class ReferenceConfig:
    """How the 10 Hz chunk knots become a continuous reference at physics rate."""

    # "linear": piecewise-linear between knots (smooth upsampling, the deployment default)
    # "hold"  : zero-order hold-right (reference == next knot), which reproduces the replay's
    #           constant-per-frame commands exactly
    interp: str = "linear"
    # command latency in physics steps: the reference is evaluated at t - latency*dt. The Isaac
    # Gym replays held the previous frame's command for the first 2 steps of each frame (stale
    # target, latency 2 + "hold" reproduced it); the play2perfect env applies the policy's target
    # at both substeps of a frame, so its exact replay is latency 0 (robot_spec.STALE_TARGET_STEPS).
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
    offset_clip_rad: float = 0.12          # rad  final per-joint |offset| clamp (uniform fallback)
    # [play2perfect] two joint-limit-aware refinements of the statics map (2026-09-08):
    # (1) a joint sitting at a position limit is rigid there - an offset pushing it further into
    #     the limit produces no torque and no force, so its Jacobian column is masked for that
    #     direction (the recorded policy holds the thumb IP pinned at its upper limit while it
    #     squeezes; the unmasked map spent most of the thumb's offset exactly there);
    limit_margin_rad: float = 0.01         # rad  "at the limit" band
    # (2) the per-joint clip comes from the effort limit: |dq_j| <= tau_max_j / K_j is the offset
    #     beyond which the PD spring saturates anyway (Sharpa: 0.71 rad IP/PIP, 0.21 rad DIP,
    #     0.25 rad CMC), instead of one uniform 0.12 rad that stops the soft joints short while
    #     the stiff ones never come near it. Evaluated 2026-09-08 on the 12 perfect episodes: WORSE
    #     than the uniform 0.12 clip on every task (slip 85->269 mm on beam1, 35->86 mm on
    #     screwing) - the soft joints run to 0.7 rad and the grasp destabilises before force
    #     builds. Off by default; --effort-clip switches it on.
    offset_clip_from_effort: bool = False
    # ---- force -> offset map (2026-09-09, from the closed-loop evaluation) ----
    # "statics":     dq = -K^-1 J^T f, then the per-joint clip (closed form; the default).
    # "constrained": the same statics model solved as a bounded least squares per fingertip:
    #                the offset that makes the finger's PD deliver the force CLOSEST to f_cmd
    #                (least squares in force space, plus a penalty on torque outside the contact
    #                Jacobian's row space, which would only move the finger) within per-joint
    #                bounds - the offset clip and the effort limit tau_max_j / K_j - with joints
    #                sitting at a position limit they would push into treated as rigid (their
    #                torque is free). Where no bound binds it equals the statics map exactly;
    #                where one does, it keeps the force direction instead of clipping joint by
    #                joint (which bends the delivered force away from the prediction, i.e. out of
    #                the friction cone). Zero-shot: only the robot model enters.
    map: str = "statics"
    map_null_arm_m: float = 0.05           # m    torque outside the row space is weighed as force
    #                                             at this lever arm (1 N.m -> 20 N at 0.05 m)
    map_reg_n_per_rad: float = 1e-2        # N/rad Tikhonov on the offset (conditioning only)
    # ---- contact-point channel gating (opt-in) ----
    # act on WHERE the contact sits only when the plan says the contact is established and
    # steady: predicted |f| >= point_engage_n and its relative rate below point_steady_rate. A
    # re-grasp shows in the plan as a fast drop/rise of the predicted force; pulling the pad
    # toward a predicted point while the plan is moving it fights the position layer.
    point_engage_n: float = 0.0            # N    0 = engage_threshold_n (previous behaviour)
    point_steady_rate: float = 0.0         # 1/s  0 = no rate gate


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
            reference=ReferenceConfig(interp="hold", latency_steps=STALE_TARGET_STEPS),
            force_law=ForceLawConfig(law="null"),
            clamp_action_to_limits=False,
        )
