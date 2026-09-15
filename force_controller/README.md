# force_controller — hybrid force controller development pipeline

> **[GUIDE.md](GUIDE.md)** — plain-words walkthrough of the whole pipeline: what each file
> does, the methods behind them, and a debugging playbook. Start there if anything below is
> unclear.
>
> **[REVIEW.md](REVIEW.md)** — code review of this package against the three design
> requirements (no directional assumptions / fingertip tactile sensors only / track contact
> point + force vector + magnitude), with the open issues and a suggested order of work.
> Some statements below are contradicted by it; the review is the current record.

The target system is a BC policy that predicts **robot states and fingertip force readings** in
chunks, and a **middle layer** that turns those predictions plus the *live* force readings into
joint-position targets for the existing low-level PD controller. Until the policy exists, recorded
replay episodes (`outputs/<run>/<stamp>/replay_data.npz`) are sliced and served with exactly the
chunk/horizon semantics a BC policy would have, so the middle layer can be developed and tuned in
simulation against ground truth.

```
                        every C frames (10 Hz frame clock)
  ┌────────────────────┐  chunk: q̂[k..k+H-1], f̂[k..k+H-1]  ┌──────────────────────────────┐
  │ ChunkedReplayPolicy├────────────────────────────────────▶│  HybridForceMiddleLayer      │
  │ (stand-in for BC)  │                                     │  ┌────────────────────────┐  │
  └────────────────────┘                                     │  │ ChunkTracker           │  │
        recorded joint_pos + fingertip forces                │  │ q_ref(t), f_ref(t)     │  │
                                                             │  └───────────┬────────────┘  │
                                                             │  ┌───────────▼────────────┐  │
  ┌────────────────────┐   measured F (120 Hz, world frame)  │  │ ForceFeedbackLaw       │  │
  │ ContactSensors     ├────────────────────────────────────▶│  │ Δq = law(f_ref−f_meas) │  │
  │ (4 fingertips)     │                                     │  └───────────┬────────────┘  │
  └─────────▲──────────┘                                     └──────────────┼───────────────┘
            │                                                action = q_ref + Δq   (120 Hz)
            │              ┌──────────────────────────────┐                 │
            └──────────────┤ Isaac Lab scene (replay.py)  │◀────────────────┘
                           │ implicit PD: arm 400/40,     │
                           │ hand 350/12                  │
                           └──────────────────────────────┘
```

## Why the middle layer exists

The policy predicts where the robot should **be** (`joint_pos`) and what it should **feel**
(fingertip forces). A PD controller must command *past* the measured position to produce force —
in the recordings the command leads the measured hand joints by up to ~0.04 rad during a 95 N
grasp. Feeding predicted positions straight to the PD therefore under-squeezes and drops the
object; the middle layer recovers the missing command lead by closing a loop on
`f_ref − f_meas`. Pass-through (`--force-law null`) is the baseline that demonstrates the problem.

## Files

| file | contents | needs Isaac? |
|---|---|---|
| `config.py` | all dataclass configs (`PolicyConfig`, `ReferenceConfig`, `ForceLawConfig`, `ControllerConfig`) | no |
| `episode.py` | `ReplayEpisode`: loads `replay_data.npz` + `summary.json`, joint-order remapping, force views | no |
| `policy.py` | `BasePolicy` interface + `ChunkedReplayPolicy` (recorded data with chunk/horizon semantics) | no |
| `middle_layer.py` | `ChunkTracker` (10 Hz knots → continuous reference), `TaskSpaceForceLaw` (+ null), `HybridForceMiddleLayer` | no |
| `plots.py` | force-tracking / command-comparison plots, magnitude force metrics | no |
| `metrics.py` | vector force-tracking error \|f_ref_vec − f_meas_vec\| (magnitude + direction split), contact-point error \|p_ref − p_meas\| (tangential + normal split); exact vector reconstruction for runs recorded before vectors were saved | no |
| `replot.py` | regenerate plots + vector metrics from a finished run's `controller_data.npz`, no sim needed | no |
| `offline_check.py` | validates the pipeline against recorded measurements, **no Isaac Sim needed** | no |
| `sim_runner.py` | closed-loop rollout using the validated replay scene | yes |
| `run_tracking.py` | CLI launcher (AppLauncher wiring, saving, plots, replay diff) | yes |

## Conventions (inherited from the replay)

* Frame = 0.1 s = 12 physics steps of 1/120 s. Array index `k` = state at the **end** of frame `k`;
  `joint_target[k]` was held **during** frame `k` (first 2 steps of each frame still held `k−1`).
* Joint arrays are in the recorded sim joint order; `ReplayEpisode.reordered()` re-maps by name
  before driving a live articulation (the orders differ between arrays and articulations — never
  index positionally across sources).
* Forces are world-frame per-fingertip and always **net** — the total force on the pad, table
  contact included. PhysX's per-object breakdown is privileged information no tactile sensor can
  produce, so it is recorded for analysis but never fed back (there is no `force_source` knob).
* The policy is queried at the start of frame `k` and predicts the states at the ends of frames
  `k … k+H−1`; a new chunk arrives every `C` frames. Prediction knot `j` lives at time `k+j+1`
  (frame units); the middle layer interpolates (`linear`) or holds (`hold`) between knots and
  anchors each new chunk at the reference it was already tracking, so chunk switches never jump.

## Validation ladder (run in this order)

```bash
# 1. no sim: chunking/interp/latency must reconstruct the recorded commands bit-exactly,
#    and the requested config is exercised open-loop against recorded readings
python force_controller/offline_check.py --episode outputs/run_2026-05-15_17-55-22

# 2. in sim: the whole runner must bit-reproduce the replay (replays are bit-deterministic here)
conda activate env_isaaclab
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
    --exact-replay --no-render

# 3. baseline: predicted states, no force feedback -> the command lead is not recovered
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
    --force-law null --no-render

# 4. the controller: predicted states + task-space force tracking, with videos
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22
```

**Repeat step 4 before believing a gain change.** The open-loop replay is bit-deterministic, but
the closed loop is not: it feeds the GPU contact-report readback jitter back through the law. With
the contact-point channel on, six identical-config runs of `17-55-22` spread **5.06 N** in force
bias (0.03 N with `--point-kp 0`) — larger than most differences between gain settings. `|dq|`
(`command_fidelity.offset_mean_mrad`) is far steadier (sd 0.61 on a 14.4 mean) and is the better
score to tune on.

Step 2 prints a per-key `max |diff|` against the episode's `replay_data.npz`; every state key
(joints, objects, bodies, frame-end contact force) must be exactly 0.0. The per-step force-report
keys (`contact_force_steps`, `contact_object_force_steps`) may show isolated ≤ 1-float32-ulp
entries — GPU contact-report readback jitter that never feeds back into the dynamics (verified
2026-08-27: 3/23040 entries at exactly 1.00 ulp while all states stayed bit-exact for 160
frames). Anything beyond that is a framework regression, not a tuning problem.

Rollouts **render by default**: the demo camera with force arrows (`rollout.mp4`), the sweeping
force plot (`contact_forces.mp4`) and the annotated side-by-side (`rollout_with_forces.mp4`).
Pass `--no-render` for physics only — a few seconds and ~3.3 GB instead of a few minutes and
~6.6 GB — which is what steps 1–3 above want, and what a gain sweep wants.

Episodes used so far: `outputs/run_2026-05-15_17-52-54`, `outputs/run_2026-05-15_17-55-22`,
`outputs/run_2026-05-16_01-27-32` (latest stamp of each; `--episode` accepts the run folder and
picks the newest stamp with data).

## Sensing contract

The controller may read **fingertip tactile sensors only**, plus proprioception and the robot
model. Concretely, `Measured` carries:

| signal | source | real-robot equivalent |
|---|---|---|
| `f_net` [4, 3] | `ContactSensor.net_forces_w` | total force on the pad |
| `tip_contact_pos` [4, 3] | \|f\|-weighted centroid of the per-pair patch centres (`pad_contact_centroid`) | the pad's pressure centroid |
| `q`, `qd` | articulation state | joint encoders |
| `tip_pos`, `tip_jacobians` | forward kinematics | URDF + FK |

Deliberately absent: the per-object force decomposition (`force_matrix_w[..., manip_col]`), object
poses, and any knowledge of *what* is being touched. The pad centroid is fused across objects for
exactly that reason — a real pad reports one centroid and cannot attribute it.

Caveat, honestly stated: the table is a static collider rather than a filtered object, so contact
with it contributes to `f_net` but reports no patch centre. A pad touching only the table therefore
reads force with a NaN centroid, and the law falls back to the predicted point (then the fingertip
origin) for its Jacobian anchor.

## The force law

**`TaskSpaceForceLaw` (the only law; `null` is the playback baseline).** The target is a full
contact description — contact point `p_ref`, world-frame force vector `f_ref` (ON the fingertip
FROM the environment), magnitude — and the offset is the sum of two channels that live in
orthogonal subspaces of the contact frame. This is the "hybrid" in hybrid force/position control,
except the subspace split is *derived from the prediction* rather than fixed in advance.

**Force channel** — from statics:

    τ = J_c^T · (−f)      (J_c: positional Jacobian at the contact point; the finger pushes with −f)
    τ = K · Δq            (implicit PD stiffness)
    ⇒ Δq_f = −K⁻¹ J_c^T f  applied to  f = kff·f_ref + task_kp·e + ∫task_ki·e·dt + task_kd·ė,
                           e = f_ref − f_meas (vectors)

**Contact-point channel** — the contact-point error mapped back through the damped pseudo-inverse
of the same Jacobian, with the depth-of-press component removed:

    e_p⊥ = (I − n̂ n̂ᵀ)(p_ref − p_meas),   n̂ = f_ref / |f_ref|
    Δq_p = point_kp · J_c^T (J_c J_c^T + λ² I)⁻¹ e_p⊥

The projection is what keeps the two channels from fighting: the component of the contact-point
error *along* `f_ref` is how hard the finger is pressing, which the force channel already
regulates; the tangential remainder is *where on the surface* the finger sits, which the force
channel cannot observe. `n̂` comes from the predicted vector, so no contact direction is assumed.

The pseudo-inverse is solved over only the joints the offset may use (hand-only by default), so
the least-squares redistributes onto them instead of producing arm motion that would be clipped
away.

No direction is assumed anywhere: squeeze, push or pull all emerge from the predicted vector and
the kinematics at the *measured* contact point (fallback: predicted point, then fingertip origin).
`kff = 1` is pure model feedforward — it supplies the command lead for the predicted force
instantly, and the PID cleans up what the statics map misses (contact compliance makes the
effective gain ~0.16 N per N of integral, hence the large `task_i_max_n`; see config.py). The sim
runner provides the contact-point Jacobians from PhysX each step and the articulation's live
stiffness. The offset is hand-only by default: per-tip superposition ignores that arm joints are
shared, and the residual wrench of a grasp measurably drifts the EE (30–80 mm lift overshoots) —
`--allow-arm-offset` enables it for single-contact tasks until the coupled solve exists.

An adaptive feedforward (2026-08-31) replaces the fixed ``kff`` scale: the law estimates, per
tip and online, how much force actually comes back per newton commanded along the predicted
direction (EMA, ``adapt_tau_s``), and multiplies the feedforward by the inverse (clamped to
``ff_gain_max``, rise slew-limited). This is what lets the integral come off its clamp
(66–72 % → 36–40 % of engaged steps) — the PID gains are never scaled. ``--no-adapt-kff``
restores the fixed feedforward.

A friction-cone cap (2026-08-31) bounds `f = f_cmd` to at most `cone_half_angle_deg` (20°)
away from the predicted force direction, and forbids a net pull along it. The prediction was
measured on the real surface, so its direction lies inside the true friction cone; the cap stops
the saturated integral from rotating the command outside it (which measurably ground a grasp out
of the hand on `run_2026-05-16_01-27-32`, mu 0.35 → cone 19.3°). `--cone-deg`, ≤0 disables.
The per-run `grasp_slip` metric (unintended object-in-palm motion vs the episode's own, with a
`dropped` flag) is the detector for this failure class; `replot.py` backfills it.

Engagement semantics: the *predicted* force gates both channels — while `|f_ref|` ≈ 0 the integral
leaks out (`release_tau_s`) and the contact-point term is off, and while engaged-but-not-touching
the law deliberately drives toward contact.

## Refinement roadmap

1. **Multi-contact coupling**: the per-tip Jacobian mapping superposes tips independently; a
   coupled solve (stacked J, one least-squares over both channels) would handle shared arm joints
   cleanly and remove the need for the hand-only `allow_arm_offset` default.
2. **Gain tuning** on the three episodes (`task_kp`/`task_ki`/`point_kp`, EMA alphas; watch for
   120 Hz chatter).
3. **Friction-cone / unilaterality projection** on `f_cmd`: the law is direction-free, so nothing
   currently stops the PID chasing a pulling force a fingertip cannot exert without adhesion, or a
   tangential force outside the friction cone.
4. **Anti-windup against the output clip**: the integral is bounded by `task_i_max_n`, but the
   offset is separately bounded by `offset_clip_rad`; there is no back-calculation, so the
   integrator winds while the output is saturated.
5. **Torque-limit awareness**: fold the hand's 50 N·m effort limit into the offset clamp.
6. **Robustness studies**: perturb the pretend policy (noise on states/forces/contact points,
   dropped chunks, longer chunks) and disturb the object, to quantify what force feedback buys
   over playback.
7. **Real BC policy**: implement `BasePolicy.predict()` on the trained model — the middle layer
   and runner do not change (`run_force_tracking(..., policy=...)`). The policy must predict the
   full force target (net pad force vector + pad contact centroid), which the recordings show is
   learnable signal. Note both are currently world-frame; predicting them in the fingertip frame
   and converting via FK inside the middle layer would be a strictly easier learning problem.
