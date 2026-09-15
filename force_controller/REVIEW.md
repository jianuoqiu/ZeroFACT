# Code review — `force_controller/` against the three design requirements

Review date: 2026-08-29. Reviewed: all 11 python files in `force_controller/`, the contact-sensor
setup in `v2s2r_isaaclab/replay.py`, and every `summary.json` under `outputs/force_controller/`.

The requirements this review checks against:

1. **No assumption about how to reduce force tracking error.** No grasp-specific design — e.g. no
   "just push the fingertips along the fingertip normal", no fixed squeeze directions.
2. **Only fingertip tactile sensors.** The controller may use proprioception and the robot model,
   but nothing a real tactile sensor cannot report.
3. **Force tracking means all three channels:** contact point position, force vector, force
   magnitude.

---

## Status

Findings 1–3 were **acted on 2026-08-29** (see *Resolution* notes inline). The exact-replay
regression stayed bit-exact through the refactor, so the framework itself is unchanged.

| Requirement | Status before | Status now |
|---|---|---|
| 1 — no directional assumptions | Mostly met, 2 leaks | **Met** — squeeze law deleted; arm restriction now solved *within* the allowed joint set instead of truncating |
| 2 — fingertip tactile only | Violated by default | **Met** — the loop reads net pad force + pad centroid only; the privileged path is unreachable from any config |
| 3 — point + vector + magnitude | 2 of 3 | **Met** — contact point is now tracked (a control channel), recorded, measured and plotted |

Still open (see *Suggested order of work*): the coupled multi-contact solve and output-clip
anti-windup. (Friction-cone projection and the adaptive feedforward — the integral-saturation
cure — were shipped 2026-08-31 after a dropped-grasp rollout was traced to them; the contact
sensor's readback buffer was raised 32 → 64 for the firmer grasps, with exact-replay re-verified
bit-exact.)

## Verdict at a glance (as found)

| Requirement | Status | Where it breaks |
|---|---|---|
| 1 — no directional assumptions | **Mostly met**, 2 leaks | Legacy squeeze law is `offline_check.py`'s default; `allow_arm_offset=False` is a squeeze prior in disguise |
| 2 — fingertip tactile only | **Violated by default** | Closed loop reads PhysX's *per-object* force and contact point — simulator-only signals |
| 3 — point + vector + magnitude | **2 of 3** | Contact point is predicted, interpolated, then never used or measured |

The core design — `TaskSpaceForceLaw` — is a correct and honest answer to requirement 1. The
problems are around it: privileged sensor signals, a structural arm restriction, and a dangling
contact-point channel.

## The pipeline

```
ChunkedReplayPolicy ──chunk every C=8 frames (10 Hz)──▶ ChunkTracker ──q_ref(t), f_ref(t)──┐
  (stands in for BC)   q̂[k..k+15], f̂[k..k+15], p̂[k..k+15]                                   ▼
ContactSensors (4 fingertips, 120 Hz) ─────────────────────────────▶ ForceFeedbackLaw ──▶ Δq
                                                                                           │
                        Isaac Lab scene, implicit PD (arm 400/40, hand 350/12) ◀── q_ref + Δq
```

## What each file does

| file | job |
|---|---|
| `config.py` | All dataclass configs: `PolicyConfig` (H=16, C=8, `state_source`, `force_source`), `ReferenceConfig` (interp, latency), `ForceLawConfig` (law choice + every gain), `ControllerConfig` + its `exact_replay()` preset. Also holds `DEFAULT_SQUEEZE_WEIGHTS` and the PD stiffness constants the statics map needs. |
| `episode.py` | Loads `replay_data.npz` + `summary.json` into a `ReplayEpisode`. Owns the index convention (`joint_pos[k]` = end of frame k, `joint_target[k]` = held *during* frame k), joint-name→articulation-order remapping (`reordered()`), and the three force-target views: `fingertip_force_mag`, `fingertip_force_vec`, `fingertip_contact_point`. |
| `policy.py` | `BasePolicy` interface (what the real BC model will implement) + `ChunkedReplayPolicy`, which slices the recording with exact ACT-style chunk semantics. `PolicyOutput` carries the full target: joint states, force magnitude, force **vector**, contact **point**. |
| `middle_layer.py` | The controller. `ChunkTracker` turns 10 Hz knots into a continuous reference (linear/hold, anchored at the previous reference so chunk switches don't jump). `TaskSpaceForceLaw` / `FingerSqueezeLaw` / `NullForceLaw` map force error → Δq. `HybridForceMiddleLayer` glues them: EMA-filter the measured force, evaluate the reference at `t = frame + (step+1−latency)/12`, `action = q_ref + clip(Δq)`. |
| `sim_runner.py` | The closed loop in Isaac Lab. Reads sensors *before* each `sim.step()` (1-step feedback delay), builds contact-point Jacobians from PhysX (`_tip_point_jacobians`: shifts the body Jacobian to the contact point via `J_p = J_v − skew(p−p_body)·J_w`), calls the middle layer, writes the PD target, records everything. `diff_against_replay` is the bit-exactness regression check. |
| `run_tracking.py` | CLI launcher. AppLauncher/Vulkan wiring, the broken-install rendering workaround, builds `ControllerConfig` from flags, runs the rollout, saves `controller_data.npz`, generates all plots + metrics + `summary.json`, prints the report. |
| `offline_check.py` | No-Isaac validation. (1) self-test that hold + latency-2 + null law reconstructs the recorded commands bit-exactly; (2) replays the requested config open-loop against recorded readings. |
| `metrics.py` | Vector force metrics: `|f_ref − f_meas|` split into magnitude bias vs angle error. Plus exact reconstruction of reference/filtered vectors for runs recorded before those arrays were saved. |
| `plots.py` | `force_tracking.png` (magnitude), `commands_vs_predicted.png` (the command-lead gap), `force_vector_error.png` (total / ∥ / ⊥ split), `force_components.png` (world XYZ), and the magnitude metrics. |
| `replot.py` | Re-runs metrics + plots from a finished `controller_data.npz`, no sim needed. |
| `__init__.py` | Numpy-only re-exports; Isaac-dependent code deliberately excluded. |

---

## Requirement 1 — no assumption about how to reduce force tracking error

### What is correct

`TaskSpaceForceLaw` (`middle_layer.py:203`) is genuinely direction-free, and the math checks out.
Static equilibrium of the arm under an external force on the fingertip gives `τ + Jᶜᵀ f_ext = 0`,
and the implicit PD gives `τ = K·Δq`, hence

```
Δq = −K⁻¹ Jᶜ(p)ᵀ f_cmd ,    f_cmd = kff·f_ref + kp·e + ∫ki·e·dt + kd·ė ,    e = f_ref − f_meas
```

with `e` a full **3-vector** (`middle_layer.py:266-275`). Nothing anywhere names a contact normal,
a flexion axis, or a squeeze direction. Push, pull and pinch all fall out of the predicted vector
and the kinematics at the contact point. Sign convention verified correct.

### Leak A — the squeeze law is still the default in the one script that runs without Isaac

`offline_check.py:141` defaults to `--force-law finger_squeeze`, i.e. `DEFAULT_SQUEEZE_WEIGHTS`
(`config.py:24`) — hardcoded per-finger MCP/PIP/DIP flexion directions. That is exactly the design
that was ruled out.

Worse, `offline_check.py` exposes `--kp/--ki/--u-max` for the squeeze law and **no flags at all**
for the task-space gains, and the task-space law cannot run offline anyway (no Jacobians → it warns
once and returns zero offsets, `middle_layer.py:240-245`). So the "no sim" rung of the validation
ladder never exercises the law that actually ships.

**Fix:** flip the default to `task_space`, add `--kff/--task-kp/--task-ki/--task-kd`, and either
delete `FingerSqueezeLaw` or move it behind an explicit `--legacy` flag.

> **Resolved 2026-08-29.** `FingerSqueezeLaw` and `DEFAULT_SQUEEZE_WEIGHTS` are deleted, along
> with the `squeeze_weights`/`kp`/`ki`/`u_max_rad` config fields and the `--kp/--ki/--u-max` CLI
> flags. `make_force_law` now offers `null` and `task_space` only, and `offline_check.py` defaults
> to `task_space` with `--kff/--task-kp/--task-ki/--task-kd/--point-kp` exposed. Rollouts recorded
> with the old law still replot: `metrics._only_known` drops removed config keys and the reference
> reconstruction forces the null law (it only needs the ChunkTracker).

### Leak B — `allow_arm_offset=False` is a squeeze prior in disguise

`config.py:107` defaults `allow_arm_offset` to `False`, and `middle_layer.py:371-374` implements
that by setting the per-joint offset clip to **0.0** for all 7 arm joints.

This encodes "the fingers generate force, the arm holds still" — a grasping assumption. And it does
not cleanly *disable* arm participation: the statics solve at `middle_layer.py:271-275` computes a
Δq that **includes** arm components, and those components are then silently truncated to zero. The
achieved force is therefore biased low, and the integrator winds against a bound it can never
reach.

Evidence from the recorded runs (`outputs/force_controller/`):

| episode | `allow_arm_offset` | index bias | thumb bias |
|---|---|---|---|
| `run_2026-05-15_17-55-22` (20260828_170551) | **True** | −1.55 N | −0.82 N |
| `run_2026-05-15_17-55-22` (20260828_170849) | **False** | **−9.58 N** | **−7.92 N** |
| `run_2026-05-16_01-27-32` (20260828_171503) | False, retuned ki=10 | −1.82 N | −3.06 N |

The "integrals pegged" symptom already noted in the project log is this mechanism.

For any press / wipe / insert task the arm carries the wrench, and with this default the controller
is **structurally unable** to track the target at all.

The stated justification (per-tip superposition ignores that arm joints are shared) is real, but
the remedy is the coupled solve, not a zero clip.

**Fix:** replace the per-tip loop with one stacked least-squares solve over all tips and all
joints — build the `[3·tips × J]` stacked Jacobian and solve
`min ‖(J_stackedᵀ K Δq) − f_cmd‖²` once. That removes the approximation and the assumption
together, and lets the arm participate correctly.

> **Partly resolved 2026-08-29.** The truncation is gone as a *silent* effect: the law now carries
> an explicit `joint_mask`, and the new contact-point channel solves its damped least-squares
> *within* the allowed joint set, so the solution redistributes onto usable joints instead of
> producing arm motion that gets clipped away. The force channel is unchanged (masking a
> projection before or after the clip is identical). The coupled stacked solve — which would
> remove the `allow_arm_offset` switch entirely — is still open and is now item 1 of the roadmap.

### Leak C (minor) — per-tip superposition

`Δq = −Σᵢ K⁻¹ Jᵢᵀ fᵢ` is exact for the hand (the LEAP finger joint sets are disjoint) and wrong for
the arm. Same fix as Leak B.

### Not an assumption, but a missing physical constraint

Because the law is direction-free, nothing stops the PID from commanding a **pulling** force a
fingertip cannot exert without adhesion, or a tangential force outside the friction cone. The
integrator will wind up chasing an unachievable component.

This is a unilaterality / friction-cone projection on `f_cmd` — physics, not a task prior — and is
worth adding. It is the *opposite* of the assumption requirement 1 forbids: it constrains the
command to what contact mechanics permits, without presuming any particular direction.

---

## Requirement 2 — only fingertip tactile sensors

### Violation D (the important one) — the closed loop feeds on privileged simulator information

`PolicyConfig.force_source` defaults to `"manipulated"` (`config.py:59`). That makes
`sim_runner.py:176` read

```python
s.data.force_matrix_w[0, 0, manip_col]   # force exchanged with the manipulated object ONLY
```

which is PhysX's per-pair contact filter. **No tactile sensor can do that.** A real fingertip
reports the *total* force on the pad: the object, plus the table, plus other fingers, plus anything
else it brushes. The filter list is built from the scene objects in
`v2s2r_isaaclab/replay.py:700-707`, so the separation is purely a simulator capability.

Every run in `outputs/force_controller/` used `force_source: manipulated`, so **all current gain
tuning rests on a signal that does not exist on hardware.**

`--force-source net` is the honest setting and is one flag away.

> **Resolved 2026-08-29.** `PolicyConfig.force_source` and `--force-source` are **deleted**: the
> loop always reads `net_forces_w`. `Measured` no longer has an `f_manip` field at all, so there is
> no code path by which the per-object breakdown can reach the law. It is still *recorded* for
> analysis (that is not a cheat), and `ChunkedReplayPolicy` keeps a constructor-only
> `force_source` kwarg used solely by `metrics.py` to reproduce pre-2026-08-29 rollouts.

### Violation E — the contact point is object-filtered too

`sim_runner.py:179` reads `contact_pos_w[0, 0, manip_col]` — the same per-object filter, returning
PhysX's contact-patch centre for that specific pair. On hardware the contact location comes from a
taxel centroid in the **fingertip frame**, converted to world by forward kinematics.

> **Resolved 2026-08-29.** `episode.pad_contact_centroid` fuses the per-(fingertip, object) patch
> centres into **one centroid per pad**, weighted by the force carried at each patch — the
> simulator's stand-in for a pressure centroid, and object-agnostic by construction. Both the sim
> runner and `ReplayEpisode.fingertip_contact_point` use it. Honest caveat now documented in the
> README: the table is a static collider rather than a filtered object, so a pad touching only the
> table reads force with a NaN centroid, and the law falls back to the predicted point.

### Frame choice — a decision worth making deliberately

Everything is world-frame: the measured force, the predicted vector, and the predicted contact
point. A tactile sensor measures in the sensor frame; converting to world needs FK, which is
available, so this is legal.

But it means the **BC policy must output world-frame force vectors and world-frame contact
points**, which is a substantially harder learning problem than the fingertip-frame equivalent.
`Measured.tip_quat` (`middle_layer.py:56`) is declared for exactly this purpose and `sim_runner`
never populates it.

Recommendation: define the force target in the fingertip frame (or the object frame) and convert
inside the middle layer.

### What is legitimate and used correctly

Joint positions and velocities (proprioception), fingertip poses and Jacobians from the robot
model (FK), and the PD stiffness read from the live articulation. All fine.

### One sim2real gap on the measurement side

PhysX contact reports carry **no torque channel** and **no inertial/gravity component**. A real
6-axis fingertip sensor reads gravity and inertia too and would need compensation before the
readings are comparable to what this pipeline treats as "measured force".

---

## Requirement 3 — contact point + force vector + magnitude

### Force vector — tracked ✅

PID on the 3-vector error, feedforward on `f_ref`, and `metrics.vector_force_metrics` reports
`rmse_vec_N` split into magnitude bias and angle error. `plot_vector_error` and
`plot_force_components` visualise it.

Latest numbers (`run_2026-05-16_01-27-32`, 20260828_172219): vector RMSE 6.4 / 3.1 / – / 14.4 N,
angle error 7.6° / 10.8° / – / 19.0°.

### Force magnitude — tracked, but carried as a second, inconsistent channel ⚠️

`f_mag` is interpolated **independently** of `f_vec` in `ChunkTracker` (`middle_layer.py:159-167`),
so after linear interpolation `f_mag ≠ |f_vec|` — the lerp of norms is ≥ the norm of the lerp. They
agree only exactly at the knots.

Consequence: the law's engagement gate uses `ref.f_mag` (`middle_layer.py:255`) while
`metrics.vector_force_metrics` uses `|f_ref_vec|`. Two different "engaged" masks inside the same
run, so the law's notion of when it is pressing and the metric's notion do not coincide.

**Fix:** derive magnitude from the interpolated vector and drop the separate channel.

> **Resolved 2026-08-29.** `ChunkTracker.eval` now derives `f_mag = |f_vec|` whenever vectors are
> available, so the law's engagement gate and the vector metrics agree by construction.

### Contact point — NOT tracked ❌

This is the real gap for requirement 3.

The policy predicts it (`PolicyOutput.contact_point`), the `ChunkTracker` interpolates it NaN-aware
into `Reference.contact_point` (`middle_layer.py:122-127`, `_blend_points`) — **and nothing ever
reads it.** Grepping the whole package: no force law, no metric, no plot consumes
`ref.contact_point`. It is dead data.

Three consequences:

1. **The Jacobian is anchored at the *measured* contact point only.** `README.md` claims
   "*fallback: predicted point, then fingertip origin*", but `sim_runner.py:184` passes only
   `tip_contact_pos` (measured) and `_tip_point_jacobians` falls back straight to the fingertip
   body origin. The predicted point is never used. **Doc/code mismatch.**
2. **`ControlStep` records no contact points**, so `controller_data.npz` has no `p_ref`/`p_meas`
   pair, and there is **no contact-point error metric or plot anywhere**. Contact-point tracking
   currently cannot even be *measured*, let alone controlled.
3. Making it a tracked quantity is a genuine design decision, not a bug fix: it needs a term that
   slides the contact along the object surface — e.g. folding the tangential position error
   `p_ref − p_meas` into the same stacked task-space solve, so force and contact location are
   traded off in one place rather than one silently dominating.

> **Resolved 2026-08-29.** The contact point is now a tracked quantity on all three counts:
>
> * **Controlled** — `TaskSpaceForceLaw._point_offset` adds a second channel,
>   `Δq_p = point_kp · J_c⁺ (I − n̂n̂ᵀ)(p_ref − p_meas)` with `n̂ = f_ref/|f_ref|` and `J_c⁺` a damped
>   pseudo-inverse over the allowed joints. Projecting out the component along `f_ref` is what
>   keeps the two channels from fighting: that component is depth-of-press, which the force channel
>   already regulates. The subspace split is *derived from the prediction*, so no direction is
>   assumed — this is the "hybrid" in hybrid force/position control, with data-defined subspaces.
> * **Used in the statics map** — `_tip_point_jacobians` now falls back measured → **predicted** →
>   body origin, which is what the README always claimed and the code never did.
> * **Measured** — `ControlStep` carries `p_ref`/`p_meas`, the runner records
>   `p_ref_steps`/`p_meas_steps`/`point_err_steps`, `metrics.contact_point_metrics` reports the
>   error in mm split tangential/normal plus a `measured_fraction` (how often a commanded contact
>   actually produced one), and `plots.plot_contact_point_error` draws it.
>
> Measured effect, both dev episodes, everything except `point_kp` at defaults (net force,
> hand-only). "old" = the last pre-change rollout, which used the *privileged* per-object force:
>
> **`run_2026-05-15_17-55-22`** (two-finger pinch, 95 N grasp; episode lift 138.7 mm)
>
> | | old (privileged) | `--point-kp 0` | **default** `--point-kp 0.5` |
> |---|---|---|---|
> | index force bias | −6.80 N | −6.58 N | **−0.14 N** |
> | thumb force bias | −2.77 N | −3.13 N | **+0.31 N** |
> | index contact-point error | not measurable | 9.53 mm | **6.16 mm** |
> | thumb contact-point error | not measurable | 11.93 mm | **8.62 mm** |
> | lift | 140.5 mm | 140.0 mm | 133.7 mm |
>
> **`run_2026-05-16_01-27-32`** (three-finger, episode lift 319.5 mm)
>
> | | old (privileged) | `--point-kp 0` | **default** `--point-kp 0.5` |
> |---|---|---|---|
> | index force bias | −1.82 N | −1.78 N | **−1.67 N** |
> | thumb force bias | −3.06 N | −2.58 N | **−1.91 N** |
> | index contact-point error | not measurable | 10.37 mm | 10.80 mm |
> | thumb contact-point error | not measurable | 14.87 mm | 14.90 mm |
> | lift | 302.2 mm | 302.4 mm | 300.3 mm |
>
> Read together: **removing the privileged force signal cost nothing** — the `--point-kp 0` column
> matches the old privileged column on both episodes. The contact-point channel is a clear win on
> `17-55-22` (magnitude bias essentially eliminated, contact-point error down ~30 %) and roughly
> neutral on `01-27-32` (bias slightly better, contact-point error unchanged).
>
> Why neutral there is worth understanding before tuning: on `01-27-32` the per-tip offset
> saturates at `point_offset_clip_rad` and the error still does not close, which suggests much of
> that 10–15 mm is *where the object is*, not where the finger is — the object settles differently
> in the rollout than in the recording, and no amount of finger motion fixes that. The contact-point
> channel can only correct the finger's share.

---

## Other concrete issues

| # | issue | status |
|---|---|---|
| 1 | **No anti-windup against the output clip.** The integral is clamped at `task_i_max_n` = 300 N, but the *output* is clamped at `offset_clip_rad` = 0.12 rad — and at 0 for arm joints — with no back-calculation. The integrator keeps winding while the output is saturated. | **open** (roadmap 4) |
| 2 | **Silent wrong-object fallback.** `fingertip_contact_point()` fell back to `col = 0` (the first filtered object) when the manipulated key was missing from the sensor filters. | **fixed** — the method no longer indexes a single object column at all; it fuses every pair into one pad centroid |
| 3 | **Meaningless ulp counts on zero references.** `sim_runner.diff_against_replay` — `np.spacing(\|b\|)` on a zero reference returns a denormal, so `max_ulp` explodes whenever a zero-valued entry differs. | **open** |
| 4 | **Multi-patch contacts collapse to one point.** | **now deliberate** — a tactile pad reports one centroid, so `pad_contact_centroid` fuses patches by design rather than by accident. What remains is the honest gap that the *table* is a static collider and reports no patch centre at all (documented in the README) |
| 5 | **`Measured.tip_quat` declared but never populated.** | **fixed** — field removed. It comes back when force targets move to the fingertip frame |
| 6 | **Contact centroid discarded zero-force patches** (bug introduced by the fix itself, found and fixed the same day). | **fixed** — see below |

### Issue 6 in detail

The first version of `pad_contact_centroid` weighted patches by `|f|` and required `w > 0`, so a
patch that PhysX reported with *exactly zero* force produced a NaN centroid. The old code took the
patch centre whenever it was finite, regardless of force. At 120 Hz the normal force crosses zero
constantly, and through the whole light-touch approach phase the pad reports a centre at ~0 N — so
the new code threw the contact location away exactly where it mattered, and the Jacobian fell back
to the fingertip origin (the wrong lever arm).

Consequences on `run_2026-05-16_01-27-32`: the controller under-closed, the integral wound up, and
it pressed hard enough to exceed `max_contact_data_count_per_prim = 32` — PhysX logged *"Incomplete
contact data is reported in GpuRigidContactView::getContactData"* and Isaac Lab's
`ContactSensor.update` tripped a CUDA device-side assert that killed the process at ~frame 90.

Found by diffing `dq_steps` against the pre-change rollout: the offsets diverged at **frame 52**,
one frame *before* `f_ref` did, which ruled out every reference-side change and pointed at the
Jacobian anchor. The fix keeps the force weighting when the pad carries load and falls back to the
unweighted mean of the reported patches when it does not; NaN now means only "no patch reported at
all".

| `run_2026-05-16_01-27-32`, `--point-kp 0 --no-predict-point` | index bias | thumb bias | lift |
|---|---|---|---|
| buggy centroid | −20.9 N | −23.4 N | 171.9 mm |
| fixed centroid | **−1.78 N** | **−2.58 N** | **302.4 mm** |
| old privileged run, for reference | −1.82 N | −3.06 N | 302.2 mm |

With the fix the overflow is gone at the **stock cap of 32** — raising it was never needed, and
`v2s2r_isaaclab/replay.py` is untouched. (It was raised to 128 for one diagnostic run and restored.)

## Suggested order of work

**Done 2026-08-29** — squeeze law deleted; the loop reads fingertip tactile only; the contact point
is a tracked quantity (controlled, recorded, measured, plotted). Exact-replay regression re-verified
bit-exact on both dev episodes.

**Next:**

1. **Replace the per-tip loop with a stacked multi-contact least-squares solve.** Removes the
   remaining superposition approximation and retires the `allow_arm_offset` switch.
2. **Sweep `point_kp`, and separate finger error from object error.** The channel eliminates the
   magnitude bias on `17-55-22` but is neutral on `01-27-32`, where the per-tip offset saturates
   without closing the error — decompose the contact-point error into the part the finger can
   reach and the part caused by the object sitting elsewhere than in the recording.
3. **Add friction-cone / unilaterality projection** on `f_cmd`, and back-calculation anti-windup
   against `offset_clip_rad`.
   > **Cone/unilaterality done 2026-08-31**, evidence-driven: a dropped-jar rollout on
   > `01-27-32` traced to the thumb's force direction running 29–35° outside the contact's
   > 19.3° friction cone (mu 0.35). `f_cmd` is now capped at 20° from the predicted direction,
   > no pulls; 3/3 repeat runs hold (was 2/3), thumb angle 17–24°, slip metric (`grasp_slip`)
   > added to detect this class automatically. Anti-windup remains open.
4. **Consider fingertip-frame force targets.** Both the force vector and the contact point are
   world-frame today, which makes the BC policy's job harder than it needs to be.
