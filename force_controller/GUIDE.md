# force_controller — plain-words guide

Open this file when you want to know **what a script does**, **what a number means**, or
**where to look when a rollout misbehaves**. It uses simple words on purpose.

The other two documents: `README.md` is the compact technical reference (math, conventions,
validation ladder). `REVIEW.md` is the code review with the open problems and their history.
This guide repeats some of both, but slower and friendlier.

Last updated: 2026-08-31. If the code changes, check the git-less way: file dates
(`ls -la force_controller/`) vs the date above.

**Map of this file**

| § | title | go here when |
|---|---|---|
| 1 | The problem, in one story | you want the why: command lead, `action = q_ref + Δq`, the sensing rule |
| 2 | The loop, step by step | you want the what-happens-when, per frame and per physics step |
| 3 | Who does what | you want to know which file to open |
| 4 | Time rules | a signal looks shifted — indexing/latency conventions |
| 5 | The methods | how each piece works: 5.1 chunk tracking · 5.2 filtering · 5.3 force channel (K, the `−K⁻¹Jᵀf` derivation, the PID) · 5.4 contact-point channel (`e_p`, the pseudo-inverse) · 5.5 the Jacobian · 5.6 the pad centroid · 5.7 state-vs-force balance · 5.8 going to a real robot |
| 6 | What a rollout writes | which output file answers which question; npz keys |
| 7 | The scoreboard | what each summary.json number means; what to tune on |
| 8 | Debugging playbook | the ladder, the noise rules, symptom → where to look, open problems |
| 9 | Defaults at a glance | every knob, value, and its flag |
| 10 | Tiny glossary | short words |

---

## 1. The problem, in one story

The future system is a **BC policy** (a neural network trained on demonstrations). Ten times per
second it will look at the robot and predict a short movie of the near future:

> "Here is where the joints should **be** for the next 16 frames, and here is what each fingertip
> should **feel**: how hard, in which direction, at which spot on the object."

The robot's joints are driven by a **PD controller**. Think of each joint as a **spring**: you
give it a target angle, and the motor pulls toward that target — the further away, the harder the
pull.

Here is the trouble. If you send the spring target exactly where the finger should *be*, the
finger arrives there and pushes with almost **nothing**. To press on something, a spring must aim
**past** the surface. In our recordings, during a hard grasp, the recorded command aims up to
~0.04 rad past where the finger actually is. This "aim past" is called the **command lead**.

The BC policy predicts positions, not commands — so the lead is missing. The **middle layer**
(this package) is the piece that puts it back:

```
action = q_ref + Δq
```

- `q_ref` = the policy's predicted joint state (where to be)
- `Δq` = a small extra angle, computed from the difference between the predicted feel and the
  live tactile reading (how to press)

There is no BC policy yet. A **recorded episode** plays its role perfectly (`ChunkedReplayPolicy`
serves slices of the recording with exactly the timing a real policy would have). This is great
for development: the recording also contains the *ideal command* (`joint_target`), so we can check
the controller against a known right answer.

**The sensing rule (do not break it):** the controller may only read what a real robot would
have — fingertip tactile pads (net force on the pad + one touch spot per pad), joint angles and
velocities, and the robot model (kinematics, spring stiffness). The simulator also knows the
force split per object and the object poses — those are **analysis only** and there is
deliberately no switch to feed them into the control loop.

---

## 2. The loop, step by step

Two clocks:

- **frame** = 0.1 s (10 Hz). The policy speaks at this rate.
- **physics step** = 1/120 s. The controller and the simulation run at this rate.
  12 steps = 1 frame.

**At the start of a frame, when `frame % 8 == 0`** (every 8th frame):

1. Ask the policy for a new **chunk**: 16 predicted frames of (joint state, force vector, force
   size, touch spot).
2. Hand it to the `ChunkTracker`. The tracker glues it to whatever it was already tracking, so
   the target never jumps at the seam.

**At every physics step:**

1. **Read tactile**: net force per fingertip pad + the pad's touch spot (centroid). This reading
   is from the end of the *previous* step — one step old, like a real sensor.
2. **Smooth** both readings a little (EMA filter, see §5.2).
3. **Ask the tracker**: "what should the state / force / touch spot be *right now*?" It
   interpolates between the 10 Hz predictions.
4. **Build the Jacobians** — the small math tables that connect joint motion to fingertip motion —
   anchored at the touch point (measured spot if touching, else the predicted spot, else the
   fingertip itself).
5. **Force law** computes `Δq` = force channel + contact-point channel (§5.3, §5.4).
6. **Clip** `Δq` (each joint at most 0.12 rad; arm joints get 0 unless `--allow-arm-offset`).
   `action = q_ref + Δq`, clipped to the joint limits.
7. **Send** the action to the PD, step the physics once, refresh the sensors, record everything.

That's the whole controller. Everything else in the package is loading, checking, scoring, or
drawing.

---

## 3. Who does what

| file | plain-words job | needs Isaac? |
|---|---|---|
| `config.py` | Every knob, with defaults and comments. **Start here** to see what is tunable. | no |
| `episode.py` | Loads a recorded replay (`replay_data.npz` + `summary.json`) as clean arrays. Knows the array layout and joint ordering. Builds the tactile "views": net pad force and the fused pad centroid (`pad_contact_centroid`). | no |
| `policy.py` | The pretend policy. `ChunkedReplayPolicy` serves recording slices with real chunk timing. `BasePolicy` is the interface the real BC policy will implement later — the rest of the pipeline will not change. | no |
| `middle_layer.py` | **The actual controller.** `ChunkTracker` (10 Hz predictions → smooth 120 Hz target), `TaskSpaceForceLaw` (the two channels), `HybridForceMiddleLayer` (filtering + glue + clipping). Pure numpy — easy to unit-test. | no |
| `sim_runner.py` | The closed loop inside Isaac Lab. Reads sensors, calls the middle layer, steps physics, records every signal. Also `diff_against_replay` for the bit-exact regression test. | yes |
| `run_tracking.py` | The launcher. Flags → config, builds the scene, runs the rollout, writes the npz + videos + plots + `summary.json`, prints the report. | yes |
| `offline_check.py` | The no-sim test bench. Proves the timing plumbing reconstructs the recorded commands **bit-exactly**, then previews your config open-loop against recorded readings. Runs in ~2 s. | no |
| `metrics.py` | The scoreboard: force size / vector / per-axis errors, contact-point error, and the offset cost `|Δq|`. Also rebuilds signals for runs recorded before those signals were saved. | no |
| `plots.py` | Every figure. | no |
| `replot.py` | Redo the figures + scores of a finished run from its `controller_data.npz`. Seconds, no sim. Use it after any plotting/metric change. | no |

---

## 4. Time rules (most debugging pain lives here)

Five rules. If a signal looks shifted by ~1 frame or ~2 steps, suspect one of these before
suspecting physics.

1. **Index `k` = the state at the END of frame `k`** (after its 12 physics steps).
2. **`joint_target[k]` = the command held DURING frame `k`.** Historical quirk: in the original
   replay the first 2 steps of each frame still held the *previous* command (the recorder wrote
   the new command late). The `--exact-replay` preset reproduces this exactly with
   `interp=hold + latency_steps=2`; the normal controller uses `linear + latency 0`.
3. **The policy is asked at the START of frame `k`** and predicts the ends of frames
   `k .. k+15`. So prediction number `j` "lives at" time `k + j + 1` (in frame units).
4. **The command computed at (frame, step) targets the end of that step**:
   `t = frame + (step + 1 − latency) / 12`.
5. **Sensor readings are one physics step old** (read before the step, so you see the previous
   step's result). The offline checker reproduces this delay with its `g − 1` indexing.

---

## 5. The methods, in plain words

### 5.1 ChunkTracker — from 10 Hz predictions to a smooth 120 Hz target

- A **chunk** = 16 predicted frames (`horizon`), arriving every 8 frames (`chunk`). The extra 8
  are spare — if a real policy is late one day, the old chunk still has predictions to run on.
- Between predictions the tracker **interpolates** (`linear`, the default) or **holds** the next
  prediction (`hold`, staircase style — only used to reproduce the replay exactly).
- **The anchor trick**: when a new chunk arrives, its first segment starts from whatever value
  was being tracked at that instant — not from the chunk's own first prediction. So the target is
  continuous across chunk switches. The offline check prints the biggest per-step target jump; it
  should be small (~0.018 rad on the dev episodes).
- **Force size is derived from the interpolated vector** (`|lerp(vector)|`), never interpolated
  separately. (Interpolating sizes and vectors separately makes them disagree between
  predictions, which once gave the controller and the metrics two different ideas of "pressing".)
- **Touch spots interpolate NaN-aware**: if one end of a segment is "not touching" (NaN), use the
  touching end instead of averaging with NaN.

### 5.2 Filtering (EMA)

Contact forces at 120 Hz are spiky. Both tactile readings are smoothed with an **EMA**
(exponential moving average):

```
smoothed = 0.4 · new_reading + 0.6 · old_smoothed        (alpha = 0.4)
```

Bigger alpha = less smoothing, less delay. Alpha 1.0 = no filter.

Centroid special rule: only blend when the old and new spot **both exist**. On a fresh touch (or
a loss of touch) restart from the raw value — otherwise you drag a ghost point across the gap.

**Before diving in: `Δq` has two parts.** The offset the law adds is the *sum* of two
contributions, computed together every step and clipped together:

```
Δq  =  Δq_force (§5.3)  +  Δq_point (§5.4)
```

`Δq_force` makes the press the right **strength and direction**; `Δq_point` slides the
fingertip **along the surface** to the right spot. They stay out of each other's way by the
direction split in §5.4 step 2: the along-force direction (pressing depth) belongs to the force
channel, the sideways directions belong to the point channel. With `point_kp = 0` the second
part is zero and `Δq` is the force channel alone. The next two sections cover one part each.

### 5.3 The force channel (the core)

**The spring math.** To push on the world with force `f` at the touch point, the joints need
extra torque. The Jacobian `J` converts between the two worlds: joint speed → fingertip speed,
and (flipped, `Jᵀ`) fingertip force → joint torque. The PD spring gives torque
`K · Δq` (stiffness times extra angle). Put together:

```
Δq = −K⁻¹ Jᵀ f      "the extra angle whose spring torque balances the push"
```

**What `K` is:** the PD controller's stiffness — the spring constant of each joint's motor, in
N·m per rad (a control *setting*, not a property of the metal): arm joints 400, LEAP joints 350
(mirroring the original Isaac Gym gains). Each joint is its own independent spring, so `K` is
diagonal and `K⁻¹` just means "divide by each joint's stiffness". The damping `D` (arm 40,
hand 12) never appears because this is a standing-still balance: `q̇ ≈ 0`, so only the spring
term carries the load. The sim runner reads the **live** values from the articulation rather
than trusting the config constants; offline, `config.default_joint_stiffness()` supplies them.
Feel for the size: 10 N at a ~2 cm lever arm → 0.2 N·m → `0.2/350 ≈ 0.6 mrad` of aim-past.

**Where the equation comes from** — two facts set equal, standing still:

1. `τ_needed = Jᵀ·f` — the torque the contact force exerts on each joint. This is the *regular*
   kinematic Jacobian, transposed; no special "force Jacobian" exists. Why the same matrix:
   energy bookkeeping (the "virtual work" principle). For any tiny imaginary joint motion `δq`
   the point moves `J·δq`, so the force does work `(Jᵀf)·δq`, and the joint torques do `τ·δq`;
   balance for *every* possible `δq` forces `τ = Jᵀf`. Plain words: a force pushes back on a
   joint exactly as strongly as that joint could move the point along the force.
2. `τ_motor = K·Δq` — what the spring produces when its target sits `Δq` past the joint.

Net torque zero at rest: `K·Δq + Jᵀ·f = 0` → `Δq = −K⁻¹Jᵀf`. (Minus sign = convention: `f` is
the force ON the fingertip; the finger pushes with `−f`.) One-joint check: lever arm 2 cm,
press down 10 N → table pushes up 10 N → 0.2 N·m uncurl torque → aim 0.6 mrad *deeper*. ✓
This is textbook **stiffness control** (Salisbury 1980); the composed map `−K⁻¹Jᵀ` is the
"force → position offset" converter, built entirely from the regular `J`. Hidden assumptions:
standing still, small `Δq`, pure force at a point, and a **rigid world** — the last one is the
big one; see the honest warning at the end of this section.

**What `f` to plug in.** Not just the predicted force — also a correction from the error:

```
f_cmd = kff·f_ref  +  engaged · ( kp·e  +  I  +  kd·ė )        e = f_ref − f_meas
```

- `kff·f_ref` (kff = 1): the **feedforward** — press what the policy asks for, instantly.
- `kp·e` (0.3): push a bit harder where the measurement falls short.
- `I`: the **integral** — a memory that keeps adding the leftover error (rate `ki` = 10 /s), so a
  steady shortfall slowly gets pushed away. Clamped at 300 N.
- `kd·ė` (0 = off): reacts to fast error changes. Off because contact is spiky.

In control-theory words: the bracket is a textbook **PID** on the force error, and the whole
line is the classic **feedforward + PID** pattern (with `kd = 0` it runs as feedforward + PI).
Non-textbook parts: its output is a force that still goes through `−K⁻¹Jᵀ` rather than the
actuator directly, and the engagement gate is *conditional integration* driven by the
prediction ("should I even be pressing" is the policy's call, not the error's).

Everything is a full 3-D **vector** — four independent controllers, one per fingertip. The
deadband and clip act on the error's *length* (they scale the vector, never bend it), so no
direction is assumed anywhere: pushing, pulling and squeezing all come out of the predicted
vector and the geometry at the touch point.

**The engagement gate.** The correction terms only act while the *prediction* says "you should be
pressing" (`|f_ref| ≥ 0.2 N`). While not engaged, the integral leaks away (time constant
0.15 s). While engaged but not yet touching, the error equals the whole `f_ref` — so the law
deliberately drives the finger toward contact. That is a feature, not a bug.

**Safety rails**: error deadband 0.1 N (ignore dust), error clip 30 N, integral clamp 300 N,
final per-joint offset clip 0.12 rad, arm joints excluded by default — and the **friction-cone
cap** (added 2026-08-31, `cone_half_angle_deg`, `--cone-deg`, ≤0 disables).

**The friction cone, in plain words.** A contact can push back without limit *into* the press,
but only `mu ×` the pressing force *sideways* (friction's limit). Turn that into a rule about
the force arrow's angle and all the "holdable" directions form a cone on the contact point,
half-angle `atan(mu)` — for this jar's mu 0.35, **19.3°**:

```
        \    |    /     arrows INSIDE the cone: contact holds them (no sliding)
         \   |   /      arrows OUTSIDE: too much sideways -> the contact SLIDES
          \  |  /
   ────────(●)────────   half-angle = atan(mu)
```

Ask the fingertip for a force outside the cone and physics answers with sliding — which is
exactly how the saturated integral (which had rotated the thumb's command 29–35° off) ground
the jar out of the hand on `01-27-32`.

**The cap.** The controller cannot know the surface normal or `mu` (tactile-only rule) — but it
does not need to: the *predicted* force comes from a real recorded contact, and any force that
really existed at a contact was inside the cone. So the prediction is a known-safe axis for
free. `f_cmd` is split into an along-prediction part ("press harder/softer" — never touched,
that is the PID's job) and a sideways part ("rotate" — shrunk until the angle is ≤ 20°), and
the along-part may never go negative (a fingertip pushes, it cannot pull). No direction is
hard-coded: the safe axis is the prediction itself. Measured effect: 3/3 repeat runs hold the
jar (was 2/3), thumb angle 17–24° (was 28–35°) — see §8.5 problem 4. 20° is a global default,
not the object's true `atan(mu)`; the prediction's own in-cone margin covers the difference.

**The rigid-world problem and the adaptive feedforward (treated 2026-08-31).** The spring math
assumes the world is rigid; real contact squishes and objects move, so one newton commanded
returns well under one newton measured. A fixed `kff` therefore left the integral to supply the
missing scale — saturated at 300 N, direction geometry-locked (the slip story of §8.5). The cure:
estimate the plant's delivery ratio online, per tip — `alpha = EMA of (measured force along the
predicted direction) / (commanded along it last step)`, time constant `adapt_tau_s` = 0.3 s —
and multiply the **feedforward only** by `1/alpha` (clamped to [0.5, `ff_gain_max` = 6], rising
at most `ff_gain_slew` = 3/s; falling freely). The PID gains are untouched, so loop stability is
unchanged; the feedforward simply supplies the missing scale *in the safe direction*. Adaptation
starts from first light touch (0.5 N) on purpose: waiting for firm contact lets the integral wind
up first, and gain-on-top-of-full-integral over-presses (measured: 4/4 crashes with a 2 N gate,
clean with 0.5 N). Measured effect: integral-at-clamp 66–72 % → 36–40 % of engaged steps, thumb
bias on `17-52-54` −17…−27 N → **+0.6 N**, thumb angle on `17-55-22` → 4–7°. Remaining
weakness: the estimator sees the *closed-loop* ratio (integral included), so a finger whose
integral already covers the gap reads `alpha ≈ 1` and keeps its integral — full desaturation
needs the estimator refinement or the stacked solve (§8.5).

**The force channel's pipeline, with the 2026-08-31 layers marked:**

```
f_ref ──► [ × adaptive gain ]  ─────┐        layer 1: the CURE (fixes the size deficit
          §5.3 adaptive ff          ▼                  that made the integral saturate)
e = f_ref − f_meas ─► PID ──►  sum = f_cmd ──► [ cone cap ] ──► −K⁻¹Jᵀ ──► Δq_force
                                               §5.3 rails      layer 2: the FENCE (no term
                                                               may rotate the command into
                                                               the slipping region)
(layer 3 — the contact sensor's readback buffer, 32 → 64 — is not in this pipeline at all;
 it is reporting capacity inside the sensor that produces f_meas, proven physics-neutral by
 a bit-exact --exact-replay after the change.)
```

**The three-layer design, as a design audit.** The three 2026-08-31 changes live at three
different levels on purpose — cause, consequence, infrastructure — so each can be judged,
switched off, and re-measured on its own:

| | layer 1: adaptive ff | layer 2: cone cap | layer 3: buffer 64 |
|---|---|---|---|
| level | the **cause** (size deficit that saturated the integral) | the **consequence** (direction leaving the friction cone) | **infrastructure** (sensor reporting capacity) |
| what it exactly is | per-tip `alpha` = EMA(measured/commanded along the prediction); feedforward × `1/alpha`, clamped [0.5, 6], rise ≤ 3/s | after the sum: along-part ≥ 0 (no pulls), sideways part shrunk until the angle ≤ 20° | the ContactSensor's per-pad readback list, 32 → 64 entries |
| what it deliberately does NOT do | never scales the PID (loop stability untouched); never touches direction | never fixes any term — only bounds the damage any term can do; magnitude passes through | nothing to physics — it changes how many already-resolved contact points get *reported* |
| off-switch | `--no-adapt-kff` | `--cone-deg 0` | the constant in `zerofact/replay.py` |
| addressed a *measured* problem | integral pinned 66–72 %, thumb −27 N | thumb 29–35° vs the 19.3° cone, dropped jar | 1–2 of 4 runs crashed mid-episode |
| validation | 4/4 jar held, `17-52-54` thumb → +0.6 N, clamp time → 36–40 % | 3/3 held pre-adaptive, thumb angle 17–24° | 0 overflows after; **exact-replay bit-exact** after the change (the physics-neutrality proof) |
| honest weakness | estimator sees the closed-loop ratio → partial desaturation only | global 20°, and the cone axis is the *predicted* normal, not the actual one | none (capacity) |

Shared philosophy: the only direction knowledge either control layer uses is the policy's own
prediction — no new assumption entered the controller. Details: adaptive ff and cone above in
this section; measured history in §8.5 problem 1.

### 5.4 The contact-point channel

The force channel controls how hard you press. It cannot see *where on the surface* the finger
sits: the press can be a perfect 10 N while the fingertip stands 5 mm beside the spot the policy
wanted. This channel closes that gap. Its input is the contact-point error

```
e_p = p_ref − p_meas        (predicted touch spot − measured pad centroid)
```

and its full life looks like this:

```
p_ref (interpolated prediction)        p_meas (EMA-filtered pad centroid)
        └───────────────┬──────────────────────┘
                        ▼
              e_p = p_ref − p_meas            (TaskSpaceForceLaw._point_offset)
                        │
        ┌───────────────┼────────────────────────────┐
        ▼ control       ▼ recording                  ▼
    Δq_point      point_err_steps (npz)     contact_point_error.png,
        │                                   summary.contact_point_tracking
        ▼
    Δq_total = Δq_force + Δq_point  →  per-joint clip  →  action = q_ref + Δq_total
```

Step by step:

1. **Existence check** — either spot NaN (pad not touching / policy says no contact) or
   `point_kp = 0` → contribute nothing (the error is still *recorded* for the metric).
2. **Split off the depth part.** Project out the component along the predicted force direction
   `n̂ = f_ref/|f_ref|`:  `e_p⊥ = (I − n̂n̂ᵀ)·e_p`. Along-force = pressing depth = the force
   channel's territory; correcting it here would fight the PID. Only the **sideways** slide
   survives. The split direction comes from the *prediction*, so no direction is assumed.
3. **Clip** `|e_p⊥|` at 2 cm (`point_clip_m`) — one bad centroid must not yank the hand.
4. **Turn the slide into joint motion** (the pseudo-inverse — explained below):
   `Δq_tip = point_kp · Jᵀ(JJᵀ + λ²I)⁻¹ · e_p⊥`, clipped per tip at 0.05 rad, only while engaged.
5. **Add** to the force channel's offset; the sum goes through the usual clip into the action.

**How Δq_point comes out of e_p, slowly.** The question is the *reverse* of §5.5's: we know the
fingertip motion we want, `d = point_kp·e_p⊥` (3 numbers), and ask which joint motion produces
it. Kinematics says `J·Δq = d` — 3 equations, but a finger chain has 4 joints (more if the arm
joins), so **many joint motions give the same fingertip motion** (curl more at the knuckle and
less at the middle joint, same tip motion). Which to pick?

- **Pick the smallest.** Among all solutions, take the one with the least total joint motion:
  `Δq = Jᵀ(JJᵀ)⁻¹·d`, the *pseudo-inverse*. Reading it right-to-left: `JJᵀ` is a tiny 3×3
  "mobility" matrix of the fingertip (big entries where the joints can move the tip easily), so
  `(JJᵀ)⁻¹d` answers "how much pull per direction does this slide need, given the tip's
  mobility", and `Jᵀ` then spreads that pull onto the joints by their lever arms. Joints whose motion doesn't move the tip get exactly 0 —
  no wasted motion.
- **Add damping.** With the finger nearly straight, some tip directions become almost
  unreachable (a *singularity* — try shortening a fully extended finger by curling: the first
  millimetre needs a huge joint swing): `(JJᵀ)⁻¹` would demand enormous joint motion for them. Damped
  least squares softens the request: minimize `‖J·Δq − d‖² + λ²‖Δq‖²` — "get the motion as
  close as you can, but big joint motion costs too". The answer is the used formula,
  `Δq = Jᵀ(JJᵀ + λ²I)⁻¹·d` (λ = 0.02 m/rad, floored at 1e-6). Far from a singularity it is
  ~the exact pseudo-inverse; near one it gracefully under-delivers instead of exploding.
- **Mask before solving.** The arm columns of `J` are zeroed *before* the solve (default
  hand-only), so the least-squares redistributes the slide onto the allowed joints — instead of
  computing arm motion that the clip would silently delete afterwards, under-delivering the slide.

Note the contrast with the force channel — same Jacobian, two different directions of use:

| channel | wants | map | why that map |
|---|---|---|---|
| force | a **force** at the tip | `−K⁻¹Jᵀ` | Jᵀ converts force → joint torque (lever arms), K⁻¹ torque → angle |
| contact point | a **motion** of the tip | `point_kp · Jᵀ(JJᵀ+λ²I)⁻¹` | inverse kinematics: smallest joint motion producing the tip motion |

Two honest notes. This is a pure P controller in position space (offset ∝ current error,
recomputed every step, nothing integrates), so friction and the 0.05 rad clip can leave a
steady residual — seen on `01-27-32`, where the offset saturates without closing the error.
If that ever needs fixing, the fix is an integral term on `e_p⊥` or a bigger per-tip clip — a
deliberate tuning decision, not a bug.
And the measured centroid is the run-to-run **noise amplifier** (mm-scale jumps between steps,
§8.3/§8.5 problem 5) — filter or rate-limit `p_meas` before tuning `point_kp` hard.

### 5.5 Where the Jacobian comes from

The Jacobian `J` of a point is a table with one **column per joint** (23 here) and one row per
direction (x, y, z). Column `j` answers: "if joint `j` alone turned at 1 rad/s, how fast and in
which direction would this point move?" For a hinge joint that is pure geometry: the point sweeps
a circle around the joint axis, so its velocity is `axis × lever‑arm`. Joints not between the
base and this fingertip give zero columns.

Flipped (`Jᵀ`), the same table converts a force at the point into joint torques: a force pushes
back on a joint exactly as strongly as that joint could move the point *along the force* — same
lever arms, same numbers. The force channel uses the flipped direction (`τ = Jᵀf`); the
contact-point channel uses the forward direction through the damped inverse (§5.4). One matrix,
both jobs.

The code (`_tip_point_jacobians`, `sim_runner.py:40`) builds it in three steps, every physics
step (the hand moves, so `J` changes):

1. **Ask PhysX.** `robot.root_physx_view.get_jacobians()` returns, for every body, a 6×23 matrix
   at the current pose — rows 0–2 linear, rows 3–5 angular — built analytically by the engine
   from the same joint axes and link poses it simulates with. Exact, always in sync, no numerical
   differencing. Quirk: for a bolted-down robot PhysX omits the unmovable root body, so every
   body index shifts by one — the code detects this by array size (the "base_offset probe").
2. **Shift it to the touch point.** PhysX reports the matrix at the fingertip's *origin* (back at
   the last joint); the force acts at the *pad*, a lever arm away, and torque depends on lever
   arms. Rigid-body rule: a point riding on a body moves like the origin plus a rotation term,
   `v_p = v_origin + ω × r`. In matrix form: `J_p = J_v − skew(r)·J_w` (`skew(r)` = the cross
   product written as a 3×3 matrix). The fingertip's lever arm is only ~1–2 cm — small, but
   skipping the shift would be a systematic bias.
3. **Choose the anchor point** `p`, in order: the **measured** pad centroid (touching) → the
   **predicted** touch spot (not touching yet, so the press aims at the right place before
   contact; disable with `--no-predict-point`) → the fingertip origin (nothing known).

Only the 3 linear rows are kept (we command point forces; there is no torque channel — matching
the sensor, which reports no torque either). Result: `[4 tips, 3, 23]`.

Fair-play note: this is robot model + joint angles, not simulator magic — on hardware the same
matrix comes from the URDF + encoders with any kinematics library. Offline there is no PhysX and
therefore no `J`, which is why the task-space law outputs zero offsets in `offline_check.py`.

DIY sanity check: in free space, nudge one joint's target by a tiny δ, let the PD settle, and
compare the fingertip point's motion with `δ · J[:, j]`. If they disagree, suspect the
base-offset probe or a joint-ordering mismatch first.

### 5.6 The pad centroid — what "the tactile sensor" reports here

A real tactile pad reports **one pressure spot** and cannot tell what it is touching. The
simulator knows per-object contact patches, so `pad_contact_centroid` fuses them into one point
per pad, **weighted by force**. Two special rules:

- Patches that exist but read exactly 0 N still count as touch (the normal force crosses zero all
  the time at 120 Hz) — with no force anywhere, use the plain average of the patches. Dropping
  zero-force patches once caused a real bug: the controller lost the touch location during light
  contact, under-closed, wound up its integral, and crashed the sim.
- No patch at all → NaN ("not touching").

Caveat: the **table** is a plain collider, not a filtered object, so touching only the table
gives force with a NaN centroid. The law then falls back to the predicted spot.

### 5.7 How state tracking and force tracking share one action

The middle layer must track two things — the predicted state and the predicted force — but it has
only one output. It does **not** track one "first" in time. Every step builds one action:

```
action = q_ref + Δq        (state = the base, force = a bounded correction on top)
```

Two facts that surprise people:

- **State tracking has no feedback loop here.** The middle layer never computes
  `q_ref − q_measured`. It sends the target to the PD and trusts the springs to regulate the
  joints. The only feedback the middle layer closes is on force (and the touch spot).
- **Under contact, the two goals are incompatible by physics.** A spring sitting exactly at the
  predicted state pushes with ~nothing; producing force *requires* deviating from the state
  (`Δq = −K⁻¹Jᵀf` — the deviation IS the force). So the real question is "what is the *minimum*
  state deviation that buys the commanded force". The recorded ideal command answers it: ~8.6 mrad
  (hand mean) on the dev episodes.

There is no explicit weight between the two objectives. The balance is settled by structure:

| mechanism | what it decides |
|---|---|
| engagement gate | free space → `Δq → 0` → pure state tracking; force feedback exists only where the prediction says "press" |
| additive form | force can only *perturb* the state target, never replace it |
| offset clip (0.12 rad) | the hard budget: at most 0.12 rad of state accuracy per joint may be spent on force, ever |
| arm mask | arm joints' budget is zero (default) — state tracking is absolute there |
| tangential projection | direction split: along `f_ref` = force's territory (depth), sideways = position's territory (spot on the surface) |

Who wins in a conflict: free space — the state, completely. Under commanded contact — the force,
up to its budget (the integral keeps converting state accuracy into force until the force is met
or the clip stops it).

The price actually paid is a recorded number: `command_fidelity.offset_mean_mrad` = `|Δq|`.
Today the law pays ~14–15 mrad where the ideal pays ~8.6 — it overpays, in the wrong joints
(§8.5 problem 1). Improving the balance is therefore not a re-weighting question; it is the
feedforward-scale fix. The two corners for intuition: the `null` law = 100 % state tracking
(costs 74.5 N of force error on the 95 N grasp); `--exact-replay` = the perfectly balanced ideal.

### 5.8 Taking the controller to a real robot — what changes

Almost nothing in the controller itself: `middle_layer.py` is pure numpy with no Isaac imports
on purpose. Only `sim_runner.py` gets replaced by a "hardware runner" that fills the same
`Measured` struct from real sources:

| field | sim source | hardware source |
|---|---|---|
| `q`, `qd` | articulation state | joint encoders |
| `f_net` | ContactSensor | tactile pad, rotated into the base frame via FK |
| `tip_contact_pos` | fused PhysX centroid | the pad's centroid, placed via FK |
| `tip_jacobians` | PhysX (§5.5) + shift | kinematics library (Pinocchio/KDL/…) at the measured `q`, then the **same** shift |
| stiffness `K` | read from the articulation | identified on the robot — see gap 1 |

The Jacobian is NOT taken from simulation — it never was. It is a function of
(robot model, current joint angles), recomputed every step; the sim merely evaluates that
function, and on hardware a kinematics library evaluates the identical one from the URDF +
encoders.

The three real sim2real gaps, ranked:

1. **Effective stiffness, not the Jacobian.** A real servo (Dynamixel + gears + friction +
   backlash) does not deliver the nameplate torque-per-radian. We already see the same disease in
   sim (effective gain ~0.16× the model, from contact squish); hardware just changes the wrong
   number. The PID absorbs it, and the planned per-tip adaptive `kff` scale (§8.5 problem 1) is
   the proper fix for both worlds at once — it estimates achieved/requested force online and does
   not care why the model is off.
2. **Kinematic calibration.** URDF link lengths / joint zeros must match the metal; errors bias
   the lever arms. Lands as a force bias the integral eats, not as instability.
3. **Touch-point frame.** The pad reports its centroid in the sensor frame; FK places it. The
   quantity that matters is the local lever arm (~1–2 cm), so fingertip-frame accuracy is enough.

---

## 6. What a rollout writes

Run: `python force_controller/run_tracking.py --episode outputs/<run>` (conda env
`env_isaaclab`). Output folder: `outputs/force_controller/<run>/<stamp>/`. Videos are **on by
default**; `--no-render` = physics only (seconds, ~3.3 GB — use it for sweeps).

**Which file answers which question:**

| file | the question it answers |
|---|---|
| `rollout.mp4` | what did the robot physically do? (force arrows at the touch points) |
| `rollout_with_forces.mp4` | same, side-by-side with the sweeping force plot + live readout |
| `contact_forces.mp4` | the force plot alone, as a video |
| `force_tracking.png` | is the **size** of the force right, per finger, over time? |
| `force_error_components.png` | **where** is the force wrong — signed error per world axis X/Y/Z, with bias + rms printed in each panel |
| `force_vector_error.png` | is the error wrong **size** or wrong **direction**? (parallel vs perpendicular split + angle) |
| `force_components.png` | the two raw signals (predicted vs measured) per axis |
| `contact_point_error.png` | is the finger touching the **right spot**? (sideways vs depth split; red shading = told to touch but not touching) |
| `commands_vs_predicted.png` | did the law recover the command lead? black = recorded ideal command, blue dashed = predicted state, red = our command. Null law: red sits on blue. Perfect law: red sits on black. |
| `joint_tracking.png`, `object_tracking.png`, `contact_forces.png` | physics-side sanity views |
| `controller_data.npz` | every signal, per physics step (see below) |
| `summary.json` | all scores + full config + outcome |

**Controller signals in `controller_data.npz`** (T frames, 12 steps, 4 fingertips, J joints):

| key | shape | meaning |
|---|---|---|
| `action_steps` | [T,12,J] | the command actually sent |
| `q_ref_steps` | [T,12,J] | interpolated predicted state |
| `dq_steps` | [T,12,J] | the offset (after clipping) |
| `f_ref_steps` / `f_meas_steps` | [T,12,4] | predicted / measured force size |
| `f_ref_vec_steps` / `f_meas_vec_steps` | [T,12,4,3] | the vectors (world frame) |
| `p_ref_steps` / `p_meas_steps` | [T,12,4,3] | predicted / measured touch spot (NaN = none) |
| `point_err_steps` | [T,12,4] | sideways touch-spot error (m, NaN = no pair) |
| `u_steps` | [T,12,4] | integral size per tip (N) — **watch this for saturation at 300** |
| `ff_gain_steps` | [T,12,4] | adaptive feedforward gain per tip (1.0 = model scale) |

Plus the same physics recording a replay makes (`joint_pos`, `object_pos`, `contact_force_steps`,
…), so a rollout can be compared with a replay key by key.

---

## 7. The scoreboard (`summary.json`)

| block | what it says | read it as |
|---|---|---|
| `force_tracking` | size-only error while pressing | `bias_engaged_N` negative = pressing too weakly |
| `force_tracking_vector` | the honest force score | `rmse_vec_N` = total; `bias_mag_N` vs `angle_mean_deg` = wrong size vs wrong direction; `bias_xyz_N` = signed bias per world axis |
| `contact_point_tracking` | touch-spot error in mm | `tangential` = this channel's job; `normal` = depth (force channel's job); `measured_fraction` = of the time we were told to touch, how often we actually touched |
| `command_fidelity` | the offset cost | see below — this is the tuning score |
| `grasp_slip` | unintended in-hand motion | the object's position in the **palm frame**, compared against the episode's own — commanded pushing/pouring/carrying cancels out, slip and drops remain. `dropped: true` = the object left the hand while a strong grasp was still commanded. Exact-replay scores 0.0 by construction |
| `outcome` | did the task work | lift vs the episode's lift. **Weak separator**: the no-feedback baseline still lifts ~97% of ground truth, so lift alone cannot rank controllers |

**`command_fidelity`, carefully — two scores in one block:**

- `offset_mean_mrad` (+ max/rms) = **`|Δq|`** — how far the controller departed from the
  predicted state to satisfy the force target. Uses nothing a real robot lacks
  (**deployable**), and it is the *steadiest* number run-to-run. **Tune on this + the force
  error.** (1 mrad = 0.001 rad ≈ 0.06°.)
- `hand_mean_mrad` + `per_joint` = distance to the **recorded ideal command**. Ground truth only
  exists in development — never a control signal, never a tuning target. Its value is the
  `per_joint` split: `needed_lead` (what the offset *should* have been) vs `produced_dq` (what
  the law made) shows **which joints** the offset went to. That is how we found the offset going
  to sideways/abduction joints instead of the curl/flexion joints.

The two rank controllers almost identically (rank correlation 0.98 over 26 runs), which is what
makes tuning on `|Δq|` safe.

---

## 8. Debugging playbook

### 8.1 Always start with the ladder

```bash
# 1. no sim, ~2 s: timing plumbing must rebuild the recorded commands bit-exactly
python force_controller/offline_check.py --episode outputs/run_2026-05-15_17-55-22

# 2. in sim: the whole runner must bit-reproduce the replay
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
    --exact-replay --no-render

# 3. baseline: no force feedback
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22 \
    --force-law null --no-render

# 4. the controller (videos on by default)
python force_controller/run_tracking.py --episode outputs/run_2026-05-15_17-55-22
```

Step 2's diff: every **state** key must be exactly 0.0. The per-step force-report keys may show
isolated ≤ 1-ulp entries (GPU readback rounding dust — see glossary). If step 1 or 2 fails:
**framework bug — stop and fix that first.** Never tune on a broken ladder.

### 8.2 Know your noise (important!)

- The open-loop replay is bit-deterministic. **The closed loop is not**: the GPU's force readback
  jitters by a few ulp, the controller reacts to it, and the difference grows.
- Measured on 6 identical default runs of `17-55-22`: force bias spreads **5.06 N**
  (with `--point-kp 0`: 0.03 N — the point channel is the amplifier). Lift spreads ~4 mm.
  `|Δq|` barely moves (sd 0.61 mrad on a mean of 14.4).
- Rules: **repeat 3× before believing a force-bias change**; treat differences under ~2 N as
  noise (with the point channel on); rank configurations by `|Δq|` + force error together.

### 8.3 Symptom → where to look

| symptom | first place to look |
|---|---|
| target jumps at chunk switches | offline check's "action continuity" number; anchor logic §5.1; timing rules §4 |
| everything shifted ~2 steps / 1 frame | `interp` / `latency_steps`; rules 2–5 in §4 |
| force size right, direction wrong | `force_error_components.png` (which axis?) + angle in `force_vector_error.png`; then centroid quality / contact geometry |
| one finger stuck below target, `u_steps` pegged at 300 | **not** a `ki` problem — the feedforward under-delivers and the integral saturated; §8.5 problem 1 |
| offset goes to strange joints (sideways instead of curl) | `command_fidelity.per_joint` in the summary |
| big touch-spot error on `01-27-32` | partly the **object's** drift, not the finger's: the stand-in policy is open-loop (ignores observations), so on episodes where the rollout diverges (~41 mm there) `p_ref` points at where the object *was*. Tune on `17-55-22` — it stays close to its recording |
| object slides / drops out of the hand near the end | `grasp_slip` in the summary (or `replot.py` to backfill it): `dropped` + `drop_frame` say when; then check the thumb's `angle_mean_deg` against the friction cone `atan(mu)` (mu 0.35 → 19.3°) and `u_steps` saturation. Root cause found 2026-08-31: force *direction* outside the cone grinds the contact; the cone cap (§5.3) is the fix, and the drop was the bad draw of the run-to-run spread |
| jittery `dq` with the point channel on | the centroid jumps 1–3.5 mm between steps (>1 mm on ~13% of steps); lower `point_kp` or strengthen `point_ema_alpha` |
| crash mid-run: "Incomplete contact data" warning, then CUDA assert | the grasp resolved more contact points than the sensor's readback buffer holds. The buffer is 64 since 2026-08-31 (was 32; raised for the adaptive feedforward's firmer grasps, with `--exact-replay` re-verified bit-exact — readback capacity, not physics). If it still fires, something is over-pressing: check `ff_gain_steps` and `u_steps` first |
| plots or scores look wrong after editing `plots.py`/`metrics.py` | `replot.py <run folder>` — rebuilds everything from the npz in seconds, no sim |
| Isaac won't start / OOM / no window | root `README.md`: DISPLAY/Vulkan guard, `free -g` first (shared box), **one** Isaac process at a time |

### 8.4 Cheap tools

- `python force_controller/replot.py outputs/force_controller/<run>/<stamp> ...` — re-score and
  re-plot finished runs; also backfills runs recorded before a metric existed.
- **Diff two runs**: load both `controller_data.npz`, `np.abs(a["dq_steps"] - b["dq_steps"])`,
  find the **first differing frame**. This locates where behaviour split — it found the centroid
  bug in minutes (the offsets diverged one frame *before* the reference did, which pointed away
  from the reference and at the Jacobian anchor).
- `offline_check.py` with your flags — previews the reference and targets with no sim. (Note: the
  task-space law outputs zero offsets offline — it needs Jacobians only the sim provides — but
  all timing/target checks still run.)
- `--max-frames 40 --no-render` — a ~3 s smoke test.
- A full rendered run ≈ 1 min wall clock (Isaac startup ~40 s + ~14 s loop). Use
  `/home/jianuoqiu/anaconda3/envs/env_isaaclab/bin/python` directly.

### 8.5 Known open problems (so you don't rediscover them)

1. **The integral saturation chain — largely treated 2026-08-31.** The chain (spring math
   under-delivers → integral supplies ~2.8× the target, pinned at its clamp 66–72 % of steps,
   direction geometry-locked → offsets in the wrong joints, and — via the cone story — grasps
   ground sideways) is now attacked from two sides: the **cone cap** fences the direction damage
   and the **adaptive feedforward** (§5.3) removes most of the magnitude burden. Measured:
   integral-at-clamp 36–40 %, `17-52-54` thumb −17…−27 N → +0.6 N, `17-55-22` thumb angle
   4–7°, jar episode 4/4 held with lifts 305–327 mm (episode 319.5). Still open, in order:
   (a) the estimator sees the closed-loop ratio, so fingers whose integral already covers the
   gap stay partially saturated; (b) per-tip coupling — on `17-52-54` the middle finger reads
   **+33 N** it did not command (the neighbours' squeeze reacts through the object; its own
   command is already at zero and a per-tip law cannot press *less* than zero) — the stacked
   multi-contact solve is the real cure for both.
2. **Per-tip superposition.** Each fingertip's offset is computed alone; the arm joints are
   shared by all fingers, so arm participation is disabled by default (`allow_arm_offset`).
   Right fix: one stacked least-squares over all tips at once.
3. **No anti-windup against the output clip.** The integral is clamped at 300 N and the offset
   separately at 0.12 rad, but nothing tells the integral "your output is being clipped" — it
   keeps winding.
4. **Physical-cone check — largely fixed 2026-08-31.** `f_cmd` is now capped to ≤ 20° from the
   predicted force direction and can never command a pull (§5.3). Measured effect on `01-27-32`
   (the episode that dropped the jar): 3/3 runs hold vs 2/3 before, unintended in-hand motion
   30 mm vs 36–90, thumb direction error 17–24° vs 28–35°, and better force bias — with no
   regression on `17-55-22` and the exact-replay ladder still bit-exact. Still open: 20° is a
   single global number, not the object's actual `atan(mu)`; and the thumb still sits *at* the
   cone edge — the deeper cure remains problem 1.
5. **Point-channel noise** (§8.3): the centroid is discontinuous step to step; rate-limit or
   filter it before tuning `point_kp` seriously.
6. **Touch-spot floor ~1.2 mm**: the episode's contact geometry is recorded at 10 Hz, so the
   interpolated `p_ref` lags the 120 Hz truth by ~1 mm even when everything is perfect.

---

## 9. Defaults at a glance

| knob | default | plain meaning | change via |
|---|---|---|---|
| `horizon` / `chunk` | 16 / 8 | predictions per chunk / frames between chunks | `--horizon` `--chunk` |
| `state_source` | `joint_pos` | policy predicts where to *be* (real BC semantics). `joint_target` = serve recorded commands (only for exact-replay) | `--state-source` |
| `interp` / `latency_steps` | `linear` / 0 | smooth targets, no artificial delay | `--interp` `--latency-steps` |
| `law` | `task_space` | the two-channel law; `null` = playback baseline | `--force-law` |
| `kff` | 1.0 | feedforward on the predicted force | `--kff` |
| `task_kp` / `task_ki` / `task_kd` | 0.3 / 10 / 0 | PID on the force error vector | `--task-kp` etc. |
| `task_i_max_n` | 300 N | integral clamp | config.py only |
| `cone_half_angle_deg` | 20° | cap on f_cmd's angle from the predicted force; no pulls (§5.3) | `--cone-deg` |
| `adapt_kff` | on | adaptive feedforward gain (the integral-saturation cure, §5.3) | `--no-adapt-kff` |
| `adapt_tau_s` / `ff_gain_max` / `ff_gain_slew` | 0.3 s / 6 / 3 s⁻¹ | estimator speed / gain clamp / max upward gain rate | `--adapt-tau` / `--ff-gain-max` / config.py |
| `allow_arm_offset` | off | offset stays in the hand | `--allow-arm-offset` |
| `point_kp` | 0.5 | sideways touch-spot correction (0 = off) | `--point-kp` |
| `point_clip_m` / `point_offset_clip_rad` | 2 cm / 0.05 rad | its rails | `--point-clip-m` / config.py |
| `point_damping` | 0.02 m/rad | pseudo-inverse damping (singularity safety, §5.4) | config.py only |
| `predict_point_before_contact` | on | aim the press at the predicted spot before touching | `--no-predict-point` |
| `engage_threshold_n` | 0.2 N | "you should be pressing" gate | config.py only |
| `deadband_n` / `error_clip_n` | 0.1 / 30 N | ignore dust / cap spikes | config.py only |
| `release_tau_s` | 0.15 s | integral leak when not engaged | config.py only |
| `meas_ema_alpha` / `point_ema_alpha` | 0.4 / 0.4 | smoothing (1.0 = none) | config.py only |
| `offset_clip_rad` | 0.12 rad | final per-joint safety clamp | config.py only |
| rendering | **on** | 3 videos per run | `--no-render` to skip |

Dev episodes: `outputs/run_2026-05-15_17-52-54`, `run_2026-05-15_17-55-22` (cleanest for
tuning), `run_2026-05-16_01-27-32` (object drifts — read §8.3 before trusting its point errors).

---

## 10. Tiny glossary

| word | meaning |
|---|---|
| `q_ref` | interpolated predicted joint state — "where to be right now" |
| `f_ref`, `p_ref` | predicted force vector / touch spot — "what to feel right now" |
| `f_meas`, `p_meas` | the (smoothed) tactile reading |
| `Δq` / `dq` | the offset the law adds; `action = q_ref + dq` |
| chunk / knot | one policy answer (16 frames) / one predicted frame inside it |
| engaged | the prediction says this fingertip should be pressing (`\|f_ref\| ≥ 0.2 N`) |
| command lead | how far past the surface a spring must aim to actually press |
| EMA | the smoother: `0.4·new + 0.6·old` |
| Jacobian (`J`) | converts joint motion ↔ fingertip motion; flipped (`Jᵀ`), fingertip force → joint torque |
| `K` | spring stiffness of the PD, in N·m per rad — a rate, not a torque: torque = `K·Δq` (arm 400, hand 350; the runner reads the live values) |
| centroid | the one touch spot a tactile pad reports (force-weighted middle of the patches) |
| mrad | 0.001 rad ≈ 0.06° |
| ulp | the smallest possible change of a float32 number — rounding dust, not a real difference |
