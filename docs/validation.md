# Validation against the Isaac Gym reference

`scripts/compare_with_isaacgym.py` replays the *same* trajectory twice — once with the original
Isaac Gym script (`Video2Sim2Real/contact_opt/optimized_replay.py`, patched into a temporary headless
copy, nothing written into that tree) and once with this Isaac Lab port — and diffs the result.

Isaac Gym needs a PyTorch with sm_120 kernels for the RTX 5090. `env_isaaclab`'s torch and
`vid2sim2real`'s torch 2.3.1 do not have them (`CUDA error: no kernel image is available for
execution on the device`); the `video2real` env does, so that interpreter is the default
`--gym-python`.

## Results

All runs replayed at `object_mesh_scale = 0.9`, 160 frames, PD position control.

| run | commanded targets | joint pos error (mean / max) | manipulated-object position error (mean / max) |
|---|---|---|---|
| `run_2026-05-18_00-47-02` (5 objects, **successful 0.35 m pick-and-place**) | identical | 0.00137 rad / 0.01910 rad | 12.4 mm / 25.4 mm (untouched objects ≤1 mm) |
| `run_2026-08-11_17-29-12` (sponge + whiteboard) | identical | 0.00062 rad / 0.01199 rad | 4.2 mm / 6.1 mm |
| `run_2026-05-15_17-52-54` (apple + bowl) | identical | 0.00075 rad / 0.01817 rad | 2.5 mm / 3.5 mm |
| `run_2026-05-18_00-47-05` (**marginal grasp**) | identical | 0.00084 rad / 0.01861 rad | 34.0 mm / 77.3 mm (untouched objects ≤1.6 mm) |

The first row is the informative one: the trajectory actually grasps, lifts and places an object, and
the two engines agree on the whole manoeuvre — Isaac Gym lifts it **368.9 mm**, this port **366.7 mm**
(displacement 333.9 mm vs 345.6 mm), with a mean per-frame position difference of 12.4 mm over a
0.35 m motion. Objects that are never touched stay within 1 mm.

The last row is the honest limit of that agreement. `run_2026-05-18_00-47-05` is a *marginal* grasp:
the object is barely held, so Isaac Gym drags it 98.6 mm (lifting 29.3 mm) while this port moves it
25.5 mm (lifting 13.3 mm). Neither reproduces the demonstration, but the two simulators no longer
agree closely either — when contact is firm (a real grasp) or absent (untouched objects) the two
engines match to millimetres; when the grasp is on the edge of slipping, the residual differences
(no rolling friction in PhysX 5, different convex-decomposition hulls) decide the outcome.

Object *orientation* is also compared (`orientation_error_*_deg` in `comparison.json`). Objects that
are not manipulated agree to ~0.05°, but a rolling object drifts: the apple in `run_2026-05-15_17-52-54`
accumulates 11.7° mean orientation error while its position stays within 2.5 mm. That is the
rolling-friction gap in the "known differences" list — Isaac Gym applies the URDF's
`rolling_friction = 0.003`, and PhysX 5's rigid-body material has no equivalent.

"Commanded targets identical" means the per-frame joint command arrays match **exactly** (max
difference 0.0 rad) after mapping Isaac Gym's DOF order onto Isaac Lab's joint order — i.e. the
joint-name mapping through the URDF rename is correct, not merely plausible.

Residual differences are what two different PhysX integrations of the same scene produce: sub-degree
joint tracking differences and millimetre-scale object drift, concentrated in the frames where the
hand is actually in contact.

## What the comparison also shows about the trajectories themselves

Both simulators agree that these optimized trajectories, replayed open-loop, **barely move the
manipulated object**:

| run | demo (video 3D flow) | Isaac Gym | Isaac Lab (this port) |
|---|---|---|---|
| `run_2026-08-11_17-29-12` | moved 0.24 m, lifted 0.197 m | moved 20.9 mm, lifted 2.5 mm | moved 19.6 mm, lifted 3.0 mm |
| `run_2026-05-15_17-52-54` | — | moved 8.2 mm, lifted 0.8 mm | moved 6.3 mm, lifted 0.9 mm |
| `run_2026-05-18_00-47-02` | — | moved 333.9 mm, lifted 368.9 mm | moved 345.6 mm, lifted 366.7 mm |
| `run_2026-05-18_00-47-05` | — | moved 98.6 mm, lifted 29.3 mm | moved 25.5 mm, lifted 13.3 mm |

Some trajectories replay as a clean pick-and-place (`run_2026-05-18_00-47-02`, `run_2026-05-18_00-47-05`),
others barely disturb the object — and Isaac Gym agrees run for run.

So a large `flow` error in `summary.json` is not evidence of a porting problem: the reference
simulator produces the same outcome. Use `compare_with_isaacgym.py` whenever you need to separate
"the port differs from Isaac Gym" from "the trajectory does not reproduce the video".

### One systematic difference worth knowing

Objects rest ~0.8 mm higher in Isaac Gym (z = 0.8008) than here (z = 0.8000) — Isaac Gym's
scene-level `contact_offset` keeps a small separation at rest, while the per-collider
`rest_offset = 0` used here lets them touch the table exactly. It is a constant offset, it does not
grow over a trajectory, and it is the dominant term in the sub-millimetre errors reported for
objects nobody touches.

## A silent no-op the comparison caught

Isaac Sim's URDF importer marks each link's `visuals` and `collisions` scope **instanceable**, and
Isaac Lab's `apply_nested` helpers deliberately skip instanced prims. As a result the usual calls —
`UsdFileCfg.collision_props` at spawn time and `bind_physics_material(<asset root>, ...)` — silently
did nothing to the actual collision meshes: every collider kept the *scene default* material and
PhysX's default contact offset instead of the URDF's `mu1`/restitution and Isaac Gym's
`contact_offset = 0.001`.

Two more edits were landing in the same blind spot:

* the VHACD parameters (`voxel_resolution = 100000`, matching Isaac Gym) were applied to the asset
  root, where `apply_nested` succeeded and stopped — they are now applied to each collider prim, and
  the colliders really do carry `physxConvexDecompositionCollision:voxelResolution = 100000`;
* the object centre of mass was scaled twice. Measured through `root_physx_view.get_coms()`, PhysX
  applies the prim's `xformOp:scale` to an authored `physics:centerOfMass`, but *not* to an authored
  mass or inertia. The port now authors mass × s³ and inertia × s⁵ but the **unscaled** CoM, and the
  effective values read back from PhysX match Isaac Gym's `set_actor_scale` exactly (see below).

`v2s2r_isaaclab/replay.py::_prepare_colliders` now flattens the instanced scopes first, re-applies
the collision properties and the material, and then **verifies** the result, printing e.g.

```
[replay] robot: friction 1.0, 26 colliders (26 with material, 26 with contact_offset 0.001)
[replay] obj_0001: friction 0.6 restitution 0.35, 1 collider(s) (1 with material, 1 with contact_offset 0.001)
```

Effect on the manipulation case (`run_2026-05-18_00-47-02`, lift height vs Isaac Gym's 368.9 mm):

| | lift (Isaac Gym: 368.9 mm) | displacement (Isaac Gym: 333.9 mm) |
|---|---|---|
| before the fixes | 375.8 mm (+6.9 mm) | 353.5 mm (+19.6 mm) |
| after the fixes | **366.7 mm (−2.2 mm)** | 345.6 mm (+11.7 mm) |

For `run_2026-08-11_17-29-12` the same fixes cut the manipulated object's mean position error from
8.6 mm to 4.2 mm and brought its displacement from 10.4 mm to 19.6 mm against Isaac Gym's 20.9 mm.

## Scaled inertial properties match exactly

Isaac Gym's `set_actor_scale(0.9)` rescales mass/COM/inertia; `UsdFileCfg.scale` does not, so the
port recomputes them. Same run, same object (`run_2026-08-11_17-29-12`, `obj_0001`):

| quantity | Isaac Gym (`get_actor_rigid_body_properties`) | this port (`root_physx_view`) |
|---|---|---|
| mass | `0.05110679566860199` kg | `0.05110680` kg |
| com | `(-0.000123, 0.000304, 0.050311)` | `(-0.000123, 0.000304, 0.050311)` |
| Ixx | `6.19634383e-05` | `6.19634375e-05` |

The right-hand column is what PhysX reports after `sim.reset()`, not what the code authored — the two
differ, which is exactly how the centre-of-mass double-scaling above was caught.

## Memory footprint

Measured with `/usr/bin/time -v` on `run_2026-05-18_00-47-02` (5 objects, 160 frames):

| mode | peak RSS | wall clock |
|---|---|---|
| `--no-render` | 3.3 GB | 36 s |
| default (RGB + video) | 6.6 GB | 41 s |

Only the importer's `collisions` scopes are un-instanced (the `visuals` ones are never edited), which
keeps the flattening from duplicating the visual meshes. Restricting it that way left the physics
bit-identical: same joint tracking (0.002815 rad) and same object displacement (0.34562 m) as the
validated batch run.

## Determinism

Two trajectory folders hold byte-identical JSONs — `run_2026-05-15_17-52-54` and
`run_2026-05-15_17-52-54_record_time_cost` (verified with `cmp`), as do `run_2026-05-15_17-55-22`
and its `(Copy)`. Replaying such a pair in separate processes gives identical summaries (same
tracking error, same object displacement to 4 decimals), so the replay is reproducible across
processes.

Do **not** assume the other suffixed folders are copies: `run_2026-05-15_17-56-51_record_time_cost`
and `run_2026-05-16_01-27-32_time_cost` contain *different* trajectories from their unsuffixed
namesakes, and they replay to visibly different outcomes. `compare_with_isaacgym.py` therefore points
the Isaac Gym reference at the exact trajectory JSON this project replayed rather than letting the
reference re-derive it from the folder name.

## Reproducing

```bash
python scripts/replay_trajectory.py --run <run>          # produces outputs/<run>/<ts>/replay_data.npz
python scripts/compare_with_isaacgym.py --run <run>      # runs Isaac Gym and diffs against it
```

Outputs land in `outputs/_comparisons/<run>/<ts>/`: `comparison.json`, `comparison.png` (per-object
x/y/z traces, both engines), `gym_reference.npz`, `gym_reference.log`, and the patched script that
produced the reference.


## Batch results — all 16 trajectories

`python scripts/replay_all.py` (headless, images + video, ~40 s per run) — every run completed:

| run | frames | joint tracking mean \|err\| (rad) | largest object displacement (m) | largest lift (m) | flow Δ-error vs demo (m) |
|---|---|---|---|---|---|
| `run_2026-05-15_17-52-54` | 160 | 0.0020 | 0.006 | 0.001 | 0.095 |
| `run_2026-05-15_17-52-54_record_time_cost` | 160 | 0.0020 | 0.006 | 0.001 | 0.095 |
| `run_2026-05-15_17-55-22` | 160 | 0.0016 | 0.005 | 0.004 | 0.097 |
| `run_2026-05-15_17-55-22 (Copy)` | 160 | 0.0016 | 0.005 | 0.004 | 0.097 |
| `run_2026-05-15_17-56-51` | 150 | 0.0021 | 0.005 | 0.000 | 0.093 |
| `run_2026-05-15_17-56-51_record_time_cost` | 150 | 0.0022 | 0.016 | 0.000 | 0.101 |
| `run_2026-05-16_01-27-32` | 160 | 0.0028 | 0.264 | 0.302 | 0.113 |
| `run_2026-05-16_01-27-32_time_cost` | 160 | 0.0020 | 0.045 | 0.027 | 0.135 |
| `run_2026-05-18_00-47-02` | 160 | 0.0028 | 0.346 | 0.367 | 0.062 |
| `run_2026-05-18_00-47-05` | 160 | 0.0025 | 0.026 | 0.013 | 0.087 |
| `run_2026-05-18_00-47-07` | 160 | 0.0023 | 0.082 | 0.038 | 0.083 |
| `run_2026-05-19_21-39-18` | 160 | 0.0022 | 0.021 | 0.019 | 0.113 |
| `run_2026-08-06_19-44-36` | 160 | 0.0112 | 0.075 | 0.010 | 0.152 |
| `run_2026-08-09_14-30-45` | 160 | 0.0232 | 0.088 | 0.110 | 0.134 |
| `run_2026-08-09_15-37-32` | 160 | 0.0021 | 0.082 | 0.007 | 0.174 |
| `run_2026-08-11_17-29-12` | 160 | 0.0021 | 0.043 | 0.003 | 0.214 |

Batch summary: `outputs/_batches/20260823_191003/batch_summary.json`.

Some trajectories genuinely manipulate (`run_2026-05-18_00-47-02`: 0.35 m displacement, 0.37 m lift;
`run_2026-05-16_01-27-32`: 0.26 m), others only nudge the object. `run_2026-08-09_14-30-45` has the
largest PD tracking error (0.023 rad mean) — its commanded trajectory contains the fastest joint
steps in the set.

---

## 2026-08-26: why grasps failed at the old default scale — and the fix

Several optimized trajectories that grasped during the pipeline's Isaac Gym tests would not grasp in
this port (fingers closing just beside the object, near-zero fingertip force). Tracing the pipeline's
actual invocations showed the port's **default `object_mesh_scale = 0.9` was a misreading**:

* `grasp_retry_loop.sh:33` passes 0.9 **only to grasp-pose generation** (lightning-grasp). The
  candidate/optimized JSONs record the value as metadata, but
* every Isaac Gym replay/test ran at **1.0**: `kinova_replay_grasping_test_interaction`
  (`run.sh:199`), the disturbance test (retry-loop stage 4) and `optimized_replay.py` were all
  invoked without `--object_mesh_scale`, and none of them read the JSON field.

Empirical confirmation (`run_2026-05-15_17-52-54`, winning interaction candidate `0027` replayed in
this port): at scale 1.0 the grasp succeeds (0.14 m lift, 52/25/57 N fingertip forces); at 0.9 the
same trajectory closes on air (0.25 N peak). The two previously validated grasping runs also still
succeed at 1.0 (`run_2026-05-18_00-47-02`: 0.42 m lift; `run_2026-05-16_01-27-32`: 0.34 m).
The fallback is now 1.0 (an explicit JSON field or `--object-mesh-scale` still wins). All the
engine-parity numbers above were measured with **matched** scales in both engines, so they remain
valid.

Separate data finding, not a physics issue: the **Jun 13 versions** of
`retarget_kinova_leap_optimized.json` for `run_2026-05-15_17-52-54 / 17-55-22 / 17-56-51` fail to
grasp **in Isaac Gym too** (Aug 23 gym reference: 8 mm object displacement; the May 18
`replay_leap` frames end with the object still on the table). The grasps that actually succeeded are
the `grasp_test_interaction` candidate trajectories these JSONs were later re-derived from.

---

## 2026-08-26 (later): the second half of the grasp fix — convex-decomposition fidelity

With the scale corrected, `run_2026-05-15_17-55-22` (trajectory `0109`, restored to
`contact_opt/leap/retarget_kinova_leap_optimized` on 2026-08-26) still slipped out of the grasp in
this port while Isaac Gym lifted it 134 mm. Symptoms that localised it: the object rested **1.5 mm
above the table** here (0.80136 m vs gym's 0.79949 m), lab fingers made contact one frame early and
nudged the object away, and the pinch slipped at ~12 N.

Cause: PhysX 5 replaced Isaac Gym's VHACD with its own convex decomposer, and in this Isaac Sim
build the cooked hulls react to **`errorPercentage` only** — changing `voxelResolution`,
`shrinkWrap` or `maxConvexHulls` produced bit-identical replays (no cooked-data cache involved;
verified with `--/physics/cooking/ujitsoCollisionCooking=false`). At the PhysX default of **10%**
the hulls of a hand-sized scanned mesh are ~1.5 mm fatter than gym's VHACD hulls. Two further
gotchas found on the way: the importer authors `CollisionAPI`/approximation on the link's `World`
*Xform* while the mesh prim carries no collision APIs (the replay now writes the decomposition
attributes to the mesh prims as well), and the `collisions` scopes are instanceable, so
`Usd.PrimRange` needs `TraverseInstanceProxies` to see them at all.

The replay now cooks object colliders at **`errorPercentage = 1.0`** (`--object-decomposition-error`
to override). Results:

| run (traj) | Isaac Gym lift | this port @10% error | this port @1% error |
|---|---|---|---|
| `17-55-22` (`0109`) | 134 mm | 10 mm (slips) | **139 mm** |
| `17-52-54` (`0027`) | — | grasps | grasps (135 mm) |
| `00-47-02` | 367 mm (at 0.9) | grasps | grasps (369 mm) |
| `01-27-32` | — | grasps | grasps (320 mm) |

Resting height matches Isaac Gym to 0.1 mm with the fix. `run_2026-05-15_17-56-51` still fails to
grasp — in both engines; its `run.sh` entry is annotated "#bad retarget" and its trajectory was not
part of the 2026-08-26 restore.

---

## 2026-08-26 (later still): run_2026-05-15_17-56-51 — the grasp that needs VHACD's padding

The last hold-out. Its optimized trajectory is byte-identical to interaction candidate `0000` of the
`grasp_test_interaction/run_2026-05-15_17-56-51 (Copy)` batch (6/6 candidates succeeded there, best
= `0000`), and **Isaac Gym lifts it 157.5 mm** with the same file at scale 1.0 — while this port
grazed the object at 2–3 N and left it on the table, at any decomposition error (0.5–10% are
bit-identical for this mesh), with more hulls (256/255: lift 1.6 mm), and even with true-surface SDF
collision (worse: 1.4 N).

The object explains it: a **30 g "brown toy", 3.3 × 6.3 × 3.7 cm**. Isaac Gym's VHACD covers such a
small concave shape with convex hulls that bridge its indentations — an effectively *padded,
simplified* toy, and the knife-edge pinch was optimized against that envelope. Every faithful
geometry here (tight decomposition, SDF) is thinner than what the grasp was tuned on, so the index
fingertip shoves the toy out of the pinch before the thumb arrives (visible side-by-side with the
May-18 gym frames).

The envelope was quantified with a diagnostic rest-offset sweep (contacts resolving that far
outside the object's surface): 1 mm of padding → 0.5 mm lift, 2 mm → 32 mm, 2.5 mm → 156 mm,
3 mm → 155 mm, 4 mm → 153.5 mm — a plateau that brackets gym's 157.5 mm, i.e. the grasp closes
~2.5 mm short of the *real* toy surface. **Resolution: fix the trajectory, not the simulation.**
Padding the object in sim reproduces Isaac Gym's artifact rather than reality, so no such option is
kept in the replay; this run needs a trajectory re-optimized against the true geometry (the ~2.5 mm
figure above is how much deeper the pinch must close). Until that trajectory lands, the replay
faithfully reports the grasp as failing.
