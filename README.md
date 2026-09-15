# ZeroFACT — Video2Sim2Real trajectory replay in Isaac Lab

Self-contained Isaac Lab 2.3 / Isaac Sim 5.1 port of the Video2Sim2Real Isaac Gym replay
(`Video2Sim2Real/contact_opt/optimized_replay.py`), for the optimized Kinova Gen3-7DoF + LEAP hand
trajectories in

```
Video2Sim2Real/video_tests/Manioulation_data/five_objects/contact_opt/leap/retarget_kinova_leap_optimized/
```

All 16 of those trajectories, the scenes they were planned in, and the robot description have been
copied into `data/` here, so nothing in this folder depends on the Video2Sim2Real tree at run time.

> **Not in the git repo:** `data/`, `assets/usd/` and `outputs/` (trajectories, converted USDs, run
> results — ~13 GB) are git-ignored. Regenerate them with `scripts/ingest_data.py` and
> `scripts/convert_assets.py`, or copy them over from an existing checkout.

The port is **validated against Isaac Gym**, not just "it runs": `scripts/compare_with_isaacgym.py`
replays the same trajectory in the original Isaac Gym script and diffs the result. On the run that
actually performs a pick-and-place, the two engines lift the object **368.9 mm vs 366.7 mm** and
their joint commands are identical; joint tracking agrees to ≤0.0014 rad mean and object positions
to 0.5–12 mm over 160 frames. One *marginal* grasp diverges (34 mm) — see
[`docs/validation.md`](docs/validation.md) for the full picture, including what the comparison
caught in this port's own physics setup.

---

## Quick start

```bash
conda activate env_isaaclab
cd ~/ZeroFACT

# 0. (already done) pull the trajectories + scenes out of Video2Sim2Real
python scripts/ingest_data.py

# 1. (already done) convert the robot and every scene's objects to USD
python scripts/convert_assets.py

# 2. replay one trajectory: physics + camera images + video + plots
python scripts/replay_trajectory.py --run run_2026-08-11_17-29-12

# 3. replay everything (one Isaac Sim process per run) and print a summary table
python scripts/replay_all.py

# 4. check the port against the original Isaac Gym script
python scripts/compare_with_isaacgym.py --run run_2026-08-11_17-29-12
```

Results land in `outputs/<run>/<timestamp>/`:

| file | what it is |
|---|---|
| `rgb_images/frame_%04d.png` | camera frames from the *demo camera pose*, one per trajectory frame |
| `replay.mp4` | the same frames as a video — **with a colour-coded force arrow drawn at each fingertip's measured contact point and an emissive sphere at every individual PhysX contact point** (sphere size grows mildly with per-point force; contacts outside the tracked objects fall back to an arrow at the fingertip body origin) |
| `handcam.mp4` | close-up video from a camera that tracks the fingertips — the view where the contact points/arrows are actually legible (also inset into the combined video) |
| `contact_forces.mp4` | the contact-force plot as a video, a cursor sweeping the frame axis in step with `replay.mp4` |
| `replay_with_forces.mp4` | annotated side-by-side video: camera view + sweeping force plot, live per-fingertip readout, key-frame call-outs |
| `replay_data.npz` | commanded actions (`joint_target`) **and measured states** — joint pos/vel at frame rate and at every physics step (`joint_pos_steps`/`joint_vel_steps`), applied actuator torques (`applied_torque[_steps]`), tracked body poses, object poses/velocities, fingertip contact forces (net **per physics step** + per-object breakdown), contact-patch positions (`contact_point_w`) **and every individual contact point with its normal + per-point force** (`contact_points_w` / `contact_points_normal_w` / `contact_points_force_N`), 3D flow points |
| `summary.json` | every constant actually used, the resolved joint map, tracking / object-motion / contact / flow metrics |
| `joint_tracking.png` | commanded vs simulated joint angles, with the trajectory's key frames marked |
| `object_tracking.png` | object world position over time |
| `contact_forces.png` | per-fingertip contact force at every physics step: net force, force from the manipulated object, and everything else |
| `flow_comparison.png` | simulated object motion vs the demo video's 3D flow |
| `hand_pose_traj.json` | palm pose per frame, in world and in the manipulated object's frame (same schema as the Isaac Gym script's output, quaternions xyzw) |
| `depth_images/frame_%04d.npy` | per-frame depth, with `--save-depth` |

---

## Memory: this is a shared box

A replay peaks at about **6.6 GB** resident with rendering, **9.4 GB** with `--gui`, and **3.3 GB**
with `--no-render` (measured with `/usr/bin/time -v` on the 5-object `run_2026-05-18_00-47-02`). The machine has 62 GB
but is shared, and a desktop session with a browser + VS Code easily holds 35-40 GB. If the free
memory drops below the peak, the kernel OOM-kills the run — the terminal just prints `Killed`, and
the desktop freezes for a while first while it swaps.

The replay prints a warning before starting Isaac Sim when there is not enough headroom. Check with
`free -g` (look at *available*, and at swap) and, if it is tight:

* close browser/editor windows, or
* run with `--no-render` (physics + npz + plots, no images/video), or
* wait for the other users' jobs to finish — `ps -eo user,rss --no-headers | awk '{a[$1]+=$2} END {for (u in a) print u, a[u]/1048576" GB"}'`
  shows who is holding what.

## This machine: two environment traps

**1. `DISPLAY` points at another user's X session.** `~/.bashrc` exports `DISPLAY=:2`, which belongs
to a different login on this shared box. The NVIDIA Vulkan ICD cannot initialise against it, so
`vkCreateInstance` returns `VK_ERROR_INCOMPATIBLE_DRIVER` and Isaac Sim dies at startup with

```
[carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, ...
```

Every script here calls `zerofact.runtime.prepare_display()` before launching Kit: it probes
Vulkan, and if it fails, switches to the X display that belongs to the current user (`:1`). Force a
specific one with `V2S2R_DISPLAY=:1` (or `V2S2R_DISPLAY=none`). Nothing else needs to change — this
applies to headless runs too, because Kit still needs a Vulkan instance to render.

`--gui` uses a **stricter** test (`prepare_display(require_window=True)`): the display must also
accept an X connection (`XOpenDisplay`), which is what GLFW needs to open a window. Vulkan happily
initialises on `:2` even though this user cannot open it, and Kit then starts *windowless* with only
a `GLFW initialization failed` warning — the symptom being a `--gui` run that produces no window.

**2. Isaac Lab's rendering and windowed experience files crash this Isaac Sim install.** Launching
with `isaaclab.python.headless.rendering.kit`, `isaaclab.python.rendering.kit` or the windowed
`isaaclab.python.kit` fails during extension startup — `module 'omni.usd' has no attribute
'UsdContext'`, then SIGSEGV. It reproduces with plain `isaacsim.SimulationApp` too, so it is the
installation, not Isaac Lab or this code. It is *not* caused by cache corruption, a stale
`user.config.json`, the `--portable` flag, or a leftover Kit process — all of those were tested.

The workaround is built into `scripts/replay_trajectory.py`: start from the *plain headless*
experience, which loads cleanly, and switch the needed extensions on by hand —

```
--experience isaaclab.python.headless.kit
--kit_args "--enable omni.replicator.core --enable omni.kit.viewport.rtx \
            --enable omni.kit.material.library --/isaaclab/cameras_enabled=true"
```

— plus, for `--gui`, the UI stack (`omni.kit.mainwindow`, `omni.kit.viewport.window`,
`omni.kit.manipulator.camera`, `omni.kit.window.toolbar`, `omni.kit.window.status_bar`). Offscreen
rendering (640×480 RGB + depth) and a normal *Omniverse Kit* window both work this way. Pass
`--no-compat-rendering` to use the stock experiences once the install is repaired.

---

## What the replay reproduces

The Isaac Gym setup was measured, not guessed. The non-obvious pieces:

| Isaac Gym | here | why it matters |
|---|---|---|
| `AssetOptions.armature = 0.01` | `+0.01` added to every link's `ixx/iyy/izz` at ingest | Isaac Gym adds armature to the **inertia diagonal**, not to the joint. The LEAP fingertip's real inertia is 3.4e-6 kg·m²; with `K=350` that is numerically explosive. The hand gains only make sense with the bump. |
| `joint_1` declared `continuous` with a stray `<limit lower="-1.5708" upper="0">` | the `lower/upper` pair is stripped at ingest | Isaac Gym ignores limits on continuous joints; the Isaac Sim URDF importer honours them and would silently clamp commands (5 runs command `joint_1 > 0`). |
| LEAP joints named `"0"…"15"` | renamed to `leap_j0…leap_j15` (`data/robot/joint_name_map.json`) | USD prim names may not start with a digit: the importer maps *every* single-digit joint onto the same name `a_`, silently destroying the hand. |
| DOF order | name-based mapping, printed every run | Isaac Gym's DOF order, the trajectory JSON's key order and Isaac Lab's joint order are all different. |
| `dt = 1/60`, `substeps = 2`, 6 `simulate()` per frame | `dt = 1/120`, 12 steps per frame | identical sim time (0.1 s/frame → 10 Hz replay) and identical substep size. |
| target set *inside* the inner loop | first 2 steps of each frame hold the previous target | the reference's first `simulate()` runs before the new command is written. `--stale-target-steps 0` turns this off. |
| `set_actor_scale(s)` when a scale ≠ 1 is requested | `UsdFileCfg.scale` **and** mass ×s³, COM ×s, inertia ×s⁵ | USD scale only scales geometry; Isaac Gym also rescales the inertial properties. (The pipeline's own gym replays all ran at s = 1.0 — see the `object_mesh_scale` note above.) |
| loaded-asset shape friction (1.0) | `SimulationCfg.physics_material` = 1.0/1.0/0.0, robot bound to the same | Isaac Lab's default is 0.5, which would change every robot contact. |
| table friction 0.45, object `mu1` from the URDF gazebo block | per-collider `RigidBodyMaterialCfg` | same numbers, including the `restitution=` value that lives in a trailing XML comment. |
| VHACD `resolution = 100000` | `ConvexDecompositionPropertiesCfg(voxel_resolution=100000, max_convex_hulls=64, hull_vertex_limit=64)` | object collision fidelity. |
| fingertip force sensors | one `ContactSensor` per fingertip (`fingertip`, `fingertip_2`, `fingertip_3`, `thumb_fingertip`), filtered against every scene object | net contact force per fingertip at **every physics step** (`contact_force_steps`), the per-frame sample (`contact_force`), and the fingertip↔object force matrix (`contact_object_force_steps`) — so grasp force on the manipulated object is separable from table/self contacts. |

`object_mesh_scale` honours the trajectory JSON's field when present (2 of the 16 optimized JSONs
carry `0.9`) and otherwise defaults to **1.0** — because that is what every Isaac Gym replay/test in
the pipeline actually ran with: `kinova_replay_grasping_test_interaction` (`run.sh:199`), the
disturbance test (`grasp_retry_loop.sh` stage 4) and `optimized_replay.py` were all invoked without
`--object_mesh_scale` and none of them read the JSON field. The `0.9` in `grasp_retry_loop.sh:33`
goes only to grasp-pose *generation* (lightning-grasp), and the JSONs record it as metadata.
Replaying at 0.9 shrinks objects out of grasps that were validated on full-size ones (verified on
interaction candidate 0027 of `run_2026-05-15_17-52-54`: lifts at 1.0, whiffs at 0.9 — in both
engines). Override with `--object-mesh-scale`.

### Harmless warnings you will see

* `Unresolved reference prim path ... /visuals` for `index_tip_head`, `thumb_tip_head`,
  `middle_tip_head`, `ring_tip_head`, `end_effector_link` — those URDF links carry no visual or
  collision geometry (they are pure frames), so the importer writes an empty `visuals` scope. They
  still exist as bodies and their poses are correct.
* `Isaac Sim shutdown is hanging; exiting the process directly` — Kit 107 regularly blocks in
  `SimulationApp.close()` on this machine. Everything is already written to disk at that point.
* `Disabling key-value database because another kit process is locking it` — a second Kit process is
  running. Run **one** Isaac Sim process at a time: two rendering processes contend on the shader
  cache and can stall each other for minutes. `replay_all.py` already serialises its runs.
* `enable_external_forces_every_iteration ... is set to False` — Isaac Lab's own advice; the Isaac
  Gym reference did not enable it either.

### Known differences from Isaac Gym

* **Rolling / torsional friction** (`rolling_friction=0.003` in the object URDFs) has no equivalent
  in PhysX 5's rigid-body material and is dropped.
* **Convex decomposition** is PhysX's, not Isaac Gym's VHACD build — and in this Isaac Sim build the
  cooked hulls react to `errorPercentage` only (voxel resolution / shrink-wrap / hull-count edits
  are ignored). At PhysX's 10% default the hulls of a hand-sized scanned object are ~1.5 mm fatter
  than gym's, enough that objects rest 1.5 mm above the table and marginal grasps slip. The replay
  therefore cooks object colliders at **1% error** (`--object-decomposition-error` to override),
  which matches Isaac Gym's resting height to 0.1 mm and restores the grasps — see
  `docs/validation.md`, 2026-08-26.
* **`mu2`** is recorded but unused: Isaac Gym has a single friction coefficient, so static and
  dynamic friction are both set to `mu1`.
* **Fingertip force reading**: Isaac Gym attached a 6-axis rigid-body force sensor to each
  fingertip (`create_asset_force_sensor` with forward-dynamics + constraint-solver forces, world
  frame) and read it once per frame. This port records the **net contact force** from PhysX's
  contact reports instead — no torque channel and no inertial/gravity component, but sampled at
  every physics step and with a per-object breakdown the Isaac Gym sensor cannot give. On the
  near-massless fingertips the two agree whenever contact dominates the reading.
* **Camera far plane** defaults to 20 m here; Isaac Gym used 1.0 m, which clips most of the scene.
  Pass `--cam-far 1.0` to match it exactly.
* **Recording is one step less stale**: Isaac Gym read its tensors before the last physics step of
  each frame; this port records after all 12.

---

## Layout

```
data/
  manifest.json                     what was ingested, from where
  robot/kinova_leap_description/    robot URDF (renamed joints, stripped limits, armature baked) + meshes
  robot/joint_name_map.json         trajectory joint name -> USD joint name
  runs/<run>/
    retarget_kinova_leap_optimized.json    the trajectory being replayed
    run_meta.json                          provenance, key frames, object poses, static objects
    scene/{scene_output_final,table_frame_pose,camera_frame_pose}.json, cam_params.txt
    scene/urdfs/obj_XXXX.urdf, scene/meshes/obj_XXXX_transformed.obj
    flow_data/                             demo 3D/2D object flow + contact step
assets/usd/
  robot/kinova_leap.usd, robot/robot_info.json   (joint/body names + limits as Isaac Lab sees them)
  scenes/<scene_run>/obj_XXXX.usd
  index.json
zerofact/
  scene_spec.py   pure-numpy scene/trajectory description (no Isaac imports)
  replay.py       Isaac Lab scene construction + control loop
  analysis.py     metrics + plots (works offline on replay_data.npz)
  naming.py       trajectory <-> simulation joint-name mapping
  runtime.py      DISPLAY/Vulkan guard, safe shutdown
scripts/
  ingest_data.py           copy trajectories + scenes here, prepare the robot URDF
  convert_assets.py        URDF -> USD for robot and objects, dump the articulation's joint names
  import_3mf_scene.py      3MF print project -> data/runs/<scene> (meshes, URDFs, poses, held start pose)
                           for teleop / controller scenes without a Video2Sim2Real reconstruction
  test_thread.py           physics check for SDF-collider scenes: torque the nut, verify height/turn = pitch
  replay_trajectory.py     replay one run
  replay_all.py            replay every run, one process each, with a summary table
  render_force_video.py    rebuild contact_forces.mp4 + replay_with_forces.mp4 from an existing
                           output folder (no Isaac Sim needed - annotation tweaks in seconds)
  compare_with_isaacgym.py validate against the original Isaac Gym script
outputs/                   per-run results, batch summaries, comparisons
```

---

## Useful flags (`scripts/replay_trajectory.py`)

| flag | effect |
|---|---|
| `--no-render` | physics only; ~7 s per 160-frame trajectory |
| `--kinematic` | teleport the joints instead of PD tracking (Isaac Gym's `--visualize`) |
| `--max-frames N` | stop early |
| `--save-depth` | also write per-frame depth `.npy` |
| `--object-mesh-scale S` | override the object scale (default: JSON field, else 1.0) |
| `--static-object-ids '0000'` | weld a scene object in place (`kinematic_enabled`) |
| `--object-decomposition-error P` | collider decomposition error tolerance in % (default 1.0; PhysX's 10 bloats small objects ~1.5 mm) |
| `--stale-target-steps N` | how many steps of each frame hold the previous command (default 2) |
| `--cam-far` / `--use-real-intrinsics` | camera clipping / use `cam_params.txt` instead of the forced 90° FOV |
| `--joint-armature` | PhysX joint armature; keep 0 (Isaac Gym's armature is in the link inertias) |
| `--no-contact-sensors` | skip fingertip contact sensing |
| `--no-force-vis` | do not draw the fingertip force arrows in the scene/video |
| `--force-vis-scale S` | force-arrow length in m per N (default 0.005 = 0.5 cm/N) |
| `--no-force-videos` | skip `contact_forces.mp4` and `replay_with_forces.mp4` |
| `--no-render` | also the low-memory option: 3.3 GB peak instead of 6.6 GB |
| `--gui` | live Omniverse Kit window, paced to real time (~9.4 GB peak) |
| `--keep-open` | with `--gui`, leave the window open when the replay ends |
| `--no-realtime` | with `--gui`, replay as fast as the machine allows |
| `--gui-render-interval N` | with `--gui`, render every Nth physics step (default 2 ≈ 60 Hz) |
| `--no-images` / `--no-video` / `--save-ratio N` | render but skip PNGs / skip the mp4 / save every Nth frame |
| `--physics-dt` / `--steps-per-frame` / `--settle-steps` | timing overrides (defaults reproduce Isaac Gym exactly) |
| `--flow-points N` | surface samples for the flow metric (0 disables) |
| `--out-dir` / `--data-dir` / `--usd-dir` | override the output, data and USD locations |
| `--video-fps` / `--seed` | video frame rate / surface-sampling seed |

Everything Isaac Lab's own `AppLauncher` accepts (`--device`, `--kit_args`, `--experience`, …) also
works; run with `--help` for the full list.

Re-running `scripts/ingest_data.py` rewrites the robot URDF; re-run
`scripts/convert_assets.py --robot-only --force` afterwards, otherwise the USD is stale. The replay
checks the things that matter (23 joints, 31 bodies, continuous joints unlimited, fingertip inertia
bumped) and warns loudly if a reconversion is needed.

---

## Interpreting a replay

`summary.json`'s `flow` block compares the simulated object motion with the demonstration's 3D
object flow. A large error is not automatically a port bug: for `run_2026-08-11_17-29-12` the demo
lifts the sponge 0.197 m while **both** Isaac Gym and this port move it ~0.02 m and never lift it,
i.e. the optimized trajectory does not reproduce the demo in either simulator. Use
`compare_with_isaacgym.py` to separate "the port differs from Isaac Gym" from "the trajectory does
not do what the video did".
