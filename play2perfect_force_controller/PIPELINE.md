# How the policy and the force controller work together

Plain-language description of the pipeline in this folder, what each part sees, and how to read
the force plots. Figures referenced below are in `outputs/play2perfect/force_controller/figures/`
and are produced with `plot_force_comparison.py`.

## 1. Three layers

```
  policy  ──(every C frames: a CHUNK)──►  middle layer  ──(every physics step: joint targets)──►  robot PD  ──►  simulator / real hand
                                              ▲
                                              └── fingertip tactile readings + joint states (nothing else)
```

* **Policy.** Outputs a chunk: for each of the next `horizon` (H = 24) frames at 60 Hz, the robot
  joint state it predicts and, per fingertip, the force it predicts: contact point, world-frame
  force vector and magnitude. A new chunk is requested every `chunk` (C = 12) frames. The policy
  is the only part that sees the object. In the final system it will be a BC policy trained on
  human data, so it predicts *reached* states, never motor commands.
* **Middle layer** (`middle_layer.py`). Turns the chunk plus the live tactile readings into a
  joint-position target at every physics step (120 Hz). It sees joints and fingertip forces
  only. It does not know where the object is.
* **Robot PD.** The joint-position controller of the hand and arm (the simulator's implicit PD,
  or the real hand's position mode). Force on the object comes from the difference between the
  commanded target and the reached position, multiplied by the joint stiffness.

## 2. Where the policy comes from today

The BC policy does not exist yet. Two stand-ins produce chunks with exactly the same interface
(`policy.py`, `oracle_policy.py`):

| stand-in | what it serves | loop |
|---|---|---|
| `ChunkedReplayPolicy` (`--policy replay`, default) | slices of a recorded episode: the recorded joint states and the recorded fingertip forces | open loop: the chunk never depends on what happened |
| `OracleChunkPolicy` (`--policy oracle`) | at every chunk boundary the released RL policy is rolled forward H frames inside the simulator from the current state; its commands, reached states and fingertip forces become the chunk; the world is then restored | closed loop at chunk rate: the plan follows the real state every C frames |

The oracle is the fair test bed: it gives the position layer the object feedback a future BC
policy will have (through its observations), while the middle layer still only sees tactile.

## 3. The two ways to serve the state: commands or reached states

`--state-source` decides what "predicted robot state" means:

* `joint_target` (**commands**): the target the policy sent to the PD. On a position-controlled
  hand a squeeze is a target *inside* the object, so commands carry the preload. Executing them
  reproduces the force by itself; the force law only corrects.
* `joint_pos` (**reached states**): where the joints actually were. The preload is invisible in
  them. This is what a policy trained on human data can predict, so this is the premise we keep.
  The force law must *generate* the preload from the force targets.

The recorded Sharpa episodes show how big that preload is: during contact the commands lead the
reached positions by 0.2 to 0.4 rad (95th percentile), up to 1.2 rad, on joints with 0.9 N·m/rad
stiffness.

## 4. What the middle layer does at every physics step

1. **Reference** (`ChunkTracker`). Interpolates the chunk's knots to the current time: joint state
   `q_ref`, force target vector `f_ref` per fingertip, contact point `p_ref`. Chunk switches are
   continuous (the new chunk is anchored at the reference being tracked).
2. **Tactile filter.** EMA of each pad's net force vector and contact centroid.
3. **Force law** (`TaskSpaceForceLaw`), per fingertip:
   * error `e = f_ref - f_meas` (vector), deadband and clip;
   * commanded force `f_cmd = g·kff·f_ref + kp·e + I + kd·de`, where `I` integrates `e` while the
     fingertip is *engaged* (`|f_ref| >= engage_threshold_n`) and leaks out otherwise, and `g` is
     an adaptive feedforward gain estimated online from achieved/commanded force;
   * friction-cone cap: `f_cmd` may not rotate more than `cone_half_angle_deg` away from the
     predicted direction and may never pull;
   * **force → joint offset**: `dq = -K⁻¹ Jᵀ f_cmd` (statics through the PD stiffness `K`; `J` is
     the contact-point Jacobian), joints sitting at a position limit they would push into are
     masked (rigid there). `--map constrained` solves the same model as a bounded least squares
     that keeps the force direction under the clip and effort bounds;
   * **contact-point channel**: the tangential part of `p_ref - p_meas` (the part not along
     `f_ref`) is mapped back through the damped pseudo-inverse of `J` with gain `point_kp`, so the
     pad also moves to *where* on the surface it should press. `--point-engage-n` /
     `--point-steady-rate` gate it to established, steady contacts.
4. **Clip and sum.** Offsets are clipped per joint (`offset_clip_rad`, 0.12) and added:
   `target = q_ref + dq`. Arm joints get no offset unless `--allow-arm-offset`.
5. The target goes to the PD. In the simulator (`sim_runner.py`) this is one raw physics step.

`--force-law null` skips step 3 and 4: the target is `q_ref`. That baseline shows what the position
layer alone does.

## 5. One rollout, frame by frame (`sim_runner.run_force_tracking`)

```
reset env to the recorded start (same seed for the oracle; identical on every key)
for frame k = 0, 1, 2, ...:
    if k % C == 0:   chunk = policy.predict(k)         # replay: slice; oracle: snapshot, roll RL H frames, restore
    (oracle) policy.advance()                          # its LSTM sees the observation of this frame
    for substep in 0, 1:                                # 120 Hz
        read pads (net force, centroid), joints, Jacobians
        target = middle_layer(t, tactile)              # sections 4.1–4.4
        physics_step(target)
    record joints, targets, forces, object poses, law internals
    (oracle) policy.after_frame(target)                # env task logic: sub-goal advance, retract, done
```

Frames are 60 Hz policy steps; value at frame k is the state at the end of frame k; the command
held during frame k is `joint_target[k]`.

## 6. How rollouts are scored

* **env verdict** (oracle runs): the env's own success counter and retract flag. Screwing is
  rotation-symmetric, so for screwing only this column counts.
* **inserted** (offline, all runs): the env's keypoint test recomputed from the saved object poses
  against the recording's final goal.
* **force bias**: mean(measured − target) over engaged steps, worst fingertip.
* **slip / drop**: unintended object motion in the palm frame, and whether the grasp was lost.
* **exact replay** (`--exact-replay`): commands, held per frame, null law; must reproduce the
  recording bit for bit. It is the regression test of the whole runner.

`evaluate_controller.py` runs a law over every perfect episode (one Kit process per rollout) and
writes `outputs/play2perfect/force_controller/evaluation_<stamp>.md`.

## 7. Watching a rollout with its force tracking

Every rendered rollout now gets `rollout_force_tracking.mp4` (`render_force_video.py`): the
camera view with the force arrows on the left; on the right one row per fingertip with the force
target dashed and the measured force solid, a red cursor at the current time, and the live
numbers (target, measured, error, engaged or not) plus the object height. Ready-made examples in
`outputs/play2perfect/force_controller/figures/videos/`:

| video | what to watch |
|---|---|
| `tight_seed2_replay_reachedstates_taskspace.mp4` | open-loop replay: after the first touch the measured forces stay at zero while the target rises, the hand is no longer at the part |
| `tight_seed2_oracle_reachedstates_taskspace.mp4` | closed loop, reached states: the law builds the grasp (index and thumb track the target with a 2 to 4 N under-press), lifts the part, loses it when the plan releases for a re-grasp |
| `tight_seed2_oracle_commands_taskspace.mp4` | closed loop, commands: measured follows target closely, insertion succeeds |
| `beam1_seed2_replay_reachedstates_taskspace.mp4` | open-loop replay of the beam: lift with a steady under-press, then the beam creeps out |

What the camera shows (since 2026-09-11): the adapter plate between the KUKA flange and the Sharpa
palm is drawn (visual-only geometry added to the URDF's `sharpa_mount` link; the physics is
unchanged and exact replays stay bit-exact), and the target pose is a translucent copy of the part
(18 % opacity, same colour) so it is told apart from the real part even when the two coincide
(`--goal-marker-opacity 1.0` gives the old opaque marker). `compose_task_video.py` stacks three
rollouts of one recording in a column, each row camera + force curves under a coloured title:
A. original action replay (sanity check), B. state-only replay (no force tracking), C. state replay
+ force tracking; `package_results.py` writes one per episode into `share_results/abc_comparison/`.

## 8. Reading the force plots (static overlays)

`plot_force_comparison.py` overlays, for several rollouts of one episode, the force target
(dashed) and the measured force (solid) per fingertip, plus the object height. The object drops
about 10 cm in the first 0.3 s in every run: that is the part settling onto the table after the
reset, not a failure.

**Figure A – open-loop replay, reached states, tight insertion seed 2**
(`figures/tight_seed2_replay_null_vs_taskspace.png`). After the first touch at 1 s the measured
force of both laws stays at zero while the recorded target rises to 10–25 N from 2 s on. The
hand is no longer where the object is: the replayed trajectory diverged and the part was pushed
away (drop marker at 1.3 s). There is nothing for the force law to track. This is the failure of
the open-loop protocol, not of the law.

**Figure B – oracle, reached states, tight insertion seed 2**
(`figures/tight_seed2_oracle_pos_null_vs_taskspace.png`). With re-planning every 0.2 s the null
law (blue) still never builds force and never lifts: reached states carry no preload. The
task-space law (red) does build it: index 5–10 N against 7–10 N target, thumb 10 N against
10–15 N, pinky 2 N against 3–5 N, and lifts the part 6.5 cm and carries it for 7 s. Two things
are visible: a steady **under-press of roughly 30 percent** (the clip binds, see the sweep), and
at 5.4 s the plan's targets drop to zero (the policy releases to re-grasp), the law releases with
them, the part then rests on the hand without pad contact and falls at 8.5 s.

**Figure C – oracle, commands, tight insertion seed 2**
(`figures/tight_seed2_oracle_cmd_null_vs_taskspace.png`). When the commands carry the preload,
measured force follows the target closely with and without the law, and both runs insert. Force
tracking is easy here; the law is a corrector.

**Figure D – oracle, commands, screwing seed 0**
(`figures/screwing_seed0_oracle_cmd_null_vs_taskspace.png`). Force tracking is good in both runs,
yet the null law finishes the ten turns at 12.7 s while the task-space run turns more slowly and
loses the screw at 14 s. The law's offsets interfere with the frequent re-grasps, not with force
tracking. That is what the point-channel gate addresses.

**Figure E – open-loop replay, reached states, beam step 1 seed 2**
(`figures/beam1_seed2_replay_null_vs_taskspace.png`). The task-space law lifts the beam 5 cm and
holds it for 2 s with the same ~30 percent under-press on every fingertip, then the beam creeps
down; the null law lifts for 0.2 s and drops it.

## 9. Commands

```bash
EP=outputs/play2perfect/episodes/tight_insertion/<stamp>/ep_0000
python play2perfect_force_controller/run_tracking.py --episode $EP --exact-replay --no-render          # regression
python play2perfect_force_controller/run_tracking.py --episode $EP --force-law null --no-render         # replay baseline
python play2perfect_force_controller/run_tracking.py --episode $EP                                       # replay + law
python play2perfect_force_controller/run_tracking.py --episode $EP --policy oracle --chunk 12 --horizon 12 --no-render
python play2perfect_force_controller/evaluate_controller.py --policy oracle --laws null task_space --tag o12pos --extra --chunk 12 --horizon 12 --state-source joint_pos
python play2perfect_force_controller/plot_force_comparison.py <rollout dir> <rollout dir> --out forces.png
python play2perfect_force_controller/render_force_video.py <rollout dir>          # rollout_force_tracking.mp4
python play2perfect_force_controller/compose_task_video.py --a <exact cmd replay> --b <reached-state replay> --c <reached states + law> --out abc.mp4
# full evaluation rerun (one Kit process per rollout, ~90 min) + share folder
python play2perfect_force_controller/evaluate_controller.py --laws null --render-laws null --tag exactvid --extra --exact-replay
python play2perfect_force_controller/evaluate_controller.py --laws null --render-laws null --tag posvid
python play2perfect_force_controller/evaluate_controller.py --laws null task_space --tag eval
python play2perfect_force_controller/evaluate_controller.py --policy oracle --laws null task_space --tag o12pos --extra --chunk 12 --horizon 12 --state-source joint_pos
python play2perfect_force_controller/package_results.py
```
