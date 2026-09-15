# play2perfect_force_controller — the hybrid force controller on contact-rich assembly

Plain-language walkthrough of how the policy and the controller fit together, and how to read the
force plots: [PIPELINE.md](PIPELINE.md).

The same middle layer as [`force_controller/`](../force_controller/README.md) (task-space hybrid
force / contact-point law, adaptive feedforward, friction-cone cap — read that README and
[GUIDE.md](../force_controller/GUIDE.md) for the method), with a different source of **perfect
data**: instead of the Kinova + LEAP video replays (pick-and-place only), episodes are recorded by
rolling the released [play2perfect](https://github.com/kushal2000/play2perfect) RL checkpoints
out in their own Isaac Lab scene — a KUKA iiwa14 + Sharpa hand doing tight insertion (0.5 mm
L-peg), two-step beam assembly and screwing.

```
   play2perfect checkpoint ──► collect_episodes.py ──► outputs/play2perfect/episodes/<problem>/<stamp>/ep_XXXX/
   (RL policy, 60 Hz)           (their env + our            replay_data.npz  summary.json  rollout.mp4 ...
                                 fingertip sensors)                 │
                                                                    │  "perfect data": robot states,
                                                                    │  fingertip force vectors,
                                                                    ▼  contact points, per step
   ChunkedReplayPolicy ──chunks──► HybridForceMiddleLayer ──PD targets──► run_tracking.py (sim_runner)
   (stand-in for the BC policy)    (config / middle_layer, unchanged)      outputs/play2perfect/force_controller/<run>/<stamp>/
```

## Setup (once)

```bash
git clone https://github.com/kushal2000/play2perfect ~/play2perfect      # or set PLAY2PERFECT_ROOT
cd ~/play2perfect && python download_checkpoints.py                       # pretrained_assembly/<problem>/model.pth
conda activate env_isaaclab            # Isaac Sim 5.1 / Isaac Lab 2.3.2.post1 == what play2perfect pins
```

Nothing is pip-installed: `p2p_paths.bootstrap()` puts the checkout and its **vendored rl_games
fork** (the checkpoints need the fork's SAPG model classes) at the front of `sys.path`.

### Which hand (`ISAACSIMENVS_HAND`)

The pipeline runs on either hand on the iiwa14 flange, selected by the same environment variable
play2perfect uses, read at import time:

| | `sharpa` (default) | `xhand` |
| --- | --- | --- |
| hand | Sharpa left, 22 joints (29 total) | RobotEra XHAND1 right, 12 joints (19 total) |
| task id | `Isaacsimenvs-PreciseAssembly-Direct-v0` | `Isaacsimenvs-PreciseAssembly-XHand-Direct-v0` |
| policies | `~/play2perfect/pretrained_assembly/<problem>/model.pth` (released) | `~/play2perfect/pretrained_assembly_xhand/<problem>/model.pth` (trained here) |
| recordings | `outputs/play2perfect/episodes/` | `outputs/play2perfect/episodes_xhand/` |

```bash
export ISAACSIMENVS_HAND=xhand      # before running anything below; unset == sharpa
```

Everything else is identical: the same recorder, the same middle layer, the same evaluation. The
fingertip order is by finger role (index, middle, ring, thumb, pinky) on both hands, so a colour
means the same finger in every plot. The XHand PD stiffness is `tau_max / ISAACSIMENVS_XHAND_ERR_RAD`
(default 0.3 rad) and **must** match the value the episodes were recorded with, because the
task-space law divides by exactly these numbers.

## Commands

```bash
cd ~/ZeroFACT
# 1. record perfect data (one Kit process per problem; rendering on by default)
python play2perfect_force_controller/collect_episodes.py --problem tight_insertion --episodes 5
#    problems: tight_insertion | beam_assembly_step1 | beam_assembly_step2 | screwing
EP=outputs/play2perfect/episodes/tight_insertion/<stamp>/ep_0000

# 2. validation ladder, in this order
python play2perfect_force_controller/offline_check.py --episode $EP                          # no sim
python play2perfect_force_controller/run_tracking.py --episode $EP --exact-replay --no-render  # must reproduce
python play2perfect_force_controller/run_tracking.py --episode $EP --force-law null --no-render # baseline
python play2perfect_force_controller/run_tracking.py --episode $EP                            # the controller

# 3. closed-loop evaluation: the RL policy itself serves the chunks (re-planned in the simulator
#    every --chunk frames, see "Oracle chunk policy" below); --exact-replay with --chunk 1 is the
#    restore-floor test, the other two are the null baseline and the controller
python play2perfect_force_controller/run_tracking.py --episode $EP --policy oracle --exact-replay --chunk 1 --horizon 1 --no-render
python play2perfect_force_controller/run_tracking.py --episode $EP --policy oracle --force-law null --chunk 12 --horizon 12 --no-render
python play2perfect_force_controller/run_tracking.py --episode $EP --policy oracle --chunk 12 --horizon 12
#    over all perfect episodes (report: outputs/play2perfect/force_controller/evaluation_<stamp>.md)
python play2perfect_force_controller/evaluate_controller.py --policy oracle --laws null task_space --tag oracle12 --extra --chunk 12 --horizon 12

# 4. re-plot a finished rollout without the simulator
python play2perfect_force_controller/replot.py outputs/play2perfect/force_controller/<run>/<stamp>

# 5. when a rollout under-presses: does the statics map even point the right way at this grasp?
python play2perfect_force_controller/probe_contact_map.py --episode $EP --frame 75
```

Shared simulation flags (`launch.py`): `--no-render`, `--gui`, `--video-fps`, `--seed`,
`--max-contact-points` (contact-point readback capacity per fingertip sensor, default 256 — the
SDF-collision parts of the beam and screwing tasks report many contact points and overflowing the
buffer trips a PhysX CUDA assert, which is how the first beam/screwing collections died at 64).

Every controller flag of `force_controller/run_tracking.py` exists here with the same name
(`--chunk/--horizon`, `--force-law`, `--task-kp/--task-ki`, `--point-kp`, `--cone-deg`,
`--allow-arm-offset`, `--offset-clip-rad`, ...).

## Files

| file | role | needs Isaac? |
|---|---|---|
| `robot_spec.py` | hand specs (Sharpa / XHand): joint names, PD stiffness, fingertip order/colours/pad offsets, timing constants | no |
| `p2p_paths.py` | where the play2perfect checkout is; `bootstrap()` makes it (and its rl_games fork) importable | no |
| `launch.py` | shared CLI flags + Isaac Sim launch settings (the machine's camera workaround, display fix) | no |
| `config.py` `episode.py` `policy.py` `middle_layer.py` `metrics.py` `plots.py` `replot.py` `offline_check.py` | **the controller, copied from `force_controller/`** — see "What differs" | no |
| `p2p_env.py` | `AssemblyBench`: play2perfect's `PreciseAssemblyEnv` + fingertip contact sensors + demo camera + recording hook + initial-state snapshot/restore; `load_policy()` for the checkpoints | yes |
| `p2p_vis.py` | per-fingertip force arrows, demo camera | yes |
| `collect_episodes.py` | CLI: run the checkpoint, record episodes in the `replay_data.npz` layout | yes |
| `sim_runner.py` | closed-loop force-tracking rollout in the bench scene | yes |
| `render_force_video.py` | rollout video with a live force-tracking plot (camera left; target dashed vs measured solid per fingertip, time cursor, live numbers, object height right); made automatically by `run_tracking.py` for rendered rollouts as `rollout_force_tracking.mp4` | no |
| `plot_force_comparison.py` | overlay force target (dashed) vs measured (solid) per fingertip for several rollouts of one episode, plus object height | no |
| `lead_diagnostic.py` | closed-loop rollouts: the law's offsets vs the policy's own preload per finger-joint group | no |
| `oracle_policy.py` | `OracleChunkPolicy`: the RL policy + the simulator as the middle layer's chunk policy (snapshot, roll the policy `horizon` frames, restore, execute the chunk) | yes |
| `evaluate_controller.py` | run the laws over every perfect episode (one Kit process each) and tabulate the outcomes | no (spawns) |
| `run_tracking.py` | CLI launcher: rollout + saving + plots + videos + metrics + exact-replay diff | yes |
| `probe_contact_map.py` | diagnostic: replay to a grasp frame, nudge one joint, measure the force change, compare with the law's `K⁻¹Jᵀf` direction | yes |
| `collect_all.sh` | record N episodes for each of the four problems, one Kit process each | yes |
| `bc_dataset.py` | BC dataset conventions: acceptance test (`validate_records` / `validate_episode_dir`), fake-motor constants, `dataset_index.json` builder, `load_episode`, review sheets | no |
| `collect_bc_dataset.py` | driver: N verified-successful episodes per problem, one cold-start Kit process per seed, resumable, `progress.json` / `rejected.json` | no (spawns) |

## What differs from `force_controller/` (and what does not)

The controller modules are verbatim copies except for four robot-specific lines, each marked
`[play2perfect]`:

* `config.py`: `ARM_JOINT_NAMES` / `default_joint_stiffness` come from `robot_spec` (iiwa14 +
  Sharpa instead of Kinova + LEAP); the exact-replay preset uses **latency 0**; chunk/horizon
  defaults are 12 / 24 frames.
* `plots.py`: fingertip colours/labels from `robot_spec` (five fingers).
* `offline_check.py`: exact-replay self-test uses latency 0; default chunk/horizon.

Everything about the law, the sensing contract (net pad force + pad centroid only), the metrics
and the plots is unchanged, so results are comparable across the two robots.

**Conventions that changed with the data source**

* A *frame* is one 60 Hz policy step = **2 physics steps** of 1/120 s (was 0.1 s = 12 steps).
  `steps_per_frame` is read from the episode; chunk/horizon flags are in these frames
  (12 / 24 = re-plan every 0.2 s, predict 0.4 s; use 48 / 96 for the old 0.8 s / 1.6 s).
* The env applies the policy's joint target at both substeps of a frame, so there is **no
  stale-target lag**: exact replay = `joint_target` + `hold` + latency 0.
* Five fingertips (`left_{index,middle,ring,thumb,pinky}_DP`; the pad and elastomer links are
  merged into the distal phalanx by the URDF importer). The Sharpa finger PD gains are tiny
  (0.9–13 N·m/rad vs LEAP's 350), so `K⁻¹Jᵀf` yields large offsets per newton — watch
  `offset_clip_rad` (0.12) in the first rollouts.
* The contact sensors attribute forces to `object` (the inserted part), `hole` (the receptacle)
  and `table`; all three are dynamic rigid bodies here, so the table contact *does* report a
  patch centre (in the LEAP scenes it was a static collider and did not).
* Objects recorded every frame: `object`, `hole`, `table`, `goal_viz` (the goal marker, a
  collision-free copy of the part at the current subgoal). In controller rollouts the marker is
  pinned at the recording's FINAL goal pose; replaying the recorded subgoal schedule made it look
  like a phantom part sinking on its own whenever the real part was elsewhere.
* **Task success** is scored the way the env scores it (`metrics.task_success_metrics`): the
  part's fixed-size box-corner keypoints within 15 mm (insertion tolerance × keypoint scale) of the
  final goal for 10 consecutive frames, retract = fingertips then > 10 cm away with the part still
  within 7.5 mm. `run_tracking.py` prints it and `evaluate_controller.py` tabulates it
  (`INSERTED`); the slip/drop metrics only say whether the part left the palm.
* The episode's initial state (joint state + targets, object/hole/table/goal root states) is
  stored in `summary.json["init_state"]` and restored before a rollout; the RL policy's actions
  (`action`, canonical order) and the env's `successes` / `retract_phase` /
  `keypoints_max_dist` per frame are in `replay_data.npz` for analysis.

## Oracle chunk policy (closed-loop evaluation)

`ChunkedReplayPolicy` serves the recording open loop, which is a fair test only for tasks that
tolerate small deviations. These do not: the recorded commands with one physics step of smoothing
(`--state-source joint_target`, default linear interpolation, null law) insert 2 of 12 episodes,
while held per frame they reproduce the recording bit for bit — tight insertion and beam step 1
are chaotic under open-loop replay, and the RL policy only succeeds because it re-observes the
object keypoints every 16 ms. The middle layer sees joints and fingertip forces only, so it cannot
be scored on pose divergence it never observes.

`--policy oracle` (`oracle_policy.OracleChunkPolicy`) gives the position layer the feedback a
learned chunk policy would have. At every chunk boundary it snapshots the world (physics state,
the env's bookkeeping, the sensor buffers, the RNG, the policy's LSTM state), rolls the released RL
policy forward `--horizon` frames inside the simulator — recording per frame the command it
applied, the joint state it reached, the net fingertip forces and the pad contact centroids — and
puts the world back. The middle layer then executes the chunk (`--state-source joint_target`: the
commands, i.e. the policy predicts what it would command; `joint_pos`: the reached states, i.e.
the BC premise where the force law has to supply the preload). Between boundaries the env's task
logic keeps running (`AssemblyBench.post_frame_bookkeeping`: sub-goal advance, retract phase,
terminations — the goal marker therefore moves through the sub-goals as in the recording, not
pinned) and the policy's recurrent state is advanced on the executed observations. The env is
reset with the recording's seed so the start is the recording's (the runner prints whether the
same-seed reset reproduced every key of `init_state`). The rollout ends when the env ends the
episode (success + retract, drop, fall, timeout) or after the recording's length plus
`--grace-frames`.

The force targets are therefore not predicted by a network: they are what the policy would have
produced over the next `horizon` frames, re-planned from the current state — the same quantity the
replay policy takes from the recording. It is optimistic (its chunks are consistent with the true
dynamics) and needs the simulator to plan, so it is an evaluation device, not a deployable policy;
its rollouts are also the (observation → state chunk + force chunk) pairs a learned chunk policy
would be trained on.

Caveat: PhysX's contact cache cannot be read back, so a restore is exact for every buffer we can
read but the first solver step afterwards starts from cold contacts. `--policy oracle
--exact-replay --chunk 1 --horizon 1` (commands, held, null law, re-planned every frame) measures
that floor: with a perfect restore it would reproduce the recording bit for bit.

Reports label these rollouts `oracle(C<chunk>):<law>`; the `env goals` column is the env's own
verdict (`successes/max_goals`, `+R` when the retract succeeded). `/cmap` and `/gate` mark the
two law variants below.

### Two law variants from the closed-loop findings (both opt-in, both zero-shot)

* **Constrained force→offset map** (`--map constrained`, `ForceLawConfig.map`). The closed-form
  statics map `dq = -K⁻¹Jᵀf` followed by a joint-by-joint clip was at its clip in 70–100 % of the
  engaged steps of the closed-loop rollouts, and raising the clip did not turn into force: the
  clipped offset delivers a force pointing elsewhere (out of the friction cone) and its torque
  outside the contact Jacobian's row space only moves the finger. The constrained map solves the
  same statics model as a bounded least squares per fingertip — the offset whose delivered force
  is closest to the command, plus a penalty on the null-space torque (`map_null_arm_m`), within
  the per-joint bounds (the clip tightened by the effort limit `τ_max/K`), with joints pinned at
  a position limit treated as rigid. It equals the statics map exactly when no bound binds
  (synthetic test: 3e-5 relative difference); when bounds bind it cuts the force residual from
  0.77 to 0.56 of the command and the self-motion torque from 0.16 to 0.10 N·m against the
  clipped map. 0.08 ms per fingertip.
* **Contact-point gate** (`--point-engage-n`, `--point-steady-rate`). The point channel pulls the
  pad toward the predicted contact point; during a re-grasp the plan is moving that point and
  the channel fights the position layer (screwing lost 2 of 3 completions to the law when the
  commands already carried the preload). The gate lets the channel act only while the plan says
  the contact is established and steady: predicted |f| above `point_engage_n` and its relative
  rate below `point_steady_rate`.

`lead_diagnostic.py` compares, on closed-loop rollouts, the law's offsets with the policy's own
preload at the same states (the plan's command minus the state it reached), per finger-joint
group — the direct check of *where* the law puts its offsets.

## BC dataset (`collect_bc_dataset.py`, 2026-09-14)

Behaviour-cloning data = verified-successful rollouts of the released Sharpa checkpoints with
clean RGB frames and the "fake tactile" channels, under `outputs/play2perfect/bc_dataset/`:

```bash
# 50 accepted episodes per problem, seeds from 1000, one cold-start Kit process per seed, resumable
python play2perfect_force_controller/collect_bc_dataset.py --target 50 --seed0 1000
# rebuild dataset_index.json / README.md / <problem>/review_sheet.png without collecting
python play2perfect_force_controller/collect_bc_dataset.py --review-only
# one episode by hand (what the driver runs per seed)
python play2perfect_force_controller/collect_episodes.py --problem screwing --episodes 1 --seed 1234 \
    --strict --max-attempts 1 --flat --out-dir /tmp/screw_1234 --image-every 2
```

Per episode (`<problem>/seed_<k>/`): the usual `replay_data.npz` + `summary.json` + plots, plus
`rgb_images/<state index>.png` (demo camera 640x480 every 2nd state = 30 Hz, s = 0 is the reset
state, **no force arrows, goal marker hidden**; `--goal-marker translucent --force-arrows` restores
the video look), `rollout.mp4` and `rollout_with_forces.mp4` (small review videos), and these
extra arrays (per frame and, as `*_steps [T, 2, ...]`, per 120 Hz substep): `obs_policy [T, 141]`
(field layout in `summary.bc.obs_fields`), `applied_torque` / `computed_torque` (implicit-PD drive
torque, clipped / unclipped), `joint_torque_measured` (PhysX projected joint force = the reaction
along the joint axis), `motor_current = applied_torque / motor_kt` (kt = effort limit / rated
current, 10 A arm / 1 A hand - FAKE constants), `joint_cmd_err = joint_target - joint_pos`,
`joint_wrench_b [T, 30, 6]` (incoming joint wrench per body), and the static `joint_stiffness`,
`joint_damping`, `joint_armature`, `joint_pos_limits`, `joint_effort_limits`. Full conventions
(state index vs frame index, substep order) are in the `bc_dataset.py` module docstring;
`bc_dataset.load_episode()` returns state-aligned arrays.

**Acceptance (`--strict`)** = env verdict (every sub-goal + retract) AND `validate_records`: clean
termination (`max_successes` only), part within 7.5 mm keypoint distance of the final goal at the
last frame, the insertion test re-run offline against the recorded final goal pose, physics sanity
(joint speed <= 40 rad/s, fingertip force <= 400 N, part speed <= 4 m/s, lifted >= 2 cm, >= 60
frames, finite), image sanity (files match `image_state_index`, not black, not frozen). The driver
re-runs it from the saved files (`validate_episode_dir`); rejected seeds go to
`<problem>/rejected.json` with the reason, Kit hangs / PhysX errors are retried (`--retries`).

**Speed on the shared GPU (2026-09-14, XHand screwing training at 86 % util alongside):** ~4.5
frames/s, i.e. a 116-frame tight_insertion episode takes ~30 s of rollout + ~35 s Kit start-up;
the renderer is NOT the bottleneck (`--render-mode capture` renders only at capture time, FXAA;
`step` = per-substep rendering with TAA), the GPU time-slicing is. Recording reads are batched into
three device syncs per frame (`scene_frame`, `tactile_frame`) - bit-identical arrays to the old
per-tensor readers.

## Signal filtering (`signal_filtering.py`, `sg_filter_study.py`, 2026-09-15)

The recorded contact forces, PhysX measured joint torques and joint wrenches carry 14-15 % of their
power above 10 Hz (120 Hz solver contact chatter); joint positions, the state-command difference
and object poses carry under 1 % and are left alone. `signal_filtering.py` is the filter library
(pure numpy/scipy) and `sg_filter_study.py` measures every candidate on real episodes:

```bash
python play2perfect_force_controller/sg_filter_study.py              # figures + REPORT.md
python play2perfect_force_controller/sg_filter_study.py --self-test  # 17 property checks
```

Outputs land in `outputs/play2perfect/bc_dataset/filtering/` (raw vs filtered per task, filter
frequency responses over the measured noise spectrum, parameter sweep, causal comparison, REPORT.md).
**The dataset on disk is untouched**; apply `signal_filtering.filter_arrays()` at load time.

Three findings decide the defaults:

1. **Filter the force VECTOR, never `|F|`.** Smoothing the magnitude undershoots below zero on
   4-6 % of samples (to -3 N) at every contact onset and release, because the Savitzky-Golay kernel
   has negative side lobes. Filtering xyz and taking the norm afterwards gives 0 % impossible values
   with peaks within 2 % (`filtered_force_magnitude`, `vector_vs_magnitude.png`).
2. **A causal Savitzky-Golay amplifies this data's noise band.** It is exact on a linear trend, so
   it has zero lag at DC, but its magnitude response overshoots unity before rolling off: window 9
   peaks at 1.40 and windows 15-31 at 1.54-1.66 around 5-8 Hz. Measured on the data, causal (9,2)
   raises 4-10 Hz power to 1.36x. For anything the policy reads online use `butter_lowpass` at
   6-10 Hz or `ema` with a 2-3 frame time constant, which never exceed unity gain.
3. **Centered filtering leaks backwards in time.** Zero lag, but contact onset moves about one frame
   earlier because the window mixes in future samples. Fine for labels and analysis, wrong as a
   policy input feature.

`DEFAULT_FILTER_SPEC` is therefore centered, window 7 (117 ms), order 2 on the contact channels
(89 % of the >10 Hz power removed for 4.5 % peak loss, the knee of the sweep) and window 5 on the
drive torque / current / velocity. Order 3 is identical to order 2 for centered smoothing.

## What an episode contains (`replay_data.npz`)

Same keys as the LEAP replays (`episode.ReplayEpisode.load` is unchanged): `joint_pos`,
`joint_vel`, `joint_target` `[T, 29]`; `contact_force` `[T, 5, 3]`, `contact_force_steps`
`[T, 2, 5, 3]`, `contact_object_force_steps` `[T, 2, 5, 3, 3]`, `contact_point_w` `[T, 5, 3, 3]`;
`object_pos/quat/lin_vel` `[T, 4, ·]`; `body_pos/quat` for palm + fingertips. `summary.json`
carries `outcome` (`insertion_complete`, `retract_success`, termination reason, max force on the
part), `key_frames` (first contact, lift, each subgoal, retract) and `init_state`.

"Perfect" = the policy reached every insertion subgoal (`outcome.perfect`); with `--only-perfect`
(what `collect_all.sh` passes) an episode is kept only if it is perfect *and* the retract
succeeded, and the collector keeps rolling until `--episodes` such episodes exist. Episodes
shorter than `--min-frames` (unstable start) are always discarded; `collection_summary.json`
lists what was kept. Controller evaluation should use perfect episodes only.

**Determinism (measured 2026-09-08).** The simulator is bit-deterministic only from a cold
start: the first episode of a process replays bit-exactly (`--exact-replay` shows 0.0 on every
physics key), but episodes recorded after an in-process auto-reset carry hidden PhysX state and do
NOT (joint positions off by up to 0.28 rad, contact forces by tens of newtons). Use
`collect_all.sh`, which starts one Kit process per episode with its own `--seed`, whenever a
recording must be replayable; `--episodes N` in one process is fine for a quick look at the
policy. The same effect makes repeated in-process replays (`probe_contact_map.py`) disagree.
