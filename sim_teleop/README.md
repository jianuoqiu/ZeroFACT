# sim_teleop — record a human hand with a camera, replay it as robot data in sim

Purpose: **more data**. You record a human hand doing a task with a camera, and this
pipeline turns the video into a Kinova + LEAP trajectory that plays inside the
validated Isaac Lab scene — producing the same `replay_data.npz` episodes the rest
of the repo (including `force_controller/`) already consumes.

Nothing heavy is re-implemented here. The hand-capture and retargeting stacks that
already work on this machine are reused in place; this folder adds the glue, the
depth correction wrapper, the packaging into this repo's run format, and the
in-sim player.

```
video.mp4 ──1──▶ VIPE ──2──▶ Dyn-HaMR ──3──▶ depth-corrected keypoints ──4──▶ IK ──5──▶ packaged run ──6──▶ sim
             camera+depth    MANO hand,      (T,21,3) in robot base        Kinova+LEAP    data/runs/       replay +
             preprocessing   world frame      frame                        joints         teleop_<name>    record
```

| stage | script | conda env | what it does |
|---|---|---|---|
| 1+2 | `run_hand_capture.sh <video>` | `vipe`, `dynhamr5090` | camera/depth preprocessing, then Dyn-HaMR: a world-grounded MANO hand sequence from the video |
| 3a | `make_hand_traj.py --seq S --export-world` | `dynhamr5090` (auto) | export world-frame (T,21,3) joints for the depth measurement |
| 3b | `estimate_depth_scale.py` | any (numpy+cv2) | **the depth correction**: Dyn-HaMR is monocular, so the hand sits a constant factor too far/near; measure it against the recording's depth PNGs (`median(Z_sensor / Z_rgb)` at wrist+knuckles) |
| 3c+4 | `make_hand_traj.py --seq S --camera-pose-json … --depth-scale-json …` | `dynhamr5090`, `vid2sim2real` (auto) | MANO → robot-base keypoints (AprilTag calibration + depth correction), then mink/mujoco IK → `retarget_kinova_leap.json` |
| 5 | `package_run.py --traj … --name S` | any | build `data/runs/teleop_S/` in the standard run layout |
| 6 | `teleop_player.py --run teleop_S --gui` | `env_isaaclab` | quick look: stream the joints into the loaded scene |
| 6' | `scripts/replay_trajectory.py --run teleop_S` | `env_isaaclab` | **the data collection step**: replay with the validated recorder → `replay_data.npz`, contact forces, videos, plots |

Stages 1-4 can also be replaced by **live teleop** (camera + sim windows, keyboard-gated
recording): `./sim_teleop/run_live_teleop.sh <name>` — see the section below.

## Quick start (a new video, same table + camera setup as the recordings)

```bash
# 1+2. video -> world-frame MANO hand   (heavy: GPU, several minutes)
sim_teleop/run_hand_capture.sh ~/videos/mug_pick.mp4

# 3a. export world joints for the depth measurement
python sim_teleop/make_hand_traj.py --seq mug_pick --export-world

# 3b. measure the depth correction against the recording's RealSense depth PNGs
python sim_teleop/estimate_depth_scale.py \
    --world-npy outputs/sim_teleop/mug_pick/hand_world.npy \
    --npz outputs/sim_teleop/mug_pick/dynhamr_result/smooth_fit/*_world_results.npz \
    --depth-dir <recording>/depth

# 3c+4. keypoints (calibrated + corrected) -> Kinova+LEAP joints
python sim_teleop/make_hand_traj.py --seq mug_pick \
    --camera-pose-json data/runs/run_2026-05-15_17-55-22/scene/camera_frame_pose.json
    # (depth_scale.json in the work dir is picked up automatically)

# 5. package as a run; borrow the donor's calibration and (optionally) its objects
python sim_teleop/package_run.py \
    --traj outputs/sim_teleop/mug_pick/retarget_kinova_leap.json \
    --name mug_pick --objects-from run_2026-05-15_17-55-22

# 6. look at it / collect the data
conda activate env_isaaclab
python sim_teleop/teleop_player.py --run teleop_mug_pick --gui
python scripts/replay_trajectory.py --run teleop_mug_pick        # -> outputs/teleop_mug_pick/...
```

## Live teleop (camera window + sim window + keyboard recording)

Instead of processing a video offline (stages 1-4 above), you can drive the sim robot with
your hand **live** and record episodes with the keyboard:

```bash
./sim_teleop/run_live_teleop.sh my_episode --objects-from run_2026-05-15_17-55-22
```

This opens two windows:

- **camera window** (`live_teleop.py`, conda env `cam`): the RealSense feed with the
  detected hand skeleton, depth-anchor dots (green = valid), and a status banner;
- **sim window** (`live_sim_view.py`, conda env `env_isaaclab`): the validated scene with
  the robot mirroring your hand in real time (donor objects shown welded in place as a
  placement reference).

Control is **delta (clutch) based by default** — absolute hand placement doesn't matter
and *no camera calibration is needed*: the robot starts at a proven initial pose (frame 0
of `--init-from`, default the scene donor's trajectory) and follows your hand's *relative*
motion once you engage. The camera can sit anywhere that sees your hand — **you don't
need the robot table at all**: operating at your own computer desk against the sim
objects in the viewer is the intended use (your hand grips air; contacts happen in sim,
and closing "through" a virtual surface is fine — it simply commands grip force in the
replay). You close the loop by watching the sim window; use the clutch (`t`) to cover
the robot's large workspace from a small desk-chair range of motion. Engaging never makes the robot jump (verified: arm moves
<5° at the engage instant — only the fingers glide to your actual curl, rate-capped).

**Conditions matter more than anything else — and the EST badge tells you live whether
they are met.** The camera window shows `EST GOOD/OK/BAD` (top right): a verdict on the
3D hand estimate itself, calibrated on the known-good rig recording (its whole session
reads GOOD; real sessions that produced unusable data read OK/BAD). Only record while it
is GOOD — the 2D skeleton overlay can look perfect while the 3D estimate is garbage, so
trust the badge, not the drawing. When it is BAD it prints the cure, which follows from
which half is failing: high *reprojection error* (MediaPipe's own 2D/3D inconsistency —
measured 2–2.5× worse in real live sessions than on the rig) means **lighting, exposure, background
or motion** — light the SCENE evenly (over a dark background, auto-exposure over-exposes and blurs
the hand - brightening the whole area beats a spotlight on the hand), no bright
window/monitor behind you, slower moves; a high *anchor
fit* with good reprojection means **viewing geometry** — face the palm or the back of
the hand to the camera 40–60 cm away, never fingers-at-the-lens, never edge-on (occluded
fingers get *hallucinated*, usually as a curl, unrecoverable downstream). Roll the
sleeve up (the wrist is a depth anchor). The proven reference placement is the original
rig's: **above and in front, looking down ~45° at the back of the hand** (`--view top`).
On save the take prints its estimation-health share (also in `live_meta.json`) — below
~60% GOOD, re-record rather than hoping. Delta mode needs no recalibration after moving
the camera: just restart.

In the **camera window**:

| key | action |
|---|---|
| `t` | engage the clutch: robot follows your relative motion from its current pose |
| `t` again | release: robot holds while you reposition your hand (like lifting a mouse), `t` re-engages — works mid-recording too |
| `SPACE` | start recording (episode frame 0 = the pose at that moment) |
| `SPACE` again | stop -> trajectory saved + packaged as `data/runs/teleop_<name>` |
| `q` / `ESC` | quit (an unfinished recording is discarded) |
| `m` | mirror the preview (display only) |

**Floating-hand collection (`--floating`, recommended):** no arm anywhere in the live
loop. The sim shows a **free-floating LEAP hand** that mirrors your hand 1:1 — wrist pose
applied directly (no arm IK, no reachability, no arm-induced artifacts), fingers solved
against a fixed palm.

**Grasping in the preview (`--dynamic-objects`):** by default the preview objects are a
non-colliding placement reference — the hand passes through them (a stable pose preview,
but no contact feedback). For manipulation tasks (picking, screwing) where you need to
*see* the grasp work, add `--dynamic-objects`: the objects become dynamic, rest on a solid
table, and can be grasped and moved by the hand for live contact feedback. This is
best-effort live physics (the hand root is teleported to your wrist, so a grasp may slip
under fast motion); the ground-truth contact is still the validated replay. Without the
flag the hand can't blow up on an immovable object; with it, dynamic objects yield so
they can't either. Example:
```bash
./sim_teleop/run_live_teleop.sh grip1 --glove live --source realsense --wrist-tag \
    --tag-size 0.025 --tag-id 0 --floating --dynamic-objects \
    --objects-from run_2026-05-15_17-55-22
``` The recording stores the raw wrist-pose series + finger joints
(`outputs/sim_teleop/<name>/float_traj.json`); on save, `retarget_float.py` automatically
fits the 7-DoF arm to the wrist series offline (warm-started, hardware-rate-capped,
frame 0 fully converged) and reports the wrist tracking error — then packaging, the
validated replay and force_controller run unchanged. Validated end-to-end: the benchmark
video collected in floating mode retargets with 0.12 cm mean wrist error and replays with
contacts on the apple at the human's grasp frames. If the error report warns, the demo
left the arm's workspace — redo it closer in. One-time asset build:
`conda run -n env_isaaclab python sim_teleop/make_float_hand.py`.

```bash
./sim_teleop/run_live_teleop.sh my_episode --floating --objects-from run_2026-05-15_17-55-22 --replay
```

**Wuji-glove input (`--glove`)** replaces vision for the hand itself — the answer to the
estimation problems above at their source. The glove streams a 21-joint wrist-local hand
skeleton at 120 Hz (pure finger articulation, flip-free by construction) and a palm IMU
whose fused world orientation is gravity-aligned z-up like the robot base; the constant
IMU-mount rotation cancels in delta control, so no calibration is needed.

**Wrist orientation vs translation — an important hardware limit.** The glove reports the
wrist's *orientation* (IMU) but NOT its *position*: an IMU measures rotation and
acceleration, and turning acceleration into position needs double integration, which
drifts unusably (measured on your own recordings: ~7 cm after 2 s, ~45 cm after 5 s — a
free-floating position estimate is worthless for grasping). So position must come from
somewhere absolute. Two ways to run it:

- **Glove + AprilTag on the camera (full 6-DoF wrist, RELIABLE)** — `--source realsense
  --wrist-tag`. Wrist *translation* comes from an **AprilTag on the back of the glove**:
  `dt_apriltags` (the same detector as the offline scene calibration) reads its 6-DoF pose
  directly (sub-3 mm depth verified in sim), immune to hand shape/orientation. This is the
  mode for real data collection. The default family is **`tagStandard41h12`** — the same
  family as the calibration tags, so you can reuse one of those (OpenCV's aruco can't
  detect 41h12, which is why we use dt_apriltags). Measure the tag's **black-square** side
  and pass it as `--tag-size` (metres). Tape the tag **flat on a rigid backing** (a soft
  glove lets the paper crumple, and a bent tag won't detect) near the wrist, then:
  ```bash
  ./sim_teleop/run_live_teleop.sh glove_try --glove live --source realsense \
      --wrist-tag --tag-size 0.053 \
      --floating --dynamic-objects --objects-from run_2026-05-15_17-55-22
  ```
  `--tag-size` defaults to 0.053 (a wrong value scales the wrist motion). `--tag-family`
  defaults to `tagStandard41h12` (use e.g. `tag36h11` for that family). Omit `--tag-id` to
  track whatever single tag is on the glove, or pass `--tag-id N` if several tags are in
  view. The status line shows `wrist tag: SEEN/LOST`, the preview marks it, and the tag's
  fixed offset from your wrist cancels in clutch control (no calibration — just keep it
  flat, facing the camera, within ~0.6–0.7 m). To *print* a fresh tag instead of reusing a
  calibration one, `sim_teleop/make_wrist_tag.py` makes a true-to-scale PDF but only for
  the OpenCV families (e.g. `--dict DICT_APRILTAG_36h11`, then run with
  `--tag-family tag36h11`); it cannot generate 41h12.

- **Glove + camera depth blob (full 6-DoF, coarse)** — `--source realsense` without
  `--wrist-tag`. Wrist translation from the nearest hand-sized depth blob: appearance/
  lighting independent, no tag needed, but it tracks a hand-region *centroid* that shifts
  with hand orientation and forearm visibility — **measured unreliable in practice**. Use
  `--wrist-tag` for real data; the blob is a no-print fallback for a quick look.
- **Glove only (`--source none`)** — wrist *position is parked* (orientation + grasp
  only). Good for a first bring-up, useless for reaching. **This is why your wrist was
  fixed: the default `--glove` with no camera parks the wrist.**

The SDK lives in its own conda env (`wuji-sdk`), so the glove runs as a bridge process
(`wuji_bridge.py`) streaming JSON over UDP :5557; the launcher starts it for you:

```bash
# full 6-DoF wrist: glove (fingers + orientation) + RealSense depth (translation):
./sim_teleop/run_live_teleop.sh glove_try --glove live --source realsense \
    --floating --objects-from run_2026-05-15_17-55-22
# glove only (wrist position parked - fingers + orientation, first bring-up):
./sim_teleop/run_live_teleop.sh glove_try --glove live --floating --objects-from run_2026-05-15_17-55-22
# debug without hardware: replay a Wuji Studio .mcap through the identical path
./sim_teleop/run_live_teleop.sh glove_dbg --glove ~/tactile_glove/test_new_glove_right/<rec>.mcap --floating
```

For the depth-blob wrist, keep your gloved hand the **closest** thing to the RealSense
(it locks onto the nearest hand-sized blob), 40–80 cm out, and let the camera see it move.

The one free parameter is the **yaw** between the IMU's gyro-initialised world and the
robot base: after engaging (`t`), push your hand forward — if the robot goes sideways,
tap `[` / `]` (5° steps; each press re-anchors so the robot never jumps) until forward
is forward; `--imu-yaw <deg>` presets it. The EST badge shows glove-stream health
(rate/staleness) instead of vision quality. Validated end-to-end without hardware: real
recorded grasps (`~/tactile_glove/*.mcap`) replayed through the bridge reproduce the
IMU's net orientation to 0.2°, produce civil finger joints (no limit rails), retarget at
0.01 cm / 0.1° wrist fit, package, and replay in sim. Glove mode is delta/clutch only.

**Steering directions** come from `--view` (delta mode): `front` (default) assumes the
camera faces you — *push away from you = robot forward, up = up, your right = robot
right*, no matter how your palm is held at engage; rotating your hand in place rotates
the robot hand in place. `top` is the same idea for a camera above/behind looking down
(image up = forward). `hand` is the old palm-relative mapping (any camera placement, but
you must engage with your palm matching the robot's — measured to be the main steering
trap). The engage message prints the active direction cheat-sheet.

Every session also writes `outputs/sim_teleop/<name>/track_log.jsonl` — per-frame state,
fit residual, reprojection error and wrist positions — so a bad-feeling session can be
diagnosed from data afterwards.

The recorded 10 Hz trajectory is packaged through the normal `package_run.py`, so the
actual episode data still comes from the validated recorder (`--replay` runs it for you
after the session):

```bash
python scripts/replay_trajectory.py --run teleop_my_episode   # -> replay_data.npz etc.
```

which is exactly the format `force_controller` consumes (`ReplayEpisode.load` verified).

**Why live can look worse than the offline Dyn-HaMR pipeline, and the flip gate.**
Dyn-HaMR's `smooth_fit` optimizes one temporally-consistent MANO trajectory over the whole
video; MediaPipe is per-frame and occasionally *flips its 3D interpretation* of the hand
(palm-toward vs palm-away) with the 2D overlay still looking perfect — measured on a real
live session: 85–118° single-frame jumps of the palm frame (a physical wrist can't exceed
~600°/s) with zero dropouts, and since finger targets are expressed in that frame, each
flip slammed fingers into their joint limits. The tracker therefore gates the derived
wrist frame: an impossible jump is rejected and the last good pose held (status shows
`WRIST FLIP rejected`), a persistent new orientation is accepted only if the depth
anchors fit it clearly better or it survives ~0.5 s. Frequent flip messages mean the
view geometry is ambiguous — angle the palm mostly toward the camera and do NOT point
the fingers straight at the lens. Two further defenses make flips survivable:
**fingers are flip-immune by construction** — their targets are built in the hand's own
raw palm frame, so a whole-hand orientation flip cancels exactly (verified: a 110° flip
injected every 1.1 s changes index/middle/ring tips by 0.3 mm mean vs the clean run) —
and the **offline arm retarget repairs conceded flips** in the recorded wrist series (a
stint entered and exited by impossible >600°/s snaps that returns to the pre-entry pose
is slerp-bridged; `repaired N wrist-flip frames` in the report; 0 false repairs on the
clean benchmark, and on a real flip-polluted session it cut the wrist-fit error from
15.1° to 3.5° mean). A flip can therefore at worst briefly freeze or swivel the floating
hand's BASE in the viewer — finger shape and the packaged episode stay sane. (The
MediaPipe handedness label is ignored — it reads
"Left" on every frame of a right-hand video; a label/determinant-based mirror correction
was tried and measured harmful, since the raw geometry is right-handed on ~82% of frames
and the palm anchors are mirror-blind.)

How the live hand pose is measured: MediaPipe gives the 21-landmark hand (2D pixels + a
metric hand shape). The pose comes from `cv2.solvePnP` of that shape against its own 2D
landmarks (all 21 correspondences, previous-frame warm start) — chirality-safe, so the
palm can never mirror-flip and cross the fingers the way a knuckle-plane fit could; the
orientation is additionally SO(3)-smoothed and rate-capped, since a flat hand leaves one
rotation direction weakly constrained. Position is re-anchored every frame to the
RealSense **aligned depth** at the wrist + 4 MCP knuckles (the anchor joints
`estimate_depth_scale.py` trusts) — depth is still needed so the *relative* motion is
metric; only the person's hand-size factor is slow-locked. At engage, one rigid map
`M = B @ H0^-1` binds your current hand frame to the robot's current bracelet pose; every
frame all 21 points map through `M`, so translation, rotation and finger articulation
carry over 1:1. **Finger mapping** (`--finger-map`, default `knuckle`): only the four
LEAP fingertips are IK targets (position-only — the same links and solver as the offline
`retarget_leap`, which also weights fingertip orientation at 0). What differs is where
the targets are placed. The offline scheme (`--finger-map v2s2r`) uses the human tip
points directly (optionally scaled about the wrist): because a human palm is much
narrower than the LEAP palm (adjacent open-hand tips ~3.2 cm apart vs the LEAP's natural
4.5 cm), those targets constantly pull the fingers toward each other — measured on a real
open-hand frame: ~20° of MCP curl and ±8° of converging abduction, which with tracking
noise becomes visibly *crossing* fingers. The default `knuckle` map instead anchors each
fingertip at the LEAP finger's **own** knuckle and adds the human tip-from-knuckle
vector scaled by the finger-length ratio: the neutral spread is the LEAP palm's own
(open→open, no convergence, solved tip error ~0.1 cm), and the mapping is invariant to
hand size, so the auto `hand scale xN` (still printed) no longer affects finger shape.
On the benchmark video the old map only poked the apple (0.007 m displacement); the
knuckle map actually grasps and carries it. `--hand-scale <number|auto>` still applies
to the `v2s2r` map; absolute mode defaults it to 1.0. The IK is the same `MinkSolver` as `retarget_leap` (warm-started, 20
iters, ~15 ms/frame; the first solve after engaging runs to full convergence so recordings
never contain a catch-up ramp), and the output joints are rate-capped at
hardware-plausible speeds (Kinova ~1.2 rad/s) so redundant-arm branch flips cannot snap
the robot. Use your **right hand**; the MediaPipe left/right label is ignored (a top-down
view flips it constantly). The IK is **decoupled by default**: the arm solves for the
wrist target only, then the fingers solve with the bracelet anchored — so finger motion
can never move the arm (in the whole-body solve it measurably did: curling fingers
dragged the bracelet ~3 cm/frame with a still wrist; `--coupled-ik` restores the old
behaviour). The wrist-target orientation gets its own One Euro filter, since finger
curls shift the knuckle landmarks the wrist frame is built from.

`--mode absolute` restores the calibrated absolute mapping (the AprilTag formula of the
offline converter): the hand lands exactly where the calibration says, which matters only
when you are simultaneously manipulating **real** objects on the **real** calibrated
table. It then needs `--calib <camera_frame_pose.json>` of the current camera setup
(defaults to the donor run's — only valid if the camera/table have not moved).

Live-specific knobs (`live_teleop.py`): `--init-from <run>` (robot start pose = that
run's trajectory frame 0), `--smooth` One Euro filtering strength (speed-adaptive: a
static hand is heavily smoothed, real motion passes with little lag; raise above 1.0 if
the robot still trembles at rest),
`--num-iter`, `--record-fps`, `--serial` (RealSense S/N; the D435 used for the recordings
is 337122071053). The sim window: `--gui-render-interval N` renders every Nth physics
step (raise it if the viewer feels slow), `--dynamic-objects` / `--no-objects`.

No camera plugged in? Test the whole chain against a recorded RGB-D session:

```bash
./sim_teleop/run_live_teleop.sh t1 \
    --source ~/Video2Sim2Real_main/keyframe_detection_test/run_2026-05-15_17-52-54 \
    --auto-record --no-window        # drop the last two flags to use the windows/keys
```

(`teleop_livefull` in `data/runs/` is exactly that, recorded in `--mode absolute`: the
17-52-54 apple-grasp video pushed through the live tracker end-to-end, replayed with
contacts on the apple at frames 47-49. A canned video can't steer, so absolute mode is
the right way to *benchmark* tracking quality against a recording; delta mode is for a
human in the loop.)

## The three things you must get right

1. **Camera calibration (`--camera-pose-json`).** The AprilTag file
   (`camera_frame_pose.json`) tells the pipeline where the camera sits relative to
   the robot base. Reuse a donor run's file **only if the camera and table did not
   move** since that recording; otherwise redo the AprilTag measurement. Without
   it the converter guesses (up-axis right, yaw unknown) and warns.
2. **The depth correction.** Monocular Dyn-HaMR gets the hand's *shape and motion*
   right but its *distance* wrong by a global factor (measured ~1.13–1.19× on the
   prior recordings). Skipping it puts the hand centimetres off in depth — grasps
   close beside the object. If you have no depth recording, the `--depth-scale`
   flag also accepts a manual value.
3. **Frame rate.** The replay convention is 1 frame = 0.1 s (10 Hz). If the video
   was processed at a different rate, pass `--source-fps` to `package_run.py`
   (nearest-frame resampling).

## Objects: with or without

- `--objects-from <donor run>`: the donor's scene objects (meshes, poses, their
  converted USDs) are placed in the scene. Contact between the LEAP fingertips and
  the objects is recorded — episodes are then usable by `force_controller`.
  Physically meaningful when the teleop video interacted with the same objects on
  the same table spots.
- No flag: robot + table only. Good for checking motion quality; no grasp forces.

### Scenes from 3MF print projects (`scene_hra`, `scene_m110`)

Objects that were never reconstructed from a video can come from a 3MF instead:
`scripts/import_3mf_scene.py` writes a normal run folder under `data/runs/<name>/`
(meshes in metres, single-link URDFs with mass/inertia, `run_meta.json` with table
poses, the donor run's calibration and a 3 s *held* start pose as its trajectory), so
`--scene-from <name> --objects-from <name>` works everywhere the donor runs do:

```bash
# already built (see run_meta.json "import_command" for the exact placement used):
#   scene_hra   nut rail with 5 threaded posts (welded) + 4 small hex nuts, nut 1 = manipulated
#   scene_m110  hex screw pen holder (welded) + its large hex nut ALREADY THREADED ON near the
#               top of the screw (manipulated) - a pure screwing task, no pick-up
./sim_teleop/run_live_teleop.sh nuts_1 --scene-from scene_hra  --objects-from scene_hra
./sim_teleop/run_live_teleop.sh m110_1 --scene-from scene_m110 --objects-from scene_m110
conda run -n env_isaaclab python scripts/replay_trajectory.py --run teleop_nuts_1
python force_controller/run_tracking.py --episode outputs/teleop_nuts_1

# new 3MF: inspect the build items first, then place them (world frame, metres, yaw deg)
conda run -n env_isaaclab python scripts/import_3mf_scene.py --3mf my.3mf --name scene_x --inspect
conda run -n env_isaaclab python scripts/import_3mf_scene.py --3mf my.3mf --name scene_x \
    --place 0=-0.30,-0.20,90 1=-0.44,-0.25 --manipulated 1 --static 0 --mass 1=2.5
conda run -n env_isaaclab python scripts/convert_assets.py --objects-only --runs scene_x
```

The fixture (rail / screw) is in `static_keys` so a jittery hand cannot slide it; remove
it from `run_meta.json` (or use `replay_trajectory.py --static-object-ids`) for a free
fixture. Masses come from the slicer's print weights when the 3MF stores them (M110) and
otherwise from volume x PLA density x a fill fraction - pass `--mass` for small parts.

**Threads need SDF colliders.** The default convex decomposition has no thread (a nut becomes
a solid ring of hulls, a screw a smooth cylinder), so interlocking parts get
`--collision <idx>=sdf`: the run_meta object carries `"collision": "sdf"` and
`zerofact.replay.build_scene` gives that object a PhysX signed-distance-field triangle
mesh (resolution 256, `SDF_RESOLUTION` in replay.py). In `scene_m110` both parts are SDF and the
nut starts 84 mm up the screw at the yaw where the helices mate (`--place 0=x,y,178.5,0.084`,
found by fitting the M110x6 helix phase of both meshes). `scripts/test_thread.py --run scene_m110`
checks the physics: it pushes the nut with 15 N (`--push-force`) down and sideways and requires
the thread to hold (slip < 1.5 mm), then a torque about Z turns the nut down the screw at
5.9 mm/turn (pitch 6 mm) until it bottoms on the head, 0.1 mm off-axis.

SDF objects also get their own contact settings (`SDF_*` in `zerofact/replay.py`: 64
position iterations, 5 mm contact offset, 2 m/s depenetration cap). With the grasping defaults a
200 N push - trivial for the fixed-base PD arm pressing against a welded screw - drove the nut
14 mm through the thread and the depenetration kick then flung it away ("the screw and nut
insert into each other"); with the SDF settings the thread holds to 400 N with < 0.6 mm slip.
Convex-decomposition scenes are untouched (bit-identical replays). Screwing needs the nut
*dynamic*: the live viewer welds objects by default, so pass `--dynamic-objects` to
`live_sim_view.py` / `run_live_teleop.sh` (works in both the full-robot and `--floating` modes;
the floating grasp mode now applies the same colliders and materials as the validated scene).

## What ends up where

| path | what |
|---|---|
| `outputs/sim_teleop/<seq>/` | work dir: world/skeleton npy, depth_scale.json, retarget json |
| `data/runs/teleop_<name>/` | the packaged run (trajectory + run_meta + scene calib [+ objects]) |
| `outputs/teleop_<name>/<stamp>/` | after `replay_trajectory.py`: `replay_data.npz`, videos, plots |

## Troubleshooting

| symptom | fix |
|---|---|
| Dyn-HaMR finds no hand / wrong track | list ids: `find $CAPTURE_ROOT/dynhamr/track_preds/<seq> -type f`; pass the id as the 2nd arg of `run_hand_capture.sh` |
| MANO forward returns zeros | you are in the `dynhamr` env — use `dynhamr5090` (RTX 5090 needs it) |
| hand floats above / stabs through the table | depth correction missing or wrong — redo 3a/3b; check the calibration file matches the recording session |
| hand is mirrored / rotated on the table | wrong `camera_frame_pose.json` (camera moved since the donor recording) |
| robot jumps at t=0 in the player | normal if the trajectory starts far from home; the player resets the robot to frame 0 first, so a jump means frame 0 itself is odd — check the video's first seconds |
| `package_run.py` complains about calibration files | the donor run isn't ingested (`data/runs/<donor>/scene/` missing) — pick another donor or pass explicit files |
| replay can't find object USDs | the donor's scene was never converted — `python scripts/convert_assets.py` |
| IK looks fine but fingers close beside the object | the object in sim is at the donor's recorded pose; your video must interact with objects at those same table spots (or edit `run_meta.json` object poses) |

## Format contracts (for debugging)

- Stage 4 output: `{"traj": [{"robot_cfg": {"joint_1": …, …, "0": …, "15": …}}, …]}` —
  joint names are URDF names; the digit names are the LEAP joints and are mapped to
  `leap_j*` by `data/robot/joint_name_map.json` at load time.
- The packaged trajectory adds `total_frames`, `traj_id`, and the six key-frame
  fields (all optional annotations; `null` is fine).
- `run_meta.json` needs: `traj_run`, `scene_run` (where object USDs live under
  `assets/usd/scenes/`), `objects` (may be `[]`), `static_keys`.
- The scale json: `{"metric_scale": s, "depth_scale_for_converter": 1/s, …}` —
  the converter **divides** camera-frame points by `depth_scale_for_converter`.
