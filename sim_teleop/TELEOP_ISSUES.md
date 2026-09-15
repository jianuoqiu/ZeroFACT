# Live Teleop: Why It Was Bad — Technical Post-Mortem & Solutions

*(status as of 2026-09-01; every number below was measured on this project's data, not guessed)*

## TL;DR

The pipeline has ~6 stages. Five of them are now measured-good. **Everything that felt
"so so so bad" traces to the very first link: per-frame 3D hand estimation from a single
RGB-D camera.** In the live sessions MediaPipe's landmarks are 2–2.5× worse than on the
original recording rig (reprojection error 7–10 px vs 4.3 px), and about once per second
it silently flips its 3D interpretation of the hand by 90–120° while the 2D skeleton
overlay still looks perfect. Three real retargeting bugs made this worse and are fixed;
the estimation quality itself is a *conditions/sensor* problem with a clear upgrade path.

The proof the rest of the chain is sound: the **same live code**, fed the original rig
recording instead of the live camera stream, grasps the apple, lifts it, and carries it toward
the bowl — end to end, through the validated replay.

---

## The pipeline, and where it breaks

```
RealSense RGB-D (aligned)                            [OK - verified aligned, correct intrinsics]
  └─> MediaPipe 21 landmarks (2D px + 3D shape)      [<<< ROOT CAUSE: noisy + flips in live sessions]
      └─> PnP pose + per-frame depth anchoring       [OK - chirality-safe, sensor-metric]
          └─> delta/clutch mapping to robot frame    [OK - rigid, det+1; delta≡absolute to 2.2°]
              └─> finger + wrist targets             [FIXED - was the #1 retargeting bug]
                  └─> IK (decoupled / floating)      [FIXED - was the #2 retargeting bug]
                      └─> viewer + 10 Hz recording   [OK - same joints you see are saved]
                          └─> package -> replay      [OK - bit-deterministic, validated]
```

## Why it *felt* incomprehensible

1. **The camera window lied by omission.** PnP re-projects whatever 3D shape MediaPipe
   produced back onto the correct pixels, so the drawn skeleton looks perfect even when
   the 3D hand behind it is mirrored, flipped, or hallucinated. You judged "hand
   estimation is good" from the one view that cannot show the problem.
2. **Each downstream stage amplified upstream noise.** Finger targets ride the wrist
   frame → a 90° wrist-estimate flip threw every fingertip sideways → position-only IK
   slammed abduction joints into their ±60° rails → rate caps smeared the snap over
   several frames → the robot looked possessed rather than merely noisy.
3. **"Offline works, so live should work" was a false intuition.** Dyn-HaMR optimizes
   one temporally-consistent MANO trajectory over the *whole video* (`smooth_fit`) — it
   structurally cannot flip between frames. MediaPipe is frame-by-frame. Same 21 points
   on paper, completely different noise processes.

---

## Issues found, in the order they were root-caused

| # | Issue | Key evidence | Status |
|---|-------|--------------|--------|
| 1 | Finger crossing from chirality-unsafe pose fit | Coplanar-anchor Umeyama could mirror the palm | **Fixed** (PnP, warm-started) |
| 2 | Open hand → closed robot hand; fingers converge/cross | Open hand solved to 20.7° curl; tips 3.2 cm apart vs LEAP's 4.5 cm | **Fixed** (knuckle map) |
| 3 | Arm moves when only fingers move | Finger-task/arm-motion correlation 0.62 in the coupled solve | **Fixed** (decoupled IK; floating mode has no arm at all) |
| 4 | Robot moves while the hand is static | 8.6°/frame wrist wobble × 12.7 cm bracelet lever | **Fixed** (One Euro filters) |
| 5 | Wrist-frame interpretation flips | 85–118° single-frame jumps, ~1/s, **zero dropouts**, wrist translating ~1 cm | **Mitigated ×3** (gate + flip-immune fingers + offline repair) |
| 6 | Live-session hand estimation globally poor | reproj 7–10 px vs rig 4.3; POOR VIEW 45–90% of frames | **OPEN — the root cause** |

### Issue 2 — the retargeting bug you correctly suspected (fixed)

The offline V2S2R scheme feeds **raw human fingertip positions** + the wrist into one
weighted IK. A human palm is much narrower than the LEAP palm, so those targets
constantly pull the fingers inward and short: measured on a real open-hand frame, ~20°
of curl and ±8° of converging abduction — with tracking noise, visibly crossing fingers.
V2S2R "resolves" the conflict by weights (wrist cost 2.0 beats fingers 1.0) and simply
accepts cm-level fingertip residuals — survivable for smooth offline power grasps, not
for live teleop. The fix: each fingertip target = **the LEAP finger's own knuckle + your
tip-from-knuckle vector × (LEAP finger length / your finger length)**. Neutral spread
becomes the robot's own, open→open (curl 20.7°→2.0°), tip residual 2.1→0.1 cm, and hand
size cancels per finger. Benchmark result: raw scheme only *poked* the apple (0.007 m
displacement); knuckle map **grasps and carries it** (0.15–0.37 m).

### Issue 5 — the flips (mitigated three ways)

MediaPipe occasionally re-decides "palm-toward vs palm-away" between frames. The output
points then rotate ~90–120° in 0.03–0.1 s — kinematically impossible (>850°/s), with the
2D overlay unchanged. Defenses now in the code:

- **Flip gate** (live): impossible palm-frame jumps are rejected and the last pose held;
  a persistent new orientation is accepted only if the depth anchors fit it clearly
  better or it survives ~0.5 s (a real fast turn never triggers the gate — it arrives as
  ~20°/frame, not 100° in one frame).
- **Flip-immune fingers** (live): finger vectors are built in the hand's *own raw palm
  frame*, so a whole-hand flip cancels exactly for the fingers. Verified: a 110° flip
  injected every 1.1 s changes index/middle/ring fingertips by **0.3 mm mean** vs the
  clean run. A flip can now at worst freeze/swivel the floating hand's *base* — the
  fingers stay correct.
- **Offline wrist repair** (retarget): a stint entered and exited by impossible snaps
  that returns to the pre-entry pose is slerp-bridged before the arm is fitted. On your
  own float3 recording: 13 frames repaired, wrist-fit error **15.1°→3.5° mean,
  140°→47° max**; zero false repairs on the clean benchmark.

### Issue 6 — the root cause that remains: live-session estimation quality

Measured split (reprojection error is MediaPipe-internal — no depth, no retargeting):

| source | reproj px (med/p90) | depth fit cm (med) | POOR VIEW |
|---|---|---|---|
| rig recording (works) | 4.3 / 5.6 | 1.27 | 4% |
| live sessions | 7.3–10.1 / 11–17 | 1.5–2.1 | 45–90% |

Ruled out with data: **distance** (corr(distance, fit) = 0.00 over 5083 live frames),
**depth-RGB misalignment** (BGR_Reader applies `rs.align` + color intrinsics),
**threshold miscalibration** (the rig sits comfortably under it). What remains is
MediaPipe's landmark quality itself. The live raw frames are not saved (upgrade #1 fixes
that), so the exact degrader is inferred, not measured; the leading suspects for the
current operating spot (hand over a **black carpet**) are:

- **Auto-exposure metering a mostly-dark scene.** The RGB sensor meters the whole
  frame; a black background drags the average down, so auto-exposure raises exposure
  time and gain — motion blur whenever the hand moves, gain noise, and the hand (the
  brightest object) drifting toward washed-out. All three degrade landmarks exactly in
  the measured way. Check live in the camera window: does the hand smear when moving,
  does it look bright/flat?
- **Dim ambient light** at the spot, with the same blur/noise consequences.
- Backlight from a window or screen behind the operator, if any.

For reference, the known-good rig recording is evenly and moderately exposed: mean
luminance 98/255 (stable 96–101 across the whole take), sharp (Laplacian var ~209),
zero blown-out pixels. Aim the live picture at that look: scene lit evenly (light the
area, not just the hand — a bright hand over a black ground makes metering worse),
hand neither smeared nor washed out. This is why "most of the space" reads POOR VIEW:
it is not a spot you haven't found — the imaging conditions are below what the model
needs everywhere in the frame.

### Dead ends (tried, measured harmful — do not retry)

- **Handedness label filtering**: the label reads "Left" on 143/143 frames of a
  right-hand video. Pure noise.
- **Determinant-sign mirror guard**: MediaPipe flattens the thumb toward the palm plane
  (normalized det ±0.1 vs 0.25–0.49 on MANO truth) — the sign is ill-conditioned; the
  guard corrupted good frames and made the benchmark 10× worse. Reverted.
- **Depth-anchor chirality vote**: the wrist+MCP anchors are near-planar → mirror-blind.
  Procrustes against MANO truth showed the raw geometry is right-handed on 131/160
  frames — there was no persistent mirror to fix.

### Minor known behaviors (accepted)

- Thumb has ±1 cm solver indeterminacy when its target is unreachable (redundant chain,
  flat optimization valley). Benign; do not chase bitwise determinism.
- The robot's fingers reach ~5 cm deeper than yours (palm sits at *your* wrist, robot
  fingers are longer). Same property as the offline pipeline; helps grasps close. If
  contacts ever land too early/deep, shift the floating root along the palm axis by the
  reach difference (one-line change).

---

## Solutions

### Do now (no code changes — conditions + the built-in feedback)

1. **Light your hand**: a lamp aimed at the hand, not the camera. No bright monitor or
   window behind you. Single highest-leverage change for MediaPipe.
2. Plain, non-skin-tone surface behind/below the hand (dark mat). Sleeve rolled up
   (the wrist is a depth anchor).
3. Hand 40–60 cm from the camera, palm or back of hand facing it; never fingers at the
   lens, never edge-on. The proven reference geometry is the rig's: camera above,
   looking down ~45° at the back of the hand (`--view top`).
4. **Trust the `EST GOOD/OK/BAD` badge, not the skeleton overlay.** It is calibrated so
   the entire known-good rig session reads GOOD and the failed live sessions read
   OK/BAD; when BAD it prints whether the cure is lighting/background/motion or
   camera geometry. Only record on GOOD.
5. After each take: the printed estimation-health share (<60% GOOD → re-record) and the
   `[float-retarget]` report (repaired-flip count, wrist error, workspace WARN).

### Upgrades, ranked by payoff per effort

| rank | upgrade | what it fixes | effort |
|---|---|---|---|
| 1 | **Hybrid workflow**: save raw RGB-D during takes; re-lift chosen takes offline through Dyn-HaMR + existing retargeting | Final data becomes offline-quality regardless of live noise; live view demoted to feedback-only. Frames are already saved in the pipeline's own folder format | ~1 day of glue |
| 2 | **AprilTag strapped to the back of the hand** (or 3-face tag cube) for the wrist pose | Eliminates flips and wrist noise entirely (tag pose is unambiguous); MediaPipe keeps only palm-relative fingers — its most reliable output, exactly what the knuckle map consumes | ~1 day |
| 3 | **Replace MediaPipe with WiLoR** (real-time MANO regressor; the RTX 5090 handles it) | Raises the floor of the whole estimate, fingers included; same 21-point interface | 1–2 days |
| 4 | **Headset hand tracking** (Quest 3 / Vision Pro, TeleVision-style) or a mocap glove | The standard modern answer for dexterous teleop collection; purpose-built, flip-free | bigger lift / hardware cost |

Recommended combination: **1 + 2**. Independent, both cheap, and together they give a
pleasant live view (tag wrist is rock solid) plus guaranteed final-data quality
(offline re-lift). #3 afterwards if the live view itself should be beautiful.

### What NOT to invest in

More filtering/gating on MediaPipe output. The flip gate, flip-immune fingers, offline
repair, and One Euro stack already extract what is extractable; the remaining error is
in the landmarks themselves, and only better input fixes that.

---

## What is validated today (the trust anchors)

- Rig recording through the **live** stack (absolute + floating, knuckle map): wrist fit
  0.35 cm / 1.5° mean; replay grasps the apple, lifts ~5 cm, carries toward the bowl;
  contacts at the human's grasp frames. Runs kept: `teleop_floatbench_kn3/kn4`
  (gold offline reference: 0.21 m to the bowl, 13.5 cm lift).
- Delta mode ≡ absolute mode for fingers: 2.2° mean difference, 0.99 correlation.
- Flip immunity: 110° flips injected every 1.1 s → fingertips differ 0.3 mm mean from
  the clean run.
- Replays are bit-deterministic; packaged episodes load in `force_controller`
  (`ReplayEpisode.load` verified).
