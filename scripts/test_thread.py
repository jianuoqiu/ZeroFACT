#!/usr/bin/env python3
"""Does the nut actually thread on the screw? Physics-only check for an SDF-collider scene.

Builds the run's scene exactly like the replay (same physics, materials, solver settings), then:

1. **settle**: steps the sim with no input and reports how far the manipulated object drifts -
   a mis-placed nut (helix out of phase, wrong height) shows up here as a jump or an explosion;
2. **turn**: applies a constant torque about the world Z axis to the manipulated object and
   reports its yaw and height over time. On a real thread the two are locked: height per turn
   equals the pitch (M110x6 -> 6.0 mm/turn, right-hand: clockwise seen from above = down).

    conda run -n env_isaaclab python scripts/test_thread.py --run scene_m110
    conda run -n env_isaaclab python scripts/test_thread.py --run scene_m110 --torque -0.3 --turn-steps 600

No rendering, nothing written to outputs/. Exit code 1 when the object explodes or leaves the screw.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from zerofact.runtime import prepare_display  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", required=True, help="run name under data/runs")
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
parser.add_argument("--settle-steps", type=int, default=240, help="physics steps with no input (2 s)")
parser.add_argument("--turn-steps", type=int, default=480, help="physics steps under torque (4 s)")
parser.add_argument("--torque", type=float, default=-0.15,
                    help="torque about world +Z on the manipulated object, N m (negative = clockwise from above)")
parser.add_argument("--pitch-mm", type=float, default=6.0, help="expected thread pitch for the verdict")
parser.add_argument("--push-force", type=float, default=15.0,
                    help="N; before the torque phase, push the object straight DOWN then SIDEWAYS with this force "
                         "(a finger pressing on the nut) and measure how far it tunnels into the screw; 0 skips")
parser.add_argument("--push-steps", type=int, default=180, help="physics steps per push direction (1.5 s)")
parser.add_argument("--report-every", type=int, default=60)
prepare_display()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True                      # no rendering: the stock headless experience loads fine
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from zerofact.replay import ReplayConfig, build_scene, make_simulation_cfg  # noqa: E402
from zerofact.scene_spec import describe, load_run_spec, quat_wxyz_to_xyzw  # noqa: E402


def resolve_assets(spec, usd_dir: Path):
    robot_usd = usd_dir / "robot" / "kinova_leap.usd"
    object_usds = {}
    for obj in spec.objects:
        scene_dir = usd_dir / "scenes" / spec.scene_run
        cand = [scene_dir / obj.name / f"{obj.name}.usd", scene_dir / f"{obj.name}.usd"]
        usd = next((c for c in cand if c.is_file()), None)
        if usd is None:
            raise SystemExit(f"object USD missing: {cand[0]} (run scripts/convert_assets.py)")
        object_usds[obj.key] = usd
    return robot_usd, object_usds


def pose_of(obj) -> tuple[np.ndarray, float]:
    pos = obj.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    quat = quat_wxyz_to_xyzw(obj.data.root_quat_w[0].cpu().numpy().astype(np.float64))
    yaw = R.from_quat(quat).as_euler("ZYX")[0]
    return pos, yaw


def main() -> int:
    spec = load_run_spec(args.data_dir / "runs" / args.run)
    print(describe(spec))
    if spec.manipulated is None:
        raise SystemExit("the run has no manipulated object")
    target = spec.manipulated
    cfg = ReplayConfig(device=args.device, render=False, save_images=False, video=False, hand_cam=False,
                       contact_sensors=False, force_vis=False, flow_points=0,
                       output_dir=PROJECT_ROOT / "outputs" / "_thread_test")
    robot_usd, object_usds = resolve_assets(spec, args.usd_dir)
    sim = SimulationContext(make_simulation_cfg(cfg, spec))
    scene = build_scene(spec, cfg, robot_usd, object_usds)
    sim.reset()
    nut = scene.objects[target.key]
    dt = cfg.physics_dt

    def step(apply_torque: float | None, force=None):
        if apply_torque is not None or force is not None:
            forces = torch.zeros((1, 1, 3), device=nut.device)
            torques = torch.zeros((1, 1, 3), device=nut.device)
            if apply_torque is not None:
                torques[0, 0, 2] = apply_torque
            if force is not None:
                forces[0, 0, :] = torch.as_tensor(force, dtype=forces.dtype, device=nut.device)
            nut.set_external_force_and_torque(forces, torques)
            nut.write_data_to_sim()
        scene.robot.set_joint_position_target(scene.robot.data.default_joint_pos)
        scene.robot.write_data_to_sim()
        sim.step(render=False)
        nut.update(dt)
        scene.robot.update(dt)

    p0, yaw0 = pose_of(nut)
    print(f"\n[thread] {target.name}: start pos {np.round(p0, 4).tolist()} yaw {math.degrees(yaw0):.1f} deg")
    print(f"[thread] settle {args.settle_steps} steps ({args.settle_steps * dt:.1f} s) ...")
    ok = True
    for i in range(1, args.settle_steps + 1):
        step(None)
        if i % args.report_every == 0 or i == args.settle_steps:
            p, yaw = pose_of(nut)
            d = p - p0
            print(f"[thread]   t={i * dt:5.2f}s  dxy {np.hypot(d[0], d[1]) * 1000:6.2f} mm  "
                  f"dz {d[2] * 1000:7.2f} mm  dyaw {math.degrees(yaw - yaw0):7.2f} deg")
    p_s, yaw_s = pose_of(nut)
    drift = float(np.linalg.norm(p_s - p0))
    if not np.isfinite(drift) or drift > 0.02:
        print(f"[thread] FAIL: object moved {drift * 1000:.1f} mm while settling (mis-placed or exploded)")
        ok = False

    # ---- push: a finger pressing the nut must not drive it through the thread ----
    # on a thread, height and yaw are locked (dz = pitch * turns); anything beyond that is
    # penetration into the screw ("thread slip"), and any XY offset is the nut leaving the axis
    pitch = args.pitch_mm / 1000.0
    if args.push_force > 0:
        yaw_acc = 0.0
        prev = yaw_s
        for label, f in (("down", (0.0, 0.0, -args.push_force)), ("sideways +x", (args.push_force, 0.0, 0.0)),
                         ("sideways +y", (0.0, args.push_force, 0.0))):
            print(f"\n[thread] push {label} with {args.push_force:.1f} N for {args.push_steps} steps ...")
            for i in range(1, args.push_steps + 1):
                step(None, force=f)
                p, yaw = pose_of(nut)
                yaw_acc += (yaw - prev + math.pi) % (2 * math.pi) - math.pi
                prev = yaw
                if i % args.report_every == 0 or i == args.push_steps:
                    dz = p[2] - p_s[2]
                    slip = (dz - pitch * yaw_acc / (2 * math.pi)) * 1000
                    off = np.hypot(*(p[:2] - p0[:2])) * 1000
                    print(f"[thread]   t={i * dt:5.2f}s  dz {dz * 1000:7.2f} mm  turned {math.degrees(yaw_acc):7.1f} deg  "
                          f"thread slip {slip:7.2f} mm  off-axis {off:5.2f} mm")
                    if abs(slip) > 1.5 or off > 3.0:
                        ok = False
        # release and let it settle again before the torque phase
        for _ in range(60):
            step(None, force=(0.0, 0.0, 0.0))
        p_s, yaw_s = pose_of(nut)
        print(f"[thread] after pushes: {'penetration/derail detected -> FAIL' if not ok else 'thread held'}")

    print(f"\n[thread] torque {args.torque:+.3f} N m about +Z for {args.turn_steps} steps ...")
    total_yaw = 0.0
    prev_yaw = yaw_s
    for i in range(1, args.turn_steps + 1):
        step(args.torque)
        p, yaw = pose_of(nut)
        dy = (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
        total_yaw += dy
        prev_yaw = yaw
        if i % args.report_every == 0 or i == args.turn_steps:
            dz = (p[2] - p_s[2]) * 1000
            turns = total_yaw / (2 * math.pi)
            per_turn = dz / turns if abs(turns) > 1e-3 else float("nan")
            print(f"[thread]   t={i * dt:5.2f}s  turned {math.degrees(total_yaw):8.1f} deg  "
                  f"dz {dz:7.2f} mm  ->  {per_turn:6.2f} mm/turn   "
                  f"xy off-axis {np.hypot(*(p[:2] - p0[:2])) * 1000:5.2f} mm")
    nut.set_external_force_and_torque(torch.zeros((1, 1, 3), device=nut.device),
                                      torch.zeros((1, 1, 3), device=nut.device))

    p_e, _ = pose_of(nut)
    turns = total_yaw / (2 * math.pi)
    dz_mm = (p_e[2] - p_s[2]) * 1000
    print("\n[thread] result")
    print(f"[thread]   turned         : {math.degrees(total_yaw):.1f} deg ({turns:+.2f} turns)")
    print(f"[thread]   height change  : {dz_mm:+.2f} mm")
    if abs(turns) > 0.1:
        per_turn = dz_mm / turns
        print(f"[thread]   height per turn: {per_turn:+.2f} mm  (pitch {args.pitch_mm} mm; "
              f"{'THREADED' if abs(abs(per_turn) - args.pitch_mm) < 0.25 * args.pitch_mm else 'NOT following the thread'})")
        if abs(abs(per_turn) - args.pitch_mm) >= 0.25 * args.pitch_mm:
            ok = False
    else:
        print("[thread]   the object did not turn: torque too small for the friction, or it is jammed")
        ok = False
    off_axis = float(np.hypot(*(p_e[:2] - p0[:2])) * 1000)
    if off_axis > 5.0:
        print(f"[thread]   FAIL: {off_axis:.1f} mm off the screw axis")
        ok = False
    print(f"[thread] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    from zerofact.runtime import hard_exit

    hard_exit(simulation_app, status)
