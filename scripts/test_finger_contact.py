#!/usr/bin/env python3
"""How deep do the LEAP fingertips sink into a scene object when the hand closes on it?

Spawns the run's objects exactly like the replay (materials, SDF/convex colliders, SDF contact
settings) plus the free-floating LEAP hand, parks the palm so the fingers curl straight down onto
the manipulated object's top face, commands a full curl, and measures the penetration of every
fingertip's collision hull into the object mesh (trimesh signed distance, positive = inside).

The hand root is teleported (kinematic) like the live viewer's floating mode, so the only thing
that can yield is the finger drive - which is what the effort limit is for::

    conda run -n env_isaaclab python scripts/test_finger_contact.py --run scene_m110 --hand-effort 50
    conda run -n env_isaaclab python scripts/test_finger_contact.py --run scene_m110 --hand-effort 0.95
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from v2s2r_isaaclab.runtime import prepare_display  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--run", required=True)
parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
parser.add_argument("--usd-dir", type=Path, default=PROJECT_ROOT / "assets" / "usd")
parser.add_argument("--hand-effort", type=float, default=None,
                    help="finger joint effort limit, N m (default: the run's hand_effort_limit, else replay HAND_EFFORT)")
parser.add_argument("--curl-steps", type=int, default=240, help="physics steps of curling (2 s)")
parser.add_argument("--curl-ramp-s", type=float, default=1.0,
                    help="ramp the curl target from open to closed over this long (a human closes in ~0.5-1 s; "
                         "an instant target lets a 50 N m finger flick past a narrow face without touching it)")
parser.add_argument("--approach-mm", type=float, default=15.0, help="open fingertip height above the face")
parser.add_argument("--full-scene", action="store_true",
                    help="keep every object and the object's own dynamics; default isolates the manipulated "
                         "object and welds it so the number measured is purely finger compliance")
parser.add_argument("--root-drive", action="store_true",
                    help="pull the hand root with the live viewer's spring-damper wrench (sim_teleop/root_drive.py) "
                         "instead of teleporting it, and after the curl push the whole hand 40 mm down into the object")
parser.add_argument("--press-mm", type=float, default=40.0, help="--root-drive: how far below the face the palm target goes")
parser.add_argument("--report-every", type=int, default=30)
prepare_display()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from v2s2r_isaaclab import replay as rp  # noqa: E402
from v2s2r_isaaclab.scene_spec import describe, load_run_spec, quat_xyzw_to_wxyz  # noqa: E402

FINGERS = {"index": ("fingertip", ["leap_j1", "leap_j0", "leap_j2", "leap_j3"]),
           "middle": ("fingertip_2", ["leap_j5", "leap_j4", "leap_j6", "leap_j7"]),
           "ring": ("fingertip_3", ["leap_j9", "leap_j8", "leap_j10", "leap_j11"])}
CURL = [1.4, 0.0, 1.2, 1.0]          # MCP flex, abduction, PIP, DIP
THUMB = {"leap_j12": 0.0, "leap_j13": 0.0, "leap_j14": -0.5, "leap_j15": 0.15}   # donor frame-0 thumb
# fingertip.stl placement inside the fingertip link (URDF <collision><origin>)
TIP_COL_XYZ = np.array([0.013286424108533503, -0.006114238386541987, 0.0145])
TIP_COL_RPY = np.array([math.pi, 0.0, 0.0])


def main() -> int:
    spec = load_run_spec(args.data_dir / "runs" / args.run)
    print(describe(spec))
    target_obj = spec.manipulated
    if target_obj is None:
        raise SystemExit("run has no manipulated object")
    if not args.full_scene:
        # the measurement is "how far do the fingers sink into a rigid part" - remove everything
        # else (the screw shaft sticks up through the nut and fouls the parked hand) and weld it
        spec.objects = [target_obj]
        target_obj.is_static = True
        print(f"[finger] isolated: only {target_obj.name}, welded (pass --full-scene for the whole scene)")
    effort = args.hand_effort
    if effort is None:
        effort = spec.hand_effort_limit if getattr(spec, "hand_effort_limit", None) is not None else rp.HAND_EFFORT
    print(f"\n[finger] hand effort limit {effort} N m")

    cfg = rp.ReplayConfig(device=args.device, render=False, save_images=False, video=False, hand_cam=False,
                          contact_sensors=False, force_vis=False, flow_points=0)
    sim = SimulationContext(rp.make_simulation_cfg(cfg, spec))

    # ---- scene: lights, ground, table, objects (as in the replay / floating viewer) ----
    dome = sim_utils.DomeLightCfg(intensity=900.0)
    dome.func("/World/DomeLight", dome)
    rp._spawn_static_box("/World/GroundPlane", size=(40.0, 40.0, 0.2), position=np.array([0.0, 0.0, -0.1]),
                         friction=rp.GROUND_FRICTION, color=(0.25, 0.25, 0.27))
    rp._spawn_static_box(f"{rp.ENV_PRIM}/Table", size=tuple(float(v) for v in spec.table_size),
                         position=spec.table_position, friction=rp.TABLE_FRICTION, restitution=rp.TABLE_RESTITUTION)
    objects = {}
    for obj in spec.objects:
        scene_dir = args.usd_dir / "scenes" / spec.scene_run
        usd = next(c for c in [scene_dir / obj.name / f"{obj.name}.usd", scene_dir / f"{obj.name}.usd"] if c.is_file())
        prim_path = f"{rp.ENV_PRIM}/{obj.name}"
        objects[obj.key] = RigidObject(RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd), rigid_props=rp.object_rigid_props(obj),
                                       collision_props=rp.object_collision_props(obj)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in obj.position),
                                                      rot=tuple(float(v) for v in quat_xyzw_to_wxyz(obj.quat_xyzw))),
        ))
        rp.prepare_object_colliders(obj, prim_path, cfg)

    hand_usd = args.usd_dir / "robot" / "leap_float.usd"
    hand = Articulation(ArticulationCfg(
        prim_path=f"{rp.ENV_PRIM}/FloatHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(hand_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=rp.MAX_DEPENETRATION_VELOCITY,
                solver_position_iteration_count=rp.SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=rp.SOLVER_VELOCITY_ITERATIONS),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, fix_root_link=False,
                solver_position_iteration_count=rp.SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=rp.SOLVER_VELOCITY_ITERATIONS),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=rp.CONTACT_OFFSET, rest_offset=rp.REST_OFFSET),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.6)),
        actuators={"hand": ImplicitActuatorCfg(joint_names_expr=[r"leap_j\d+"], stiffness=rp.HAND_STIFFNESS,
                                               damping=rp.HAND_DAMPING, effort_limit_sim=effort)},
    ))
    rp._prepare_colliders(f"{rp.ENV_PRIM}/FloatHand", "/World/Materials/hand",
                          sim_utils.RigidBodyMaterialCfg(static_friction=rp.ROBOT_FRICTION,
                                                         dynamic_friction=rp.ROBOT_FRICTION, restitution=0.0))
    sim.reset()
    dt = cfg.physics_dt
    device = hand.device
    jn = list(hand.joint_names)
    bn = list(hand.body_names)
    print(f"[finger] hand joints {jn}\n[finger] hand bodies {bn}")

    def joint_targets(curl: float) -> torch.Tensor:
        q = torch.zeros((1, hand.num_joints), device=device)
        for _, (_, joints) in FINGERS.items():
            for name, val in zip(joints, CURL):
                q[0, jn.index(name)] = val * curl
        for name, val in THUMB.items():
            q[0, jn.index(name)] = val
        return q

    root_pose = torch.tensor([[0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]], device=device)
    drive = None
    if args.root_drive:
        from sim_teleop.root_drive import RootPoseDrive

        drive = RootPoseDrive(hand)
        print(f"[finger] {drive.describe()}")
    drive_active = [False]        # calibration + parking are kinematic; the drive takes over once parked

    def step(curl: float):
        if drive is None or not drive_active[0]:
            hand.write_root_pose_to_sim(root_pose)
            hand.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
        else:
            drive.set_target(root_pose)
            drive.apply()
        hand.set_joint_position_target(joint_targets(curl))
        hand.write_data_to_sim()
        sim.step(render=False)
        hand.update(dt)
        for o in objects.values():
            o.update(dt)

    def tip_pose(body: str):
        i = bn.index(body)
        p = hand.data.body_pos_w[0, i].cpu().numpy().astype(np.float64)
        q = hand.data.body_quat_w[0, i].cpu().numpy().astype(np.float64)   # wxyz
        return p, R.from_quat([q[1], q[2], q[3], q[0]])

    # ---- calibrate the curl sweep of the index fingertip, far above the table ----
    hand.write_joint_state_to_sim(joint_targets(0.0), torch.zeros((1, hand.num_joints), device=device))
    for _ in range(60):
        step(0.0)
    p_open, _ = tip_pose("fingertip")
    root_p0 = hand.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    for _ in range(120):
        step(1.0)
    p_closed, _ = tip_pose("fingertip")
    sweep = p_closed - p_open
    print(f"[finger] index tip sweep during curl: {np.round(sweep * 1000, 1).tolist()} mm")

    # ---- park the hand palm-down above the object: at the asset's identity pose the open
    # fingers point +Z and curl toward -X/-Z (palm side = -X). Rotating -90 deg about Y lays the
    # fingers horizontal (pointing -X) with the palm facing down, so a curl sweeps the pads
    # down (-Z) onto the face and back toward the palm (+X). (Aligning the raw sweep chord to -Z
    # instead puts the palm *below* the fingertips - 13 mm inside the table.)
    rot = R.from_euler("y", -90.0, degrees=True)
    print(f"[finger] curl sweep after parking: {np.round(rot.apply(sweep) * 1000, 1).tolist()} mm (want -Z)")
    obj_body = objects[target_obj.key]
    obj_p = obj_body.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    obj_q = obj_body.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    obj_mesh = trimesh.load(target_obj.mesh_path, force="mesh", process=False)
    obj_top = obj_p[2] + float(obj_mesh.bounds[1, 2])
    # aim the open index tip at the middle of the top face's solid ring (the nut has a bore:
    # anything inside the bore radius drops into the hole and lands on the thread instead)
    v = np.asarray(obj_mesh.vertices)
    top_ring = v[v[:, 2] > float(obj_mesh.bounds[1, 2]) - 0.002]
    r_top = np.hypot(top_ring[:, 0], top_ring[:, 1])
    r_aim = 0.5 * (float(r_top.min()) + float(r_top.max()))
    print(f"[finger] top face ring r = {r_top.min() * 1000:.1f}..{r_top.max() * 1000:.1f} mm -> aiming the index tip at r = {r_aim * 1000:.1f} mm "
          "(the other fingers sit beside it and may miss the ring)")
    aim = np.array([obj_p[0] + r_aim, obj_p[1], obj_top + args.approach_mm / 1000.0])
    # re-open far away, then compute where the root must be so that R*(p_open - root) + root_new = aim
    for _ in range(90):
        step(0.0)
    p_open, _ = tip_pose("fingertip")
    root_p0 = hand.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    root_q0 = hand.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    root_new = aim - rot.apply(p_open - root_p0)
    q_new = (rot * R.from_quat([root_q0[1], root_q0[2], root_q0[3], root_q0[0]])).as_quat()   # xyzw
    tips_open = {k: tip_pose(b)[0] for k, (b, _) in FINGERS.items()}
    predicted = {k: root_new + rot.apply(v - root_p0) for k, v in tips_open.items()}
    root_pose = torch.tensor([[*root_new, q_new[3], q_new[0], q_new[1], q_new[2]]], dtype=torch.float32, device=device)
    print(f"[finger] teleport root {np.round(root_p0, 3).tolist()} -> {np.round(root_new, 3).tolist()}, "
          f"rot {np.round(rot.as_rotvec() * 180 / math.pi, 1).tolist()} deg")
    if root_new[2] < spec.table_position[2] + spec.table_size[2] / 2 + 0.03:
        raise SystemExit(f"[finger] parked root at z={root_new[2]:.3f} would be in the table - bad hand orientation")
    for n in (1, 5, 60):
        for _ in range(n if n == 1 else n - (1 if n == 5 else 5)):
            step(0.0)
        rp_w = hand.data.root_pos_w[0].cpu().numpy()
        print(f"[finger]   after {n:2d} step(s): root err {np.round((rp_w - root_new) * 1000, 1).tolist()} mm; tip err mm: "
              + "  ".join(f"{k} {np.round((tip_pose(b)[0] - predicted[k]) * 1000, 1).tolist()}" for k, (b, _) in FINGERS.items()))
    p_chk, _ = tip_pose("fingertip")
    print(f"[finger] parked: index tip {np.round((p_chk - aim) * 1000, 1).tolist()} mm from aim; tips above the "
          f"{target_obj.name} top face: " + ", ".join(
              f"{k} {(tip_pose(b)[0][2] - obj_top) * 1000:.1f} mm" for k, (b, _) in FINGERS.items()))

    if drive is not None:
        drive_active[0] = True
        print("[finger] root drive engaged (root is now a dynamic body pulled toward the parked pose)")

    # ---- curl onto the object and measure hull penetration ----
    tip_mesh = trimesh.load(PROJECT_ROOT / "data/robot/kinova_leap_description/meshes/leap/fingertip.stl", force="mesh")
    hull = np.asarray(tip_mesh.convex_hull.vertices, dtype=np.float64)
    if hull.max() > 1.0:          # STL in millimetres
        hull = hull / 1000.0
    R_col = R.from_euler("xyz", TIP_COL_RPY)
    hull_link = R_col.apply(hull) + TIP_COL_XYZ

    def penetration() -> dict[str, float]:
        p = obj_body.data.root_pos_w[0].cpu().numpy().astype(np.float64)
        q = obj_body.data.root_quat_w[0].cpu().numpy().astype(np.float64)
        R_obj = R.from_quat([q[1], q[2], q[3], q[0]])
        out = {}
        for label, (body, _) in FINGERS.items():
            tp, tR = tip_pose(body)
            pts_w = tR.apply(hull_link) + tp
            pts_obj = R_obj.inv().apply(pts_w - p)                       # into the object's mesh frame
            sd = trimesh.proximity.signed_distance(obj_mesh, pts_obj)    # >0 inside
            out[label] = float(sd.max()) * 1000
        return out

    obj_p_start = obj_p.copy()
    obj_yaw_start = R.from_quat([obj_q[1], obj_q[2], obj_q[3], obj_q[0]]).as_euler("ZYX")[0]
    print(f"[finger] curling for {args.curl_steps} steps ...")
    worst = {k: -1e9 for k in FINGERS}
    for i in range(1, args.curl_steps + 1):
        step(min(1.0, i * dt / max(args.curl_ramp_s, 1e-6)))
        if i % args.report_every == 0 or i == args.curl_steps:
            pen = penetration()
            for k, v in pen.items():
                worst[k] = max(worst[k], v)
            tau = hand.data.applied_torque[0].cpu().numpy()
            p = obj_body.data.root_pos_w[0].cpu().numpy().astype(np.float64)
            q = obj_body.data.root_quat_w[0].cpu().numpy().astype(np.float64)
            yaw = R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX")[0]
            heights = {k: (tip_pose(b)[0][2] - obj_top) * 1000 for k, (b, _) in FINGERS.items()}
            print(f"[finger]   t={i * dt:4.2f}s  penetration mm  "
                  + "  ".join(f"{k} {v:6.2f}" for k, v in pen.items())
                  + f"   index tip {heights['index']:6.1f} mm above face"
                  + f"   |tau|max {np.abs(tau).max():6.2f} N m   object moved {np.linalg.norm(p - obj_p_start) * 1000:5.2f} mm, "
                    f"turned {math.degrees((yaw - obj_yaw_start + math.pi) % (2 * math.pi) - math.pi):6.1f} deg")
    print("\n[finger] result: worst hull penetration into the object (mm, negative = never touched)")
    for k, v in worst.items():
        print(f"[finger]   {k:7s} {v:7.2f}")

    if drive is not None:
        # ---- the tracker says "hand 40 mm lower" while the object is in the way ----
        pressed = root_pose.clone()
        pressed[0, 2] -= args.press_mm / 1000.0
        root_pose = pressed
        print(f"\n[finger] root target lowered {args.press_mm:.0f} mm into the object; holding for {args.curl_steps} steps ...")
        worst_press = {k: -1e9 for k in FINGERS}
        for i in range(1, args.curl_steps + 1):
            step(1.0)
            if i % args.report_every == 0 or i == args.curl_steps:
                pen = penetration()
                for k, v in pen.items():
                    worst_press[k] = max(worst_press[k], v)
                f, tq = drive.apply()
                print(f"[finger]   t={i * dt:4.2f}s  penetration mm  " + "  ".join(f"{k} {v:6.2f}" for k, v in pen.items())
                      + f"   root lag {drive.position_error() * 1000:5.1f} mm   drive |F| {f:5.1f} N |tau| {tq:4.2f} N m")
        print("[finger] result with the palm pushed down: worst hull penetration (mm)")
        for k, v in worst_press.items():
            print(f"[finger]   {k:7s} {v:7.2f}")
        print(f"[finger]   root lag {drive.position_error() * 1000:.1f} mm of the {args.press_mm:.0f} mm commanded "
              "(the hand stopped at the object instead of entering it)")
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    from v2s2r_isaaclab.runtime import hard_exit

    hard_exit(simulation_app, status)
