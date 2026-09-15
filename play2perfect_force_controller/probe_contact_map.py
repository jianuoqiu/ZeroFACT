#!/usr/bin/env python3
"""Diagnostic: does the statics map ``dq = -K^-1 J_c^T f`` point the way that raises the force?

Replays a recorded episode's commands bit-exactly up to ``--frame`` (a frame inside the grasp),
then, once per probed joint and sign, nudges that joint's PD target by ``--delta`` for
``--hold-steps`` physics steps and reports how each fingertip's force on the manipulated object
changed. Next to it prints the direction the task-space law would move that joint for the force it
measured at that frame. A joint whose law direction *lowers* the force is a modelling problem
(sign / lever arm / joint limit), not a gain problem.

    python play2perfect_force_controller/probe_contact_map.py --episode <ep dir> --frame 75

CAVEAT (measured 2026-09-08): a fresh process replays an episode bit-exactly, but repeated
``reset()`` + ``restore_state()`` replays *inside one process* do not - at a contact-rich frame the
held-command baseline differed by up to 15 N of fingertip force between two identical replays
(PhysX keeps warm-start / contact caches across resets). The tool prints that repeat-to-repeat
spread first; only trust the nudge table when it is well below the effects you are looking at.
Probing each nudge in its own process is the reliable (slow) alternative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from play2perfect_force_controller.launch import hard_exit, prepare_display  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episode", required=True)
parser.add_argument("--frame", type=int, required=True, help="frame to probe (inside the grasp)")
parser.add_argument("--delta", type=float, default=0.03, help="target nudge per probe (rad)")
parser.add_argument("--hold-steps", type=int, default=12, help="physics steps the nudge is held")
parser.add_argument("--joints", nargs="*", default=None, help="joint names (default: all hand joints)")
parser.add_argument("--seed", type=int, default=0)
prepare_display()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

from play2perfect_force_controller.episode import ReplayEpisode, pad_contact_centroid  # noqa: E402
from play2perfect_force_controller.p2p_env import AssemblyBench, SceneOptions, make_env_cfg  # noqa: E402
from play2perfect_force_controller.robot_spec import ARM_JOINT_NAMES, FINGERTIP_LABELS  # noqa: E402
from play2perfect_force_controller.sim_runner import _tip_point_jacobians  # noqa: E402


def main() -> int:
    episode = ReplayEpisode.load(ReplayEpisode.resolve_dir(args.episode, PROJECT_ROOT / "outputs"))
    cfg = make_env_cfg(episode.summary["problem"], seed=args.seed, sim_device=args.device)
    env = AssemblyBench(cfg, SceneOptions(render=False))
    episode = episode.reordered(list(env.robot.joint_names))
    names = episode.joint_names
    steps = int(env.cfg.decimation)
    manip = episode.contact_object_keys.index(episode.manipulated_key)
    dev = env.device
    tips = episode.fingertip_bodies
    K = env.robot.data.joint_stiffness[0].cpu().numpy().astype(np.float64)

    def replay_to(frame: int) -> None:
        env.reset()
        env.restore_state(episode.summary["init_state"])
        for f in range(frame):
            cmd = torch.as_tensor(episode.joint_target[f], dtype=torch.float32, device=dev).unsqueeze(0)
            for _ in range(steps):
                env.physics_step(cmd)

    def object_forces() -> np.ndarray:
        _, pair_f, _ = env.tactile_latest()
        return np.linalg.norm(pair_f[:, manip], axis=-1)              # [tips] |F| from the object

    # ---- the law's direction at the probed frame ----
    replay_to(args.frame)
    f_net, pair_f, pair_p = env.tactile_latest()
    f0 = object_forces()
    centroid = pad_contact_centroid(pair_f, pair_p)
    jac = _tip_point_jacobians(env.robot, env.tip_body_ids, env.tip_positions(), centroid, None, env.tip_positions())
    dq_law = np.zeros((len(tips), len(names)))
    for t in range(len(tips)):
        dq_law[t] = -(1.0 / K) * (jac[t].T @ f_net[t])                # per-tip statics map, f = measured
    q = env.robot.data.joint_pos[0].cpu().numpy()
    lim = env.robot.data.joint_pos_limits[0].cpu().numpy()
    print("=" * 100)
    print(f"frame {args.frame}: |F| from object per tip:",
          {FINGERTIP_LABELS[b]: round(float(v), 1) for b, v in zip(tips, f0)})
    joints = args.joints or [n for n in names if n not in ARM_JOINT_NAMES]
    loaded = [t for t in range(len(tips)) if f0[t] > 1.0]
    header = f"{'joint':22s} {'q':>6s} {'lo..hi':>13s} |" + "".join(
        f" {'law dq(' + FINGERTIP_LABELS[tips[t]][:5] + ')':>16s}" for t in loaded) + " |" + "".join(
        f" {'d|F| ' + FINGERTIP_LABELS[tips[t]][:5] + ' +/-':>20s}" for t in loaded)
    print(header)
    def hold_and_measure(cmd: np.ndarray) -> np.ndarray:
        replay_to(args.frame)
        cmd_t = torch.as_tensor(cmd, dtype=torch.float32, device=dev).unsqueeze(0)
        for _ in range(args.hold_steps):
            env.physics_step(cmd_t)
        return object_forces()

    # control: the recorded command held unchanged for the same steps (the grasp keeps evolving
    # on its own - arm motion, settling - so every nudge is measured against this drift). It is
    # re-measured for every joint, right before its two nudges: the first replay after the env
    # comes up was seen to differ from all later ones, so a single shared baseline is not safe.
    held = episode.joint_target[args.frame].copy()
    b0, b1 = hold_and_measure(held), hold_and_measure(held)
    print("held-command baseline change over the hold:",
          {FINGERTIP_LABELS[b]: round(float(v), 1) for b, v in zip(tips, b1 - f0)},
          " (repeat-to-repeat spread", np.abs(b1 - b0).max().round(2), "N)")
    rows = []
    for jn in joints:
        j = names.index(jn)
        baseline = hold_and_measure(held)
        deltas = {}
        for sign in (+1.0, -1.0):
            cmd = held.copy()
            cmd[j] += sign * args.delta
            deltas[sign] = hold_and_measure(cmd) - baseline
        line = f"{jn:22s} {q[j]:6.3f} {lim[j, 0]:6.2f}..{lim[j, 1]:5.2f} |"
        for t in loaded:
            line += f" {dq_law[t, j] * 1000:+16.1f}"
        line += " |"
        for t in loaded:
            line += f"   {deltas[+1.0][t]:+7.2f} / {deltas[-1.0][t]:+7.2f}"
        print(line)
        rows.append((jn, {tips[t]: (float(dq_law[t, j]), float(deltas[+1.0][t]), float(deltas[-1.0][t])) for t in loaded}))
    print("law dq in mrad for the force measured at this frame (per loaded tip); d|F| in N versus the "
          f"held-command baseline after {args.hold_steps} steps of a {args.delta:+.3f} / {-args.delta:+.3f} rad "
          "nudge of that joint's target (a joint at its limit shows +0.00 on the blocked side)")
    print("=" * 100, flush=True)
    return 0


if __name__ == "__main__":
    status = 0
    try:
        status = main()
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        hard_exit(app, status)
