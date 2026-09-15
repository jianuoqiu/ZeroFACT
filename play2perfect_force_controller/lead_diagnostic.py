#!/usr/bin/env python3
"""Where does the law put its offsets, compared with the RL policy's own preload? (no simulator)

Closed-loop (oracle) rollouts record, for every executed frame, the plan knot the middle layer
was given: the policy's command and the state it reached in the planner. Their difference is the
policy's own preload ("lead") per joint - the offset it would have applied at this very state.
The law's offset ``dq`` is recorded per substep. This script compares the two on the hand
joints, per finger-joint group and per fingertip force level::

    python play2perfect_force_controller/lead_diagnostic.py --tag o12pos p_kp2   # rollouts by tag
    python play2perfect_force_controller/lead_diagnostic.py --dirs <rollout dir> ...

Reported per problem: |lead| and |dq| per joint group (MCP flexion, abduction, PIP/DIP/IP, thumb
CMC), the cosine between the law's hand-offset vector and the plan's lead vector, and how much
of each vector's magnitude sits in the soft distal joints.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from play2perfect_force_controller.robot_spec import ARM_JOINT_NAMES  # noqa: E402

ROLLOUTS = PROJECT_ROOT / "outputs" / "play2perfect" / "force_controller"
GROUPS = {
    "MCP flex": lambda n: n.endswith("MCP_FE") or n == "left_5_pinky_CMC",
    "abduction": lambda n: n.endswith("_AA"),
    "PIP/DIP/IP": lambda n: n.endswith("PIP") or n.endswith("DIP") or n.endswith("_IP"),
    "thumb CMC": lambda n: "thumb_CMC" in n,
}


def analyse(d: Path) -> dict | None:
    c = np.load(d / "controller_data.npz", allow_pickle=True)
    if "plan_joint_target" not in c.files:
        return None
    s = json.loads((d / "summary.json").read_text())
    names = list(c["joint_names"])
    hand = np.array([i for i, n in enumerate(names) if n not in ARM_JOINT_NAMES])
    lead = (c["plan_joint_target"] - c["plan_joint_pos"])[:, hand]           # [T, Jh] policy preload
    dq = c["dq_steps"][:, -1, hand]                                          # [T, Jh] law offset (frame end)
    f_ref = c["f_ref_steps"][:, -1]                                          # [T, tips]
    eng = (f_ref >= 2.0).any(axis=1)                                         # firm contact planned
    if eng.sum() < 5:
        return None
    lead, dq = lead[eng], dq[eng]
    # a lead on a joint the policy holds AT a position limit is a push into the limit, not a
    # torque request through the spring: drop those entries from both vectors
    if "joint_limits" in c.files:
        lim = c["joint_limits"][hand]
        qp = c["plan_joint_pos"][eng][:, hand]
        at_limit = (qp <= lim[:, 0] + 0.01) | (qp >= lim[:, 1] - 0.01)
        lead = np.where(at_limit, 0.0, lead)
        dq = np.where(at_limit, 0.0, dq)
    hn = [names[i] for i in hand]
    out = {"problem": s["problem"], "rollout": d.name, "frames": int(eng.sum())}
    for g, pred in GROUPS.items():
        idx = [j for j, n in enumerate(hn) if pred(n)]
        out[g] = (float(np.abs(lead[:, idx]).mean()), float(np.abs(dq[:, idx]).mean()))
    dist = [j for j, n in enumerate(hn) if GROUPS["PIP/DIP/IP"](n)]
    out["distal share"] = (float(np.abs(lead[:, dist]).sum() / max(np.abs(lead).sum(), 1e-9)),
                           float(np.abs(dq[:, dist]).sum() / max(np.abs(dq).sum(), 1e-9)))
    cos = np.sum(lead * dq, axis=1) / (np.linalg.norm(lead, axis=1) * np.linalg.norm(dq, axis=1) + 1e-9)
    out["cosine"] = float(np.nanmean(cos))
    out["magnitude ratio"] = float(np.linalg.norm(dq, axis=1).mean() / max(np.linalg.norm(lead, axis=1).mean(), 1e-9))
    env = (s["outcome"].get("env") or {}).get("last") or {}
    out["env"] = f"{env.get('successes', '?')}/{env.get('max_goals', '?')}{'+R' if env.get('retract_succeeded') else ''}"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", nargs="*", default=[], help="rollout tags (folder suffix <stamp>_<tag>_task_space)")
    p.add_argument("--dirs", nargs="*", default=[], help="explicit rollout folders")
    a = p.parse_args()
    dirs = [Path(x) for x in a.dirs]
    for tag in a.tag:
        dirs += [Path(x) for x in sorted(glob.glob(str(ROLLOUTS / "*" / f"*_{tag}_task_space")))]
    rows = [r for r in (analyse(d) for d in dirs) if r]
    if not rows:
        print("no closed-loop task_space rollouts with recorded plans found"); return 1
    print("mean |offset| in rad over frames with a planned force >= 2 N: policy lead / law dq")
    print(f"{'problem':20s} {'rollout':34s} {'env':8s} " + " ".join(f"{g:>17s}" for g in GROUPS)
          + "   distal share   cosine   |dq|/|lead|")
    for r in rows:
        cells = " ".join(f"{r[g][0]:7.3f} / {r[g][1]:7.3f}" for g in GROUPS)
        print(f"{r['problem']:20s} {r['rollout'][:34]:34s} {r['env']:8s} {cells}   "
              f"{r['distal share'][0]:.2f} / {r['distal share'][1]:.2f}   {r['cosine']:+.2f}   {r['magnitude ratio']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
