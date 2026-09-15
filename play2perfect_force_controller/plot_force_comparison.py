#!/usr/bin/env python3
"""Recorded (target) force vs rollout (measured) force, several rollouts of one episode overlaid.

No simulator. Each rollout folder holds ``controller_data.npz`` with the per-substep force target
``f_ref_steps`` (the recording's fingertip force for the replay policy, the plan's for the oracle)
and the rollout's measured force ``f_meas_steps``. One row per fingertip: the target dashed, the
measured solid, one colour per rollout; the last row shows the manipulated object's height and
marks where the grasp-slip detector declared a drop.

    python play2perfect_force_controller/plot_force_comparison.py \
        outputs/play2perfect/force_controller/<run>/<stamp>_eval_null \
        outputs/play2perfect/force_controller/<run>/<stamp>_eval_task_space --out forces.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRAME_DT = 1.0 / 60.0
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def label_of(summary: dict) -> str:
    ctrl = summary["controller"]
    law = ctrl["force_law"]["law"]
    src = ctrl["policy"]["state_source"]
    kind = "oracle C%d" % ctrl["policy"]["chunk"] if ctrl["policy"].get("source") == "oracle" else "replay"
    prem = "commands" if src == "joint_target" else "reached states"
    extra = ""
    if law == "task_space":
        extra = f", clip {ctrl['force_law']['offset_clip_rad']}"
        if ctrl["force_law"].get("map") == "constrained":
            extra += ", constrained map"
    return f"{kind}, {prem}, {law}{extra}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollouts", nargs="+", help="rollout folders (same episode)")
    p.add_argument("--out", default=None, help="output png (default: next to the first rollout)")
    p.add_argument("--title", default=None)
    a = p.parse_args()
    runs = []
    for d in a.rollouts:
        d = Path(d)
        c = np.load(d / "controller_data.npz", allow_pickle=True)
        s = json.loads((d / "summary.json").read_text())
        runs.append((d, c, s))
    tips = [b.replace("left_", "").replace("_DP", "").replace("right_hand_", "").replace("_link2", "") for b in runs[0][1]["fingertip_bodies"]]
    n_tips = len(tips)
    fig, axes = plt.subplots(n_tips + 1, 1, figsize=(13, 2.1 * (n_tips + 1)), sharex=True)
    for k, (d, c, s) in enumerate(runs):
        col = PALETTE[k % len(PALETTE)]
        f_ref = c["f_ref_steps"][:, -1, :]            # frame-end target  [T, tips]
        f_meas = c["f_meas_steps"][:, -1, :]          # frame-end measured
        t = np.arange(len(f_ref)) * FRAME_DT
        lab = label_of(s)
        for i in range(n_tips):
            ax = axes[i]
            ax.plot(t, f_ref[:, i], "--", color=col, lw=1.0, alpha=0.9, label=f"target ({lab})" if i == 0 else None)
            ax.plot(t, f_meas[:, i], "-", color=col, lw=1.2, label=f"measured ({lab})" if i == 0 else None)
        keys = list(c["object_keys"]); oi = keys.index("object")
        z = c["object_pos"][:, oi, 2]
        axes[-1].plot(t, 100 * (z - z[0]), color=col, lw=1.4, label=lab)
        slip = s.get("grasp_slip") or {}
        if slip.get("dropped") and slip.get("drop_frame") is not None:
            fr = int(slip["drop_frame"])
            for ax in axes:
                ax.axvline(fr * FRAME_DT, color=col, ls=":", lw=1.0, alpha=0.8)
            axes[-1].annotate("drop", (fr * FRAME_DT, 100 * (z[min(fr, len(z) - 1)] - z[0])), color=col, fontsize=8)
        task = (s.get("outcome") or {}).get("task") or {}
        if task.get("inserted") and task.get("inserted_frame") is not None:
            axes[-1].axvline(task["inserted_frame"] * FRAME_DT, color=col, ls="-.", lw=1.0, alpha=0.8)
            axes[-1].annotate("inserted", (task["inserted_frame"] * FRAME_DT, 100 * (z[-1] - z[0])), color=col, fontsize=8)
    for i in range(n_tips):
        axes[i].set_ylabel(f"{tips[i]}\n|F| [N]")
        axes[i].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_ylabel("object height\nvs start [cm]")
    axes[-1].set_xlabel("time [s]")
    axes[-1].grid(alpha=0.3)
    axes[-1].legend(fontsize=8, loc="upper left")
    title = a.title or f"{runs[0][2]['run']}: force target (dashed) vs measured (solid)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out) if a.out else runs[0][0] / "force_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
