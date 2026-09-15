#!/usr/bin/env python3
"""Controller evaluation over the perfect episodes: run each force law on every perfect episode
(one Kit process per rollout, sequentially) and aggregate the rollouts' ``summary.json`` into one
table.

    # everything perfect under outputs/play2perfect/episodes, null + task_space, videos for task_space
    python play2perfect_force_controller/evaluate_controller.py
    # a subset, extra controller flags, or just re-aggregate finished rollouts
    python play2perfect_force_controller/evaluate_controller.py --problems tight_insertion --laws task_space \\
        --extra --offset-clip-rad 0.5
    python play2perfect_force_controller/evaluate_controller.py --report-only

Writes ``outputs/play2perfect/force_controller/evaluation_<stamp>.{md,json}``. Metrics per rollout:
force bias / rmse per fingertip (engaged steps), vector error, contact-point error, hand offset
cost |dq|, unintended object slip in the palm frame with the drop flag, and where the part ended
relative to the recording.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from play2perfect_force_controller.metrics import task_success_metrics  # noqa: E402
from play2perfect_force_controller.p2p_paths import EPISODE_ROOT, PROBLEMS  # noqa: E402
import numpy as np  # noqa: E402

EPISODES_ROOT = EPISODE_ROOT                     # per hand (ISAACSIMENVS_HAND)
ROLLOUTS_ROOT = PROJECT_ROOT / "outputs" / "play2perfect" / "force_controller"
PYTHON = Path(sys.executable)
ROLLOUT_TIMEOUT_S = 900          # a rendered oracle rollout takes ~3 min; a hung Kit never returns


def perfect_episodes(problems: list[str]) -> list[Path]:
    eps = []
    for problem in problems:
        for summary in sorted((EPISODES_ROOT / problem).glob("*/ep_*/summary.json")):
            s = json.loads(summary.read_text())
            if s["outcome"].get("perfect") and s["outcome"].get("retract_success"):
                eps.append(summary.parent)
    return eps


def run_rollout(episode: Path, law: str, render: bool, extra: list[str], tag: str,
                policy: str = "replay") -> Path | None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = json.loads((episode / "summary.json").read_text())["run"]
    out_dir = ROLLOUTS_ROOT / run_name / f"{stamp}_{tag}_{law}"
    cmd = [str(PYTHON), str(PROJECT_ROOT / "play2perfect_force_controller" / "run_tracking.py"),
           "--episode", str(episode), "--force-law", law, "--out-dir", str(out_dir)]
    if policy != "replay":
        cmd += ["--policy", policy]
    cmd += extra
    if not render:
        cmd.append("--no-render")
    print(f"[eval] {run_name} law={law}: {' '.join(cmd[2:])}", flush=True)
    for attempt in range(3):
        try:
            # Kit occasionally hangs at start-up on this host (one thread spinning, nothing logged);
            # no rollout legitimately takes this long, so kill it (SIGKILL: Kit ignores SIGTERM) and retry
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  timeout=ROLLOUT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"[eval]   TIMEOUT after {ROLLOUT_TIMEOUT_S} s (Kit hang?); attempt {attempt + 1}/3", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            time.sleep(30)
            continue
        # a PhysX GPU allocation failure (another Kit process holding the GPU) leaves a rollout with
        # frozen physics and exit code 0: treat it as a failure and retry after a pause. Kernel-launch
        # errors in the first frames of a rendered rollout (the renderer's first allocations) are
        # transient on a shared GPU - such rollouts still reproduce the recording bit for bit - so
        # only a failed device-memory allocation / CUDA error, or errors after the second progress
        # line (frame 121, two seconds in), count
        lines = proc.stdout.splitlines()
        progress = [i for i, l in enumerate(lines) if l.startswith("[force] frame")]
        settled = progress[1] if len(progress) > 1 else len(lines)
        sticky = ("PxgCudaDeviceMemoryAllocator failed", "CUDA error")
        physx_errors = [l for i, l in enumerate(lines) if "PhysX error" in l
                        and (i > settled or any(k in l for k in sticky))]
        if physx_errors:
            print(f"[eval]   PhysX errors ({len(physx_errors)} lines, e.g. {physx_errors[0][-120:]}); "
                  f"attempt {attempt + 1}/3", flush=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            time.sleep(60)
            continue
        break
    if not (out_dir / "summary.json").is_file():
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        print(f"[eval]   FAILED (exit {proc.returncode}):\n{tail}", flush=True)
        return None
    return out_dir


def score_task(rollout_dir: Path, episode_dir: Path) -> dict:
    """The env's insertion test, applied to the saved rollout against the recording's final goal."""
    c = np.load(rollout_dir / "controller_data.npz", allow_pickle=True)
    e = np.load(episode_dir / "replay_data.npz", allow_pickle=True)
    keys, ekeys = list(c["object_keys"]), list(e["object_keys"])
    tracked, tips = list(c["tracked_links"]), list(c["fingertip_bodies"])
    tip_ids = [tracked.index(b) for b in tips if b in tracked]
    return task_success_metrics(
        c["object_pos"][:, keys.index("object")], c["object_quat"][:, keys.index("object")],
        e["object_pos"][-1, ekeys.index("goal_viz")], e["object_quat"][-1, ekeys.index("goal_viz")],
        fingertip_pos=c["body_pos"][:, tip_ids] if tip_ids else None,
    )


def collect_rows(problems: list[str]) -> list[dict]:
    rows = []
    for summary in sorted(ROLLOUTS_ROOT.glob("*/*/summary.json")):
        s = json.loads(summary.read_text())
        if s.get("problem") not in problems or summary.parent.parent.name != s.get("run"):
            continue
        ep = Path(s["episode"])
        ep_summary = json.loads((ep / "summary.json").read_text()) if (ep / "summary.json").is_file() else {}
        if not (ep_summary.get("outcome", {}).get("perfect") and ep_summary["outcome"].get("retract_success")):
            continue
        ctrl = s["controller"]
        law = ctrl["force_law"]["law"]
        task = score_task(summary.parent, ep)
        ft, slip, out = s["force_tracking"], s.get("grasp_slip") or {}, s["outcome"]
        # "exact_replay" only when the middle layer rebuilds the recorded commands 1:1 (commands +
        # zero-order hold + null law); commands served through the default linear interpolation
        # are a distinct diagnostic and are labelled "<law>(cmd)".
        src = ctrl["policy"]["state_source"]
        interp = ctrl.get("reference", {}).get("interp", "linear")
        oracle = ctrl["policy"].get("source") == "oracle"
        exact = src == "joint_target" and interp == "hold" and law == "null" and not oracle
        if exact:
            base = "exact_replay"
        else:
            base = law + (("(cmd" + (",hold" if interp == "hold" else "") + ")") if src == "joint_target" else "")
        fl = ctrl["force_law"]
        if law == "task_space" and fl.get("map") == "constrained":
            base += "/cmap"
        if law == "task_space" and (fl.get("point_engage_n", 0) > 0 or fl.get("point_steady_rate", 0) > 0):
            base += "/gate"
        if oracle:      # closed loop: the RL policy re-planned every C frames
            base = f"oracle(C{ctrl['policy']['chunk']}):" + base
        env_v = (out.get("env") or {}).get("last") if isinstance(out, dict) else None
        bodies = ft["bodies"]
        name = summary.parent.name                       # <YYYYMMDD>_<HHMMSS>_<tag>_<law>
        tag = name[16:-len(law) - 1] if name.endswith("_" + law) and len(name) > 16 else ""
        rows.append({
            "problem": s["problem"], "run": s["run"], "rollout": summary.parent.name, "tag": tag,
            "law": base + (f"[{tag}]" if tag else ""),
            "clip_rad": ("effort" if ctrl["force_law"].get("offset_clip_from_effort") else ctrl["force_law"]["offset_clip_rad"]),
            "point_kp": ctrl["force_law"]["point_kp"],
            "frames": s["num_frames"],
            "bias_N": {b.replace("left_", "").replace("_DP", ""): round(v, 2) for b, v in zip(bodies, ft["bias_engaged_N"])},
            "rmse_N": {b.replace("left_", "").replace("_DP", ""): round(v, 2) for b, v in zip(bodies, ft["rmse_engaged_N"])},
            "worst_bias_N": round(min(ft["bias_engaged_N"]), 2),
            "dq_hand_mean_mrad": round(s["command_fidelity"]["offset_mean_mrad"], 1) if "offset_mean_mrad" in s.get("command_fidelity", {}) else None,
            "slip_mean_mm": round(slip.get("dev_mean_mm", float("nan")), 1) if slip else None,
            "slip_max_mm": round(slip.get("dev_max_mm", float("nan")), 1) if slip else None,
            "dropped": slip.get("dropped") if slip else None,
            "drop_frame": slip.get("drop_frame") if slip else None,
            "env_verdict": (f"{env_v['successes']}/{env_v['max_goals']}" + ("+R" if env_v["retract_succeeded"] else ""))
            if env_v else "-",
            "env_retract": bool(env_v["retract_succeeded"]) if env_v else None,
            "inserted": task["inserted"], "inserted_frame": task["inserted_frame"],
            "retract_success": task["retract_success"],
            "min_keypoint_dist_mm": round(task["min_keypoint_dist_mm"], 1),
            "final_keypoint_dist_mm": round(task["final_keypoint_dist_mm"], 1),
            "final_pos_err_mm": round(1000 * out["manipulated_final_position_error_vs_episode_m"], 1)
            if out.get("manipulated_final_position_error_vs_episode_m") is not None else None,
            "max_lift_mm": round(1000 * out["manipulated_max_lift_m"], 1) if out.get("manipulated_max_lift_m") is not None else None,
            "episode_lift_mm": round(1000 * out["episode_manipulated_max_lift_m"], 1) if out.get("episode_manipulated_max_lift_m") is not None else None,
        })
    return rows


def write_report(rows: list[dict], path_md: Path, path_json: Path) -> str:
    path_json.write_text(json.dumps(rows, indent=1))
    lines = ["# Force-controller evaluation on perfect play2perfect episodes", "",
             f"{len(rows)} rollouts. bias = mean(measured − target) over engaged steps, worst fingertip; "
             "INSERTED = the play2perfect env's own success test (fixed-size keypoints of the part within "
             "insertion_tolerance x keypoint_scale = 15 mm of the FINAL goal for 10 consecutive frames; "
             "retract = fingertips then > 10 cm away while the part stays within 7.5 mm); slip = unintended "
             "object motion in the palm frame vs the recording; dropped = slip detector lost the grasp.", "",
             "| problem | episode | law | clip | frames | INSERTED (env test) | env goals | closest / final keypoint mm | worst bias N | slip mean/max mm | dropped | lift mm (ep) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        drop = "-" if r["dropped"] is None else (f"yes @f{r['drop_frame']}" if r["dropped"] else "no")
        slip = "-" if r["slip_mean_mm"] is None else f"{r['slip_mean_mm']} / {r['slip_max_mm']}"
        ins = (f"YES @f{r['inserted_frame']}, retract {'ok' if r['retract_success'] else 'no'}" if r["inserted"] else "no")
        lines.append(f"| {r['problem']} | {r['run'].split('_', 1)[1]} | {r['law']} | {r['clip_rad']} | {r['frames']} | {ins} | {r.get('env_verdict', '-')} | "
                     f"{r['min_keypoint_dist_mm']} / {r['final_keypoint_dist_mm']} | "
                     f"{r['worst_bias_N']} | {slip} | {drop} | {r['max_lift_mm']} ({r['episode_lift_mm']}) |")
    # per problem / law aggregates
    lines += ["", "## Per problem and law (mean over rollouts)", "",
              "| problem | law | n | inserted | env retract ok | worst bias N | slip mean mm | dropped | closest keypoint mm |", "|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["problem"], r["law"], r["clip_rad"]) for r in rows})
    for problem, law, clip in keys:
        sel = [r for r in rows if (r["problem"], r["law"], r["clip_rad"]) == (problem, law, clip)]
        mean = lambda k: (sum(r[k] for r in sel if r[k] is not None) / max(1, sum(r[k] is not None for r in sel)))  # noqa: E731
        dropped = sum(1 for r in sel if r["dropped"])
        inserted = sum(1 for r in sel if r["inserted"])
        env_ok = sum(1 for r in sel if r.get("env_retract"))
        env_col = f"{env_ok}/{len(sel)}" if any(r.get("env_retract") is not None for r in sel) else "-"
        lines.append(f"| {problem} | {law} (clip {clip}) | {len(sel)} | {inserted}/{len(sel)} | {env_col} | {mean('worst_bias_N'):.2f} | "
                     f"{mean('slip_mean_mm'):.1f} | {dropped}/{len(sel)} | {mean('min_keypoint_dist_mm'):.1f} |")
    text = "\n".join(lines) + "\n"
    path_md.write_text(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    parser.add_argument("--laws", nargs="+", default=["null", "task_space"], choices=["null", "task_space"])
    parser.add_argument("--render-laws", nargs="*", default=["task_space"], help="laws that get videos")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra run_tracking flags (after --extra)")
    parser.add_argument("--tag", default="eval")
    parser.add_argument("--policy", choices=["replay", "oracle"], default="replay",
                        help="chunk source for the rollouts: the recording (replay) or the RL policy "
                             "re-planned in the simulator (oracle)")
    parser.add_argument("--report-only", action="store_true", help="aggregate existing rollouts, run nothing")
    args = parser.parse_args()

    if not args.report_only:
        eps = perfect_episodes(args.problems)
        print(f"[eval] {len(eps)} perfect episodes: " + ", ".join(str(e.relative_to(EPISODES_ROOT)) for e in eps), flush=True)
        for ep in eps:
            for law in args.laws:
                run_rollout(ep, law, render=law in args.render_laws, extra=args.extra, tag=args.tag,
                            policy=args.policy)
    rows = collect_rows(args.problems)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = write_report(rows, ROLLOUTS_ROOT / f"evaluation_{stamp}.md", ROLLOUTS_ROOT / f"evaluation_{stamp}.json")
    print(text)
    print(f"[eval] report -> {ROLLOUTS_ROOT / f'evaluation_{stamp}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
