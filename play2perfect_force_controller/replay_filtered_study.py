#!/usr/bin/env python3
"""Replay filtered episodes in the simulator and record the success rate.

The question this answers: does Savitzky-Golay smoothing keep the recorded data good enough to be
a behaviour-cloning target? It replays each accepted BC episode under several filtering conditions
and scores the outcome with the env's own insertion test (``metrics.task_success_metrics``, the
same test the episodes were accepted with).

Conditions (each is one cold-start Kit process per episode, ``run_tracking.py --exact-replay``,
which holds the recorded command for both physics substeps and runs no force law)::

    raw      nothing filtered - the regression baseline, must reproduce the recording bit for bit
    force    the contact channels filtered (this is what DEFAULT_FILTER_SPEC does); the commands
             are untouched, so any change here would mean the filter leaked into the trajectory
    traj5    the joint trajectory filtered with a 5 frame window - what a BC policy would emit if
    traj7    ... or a 7 frame window (the study's recommended offline setting)
    all7     both, window 7

``--closed-loop`` adds the oracle chunk policy (``--policy oracle --chunk 12 --horizon 12``), which
re-plans from the true simulator state every 12 frames. That is the regime a trained BC policy
actually runs in, so it separates "smoothing destroyed information" from "open-loop replay of a
contact-rich task is chaotic".

Usage::

    python play2perfect_force_controller/replay_filtered_study.py --episodes 8
    python play2perfect_force_controller/replay_filtered_study.py --episodes 3 --closed-loop
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

from play2perfect_force_controller.p2p_paths import PROBLEMS  # noqa: E402

PYTHON = Path("/home/jianuoqiu/anaconda3/envs/env_isaaclab/bin/python")
RUN_TRACKING = PROJECT_ROOT / "play2perfect_force_controller" / "run_tracking.py"
DATASET = PROJECT_ROOT / "outputs" / "play2perfect" / "bc_dataset"
DEFAULT_OUT = DATASET / "filtering" / "replay"

# tag -> extra run_tracking.py flags (on top of --exact-replay --no-render)
OPEN_LOOP_CONDITIONS = {
    "raw": [],
    "force": ["--filter-signals", "force"],
    "traj5": ["--filter-signals", "traj", "--filter-window", "5"],
    "traj7": ["--filter-signals", "traj", "--filter-window", "7"],
    "all7": ["--filter-signals", "all", "--filter-window", "7"],
}
# closed loop: the policy re-plans every chunk, so the state-source is its own prediction
CLOSED_LOOP_CONDITIONS = {
    "oracle_raw": ["--policy", "oracle", "--extra-marker"],
    "oracle_traj7": ["--filter-signals", "traj", "--filter-window", "7", "--policy", "oracle", "--extra-marker"],
}
CLOSED_LOOP_EXTRA = ["--chunk", "12", "--horizon", "12", "--state-source", "joint_target"]

TIMEOUT_S = 1800


def log(msg: str) -> None:
    print(f"[replay {datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def episodes_for(problem: str, n: int) -> list[Path]:
    return [p.parent for p in sorted((DATASET / problem).glob("seed_*/replay_data.npz"))[:n]]


def physx_errors(lines: list[str]) -> list[str]:
    progress = [i for i, l in enumerate(lines) if l.startswith("[force] frame")]
    settled = progress[1] if len(progress) > 1 else len(lines)
    sticky = ("PxgCudaDeviceMemoryAllocator failed", "CUDA error")
    return [l for i, l in enumerate(lines) if "PhysX error" in l and (i > settled or any(k in l for k in sticky))]


def run_one(ep: Path, tag: str, flags: list[str], out_root: Path, retries: int = 2) -> dict | None:
    out_dir = out_root / ep.parent.name / f"{ep.name}_{tag}"
    cmd = [str(PYTHON), str(RUN_TRACKING), "--episode", str(ep), "--exact-replay", "--no-render",
           "--out-dir", str(out_dir)]
    # "--extra-marker" is where run_tracking's own trailing flags must go (chunk/horizon/state-source)
    if "--extra-marker" in flags:
        i = flags.index("--extra-marker")
        cmd += flags[:i] + CLOSED_LOOP_EXTRA + flags[i + 1:]
    else:
        cmd += flags
    for attempt in range(1, retries + 2):
        shutil.rmtree(out_dir, ignore_errors=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=TIMEOUT_S)
            out, rc = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
            rc = -9
        lines = out.splitlines()
        errs = physx_errors(lines)
        if (out_dir / "summary.json").is_file() and not errs and rc == 0:
            s = json.loads((out_dir / "summary.json").read_text())
            task = s["outcome"]["task"]
            slip = s.get("grasp_slip", {})
            diff = s.get("diff_against", {}).get("max_abs_diff", {})
            return {
                "problem": ep.parent.name, "episode": ep.name, "condition": tag,
                "inserted": bool(task["inserted"]), "retract": bool(task["retract_success"]),
                "final_keypoint_mm": float(task["final_keypoint_dist_mm"]),
                "min_keypoint_mm": float(task["min_keypoint_dist_mm"]),
                "dropped": bool(slip.get("dropped", False)),
                "joint_target_diff_rad": float(diff.get("joint_target", {}).get("max_abs_diff", 0.0)),
                "joint_pos_diff_rad": float(diff.get("joint_pos", {}).get("max_abs_diff", 0.0)),
                "wall_s": round(time.time() - t0, 1),
            }
        why = "timeout" if rc == -9 else (f"PhysX errors ({len(errs)})" if errs else f"exit {rc}")
        log(f"  {ep.parent.name}/{ep.name} {tag}: {why}, attempt {attempt}/{retries + 1}")
        time.sleep(30 if errs else 10)
    return None


def aggregate(rows: list[dict], conditions: list[str]) -> list[dict]:
    table = []
    for tag in conditions:
        for problem in PROBLEMS + ["ALL"]:
            sel = [r for r in rows if r["condition"] == tag and (problem == "ALL" or r["problem"] == problem)]
            if not sel:
                continue
            n = len(sel)
            table.append({
                "condition": tag, "problem": problem, "n": n,
                "inserted": sum(r["inserted"] for r in sel),
                "inserted_and_retracted": sum(r["inserted"] and r["retract"] for r in sel),
                "dropped": sum(r["dropped"] for r in sel),
                "median_final_keypoint_mm": float(sorted(r["final_keypoint_mm"] for r in sel)[n // 2]),
                "max_joint_target_diff_rad": max(r["joint_target_diff_rad"] for r in sel),
            })
    return table


def write_report(out_root: Path, rows: list[dict], table: list[dict], conditions: list[str],
                 per_task: int, closed_loop: bool) -> None:
    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a / b:.0%})" if b else "-"

    lines = [
        "# Replaying filtered episodes: success rate",
        "",
        f"`replay_filtered_study.py`, {per_task} episodes per task, "
        f"{'closed loop (oracle chunk policy, re-plans every 12 frames)' if closed_loop else 'open loop (exact replay of the recorded commands)'}.",
        "Scored with the env's own insertion test against the recording's final goal pose.",
        "",
        "| condition | task | n | inserted | inserted + retracted | dropped | median final keypoint (mm) | max command change (rad) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        lines.append(f"| {r['condition']} | {r['problem']} | {r['n']} | {pct(r['inserted'], r['n'])} | "
                     f"{pct(r['inserted_and_retracted'], r['n'])} | {r['dropped']} | "
                     f"{r['median_final_keypoint_mm']:.1f} | {r['max_joint_target_diff_rad']:.4f} |")
    overall = {r["condition"]: r for r in table if r["problem"] == "ALL"}
    raw = overall.get("raw") or overall.get("oracle_raw")
    lines += [
        "",
        "## Reading this",
        "",
        "* `raw` is the regression baseline: replaying the recorded commands must reproduce the",
        "  recording bit for bit, so anything below 100 % there would mean the harness is broken,",
        "  not the filter.",
        "* `force` filters the contact channels only, which is what `DEFAULT_FILTER_SPEC` does. The",
        "  commands are untouched (`max command change` is 0), so the replay is identical to `raw`.",
        "  **This is the configuration the BC dataset is meant to be used with.**",
        "* `traj5` / `traj7` / `all7` smooth the joint trajectory itself. That is not part of the",
        "  recommended pipeline; it is here to measure how much trajectory precision these tasks need.",
        "",
    ]
    if raw:
        lines += [f"Baseline `{raw['condition']}`: {pct(raw['inserted'], raw['n'])} inserted, "
                  f"{pct(raw['inserted_and_retracted'], raw['n'])} inserted and retracted.", ""]
    (out_root / "REPORT.md").write_text("\n".join(lines))
    (out_root / "rows.json").write_text(json.dumps(rows, indent=2))
    (out_root / "table.json").write_text(json.dumps(table, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=8, help="episodes per task")
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    parser.add_argument("--conditions", nargs="+", default=None, help="subset of the condition tags")
    parser.add_argument("--closed-loop", action="store_true",
                        help="oracle chunk policy instead of open-loop replay")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--keep-rollouts", action="store_true", help="keep each rollout directory")
    args = parser.parse_args()

    conds = CLOSED_LOOP_CONDITIONS if args.closed_loop else OPEN_LOOP_CONDITIONS
    tags = args.conditions or list(conds)
    out_root = args.out_dir or (DEFAULT_OUT.parent / ("replay_closed_loop" if args.closed_loop else "replay"))
    out_root.mkdir(parents=True, exist_ok=True)
    rows_path = out_root / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.is_file() else []
    done = {(r["problem"], r["episode"], r["condition"]) for r in rows}

    jobs = [(ep, tag) for problem in args.problems for ep in episodes_for(problem, args.episodes)
            for tag in tags if (problem, ep.name, tag) not in done]
    log(f"{len(jobs)} rollouts to run ({len(tags)} conditions x {args.episodes} episodes x "
        f"{len(args.problems)} tasks), {len(rows)} already on disk -> {out_root}")
    t0 = time.time()
    for i, (ep, tag) in enumerate(jobs, 1):
        r = run_one(ep, tag, conds[tag], out_root)
        if r is None:
            log(f"[{i}/{len(jobs)}] {ep.parent.name}/{ep.name} {tag}: FAILED")
            continue
        rows.append(r)
        rows_path.write_text(json.dumps(rows, indent=2))
        eta = (time.time() - t0) / i * (len(jobs) - i) / 60
        log(f"[{i}/{len(jobs)}] {ep.parent.name[:18]}/{ep.name} {tag:12s} "
            f"inserted={'yes' if r['inserted'] else 'NO '} retract={'yes' if r['retract'] else 'no '} "
            f"final {r['final_keypoint_mm']:7.1f} mm  ({r['wall_s']}s, eta {eta:.0f} min)")
        if not args.keep_rollouts:
            shutil.rmtree(out_root / ep.parent.name / f"{ep.name}_{tag}", ignore_errors=True)

    table = aggregate(rows, tags)
    write_report(out_root, rows, table, tags, args.episodes, args.closed_loop)
    log("summary:")
    for r in table:
        if r["problem"] == "ALL":
            log(f"  {r['condition']:12s} inserted {r['inserted']:3d}/{r['n']:<3d} "
                f"({r['inserted'] / r['n']:.0%})  +retract {r['inserted_and_retracted']:3d}/{r['n']:<3d}  "
                f"dropped {r['dropped']:3d}  median final {r['median_final_keypoint_mm']:7.1f} mm")
    log(f"report -> {out_root / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
