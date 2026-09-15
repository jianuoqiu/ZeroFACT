#!/usr/bin/env python3
"""Replay every ingested trajectory, one Isaac Sim process per run, and summarise the results.

A fresh process per run is deliberate: Isaac Sim 5.1 does not reliably tear a stage down and rebuild
it in-process, and a crash in one run then cannot take the whole batch with it.

Examples::

    python scripts/replay_all.py                          # all runs, headless, images + video
    python scripts/replay_all.py --no-render               # physics only, fastest
    python scripts/replay_all.py --runs run_2026-08-11_17-29-12 run_2026-05-18_00-47-02
    python scripts/replay_all.py --extra-args "--kinematic --max-frames 40"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--runs", nargs="*", default=None, help="subset of run names (default: all)")
    parser.add_argument("--python", default=sys.executable, help="python interpreter with Isaac Lab")
    parser.add_argument("--no-render", action="store_true", help="physics only (no images / video)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per run")
    parser.add_argument("--extra-args", default="", help="extra flags passed to replay_trajectory.py")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    runs_dir = args.data_dir / "runs"
    runs = args.runs or sorted(p.name for p in runs_dir.iterdir() if (p / "run_meta.json").is_file())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = PROJECT_ROOT / "outputs" / "_batches" / stamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, run in enumerate(runs, start=1):
        out_dir = batch_dir / run
        cmd = [
            args.python,
            "-u",
            str(PROJECT_ROOT / "scripts" / "replay_trajectory.py"),
            "--run",
            run,
            "--out-dir",
            str(out_dir),
        ]
        if args.no_render:
            cmd.append("--no-render")
        if args.extra_args:
            cmd.extend(shlex.split(args.extra_args))

        log_path = batch_dir / f"{run}.log"
        print(f"[{i}/{len(runs)}] {run} -> {out_dir}", flush=True)
        start = time.time()
        timed_out = False
        with open(log_path, "w") as log:
            try:
                proc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False
                )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run already killed the child; keep the batch going
                timed_out = True
                returncode = -1
                log.write(f"\n[replay_all] timed out after {args.timeout}s\n")
        elapsed = time.time() - start

        summary_path = out_dir / "summary.json"
        entry = {
            "run": run,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_time_s": round(elapsed, 1),
            "log": str(log_path),
            "output_dir": str(out_dir),
        }
        if summary_path.is_file():
            with open(summary_path) as f:
                summary = json.load(f)
            entry["frames"] = summary.get("num_frames")
            entry["tracking_mean_abs_rad"] = summary.get("tracking", {}).get("mean_abs_rad")
            entry["tracking_max_abs_rad"] = summary.get("tracking", {}).get("max_abs_rad")
            entry["flow"] = {
                k: summary.get("flow", {}).get(k)
                for k in ["centroid_delta_error_mean_m", "centroid_delta_error_final_m", "sim_max_lift_m", "demo_max_lift_m"]
            }
            entry["objects"] = {
                name: {"moved_m": stats["total_displacement_m"], "max_lift_m": stats["max_lift_m"]}
                for name, stats in summary.get("objects", {}).items()
            }
        results.append(entry)

        status = "ok" if returncode == 0 else ("TIMED OUT" if timed_out else f"FAILED ({returncode})")
        print(f"      {status} in {elapsed:.0f}s   log: {log_path}", flush=True)
        if returncode != 0 and args.stop_on_error:
            break

    with open(batch_dir / "batch_summary.json", "w") as f:
        json.dump({"timestamp": stamp, "runs": results}, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'run':<45s} {'ok':>3s} {'frames':>7s} {'trk mean':>9s} {'flow err':>10s} {'max obj moved':>14s}")
    print("-" * 100)
    for entry in results:
        flow = (entry.get("flow") or {}).get("centroid_delta_error_mean_m")
        moved = max((o["moved_m"] for o in (entry.get("objects") or {}).values()), default=None)
        track = entry.get("tracking_mean_abs_rad")
        print(
            f"{entry['run']:<45s} "
            f"{('y' if entry['returncode'] == 0 else 'n'):>3s} "
            f"{str(entry.get('frames', '-')):>7s} "
            f"{(f'{track:.4f}' if track is not None else '-'):>9s} "
            f"{(f'{flow:.4f}' if flow is not None else '-'):>10s} "
            f"{(f'{moved:.4f}' if moved is not None else '-'):>14s}"
        )
    print("=" * 100)
    print(f"batch summary -> {batch_dir / 'batch_summary.json'}")
    return 0 if all(e["returncode"] == 0 for e in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
