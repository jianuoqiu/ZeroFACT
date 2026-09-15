#!/usr/bin/env python3
"""Build a behaviour-cloning dataset of VERIFIED successful play2perfect episodes (Sharpa hand).

For every problem, seeds are tried in order (default from 1000) with ONE cold-start Kit process
per seed (``collect_episodes.py --strict --flat``): every episode is a distinct reset draw of the
scene (object pose, hole pose), replays bit-exactly from its saved initial state, and a Kit hang
costs one seed - until ``--target`` accepted episodes exist per problem.

Acceptance = the env's own verdict (every insertion sub-goal reached AND the retract succeeded)
AND ``bc_dataset.validate_records`` (clean termination, part at the goal at the last frame, the
insertion test re-run offline, physics / image sanity), re-checked from the saved files
(``validate_episode_dir``) by this driver. Rejected seeds are logged with the reason (and the log
tail) in ``<root>/<problem>/rejected.json`` and their directories removed. Resumable: accepted
episodes and rejected seeds already on disk are skipped.

Layout::

    <root>/
        dataset_index.json        accepted episodes (problem, seed, frames, images, ...)
        README.md                 conventions + counts (written at the end / --review-only)
        progress.json             live progress per problem
        STOP                      create it to stop after the running seed
        <problem>/seed_<k>/       one accepted episode (see bc_dataset.py for the layout)
        <problem>/rejected.json   seed -> reason
        <problem>/logs/seed_<k>.log   Kit log (full for accepted seeds, tail for rejected)
        <problem>/review_sheet.png    thumbnails of every accepted episode

Run detached so neither the Claude harness nor a desktop OOM event kills it::

    PY=/home/jianuoqiu/anaconda3/envs/env_isaaclab/bin/python
    systemd-run --user --scope --property=OOMPolicy=continue --unit bc-collect-$(date +%s) -- \\
        setsid -f bash -c "cd ~/ZeroFACT && $PY play2perfect_force_controller/collect_bc_dataset.py \\
        --target 50 > outputs/play2perfect/bc_dataset/driver.log 2>&1 < /dev/null"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from play2perfect_force_controller.bc_dataset import (  # noqa: E402
    MOTOR_RATED_CURRENT_A, SANITY, build_index, validate_episode_dir, write_final_frame_grid,
    write_review_sheet,
)
from play2perfect_force_controller.p2p_paths import PROBLEMS  # noqa: E402

PYTHON = Path("/home/jianuoqiu/anaconda3/envs/env_isaaclab/bin/python")
COLLECT = PROJECT_ROOT / "play2perfect_force_controller" / "collect_episodes.py"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "play2perfect" / "bc_dataset"


def log(msg: str) -> None:
    print(f"[bc {datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def mem_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return -1.0


def wait_for_ram(min_gb: float, max_wait_s: float = 1800.0) -> None:
    t0 = time.time()
    warned = False
    while (avail := mem_available_gb()) >= 0 and avail < min_gb and time.time() - t0 < max_wait_s:
        if not warned:
            log(f"waiting for host RAM: {avail:.1f} GB available < {min_gb:.0f} GB")
            warned = True
        time.sleep(30)


def _prefer_as_oom_victim() -> None:
    """Run in the child before exec: if the host runs out of memory (swap is full here and a training
    job shares the box), the kernel should kill THIS Kit process - the driver retries the seed - and
    never the training."""
    try:
        Path("/proc/self/oom_score_adj").write_text("800")
    except OSError:
        pass


def physx_errors(lines: list[str]) -> list[str]:
    """Same rule as evaluate_controller.run_rollout: kernel-launch noise in the first frames of a
    rendered run is transient on the shared GPU; a failed device allocation / CUDA error, or any
    PhysX error after the second progress line, is not."""
    progress = [i for i, l in enumerate(lines) if l.startswith("[collect] ep") and " frame " in l]
    settled = progress[1] if len(progress) > 1 else len(lines)
    sticky = ("PxgCudaDeviceMemoryAllocator failed", "CUDA error")
    return [l for i, l in enumerate(lines) if "PhysX error" in l and (i > settled or any(k in l for k in sticky))]


def run_seed(problem: str, seed: int, root: Path, args) -> tuple[str, dict]:
    """Returns (status, info); status in accepted / rejected / error."""
    ep_dir = root / problem / f"seed_{seed}"
    log_dir = root / problem / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"seed_{seed}.log"
    cmd = [str(PYTHON), str(COLLECT), "--problem", problem, "--episodes", "1", "--seed", str(seed),
           "--strict", "--max-attempts", "1", "--flat", "--out-dir", str(ep_dir),
           "--image-every", str(args.image_every), "--image-format", args.image_format,
           "--goal-marker", args.goal_marker, "--videos", args.videos] + list(args.extra)
    info: dict = {"seed": seed, "attempts": 0}
    status = "error"
    for attempt in range(1, args.retries + 2):
        info["attempts"] = attempt
        wait_for_ram(args.min_avail_gb)
        shutil.rmtree(ep_dir, ignore_errors=True)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=args.timeout, preexec_fn=_prefer_as_oom_victim)
            out = proc.stdout
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) + \
                  f"\n[bc] TIMEOUT after {args.timeout} s (Kit hang?) - killed\n"
            rc = -9
        info["wall_s"] = round(time.time() - t0, 1)
        lines = out.splitlines()
        log_path.write_text(out)
        errs = physx_errors(lines)
        killed = rc in (-9, 137) and "[bc] TIMEOUT" not in out
        if rc == -9 and not killed:
            info["reason"] = f"timeout after {args.timeout} s"
        elif killed:
            info["reason"] = "killed by SIGKILL (host OOM killer, most likely)"
        elif errs:
            info["reason"] = f"PhysX errors ({len(errs)}): {errs[0][-160:]}"
        if killed or errs or (rc == -9):
            log(f"{problem} seed {seed}: {info['reason']} - attempt {attempt}/{args.retries + 1}")
            shutil.rmtree(ep_dir, ignore_errors=True)
            time.sleep(120 if killed else (60 if errs else 20))
            continue
        if (ep_dir / "summary.json").is_file():
            report = validate_episode_dir(ep_dir)
            summary = json.loads((ep_dir / "summary.json").read_text())
            o = summary["outcome"]
            info.update({"frames": o["num_frames"], "sim_time_s": round(o["sim_time_s"], 2),
                         "max_force_N": round(o["max_fingertip_force_on_object_N"], 1),
                         "images": len(summary.get("bc", {}).get("image_state_index", []))})
            if report["ok"] and summary.get("validation", {}).get("ok"):
                return "accepted", info
            info["reason"] = "post-validation failed: " + ", ".join(report["failed"] or summary.get("validation", {}).get("failed", []))
            info["checks"] = {k: report["checks"][k] for k in report["failed"]}
            shutil.rmtree(ep_dir, ignore_errors=True)
            _truncate_log(log_path, lines)
            return "rejected", info
        m = re.search(r"\[collect\] attempt 1: (.*) - discarded", out)
        if m:
            info["reason"] = m.group(1)
            shutil.rmtree(ep_dir, ignore_errors=True)
            _truncate_log(log_path, lines)
            return "rejected", info
        tail = [l for l in lines[-40:] if l.strip()]
        info["reason"] = f"no episode written (exit {rc}); log tail: " + " | ".join(tail[-6:])[-600:]
        log(f"{problem} seed {seed}: {info['reason'][:200]} - attempt {attempt}/{args.retries + 1}")
        shutil.rmtree(ep_dir, ignore_errors=True)
        time.sleep(30)
    return status, info


def _truncate_log(log_path: Path, lines: list[str], keep: int = 300) -> None:
    if len(lines) > keep:
        log_path.write_text("\n".join(["[bc] log truncated to the last %d lines" % keep] + lines[-keep:]) + "\n")


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def accepted_seeds(root: Path, problem: str) -> dict[int, dict]:
    out = {}
    for summary in sorted((root / problem).glob("seed_*/summary.json")):
        s = json.loads(summary.read_text())
        if s.get("validation", {}).get("ok"):
            out[int(summary.parent.name.split("_")[1])] = {"frames": s["outcome"]["num_frames"]}
    return out


def write_readme(root: Path, index: dict, args) -> None:
    counts = index["counts"]
    lines = [
        "# play2perfect BC dataset (KUKA iiwa14 + Sharpa hand)",
        "",
        f"Generated by `play2perfect_force_controller/collect_bc_dataset.py` (finished {datetime.now():%Y-%m-%d %H:%M}).",
        "Policy: the released play2perfect SAPG checkpoints (`~/play2perfect/pretrained_assembly/<problem>/model.pth`),",
        "deterministic (mean action), conditioned on the exploit block; env = play2perfect PreciseAssembly with the",
        "evaluation overrides (domain randomisation off: no obs/action delay, no random wrenches, no joint noise;",
        "the reset draws of object pose, hole pose and table height stay on). One cold-start Kit process per episode,",
        f"seeds from {args.seed0} (distinct per episode; the seed is `config.seed` in summary.json).",
        "",
        "## Contents",
        "",
        "| problem | accepted episodes | frames (min / mean / max) | sim time s (mean) |",
        "|---|---|---|---|",
    ]
    for problem in PROBLEMS:
        eps = [e for e in index["episodes"] if e["problem"] == problem]
        if not eps:
            lines.append(f"| {problem} | 0 | - | - |")
            continue
        fr = [e["num_frames"] for e in eps]
        lines.append(f"| {problem} | {len(eps)} | {min(fr)} / {sum(fr) / len(fr):.0f} / {max(fr)} | "
                     f"{sum(e['sim_time_s'] for e in eps) / len(eps):.1f} |")
    lines += [
        "",
        "Every episode directory `<problem>/seed_<k>/` holds `replay_data.npz`, `summary.json`, `rgb_images/`,",
        "`rollout.mp4`, `rollout_with_forces.mp4` and the three plots. `dataset_index.json` lists the accepted",
        "episodes; `<problem>/review_sheet.png` shows five thumbnails per accepted episode; `<problem>/rejected.json`",
        "lists every rejected seed with the reason.",
        "",
        "## Acceptance (why the videos are optional)",
        "",
        "An episode is kept only if ALL of these hold (checked inside the recorder and again from the saved files):",
        "",
        "1. the env's own verdict: every insertion sub-goal reached (fixed-size keypoints within",
        "   `insertion_success_tolerance` x 1.5 for 10 consecutive steps) AND the retract succeeded (fingertips > 10 cm",
        "   away while the part stays within `retract_success_tolerance` x 1.5 = 7.5 mm of the final goal);",
        "2. termination reason exactly `max_successes` (no fall / hand_far / dropped / timeout / forced cut-off);",
        "3. part within 7.5 mm keypoint distance of the final goal at the LAST frame;",
        "4. the insertion test re-run offline on the recorded object trajectory vs the recorded final goal pose;",
        f"5. physics sanity: |joint vel| <= {SANITY['max_joint_vel_rad_s']} rad/s, fingertip force <= {SANITY['max_fingertip_force_n']} N,",
        f"   part speed <= {SANITY['max_object_speed_m_s']} m/s, untouched part <= {SANITY['max_free_object_speed_m_s']} m/s (no reset bounce),",
        f"   part moved >= {SANITY['min_displacement_m'] * 100:.0f} cm or turned >= {SANITY['min_rotation_deg']:.0f} deg, >= {SANITY['min_frames']} frames, all values finite;",
        "6. image sanity: frames on disk match `image_state_index`, not black, not frozen.",
        "",
        "The verdict and every measured value are in `summary.json['validation']`.",
        "",
        "## Time convention",
        "",
        "T frames of 1/60 s (policy steps), S = 2 physics substeps of 1/120 s each.",
        "",
        "```",
        "frame t:  obs_policy[t] -> action[t] -> joint_target[t] (held for both substeps)",
        "          -> joint_pos[t], contact_force[t], object_pos[t], ... = state at the END of frame t",
        "state index s: s = 0 is the reset state (init_* arrays), s = t + 1 is the state at the end of frame t",
        "rgb_images/<s>.png = camera at state s (every %d-th state, s = 0 included -> %d Hz); image_state_index lists them" % (args.image_every, round(60 / args.image_every)),
        "*_steps arrays are [T, S, ...] (substeps in order); the per-frame array is the last substep",
        "```",
        "",
        "`bc_dataset.load_episode(ep_dir)` returns state-aligned arrays (`states[key][s]`, `actions[t]`, image paths).",
        "",
        "## Signals in replay_data.npz",
        "",
        "* proprio: `joint_pos`, `joint_vel`, `joint_target` [T, 29] (`joint_names` order: 7 arm + 22 hand), `body_pos`/`body_quat`",
        "  [T, 6, ...] (`tracked_links`: palm link + 5 fingertips), `action` [T, 29], `obs_policy` [T, D] (`summary.bc.obs_fields`);",
        "* force: `contact_force` [T, 5, 3] net fingertip force (world), `contact_force_steps` [T, 2, 5, 3],",
        "  `contact_object_force_steps` [T, 2, 5, 3 objects, 3] (object / hole / table), `contact_point_w` [T, 5, 3, 3] (NaN off-contact);",
        "* scene: `object_pos`/`object_quat`/`object_lin_vel` [T, 4, ...] for object, hole, table, goal_viz (wxyz quaternions),",
        "  `keypoints_max_dist`, `successes`, `retract_phase`, `init_*` reset state;",
        "* fake tactile (per frame and `*_steps` per substep):",
        "  * `applied_torque` [N m] implicit-PD drive torque K (q_target - q) - D qd clipped to the URDF effort limit",
        "    (`joint_effort_limit_nm`); `computed_torque` unclipped;",
        "  * `joint_torque_measured` [N m] PhysX projected joint force (reaction along the joint axis: drive + gravity + contact);",
        "  * `motor_current` [A] = applied_torque / `motor_kt`, `motor_kt` = effort limit / rated current",
        f"    (rated current {MOTOR_RATED_CURRENT_A['arm']:.0f} A arm, {MOTOR_RATED_CURRENT_A['hand']:.0f} A hand: FAKE constants, rescale as needed);",
        "  * `joint_cmd_err` [rad] = joint_target - joint_pos (state-command difference);",
        "  * `joint_wrench_b` [T, bodies, 6] incoming joint wrench of every body (`body_names`) in the joint child frame;",
        "  * static: `joint_stiffness`, `joint_damping`, `joint_armature`, `joint_pos_limits`, `joint_effort_limits`.",
        "",
        "Images are the fixed demo camera (`summary.bc.camera`), 640x480, with NO force arrows and the goal-pose",
        "marker hidden (`--goal-marker hidden`); the marker's pose is still recorded (`object_keys` index of `goal_viz`).",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))


def finalize(root: Path, args) -> dict:
    index = build_index(root, revalidate=args.revalidate)
    for problem in PROBLEMS:
        if (root / problem).is_dir():
            sheet = write_review_sheet(root / problem)
            if sheet:
                log(f"review sheet -> {sheet}")
            for state in ("final", "reset"):
                grid = write_final_frame_grid(root / problem, state=state)
                if grid:
                    log(f"{state} frame grid -> {grid}")
    write_readme(root, index, args)
    log(f"index -> {root / 'dataset_index.json'}: {index['counts']}")
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--problems", nargs="+", default=PROBLEMS, choices=PROBLEMS)
    parser.add_argument("--target", type=int, default=50, help="accepted episodes per problem")
    parser.add_argument("--seed0", type=int, default=1000, help="first seed to try")
    parser.add_argument("--max-seeds", type=int, default=400, help="seeds to try per problem before giving up")
    parser.add_argument("--image-every", type=int, default=2)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--goal-marker", choices=["hidden", "translucent", "opaque"], default="hidden")
    parser.add_argument("--videos", choices=["none", "review", "all"], default="review")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per Kit process before it is killed")
    parser.add_argument("--retries", type=int, default=2, help="extra attempts per seed after a hang / PhysX error / crash")
    parser.add_argument("--min-avail-gb", type=float, default=12.0, help="host RAM to wait for before launching Kit")
    parser.add_argument("--extra", nargs="*", default=[], help="extra collect_episodes.py flags")
    parser.add_argument("--review-only", action="store_true", help="only rebuild the index, review sheets and README")
    parser.add_argument("--revalidate", action="store_true", help="re-run validate_episode_dir on every episode when indexing")
    args = parser.parse_args()

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        finalize(root, args)
        return 0
    stop_file = root / "STOP"
    progress_path = root / "progress.json"
    progress = load_json(progress_path, {})
    progress["started"] = progress.get("started") or datetime.now().isoformat(timespec="seconds")
    progress["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    log(f"root {root}; problems {args.problems}; target {args.target}/problem; seeds from {args.seed0}")

    for problem in args.problems:
        pdir = root / problem
        pdir.mkdir(exist_ok=True)
        rejected_path = pdir / "rejected.json"
        rejected = {int(k): v for k, v in load_json(rejected_path, {}).items()}
        accepted = accepted_seeds(root, problem)
        errors = {int(k): v for k, v in load_json(pdir / "errors.json", {}).items()}
        log(f"{problem}: {len(accepted)} accepted, {len(rejected)} rejected, {len(errors)} errors already on disk")
        seed = args.seed0
        tried = 0
        t_problem = time.time()
        while len(accepted) < args.target and tried < args.max_seeds:
            if stop_file.is_file():
                log("STOP file found - exiting after the current seed")
                progress["stopped"] = datetime.now().isoformat(timespec="seconds")
                progress_path.write_text(json.dumps(progress, indent=2))
                finalize(root, args)
                return 0
            if seed in accepted or seed in rejected or seed in errors:
                seed += 1
                continue
            tried += 1
            progress[problem] = {"accepted": len(accepted), "rejected": len(rejected), "errors": len(errors),
                                 "running_seed": seed, "updated": datetime.now().isoformat(timespec="seconds")}
            progress_path.write_text(json.dumps(progress, indent=2))
            status, info = run_seed(problem, seed, root, args)
            if status == "accepted":
                accepted[seed] = info
                log(f"{problem} seed {seed}: ACCEPTED ({len(accepted)}/{args.target}) {info['frames']} frames, "
                    f"{info['sim_time_s']} s, peak |F| {info['max_force_N']} N, {info['images']} images, "
                    f"{info['wall_s']} s wall")
            elif status == "rejected":
                rejected[seed] = info
                rejected_path.write_text(json.dumps(rejected, indent=2))
                log(f"{problem} seed {seed}: rejected - {info['reason'][:160]} ({info['wall_s']} s)")
            else:
                errors[seed] = info
                (pdir / "errors.json").write_text(json.dumps(errors, indent=2))
                log(f"{problem} seed {seed}: ERROR - {info.get('reason', '?')[:160]}")
            seed += 1
            done = len(accepted)
            rate = (time.time() - t_problem) / max(1, tried)
            progress[problem] = {"accepted": done, "rejected": len(rejected), "errors": len(errors),
                                 "tried_this_run": tried, "next_seed": seed, "s_per_seed": round(rate, 1),
                                 "eta_min": round(rate * max(0, args.target - done) / max(0.05, done / tried) / 60, 1),
                                 "updated": datetime.now().isoformat(timespec="seconds")}
            progress_path.write_text(json.dumps(progress, indent=2))
        log(f"{problem}: {len(accepted)} accepted after {tried} seeds this run "
            f"({len(rejected)} rejected, {len(errors)} errors total)")
    progress["finished"] = datetime.now().isoformat(timespec="seconds")
    progress_path.write_text(json.dumps(progress, indent=2))
    finalize(root, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
