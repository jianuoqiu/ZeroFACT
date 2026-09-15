#!/usr/bin/env python3
"""Pack the force-controller evaluation into one shareable folder (videos, plots, metrics, gallery).

Settings packed (see SETTINGS below), plus an A/B/C column video per episode (compose_task_video.py):
  replay_reachedstates_forcelaw   the recorded episode replayed open loop, reached states served
  oracle_reachedstates_forcelaw   the RL policy re-planned every 12 frames in sim, reached states served
For each of the perfect episodes: the live force-tracking video, the camera video, a target-vs-
measured plot with the null-law rollout overlaid (with vs without the force law), summary.json and
a metrics.md. Top level: README.md with the aggregate tables and index.html as a gallery.

    python play2perfect_force_controller/package_results.py [--out outputs/play2perfect/force_controller/share_results]
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLL = PROJECT_ROOT / "outputs" / "play2perfect" / "force_controller"
PY = sys.executable
# rows of the A/B/C column video (compose_task_video.py): row -> (rollout tag, law)
ABC_ROLLOUTS = {"a": ("exactvid", "null"), "b": ("posvid", "null"), "c": ("eval", "task_space")}
SETTING_OF_ROW = {"a": "replay_commands_exact_nolaw", "b": "replay_reachedstates_nolaw", "c": "replay_reachedstates_forcelaw"}
# setting folder -> (rollout tag, law of the main rollout, tag of the OFF baseline or None, description)
SETTINGS = {
    "replay_commands_exact_nolaw": ("exactvid", "null", None,
                                    "Original action replay: the recorded motor COMMANDS replayed exactly (held per frame), no force law. "
                                    "Reproduces the recording bit for bit; the force target and the measured force coincide."),
    "replay_reachedstates_nolaw": ("posvid", "null", None,
                                   "State-only replay: the recorded REACHED joint states replayed, no force law. The position layer alone, "
                                   "without the preload the commands carried."),
    "replay_reachedstates_forcelaw": ("eval", "task_space", "eval",
                                      "Recorded episode replayed open loop (the recording plays the policy's part), "
                                      "reached states served, task-space force law on."),
    "oracle_reachedstates_forcelaw": ("o12pos", "task_space", "o12pos",
                                      "RL policy re-planned every 12 frames inside the simulator (closed loop at chunk rate), "
                                      "reached states served, task-space force law on."),
}


def verdict(s: dict, rollout_dir: Path | None = None) -> dict:
    o = s["outcome"]; t = o.get("task") or {}; env = (o.get("env") or {}).get("last") or {}
    if not t and rollout_dir is not None:      # older rollouts: score the saved data with the env's test
        sys.path.insert(0, str(PROJECT_ROOT))
        from play2perfect_force_controller.evaluate_controller import score_task
        ep = Path(s["episode"]); ep = ep if ep.is_absolute() else PROJECT_ROOT / ep
        t = score_task(rollout_dir, ep)
    slip = s.get("grasp_slip") or {}; ft = s["force_tracking"]
    return {
        "inserted (offline env test)": bool(t.get("inserted")),
        "env verdict": (f"{env.get('successes')}/{env.get('max_goals')} goals" + (", retract ok" if env.get("retract_succeeded") else "")) if env else "n/a (open-loop replay)",
        "closest keypoint distance mm": round(t.get("min_keypoint_dist_mm", float("nan")), 1),
        "worst fingertip force bias N": round(min(ft["bias_engaged_N"]), 2),
        "slip mean mm": round(slip.get("dev_mean_mm", float("nan")), 1) if slip else None,
        "dropped": slip.get("dropped") if slip else None,
        "frames": s["num_frames"],
    }


def completed(v: dict | None) -> bool:
    if not v:
        return False
    return v["env verdict"].endswith("retract ok") if "goals" in v["env verdict"] else bool(v["inserted (offline env test)"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROLL / "share_results"))
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    gallery = []
    agg = defaultdict(list)
    episodes = {}
    for setting, (tag, law, base_tag, desc) in SETTINGS.items():
        for run_dir in sorted(ROLL.glob("*_ep0000")):
            ts = sorted(run_dir.glob(f"*_{tag}_{law}")); nl = sorted(run_dir.glob(f"*_{base_tag}_null")) if base_tag else []
            if not ts:
                continue
            ts, nl = ts[-1], (nl[-1] if nl else None)
            s = json.loads((ts / "summary.json").read_text())
            prob = s["problem"]; ep = run_dir.name.replace(prob + "_", "")
            dst = out / setting / prob / ep; dst.mkdir(parents=True)
            if (ts / "rollout_force_tracking.mp4").is_file():
                shutil.copy(ts / "rollout_force_tracking.mp4", dst / "force_tracking_video.mp4")
            elif (ts / "rollout.mp4").is_file():
                subprocess.run([PY, str(PROJECT_ROOT / "play2perfect_force_controller" / "render_force_video.py"), str(ts),
                                "--out", str(dst / "force_tracking_video.mp4")], check=False, stdout=subprocess.DEVNULL)
            if (ts / "rollout.mp4").is_file():
                shutil.copy(ts / "rollout.mp4", dst / "camera_video.mp4")
            for name in ("force_tracking.png", "force_components.png", "contact_point_error.png", "commands_vs_predicted.png"):
                if (ts / name).is_file():
                    shutil.copy(ts / name, dst / name)
            if nl is not None:
                subprocess.run([PY, str(PROJECT_ROOT / "play2perfect_force_controller" / "plot_force_comparison.py"), str(nl), str(ts),
                                "--out", str(dst / "with_vs_without_forcelaw.png"),
                                "--title", f"{prob} {ep}: {setting} - null law (blue) vs task-space force law (red)"],
                               check=False, stdout=subprocess.DEVNULL)
            shutil.copy(ts / "summary.json", dst / "summary.json")
            # the recording itself: the released policy's own run that the rollout starts from / replays
            ep_dir = Path(s["episode"])
            if not ep_dir.is_absolute():
                ep_dir = PROJECT_ROOT / ep_dir
            ep_sum = json.loads((ep_dir / "summary.json").read_text())
            if (ep_dir / "rollout.mp4").is_file():
                shutil.copy(ep_dir / "rollout.mp4", dst / "recorded_episode_video.mp4")
            shutil.copy(ep_dir / "summary.json", dst / "recorded_episode_summary.json")
            ep_rel = str(ep_dir.relative_to(PROJECT_ROOT)) if ep_dir.is_relative_to(PROJECT_ROOT) else str(ep_dir)
            episodes[run_dir.name] = (ep_rel, ep_sum["config"]["seed"], ep_sum["outcome"]["num_frames"])
            v = verdict(s, ts); v_null = verdict(json.loads((nl / "summary.json").read_text()), nl) if nl else None
            lines = [f"# {prob} / {ep} / {setting}", "", desc, "", "| metric | force law ON (task-space) | force law OFF (null law, baseline) |", "|---|---|---|"]
            for k in v:
                lines.append(f"| {k} | {v[k]} | {v_null[k] if v_null else '-'} |")
            lines.insert(3, f"Recorded episode: `{episodes[run_dir.name][0]}` (collection seed {episodes[run_dir.name][1]}, {episodes[run_dir.name][2]} frames); its own video is `recorded_episode_video.mp4`.")
            lines += ["", "Files: `force_tracking_video.mp4` (camera + live target vs measured force), `camera_video.mp4`, "
                      "`with_vs_without_forcelaw.png` (target dashed, measured solid; blue = null law, red = force law), "
                      "`force_tracking.png`, `force_components.png`, `contact_point_error.png`, `summary.json`."]
            (dst / "metrics.md").write_text("\n".join(lines) + "\n")
            agg[(setting, prob)].append((v, v_null))
            gallery.append((setting, prob, ep, dst.relative_to(out), v))
    # A/B/C column video per episode: exact command replay, reached-state replay, reached states + force law
    sys.path.insert(0, str(PROJECT_ROOT))
    from play2perfect_force_controller.compose_task_video import ROWS, compose
    abc = []
    for run_dir in sorted(ROLL.glob("*_ep0000")):
        picks = {}
        for key, (tag, law) in ABC_ROLLOUTS.items():
            found = sorted(run_dir.glob(f"*_{tag}_{law}"))
            found = [f for f in found if (f / "rollout.mp4").is_file()]
            if found:
                picks[key] = found[-1]
        if len(picks) < 3:
            print(f"[pack] {run_dir.name}: A/B/C video skipped (have {sorted(picks)})", flush=True)
            continue
        prob = json.loads((picks["a"] / "summary.json").read_text())["problem"]; ep = run_dir.name.replace(prob + "_", "")
        dst = out / "abc_comparison" / prob / ep; dst.mkdir(parents=True)
        compose(picks, dst / "abc_comparison.mp4", label=f"{prob} / {ep}")
        abc.append((prob, ep, dst.relative_to(out)))
    R = ["# Force-controller evaluation on play2perfect assembly tasks", "",
         "Hybrid force/position middle layer (fingertip tactile + proprioception only) between a chunk policy and the",
         "joint PD of a KUKA iiwa14 + Sharpa hand. Twelve recorded 'perfect' episodes of the released play2perfect",
         "policies (three per task) are the test set. Four settings are shown: two baselines without the force law",
         "(the recorded commands replayed exactly, and the recorded reached states replayed), and two with the task-space",
         "force law where the policy serves *reached* joint states (no motor commands, the premise of a policy trained",
         "on human data) so the law has to supply the grasp preload from the force targets:", ""]
    for setting, (tag, law, base_tag, desc) in SETTINGS.items():
        R.append(f"* `{setting}/` — {desc}")
    R += ["", "`abc_comparison/<task>/<episode>/abc_comparison.mp4` stacks three of these for the same recording in one column, each row",
          "the camera plus the live force curves:", ""]
    for key in ("a", "b", "c"):
        letter, title, sub, _ = ROWS[key]
        R.append(f"* **{letter}. {title}** - {sub} (`{SETTING_OF_ROW[key]}/`)")
    R += ["", "The target pose is shown as a faint translucent copy of the part (same colour, 18 % opacity) so it is told apart from the",
          "real part even when the two coincide; the adapter plate between the KUKA flange and the Sharpa palm is drawn (visual only).", "",
          "Each episode folder holds `force_tracking_video.mp4` (camera left; per fingertip the force target dashed and",
          "the measured force solid, time cursor, live numbers, object height), the camera video, a with/without-force-law",
          "overlay plot, the rollout's own plots, `summary.json` and `metrics.md`.", "",
          "## Aggregate (means over the 3 episodes per task)", "",
          "Every rollout was run twice from the same recording with the same chunk source: force law ON (task-space law) and",
          "force law OFF (null law: predicted states passed straight to the PD, no force feedback). The OFF column is the baseline.", "",
          "| setting | task | completed, force law ON | completed, force law OFF (null law) | worst force bias N, ON / OFF | closest keypoint mm, ON / OFF |",
          "|---|---|---|---|---|---|"]
    for (setting, prob), items in sorted(agg.items()):
        n = len(items)
        with_ = sum(1 for v, _ in items if completed(v))
        bias_w = sum(v["worst fingertip force bias N"] for v, _ in items) / n
        kp_w = sum(v["closest keypoint distance mm"] for v, _ in items) / n
        has_off = any(vn for _, vn in items)
        if has_off:
            without = sum(1 for _, vn in items if completed(vn)); nn = max(1, sum(1 for _, vn in items if vn))
            bias_n = sum(vn["worst fingertip force bias N"] for _, vn in items if vn) / nn
            kp_n = sum(vn["closest keypoint distance mm"] for _, vn in items if vn) / nn
            R.append(f"| {setting} | {prob} | {with_}/{n} | {without}/{n} | {bias_w:+.1f} / {bias_n:+.1f} | {kp_w:.0f} / {kp_n:.0f} |")
        else:   # a baseline setting: the force law is OFF in the rollout itself
            R.append(f"| {setting} | {prob} | - | {with_}/{n} | - / {bias_w:+.1f} | - / {kp_w:.0f} |")
    R += ["", "For the two baseline settings the force law is off in the rollout itself, so their numbers stand in the OFF columns.",
          "In the exact command replay the raw fingertip forces reproduce the recording bit for bit; the small nonzero bias there",
          "comes from the measurement filter the metric is computed with, not from the physics.", "", "## Episodes", "",
          "Rollout folders are named after the recorded episode's run name (`<task>_<time the collection process started>_ep0000`).",
          "The recordings themselves live under `outputs/play2perfect/episodes/<task>/<batch stamp>_seed<k>/ep_0000/`; every episode",
          "folder here carries the recording's video (`recorded_episode_video.mp4`) and summary.", "",
          "| rollout / episode name | recorded episode folder | collection seed | frames |", "|---|---|---|---|"]
    for name, (folder, seed, frames) in sorted(episodes.items()):
        R.append(f"| {name} | `{folder}` | {seed} | {frames} |")
    R += ["", "'Completed' = the env's own verdict (all sub-goals + retract) for the oracle setting, the offline keypoint insertion",
          "test for the open-loop replay. Force bias = mean(measured - target) over engaged steps, worst fingertip.", "",
          "## Reading the videos", "",
          "* dashed = force target the policy predicted, solid = what the tactile pad measured; the red cursor is the current time;",
          "  the text column shows target, measured and error in newtons and whether the fingertip is engaged (target above 0.2 N).",
          "* In the open-loop replay the hand often leaves the part after the first touch: measured force stays at zero while the",
          "  target keeps rising - there is nothing to track once the replayed trajectory has diverged.",
          "* In the oracle setting the position layer re-plans every 0.2 s, so the law is judged on what it does between re-plans:",
          "  a steady under-press of roughly 30 % and the loss of the grasp when the plan releases for a re-grasp are the typical failure modes.", "",
          "Code: `play2perfect_force_controller/` (this package); pipeline description in its `PIPELINE.md`."]
    (out / "README.md").write_text("\n".join(R) + "\n")
    H = ["<!doctype html><html><head><meta charset='utf-8'><title>Force-controller evaluation</title>",
         "<style>body{font-family:sans-serif;margin:24px} .ep{margin:18px 0;padding:12px;border:1px solid #ddd} video{max-width:100%} "
         "table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px} h2{margin-top:36px}</style></head><body>",
         "<h1>Force-controller evaluation on play2perfect assembly tasks</h1><p>See README.md for the description and the aggregate tables.</p>"]
    if abc:
        H.append("<h2>A / B / C comparison per episode</h2><p>" + " &nbsp;|&nbsp; ".join(
            f"<b style='color:rgb{ROWS[k][3]}'>{ROWS[k][0]}. {html.escape(ROWS[k][1])}</b>: {html.escape(ROWS[k][2])}" for k in ("a", "b", "c")) + "</p>")
        for prob, ep, rel in abc:
            H.append(f"<div class='ep'><h3>{html.escape(prob)} / {html.escape(ep)}</h3>"
                     f"<video controls preload='metadata' src='{rel.as_posix()}/abc_comparison.mp4'></video></div>")
    for setting in SETTINGS:
        H.append(f"<h2>{html.escape(setting)}</h2><p>{html.escape(SETTINGS[setting][3])}</p>")
        for st, prob, ep, rel, v in gallery:
            if st != setting:
                continue
            H.append(f"<div class='ep'><h3>{html.escape(prob)} / {html.escape(ep)}</h3>")
            H.append(f"<video controls preload='metadata' src='{rel.as_posix()}/force_tracking_video.mp4'></video>")
            H.append("<table>" + "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(val))}</td></tr>" for k, val in v.items()) + "</table>")
            H.append(f"<p><a href='{rel.as_posix()}/with_vs_without_forcelaw.png'>with vs without force law (plot)</a> · "
                     f"<a href='{rel.as_posix()}/camera_video.mp4'>camera video</a> · <a href='{rel.as_posix()}/metrics.md'>metrics</a></p></div>")
    H.append("</body></html>")
    (out / "index.html").write_text("\n".join(H))
    zip_path = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"packed {len(gallery)} episode folders -> {out}  ({total:.0f} MB), zip {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
