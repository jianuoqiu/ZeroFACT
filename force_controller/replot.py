#!/usr/bin/env python3
"""Regenerate the analysis plots of a finished force-controller rollout - no Isaac Sim needed.

Reads ``controller_data.npz`` (plus the episode it points at, for the recorded commands and key
frames) and rewrites ``force_tracking.png`` and ``commands_vs_predicted.png`` in place, so plot
tweaks take seconds instead of a sim re-run (same idea as ``scripts/render_force_video.py``).

Usage::

    python force_controller/replot.py outputs/force_controller/run_2026-05-15_17-55-22/<stamp>
    python force_controller/replot.py outputs/force_controller/run_*/<stamp1> <stamp2> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from force_controller.episode import ReplayEpisode  # noqa: E402
from force_controller.metrics import (  # noqa: E402
    command_fidelity_metrics,
    contact_point_metrics,
    format_command_fidelity,
    format_contact_point_metrics,
    format_grasp_slip,
    format_vector_metrics,
    grasp_slip_metrics,
    load_force_vectors,
    vector_force_metrics,
)
from force_controller.plots import (  # noqa: E402
    plot_action_comparison,
    plot_contact_point_error,
    plot_force_components,
    plot_force_error_components,
    plot_force_tracking,
    plot_vector_error,
)


def replot(run_dir: Path) -> None:
    data = np.load(run_dir / "controller_data.npz", allow_pickle=True)
    summary = json.loads((run_dir / "summary.json").read_text())
    law = summary["controller"]["force_law"]["law"]
    engage = summary["controller"]["force_law"].get("engage_threshold_n")
    run_name = summary.get("run", run_dir.parent.name)

    joint_names = [str(n) for n in data["joint_names"]]
    fingertips = [str(n) for n in data["fingertip_bodies"]]
    T = data["joint_target"].shape[0]
    steps_per_frame = data["f_ref_steps"].shape[1]

    episode = ReplayEpisode.load(Path(str(data["episode_path"]))).reordered(joint_names)

    plot_force_tracking(
        data["f_ref_steps"], data["f_meas_steps"], fingertips,
        run_dir / "force_tracking.png", steps_per_frame, episode.key_frames,
        u=data["u_steps"] if law != "null" else None,
        title=f"{run_name}: closed-loop force tracking (law={law})",
        u_label="|integral| [N]" if law == "task_space" else "u [rad]",
        engage_threshold_n=engage,
    )
    plot_action_comparison(
        data["joint_target"], episode.joint_target[:T], joint_names,
        run_dir / "commands_vs_predicted.png", episode.key_frames,
        title=f"{run_name}: recorded vs rollout commands vs predicted states (law={law})",
        predicted=data["q_ref_steps"][:, -1, :],
    )

    # vector force-tracking error |f_ref_vec - f_meas_vec| (reconstructed for older runs that
    # only recorded magnitudes); written back into the run's summary.json + plotted
    f_ref_vec, f_meas_vec, _ = load_force_vectors(run_dir)
    recorded = "recorded" if "f_ref_vec_steps" in data.files else "reconstructed"
    vec_metrics = vector_force_metrics(f_ref_vec, f_meas_vec, fingertips,
                                       engage if engage is not None else 0.2)
    vec_metrics["source"] = recorded
    summary["force_tracking_vector"] = vec_metrics

    plot_vector_error(
        f_ref_vec, f_meas_vec, fingertips, run_dir / "force_vector_error.png",
        steps_per_frame, episode.key_frames, engage,
        title=f"{run_name}: vector force-tracking error |f_ref − f_meas| (law={law})",
    )
    plot_force_components(
        f_ref_vec, f_meas_vec, fingertips, run_dir / "force_components.png",
        steps_per_frame, episode.key_frames, engage if engage is not None else 0.2,
        title=f"{run_name}: force vector components, world frame (law={law})",
    )

    plot_force_error_components(
        f_ref_vec, f_meas_vec, fingertips, run_dir / "force_error_components.png",
        steps_per_frame, episode.key_frames, engage if engage is not None else 0.2,
        title=f"{run_name}: force-tracking error per world axis (law={law})",
    )

    # contact-point tracking, for runs that recorded it (added 2026-08-29)
    made_point_plot = False
    if "p_ref_steps" in data.files and "p_meas_steps" in data.files:
        point_metrics = contact_point_metrics(
            data["p_ref_steps"], data["p_meas_steps"], f_ref_vec, fingertips,
            engage if engage is not None else 0.2,
        )
        summary["contact_point_tracking"] = point_metrics
        plot_contact_point_error(
            data["p_ref_steps"], data["p_meas_steps"], f_ref_vec, fingertips,
            run_dir / "contact_point_error.png", steps_per_frame, episode.key_frames,
            engage if engage is not None else 0.2,
            title=f"{run_name}: contact-point tracking error |p_ref − p_meas| (law={law})",
        )
        made_point_plot = True
    contact_frames = np.asarray(data["f_ref_steps"])[:, -1].max(axis=1) >= (
        engage if engage is not None else 0.2)
    cmd_metrics = command_fidelity_metrics(
        data["joint_target"], episode.joint_target[:T], data["q_ref_steps"][:, -1],
        joint_names, contact_frames,
    )
    summary["command_fidelity"] = cmd_metrics

    # in-hand slip (backfillable: object_pos / body_pos / tracked_links are in every npz)
    slip_metrics = None
    tracked = [str(l) for l in data["tracked_links"]]
    obj_keys = [str(k) for k in data["object_keys"]]
    if (episode.manipulated_key in obj_keys and "palm_lower" in tracked
            and episode.manipulated_key in episode.object_keys
            and "palm_lower" in episode.tracked_links):
        pi = tracked.index("palm_lower")
        ep_pi = episode.tracked_links.index("palm_lower")
        ep_mi = episode.object_keys.index(episode.manipulated_key)
        slip_metrics = grasp_slip_metrics(
            data["object_pos"][:, obj_keys.index(episode.manipulated_key)],
            data["body_pos"][:, pi], data["body_quat"][:, pi],
            episode.object_pos[:T, ep_mi],
            episode.body_pos[:T, ep_pi], episode.body_quat[:T, ep_pi],
            data["f_ref_steps"][:, -1], data["f_meas_steps"][:, -1],
        )
        summary["grasp_slip"] = slip_metrics
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plots = ["force_tracking.png", "commands_vs_predicted.png", "force_vector_error.png",
             "force_components.png", "force_error_components.png"] + (["contact_point_error.png"] if made_point_plot else [])
    print(f"[replot] {run_dir}: " + " + ".join(plots))
    print(f"[replot] vector error |f_ref_vec - f_meas_vec| ({recorded} vectors):")
    for line in format_vector_metrics(vec_metrics, prefix="[replot]   "):
        print(line)
    if made_point_plot:
        print("[replot] contact-point error |p_ref - p_meas|:")
        for line in format_contact_point_metrics(summary["contact_point_tracking"],
                                                 prefix="[replot]   "):
            print(line)
    else:
        print("[replot] contact-point tracking: not recorded by this run (pre-2026-08-29)")
    for line in format_command_fidelity(cmd_metrics, prefix="[replot] "):
        print(line)
    if slip_metrics is not None:
        print(format_grasp_slip(slip_metrics, prefix="[replot] "))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="force-controller output folder(s) containing controller_data.npz")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        if not (run_dir / "controller_data.npz").is_file():
            print(f"[replot][WARN] skipping {run_dir}: no controller_data.npz")
            continue
        replot(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
