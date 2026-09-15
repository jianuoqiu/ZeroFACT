#!/usr/bin/env python3
"""Regenerate the contact-force videos for an existing replay output folder.

No Isaac Sim involved: everything is rebuilt from ``replay.mp4`` + ``replay_data.npz`` +
``summary.json``, so annotation/styling changes can be iterated in seconds.

Example::

    python scripts/render_force_video.py outputs/run_2026-05-18_00-47-02/20260825_182240

writes (or overwrites) ``contact_forces.mp4`` and ``replay_with_forces.mp4`` in that folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from zerofact import analysis  # noqa: E402


def read_video_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """All frames of an mp4 as RGB arrays, plus its fps."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="an outputs/<run>/<timestamp> folder")
    parser.add_argument("--fps", type=float, default=None, help="override the fps read from replay.mp4")
    args = parser.parse_args()

    out_dir = args.output_dir
    npz_path, summary_path, video_path = (
        out_dir / "replay_data.npz",
        out_dir / "summary.json",
        out_dir / "replay.mp4",
    )
    for path in (npz_path, summary_path, video_path):
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing - the folder must come from a replay with rendering + video enabled"
            )

    data = np.load(npz_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())
    if "contact_force_steps" not in data or data["contact_force_steps"].size == 0:
        raise SystemExit("no contact_force_steps in replay_data.npz - re-run the replay with contact sensors")

    fingertip_bodies = [str(b) for b in data["fingertip_bodies"]]
    object_keys = [str(k) for k in data["object_keys"]]
    object_names = [str(n) for n in data["object_names"]]
    contact_object_keys = [str(k) for k in data.get("contact_object_keys", np.array([]))]
    contact_object_names = [object_names[object_keys.index(k)] for k in contact_object_keys]
    manipulated_key = summary.get("manipulated_object")
    manip_idx = contact_object_keys.index(manipulated_key) if manipulated_key in contact_object_keys else None

    key_frames = summary.get("key_frames") or {}
    sim_time_per_frame = summary.get("config", {}).get("sim_time_per_frame_s", 0.1)
    force_vis_scale = summary.get("config", {}).get("force_vis_scale")

    sim_frames, video_fps = read_video_frames(video_path)
    fps = args.fps or video_fps
    print(f"{len(sim_frames)} sim frames @ {fps:g} fps from {video_path}")

    handcam_path = out_dir / "handcam.mp4"
    inset_frames = read_video_frames(handcam_path)[0] if handcam_path.is_file() else None

    object_force_steps = data["contact_object_force_steps"] if data["contact_object_force_steps"].size else None
    plot_frames = analysis.render_contact_force_video(
        data["contact_force_steps"],
        fingertip_bodies,
        out_dir / "contact_forces.mp4",
        fps=fps,
        key_frames=key_frames,
        object_force_steps=object_force_steps,
        object_names=contact_object_names,
        manipulated_idx=manip_idx,
        sim_time_per_frame=sim_time_per_frame,
        return_frames=True,
    )
    if not plot_frames:
        raise SystemExit("force-plot video could not be rendered")
    print(f"force-plot video -> {out_dir / 'contact_forces.mp4'}")

    ok = analysis.render_composite_video(
        sim_frames,
        plot_frames,
        out_dir / "replay_with_forces.mp4",
        fps=fps,
        run_name=summary.get("run", out_dir.parent.name),
        fingertip_bodies=fingertip_bodies,
        contact_force=data["contact_force"],
        key_frames=key_frames,
        sim_time_per_frame=sim_time_per_frame,
        force_vis_scale=force_vis_scale,
        manipulated_name=(contact_object_names[manip_idx] if manip_idx is not None else None),
        inset_frames=inset_frames,
    )
    if not ok:
        raise SystemExit("combined video could not be rendered")
    print(f"combined video   -> {out_dir / 'replay_with_forces.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
