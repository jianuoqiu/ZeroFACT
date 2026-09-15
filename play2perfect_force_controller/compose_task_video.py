#!/usr/bin/env python3
"""Three rollouts of one recorded episode stacked in a column, each row = camera + live force
curves (``render_force_video.build_frames``) under a coloured title:

    A. Original action replay: sanity check      (recorded motor commands replayed exactly)
    B. State-only replay: no force tracking       (recorded reached joint states replayed)
    C. State replay + force tracking              (reached states served, task-space force law on)

    python play2perfect_force_controller/compose_task_video.py --a <rollout> --b <rollout> --c <rollout> \\
        --out abc_comparison.mp4 [--label "tight_insertion / 20260908_211656_ep0000"]

Rows of different length are padded with their last frame.  No simulator needed: it reads each
rollout's ``rollout.mp4`` and ``controller_data.npz``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from play2perfect_force_controller.render_force_video import _font, build_frames, encode  # noqa: E402

# (letter, title, subtitle, header colour)
ROWS = {
    "a": ("A", "Original action replay: sanity check",
          "the recorded motor commands replayed exactly; reproduces the recording", (31, 90, 170)),
    "b": ("B", "State-only replay: no force tracking",
          "the recorded reached joint states replayed through the position layer alone", (204, 102, 0)),
    "c": ("C", "State replay + force tracking",
          "reached states served, the task-space force law supplies the grasp preload", (34, 120, 60)),
}
HEADER_H = 60
BANNER_H = 44
TITLE_FONT = 34


def compose(rollouts: dict[str, Path], out: Path, label: str | None = None, fps: int = 30) -> Path:
    work = Path(tempfile.mkdtemp(prefix="abc_video_"))
    rows: list[list[Path]] = []
    for key in ("a", "b", "c"):
        letter, title, sub, color = ROWS[key]
        # the header is the short label only; the longer role description lives in the README
        rows.append(build_frames(rollouts[key], work / key, header=f"{letter}.  {title}",
                                 header_color=color, header_h=HEADER_H, header_font=TITLE_FONT, header_bold=True))
    n = max(len(r) for r in rows)
    w = max(Image.open(r[0]).size[0] for r in rows)
    heights = [Image.open(r[0]).size[1] for r in rows]
    total_h = sum(heights) + BANNER_H
    frames_dir = work / "out"; frames_dir.mkdir()
    bfont = _font(26, bold=True)
    for i in range(n):
        frame = Image.new("RGB", (w, total_h), (255, 255, 255))
        if label:
            ImageDraw.Draw(frame).text((10, (BANNER_H - 26) // 2), label, fill=(40, 40, 40), font=bfont)
        y = BANNER_H
        for r, h in zip(rows, heights):
            img = Image.open(r[min(i, len(r) - 1)]).convert("RGB")
            frame.paste(img, (0, y))
            ImageDraw.Draw(frame).line([(0, y), (w, y)], fill=(200, 200, 200), width=1)
            y += h
        frame.save(frames_dir / f"f_{i:05d}.png")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    encode(frames_dir, out, fps)
    shutil.copy(frames_dir / f"f_{min(n - 1, n // 2):05d}.png", out.with_suffix(".sample.png"))
    shutil.rmtree(work)
    print(f"wrote {out}  ({n} frames, {fps} fps, {w}x{total_h})")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="rollout dir: recorded commands replayed exactly (null law)")
    p.add_argument("--b", required=True, help="rollout dir: recorded reached states replayed (null law)")
    p.add_argument("--c", required=True, help="rollout dir: reached states served + task-space force law")
    p.add_argument("--out", required=True)
    p.add_argument("--label", default=None, help="banner text above the three rows")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()
    compose({"a": Path(a.a), "b": Path(a.b), "c": Path(a.c)}, Path(a.out), a.label, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
