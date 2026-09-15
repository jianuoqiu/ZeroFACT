#!/usr/bin/env python3
"""Rollout video with a live force-tracking plot: camera (with force arrows) on the left, on the
right the force TARGET (dashed) and the MEASURED force (solid) per fingertip with a moving time
cursor and the current values, plus the object height. No simulator: it uses the rollout's
``rollout.mp4`` (camera frames captured every ``video_every`` frames) and ``controller_data.npz``.

    python play2perfect_force_controller/render_force_video.py <rollout dir> [--out file.mp4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAME_DT = 1.0 / 60.0
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def render(rollout: str | Path, out: str | Path | None = None, fps: int = 30) -> Path:
    """Build ``rollout_force_tracking.mp4`` for one rollout folder; returns the video path."""
    a = argparse.Namespace(rollout=str(rollout), out=None if out is None else str(out), fps=int(fps))
    return _render(a)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout"); p.add_argument("--out", default=None); p.add_argument("--fps", type=int, default=30)
    _render(p.parse_args())
    return 0


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def rollout_title(s: dict) -> str:
    """Default one-line description of a rollout from its summary.json."""
    ctrl = s["controller"]; law = ctrl["force_law"]["law"]
    src = "commands" if ctrl["policy"]["state_source"] == "joint_target" else "reached states"
    kind = f"oracle C{ctrl['policy']['chunk']}" if ctrl["policy"].get("source") == "oracle" else "replay"
    return f"{s['run']}  |  {kind}, {src}, law {law}"


def build_frames(rollout: str | Path, out_dir: Path, header: str | None = None,
                 header_color=(0, 0, 0), header_h: int = 22, header_font: int = 12, header_bold: bool = False,
                 clock: bool = True) -> list[Path]:
    """Write one PNG per captured camera frame into *out_dir*: camera (left) + live force plot
    (right) under a header line.  Returns the frame paths in order.  The header text is drawn in
    *header_color*; the running clock is appended in black when *clock* is set."""
    d = Path(rollout)
    c = np.load(d / "controller_data.npz", allow_pickle=True)
    s = json.loads((d / "summary.json").read_text())
    video = d / "rollout.mp4"
    if not video.is_file():
        raise SystemExit(f"{d} has no rollout.mp4 (run the rollout without --no-render)")
    tmp = Path(tempfile.mkdtemp(prefix="force_video_"))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), str(tmp / "cam_%05d.png")], check=True)
    cam = sorted(tmp.glob("cam_*.png"))
    T = int(c["f_ref_steps"].shape[0]); n_cam = len(cam)
    every = max(1, int(round(T / n_cam)))
    f_ref = c["f_ref_steps"][:, -1, :]; f_meas = c["f_meas_steps"][:, -1, :]
    tips = [b.replace("left_", "").replace("_DP", "").replace("right_hand_", "").replace("_link2", "") for b in c["fingertip_bodies"]]
    n = len(tips); t = np.arange(T) * FRAME_DT
    keys = list(c["object_keys"]); z = c["object_pos"][:, keys.index("object"), 2]; z = 100 * (z - z[0])
    thr = s["controller"]["force_law"].get("engage_threshold_n", 0.2)
    title = rollout_title(s) if header is None else header

    # static plot: all curves, drawn once; the cursor and markers are painted per frame with PIL
    cam0 = Image.open(cam[0]); W, H = cam0.size
    dpi = 100; Wp = int(W * 1.55)                      # plot panel: wider than the camera, same height
    fig, axes = plt.subplots(n + 1, 1, figsize=(Wp / dpi, H / dpi), dpi=dpi, sharex=True)
    for i in range(n):
        ax = axes[i]
        ax.plot(t, f_ref[:, i], "--", color="0.3", lw=1.2, label="target (dashed)" if i == 0 else None)
        ax.plot(t, f_meas[:, i], "-", color=COLORS[i % 5], lw=1.6, label="measured (solid)" if i == 0 else None)
        top = max(3.0, float(np.nanmax(np.maximum(f_ref[:, i], f_meas[:, i]))) * 1.15)   # per-fingertip scale
        ax.set_ylim(0, top); ax.set_ylabel(f"{tips[i]}\n[N]", fontsize=13); ax.tick_params(labelsize=11); ax.grid(alpha=0.25)
    axes[-1].plot(t, z, color="k", lw=1.5); axes[-1].set_ylabel("obj z\n[cm]", fontsize=13); axes[-1].tick_params(labelsize=11); axes[-1].grid(alpha=0.25)
    axes[-1].set_xlabel("time [s]", fontsize=13); axes[-1].set_xlim(0, t[-1] if T > 1 else 1)
    fig.legend(loc="upper left", fontsize=13, ncol=2, frameon=False, bbox_to_anchor=(0.055, 1.005))
    fig.subplots_adjust(left=0.105, right=0.70, top=0.90, bottom=0.11, hspace=0.45)
    fig.canvas.draw()
    bg = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes(), "raw", "RGBA", 0, 1).convert("RGB")
    bg = bg.resize((Wp, H))
    sx = Wp / fig.canvas.get_width_height()[0]; sy = H / fig.canvas.get_width_height()[1]
    col_x = int(0.725 * Wp)                            # value column, right of the axes
    def to_px(ax, x, y):
        X, Y = ax.transData.transform((x, y)); return X * sx, (fig.canvas.get_width_height()[1] - Y) * sy
    plt.close(fig)
    font = _font(12); big = _font(26, bold=True); hfont = _font(header_font, header_bold)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, cpath in enumerate(cam):
        k = min(i * every, T - 1)
        right = bg.copy(); dr = ImageDraw.Draw(right)
        for j, ax in enumerate(axes):
            x0, y0 = to_px(ax, t[k], ax.get_ylim()[0]); x1, y1 = to_px(ax, t[k], ax.get_ylim()[1])
            dr.line([(x0, y0), (x1, y1)], fill=(200, 0, 0), width=2)
            ymid = 0.5 * (y0 + y1)
            if j < n:
                for val, col in ((f_ref[k, j], (80, 80, 80)), (f_meas[k, j], tuple(int(v * 255) for v in matplotlib.colors.to_rgb(COLORS[j % 5])))):
                    px, py = to_px(ax, t[k], min(val, ax.get_ylim()[1])); dr.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col)
                # the one number worth reading per fingertip: how far the measured force is from the
                # target right now (grey while the target is below the engage threshold)
                eng = f_ref[k, j] >= thr
                dr.text((col_x, ymid - 15), f"err {f_meas[k, j] - f_ref[k, j]:+5.1f} N",
                        fill=(170, 0, 0) if eng else (130, 130, 130), font=big)
            else:
                dr.text((col_x, ymid - 15), f"obj z {z[k]:+5.1f} cm", fill=(0, 0, 0), font=big)
        left = Image.open(cpath).convert("RGB")
        frame = Image.new("RGB", (W + Wp, H + header_h), (255, 255, 255))
        frame.paste(left, (0, header_h)); frame.paste(right, (W, header_h))
        hd = ImageDraw.Draw(frame)
        hd.text((6, (header_h - header_font) // 2 - 1), title, fill=tuple(header_color), font=hfont)
        if clock:
            cfont = _font(max(14, header_font - 12))
            hd.text((W + Wp - 260, (header_h - cfont.size if hasattr(cfont, "size") else header_h - 16) // 2),
                    f"t = {t[k]:5.2f} s   frame {k}", fill=(0, 0, 0), font=cfont)
        path = out_dir / f"f_{i:05d}.png"; frame.save(path); paths.append(path)
    shutil.rmtree(tmp)
    return paths


def encode(frames_dir: Path, out: Path, fps: int) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)], check=True)


def _render(a) -> Path:
    d = Path(a.rollout)
    work = Path(tempfile.mkdtemp(prefix="force_video_frames_"))
    frames = build_frames(d, work)
    out = Path(a.out) if a.out else d / "rollout_force_tracking.mp4"
    encode(work, out, a.fps)
    sample = out.with_suffix(".sample.png"); shutil.copy(frames[min(len(frames) - 1, len(frames) // 2)], sample)
    shutil.rmtree(work)
    print(f"wrote {out}  ({len(frames)} frames, {a.fps} fps; sample frame {sample})")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
