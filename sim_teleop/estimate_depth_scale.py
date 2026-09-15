#!/usr/bin/env python3
"""The depth correction: anchor RGB-only Dyn-HaMR to a metric depth sensor.

Dyn-HaMR is monocular, so its hand sits at the wrong distance from the camera by
some global factor (partly from an assumed focal length that differs from the
real intrinsics). This script measures that factor the same way the proven
``Video2Sim2Real_main/retargeting_kinova/depth_anchor.py`` does:

  1. take the Dyn-HaMR hand joints in the CAMERA frame,
  2. project wrist + the 4 MCP knuckles to pixels with the real intrinsics
     (fingertips are skipped on purpose - they sit at depth edges / get occluded),
  3. read the recorded sensor depth at those pixels,
  4. metric_scale = median(Z_sensor / Z_rgb) over every valid sample.

The number handed to the converter is ``depth_scale = 1 / metric_scale``
(the converter DIVIDES camera-frame points by it). Output json matches the
prior pipeline's ``skeleton_motion_robot_base_rgbd_scale.json`` format.

Inputs:
  --world-npy   (T,21,3) hand joints in Dyn-HaMR WORLD frame
                (make_hand_traj.py --export-world writes it)
  --npz         the *_world_results.npz (for the per-frame cam_R / cam_t)
  --depth-dir   folder of per-frame uint16 depth PNGs in millimetres
                (RealSense recording; frame t named t.png / 000t.png / ...)

Typical use (env with numpy+cv2, e.g. env_isaaclab):
    python sim_teleop/estimate_depth_scale.py \
        --world-npy outputs/sim_teleop/<seq>/hand_world.npy \
        --npz outputs/sim_teleop/<seq>/dynhamr_result/smooth_fit/<...>_world_results.npz \
        --depth-dir <recording>/depth \
        --out outputs/sim_teleop/<seq>/depth_scale.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_teleop.teleop_config import REAL_INTRINSICS  # noqa: E402

# wrist + MCP knuckles: reliable hand-surface points for depth sampling
ANCHOR_JOINTS = [0, 5, 9, 13, 17]


def world_to_camera(world: np.ndarray, npz_path: str | Path) -> np.ndarray:
    """Dyn-HaMR world joints (T,21,3) -> OpenCV camera frame via per-frame cam_R/cam_t."""
    d = np.load(npz_path)
    cam_r = np.asarray(d["cam_R"])[0].astype(np.float64)   # (T,3,3) world->cam
    cam_t = np.asarray(d["cam_t"])[0].astype(np.float64)   # (T,3)
    T = min(world.shape[0], cam_r.shape[0])
    return np.einsum("tij,tkj->tki", cam_r[:T], world[:T]) + cam_t[:T, None, :]


def _find_depth(depth_dir: Path, t: int) -> Path | None:
    for name in (f"{t}.png", f"{t:04d}.png", f"{t:05d}.png", f"{t:06d}.png",
                 f"frame_{t:04d}.png", f"depth_{t:04d}.png"):
        p = depth_dir / name
        if p.is_file():
            return p
    return None


def depth_scale(cam: np.ndarray, depth_dir: Path, intr, joints=ANCHOR_JOINTS,
                patch: int = 2, zmin: float = 0.2, zmax: float = 1.5,
                depth_unit_mm: bool = True) -> tuple[float, np.ndarray]:
    """metric_scale = median(Z_sensor / Z_rgb) over all valid (frame, joint) samples."""
    import cv2

    fx, fy, cx, cy = intr
    ratios = []
    matched = 0
    for t in range(cam.shape[0]):
        path = _find_depth(depth_dir, t)
        if path is None:
            continue
        dp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if dp is None:
            continue
        matched += 1
        dp = dp.astype(np.float32) / (1000.0 if depth_unit_mm else 1.0)
        h, w = dp.shape[:2]
        for j in joints:
            x, y, z = cam[t, j]
            if z <= 0:
                continue
            u = int(round(fx * x / z + cx))
            v = int(round(fy * y / z + cy))
            if not (patch <= u < w - patch and patch <= v < h - patch):
                continue
            win = dp[v - patch:v + patch + 1, u - patch:u + patch + 1]
            win = win[win > 0]
            if win.size < patch:
                continue
            z_sensor = float(np.median(win))
            if zmin < z_sensor < zmax:
                ratios.append(z_sensor / z)
    if matched == 0:
        raise FileNotFoundError(f"no depth frames found in {depth_dir} "
                                f"(tried t.png / 0000t.png / frame_000t.png ...)")
    ratios = np.asarray(ratios)
    if ratios.size < 20:
        raise RuntimeError(f"only {ratios.size} valid depth samples - too few to trust "
                           f"(need the hand visible + in the {zmin}-{zmax} m range)")
    return float(np.median(ratios)), ratios


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-npy", required=True, help="(T,21,3) Dyn-HaMR WORLD-frame joints")
    ap.add_argument("--npz", required=True, help="*_world_results.npz (cam_R/cam_t)")
    ap.add_argument("--depth-dir", required=True, help="per-frame uint16-mm depth PNGs")
    ap.add_argument("--intrins", nargs=4, type=float, default=REAL_INTRINSICS,
                    metavar=("FX", "FY", "CX", "CY"))
    ap.add_argument("--zmin", type=float, default=0.2)
    ap.add_argument("--zmax", type=float, default=1.5)
    ap.add_argument("--out", default=None, help="output json (default: alongside --world-npy)")
    args = ap.parse_args()

    world = np.load(args.world_npy).astype(np.float64)
    assert world.ndim == 3 and world.shape[1:] == (21, 3), f"bad shape {world.shape}"
    cam = world_to_camera(world, args.npz)

    scale, ratios = depth_scale(cam, Path(args.depth_dir), args.intrins,
                                zmin=args.zmin, zmax=args.zmax)
    out = {
        "metric_scale": scale,                      # Z_sensor / Z_rgb (median)
        "depth_scale_for_converter": 1.0 / scale,   # what dynhamr_to_skeleton divides by
        "n_samples": int(ratios.size),
        "ratio_p25_p75": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
    }
    out_path = Path(args.out) if args.out else Path(args.world_npy).with_name("depth_scale.json")
    out_path.write_text(json.dumps(out, indent=2))
    spread = out["ratio_p25_p75"]
    print(f"[depth] metric_scale = {scale:.4f}  ->  depth_scale_for_converter = {1/scale:.4f}")
    print(f"[depth] {ratios.size} samples, inter-quartile {spread[0]:.3f}..{spread[1]:.3f}"
          + ("   [WARN: wide spread - check calibration/sync]" if spread[1] - spread[0] > 0.15 else ""))
    print(f"[depth] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
