"""Isaac Sim launch plumbing shared by the two simulator scripts (runs BEFORE ``AppLauncher``).

Kept separate from ``p2p_env`` because that module imports Isaac Lab, which only works once the
Kit app is up. The rendering workaround is the same one ``scripts/replay_trajectory.py`` and
``force_controller/run_tracking.py`` use on this machine: Isaac Sim 5.1's ``*.rendering.kit``
experience files crash at startup here, so cameras are enabled on top of the plain headless
experience instead (see the project README, "compat rendering").
"""

from __future__ import annotations

import argparse

from .p2p_paths import PROJECT_ROOT  # noqa: F401  (side effect: none; documents the dependency)
from zerofact.runtime import check_memory, hard_exit, prepare_display  # noqa: F401

COMPAT_EXTENSIONS = ["omni.replicator.core", "omni.kit.viewport.rtx", "omni.kit.material.library"]
COMPAT_GUI_EXTENSIONS = [
    "omni.kit.mainwindow", "omni.kit.viewport.window", "omni.kit.manipulator.camera",
    "omni.kit.window.toolbar", "omni.kit.window.status_bar",
]


def add_sim_args(parser: argparse.ArgumentParser) -> None:
    """The simulation flags both scripts share (rendering / GUI / device)."""
    parser.add_argument("--no-render", action="store_true",
                        help="physics only: no camera, no videos (faster, lighter)")
    parser.add_argument("--gui", action="store_true", help="live Isaac Sim window")
    parser.add_argument("--no-compat-rendering", action="store_true",
                        help="use Isaac Lab's stock rendering experience instead of the "
                             "compatibility launch (once the install is repaired)")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="video frame rate; frames are captured every 60/fps policy steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-contact-points", type=int, default=256,
                        help="contact-point readback capacity per fingertip sensor (SDF parts "
                             "report many points; too small trips a CUDA assert)")


def finalize_launcher_args(args: argparse.Namespace) -> bool:
    """Turn the parsed flags into ``AppLauncher`` settings. Returns whether rendering is on."""
    render = not args.no_render
    args.headless = not args.gui
    if render or args.gui:
        args.enable_cameras = True
        if not args.no_compat_rendering:
            args.experience = "isaaclab.python.headless.kit"
            exts = list(COMPAT_EXTENSIONS) + (COMPAT_GUI_EXTENSIONS if args.gui else [])
            kit_args = (" ".join(f"--enable {e}" for e in exts) + " --/isaaclab/cameras_enabled=true"
                        # translucent goal marker: fractional cutout opacity must be on when the
                        # renderer starts (setting it later from Python has no effect)
                        " --/rtx/raytracing/fractionalCutoutOpacity=true")
            existing = getattr(args, "kit_args", "") or ""
            args.kit_args = f"{existing} {kit_args}".strip()
    check_memory(required_gb=8.0 if not (render or args.gui) else 12.0)
    return render
