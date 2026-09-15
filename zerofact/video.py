"""Browser-playable mp4 writing.

Chromium-based players (VS Code webviews, Chrome) ship no MPEG-4 part 2 decoder, and
"mp4v" — the only mp4 fourcc OpenCV's bundled encoder reliably offers — is exactly that
codec, so cv2.VideoWriter output cannot be previewed in VS Code. Frames are piped to the
system ffmpeg (libx264 -> H.264 + yuv420p) instead; cv2 remains a last-resort fallback
for machines without ffmpeg.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class VideoWriter:
    """Drop-in for cv2.VideoWriter: uint8 BGR frames in via write(), release() when done.

    release() returns whether the file was actually encoded, unlike cv2's None.
    """

    def __init__(self, path: str | Path, fps: int, size: tuple[int, int]):
        self._path = Path(path)
        self._proc: subprocess.Popen | None = None
        self._cv2_writer = None
        width, height = size
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            self._proc = subprocess.Popen(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                    "-r", str(fps), "-i", "-",
                    # yuv420p demands even dimensions; dropping one edge pixel is invisible
                    "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(self._path),
                ],
                stdin=subprocess.PIPE,
            )
            return
        import cv2

        for fourcc in ("avc1", "mp4v"):  # avc1 (H.264) exists only in some cv2 builds
            writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
            if writer.isOpened():
                if fourcc == "mp4v":
                    print(
                        f"[video][WARN] no ffmpeg and no H.264 in cv2 -> "
                        f"{self._path} will not play in VS Code or browsers"
                    )
                self._cv2_writer = writer
                return
            writer.release()

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc is not None:
            if self._proc.stdin.closed:
                return
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame_bgr, dtype=np.uint8).tobytes())
            except BrokenPipeError:  # ffmpeg died; surfaced as False from release()
                self._proc.stdin.close()
        elif self._cv2_writer is not None:
            self._cv2_writer.write(frame_bgr)

    def release(self) -> bool:
        if self._proc is not None:
            if not self._proc.stdin.closed:
                try:
                    self._proc.stdin.close()
                except BrokenPipeError:
                    pass
            if self._proc.wait() != 0:
                print(f"[video][WARN] ffmpeg exited with an error -> {self._path} may be broken")
                return False
            return True
        if self._cv2_writer is not None:
            self._cv2_writer.release()
            return True
        print(f"[video][WARN] no usable video backend -> {self._path} not written")
        return False
