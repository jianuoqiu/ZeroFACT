"""The policy side of the pipeline.

The eventual system is a BC policy that, from observations, predicts the next ``horizon`` robot
states AND force readings; a new chunk is requested every ``chunk`` frames (ACT-style chunking).
Until that policy exists, :class:`ChunkedReplayPolicy` serves slices of a recorded replay episode
with exactly those semantics, so the middle layer downstream cannot tell the difference.

Frame convention: ``predict(k)`` is called at the START of frame ``k`` (the newest available
observation is the state at the end of frame ``k-1``) and returns predictions for the states at the
ends of frames ``k .. k+horizon-1``. Predictions past the episode end repeat the last frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .config import PolicyConfig
from .episode import ReplayEpisode


@dataclass
class PolicyOutput:
    """One prediction chunk. ``knot j`` describes the state at the end of frame ``start_frame+j``.

    The force target is a full contact description: magnitude, world-frame vector and the contact
    point it acts at (NaN while the fingertip is not meant to touch). All three describe the
    fingertip PAD — the net force on it and its pressure centroid — because that is what the
    tactile sensor measures and therefore the only thing the tracking error can be formed on.
    """

    start_frame: int
    joint_pos: np.ndarray            # [H, J]  predicted robot joint states (sim joint order)
    fingertip_force: np.ndarray      # [H, 4]  predicted per-fingertip force magnitude (N)
    fingertip_force_vec: np.ndarray | None = None   # [H, 4, 3] world-frame force vectors
    contact_point: np.ndarray | None = None         # [H, 4, 3] world contact points, NaN-padded
    info: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return int(self.joint_pos.shape[0])


class BasePolicy(ABC):
    """What the middle layer expects from any policy (recorded, BC, or otherwise)."""

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

    @property
    def chunk(self) -> int:
        return int(self.cfg.chunk)

    def reset(self) -> None:  # noqa: B027 - stateless policies need nothing
        pass

    @abstractmethod
    def predict(self, frame: int, observation: dict | None = None) -> PolicyOutput:
        """Predict the next ``horizon`` frames starting at *frame*. *observation* is ignored by
        the replay policy but is part of the interface the real BC policy will use."""


class ChunkedReplayPolicy(BasePolicy):
    """Recorded replay data pretending to be the BC policy's output."""

    def __init__(self, episode: ReplayEpisode, cfg: PolicyConfig, force_source: str = "net"):
        """*force_source* is an ANALYSIS-ONLY escape hatch (default, and the only value the
        control path ever uses: ``"net"`` — the fingertip pad reading). ``metrics.py`` passes
        ``"manipulated"`` to reproduce the reference of rollouts recorded before 2026-08-29, when
        the privileged per-object force was still wired into the loop. It is deliberately not a
        ``PolicyConfig`` field, so no CLI or config can select it for control."""
        super().__init__(cfg)
        self.episode = episode
        if cfg.state_source == "joint_pos":
            self._states = episode.joint_pos
        elif cfg.state_source == "joint_target":
            self._states = episode.joint_target
        else:
            raise ValueError(f"unknown state_source {cfg.state_source!r}")
        self.force_source = force_source
        self._force_vec = episode.fingertip_force_vec(force_source)          # [T, 4, 3]
        self._force_mag = np.linalg.norm(self._force_vec, axis=-1)           # [T, 4]
        self._contact_point = episode.fingertip_contact_point()              # [T, 4, 3]

    def predict(self, frame: int, observation: dict | None = None) -> PolicyOutput:
        idx = np.clip(np.arange(frame, frame + self.cfg.horizon), 0, self.episode.num_frames - 1)
        return PolicyOutput(
            start_frame=int(frame),
            joint_pos=self._states[idx].copy(),
            fingertip_force=self._force_mag[idx].copy(),
            fingertip_force_vec=self._force_vec[idx].copy(),
            contact_point=self._contact_point[idx].copy(),
            info={
                "source": "replay",
                "episode": str(self.episode.path),
                "state_source": self.cfg.state_source,
                "force_source": self.force_source,
                "padded_frames": int(np.sum(np.arange(frame, frame + self.cfg.horizon) > idx[-1])),
            },
        )
