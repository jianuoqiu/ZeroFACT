"""Load a recorded replay (``outputs/<run>/<stamp>/replay_data.npz``) as a controller episode.

Pure numpy - no Isaac imports - so the same loader serves the offline checker and the sim runner.

Index convention (inherited from the replay): arrays are indexed by trajectory frame ``k`` and the
value at ``k`` is what was measured at the END of frame ``k`` (after its 12 physics steps).
``joint_target[k]`` is the command that was held DURING frame ``k``. All joint-indexed arrays are
stored in the recorded simulation joint order (``joint_names``); use :meth:`reordered` to move them
into a live articulation's order before driving a robot with them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


def pad_contact_centroid(pair_forces: np.ndarray, pair_points: np.ndarray) -> np.ndarray:
    """Fuse per-(fingertip, object) contact patches into one centroid per fingertip pad.

    ``pair_forces``  ``[..., tips, M, 3]`` force carried at each patch,
    ``pair_points``  ``[..., tips, M, 3]`` patch centre, NaN where that pair is not touching.
    Returns ``[..., tips, 3]``: the |f|-weighted mean of the touching patches, NaN when the pad
    carries no reported patch at all.

    This models what a tactile pad actually reports — a single pressure centroid — rather than
    the simulator's per-object decomposition. Patches against unfiltered geometry (the table is a
    static collider, not a filtered object) carry force in ``contact_force`` but report no centre,
    so the centroid covers the filtered objects only; the fallback is then NaN and the law anchors
    at the predicted point or the fingertip origin.
    """
    ok = np.isfinite(pair_points).all(axis=-1)                      # [..., tips, M] reported patch
    w = np.where(ok, np.linalg.norm(pair_forces, axis=-1), 0.0)     # [..., tips, M]
    total = w.sum(axis=-1)                                          # [..., tips]
    # a reported patch carrying exactly 0 N is still a touch - PhysX reports the centre whenever
    # the geometry is in contact, and the normal force passes through zero constantly at 120 Hz.
    # Dropping those loses the contact location through the whole light-touch approach phase, so
    # fall back to the unweighted mean of the reported patches when the pad carries no force yet.
    w = np.where((total > 0.0)[..., None], w, ok.astype(np.float64))
    total = w.sum(axis=-1)
    pts = np.where(ok[..., None], np.nan_to_num(pair_points), 0.0)
    out = (w[..., None] * pts).sum(axis=-2) / np.maximum(total, 1e-12)[..., None]
    out[total <= 0.0] = np.nan                                      # no patch reported at all
    return out


@dataclass
class ReplayEpisode:
    path: Path
    run_name: str
    # joints (sim order as recorded)
    joint_names: list[str]
    joint_pos: np.ndarray                  # [T, J] measured state at frame end
    joint_vel: np.ndarray                  # [T, J]
    joint_target: np.ndarray               # [T, J] command held during the frame
    # fingertip contact forces, world frame
    fingertip_bodies: list[str]            # [4]
    contact_force: np.ndarray              # [T, 4, 3]        net, frame-end sample
    contact_force_steps: np.ndarray        # [T, S, 4, 3]     net, every physics step
    contact_object_force_steps: np.ndarray  # [T, S, 4, M, 3] per filtered object
    contact_object_keys: list[str]         # [M]
    contact_point_w: np.ndarray            # [T, 4, M, 3]     contact-patch centre, NaN off-contact
    # objects
    object_keys: list[str]
    object_names: list[str]
    object_pos: np.ndarray                 # [T, N, 3]
    object_quat: np.ndarray                # [T, N, 4] wxyz
    object_lin_vel: np.ndarray             # [T, N, 3]
    manipulated_key: str | None
    # tracked robot links (palm, mount, bracelet, fingertips)
    tracked_links: list[str]
    body_pos: np.ndarray                   # [T, L, 3]
    body_quat: np.ndarray                  # [T, L, 4] wxyz
    # timing / annotations
    key_frames: dict
    frame_dt: float                        # seconds of sim time per trajectory frame
    summary: dict

    # ------------------------------------------------------------------ properties
    @property
    def num_frames(self) -> int:
        return int(self.joint_target.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.joint_target.shape[1])

    @property
    def steps_per_frame(self) -> int:
        return int(self.contact_force_steps.shape[1])

    @property
    def manipulated_contact_idx(self) -> int | None:
        if self.manipulated_key is None or self.manipulated_key not in self.contact_object_keys:
            return None
        return self.contact_object_keys.index(self.manipulated_key)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, episode_dir: str | Path) -> "ReplayEpisode":
        episode_dir = Path(episode_dir)
        npz_path = episode_dir / "replay_data.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"{episode_dir} has no replay_data.npz")
        data = np.load(npz_path, allow_pickle=True)

        summary_path = episode_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        run_name = summary.get("run", episode_dir.parent.name)
        frame_dt = float(summary.get("config", {}).get("sim_time_per_frame_s", 0.1))

        return cls(
            path=episode_dir,
            run_name=run_name,
            joint_names=[str(n) for n in data["joint_names"]],
            joint_pos=np.asarray(data["joint_pos"], dtype=np.float64),
            joint_vel=np.asarray(data["joint_vel"], dtype=np.float64),
            joint_target=np.asarray(data["joint_target"], dtype=np.float64),
            fingertip_bodies=[str(n) for n in data["fingertip_bodies"]],
            contact_force=np.asarray(data["contact_force"], dtype=np.float64),
            contact_force_steps=np.asarray(data["contact_force_steps"], dtype=np.float64),
            contact_object_force_steps=np.asarray(
                data["contact_object_force_steps"], dtype=np.float64
            ),
            contact_object_keys=[str(n) for n in data["contact_object_keys"]],
            contact_point_w=np.asarray(data["contact_point_w"], dtype=np.float64),
            object_keys=[str(n) for n in data["object_keys"]],
            object_names=[str(n) for n in data["object_names"]],
            object_pos=np.asarray(data["object_pos"], dtype=np.float64),
            object_quat=np.asarray(data["object_quat"], dtype=np.float64),
            object_lin_vel=np.asarray(data["object_lin_vel"], dtype=np.float64),
            manipulated_key=summary.get("manipulated_object"),
            tracked_links=[str(n) for n in data["tracked_links"]],
            body_pos=np.asarray(data["body_pos"], dtype=np.float64),
            body_quat=np.asarray(data["body_quat"], dtype=np.float64),
            key_frames=summary.get("key_frames", {}),
            frame_dt=frame_dt,
            summary=summary,
        )

    @staticmethod
    def resolve_dir(spec: str | Path, outputs_root: str | Path = "outputs") -> Path:
        """Resolve an episode folder from a path or a run name (latest stamp with data wins)."""
        candidate = Path(spec)
        if (candidate / "replay_data.npz").is_file():
            return candidate
        run_dir = candidate if candidate.is_dir() else Path(outputs_root) / str(spec)
        if run_dir.is_dir():
            stamps = sorted(
                (p for p in run_dir.iterdir() if (p / "replay_data.npz").is_file()),
                key=lambda p: p.name,
            )
            if stamps:
                return stamps[-1]
        raise FileNotFoundError(
            f"no replay_data.npz under {spec!r} (looked in {candidate} and {run_dir})"
        )

    # ------------------------------------------------------------------ views
    #
    # Sensing contract: the CONTROL path may only use what a fingertip tactile pad reports — the
    # net force on the pad and the pad's contact centroid. Both default to that here. The
    # per-object breakdown (``source="manipulated"``) is privileged simulator information; it is
    # kept for *analysis* (and for reproducing pre-2026-08-29 rollouts) and is never reachable
    # from ControllerConfig.
    def fingertip_force_mag(self, source: str = "net", per_step: bool = False) -> np.ndarray:
        """Per-fingertip force magnitude, ``[T, 4]`` (frame-end) or ``[T, S, 4]`` (per step).

        ``source="net"`` (the default, and the only thing a tactile pad can measure) keeps
        everything the fingertip touches: the manipulated object, the table, other objects.
        ``"manipulated"`` keeps only the force exchanged with the manipulated object — analysis
        only. Falls back to net when the manipulated object is not among the sensor filters.
        """
        return np.linalg.norm(self.fingertip_force_vec_steps(source), axis=-1) if per_step \
            else np.linalg.norm(self.fingertip_force_vec(source), axis=-1)

    def fingertip_force_vec_steps(self, source: str = "net") -> np.ndarray:
        """Per-step per-fingertip force vectors ``[T, S, 4, 3]`` (world frame)."""
        midx = self.manipulated_contact_idx
        if source == "manipulated" and midx is not None:
            return self.contact_object_force_steps[:, :, :, midx, :]
        if source in ("manipulated", "net"):
            return self.contact_force_steps
        raise ValueError(f"unknown force source {source!r}")

    def fingertip_force_vec(self, source: str = "net") -> np.ndarray:
        """Frame-end per-fingertip force vectors ``[T, 4, 3]`` (world frame)."""
        if source == "net":
            return self.contact_force
        return self.fingertip_force_vec_steps(source)[:, -1]

    def fingertip_contact_point(self) -> np.ndarray:
        """Frame-end contact centroid per fingertip ``[T, 4, 3]``, world frame, NaN off-contact.

        This is the "contact point position" part of the force target the policy predicts. It is
        object-agnostic on purpose: a tactile pad reports a single pressure centroid and cannot
        say *what* it is touching, so the per-(fingertip, object) patch centres PhysX reports are
        fused into one point per pad, weighted by the force carried at each patch — see
        :func:`pad_contact_centroid`.
        """
        return pad_contact_centroid(self.contact_object_force_steps[:, -1], self.contact_point_w)

    def reordered(self, joint_names: list[str]) -> "ReplayEpisode":
        """A copy with every joint-indexed array reordered to match *joint_names*."""
        if list(joint_names) == self.joint_names:
            return self
        missing = [n for n in joint_names if n not in self.joint_names]
        extra = [n for n in self.joint_names if n not in joint_names]
        if missing or extra:
            raise KeyError(
                f"episode joints do not match the requested order (missing {missing}, extra {extra})"
            )
        idx = [self.joint_names.index(n) for n in joint_names]
        return replace(
            self,
            joint_names=list(joint_names),
            joint_pos=self.joint_pos[:, idx],
            joint_vel=self.joint_vel[:, idx],
            joint_target=self.joint_target[:, idx],
        )
