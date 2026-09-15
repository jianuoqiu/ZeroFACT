"""KUKA iiwa14 + a dexterous hand: the robot-specific constants the controller pipeline needs.

Pure python (no Isaac, no torch) so the offline tools can import it. Everything here mirrors
``play2perfect/isaacsimenvs/tasks/play/utils/scene_utils.py`` / ``obs_utils.py`` - the numbers are
copied rather than imported so this package stays importable without the play2perfect checkout
(and so a play2perfect update cannot silently change what a recorded episode meant).

Two hands are supported, selected exactly as play2perfect selects them: the environment variable
``ISAACSIMENVS_HAND`` ("sharpa", the default, or "xhand"), read at import time. Set it before
importing this package and the whole pipeline - recording, rollout, plots, videos - follows.
The module-level names below are derived from the selected spec, so every importer is unchanged.

Fingertip order is fixed here and used everywhere (sensors, recordings, plots): index, middle,
ring, thumb, pinky - by FINGER ROLE, so the same colour means the same finger on both hands. The
fingertip *bodies* are the distal phalanges: the URDF's fixed tip/pad links hang off them through
fixed joints and the importer merges them (``merge_fixed_joints=True``), so the distal body carries
the pad's collision shapes and is where the contact sensor lives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ---- timing (play2perfect PreciseAssembly: 120 Hz physics, 60 Hz policy) ----------------------
PHYSICS_DT = 1.0 / 120.0
STEPS_PER_FRAME = 2                 # decimation: physics steps per policy step
FRAME_DT = PHYSICS_DT * STEPS_PER_FRAME   # one recorded "frame" == one 60 Hz policy step
# The env applies the policy's joint target at BOTH physics substeps of a frame (no stale-target
# lag as in the Isaac Gym replays), so an exact replay uses latency 0.
STALE_TARGET_STEPS = 0

# ---- arm (shared by both hands) -----------------------------------------------------------------
ARM_JOINT_NAMES = [f"iiwa14_joint_{i}" for i in range(1, 8)]

# implicit-PD stiffness the play2perfect articulation is built with (N m / rad). The task-space
# law maps forces to command offsets through K^-1 J^T f, so it needs these; the sim runner reads
# the live articulation values and only falls back to this table offline.
ARM_JOINT_STIFFNESS = {
    "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}

PALM_LINK = "iiwa14_link_7"        # the hand mount is merged into the last arm link

# scene bodies the fingertip sensors attribute their contacts to, in force_matrix_w filter order
CONTACT_OBJECT_KEYS = ["object", "hole", "table"]
MANIPULATED_KEY = "object"         # the insertion part (peg / beam part / table leg)

# matplotlib tab10 as RGB in [0, 1], by finger ROLE; shared by the in-sim force arrows and every
# plot/video, and the first four match the LEAP palette of force_controller/ so cross-robot plots
# read the same
_ROLE_COLORS = {
    "index": (0.121, 0.466, 0.705),    # tab:blue
    "middle": (1.000, 0.498, 0.054),   # tab:orange
    "ring": (0.172, 0.627, 0.172),     # tab:green
    "thumb": (0.839, 0.152, 0.156),    # tab:red
    "pinky": (0.580, 0.404, 0.741),    # tab:purple
}
_ROLE_ORDER = ("index", "middle", "ring", "thumb", "pinky")


@dataclass(frozen=True)
class HandSpec:
    """Everything the controller needs to know about the hand on the iiwa14 flange."""

    name: str
    joint_names: list[str]                      # canonical (policy) order of the hand joints
    joint_stiffness: dict[str, float]           # implicit-PD stiffness, N m / rad
    fingertip_bodies: list[str]                 # distal bodies, in _ROLE_ORDER
    # distal-body-frame offset from the body origin to the approximate pad centre (the point the
    # policy observes as "fingertip position"), one per fingertip body
    pad_offsets: dict[str, tuple[float, float, float]]

    @property
    def fingertip_labels(self) -> dict[str, str]:
        return dict(zip(self.fingertip_bodies, _ROLE_ORDER))

    @property
    def fingertip_colors(self) -> dict[str, tuple[float, float, float]]:
        return {b: _ROLE_COLORS[r] for b, r in zip(self.fingertip_bodies, _ROLE_ORDER)}


# ---- Sharpa (left), the hand play2perfect ships its pretrained policies for --------------------
SHARPA_SPEC = HandSpec(
    name="sharpa",
    joint_names=[
        "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE", "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
        "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
        "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
        "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA", "left_pinky_PIP",
        "left_pinky_DIP",
    ],
    joint_stiffness={
        "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
        "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
        "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
        "left_index_PIP": 0.9, "left_index_DIP": 0.9,
        "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
        "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
        "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
        "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
        "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
        "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
    },
    fingertip_bodies=[
        "left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP",
    ],
    pad_offsets={n: (0.02, 0.002, 0.0) for n in
                 ("left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP",
                  "left_pinky_DP")},
)

# ---- RobotEra XHAND1 (right), official URDF v1.3 ------------------------------------------------
# 12 joints: thumb bend (abd/opposition) + 2 flexions, index abduction + 2 flexions, 3 x 2 flexions.
# URDF effort limits: 1.1 N.m on the proximal joints and the thumb, 0.4 N.m on the distal joints
# and the index abduction. No verified gains exist for this hand: play2perfect sets the PD
# stiffness so the effort limit is reached at ISAACSIMENVS_XHAND_ERR_RAD (default 0.3) rad of
# tracking error, and this table must match the articulation the episodes were recorded with -
# the task-space law divides by exactly these numbers.
_XHAND_PROX = ("right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
               "right_hand_index_joint1", "right_hand_mid_joint1", "right_hand_ring_joint1",
               "right_hand_pinky_joint1")
_XHAND_DIST = ("right_hand_thumb_rota_joint2", "right_hand_index_bend_joint",
               "right_hand_index_joint2", "right_hand_mid_joint2", "right_hand_ring_joint2",
               "right_hand_pinky_joint2")
XHAND_ERR_RAD = float(os.environ.get("ISAACSIMENVS_XHAND_ERR_RAD", "0.3"))

XHAND_SPEC = HandSpec(
    name="xhand",
    joint_names=[
        "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1",
        "right_hand_thumb_rota_joint2",
        "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
        "right_hand_mid_joint1", "right_hand_mid_joint2",
        "right_hand_ring_joint1", "right_hand_ring_joint2",
        "right_hand_pinky_joint1", "right_hand_pinky_joint2",
    ],
    joint_stiffness={**{n: 1.1 / XHAND_ERR_RAD for n in _XHAND_PROX},
                     **{n: 0.4 / XHAND_ERR_RAD for n in _XHAND_DIST}},
    # role order index, middle, ring, thumb, pinky (play2perfect lists the thumb first; the bodies
    # are looked up by name, so this order is the pipeline's own)
    fingertip_bodies=[
        "right_hand_index_rota_link2", "right_hand_mid_link2", "right_hand_ring_link2",
        "right_hand_thumb_rota_link2", "right_hand_pinky_link2",
    ],
    # pad centres: the URDF tip frames sit 42.2 mm along +z of each finger's link2 (fingers curl
    # toward +x, the pad faces +x) and 50.2 mm along +y of the thumb's link2 (curls toward +z)
    pad_offsets={
        "right_hand_index_rota_link2": (0.006, 0.0, 0.035),
        "right_hand_mid_link2": (0.006, 0.0, 0.035),
        "right_hand_ring_link2": (0.006, 0.0, 0.035),
        "right_hand_thumb_rota_link2": (0.0, 0.042, 0.006),
        "right_hand_pinky_link2": (0.006, 0.0, 0.035),
    },
)

HAND_SPECS = {"sharpa": SHARPA_SPEC, "xhand": XHAND_SPEC}
HAND_NAME = os.environ.get("ISAACSIMENVS_HAND", "sharpa").lower()
if HAND_NAME not in HAND_SPECS:
    raise ValueError(f"ISAACSIMENVS_HAND={HAND_NAME!r}; choose one of {sorted(HAND_SPECS)}")
HAND_SPEC = HAND_SPECS[HAND_NAME]

# ---- names derived from the selected hand (what the rest of the package imports) ---------------
HAND_JOINT_NAMES = list(HAND_SPEC.joint_names)
HAND_JOINT_STIFFNESS = dict(HAND_SPEC.joint_stiffness)
FINGERTIP_BODIES = list(HAND_SPEC.fingertip_bodies)
FINGERTIP_LABELS = HAND_SPEC.fingertip_labels
FINGERTIP_COLORS = HAND_SPEC.fingertip_colors
# per-fingertip pad centre in the distal body frame, in FINGERTIP_BODIES order
FINGERTIP_PAD_OFFSETS = [HAND_SPEC.pad_offsets[b] for b in FINGERTIP_BODIES]
TRACKED_LINKS = [PALM_LINK] + FINGERTIP_BODIES


def default_joint_stiffness(joint_names: list[str]) -> list[float]:
    """Per-joint implicit-PD stiffness by name (fallback when the live articulation is absent)."""
    table = {**ARM_JOINT_STIFFNESS, **HAND_JOINT_STIFFNESS}
    missing = [n for n in joint_names if n not in table]
    if missing:
        raise KeyError(
            f"no PD stiffness known for joints {missing} (hand spec {HAND_NAME!r}; set "
            f"ISAACSIMENVS_HAND to the hand the episode was recorded with)"
        )
    return [table[n] for n in joint_names]


def register_with_zerofact_analysis() -> None:
    """Teach the shared zerofact plotting/video code this hand's fingertip names.

    ``zerofact.analysis`` colours and labels fingertips through the module-level dicts of
    ``zerofact.scene_spec``; extending those dicts in place is all it takes for the contact
    force plots and the annotated videos to name the five fingertips instead of falling back to
    grey + raw body names. Call once before plotting.
    """
    from zerofact import scene_spec

    scene_spec.FINGERTIP_COLORS.update(FINGERTIP_COLORS)
    scene_spec.FINGERTIP_LABELS.update(FINGERTIP_LABELS)
