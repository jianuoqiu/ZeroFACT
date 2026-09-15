"""Where the play2perfect checkout lives and how to make it importable.

play2perfect is used as a library from its git checkout (nothing is pip-installed): the env
package ``isaacsimenvs``, the problem registry ``evaluation.problems`` and the *vendored*
``rl_games`` fork (PPO + SAPG; the released checkpoints need the fork's model classes, and the
``rl_games 1.6.1`` that ships in the conda env lacks them). :func:`bootstrap` puts both at the
front of ``sys.path`` so the fork shadows the installed package, and sets the Kit EULA variable
that every non-interactive Isaac Sim launch needs.

Override the location with ``PLAY2PERFECT_ROOT=/path/to/play2perfect``.

The task id and the checkpoint layout follow the hand selected by ``ISAACSIMENVS_HAND`` (see
:mod:`robot_spec`): the Sharpa task and play2perfect's released ``pretrained_assembly/`` for the
default, the XHand task and the locally trained ``pretrained_assembly_xhand/`` for "xhand".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .robot_spec import HAND_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]              # ~/ZeroFACT
P2P_ROOT = Path(os.environ.get("PLAY2PERFECT_ROOT", Path.home() / "play2perfect")).resolve()

# task id + checkpoint directory per hand (ISAACSIMENVS_HAND, read in robot_spec)
_TASK_IDS = {
    "sharpa": "Isaacsimenvs-PreciseAssembly-Direct-v0",
    "xhand": "Isaacsimenvs-PreciseAssembly-XHand-Direct-v0",
}
_CHECKPOINT_DIRS = {
    "sharpa": "pretrained_assembly",         # play2perfect's released policies
    "xhand": "pretrained_assembly_xhand",    # trained here (see docs/xhand.md, stage 2)
}
TASK_ID = _TASK_IDS[HAND_NAME]
AGENT_ENTRY = "rl_games_sapg_cfg_entry_point"       # the released checkpoints are SAPG policies
PROBLEMS = ["tight_insertion", "beam_assembly_step1", "beam_assembly_step2", "screwing"]


# recordings are kept per hand so a Sharpa and an XHand episode can never be confused (the
# Sharpa tree keeps its historical name, so every existing report/share path still resolves)
EPISODE_ROOT = PROJECT_ROOT / "outputs" / "play2perfect" / (
    "episodes" if HAND_NAME == "sharpa" else f"episodes_{HAND_NAME}")


def checkpoint_path(problem: str) -> Path:
    """The per-problem policy for the selected hand: ``<checkpoint dir>/<problem>/model.pth``."""
    return P2P_ROOT / _CHECKPOINT_DIRS[HAND_NAME] / problem / "model.pth"


def bootstrap() -> Path:
    """Make ``isaacsimenvs`` / ``evaluation`` / the vendored ``rl_games`` importable. Idempotent."""
    if not (P2P_ROOT / "isaacsimenvs").is_dir():
        raise FileNotFoundError(
            f"play2perfect checkout not found at {P2P_ROOT} "
            "(git clone https://github.com/kushal2000/play2perfect, or set PLAY2PERFECT_ROOT)"
        )
    for path in (P2P_ROOT / "rl_games", P2P_ROOT, PROJECT_ROOT):
        p = str(path)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return P2P_ROOT
