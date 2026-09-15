"""Map the trajectory's URDF joint names onto the joint names Isaac Lab reports.

The LEAP hand joints are literally named ``"0" ... "15"`` in ``v12_vision.urdf``. USD prim names may
not start with a digit, so the URDF importer renames them (typically ``0`` -> ``_0`` or ``joint_0``).
Isaac Lab's ``Articulation.joint_names`` therefore does not necessarily equal the URDF joint names,
and the mapping has to be recovered rather than assumed.

The matcher is deliberately conservative: it only accepts a one-to-one mapping and raises with the
full name lists when anything is ambiguous, so a silent mis-mapping (fingers driven by arm targets)
can never happen.
"""

from __future__ import annotations

import re


def _normalize(name: str) -> str:
    """Lowercase and drop separators, so ``_0`` / ``0`` / ``J-0`` all normalise to ``0``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _candidates(urdf_name: str) -> list[str]:
    """Plausible USD/PhysX spellings of one URDF joint name, most specific first."""
    base = urdf_name
    out = [base, f"_{base}", f"joint_{base}", f"joint{base}", f"j_{base}", f"j{base}", f"dof_{base}"]
    # names such as "0" are also seen prefixed with the parent link name by some importers
    return out


def match_joint_names(urdf_names: list[str], sim_names: list[str]) -> dict[str, str]:
    """Return ``{urdf_joint_name: sim_joint_name}``.

    Raises:
        KeyError: if a URDF joint cannot be matched, or two URDF joints claim the same sim joint.
    """
    sim_by_exact = {n: n for n in sim_names}
    sim_by_norm: dict[str, list[str]] = {}
    for n in sim_names:
        sim_by_norm.setdefault(_normalize(n), []).append(n)

    mapping: dict[str, str] = {}
    unmatched: list[str] = []

    for urdf_name in urdf_names:
        hit = sim_by_exact.get(urdf_name)
        if hit is None:
            for cand in _candidates(urdf_name):
                options = sim_by_norm.get(_normalize(cand), [])
                if len(options) == 1:
                    hit = options[0]
                    break
                if len(options) > 1:
                    raise KeyError(
                        f"URDF joint {urdf_name!r} matches several simulation joints {options}; "
                        "cannot build an unambiguous mapping."
                    )
        if hit is None:
            unmatched.append(urdf_name)
        else:
            mapping[urdf_name] = hit

    if unmatched:
        raise KeyError(
            f"Could not map URDF joint(s) {unmatched} onto the simulation joints.\n"
            f"  URDF joints     : {urdf_names}\n"
            f"  simulation joints: {sim_names}"
        )

    used = {}
    for urdf_name, sim_name in mapping.items():
        if sim_name in used:
            raise KeyError(
                f"URDF joints {used[sim_name]!r} and {urdf_name!r} both map to simulation joint "
                f"{sim_name!r}."
            )
        used[sim_name] = urdf_name

    return mapping


def format_mapping(mapping: dict[str, str], sim_names: list[str]) -> str:
    """Human-readable mapping table, ordered by the simulation's joint index."""
    inverse = {v: k for k, v in mapping.items()}
    lines = [f"{'idx':>3s}  {'sim joint':<22s} {'urdf joint':<12s}"]
    for i, sim_name in enumerate(sim_names):
        lines.append(f"{i:>3d}  {sim_name:<22s} {inverse.get(sim_name, '-'):<12s}")
    return "\n".join(lines)
