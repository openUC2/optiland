"""Component categorization for Optiland surface groups.

Identifies optical components (Lens, Mirror, Objective) from a list of
surface dictionaries and groups surfaces into logical parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ComponentType(Enum):
    """Optical component categories."""

    LENS = "lens"
    MIRROR = "mirror"
    OBJECTIVE = "objective"   # multi-element lens assembly


@dataclass
class OpticalComponent:
    """A single categorised optical component.

    Attributes:
        name: Human-readable label (e.g. ``"Lens_1"``).
        component_type: One of :class:`ComponentType`.
        surface_indices: Indices into the *optical* surfaces list (excluding
            object/image surfaces).
        surfaces: The raw surface dicts for this component.
    """

    name: str
    component_type: ComponentType
    surface_indices: list[int] = field(default_factory=list)
    surfaces: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_reflective(surf: dict) -> bool:
    """Return True when a surface has a reflective interaction model."""
    im = surf.get("interaction_model")
    if im is None:
        return False
    return im.get("is_reflective", False)


def _material_is_air(mat: dict | None) -> bool:
    """Return True when the material represents air / vacuum."""
    if mat is None:
        return True
    mat_type = mat.get("type", "")
    if mat_type == "IdealMaterial":
        idx = mat.get("index", 1.0)
        return abs(idx - 1.0) < 1e-6
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def categorize_surfaces(surface_group: dict) -> List[OpticalComponent]:
    """Categorize a surface_group into optical components.

    Rules (v1):
      * **Mirror**: any surface whose ``interaction_model.is_reflective`` is
        ``True``.  Consecutive mirrors sharing the same assembly are grouped
        together (e.g. Cassegrain telescopes).
      * **Lens**: a pair of surfaces that bound a glass element – the front
        surface has ``material_post`` that is *not* air, and the back surface
        transitions back to air (or to a different glass for cemented
        elements).
      * **Objective**: when more than one lens element is found, the whole
        assembly is also wrapped as a single *Objective* component (the
        individual lens components remain available).

    Args:
        surface_group: Dict with ``"surfaces"`` key as produced by
            ``Optic.to_dict()["surface_group"]``.

    Returns:
        A list of :class:`OpticalComponent` objects.
    """
    surfaces = surface_group["surfaces"]

    # Filter out object/image surfaces (first and last)
    optical: list[tuple[int, dict]] = []
    for i, s in enumerate(surfaces):
        stype = s.get("type", "")
        if stype in ("ObjectSurface",):
            continue
        if stype in ("ImageSurface",):
            continue
        optical.append((i, s))

    if not optical:
        return []

    components: list[OpticalComponent] = []
    mirrors: list[tuple[int, dict]] = []
    lenses: list[OpticalComponent] = []

    # --- pass 1: identify mirrors ---
    mirror_idx = 1
    for idx, (abs_i, surf) in enumerate(optical):
        if _is_reflective(surf):
            comp = OpticalComponent(
                name=f"Mirror_{mirror_idx}",
                component_type=ComponentType.MIRROR,
                surface_indices=[abs_i],
                surfaces=[surf],
            )
            mirrors.append((abs_i, surf))
            components.append(comp)
            mirror_idx += 1

    # --- pass 2: identify lens elements ---
    # Walk optical surfaces and pair them by glass boundaries.
    lens_idx = 1
    i = 0
    while i < len(optical):
        abs_i, surf = optical[i]

        # Skip mirrors
        if _is_reflective(surf):
            i += 1
            continue

        mat_post = surf.get("material_post")
        if not _material_is_air(mat_post):
            # This surface starts a glass element.
            # Collect surfaces until we return to air.
            element_indices = [abs_i]
            element_surfaces = [surf]
            j = i + 1
            while j < len(optical):
                abs_j, surf_j = optical[j]
                element_indices.append(abs_j)
                element_surfaces.append(surf_j)
                mat_post_j = surf_j.get("material_post")
                if _material_is_air(mat_post_j):
                    break
                j += 1
            comp = OpticalComponent(
                name=f"Lens_{lens_idx}",
                component_type=ComponentType.LENS,
                surface_indices=element_indices,
                surfaces=element_surfaces,
            )
            lenses.append(comp)
            components.append(comp)
            lens_idx += 1
            i = j + 1
        else:
            i += 1

    # --- pass 3: wrap multi-element assemblies as Objective ---
    if len(lenses) > 1:
        all_indices = []
        all_surfs = []
        for lc in lenses:
            all_indices.extend(lc.surface_indices)
            all_surfs.extend(lc.surfaces)
        objective = OpticalComponent(
            name="Objective_1",
            component_type=ComponentType.OBJECTIVE,
            surface_indices=sorted(set(all_indices)),
            surfaces=all_surfs,
        )
        components.append(objective)

    return components
