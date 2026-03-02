"""STEP exporter – builds CadQuery solids from Optiland surface groups.

Coordinate convention:
    +Z  = optical axis (propagation direction)
    XY  = plane perpendicular to the optical axis
    z=0 = vertex of the first optical surface
    Units: mm

Each optical element (lens or mirror) is exported as an individual STEP file.
Lens elements are created by revolving the front & back sag profiles around
the Z-axis and combining them via boolean operations.  Mirrors are created
similarly from a single reflective surface.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import cadquery as cq

from optiland_cad.categorize import (
    ComponentType,
    OpticalComponent,
    categorize_surfaces,
    _material_is_air,
    _is_reflective,
)
from optiland_cad.geometry import (
    UnsupportedGeometryError,
    make_sag_function,
    validate_geometries,
)


# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_SEMI_APERTURE = 12.7  # mm fallback half-diameter
_MIRROR_THICKNESS = 3.0        # mm  nominal substrate thickness for mirrors
_EDGE_MARGIN = 0.5             # mm  minimum edge thickness for lenses
_PROFILE_POINTS = 128          # number of sample points for revolve profile


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_semi_aperture(surf: dict, default: float = _DEFAULT_SEMI_APERTURE) -> float:
    """Extract semi-aperture from a surface dict, with fallback."""
    sa = surf.get("semi_aperture")
    if sa is not None:
        return float(sa)
    # Try physical aperture extent
    aper = surf.get("aperture")
    if aper and isinstance(aper, dict):
        r_max = aper.get("r_max")
        if r_max is not None and not math.isinf(float(r_max)):
            return float(r_max)
    return default


def _z_of_surface(surf: dict) -> float:
    """Return the absolute z-position of a surface from its coordinate system."""
    geom = surf.get("geometry", {})
    cs = geom.get("cs", {})
    return float(cs.get("z", 0.0))


def _revolve_profile(
    sag_func,
    semi_aperture: float,
    z_offset: float,
    n_pts: int = _PROFILE_POINTS,
) -> list[tuple[float, float]]:
    """Sample a rotationally symmetric sag function into (r, z) pairs.

    Returns points from r=0 to r=semi_aperture suitable for building a
    CadQuery wire/spline in the RZ half-plane (Y=0, X=r, Z=z).
    """
    pts = []
    for i in range(n_pts + 1):
        r = semi_aperture * i / n_pts
        z = sag_func(r) + z_offset
        pts.append((r, z))
    return pts


# ── Solid builders ─────────────────────────────────────────────────────────

def _dedup_profile(pts: list[tuple[float, float]], tol: float = 1e-9) -> list[tuple[float, float]]:
    """Remove consecutive duplicate points from a profile."""
    if not pts:
        return pts
    result = [pts[0]]
    for p in pts[1:]:
        prev = result[-1]
        if abs(p[0] - prev[0]) > tol or abs(p[1] - prev[1]) > tol:
            result.append(p)
    # Also check that the last point isn't the same as the first (close will handle it)
    if len(result) > 1:
        if abs(result[-1][0] - result[0][0]) < tol and abs(result[-1][1] - result[0][1]) < tol:
            result = result[:-1]
    return result


def _profile_to_workplane(profile_pts: list[tuple[float, float]]) -> cq.Workplane:
    """Create a CadQuery Workplane with a closed polyline from (x, z) profile pts.

    The profile is drawn on the XZ plane (Y=0).  CadQuery's ``revolve()``
    on a Workplane rotates around the Z-axis by default when axisStart/
    axisEnd reference (0,0,0)→(0,0,1).

    Points must form a closed polygon with all X >= 0 (half-plane).
    """
    # CadQuery Workplane.polyline expects 2D tuples when working on a plane.
    # We work on the "XZ" workplane so (u, v) maps to (x, z).
    wp = cq.Workplane("XZ")

    # Move to the first point, then draw lines through the rest, then close
    first = profile_pts[0]
    wp = wp.moveTo(first[0], first[1])
    for pt in profile_pts[1:]:
        wp = wp.lineTo(pt[0], pt[1])
    wp = wp.close()

    return wp


def _build_lens_solid(
    front_sag,
    back_sag,
    front_z: float,
    back_z: float,
    semi_aperture: float,
) -> cq.Workplane:
    """Build a rotationally symmetric lens element by revolving the profile.

    The lens is bounded between the front surface (lower z) and back surface
    (higher z).  A cylindrical barrel connects them at ``r = semi_aperture``.

    The solid is centred on x=y=0 with the front vertex at ``front_z``.
    """
    n = _PROFILE_POINTS

    # Sample sag profiles → list of (r, z) pairs
    front_pts = _revolve_profile(front_sag, semi_aperture, front_z, n)
    back_pts = _revolve_profile(back_sag, semi_aperture, back_z, n)

    # Build the closed profile as a polyline in the XZ half-plane.
    # Sequence: front (r=0→SA) → barrel → back (r=SA→0) → implicit close
    profile: list[tuple[float, float]] = []

    # Front surface from axis outward
    for r, z in front_pts:
        profile.append((r, z))

    # Barrel edge connecting front edge to back edge at r = semi_aperture
    back_edge_z = back_pts[-1][1]
    front_edge_z = front_pts[-1][1]
    if abs(back_edge_z - front_edge_z) > 1e-9:
        profile.append((semi_aperture, back_edge_z))

    # Back surface from SA back to axis
    for r, z in reversed(back_pts):
        profile.append((r, z))

    # Deduplicate consecutive identical points (avoids zero-length edges)
    profile = _dedup_profile(profile)

    # Revolve the closed profile 360° around the Z-axis
    wp = _profile_to_workplane(profile)
    result = wp.revolve(
        angleDegrees=360.0,
        axisStart=(0, 0),
        axisEnd=(0, 1),
    )
    return result


def _build_mirror_solid(
    sag_func,
    z_offset: float,
    semi_aperture: float,
    substrate_thickness: float = _MIRROR_THICKNESS,
) -> cq.Workplane:
    """Build a rotationally symmetric mirror substrate.

    Creates a solid with the reflective surface on the front and a flat back.
    """
    n = _PROFILE_POINTS

    front_pts = _revolve_profile(sag_func, semi_aperture, z_offset, n)

    # Back-plane behind the deepest sag point
    min_z = min(z for _, z in front_pts)
    back_z = min_z - substrate_thickness

    # Profile: front (r=0→SA) → barrel → flat back (r=SA→0)
    profile: list[tuple[float, float]] = []

    for r, z in front_pts:
        profile.append((r, z))

    # barrel
    profile.append((semi_aperture, back_z))

    # flat back
    profile.append((0.0, back_z))

    profile = _dedup_profile(profile)
    wp = _profile_to_workplane(profile)
    result = wp.revolve(
        angleDegrees=360.0,
        axisStart=(0, 0),
        axisEnd=(0, 1),
    )
    return result


# ── High-level export ─────────────────────────────────────────────────────

def _surfaces_list(surface_group: dict) -> list[dict]:
    """Return the list of surface dicts."""
    return surface_group.get("surfaces", [])


def _compute_z_positions(surfaces: list[dict]) -> list[float]:
    """Compute absolute z-positions from thickness chain.

    Surface 0 (object) is skipped.  The first real surface is placed at z=0.
    """
    positions = [0.0] * len(surfaces)
    # surface_group stores cs.z already as absolute positions when exported
    # via Optic.to_dict().  Use those if available.
    has_cs = all(
        "geometry" in s and "cs" in s.get("geometry", {})
        for s in surfaces
    )
    if has_cs:
        # Use the coordinate-system z values
        for i, s in enumerate(surfaces):
            positions[i] = _z_of_surface(s)
        # Normalise so that the first non-object surface is at z=0
        first_optical_z = None
        for i, s in enumerate(surfaces):
            if s.get("type") not in ("ObjectSurface",):
                first_optical_z = positions[i]
                break
        if first_optical_z is not None:
            positions = [z - first_optical_z for z in positions]
        return positions

    # Fallback: accumulate thickness
    z = 0.0
    for i, s in enumerate(surfaces):
        if i == 0:
            positions[i] = 0.0
            continue
        positions[i] = z
        t = s.get("thickness", 0.0)
        if t is not None and not math.isinf(float(t)):
            z += float(t)
    return positions


def build_component_solid(
    component: OpticalComponent,
    surfaces: list[dict],
    z_positions: list[float],
) -> cq.Workplane:
    """Build a CadQuery Workplane solid for an :class:`OpticalComponent`.

    Args:
        component: The categorised component.
        surfaces: The *full* list of surface dicts (including object/image).
        z_positions: Pre-computed absolute z positions aligned with *surfaces*.

    Returns:
        A CadQuery :class:`~cadquery.Workplane` containing the solid.

    Raises:
        UnsupportedGeometryError: If any referenced surface has an
            unsupported geometry type.
        ValueError: If the component cannot be built (e.g. missing surfaces).
    """
    idx = component.surface_indices

    if component.component_type == ComponentType.MIRROR:
        # Single reflective surface
        si = idx[0]
        surf = surfaces[si]
        geom = surf.get("geometry", {})
        sag_func = make_sag_function(geom)
        sa = _get_semi_aperture(surf)
        z_pos = z_positions[si]
        return _build_mirror_solid(sag_func, z_pos, sa)

    # Lens (or Objective – build each element separately and fuse)
    if component.component_type in (ComponentType.LENS, ComponentType.OBJECTIVE):
        element_results: list[cq.Workplane] = []

        # Walk idx: pairs of surfaces bounding glass elements
        i = 0
        while i < len(idx):
            si_front = idx[i]
            if i + 1 >= len(idx):
                break
            si_back = idx[i + 1]

            surf_front = surfaces[si_front]
            surf_back = surfaces[si_back]

            geom_front = surf_front.get("geometry", {})
            geom_back = surf_back.get("geometry", {})

            sag_front = make_sag_function(geom_front)
            sag_back = make_sag_function(geom_back)

            # Use maximum semi-aperture of the pair
            sa_front = _get_semi_aperture(surf_front)
            sa_back = _get_semi_aperture(surf_back)
            sa = max(sa_front, sa_back)

            z_front = z_positions[si_front]
            z_back = z_positions[si_back]

            result = _build_lens_solid(sag_front, sag_back, z_front, z_back, sa)
            element_results.append(result)

            # Check if back surface material_post is also glass (cemented)
            mat_post = surf_back.get("material_post")
            if not _material_is_air(mat_post):
                i += 1  # next element starts at current back surface
            else:
                i += 2  # skip to next element pair

        if not element_results:
            raise ValueError(
                f"Could not build solid for component '{component.name}': "
                "no valid surface pairs found."
            )

        # Fuse all elements into one workplane
        if len(element_results) == 1:
            return element_results[0]

        # Union multiple solids
        result = element_results[0]
        for wp in element_results[1:]:
            result = result.union(wp)
        return result

    raise ValueError(f"Unknown component type: {component.component_type}")


def export_surface_group(
    surface_group: dict,
    output_dir: str,
    file_prefix: str = "optiland",
) -> list[str]:
    """Export all components of a surface_group to individual STEP files.

    Args:
        surface_group: The surface-group dictionary (with ``"surfaces"`` key).
        output_dir: Directory to write STEP files into (created if missing).
        file_prefix: Prefix for output filenames.

    Returns:
        A list of created STEP file paths.

    Raises:
        UnsupportedGeometryError: If any surface uses an unsupported geometry.
    """
    # Validate first
    validate_geometries(surface_group)

    surfaces = _surfaces_list(surface_group)
    z_positions = _compute_z_positions(surfaces)
    components = categorize_surfaces(surface_group)

    os.makedirs(output_dir, exist_ok=True)

    created_files: list[str] = []
    for comp in components:
        wp = build_component_solid(comp, surfaces, z_positions)

        filename = f"{file_prefix}_{comp.name}.step"
        filepath = os.path.join(output_dir, filename)

        cq.exporters.export(wp, filepath, exportType="STEP")

        created_files.append(filepath)

    return created_files
