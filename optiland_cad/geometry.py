"""Surface profile computation from Optiland geometry dictionaries.

Translates the ``geometry`` dict (type, radius, conic, coefficients …) into
a callable ``sag(r) -> z`` for rotationally symmetric surfaces, plus a
``sag_xy(x, y) -> z`` for the general case.

Only **v1** geometry types are supported:
  - Plane
  - StandardGeometry  (sphere / conic)
  - EvenAsphere       (rotationally symmetric polynomial)

Unsupported types raise :class:`UnsupportedGeometryError`.
"""

from __future__ import annotations

import math
from typing import Callable


# ── Exceptions ─────────────────────────────────────────────────────────────

class UnsupportedGeometryError(Exception):
    """Raised when an unsupported geometry type is encountered."""


# ── Supported geometry types (v1) ─────────────────────────────────────────

SUPPORTED_TYPES = frozenset({"Plane", "StandardGeometry", "EvenAsphere"})

UNSUPPORTED_TYPES = frozenset({
    "OddAsphere",
    "PolynomialGeometry",
    "ChebyshevPolynomialGeometry",
    "ZernikePolynomialGeometry",
    "BiconicGeometry",
    "ToroidalGeometry",
    "PlaneGrating",
    "StandardGratingGeometry",
    "GridSagGeometry",
    "NurbsGeometry",
    "ForbesQ2dGeometry",
    "ForbesQNormalSlopeGeometry",
})


# ── Sag functions ─────────────────────────────────────────────────────────

def _conic_sag(r: float, R: float, k: float) -> float:
    """Standard conic sag: z = r^2 / (R*(1 + sqrt(1 - (1+k)*r^2/R^2)))."""
    if math.isinf(R) or R == 0:
        return 0.0
    r2 = r * r
    denom_inner = 1.0 - (1.0 + k) * r2 / (R * R)
    # Clamp to avoid sqrt of negative due to floating point
    denom_inner = max(denom_inner, 0.0)
    return r2 / (R * (1.0 + math.sqrt(denom_inner)))


def _even_asphere_sag(r: float, R: float, k: float, coefficients: list[float]) -> float:
    """Even asphere sag = conic_sag + sum(C_i * r^(2i)), i starting at 1."""
    z = _conic_sag(r, R, k)
    r2 = r * r
    rp = r2  # r^2 initially
    for c in coefficients:
        z += c * rp
        rp *= r2  # next even power
    return z


def make_sag_function(geometry: dict) -> Callable[[float], float]:
    """Return a function ``sag(r) -> z`` for a rotationally symmetric surface.

    Args:
        geometry: The ``"geometry"`` dict from a surface.

    Returns:
        A callable that takes radial distance *r* and returns the sag *z*.

    Raises:
        UnsupportedGeometryError: If the geometry type is not in v1 scope.
    """
    gtype = geometry.get("type", "unknown")

    if gtype == "Plane":
        return lambda r: 0.0

    if gtype == "StandardGeometry":
        R = float(geometry.get("radius", math.inf))
        k = float(geometry.get("conic", 0.0))
        return lambda r, _R=R, _k=k: _conic_sag(r, _R, _k)

    if gtype == "EvenAsphere":
        R = float(geometry.get("radius", math.inf))
        k = float(geometry.get("conic", 0.0))
        coeffs = [float(c) for c in geometry.get("coefficients", [])]
        return lambda r, _R=R, _k=k, _c=coeffs: _even_asphere_sag(r, _R, _k, _c)

    # Unsupported
    raise UnsupportedGeometryError(
        f"Geometry type '{gtype}' is not supported in v1. "
        f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}."
    )


def validate_geometries(surface_group: dict) -> None:
    """Check all surfaces for unsupported geometry types.

    Raises:
        UnsupportedGeometryError: With a message listing the first
            unsupported surface.
    """
    surfaces = surface_group.get("surfaces", [])
    for i, surf in enumerate(surfaces):
        geom = surf.get("geometry", {})
        gtype = geom.get("type", "unknown")
        if gtype not in SUPPORTED_TYPES and gtype != "unknown":
            raise UnsupportedGeometryError(
                f"Surface {i} has unsupported geometry type '{gtype}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}."
            )
