"""optiland_cad – Optiland JSON to STEP exporter using CadQuery.

Converts Optiland surface_group data into CAD-ready STEP models suitable for
import into Autodesk Inventor and other CAD systems.

Coordinate convention (matches optics):
    +Z  = optical axis
    XY  = plane perpendicular to optical axis
    z=0 = first optical surface vertex
    Units: mm
"""

from __future__ import annotations

__version__ = "0.1.0"
