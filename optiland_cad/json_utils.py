"""JSON generation utilities for Optiland optical systems.

Provides helpers to:
  - Export Optiland Optic objects to JSON (surface_group).
  - Load Zemax .zmx files and convert them to Optiland JSON.
  - Extract only the surface_group portion of the JSON.
"""

from __future__ import annotations

import json
import os
from typing import Union

from optiland.fileio import load_zemax_file
from optiland.fileio.optiland_handler import save_optiland_file, load_optiland_file
from optiland.optic import Optic


def optic_to_json(optic: Optic, filepath: str) -> None:
    """Save an Optiland Optic to a JSON file.

    The Optic is first updated (ray-trace) so that semi_apertures are
    populated, then serialised via ``Optic.to_dict()``.

    Args:
        optic: An Optiland Optic instance.
        filepath: Destination path for the JSON file.
    """
    # Ensure semi-apertures are computed
    optic.update()
    save_optiland_file(optic, filepath)


def _inject_semi_apertures(optic: Optic, sg: dict) -> None:
    """Inject semi_aperture values into a surface_group dict.

    Optiland stores semi_aperture on Surface objects only after a paraxial
    ray trace.  This helper ensures the values are computed and written into
    the dict so the STEP exporter can determine physical extent.
    """
    # Force paraxial update so semi_apertures are populated
    optic.update()
    try:
        optic._updater.update_paraxial()
    except Exception:
        pass  # Best-effort; some systems may not support paraxial tracing

    for i, surf in enumerate(optic.surface_group.surfaces):
        sa = getattr(surf, "semi_aperture", None)
        if sa is not None:
            try:
                sa_val = float(sa)
            except (TypeError, ValueError):
                sa_val = None
        else:
            sa_val = None
        sg["surfaces"][i]["semi_aperture"] = sa_val


def optic_to_surface_group_json(optic: Optic, filepath: str) -> None:
    """Save only the surface_group portion of an Optic to JSON.

    This is the minimal data needed by the STEP exporter.

    Args:
        optic: An Optiland Optic instance.
        filepath: Destination path for the JSON file.
    """
    data = optic.to_dict()
    sg = data["surface_group"]
    _inject_semi_apertures(optic, sg)

    with open(filepath, "w") as f:
        json.dump(sg, f, indent=4)


def zemax_to_json(
    zmx_path_or_url: str,
    output_path: str,
    surface_group_only: bool = False,
) -> Optic:
    """Load a Zemax .zmx file and export it as Optiland JSON.

    Args:
        zmx_path_or_url: Local path or URL to a .zmx file.
        output_path: Destination path for the JSON file.
        surface_group_only: If True, export only the surface_group dict.

    Returns:
        The loaded Optic instance.
    """
    optic = load_zemax_file(zmx_path_or_url)
    if surface_group_only:
        optic_to_surface_group_json(optic, output_path)
    else:
        optic_to_json(optic, output_path)
    return optic


def load_surface_group(filepath: str) -> dict:
    """Load a JSON file and return the surface_group dictionary.

    Accepts either a full Optiland JSON (with top-level ``surface_group`` key)
    or a surface-group-only JSON (list of surfaces at top-level ``surfaces``
    key).

    Args:
        filepath: Path to the JSON file.

    Returns:
        A dict with key ``"surfaces"`` containing the list of surface dicts.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON structure is not recognised.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")
    with open(filepath) as f:
        data = json.load(f)

    if "surface_group" in data:
        return data["surface_group"]
    if "surfaces" in data:
        return data
    raise ValueError(
        "JSON does not contain a recognised Optiland structure. "
        "Expected a top-level 'surface_group' or 'surfaces' key."
    )
