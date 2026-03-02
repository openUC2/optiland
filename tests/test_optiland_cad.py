"""Tests for optiland_cad – the Optiland JSON to STEP exporter.

Requires: cadquery, optiland (with samples).
Run with: pytest tests/test_optiland_cad.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path (pytest built-in)."""
    return str(tmp_path)


@pytest.fixture
def edmund_surface_group():
    """Surface group dict for Edmund_49_847 (singlet lens)."""
    from optiland.samples import Edmund_49_847
    from optiland_cad.json_utils import _inject_semi_apertures

    optic = Edmund_49_847()
    data = optic.to_dict()
    sg = data["surface_group"]
    _inject_semi_apertures(optic, sg)
    return sg


@pytest.fixture
def aspheric_surface_group():
    """Surface group dict for AsphericSinglet (even asphere)."""
    from optiland.samples import AsphericSinglet
    from optiland_cad.json_utils import _inject_semi_apertures

    optic = AsphericSinglet()
    data = optic.to_dict()
    sg = data["surface_group"]
    _inject_semi_apertures(optic, sg)
    return sg


@pytest.fixture
def doublet_surface_group():
    """Surface group dict for TelescopeDoublet (two-element objective)."""
    from optiland.samples import TelescopeDoublet
    from optiland_cad.json_utils import _inject_semi_apertures

    optic = TelescopeDoublet()
    data = optic.to_dict()
    sg = data["surface_group"]
    _inject_semi_apertures(optic, sg)
    return sg


# ── Test: geometry module ─────────────────────────────────────────────────


class TestGeometry:
    """Tests for optiland_cad.geometry sag functions and validation."""

    def test_plane_sag(self):
        from optiland_cad.geometry import make_sag_function

        sag = make_sag_function({"type": "Plane"})
        assert sag(0.0) == 0.0
        assert sag(10.0) == 0.0

    def test_standard_sag_sphere(self):
        from optiland_cad.geometry import make_sag_function

        R = 20.0
        geom = {"type": "StandardGeometry", "radius": R, "conic": 0.0}
        sag = make_sag_function(geom)
        # At r=0, sag should be 0
        assert abs(sag(0.0)) < 1e-12
        # At small r, sag ≈ r^2/(2R) (paraxial approximation, not exact)
        r = 1.0
        expected = r**2 / (2 * R)
        assert abs(sag(r) - expected) < 1e-4

    def test_standard_sag_parabola(self):
        from optiland_cad.geometry import make_sag_function

        R = 50.0
        geom = {"type": "StandardGeometry", "radius": R, "conic": -1.0}
        sag = make_sag_function(geom)
        # Parabola: z = r^2 / (2R)
        r = 5.0
        expected = r**2 / (2 * R)
        assert abs(sag(r) - expected) < 1e-10

    def test_even_asphere_sag(self):
        from optiland_cad.geometry import make_sag_function

        geom = {
            "type": "EvenAsphere",
            "radius": 20.0,
            "conic": 0.0,
            "coefficients": [-2.248851e-4, -4.690412e-6],
        }
        sag = make_sag_function(geom)
        assert abs(sag(0.0)) < 1e-12
        # The aspheric terms modify the sag; just ensure it's finite
        val = sag(5.0)
        assert math.isfinite(val)

    def test_unsupported_geometry_raises(self):
        from optiland_cad.geometry import make_sag_function, UnsupportedGeometryError

        with pytest.raises(UnsupportedGeometryError, match="BiconicGeometry"):
            make_sag_function({"type": "BiconicGeometry"})

    def test_validate_geometries_ok(self, edmund_surface_group):
        from optiland_cad.geometry import validate_geometries

        # Should not raise
        validate_geometries(edmund_surface_group)

    def test_validate_geometries_fail(self):
        from optiland_cad.geometry import validate_geometries, UnsupportedGeometryError

        bad_sg = {
            "surfaces": [
                {"type": "ObjectSurface", "geometry": {"type": "Plane"}},
                {
                    "type": "Surface",
                    "geometry": {"type": "BiconicGeometry"},
                    "interaction_model": {"is_reflective": False},
                },
            ]
        }
        with pytest.raises(UnsupportedGeometryError, match="BiconicGeometry"):
            validate_geometries(bad_sg)


# ── Test: categorization ──────────────────────────────────────────────────


class TestCategorize:
    """Tests for optiland_cad.categorize."""

    def test_singlet_lens(self, edmund_surface_group):
        from optiland_cad.categorize import categorize_surfaces, ComponentType

        comps = categorize_surfaces(edmund_surface_group)
        lens_comps = [c for c in comps if c.component_type == ComponentType.LENS]
        assert len(lens_comps) == 1
        assert lens_comps[0].name == "Lens_1"

    def test_doublet_objective(self, doublet_surface_group):
        from optiland_cad.categorize import categorize_surfaces, ComponentType

        comps = categorize_surfaces(doublet_surface_group)
        lens_comps = [c for c in comps if c.component_type == ComponentType.LENS]
        obj_comps = [c for c in comps if c.component_type == ComponentType.OBJECTIVE]
        assert len(lens_comps) == 2
        # Objective wraps all lenses
        assert len(obj_comps) == 1

    def test_mirror_detection(self):
        from optiland_cad.categorize import categorize_surfaces, ComponentType

        sg = {
            "surfaces": [
                {
                    "type": "ObjectSurface",
                    "geometry": {"type": "Plane", "cs": {"z": 0}},
                },
                {
                    "type": "Surface",
                    "geometry": {
                        "type": "StandardGeometry",
                        "cs": {"z": 0},
                        "radius": -100.0,
                        "conic": -1.0,
                    },
                    "interaction_model": {"is_reflective": True},
                    "material_post": {"type": "IdealMaterial", "index": 1.0},
                    "thickness": -50,
                },
                {
                    "type": "ImageSurface",
                    "geometry": {"type": "Plane", "cs": {"z": -50}},
                },
            ]
        }
        comps = categorize_surfaces(sg)
        mirror_comps = [c for c in comps if c.component_type == ComponentType.MIRROR]
        assert len(mirror_comps) == 1
        assert mirror_comps[0].name == "Mirror_1"


# ── Test: JSON utilities ──────────────────────────────────────────────────


class TestJsonUtils:
    """Tests for optiland_cad.json_utils."""

    def test_optic_to_json_roundtrip(self, tmp_dir):
        from optiland.samples import Edmund_49_847
        from optiland_cad.json_utils import optic_to_json, load_surface_group

        optic = Edmund_49_847()
        path = os.path.join(tmp_dir, "test.json")
        optic_to_json(optic, path)

        assert os.path.exists(path)
        sg = load_surface_group(path)
        assert "surfaces" in sg
        assert len(sg["surfaces"]) > 0

    def test_surface_group_only(self, tmp_dir):
        from optiland.samples import Edmund_49_847
        from optiland_cad.json_utils import optic_to_surface_group_json, load_surface_group

        optic = Edmund_49_847()
        path = os.path.join(tmp_dir, "sg.json")
        optic_to_surface_group_json(optic, path)

        sg = load_surface_group(path)
        assert "surfaces" in sg
        # semi_aperture should be injected
        has_sa = any(
            s.get("semi_aperture") is not None for s in sg["surfaces"]
        )
        assert has_sa

    def test_load_nonexistent_raises(self, tmp_dir):
        from optiland_cad.json_utils import load_surface_group

        with pytest.raises(FileNotFoundError):
            load_surface_group(os.path.join(tmp_dir, "nope.json"))

    def test_load_bad_json_raises(self, tmp_dir):
        from optiland_cad.json_utils import load_surface_group

        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            json.dump({"foo": "bar"}, f)
        with pytest.raises(ValueError, match="recognised"):
            load_surface_group(path)


# ── Test: STEP export (requires cadquery) ─────────────────────────────────

# These tests are skipped when cadquery is not installed.
try:
    import cadquery  # noqa: F401
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False


@pytest.mark.skipif(not HAS_CADQUERY, reason="cadquery not installed")
class TestStepExport:
    """Integration tests for STEP export."""

    def test_singlet_export(self, tmp_dir, edmund_surface_group):
        from optiland_cad.exporter import export_surface_group

        files = export_surface_group(edmund_surface_group, tmp_dir, "test")
        assert len(files) >= 1
        for f in files:
            assert os.path.exists(f)
            assert f.endswith(".step")
            # File should not be empty
            assert os.path.getsize(f) > 100

    def test_asphere_export(self, tmp_dir, aspheric_surface_group):
        from optiland_cad.exporter import export_surface_group

        files = export_surface_group(aspheric_surface_group, tmp_dir, "asphere")
        assert len(files) >= 1
        for f in files:
            assert os.path.exists(f)

    def test_doublet_export(self, tmp_dir, doublet_surface_group):
        from optiland_cad.exporter import export_surface_group

        files = export_surface_group(doublet_surface_group, tmp_dir, "doublet")
        assert len(files) >= 1
        # Should have individual lens files plus an objective
        step_names = [os.path.basename(f) for f in files]
        assert any("Lens" in n for n in step_names)

    def test_solid_centered_on_axis(self, edmund_surface_group):
        """Assert the exported solid is centred on x=y=0."""
        from optiland_cad.exporter import (
            build_component_solid,
            _compute_z_positions,
            _surfaces_list,
        )
        from optiland_cad.categorize import categorize_surfaces, ComponentType

        surfaces = _surfaces_list(edmund_surface_group)
        z_positions = _compute_z_positions(surfaces)
        comps = categorize_surfaces(edmund_surface_group)
        lens_comps = [c for c in comps if c.component_type == ComponentType.LENS]
        assert len(lens_comps) > 0

        wp = build_component_solid(lens_comps[0], surfaces, z_positions)
        bb = wp.val().BoundingBox()
        # Center of bounding box in x and y should be near zero
        cx = (bb.xmin + bb.xmax) / 2.0
        cy = (bb.ymin + bb.ymax) / 2.0
        assert abs(cx) < 0.01, f"Center X = {cx}, expected ~0"
        assert abs(cy) < 0.01, f"Center Y = {cy}, expected ~0"

    def test_first_surface_at_z0(self, edmund_surface_group):
        """Assert the first optical surface is aligned to z=0."""
        from optiland_cad.exporter import (
            build_component_solid,
            _compute_z_positions,
            _surfaces_list,
        )
        from optiland_cad.categorize import categorize_surfaces, ComponentType

        surfaces = _surfaces_list(edmund_surface_group)
        z_positions = _compute_z_positions(surfaces)
        comps = categorize_surfaces(edmund_surface_group)
        lens_comps = [c for c in comps if c.component_type == ComponentType.LENS]
        assert len(lens_comps) > 0

        wp = build_component_solid(lens_comps[0], surfaces, z_positions)
        bb = wp.val().BoundingBox()
        # z_min of the solid should be at or very near z=0
        # (front surface vertex is at z=0)
        assert abs(bb.zmin) < 0.5, f"zmin = {bb.zmin}, expected ~0"

    def test_deterministic_output(self, tmp_dir, edmund_surface_group):
        """Same input must produce same number of files with same names."""
        from optiland_cad.exporter import export_surface_group

        dir1 = os.path.join(tmp_dir, "run1")
        dir2 = os.path.join(tmp_dir, "run2")

        files1 = export_surface_group(edmund_surface_group, dir1, "det")
        files2 = export_surface_group(edmund_surface_group, dir2, "det")

        names1 = sorted(os.path.basename(f) for f in files1)
        names2 = sorted(os.path.basename(f) for f in files2)
        assert names1 == names2, "Exported file names differ between runs"

        # Both runs should produce non-empty files of similar size
        for f1, f2 in zip(sorted(files1), sorted(files2)):
            s1, s2 = os.path.getsize(f1), os.path.getsize(f2)
            assert s1 > 0 and s2 > 0
            assert abs(s1 - s2) / max(s1, s2) < 0.01, (
                f"File sizes differ >1%: {s1} vs {s2}"
            )

    def test_unsupported_geometry_error(self, tmp_dir):
        """Unsupported geometry should raise with clear message."""
        from optiland_cad.exporter import export_surface_group
        from optiland_cad.geometry import UnsupportedGeometryError

        bad_sg = {
            "surfaces": [
                {"type": "ObjectSurface", "geometry": {"type": "Plane", "cs": {"z": 0}}},
                {
                    "type": "Surface",
                    "geometry": {"type": "ZernikePolynomialGeometry", "cs": {"z": 0}},
                    "interaction_model": {"is_reflective": False},
                    "material_post": {"type": "Material", "name": "N-BK7"},
                    "thickness": 5,
                },
                {
                    "type": "Surface",
                    "geometry": {"type": "Plane", "cs": {"z": 5}},
                    "interaction_model": {"is_reflective": False},
                    "material_post": {"type": "IdealMaterial", "index": 1.0},
                    "thickness": 20,
                },
                {"type": "ImageSurface", "geometry": {"type": "Plane", "cs": {"z": 25}}},
            ]
        }
        with pytest.raises(UnsupportedGeometryError, match="ZernikePolynomialGeometry"):
            export_surface_group(bad_sg, tmp_dir, "bad")
