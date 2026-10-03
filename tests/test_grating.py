from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays import RealRays

from .utils import assert_allclose


@pytest.fixture
def flat_transmission_grating():
    """flat transmission grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=10)
    lens.surfaces.add(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.surfaces.add(
        index=3,
        radius=be.inf,
        thickness=30,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.surfaces.add(index=4)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


@pytest.fixture
def curved_transmission_grating():
    """curved transmission grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, radius=be.inf, thickness=10)
    lens.surfaces.add(index=2, radius=be.inf, thickness=5, material="N-BK7")
    lens.surfaces.add(
        index=3,
        radius=50.0,
        thickness=30,
        conic=1.0,
        surface_type="grating",
        grating_order=-1,
        grating_period=5.0,
        groove_orientation_angle=0.0,
        is_stop=True,
    )
    lens.surfaces.add(index=4)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


@pytest.fixture
def curved_reflective_grating():
    """curved reflective grating with 3 fields and 1 wavelength"""
    lens = Optic()

    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=70,
        thickness=-30,
        material="mirror",
        surface_type="grating",
        is_stop=True,
        grating_period=5.0,
        grating_order=1,
        groove_orientation_angle=0.0,
    )
    lens.surfaces.add(index=2)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=15)

    # add field
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.fields.add(y=10)
    lens.fields.add(y=0, x=10)

    # add wavelength
    lens.wavelengths.add(value=0.587, is_primary=True)

    lens.updater.update_paraxial()

    return lens


def test_flat_grating_transmission(set_test_backend, flat_transmission_grating):
    lens = flat_transmission_grating
    wv = 0.587
    Px = 0.0
    Py = 0.0
    Hx = 0.0
    Hy = 0.0
    # axial ray central field
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # marginal ray central field
    Px = 0.0
    Py = 1.0
    Hx = 0.0
    Hy = 0.0
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0345602649, 0.0216899611, 0.9991672201]
    )


def test_curved_grating_transmission(set_test_backend, curved_transmission_grating):
    lens = curved_transmission_grating
    wv = 0.587
    Px = 0.0
    Py = 0.0
    Hx = 0.0
    Hy = 0.0
    # axial ray central field
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.1174, 0.9930847094])
    # marginal ray central field
    Px = 0.0
    Py = 1.0
    Hx = 0.0
    Hy = 0.0
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], [0.0, -0.0379603895, 0.9992792447])
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0229384233, 0.0764682608, 0.9968081229]
    )


def test_curved_grating_reflection(set_test_backend, curved_reflective_grating):
    lens = curved_reflective_grating
    wv = 0.587
    # generic ray
    Px = -0.15
    Py = 0.7
    Hx = 0.2
    Hy = 0.8
    ray = lens.trace_generic(Hx=Hx, Hy=Hy, Px=Px, Py=Py, wavelength=wv)
    # the image lies at negative thickness: the reflected ray travels back
    assert_allclose(
        [ray.L[0], ray.M[0], ray.N[0]], [0.0040370337, 0.4006583117, -0.9162186527]
    )


def test_gratingdiffract_reflective_order_zero_is_a_reflection(set_test_backend):
    """At order 0 the grating kernel reflects exactly as a mirror does."""
    theta = be.array([0.0, 0.3, 0.8])
    normal = (be.zeros(3), be.full((3,), -math.sin(0.2)), be.full((3,), math.cos(0.2)))
    grating_vector = (
        be.zeros(3),
        be.full((3,), math.cos(0.2)),
        be.full((3,), math.sin(0.2)),
    )

    def incident():
        zero = be.zeros(3)
        return RealRays(
            zero, zero, zero, zero, be.sin(theta), be.cos(theta), be.ones(3), 0.61
        )

    mirror = incident()
    mirror.reflect(*normal)
    grating = incident()
    grating.gratingdiffract(*normal, *grating_vector, 0, 1 / 1.2, 1.0, 1.0, True)

    assert_allclose(grating.L, mirror.L)
    assert_allclose(grating.M, mirror.M)
    assert_allclose(grating.N, mirror.N)


@pytest.mark.parametrize("radius", [be.inf, 70.0])
def test_reflective_grating_order_zero_traces_as_a_mirror(set_test_backend, radius):
    """Same ray through a reflective grating at order 0 and through a mirror."""
    rays = {}
    for kind in ("grating", "mirror"):
        grating = {
            "surface_type": "grating",
            "grating_period": 5.0,
            "grating_order": 0,
            "groove_orientation_angle": 0.0,
        }
        lens = Optic()
        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        lens.surfaces.add(
            index=1,
            radius=radius,
            thickness=-30,
            material="mirror",
            is_stop=True,
            **(grating if kind == "grating" else {}),
        )
        lens.surfaces.add(index=2)
        lens.set_aperture(aperture_type="EPD", value=15)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)
        lens.fields.add(y=10)
        lens.wavelengths.add(value=0.587, is_primary=True)
        rays[kind] = lens.trace_generic(
            Hx=0.2, Hy=0.8, Px=-0.15, Py=0.7, wavelength=0.587
        )

    for attr in ("x", "y", "z", "L", "M", "N", "opd"):
        assert_allclose(getattr(rays["grating"], attr), getattr(rays["mirror"], attr))


def test_tilted_reflective_grating_follows_the_grating_equation(set_test_backend):
    """One reflective grating surface tilted by 47.5 deg, 1200 l/mm, 610 nm."""
    theta = math.radians(47.5)
    period, order, wavelength = 1 / 1.2, -1, 0.61

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, x=0.0, y=0.0, z=0.0, is_stop=True)
    lens.surfaces.add(
        index=2,
        x=0.0,
        y=0.0,
        z=10.0,
        rx=theta,
        material="mirror",
        surface_type="grating",
        grating_period=period,
        grating_order=order,
        groove_orientation_angle=0.0,
    )
    lens.surfaces.add(index=3, x=0.0, y=0.0, z=-20.0)
    lens.set_aperture(aperture_type="EPD", value=2.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=wavelength, is_primary=True)
    ray = lens.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.0, wavelength=wavelength)

    # In the grating's frame the beam arrives at (0, sin θ, cos θ); the order
    # adds m λ / d along the grating vector (local y) and the normal part
    # reverses.
    s = math.sin(theta) + order * wavelength / period
    c = math.sqrt(1 - s**2)
    expected = [
        0.0,
        s * math.cos(theta) + c * math.sin(theta),
        s * math.sin(theta) - c * math.cos(theta),
    ]
    assert_allclose([ray.L[0], ray.M[0], ray.N[0]], expected)


def test_paraxial_flat_grating_transmission(
    set_test_backend, flat_transmission_grating
):
    lens = flat_transmission_grating
    wv = 0.587
    Hy = 0.0
    Py = 0.0
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.1174, 3.522])
    Hy = 0.0
    Py = 1.0
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.1174, 11.022])
    Hy = 0.8
    Py = 0.8
    lens.paraxial.trace(Hy=Hy, Py=Py, wavelength=wv)
    u = lens.surfaces.u[-1].item()
    y = lens.surfaces.y[-1].item()
    assert_allclose([u, y], [0.25794083, 13.73822504])
