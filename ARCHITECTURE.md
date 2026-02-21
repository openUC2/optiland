# Optiland – Architecture & Technical Deep-Dive

> **Version**: 0.5.9 | **Language**: Python 3.10–3.13 | **License**: MIT

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Core Architecture & Module Map](#2-core-architecture--module-map)
3. [Component Loading & Data Formats](#3-component-loading--data-formats)
4. [The Ray Tracing Engine](#4-the-ray-tracing-engine)
5. [Physical Principles](#5-physical-principles)
6. [Simulation Pipeline](#6-simulation-pipeline)
7. [Visualization System](#7-visualization-system)
8. [Analysis & Optimization](#8-analysis--optimization)
9. [Backend Abstraction (NumPy / PyTorch)](#9-backend-abstraction-numpy--pytorch)
10. [Adding a Web-Based Frontend / API](#10-adding-a-web-based-frontend--api)
11. [Integrating a 50 mm Grid for Component Placement](#11-integrating-a-50-mm-grid-for-component-placement)
12. [Adding a Thorlabs-Style Component Library](#12-adding-a-thorlabs-style-component-library)
13. [Putting It All Together – Roadmap](#13-putting-it-all-together--roadmap)

---

## 1. High-Level Overview

Optiland is an open-source **sequential ray tracing** library. Optical systems
are modelled as an ordered chain of **surfaces** separated by **materials**.
Light is represented as collections of **rays** that propagate surface by
surface from the object plane to the image plane.

```
Object → Surface₁ → Surface₂ → … → Surfaceₙ → Image
         ↑ material  ↑ material        ↑ material
```

The central class is **`Optic`** (`optiland/optic/optic.py`). Everything else
(analysis, optimization, visualization) operates *on* an `Optic` instance.

---

## 2. Core Architecture & Module Map

```
optiland/
├── optic/              # Optic class – the top-level container
│   ├── optic.py        #   Main class: surfaces, fields, wavelengths, aperture
│   └── optic_updater.py#   Mutation helpers (set_radius, set_thickness, …)
│
├── surfaces/           # Surface definitions
│   ├── standard_surface.py   # `Surface` class (geometry + material + interaction)
│   ├── surface_group.py      # Ordered list of surfaces; the `trace()` loop lives here
│   ├── object_surface.py     # Special surface for the object plane
│   ├── image_surface.py      # Special surface for the image plane
│   ├── converters.py         # Dict ↔ Surface conversion
│   └── factories/            # Factory pattern for creating surfaces from parameters
│       ├── surface_factory.py
│       ├── geometry_factory.py
│       ├── material_factory.py
│       ├── coating_factory.py
│       ├── interaction_model_factory.py
│       └── coordinate_system_factory.py
│
├── geometries/         # Surface shape definitions (sag, intersection, normals)
│   ├── base.py         #   `BaseGeometry` ABC
│   ├── standard.py     #   Sphere / conic (z = r²/R(1+√(1-(1+k)r²/R²)))
│   ├── even_asphere.py #   Even asphere polynomial
│   ├── odd_asphere.py  #   Odd asphere polynomial
│   ├── biconic.py      #   Biconic
│   ├── toroidal.py     #   Toroidal
│   ├── chebyshev.py    #   Chebyshev polynomial
│   ├── zernike.py      #   Zernike polynomial
│   ├── grid_sag.py     #   Gridded sag data
│   ├── plane.py        #   Flat / plane surface
│   ├── plane_grating.py#   Plane diffraction grating
│   ├── standard_grating.py # Curved diffraction grating
│   ├── polynomial.py   #   General polynomial
│   ├── nurbs/          #   NURBS surfaces
│   └── forbes/         #   Forbes Q-type polynomials
│
├── materials/          # Optical materials & refractive index models
│   ├── base.py         #   `BaseMaterial` ABC
│   ├── material.py     #   `Material` – lookup from refractiveindex.info DB
│   ├── material_file.py#   `MaterialFile` – YAML dispersion formula parser
│   ├── abbe.py         #   `AbbeMaterial` – nd/Vd parametric material
│   ├── ideal.py        #   `IdealMaterial` – constant index
│   └── material_utils.py
│
├── database/           # Material database (refractiveindex.info mirror)
│   ├── catalog_nk.csv  #   3200+ materials index (name, reference, wavelength range)
│   ├── data-nk/        #   YAML files with dispersion coefficients/tabulated data
│   │   ├── glass/      #     Schott, Ohara, Hoya, CDGM, Hikari, … (14 manufacturers)
│   │   ├── main/       #     Metals, crystals, gases, …
│   │   ├── organic/    #     Organic materials
│   │   └── other/      #     Miscellaneous
│   └── catalog-nk.yml  #   Master YAML catalogue
│
├── rays/               # Ray representations
│   ├── base.py         #   `BaseRays` ABC
│   ├── real_rays.py    #   `RealRays` – (x,y,z) + direction cosines (L,M,N) + OPD
│   ├── paraxial_rays.py#   `ParaxialRays` – height y + slope u (1D)
│   ├── polarized_rays.py#  `PolarizedRays` – extends RealRays with Jones vectors
│   ├── ray_generator.py#   Generates initial rays from field/pupil coordinates
│   ├── polarization_state.py
│   └── ray_aiming/     #   Strategies to aim rays at the entrance pupil
│       └── registry.py #     "paraxial", "iterative", "robust"
│
├── raytrace/           # Ray tracer orchestration
│   ├── real_ray_tracer.py    # `RealRayTracer` – full 3D sequential trace
│   └── paraxial_ray_tracer.py# `ParaxialRayTracer` – first-order (linear) trace
│
├── interactions/       # Surface interaction models
│   ├── base.py         #   `BaseInteractionModel` ABC
│   ├── refractive_reflective_model.py  # Snell's law refraction & reflection
│   ├── diffractive_model.py            # Grating diffraction
│   ├── phase_interaction_model.py      # Phase-modifying elements
│   └── thin_lens_interaction_model.py  # Ideal thin-lens model
│
├── propagation/        # Medium propagation models
│   ├── base.py         #   `BasePropagationModel` ABC
│   ├── homogeneous.py  #   Straight-line propagation + Beer-Lambert absorption
│   └── grin.py         #   Gradient-index (GRIN) propagation
│
├── analysis/           # Optical analysis tools
│   ├── spot_diagram.py, ray_fan.py, field_curvature.py, distortion.py, …
│   ├── encircled_energy.py, intensity.py, irradiance.py
│   ├── jones_pupil.py, pupil_aberration.py
│   ├── through_focus.py, through_focus_mtf.py, through_focus_spot_diagram.py
│   ├── y_ybar.py, rms_vs_field.py, angle_vs_height.py
│   └── image_simulation/  # Full image simulation
│
├── mtf/                # Modulation Transfer Function
├── psf/                # Point Spread Function (FFT-based, Huygens)
├── wavefront/          # Wavefront analysis (OPD maps, Zernike decomposition)
├── zernike/            # Zernike polynomial utilities
├── aberrations.py      # Seidel aberration calculations
│
├── optimization/       # Lens optimization
│   ├── problem.py      #   `OptimizationProblem` – merit function definition
│   ├── operand/        #   Merit function operands (RMS spot size, EFL, …)
│   ├── variable/       #   Design variables (radius, thickness, conic, …)
│   ├── optimizer/      #   Optimizer backends
│   │   ├── scipy/      #     SciPy (least_squares, minimize, …)
│   │   └── torch/      #     PyTorch (gradient-based, differentiable)
│   └── scaling/        #   Variable scaling strategies
│
├── visualization/      # Plotting & rendering
│   ├── base.py         #   `BaseViewer` ABC
│   ├── system/         #   Optical system drawing
│   │   ├── optic_viewer.py     # 2D (Matplotlib)
│   │   ├── optic_viewer_3d.py  # 3D (VTK)
│   │   ├── rays.py             # Ray path rendering
│   │   ├── lens.py, mirror.py  # Element drawing
│   │   └── system.py           # Composite system drawing
│   ├── analysis/       #   Analysis plot viewers (surface sag, …)
│   ├── info/           #   Tabular lens info display
│   ├── themes.py       #   Plot theming
│   └── palettes.py     #   Colour palettes
│
├── fileio/             # File I/O
│   ├── optiland_handler.py  # Native JSON save/load
│   ├── zemax_handler.py     # Zemax .zmx file import
│   └── converters.py        # Data model conversion
│
├── backend/            # NumPy / PyTorch abstraction layer
│   ├── numpy_backend.py
│   ├── torch_backend.py
│   ├── utils.py
│   └── linalg/         # Linear algebra wrappers
│
├── fields/             # Field of view definitions
│   ├── field.py        #   `Field` – (x, y) + vignetting factors
│   ├── field_group.py  #   `FieldGroup` – collection of fields
│   └── field_types/    #   AngleField, ObjectHeightField, ImageHeightField
│
├── physical_apertures/ # Aperture shapes for clipping/vignetting
│   ├── radial.py, rectangular.py, elliptical.py, polygon.py, offset_radial.py
│
├── coatings.py         # Thin-film coatings (Fresnel, custom, polarized)
├── aperture.py         # System aperture (EPD, F/#, NA)
├── wavelength.py       # Wavelength definitions
├── coordinate_system.py# 3D coordinate transforms (position + rotation)
├── paraxial.py         # First-order optics (EFL, BFL, magnification, …)
├── distribution.py     # Pupil/field point distributions (hexapolar, ring, …)
├── jones.py            # Jones matrix calculus
├── scatter.py          # BSDF / scattering models
├── pickup.py           # Cross-surface parameter linking
├── solves.py           # Automatic constraint solving (marginal ray height, …)
├── tolerancing/        # Tolerance analysis
├── multiconfig/        # Multi-configuration systems (zoom lenses, …)
├── environment/        # Environmental conditions (temperature, pressure)
├── phase/              # Phase elements
├── apodization/        # Pupil apodization (Gaussian, uniform, …)
├── ml/                 # Machine learning utilities
└── samples/            # Pre-defined optical systems
    ├── simple.py       #   Edmund 49-847, singlets, doublets
    ├── objectives.py   #   Camera objectives (Cooke triplet, …)
    ├── telescopes.py   #   Telescope designs
    ├── microscopes.py  #   Microscope objectives
    ├── eyepieces.py    #   Eyepiece designs
    ├── infrared.py     #   IR systems
    └── lithography.py  #   Lithography lenses
```

---

## 3. Component Loading & Data Formats

### 3.1 Material Database

Materials are loaded from a **mirror of the refractiveindex.info database**
bundled inside `optiland/database/`.

| Item | Path | Format |
|------|------|--------|
| Master index | `database/catalog_nk.csv` | CSV (3200+ rows) |
| Individual materials | `database/data-nk/{group}/{manufacturer}/{glass}.yml` | YAML |
| Glass manufacturers | 14 directories under `data-nk/glass/`: Schott, Ohara, Hoya, CDGM, Hikari, … | — |

**Material YAML format** (from refractiveindex.info):
```yaml
REFERENCES: "M. N. Polyanskiy, ..."
DATA:
  - type: formula 2          # Sellmeier-2 formula
    wavelength_range: 0.36 5.0
    coefficients: 1.03961212 0.00600069867 0.231792344 ...
  - type: tabulated nk       # or tabulated data
    data: |
      0.3 1.5469 0.0000
      0.35 1.5393 0.0000
      ...
```

The `Material` class (`materials/material.py`) performs a **fuzzy search**
(Levenshtein distance) on `catalog_nk.csv` to find the best match for a given
material name (e.g., `"N-BK7"`, `"N-SF11"`). It then reads the corresponding
YAML file and selects the appropriate dispersion formula.

**Supported dispersion formulas** (9 formulas + tabulated data):

| Formula | Description |
|---------|-------------|
| Formula 1 | Sellmeier (original form) |
| Formula 2 | Sellmeier-2 (modified form) |
| Formula 3 | Polynomial |
| Formula 4 | RefractiveIndex.INFO formula |
| Formula 5 | Cauchy |
| Formula 6 | Gases |
| Formula 7 | Herzberger |
| Formula 8 | Retro |
| Formula 9 | Exotic |
| Tabulated n / nk | Interpolated from wavelength–n(–k) tables |

### 3.2 Optical Systems (File I/O)

**Native format**: JSON (via `to_dict()` / `from_dict()`)

```python
# Save
from optiland.fileio import save_optiland_file, load_optiland_file
save_optiland_file(optic, "my_system.json")

# Load
optic = load_optiland_file("my_system.json")
```

The JSON dictionary contains:
```json
{
  "version": 1.0,
  "aperture": { "type": "EPD", "value": 25.4 },
  "fields": { "field_definition": {...}, "fields": [...] },
  "wavelengths": { "wavelengths": [...], "polarization": "ignore" },
  "surface_group": {
    "surfaces": [
      {
        "type": "standard",
        "geometry": { "type": "StandardGeometry", "radius": 19.93, "conic": 0.0, "cs": {...} },
        "material_post": { "type": "Material", "name": "N-SF11" },
        "interaction_model": { "type": "refractive_reflective", "is_reflective": false },
        "is_stop": true
      },
      ...
    ]
  }
}
```

**Zemax import**: `.zmx` files (both local and URL) are parsed by
`zemax_handler.py`, which extracts surfaces, materials, aperture, fields, and
wavelengths, then converts them to an `Optic` instance via `ZemaxToOpticConverter`.

### 3.3 Surface Construction

Surfaces are created through a **factory pattern**
(`surfaces/factories/surface_factory.py`). When you call
`optic.add_surface(...)`, the factory:

1. **GeometryFactory** → creates the appropriate geometry (Standard, EvenAsphere, …)
2. **MaterialFactory** → resolves the material string to a `Material` / `IdealMaterial`
3. **CoordinateSystemFactory** → positions the surface in global space
4. **InteractionModelFactory** → selects refraction, reflection, diffraction, or thin-lens
5. **CoatingFactory** → applies thin-film coatings if specified

---

## 4. The Ray Tracing Engine

### 4.1 Sequential Trace Loop

The core trace loop is in `SurfaceGroup.trace()`:

```python
def trace(self, rays, skip=0):
    self.reset()
    for surface in self.surfaces[skip:]:
        surface.trace(rays)    # each surface: propagate → interact
    return rays
```

Each `Surface.trace()` call performs:

1. **Propagation** – move rays from their current position to the intersection
   with the surface geometry
2. **Coordinate transform** – transform ray positions into the local coordinate
   system of the surface
3. **Intersection finding** – solve for the ray–surface intersection distance
   using the geometry's `distance()` method
4. **Update position** – advance rays to the intersection point
5. **Interaction** – apply Snell's law (refraction), reflection law, or
   diffraction equation
6. **Coating / BSDF** – apply intensity modifications from coatings or scatter
7. **Physical aperture clipping** – set intensity to 0 for vignetted rays
8. **Record data** – store intersection coordinates, direction cosines, OPD,
   and intensity

### 4.2 Ray Representation

**Real Rays** (`RealRays`):
- Position: `(x, y, z)` – 3D Cartesian coordinates [mm]
- Direction: `(L, M, N)` – direction cosines, normalised: L² + M² + N² = 1
- Intensity: `i` – ray intensity (0 = vignetted)
- Wavelength: `w` – wavelength [µm]
- OPD: `opd` – accumulated optical path difference

**Paraxial Rays** (`ParaxialRays`):
- Height: `y` – ray height at surface
- Slope: `u` – paraxial ray angle (y-u trace)
- Used for first-order calculations (EFL, BFL, pupil positions, …)

### 4.3 Ray Generation

`RayGenerator` creates initial rays given normalised field coordinates (Hx, Hy)
and pupil coordinates (Px, Py). It uses a **ray aiming** strategy to map pupil
coordinates to actual ray starting positions:

- **Paraxial**: Uses first-order optics to estimate starting position (fast)
- **Iterative**: Newton-Raphson iteration on real rays (more accurate)
- **Robust**: A more robust iterative method for difficult systems

### 4.4 Intersection Finding (Geometry)

For a **standard conic surface** (the most common case):

```
z(r) = r² / (R · (1 + √(1 - (1+k) · r²/R²)))
```

The intersection is found by solving a **quadratic equation** for the
propagation distance `t`:

```
a·t² + b·t + c = 0
```

where:
- `a = k·N² + L² + M² + N²`
- `b = 2·(k·N·z + L·x + M·y - N·R + N·z)`
- `c = k·z² - 2·R·z + x² + y² + z²`

The solution closest to z = 0 (the surface vertex) is chosen.

For **aspheric and freeform surfaces**, a **Newton-Raphson iteration**
(`geometries/newton_raphson.py`) is used after an initial estimate from the
base sphere.

---

## 5. Physical Principles

### 5.1 Snell's Law (Refraction)

Implemented in `RealRays.refract()` using the **vector form of Snell's law**:

```
t⃗ = (n₁/n₂) · d⃗ + n̂ · (√(1 - (n₁/n₂)²·(1 - (d⃗·n̂)²)) - (n₁/n₂)·(d⃗·n̂))
```

where:
- `d⃗` = incident direction (L, M, N)
- `n̂` = surface normal (nx, ny, nz)
- `n₁, n₂` = refractive indices before and after the surface
- `t⃗` = refracted direction

This handles **total internal reflection** gracefully (NaN from the square
root → invalid rays).

### 5.2 Reflection

Implemented in `RealRays.reflect()`:

```
r⃗ = d⃗ - 2·(d⃗·n̂)·n̂
```

### 5.3 Diffraction (Gratings)

Implemented in `DiffractiveInteractionModel` and `RealRays.gratingdiffract()`:

Uses the **3D grating equation**:

```
n₂·sin(θ_m) = n₁·sin(θ_i) + m·λ/d
```

generalised to full vector form for arbitrary grating orientations.

### 5.4 Paraxial Optics

The paraxial trace (`ParaxialRayTracer`) uses the **matrix method** / y-u trace:

- **Refraction**: `u' = (n₁·u - y·φ) / n₂`  where `φ = (n₂-n₁)/R` (surface power)
- **Reflection**: `u' = -u - 2·y/R`
- **Transfer**: `y' = y + t·u`

### 5.5 Material Absorption (Beer-Lambert Law)

In `HomogeneousPropagation.propagate()`:

```
I = I₀ · exp(-α · t)
α = 4π·k/λ
```

where `k` is the extinction coefficient and `t` is the propagation distance.

### 5.6 Optical Path Difference (OPD)

OPD is accumulated at each surface. The optical path length between two
surfaces is `n · t` where `n` is the refractive index and `t` is the
geometric path length. This is used for wavefront analysis and diffraction-
based PSF calculations.

### 5.7 Polarization (Jones Calculus)

When polarization is not "ignore", `PolarizedRays` carry Jones vectors.
Coatings can be defined with Jones matrices that modify the polarization
state at each surface.

---

## 6. Simulation Pipeline

A typical simulation follows this pipeline:

```
┌─────────────────┐
│  1. Define Optic │  Surfaces, materials, aperture, fields, wavelengths
└────────┬────────┘
         │
┌────────▼────────┐
│ 2. Update Parax │  Compute first-order properties (EPD, EFL, pupils)
└────────┬────────┘
         │
┌────────▼────────┐
│ 3. Generate Rays│  RayGenerator → RealRays from (Hx,Hy,Px,Py)
└────────┬────────┘
         │
┌────────▼────────┐
│  4. Trace Rays  │  SurfaceGroup.trace() → propagate + interact per surface
└────────┬────────┘
         │
┌────────▼────────┐
│ 5. Collect Data │  (x,y,z), (L,M,N), OPD, intensity at each surface
└────────┬────────┘
         │
┌────────▼────────┐
│  6. Analyse     │  SpotDiagram, MTF, PSF, Wavefront, Aberrations, …
└────────┬────────┘
         │
┌────────▼────────┐
│  7. Optimise    │  Define variables + operands → optimizer loop
└────────┬────────┘
         │
┌────────▼────────┐
│  8. Visualise   │  2D (Matplotlib) or 3D (VTK) system + ray drawings
└─────────────────┘
```

**Example**:
```python
from optiland import optic
from optiland.analysis import SpotDiagram

lens = optic.Optic()
lens.add_surface(index=0, thickness=float('inf'))
lens.add_surface(index=1, thickness=7, radius=20.0, is_stop=True, material='N-SF11')
lens.add_surface(index=2, thickness=23.0)
lens.add_surface(index=3)
lens.set_aperture(aperture_type='EPD', value=20)
lens.set_field_type(field_type='angle')
lens.add_field(y=0)
lens.add_field(y=10)
lens.add_wavelength(value=0.587, is_primary=True)

# Trace & analyse
spot = SpotDiagram(lens)
spot.plot()

# Draw system
lens.draw()
```

---

## 7. Visualization System

### 7.1 2D Visualization (Matplotlib)

- **Class**: `OpticViewer` (`visualization/system/optic_viewer.py`)
- **Library**: Matplotlib
- **Features**:
  - Lens cross-sections (filled polygons)
  - Mirror surfaces
  - Ray paths (coloured per wavelength, grouped per field)
  - Physical aperture overlays
  - Interactive tooltips (via `InteractionManager`)
  - Multiple projections: YZ, XZ, XY
  - Theming support (light/dark modes)

**Architecture**:
```
OpticViewer
├── Rays2D          → traces rays and plots ray paths
├── OpticalSystem   → draws lens/mirror elements
│   ├── Lens        → filled polygon between two surfaces
│   ├── Mirror      → reflective surface drawing
│   └── Surface     → generic surface outline
└── InteractionManager → click/hover tooltips
```

### 7.2 3D Visualization (VTK)

- **Class**: `OpticViewer3D` (`visualization/system/optic_viewer_3d.py`)
- **Library**: VTK (Visualization Toolkit)
- **Features**:
  - Full 3D rendering of lenses, mirrors, ray bundles
  - Interactive rotation, zoom, pan (trackball camera)
  - Dark mode support

### 7.3 Analysis Plots

Each analysis module (SpotDiagram, RayFan, MTF, PSF, …) has its own
`plot()` method that creates Matplotlib figures. These are standalone and
not part of the system viewer.

---

## 8. Analysis & Optimization

### 8.1 Analysis Tools

| Analysis | Module | Description |
|----------|--------|-------------|
| Spot Diagram | `analysis/spot_diagram.py` | Ray intersection pattern on image plane |
| Ray Fan | `analysis/ray_fan.py` | Transverse/longitudinal ray aberrations vs pupil |
| Field Curvature | `analysis/field_curvature.py` | Tangential/sagittal focus vs field |
| Distortion | `analysis/distortion.py` | Percent distortion vs field |
| Encircled Energy | `analysis/encircled_energy.py` | Fraction of energy within radius |
| MTF | `mtf/` | Modulation Transfer Function (geometric + diffraction) |
| PSF | `psf/` | Point Spread Function (FFT-based + Huygens) |
| Wavefront | `wavefront/` | OPD maps, Zernike decomposition |
| Seidel Aberrations | `aberrations.py` | 3rd-order aberration coefficients |
| Jones Pupil | `analysis/jones_pupil.py` | Polarization pupil maps |
| Image Simulation | `analysis/image_simulation/` | Full image convolution |

### 8.2 Optimization

The optimization system follows a **merit function** paradigm:

1. **Variables**: Parameters to optimise (radius, thickness, conic constant, …)
2. **Operands**: Target metrics (spot size, EFL, distortion, …) with weights
3. **Optimizer**: Minimises the weighted sum of squared residuals

**Backends**:
- **SciPy** (`optimization/optimizer/scipy/`): `least_squares`, `minimize`
- **PyTorch** (`optimization/optimizer/torch/`): Gradient-based (differentiable ray tracing)

```python
from optiland.optimization import OptimizationProblem

problem = OptimizationProblem()
problem.add_variable(optic, 'radius', surface_number=1)
problem.add_variable(optic, 'thickness', surface_number=1)
problem.add_operand('f_number', target=2.8, weight=1.0)
problem.add_operand('rms_spot_size', target=0, weight=10.0)

from optiland.optimization.optimizer.scipy import LeastSquaresOptimizer
optimizer = LeastSquaresOptimizer(problem)
optimizer.optimize()
```

---

## 9. Backend Abstraction (NumPy / PyTorch)

All numerical operations use `import optiland.backend as be` instead of
directly importing NumPy or PyTorch. This provides:

- **NumPy backend** (default): `numpy_backend.py` – standard CPU computation
- **PyTorch backend**: `torch_backend.py` – GPU acceleration + automatic
  differentiation (for gradient-based optimisation)

The backend can be switched at runtime:
```python
import optiland.backend as be
be.set_backend('torch')  # or 'numpy'
```

All array operations (`be.array()`, `be.sqrt()`, `be.stack()`, …) are routed
to the active backend.

---

## 10. Adding a Web-Based Frontend / API

### 10.1 Architecture Recommendation

```
┌─────────────────────────────────────────────────┐
│                  Browser Client                  │
│  React / Vue / Svelte + Three.js / Plotly.js     │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ System View  │ │ Analysis     │ │ Component │ │
│  │ (3D Canvas)  │ │ Panels       │ │ Library   │ │
│  └──────┬──────┘ └──────┬───────┘ └─────┬─────┘ │
└─────────┼───────────────┼───────────────┼───────┘
          │   REST / WebSocket / gRPC     │
┌─────────▼───────────────▼───────────────▼───────┐
│              FastAPI / Flask Backend              │
│  ┌──────────────────────────────────────────┐    │
│  │            Optiland Core Engine           │    │
│  │  Optic ← SurfaceGroup ← RealRayTracer    │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 10.2 Backend API (FastAPI)

Create a REST API layer around the Optiland core:

```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from optiland import optic
from optiland.analysis import SpotDiagram

app = FastAPI(title="Optiland API")

# In-memory session store (use Redis for production)
sessions: dict[str, optic.Optic] = {}

class SurfaceSpec(BaseModel):
    index: int
    radius: float = float('inf')
    thickness: float = 0.0
    material: str = "air"
    is_stop: bool = False
    conic: float = 0.0

class SystemSpec(BaseModel):
    name: str
    aperture_type: str   # "EPD", "imageFNO", "objectNA"
    aperture_value: float
    field_type: str      # "angle", "object_height"
    fields: list[dict]   # [{"x": 0, "y": 0}, {"x": 0, "y": 10}]
    wavelengths: list[dict]  # [{"value": 0.587, "is_primary": true}]
    surfaces: list[SurfaceSpec]

@app.post("/api/systems")
def create_system(spec: SystemSpec):
    """Create a new optical system."""
    lens = optic.Optic(name=spec.name)
    for surf in spec.surfaces:
        lens.add_surface(**surf.dict())
    lens.set_aperture(spec.aperture_type, spec.aperture_value)
    lens.set_field_type(spec.field_type)
    for f in spec.fields:
        lens.add_field(**f)
    for w in spec.wavelengths:
        lens.add_wavelength(**w)

    session_id = spec.name  # or use uuid4()
    sessions[session_id] = lens
    return {"session_id": session_id, "num_surfaces": lens.surface_group.num_surfaces}

@app.get("/api/systems/{session_id}/trace")
def trace_rays(session_id: str, num_rays: int = 100):
    """Trace rays and return intersection data."""
    lens = sessions[session_id]
    rays = lens.trace(Hx=0, Hy=0, wavelength=lens.primary_wavelength, num_rays=num_rays)
    return {
        "x": rays.x.tolist(),
        "y": rays.y.tolist(),
        "z": rays.z.tolist(),
    }

@app.get("/api/systems/{session_id}/draw")
def get_system_drawing(session_id: str):
    """Return system drawing data as JSON for frontend rendering."""
    lens = sessions[session_id]
    # Extract surface profiles and ray paths for client-side rendering
    surfaces_data = []
    for i, surf in enumerate(lens.surface_group.surfaces):
        surfaces_data.append({
            "index": i,
            "z_position": float(surf.geometry.cs.z),
            "radius": float(surf.geometry.radius),
            "semi_aperture": float(surf.semi_aperture) if surf.semi_aperture else None,
            "is_stop": surf.is_stop,
        })
    return {"surfaces": surfaces_data}

@app.get("/api/systems/{session_id}/spot")
def get_spot_diagram(session_id: str):
    """Return spot diagram data."""
    lens = sessions[session_id]
    spot = SpotDiagram(lens)
    # Return spot data as JSON
    result = []
    for field_data in spot.data:
        for wavelength_data in field_data:
            result.append({
                "x": wavelength_data.x.tolist(),
                "y": wavelength_data.y.tolist(),
            })
    return {"spots": result}

@app.get("/api/materials/search")
def search_materials(query: str):
    """Search the material database."""
    import pandas as pd
    from optiland.materials.material import Material
    df = Material._load_dataframe()
    matches = df[df['name'].str.lower().str.contains(query.lower())]
    return matches[['name', 'reference', 'min_wavelength', 'max_wavelength']].head(20).to_dict(orient='records')
```

### 10.3 Frontend Rendering

For the **3D system view** in the browser, use **Three.js** or **Plotly.js**:

- **Surfaces**: Parametric meshes generated from geometry `sag(x, y)` data
- **Rays**: Line segments between surface intersection points
- **Lenses**: Filled volumes between front/back surface meshes

The API provides the raw geometry data; the frontend handles rendering. This
avoids sending Matplotlib images and enables **interactive** pan/zoom/rotate.

### 10.4 WebSocket for Live Updates

For real-time parameter adjustments (e.g., dragging a slider to change radius):

```python
from fastapi import WebSocket

@app.websocket("/ws/systems/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        lens = sessions[session_id]

        if data["action"] == "set_radius":
            lens.set_radius(data["value"], data["surface"])
        elif data["action"] == "set_thickness":
            lens.set_thickness(data["value"], data["surface"])

        # Re-trace and send updated data
        rays = lens.trace(Hx=0, Hy=0, wavelength=lens.primary_wavelength)
        await websocket.send_json({
            "rays": {"x": rays.x.tolist(), "y": rays.y.tolist(), "z": rays.z.tolist()},
        })
```

### 10.5 Technology Stack Recommendation

| Layer | Technology | Reason |
|-------|-----------|--------|
| Backend API | **FastAPI** | Async, automatic OpenAPI docs, Pydantic validation |
| Session store | **Redis** | Persistent sessions, multi-worker support |
| 3D Rendering | **Three.js** | Industry standard for WebGL, large ecosystem |
| 2D Charts | **Plotly.js** | Interactive, zoomable, export-friendly |
| UI Framework | **React** or **Svelte** | Component-based, large ecosystem |
| Realtime | **WebSocket** | Low-latency parameter updates |
| Serialisation | **JSON** | Optiland already uses `to_dict()` / `from_dict()` |

---

## 11. Integrating a 50 mm Grid for Component Placement

### 11.1 Concept

A **50 mm optical breadboard grid** (similar to Thorlabs optical tables) where
users can place optical components at discrete positions. This bridges the gap
between Optiland's sequential model and a **free-space / component-level** paradigm.

### 11.2 Grid Data Model

```python
# optiland/grid/optical_grid.py

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class GridPosition:
    """A position on the 50mm optical breadboard grid."""
    row: int        # Grid row index
    col: int        # Grid column index
    x_mm: float     # Absolute X position [mm]
    z_mm: float     # Absolute Z position [mm] (optical axis direction)

@dataclass
class PlacedComponent:
    """An optical component placed on the grid."""
    component_id: str          # Reference to component library entry
    grid_position: GridPosition
    rotation_deg: float = 0.0  # Rotation around Y-axis
    fine_x: float = 0.0       # Fine adjustment within cell [mm]
    fine_z: float = 0.0       # Fine adjustment within cell [mm]
    fine_y: float = 0.0       # Height adjustment [mm]

class OpticalGrid:
    """A 50mm optical breadboard grid for component placement."""

    PITCH = 50.0  # mm (standard Thorlabs breadboard pitch)

    def __init__(self, rows: int = 12, cols: int = 24):
        """Create a grid of rows×cols positions.

        Default: 600mm × 1200mm breadboard (12×24 @ 50mm pitch)
        """
        self.rows = rows
        self.cols = cols
        self.components: list[PlacedComponent] = []

    def grid_to_mm(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid coordinates to mm coordinates."""
        return (col * self.PITCH, row * self.PITCH)

    def place_component(self, component_id: str, row: int, col: int,
                        rotation_deg: float = 0.0, **fine_adjustments):
        """Place a component on the grid."""
        x_mm, z_mm = self.grid_to_mm(row, col)
        pos = GridPosition(row, col, x_mm, z_mm)
        placed = PlacedComponent(
            component_id=component_id,
            grid_position=pos,
            rotation_deg=rotation_deg,
            **fine_adjustments,
        )
        self.components.append(placed)
        return placed

    def to_optic(self, source_config: dict) -> "Optic":
        """Convert the grid layout to a sequential Optic.

        This method sorts components by Z-position and creates a sequential
        optical system. Each component contributes one or more surfaces.
        """
        from optiland import optic
        lens = optic.Optic()
        # Sort by z-position (optical axis direction)
        sorted_components = sorted(self.components, key=lambda c: c.grid_position.z_mm)
        # ... convert each component to surfaces and add to lens
        return lens

    def to_dict(self) -> dict:
        """Serialise the grid to JSON."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "OpticalGrid":
        """Deserialise from JSON."""
        ...
```

### 11.3 Frontend Grid View

The web frontend should render:

1. **Grid background**: 50mm pitch squares with hole marks (SVG or Canvas)
2. **Component icons**: Drag-and-drop from a sidebar library onto the grid
3. **Snap-to-grid**: Components snap to the nearest grid position
4. **Fine adjustment**: Sub-millimetre offset within each cell
5. **Rotation**: Components can be rotated (0°, 45°, 90°, …)
6. **Ray path overlay**: Show traced rays through the placed components

```
┌─────────────────────────────────────────────────────┐
│  Component Library  │       50mm Grid View          │
│  ┌───────────┐      │  ┌──┬──┬──┬──┬──┬──┬──┬──┐   │
│  │ ⊕ Lens    │      │  │  │  │  │  │  │  │  │  │   │
│  │ ⊕ Mirror  │      │  ├──┼──┼──┼──┼──┼──┼──┼──┤   │
│  │ ⊕ Prism   │      │  │  │  │▓▓│  │  │▓▓│  │  │   │
│  │ ⊕ Filter  │      │  ├──┼──┼──┼──┼──┼──┼──┼──┤   │
│  │ ⊕ Source  │      │  │  │  │  │  │  │  │  │  │   │
│  │ ⊕ Detector│      │  ├──┼──┼──┼──┼──┼──┼──┼──┤   │
│  └───────────┘      │  │  │  │  │  │  │  │  │  │   │
│                     │  └──┴──┴──┴──┴──┴──┴──┴──┘   │
│  Properties Panel   │  ▓▓ = placed component        │
└─────────────────────┴───────────────────────────────┘
```

### 11.4 Integration with Optiland

The grid-based layout would need a **conversion layer** to translate from
the "component on a breadboard" paradigm to Optiland's sequential surface model:

1. **Component → Surfaces**: Each catalogue component maps to one or more
   surfaces (e.g., a lens = 2 surfaces + material)
2. **Position → Thickness**: The distance between consecutive components on
   the grid becomes the air gap thickness
3. **Rotation → Decentre/Tilt**: Component rotation maps to coordinate system
   rotations on the surfaces

---

## 12. Adding a Thorlabs-Style Component Library

### 12.1 Component Catalogue Structure

```
optiland/
└── components/                 # NEW module
    ├── __init__.py
    ├── base.py                 # BaseComponent ABC
    ├── catalog.py              # ComponentCatalog – search & retrieval
    ├── registry.py             # Component type registry
    ├── lenses/
    │   ├── plano_convex.py     # e.g., Thorlabs LA1 series
    │   ├── plano_concave.py    # e.g., Thorlabs LC1 series
    │   ├── biconvex.py         # e.g., Thorlabs LB1 series
    │   ├── achromatic.py       # e.g., Thorlabs AC254 series
    │   └── aspheric.py         # e.g., Thorlabs AL series
    ├── mirrors/
    │   ├── flat.py             # e.g., Thorlabs PF series
    │   ├── concave.py          # Thorlabs CM series
    │   └── dichroic.py         # Thorlabs DMSP series
    ├── prisms/
    │   ├── right_angle.py
    │   └── beamsplitter.py
    ├── filters/
    │   ├── bandpass.py
    │   ├── longpass.py
    │   └── neutral_density.py
    ├── sources/
    │   ├── point_source.py
    │   └── collimated.py
    ├── detectors/
    │   ├── sensor.py
    │   └── screen.py
    └── data/
        ├── thorlabs_lenses.json    # Catalogue data
        ├── thorlabs_mirrors.json
        ├── edmund_lenses.json
        └── custom_components.json
```

### 12.2 Component Base Class

```python
# optiland/components/base.py

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional

@dataclass
class ComponentSpec:
    """Specification for an optical component."""
    part_number: str          # e.g., "LA1131-A"
    manufacturer: str         # e.g., "Thorlabs"
    description: str          # e.g., "N-BK7 Plano-Convex Lens, f=50mm, Ø25.4mm"
    category: str             # e.g., "plano_convex_lens"
    diameter_mm: float        # e.g., 25.4
    thickness_mm: float       # e.g., 8.6
    clear_aperture_mm: float  # e.g., 22.9
    coating: str              # e.g., "A" (350-700nm AR), "B" (650-1050nm AR)
    price_usd: Optional[float] = None
    datasheet_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: list[str] = field(default_factory=list)

class BaseComponent(ABC):
    """Abstract base class for optical components."""

    def __init__(self, spec: ComponentSpec):
        self.spec = spec

    @abstractmethod
    def to_surfaces(self) -> list[dict]:
        """Convert this component into Optiland surface definitions.

        Returns a list of dicts that can be passed to optic.add_surface().
        """
        pass

    @abstractmethod
    def bounding_box(self) -> tuple[float, float, float]:
        """Return (width, height, length) in mm for grid placement."""
        pass

    def add_to_optic(self, optic, index: int = None):
        """Add this component's surfaces to an Optic instance."""
        surfaces = self.to_surfaces()
        for surf_kwargs in surfaces:
            if index is not None:
                surf_kwargs['index'] = index
                index += 1
            optic.add_surface(**surf_kwargs)
```

### 12.3 Example: Plano-Convex Lens

```python
# optiland/components/lenses/plano_convex.py

from optiland.components.base import BaseComponent, ComponentSpec

class PlanoConvexLens(BaseComponent):
    """A plano-convex lens component."""

    def __init__(self, spec: ComponentSpec, radius: float, material: str = "N-BK7"):
        super().__init__(spec)
        self.radius = radius
        self.material = material

    def to_surfaces(self) -> list[dict]:
        return [
            {
                "radius": self.radius,
                "thickness": self.spec.thickness_mm,
                "material": self.material,
                "aperture": self.spec.clear_aperture_mm,
            },
            {
                "radius": float('inf'),  # plano side
                "thickness": 0,
                "material": "air",
            },
        ]

    def bounding_box(self) -> tuple[float, float, float]:
        d = self.spec.diameter_mm
        return (d, d, self.spec.thickness_mm)
```

### 12.4 Component Catalogue (JSON Format)

```json
// optiland/components/data/thorlabs_lenses.json
{
  "catalog": "Thorlabs",
  "category": "Plano-Convex Lenses",
  "components": [
    {
      "part_number": "LA1131-A",
      "description": "N-BK7 Plano-Convex Lens, f=50mm, Ø25.4mm, ARC: 350-700nm",
      "category": "plano_convex_lens",
      "diameter_mm": 25.4,
      "thickness_mm": 8.6,
      "clear_aperture_mm": 22.9,
      "back_focal_length_mm": 46.0,
      "effective_focal_length_mm": 50.0,
      "radius_mm": 25.8,
      "material": "N-BK7",
      "coating": "A",
      "coating_range_nm": [350, 700],
      "price_usd": 28.56,
      "datasheet_url": "https://www.thorlabs.com/thorproduct.cfm?partnumber=LA1131-A",
      "tags": ["visible", "bk7", "1inch", "f50"]
    },
    {
      "part_number": "LA1134-A",
      "description": "N-BK7 Plano-Convex Lens, f=60mm, Ø25.4mm, ARC: 350-700nm",
      "category": "plano_convex_lens",
      "diameter_mm": 25.4,
      "thickness_mm": 7.5,
      "clear_aperture_mm": 22.9,
      "back_focal_length_mm": 56.7,
      "effective_focal_length_mm": 60.0,
      "radius_mm": 30.9,
      "material": "N-BK7",
      "coating": "A",
      "coating_range_nm": [350, 700],
      "price_usd": 28.56,
      "tags": ["visible", "bk7", "1inch", "f60"]
    }
  ]
}
```

### 12.5 Catalogue Search API

```python
# optiland/components/catalog.py

import json
from pathlib import Path

class ComponentCatalog:
    """Searchable catalogue of optical components."""

    def __init__(self):
        self._components = []
        self._load_builtin_catalogs()

    def _load_builtin_catalogs(self):
        data_dir = Path(__file__).parent / "data"
        for json_file in data_dir.glob("*.json"):
            with open(json_file) as f:
                catalog = json.load(f)
            for comp in catalog.get("components", []):
                comp["_source_file"] = json_file.name
                self._components.append(comp)

    def search(self, query: str = "", category: str = None,
               diameter: float = None, focal_length: float = None,
               manufacturer: str = None, max_results: int = 50) -> list[dict]:
        """Search the component catalogue."""
        results = self._components

        if query:
            q = query.lower()
            results = [c for c in results if q in c.get("description", "").lower()
                       or q in c.get("part_number", "").lower()
                       or any(q in tag for tag in c.get("tags", []))]

        if category:
            results = [c for c in results if c.get("category") == category]

        if diameter:
            results = [c for c in results if abs(c.get("diameter_mm", 0) - diameter) < 1.0]

        if focal_length:
            results = [c for c in results
                       if abs(c.get("effective_focal_length_mm", 0) - focal_length) < 5.0]

        if manufacturer:
            results = [c for c in results
                       if c.get("_source_file", "").startswith(manufacturer.lower())]

        return results[:max_results]

    def get_by_part_number(self, part_number: str) -> dict | None:
        """Get a component by its exact part number."""
        for c in self._components:
            if c.get("part_number") == part_number:
                return c
        return None
```

---

## 13. Putting It All Together – Roadmap

### Phase 1: Web API Foundation
1. ✅ Wrap Optiland core in **FastAPI** endpoints
2. ✅ Add JSON serialisation endpoints (already supported via `to_dict()`)
3. Create session management (in-memory → Redis)
4. Add WebSocket endpoint for real-time updates

### Phase 2: Web Frontend
1. Set up **React** + **Three.js** application
2. Implement 3D system viewer (lens elements + ray paths)
3. Add 2D analysis panels (spot diagram, MTF, ray fan) via **Plotly.js**
4. Create parameter editing sidebar with live-update

### Phase 3: Component Library
1. Define the `BaseComponent` class and catalogue format (JSON)
2. Populate initial catalogue with common Thorlabs / Edmund lenses
3. Build a web-scrapable pipeline or manual entry process for component data
4. Implement catalogue search API endpoints

### Phase 4: Grid System
1. Build the `OpticalGrid` data model (50mm pitch)
2. Create grid-to-sequential converter
3. Implement drag-and-drop grid UI in the frontend
4. Add snap-to-grid, rotation, and fine-adjustment controls
5. Wire grid changes to the Optiland trace engine via WebSocket

### Phase 5: Integration & Polish
1. Save/load grid layouts (JSON)
2. Export to Zemax / Optiland native format
3. Component library browser with filtering, thumbnails, datasheets
4. Collaborative editing (multi-user via shared session)
5. Deploy as a Docker container with `uvicorn` + `nginx`

### Recommended Technology Stack

```
┌───────────────────────────────────────┐
│              Docker Container          │
│  ┌─────────────────────────────────┐  │
│  │  nginx (reverse proxy, static)  │  │
│  └───────────┬─────────────────────┘  │
│  ┌───────────▼─────────────────────┐  │
│  │  uvicorn + FastAPI              │  │
│  │  ┌────────────────────────────┐ │  │
│  │  │    Optiland Core Engine    │ │  │
│  │  │  + Component Library       │ │  │
│  │  │  + Grid System             │ │  │
│  │  └────────────────────────────┘ │  │
│  └─────────────────────────────────┘  │
│  ┌─────────────────────────────────┐  │
│  │  Redis (session store)          │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
```

---

## Appendix A: Key Classes Quick Reference

| Class | Module | Purpose |
|-------|--------|---------|
| `Optic` | `optiland.optic` | Top-level optical system container |
| `Surface` | `optiland.surfaces.standard_surface` | Single optical surface |
| `SurfaceGroup` | `optiland.surfaces.surface_group` | Ordered surface collection + trace loop |
| `RealRays` | `optiland.rays.real_rays` | 3D ray bundle (x,y,z,L,M,N) |
| `ParaxialRays` | `optiland.rays.paraxial_rays` | 1D paraxial ray (y, u) |
| `RayGenerator` | `optiland.rays.ray_generator` | Creates initial rays from (H,P) |
| `RealRayTracer` | `optiland.raytrace.real_ray_tracer` | Orchestrates full ray trace |
| `StandardGeometry` | `optiland.geometries.standard` | Sphere/conic surface shape |
| `Material` | `optiland.materials.material` | Dispersion-based optical material |
| `MaterialFile` | `optiland.materials.material_file` | YAML material file parser |
| `RefractiveReflectiveModel` | `optiland.interactions` | Snell's law / reflection |
| `HomogeneousPropagation` | `optiland.propagation` | Straight-line propagation |
| `OpticViewer` | `optiland.visualization` | 2D Matplotlib viewer |
| `OpticViewer3D` | `optiland.visualization` | 3D VTK viewer |
| `OptimizationProblem` | `optiland.optimization` | Merit function definition |
| `SpotDiagram` | `optiland.analysis` | Spot diagram analysis |

## Appendix B: Material Database Statistics

- **3200+ materials** indexed in `catalog_nk.csv`
- **14 glass manufacturers**: Schott, Ohara, Hoya, CDGM, Hikari, Corning, Sumita, Vitron, LZOS, NSG, Lightpath, Barberini, AMI, misc
- **Metals, crystals, gases, organics**: Au, Ag, Al, Si, Ge, ZnSe, CaF₂, BaF₂, sapphire, water, air, …
- **Dispersion models**: 9 analytic formulas + tabulated n/nk interpolation
- **Source**: refractiveindex.info database

## Appendix C: Sample Systems Included

| Module | Systems |
|--------|---------|
| `samples/simple.py` | Edmund 49-847, singlets (stop surf 1/2) |
| `samples/objectives.py` | Cooke triplet, Double Gauss, Petzval, … |
| `samples/telescopes.py` | Newtonian, Cassegrain, Schmidt, … |
| `samples/microscopes.py` | Microscope objectives |
| `samples/eyepieces.py` | Plössl, Erfle, Huygens, … |
| `samples/infrared.py` | IR optical systems |
| `samples/lithography.py` | Lithography projection lenses |
| `samples/miscellaneous.py` | Misc. optical systems |
