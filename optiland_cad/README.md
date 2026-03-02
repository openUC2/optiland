# optiland_cad – Optiland JSON → STEP Exporter

A Python tool that converts [Optiland](https://github.com/HarrisonKramer/optiland) optical system definitions (JSON) into CAD-ready **STEP files** (.step/.stp) using [CadQuery](https://cadquery.readthedocs.io/) (OCCT kernel).

The generated STEP files are intended for import into **Autodesk Inventor** (and any other CAD system supporting STEP AP214/AP203) without manual reorientation.

---

## Quick Start

### Installation

```bash
# Inside the optiland repository root
pip install cadquery           # CadQuery + OCCT
pip install -e .               # Optiland itself (if not already installed)
```

### CLI Usage

```bash
# Export a built-in sample system to STEP
python -m optiland_cad.cli --sample Edmund_49_847 --out step_output/

# Also save the intermediate JSON
python -m optiland_cad.cli --sample Edmund_49_847 --json-out edmund.json --out step_output/

# Export from an existing Optiland JSON file
python -m optiland_cad.cli my_system.json --out step_output/

# Convert a Zemax .zmx file to STEP
python -m optiland_cad.cli --zemax path/to/file.zmx --out step_output/

# Only generate JSON (no STEP)
python -m optiland_cad.cli --sample CookeTriplet --json-out triplet.json --json-only

# List available sample systems
python -m optiland_cad.cli --list-samples
```

### Programmatic API

```python
from optiland.samples import Edmund_49_847
from optiland_cad.json_utils import optic_to_surface_group_json, load_surface_group
from optiland_cad.exporter import export_surface_group

# 1. Create optical system and save JSON
optic = Edmund_49_847()
optic_to_surface_group_json(optic, "edmund_sg.json")

# 2. Load surface_group and export to STEP
sg = load_surface_group("edmund_sg.json")
files = export_surface_group(sg, output_dir="step_out/", file_prefix="edmund")
print(files)
# ['step_out/edmund_Lens_1.step']
```

---

## Coordinate Convention

| Property | Convention          |
| -------- | ------------------- |
| **Optical axis** | **+Z** (right-handed coordinate system) |
| **First surface** | XY plane at **z = 0** |
| **Centre** | **x = 0, y = 0** (on optical axis) |
| **Units** | **mm** |

This matches Optiland's internal convention and ensures that imported STEP solids align correctly in Inventor without reorientation.

```
         y
         │
         │     x
         │   ╱
         │ ╱
  (0,0,0)└──────────→ z   (optical axis)
```

---

## Component Categories

The exporter automatically classifies optical surfaces:

| Category | Criterion | STEP Output |
| -------- | --------- | ----------- |
| **Lens** | Non-reflective surfaces with material transitions (air → glass → air) | One STEP per element |
| **Mirror** | `interaction_model.is_reflective == true` | One STEP per reflective surface |
| **Objective** | Multi-element lens assembly (>1 lens) | Single fused STEP of all elements |

---

## Supported Geometry Types (v1)

| Type | Description |
| ---- | ----------- |
| `Plane` | Flat surface (z = 0 everywhere) |
| `StandardGeometry` | Sphere or conic section |
| `EvenAsphere` | Even-order polynomial asphere (rotationally symmetric) |

### Unsupported (fail with clear error)

`OddAsphere`, `BiconicGeometry`, `ToroidalGeometry`, `PolynomialGeometry`, `ChebyshevPolynomialGeometry`, `ZernikePolynomialGeometry`, `GridSagGeometry`, `NurbsGeometry`, `ForbesQ2dGeometry`, `PlaneGrating`, `StandardGratingGeometry`

---

## Architecture / Data Flow

```
┌──────────┐     ┌────────────┐     ┌────────────┐     ┌──────────┐
│ Optiland │────▶│ JSON utils │────▶│ CadQuery   │────▶│  .STEP   │
│  .JSON   │     │  (surface  │     │ (revolve   │     │  files   │
│          │     │   group)   │     │  profiles) │     │          │
└──────────┘     └────────────┘     └────────────┘     └──────────┘
                       │                    │
                       │                    ▼
                 ┌─────▼─────┐     ┌────────────────┐
                 │ Categorize│     │ Autodesk       │
                 │ (Lens /   │     │ Inventor       │
                 │  Mirror / │     │ (.IAM import)  │
                 │  Obj.)    │     └────────────────┘
                 └───────────┘
```

---

## Running Tests

```bash
# Run only the CAD exporter tests
uv run pytest tests/test_optiland_cad.py -v

# Run without cadquery (geometry + categorize + JSON tests still pass)
uv run pytest tests/test_optiland_cad.py -v -k "not StepExport"
```

---

## File Structure

```
optiland_cad/
├── __init__.py        # Package metadata
├── cli.py             # CLI entry point (optiland_to_step)
├── json_utils.py      # JSON generation / loading utilities
├── categorize.py      # Component classification (Lens/Mirror/Objective)
├── geometry.py        # Surface profile computation (sag functions)
└── exporter.py        # CadQuery solid builder + STEP writer
```

---

## Known Limitations

- **Semi-aperture required**: The STEP exporter needs `semi_aperture` values to determine physical extent. These are computed by `Optic.update()` (ray trace). When loading from JSON only, ensure the JSON contains `semi_aperture` fields or use default values.
- **Rotationally symmetric only**: v1 only supports surfaces of revolution. Non-rotationally-symmetric types (biconic, zernike, freeform …) will raise `UnsupportedGeometryError`.
- **No edge chamfer / bevel**: Lens edges are sharp cylindrical barrels.
- **No AR coating geometry**: Coatings are metadata only, not modelled in CAD.
- **Mirror substrate**: Mirrors get a default 3 mm flat-back substrate; real substrate shapes are not modelled.
- **Cemented elements**: Cemented doublets/triplets are fused into a single solid (no internal glue layer).
- **Decentered / tilted surfaces**: Coordinate system rotations (`rx`, `ry`, `rz`) are not yet applied to the CAD geometry.

---

## License

Same as the parent Optiland project (MIT).
