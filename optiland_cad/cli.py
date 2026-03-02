"""CLI entry point for optiland_cad.

Usage:
    # Export Optiland JSON to STEP
    optiland_to_step input.json --out out_dir/

    # Generate JSON from a sample system
    optiland_to_step --sample Edmund_49_847 --json-out sample.json

    # Convert Zemax .zmx to JSON then STEP
    optiland_to_step --zemax input.zmx --out out_dir/
"""

from __future__ import annotations

import argparse
import sys
import json
import os


def _get_sample_optic(name: str):
    """Instantiate a named sample from ``optiland.samples``."""
    from optiland import samples

    cls = getattr(samples, name, None)
    if cls is None:
        available = [
            n
            for n in dir(samples)
            if not n.startswith("_") and isinstance(getattr(samples, n), type)
        ]
        print(f"Unknown sample '{name}'. Available samples:", file=sys.stderr)
        for a in sorted(available):
            print(f"  {a}", file=sys.stderr)
        sys.exit(1)
    return cls()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="optiland_to_step",
        description="Convert Optiland optical systems (JSON) to STEP files.",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "input_json",
        nargs="?",
        default=None,
        help="Path to an Optiland JSON file.",
    )
    input_group.add_argument(
        "--sample",
        metavar="NAME",
        help="Name of a built-in sample system (e.g. Edmund_49_847).",
    )
    input_group.add_argument(
        "--zemax",
        metavar="PATH_OR_URL",
        help="Path or URL to a Zemax .zmx file.",
    )

    parser.add_argument(
        "--out",
        metavar="DIR",
        default="step_output",
        help="Output directory for STEP files (default: step_output/).",
    )
    parser.add_argument(
        "--json-out",
        metavar="PATH",
        default=None,
        help=(
            "Also save the Optiland JSON file. "
            "When used with --sample or --zemax this is required for "
            "reproducibility."
        ),
    )
    parser.add_argument(
        "--surface-group-only",
        action="store_true",
        help="When saving JSON, export only the surface_group portion.",
    )
    parser.add_argument(
        "--prefix",
        default="optiland",
        help="Filename prefix for STEP outputs (default: optiland).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only generate the JSON file, skip STEP export.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available sample optical systems and exit.",
    )

    args = parser.parse_args(argv)

    # List samples and exit
    if args.list_samples:
        from optiland import samples

        available = [
            n
            for n in dir(samples)
            if not n.startswith("_") and isinstance(getattr(samples, n), type)
        ]
        print("Available sample systems:")
        for a in sorted(available):
            print(f"  {a}")
        return

    # Resolve input
    surface_group: dict | None = None

    if args.sample:
        from optiland_cad.json_utils import (
            optic_to_json,
            optic_to_surface_group_json,
            _inject_semi_apertures,
        )
        optic = _get_sample_optic(args.sample)

        if args.json_out:
            os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                        exist_ok=True)
            if args.surface_group_only:
                optic_to_surface_group_json(optic, args.json_out)
            else:
                optic_to_json(optic, args.json_out)
            print(f"JSON saved to {args.json_out}")

        if args.json_only:
            return

        # Build surface_group dict in memory
        data = optic.to_dict()
        surface_group = data["surface_group"]
        _inject_semi_apertures(optic, surface_group)

    elif args.zemax:
        from optiland_cad.json_utils import zemax_to_json, load_surface_group, _inject_semi_apertures

        json_path = args.json_out or os.path.join(args.out, "_temp_zemax.json")
        optic = zemax_to_json(
            args.zemax,
            json_path,
            surface_group_only=args.surface_group_only,
        )
        if args.json_out:
            print(f"JSON saved to {args.json_out}")

        if args.json_only:
            return

        surface_group = load_surface_group(json_path)
        _inject_semi_apertures(optic, surface_group)

    elif args.input_json:
        from optiland_cad.json_utils import load_surface_group
        surface_group = load_surface_group(args.input_json)

        if args.json_only:
            print("Input is already JSON, nothing to do.")
            return
    else:
        parser.error(
            "Provide an input JSON file, --sample NAME, or --zemax PATH."
        )

    # Export to STEP
    from optiland_cad.exporter import export_surface_group
    from optiland_cad.geometry import UnsupportedGeometryError

    try:
        files = export_surface_group(
            surface_group,
            output_dir=args.out,
            file_prefix=args.prefix,
        )
    except UnsupportedGeometryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Exported {len(files)} STEP file(s) to '{args.out}':")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
