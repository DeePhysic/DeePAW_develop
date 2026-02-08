"""CLI entry point for Hirshfeld charge analysis.

Provides two subcommands:
    analyze     Run Hirshfeld charge analysis on a crystal structure.
    precompute  Build a radial lookup table for specified elements.

Usage examples::

    # Analyze a POSCAR file using a precomputed lookup table
    python -m deepaw.hirshfeld.cli analyze --poscar POSCAR --lookup-table table.npz

    # Analyze with on-the-fly radial density computation
    python -m deepaw.hirshfeld.cli analyze --poscar POSCAR --mode onthefly

    # Precompute a radial lookup table for Si and O
    python -m deepaw.hirshfeld.cli precompute --elements Si O --output table.npz

    # Precompute all elements of the periodic table (skips failures)
    python -m deepaw.hirshfeld.cli precompute --all --output all_elements.npz

When no subcommand is given, ``analyze`` is used by default.
"""

import argparse
import sys


def _run_analyze(args):
    """Execute the analyze subcommand."""
    from ase.io import read
    from deepaw.inference import InferenceEngine
    from deepaw.hirshfeld import HirshfeldAnalysis

    atoms = read(args.poscar)
    engine = InferenceEngine(use_dual_model=not args.no_dual_model)

    grid_shape = tuple(args.grid) if args.grid else None

    analyzer = HirshfeldAnalysis(
        engine=engine,
        mode=args.mode,
        lookup_path=args.lookup_table,
        encut=args.encut,
    )
    result = analyzer.analyze(atoms, grid_shape=grid_shape)
    print(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write("# Hirshfeld Charge Analysis\n")
            f.write(f"# Source: {args.poscar}\n")
            f.write(f"# Grid: {result.charges.shape}\n")
            f.write("# Index\tSpecies\tCharge\n")
            for i, (sym, q) in enumerate(zip(result.symbols, result.charges)):
                f.write(f"{i}\t{sym}\t{q:.6f}\n")
            f.write(f"# Total charge: {result.total_charge:.6f}\n")
            f.write(f"# Deformation integral: {result.deformation_integral:.6f}\n")
        print(f"\nResults written to: {args.output}")


def _run_precompute(args):
    """Execute the precompute subcommand."""
    from deepaw.inference import InferenceEngine
    from deepaw.hirshfeld.radial_lookup import build_radial_lookup

    engine = InferenceEngine(use_dual_model=not args.no_dual_model)

    # Resolve element list: "all" or explicit list
    elements = "all" if args.all else args.elements
    if elements is None:
        print("Error: specify --elements or --all", file=sys.stderr)
        sys.exit(1)

    lookup = build_radial_lookup(
        elements=elements,
        engine=engine,
        cell_size=args.cell_size,
        encut=args.encut,
        bin_width=args.bin_width,
        max_radius=args.max_radius,
        save_path=args.output,
        save_interval=args.save_interval,
    )
    print(f"\nSaved radial lookup table to: {args.output}")
    print(f"Supported elements ({len(lookup.elements)}): {lookup.elements}")
    if lookup.failed_elements:
        print(f"Failed elements ({len(lookup.failed_elements)}):")
        for sym, msg in lookup.failed_elements:
            print(f"  {sym}: {msg}")


def _add_analyze_args(parser):
    """Add arguments for the analyze subcommand to *parser*."""
    parser.add_argument(
        "--poscar", required=True, help="Path to crystal POSCAR/CIF file"
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=None,
        help="Grid dimensions (NX NY NZ). If omitted, determined automatically from --encut.",
    )
    parser.add_argument(
        "--encut",
        type=float,
        default=500.0,
        help="Plane-wave energy cutoff for automatic grid determination (default: 500.0)",
    )
    parser.add_argument(
        "--mode",
        choices=["lookup", "onthefly"],
        default="lookup",
        help="Radial density mode: 'lookup' uses a precomputed table, "
        "'onthefly' computes densities during analysis (default: lookup)",
    )
    parser.add_argument(
        "--lookup-table",
        default=None,
        help="Path to .npz radial lookup table (required for lookup mode)",
    )
    parser.add_argument(
        "--use-dual-model",
        action="store_true",
        default=True,
        help="Use dual model (F_nonlocal + F_local) for prediction (default: True)",
    )
    parser.add_argument(
        "--no-dual-model",
        action="store_true",
        help="Disable dual model; use F_nonlocal only",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for results. If omitted, results are printed to stdout.",
    )
    parser.set_defaults(func=_run_analyze)


def _add_precompute_args(parser):
    """Add arguments for the precompute subcommand to *parser*."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--elements",
        nargs="+",
        help="Element symbols to include in the lookup table (e.g. Si O Hf)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Precompute all 118 elements of the periodic table (skips failures)",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=20.0,
        help="Cubic cell size in Angstroms for isolated-atom calculations (default: 20.0)",
    )
    parser.add_argument(
        "--encut",
        type=float,
        default=500.0,
        help="Plane-wave energy cutoff for grid determination (default: 500.0)",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.02,
        help="Radial bin width in Angstroms (default: 0.02)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=5.0,
        help="Maximum radius in Angstroms for the lookup table (default: 5.0)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file path for the radial lookup table",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save checkpoint every N elements during precomputation (default: 5)",
    )
    parser.add_argument(
        "--use-dual-model",
        action="store_true",
        default=True,
        help="Use dual model (F_nonlocal + F_local) for prediction (default: True)",
    )
    parser.add_argument(
        "--no-dual-model",
        action="store_true",
        help="Disable dual model; use F_nonlocal only",
    )
    parser.set_defaults(func=_run_precompute)


def main():
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="deepaw-hirshfeld",
        description="Hirshfeld charge analysis using DeePAW charge density predictions.",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run Hirshfeld charge analysis on a crystal structure",
    )
    _add_analyze_args(analyze_parser)

    # precompute subcommand
    precompute_parser = subparsers.add_parser(
        "precompute",
        help="Build a radial lookup table for specified elements",
    )
    _add_precompute_args(precompute_parser)

    args = parser.parse_args()

    # Default to "analyze" when no subcommand is given
    if args.subcommand is None:
        # Re-parse with "analyze" prepended so that analyze arguments are accepted
        args = parser.parse_args(["analyze"] + sys.argv[1:])

    args.func(args)


if __name__ == "__main__":
    main()
