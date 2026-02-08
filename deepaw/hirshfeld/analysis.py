"""
Hirshfeld charge analysis module.

Computes per-atom Hirshfeld charges from crystal charge densities using the
deformation density method (Hirshfeld Method B).

The analysis partitions the deformation density (crystal minus promolecule)
among atoms using free-atom density weights, yielding partial atomic charges
that sum to the total excess/deficit of electrons relative to the neutral
promolecule reference.

Reference:
    F. L. Hirshfeld, Theoretica Chimica Acta, 44, 129-138 (1977)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HirshfeldResult:
    """Result container for a Hirshfeld charge analysis.

    Attributes:
        charges: Per-atom Hirshfeld charges, shape (n_atoms,). Positive values
            indicate electron depletion (cation-like), negative values indicate
            electron accumulation (anion-like).
        symbols: Element symbols for each atom.
        deformation_integral: Total deformation density integral. For a neutral
            system this should be approximately zero; large deviations indicate
            grid or density issues.
        total_charge: Sum of all Hirshfeld charges.
        crystal_electrons: Integrated crystal electron count over the grid.
        promolecule_electrons: Integrated promolecule electron count over the
            grid.
    """

    charges: np.ndarray
    symbols: List[str]
    deformation_integral: float
    total_charge: float
    crystal_electrons: float
    promolecule_electrons: float

    def __str__(self) -> str:
        """Return a formatted table of Hirshfeld charges and diagnostics."""
        lines = []
        lines.append("Hirshfeld Charge Analysis")
        lines.append("=" * 45)
        lines.append(f"{'Index':>6}  {'Species':>8}  {'Charge':>12}")
        lines.append("-" * 45)
        for i, (sym, q) in enumerate(zip(self.symbols, self.charges)):
            lines.append(f"{i:>6d}  {sym:>8s}  {q:>12.6f}")
        lines.append("-" * 45)
        lines.append(f"{'Total charge':>20s}: {self.total_charge:>12.6f}")
        lines.append(f"{'Crystal electrons':>20s}: {self.crystal_electrons:>12.6f}")
        lines.append(f"{'Promolecule electrons':>20s}: {self.promolecule_electrons:>12.6f}")
        lines.append(f"{'Deformation integral':>20s}: {self.deformation_integral:>12.6f}")
        lines.append("=" * 45)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of the result.

        Returns:
            dict with keys: charges (list), symbols, deformation_integral,
            total_charge, crystal_electrons, promolecule_electrons.
        """
        return {
            "charges": self.charges.tolist(),
            "symbols": list(self.symbols),
            "deformation_integral": float(self.deformation_integral),
            "total_charge": float(self.total_charge),
            "crystal_electrons": float(self.crystal_electrons),
            "promolecule_electrons": float(self.promolecule_electrons),
        }

    def save(self, path: str) -> None:
        """Save the analysis results to a text file.

        Args:
            path: Output file path.
        """
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(str(self))
            f.write("\n\n")
            f.write("JSON:\n")
            f.write(json.dumps(self.to_dict(), indent=2))
            f.write("\n")


class HirshfeldAnalysis:
    """Hirshfeld charge analysis for crystalline materials.

    Computes per-atom partial charges by partitioning the deformation density
    (crystal density minus promolecule density) using free-atom density weights
    (Hirshfeld Method B).

    The promolecule density is the superposition of spherical free-atom
    densities placed at the crystal atomic positions. Each atom's share of the
    deformation density is determined by its weight function:

        w_i(r) = rho_free_i(r) / rho_promolecule(r)

    and the Hirshfeld charge is:

        Q_i = -integral[ w_i(r) * (rho_crystal(r) - rho_promolecule(r)) ] dV

    Args:
        engine: An ``InferenceEngine`` instance used to predict crystal charge
            densities when they are not supplied directly, and/or to compute
            free-atom densities in ``"onthefly"`` mode.
        mode: Free-atom density source. ``"lookup"`` uses a precomputed radial
            lookup table (fast). ``"onthefly"`` predicts free-atom densities
            with the inference engine (slower but requires no lookup table).
        lookup_path: Path to a ``.npz`` file containing the radial density
            lookup table. Used only when *mode* is ``"lookup"``.
        lookup_table: A ``RadialDensityLookup`` instance. If provided, takes
            precedence over *lookup_path*.
        encut: Plane-wave energy cutoff in eV for automatic grid size
            calculation when no grid shape is specified. Default 500.0 eV.

    Examples:
        Basic usage with a precomputed crystal density::

            from deepaw.inference import InferenceEngine
            from deepaw.hirshfeld import HirshfeldAnalysis
            from ase.build import bulk

            engine = InferenceEngine()
            ha = HirshfeldAnalysis(engine=engine, mode="lookup",
                                   lookup_path="free_atom_radial.npz")

            atoms = bulk("NaCl", "rocksalt", a=5.64)
            result = ha.analyze(atoms)
            print(result)

        Using a pre-predicted density array::

            import numpy as np
            density = np.random.rand(80, 80, 80)  # placeholder
            result = ha.analyze(atoms, crystal_density=density)
            print(result.charges)

        Accessing results programmatically::

            result_dict = result.to_dict()
            na_charge = result.charges[0]
    """

    def __init__(
        self,
        engine=None,
        mode: str = "lookup",
        lookup_path: Optional[str] = None,
        lookup_table=None,
        encut: float = 500.0,
    ):
        self.engine = engine
        self.mode = mode
        self.encut = encut

        from .free_atom import FreeAtomDensityProvider

        self._provider = FreeAtomDensityProvider(
            mode=mode,
            engine=engine,
            lookup_table=lookup_table,
            lookup_path=lookup_path,
        )

    def analyze(
        self,
        atoms,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        crystal_density: Optional[np.ndarray] = None,
    ) -> HirshfeldResult:
        """Run Hirshfeld charge analysis on a crystal structure.

        Args:
            atoms: ASE ``Atoms`` object representing the crystal structure.
            grid_shape: Tuple ``(nx, ny, nz)`` specifying the real-space grid
                dimensions. If *crystal_density* is provided and *grid_shape*
                is ``None``, the shape is inferred from the density array. If
                both are ``None``, the grid is auto-computed from *encut*.
            crystal_density: 3-D numpy array of the crystal charge density on
                the real-space grid. If ``None``, the density is predicted
                using the inference engine.

        Returns:
            HirshfeldResult: Dataclass containing per-atom charges, element
            symbols, and diagnostic quantities.

        Raises:
            ValueError: If *crystal_density* is ``None`` and no engine was
                provided, or if the density shape does not match *grid_shape*.
        """
        # ------------------------------------------------------------------
        # 1. Determine grid_shape
        # ------------------------------------------------------------------
        if crystal_density is not None and grid_shape is None:
            grid_shape = crystal_density.shape
            logger.debug(
                "Inferred grid_shape from crystal_density: %s", grid_shape
            )
        elif crystal_density is None and grid_shape is None:
            from deepaw.utils import calculate_grid_size

            grid_shape = calculate_grid_size(atoms, encut=self.encut)
            logger.info(
                "Auto-computed grid_shape: %s (encut=%.1f eV)",
                grid_shape,
                self.encut,
            )

        nx, ny, nz = grid_shape

        # ------------------------------------------------------------------
        # 2. Obtain crystal density
        # ------------------------------------------------------------------
        if crystal_density is None:
            if self.engine is None:
                raise ValueError(
                    "crystal_density is None and no InferenceEngine was "
                    "provided. Either supply a density array or pass an "
                    "engine to HirshfeldAnalysis."
                )
            logger.info("Predicting crystal density with InferenceEngine ...")
            result = self.engine.predict(atoms=atoms, grid_shape=grid_shape)
            crystal_density = result["density_3d"]

        if crystal_density.shape != (nx, ny, nz):
            raise ValueError(
                f"crystal_density shape {crystal_density.shape} does not "
                f"match grid_shape ({nx}, {ny}, {nz})."
            )

        # ------------------------------------------------------------------
        # 3. Get free-atom densities on the grid
        # ------------------------------------------------------------------
        logger.info("Computing free-atom densities on grid ...")
        atom_densities = self._provider.get_free_atom_densities(
            atoms, grid_shape
        )

        # ------------------------------------------------------------------
        # 4. Compute Hirshfeld charges
        # ------------------------------------------------------------------
        logger.info("Computing Hirshfeld charges ...")
        charges, deformation_total = self._compute_charges(
            crystal_density, atom_densities, atoms, nx, ny, nz
        )

        # ------------------------------------------------------------------
        # 5. Diagnostics
        # ------------------------------------------------------------------
        cell = np.array(atoms.get_cell())
        cell_volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        dV = cell_volume / (nx * ny * nz)

        crystal_electrons = float(np.sum(crystal_density) * dV)
        promolecule_electrons = float(
            np.sum(np.sum(atom_densities, axis=0)) * dV
        )
        total_charge = float(np.sum(charges))

        symbols = list(atoms.get_chemical_symbols())

        logger.info(
            "Hirshfeld analysis complete: total_charge=%.6f, "
            "deformation_integral=%.6f",
            total_charge,
            deformation_total,
        )

        # ------------------------------------------------------------------
        # 6. Return result
        # ------------------------------------------------------------------
        return HirshfeldResult(
            charges=charges,
            symbols=symbols,
            deformation_integral=float(deformation_total),
            total_charge=total_charge,
            crystal_electrons=crystal_electrons,
            promolecule_electrons=promolecule_electrons,
        )

    @staticmethod
    def _compute_charges(
        cryst_density: np.ndarray,
        atom_densities: np.ndarray,
        atoms,
        nx: int,
        ny: int,
        nz: int,
    ) -> Tuple[np.ndarray, float]:
        """Core Hirshfeld Method B charge computation.

        Partitions the deformation density among atoms using free-atom weight
        functions and integrates to obtain per-atom charges.

        Args:
            cryst_density: Crystal charge density, shape ``(nx, ny, nz)``.
            atom_densities: Free-atom densities, shape ``(n_atoms, nx, ny, nz)``.
            atoms: ASE ``Atoms`` object.
            nx: Grid points along the first lattice vector.
            ny: Grid points along the second lattice vector.
            nz: Grid points along the third lattice vector.

        Returns:
            Tuple of (charges, deformation_total) where *charges* is a 1-D
            array of shape ``(n_atoms,)`` and *deformation_total* is the
            integrated total deformation density (should be near zero for a
            neutral system).
        """
        cell = np.array(atoms.get_cell())
        cell_volume = np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))
        dV = cell_volume / (nx * ny * nz)

        # Pro-crystal (promolecule) density: superposition of free-atom densities
        rho_pro = np.zeros_like(cryst_density)
        for rho_at in atom_densities:
            rho_pro += rho_at

        # Deformation density
        delta_rho = cryst_density - rho_pro

        # Safe division to avoid division by zero in vacuum regions
        rho_pro_safe = np.where(rho_pro > 1e-30, rho_pro, 1e-30)

        # Per-atom charges
        n_atoms = len(atom_densities)
        charges = np.zeros(n_atoms)
        for i in range(n_atoms):
            w_i = atom_densities[i] / rho_pro_safe
            delta_rho_i = w_i * delta_rho
            integral_i = np.sum(delta_rho_i) * dV
            charges[i] = -integral_i

        deformation_total = float(np.sum(delta_rho) * dV)
        return charges, deformation_total
