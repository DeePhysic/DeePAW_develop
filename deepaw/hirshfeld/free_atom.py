"""Free-atom density provider for Hirshfeld charge analysis.

Supports two modes for obtaining isolated free-atom charge densities:

- **lookup**: Uses a precomputed radial lookup table (fast, recommended for
  production). Requires either a :class:`RadialDensityLookup` instance or a
  path to a ``.npz`` file produced by :func:`radial_lookup.build_radial_lookup`.

- **onthefly**: Computes free-atom densities on the fly by running the
  :class:`~deepaw.inference.InferenceEngine` on single-atom cells. Slower but
  does not require a precomputed table.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms

logger = logging.getLogger(__name__)


class FreeAtomDensityProvider:
    """Provide free-atom charge densities for Hirshfeld partitioning.

    Parameters
    ----------
    mode : str
        Either ``"lookup"`` (use a precomputed radial table) or ``"onthefly"``
        (compute densities via the inference engine).
    engine : InferenceEngine, optional
        A :class:`~deepaw.inference.InferenceEngine` instance.  Required when
        *mode* is ``"onthefly"``; optional for ``"lookup"``.
    lookup_table : RadialDensityLookup, optional
        A preloaded :class:`RadialDensityLookup` instance.  Used only when
        *mode* is ``"lookup"``.
    lookup_path : str, optional
        Path to a ``.npz`` file that can be loaded as a
        :class:`RadialDensityLookup`.  An alternative to *lookup_table* for
        ``"lookup"`` mode.

    Raises
    ------
    ValueError
        If the required arguments for the chosen *mode* are not provided, or
        if *mode* is not one of the recognised values.
    """

    def __init__(
        self,
        mode: str = "lookup",
        engine=None,
        lookup_table=None,
        lookup_path: Optional[str] = None,
    ):
        if mode not in ("lookup", "onthefly"):
            raise ValueError(
                f"Unknown mode '{mode}'. Must be 'lookup' or 'onthefly'."
            )

        self.mode = mode
        self.engine = engine

        if mode == "lookup":
            if lookup_table is not None:
                self.lookup = lookup_table
            elif lookup_path is not None:
                from .radial_lookup import RadialDensityLookup

                self.lookup = RadialDensityLookup.load(lookup_path)
            else:
                raise ValueError(
                    "mode='lookup' requires either 'lookup_table' or "
                    "'lookup_path' to be provided."
                )
        elif mode == "onthefly":
            if engine is None:
                raise ValueError(
                    "mode='onthefly' requires an 'engine' "
                    "(InferenceEngine instance) to be provided."
                )
            self.lookup = None

    def get_free_atom_densities(
        self, atoms: Atoms, grid_shape: Tuple[int, int, int]
    ) -> List[np.ndarray]:
        """Compute free-atom charge densities for every atom in *atoms*.

        Parameters
        ----------
        atoms : ase.Atoms
            The crystal structure whose atoms define the free-atom references.
        grid_shape : tuple of int
            ``(nx, ny, nz)`` dimensions of the real-space charge density grid.

        Returns
        -------
        list of numpy.ndarray
            One ``(nx, ny, nz)`` array per atom, representing the spherically
            symmetric free-atom density placed at that atom's position and
            evaluated on the specified grid.
        """
        if self.mode == "lookup":
            return self._get_densities_lookup(atoms, grid_shape)
        else:
            return self._get_densities_onthefly(atoms, grid_shape)

    def _get_densities_lookup(
        self, atoms: Atoms, grid_shape: Tuple[int, int, int]
    ) -> List[np.ndarray]:
        """Retrieve free-atom densities from the precomputed lookup table.

        For each atom the radial interpolator is fetched from the lookup table
        and the 3-D density is reconstructed on the target grid.

        Parameters
        ----------
        atoms : ase.Atoms
            Crystal structure.
        grid_shape : tuple of int
            ``(nx, ny, nz)`` grid dimensions.

        Returns
        -------
        list of numpy.ndarray
            Free-atom densities, one per atom.
        """
        from .radial_lookup import RadialDensityLookup

        cell = atoms.get_cell()
        scaled_positions = atoms.get_scaled_positions()
        symbols = atoms.get_chemical_symbols()

        densities: List[np.ndarray] = []
        for i, (symbol, frac_pos) in enumerate(zip(symbols, scaled_positions)):
            if not self.lookup.has_element(symbol):
                raise ValueError(
                    f"Element '{symbol}' (atom index {i}) is not present in "
                    f"the radial lookup table. Available elements: "
                    f"{self.lookup.elements}. Please precompute a table that "
                    f"includes '{symbol}'."
                )

            interpolator = self.lookup.get_interpolator(symbol)
            density_3d = RadialDensityLookup.reconstruct_3d_density(
                interpolator, frac_pos, cell, grid_shape
            )
            densities.append(density_3d)

        return densities

    def _get_densities_onthefly(
        self, atoms: Atoms, grid_shape: Tuple[int, int, int]
    ) -> List[np.ndarray]:
        """Compute free-atom densities on the fly using the inference engine.

        For each atom a single-atom periodic cell is constructed (preserving
        the original unit cell) and the engine predicts its charge density.

        Parameters
        ----------
        atoms : ase.Atoms
            Crystal structure.
        grid_shape : tuple of int
            ``(nx, ny, nz)`` grid dimensions.

        Returns
        -------
        list of numpy.ndarray
            Free-atom densities, one per atom.
        """
        cell = atoms.get_cell()
        scaled_positions = atoms.get_scaled_positions()
        symbols = atoms.get_chemical_symbols()

        densities: List[np.ndarray] = []
        n_atoms = len(symbols)

        for i, (symbol, frac_pos) in enumerate(zip(symbols, scaled_positions)):
            logger.info(
                "Computing free-atom density for atom %d/%d (%s)",
                i + 1,
                n_atoms,
                symbol,
            )

            single_atom = Atoms(
                symbols=[symbol],
                scaled_positions=[frac_pos],
                cell=cell,
                pbc=True,
            )

            result = self.engine.predict(
                atoms=single_atom, grid_shape=grid_shape
            )
            densities.append(result["density_3d"])

        return densities
