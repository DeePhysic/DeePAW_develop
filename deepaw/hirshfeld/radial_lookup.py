"""Pre-computation, storage, loading, and reconstruction of 1D radial density
profiles for free atoms used in Hirshfeld charge analysis.

This module provides :class:`RadialDensityLookup` for managing spherically
averaged radial density profiles and :func:`build_radial_lookup` for
generating those profiles from DeePAW predictions of isolated atoms placed
in large cubic cells.

Typical workflow::

    from deepaw.inference import InferenceEngine
    from deepaw.hirshfeld.radial_lookup import build_radial_lookup

    engine = InferenceEngine()
    lookup = build_radial_lookup(["Si", "O"], engine=engine)
    lookup.save("radial_table.npz")

    # Later, reload:
    lookup = RadialDensityLookup.load("radial_table.npz")
    interp = lookup.get_interpolator("Si")
    rho_at_r = interp(2.0)

    # Precompute all supported elements (skips failures automatically):
    lookup = build_radial_lookup("all", engine=engine)
    print(f"Supported: {lookup.elements}")
    print(f"Failed: {lookup.failed_elements}")
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
DEFAULT_CELL_SIZE: float = 20.0
DEFAULT_BIN_WIDTH: float = 0.02
DEFAULT_MAX_RADIUS: float = 5.0
DEFAULT_ENCUT: float = 500.0

# All elements from H (Z=1) to Og (Z=118)
ALL_ELEMENTS: List[str] = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


class RadialDensityLookup:
    """Manages pre-computed radial density profiles for free atoms.

    Each profile maps an element symbol to a pair of 1-D arrays
    ``(r_bins, rho_values)`` representing the spherically averaged electron
    density as a function of distance from the nucleus.

    Parameters
    ----------
    data : dict, optional
        Mapping of element symbol -> ``(r_bins, rho_values)`` where both
        arrays are 1-D :class:`numpy.ndarray`.
    failed_elements : list of tuple, optional
        List of ``(symbol, error_message)`` for elements that failed during
        precomputation.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        failed_elements: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self._data: Dict[str, Tuple[np.ndarray, np.ndarray]] = data if data is not None else {}
        self._interpolators: Dict[str, interp1d] = {}
        self._failed_elements: List[Tuple[str, str]] = failed_elements if failed_elements is not None else []

    # ------------------------------------------------------------------
    # Properties / queries
    # ------------------------------------------------------------------

    @property
    def elements(self) -> List[str]:
        """Return a sorted list of element symbols with stored profiles."""
        return sorted(self._data.keys())

    @property
    def failed_elements(self) -> List[Tuple[str, str]]:
        """Return list of ``(symbol, error_message)`` for failed elements."""
        return list(self._failed_elements)

    def has_element(self, symbol: str) -> bool:
        """Check whether a radial profile exists for *symbol*."""
        return symbol in self._data

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def get_interpolator(self, symbol: str) -> interp1d:
        """Return a cubic-spline interpolator for the given element.

        The interpolator extrapolates to ``0.0`` outside the stored radial
        range.  Interpolators are cached after the first call for each
        element.

        Parameters
        ----------
        symbol : str
            Element symbol (e.g. ``"Si"``).

        Returns
        -------
        scipy.interpolate.interp1d
            Cubic spline interpolator ``rho(r)``.

        Raises
        ------
        KeyError
            If no profile is stored for *symbol*.
        """
        if symbol in self._interpolators:
            return self._interpolators[symbol]

        if symbol not in self._data:
            raise KeyError(
                f"No radial density profile for element '{symbol}'. "
                f"Available elements: {self.elements}"
            )

        r_bins, rho_values = self._data[symbol]
        interpolator = interp1d(
            r_bins,
            rho_values,
            kind="cubic",
            bounds_error=False,
            fill_value=0.0,
        )
        self._interpolators[symbol] = interpolator
        return interpolator

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save all profiles to a ``.npz`` file.

        The archive contains:
        - ``_elements``: 1-D array of element symbol strings.
        - ``_failed_elements``: 1-D array of failed element symbols.
        - ``_failed_reasons``: 1-D array of error messages for failed elements.
        - ``{symbol}_r``: radial bin centres for each element.
        - ``{symbol}_rho``: density values for each element.

        Parameters
        ----------
        filepath : str
            Destination path (should end with ``.npz``).
        """
        arrays: Dict[str, np.ndarray] = {}
        elements = self.elements
        arrays["_elements"] = np.array(elements, dtype=str)
        for symbol in elements:
            r_bins, rho_values = self._data[symbol]
            arrays[f"{symbol}_r"] = r_bins
            arrays[f"{symbol}_rho"] = rho_values

        if self._failed_elements:
            failed_syms = [s for s, _ in self._failed_elements]
            failed_msgs = [m for _, m in self._failed_elements]
            arrays["_failed_elements"] = np.array(failed_syms, dtype=str)
            arrays["_failed_reasons"] = np.array(failed_msgs, dtype=str)

        np.savez(filepath, **arrays)
        logger.info("Saved radial lookup table to %s (%d elements, %d failed)",
                     filepath, len(elements), len(self._failed_elements))

    @classmethod
    def load(cls, filepath: str) -> "RadialDensityLookup":
        """Load a :class:`RadialDensityLookup` from a ``.npz`` file.

        Parameters
        ----------
        filepath : str
            Path to the ``.npz`` archive previously created by :meth:`save`.

        Returns
        -------
        RadialDensityLookup
        """
        npz = np.load(filepath, allow_pickle=False)
        elements = npz["_elements"]
        data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for symbol in elements:
            r_bins = npz[f"{symbol}_r"]
            rho_values = npz[f"{symbol}_rho"]
            data[str(symbol)] = (r_bins, rho_values)

        failed: List[Tuple[str, str]] = []
        if "_failed_elements" in npz:
            failed_syms = npz["_failed_elements"]
            failed_msgs = npz["_failed_reasons"]
            failed = [(str(s), str(m)) for s, m in zip(failed_syms, failed_msgs)]

        logger.info("Loaded radial lookup table from %s (%d elements, %d failed)",
                     filepath, len(data), len(failed))
        return cls(data=data, failed_elements=failed)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def spherical_average(
        density_3d: np.ndarray,
        cell: np.ndarray,
        atom_frac_pos: np.ndarray,
        bin_width: float = DEFAULT_BIN_WIDTH,
        max_radius: float = DEFAULT_MAX_RADIUS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the spherically averaged radial density around an atom.

        Parameters
        ----------
        density_3d : np.ndarray
            Charge density on a regular grid with shape ``(nx, ny, nz)``.
        cell : np.ndarray
            Lattice vectors as a ``(3, 3)`` array (rows are lattice vectors).
        atom_frac_pos : np.ndarray
            Fractional coordinates ``(3,)`` of the atom centre.
        bin_width : float
            Width of each radial bin in Angstroms.
        max_radius : float
            Maximum radius for the radial profile in Angstroms.

        Returns
        -------
        r_bins : np.ndarray
            Bin centres (1-D).
        rho_avg : np.ndarray
            Average density in each radial shell (1-D).
        """
        nx, ny, nz = density_3d.shape

        # 1. Build fractional grid coordinates
        fx = np.arange(nx) / nx
        fy = np.arange(ny) / ny
        fz = np.arange(nz) / nz
        gfx, gfy, gfz = np.meshgrid(fx, fy, fz, indexing="ij")

        # 2. Fractional displacement from atom, wrapped to [-0.5, 0.5]
        dfx = gfx - atom_frac_pos[0]
        dfy = gfy - atom_frac_pos[1]
        dfz = gfz - atom_frac_pos[2]
        dfx = dfx - np.round(dfx)
        dfy = dfy - np.round(dfy)
        dfz = dfz - np.round(dfz)

        # 3. Convert fractional displacements to Cartesian
        dx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
        dy = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
        dz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]

        # 4. Distances
        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # 5. Bin into radial shells
        n_bins = int(np.ceil(max_radius / bin_width))
        bin_indices = (distances / bin_width).astype(int)
        mask = bin_indices < n_bins

        # 6. Accumulate and average
        rho_sum = np.zeros(n_bins, dtype=np.float64)
        bin_count = np.zeros(n_bins, dtype=np.float64)
        np.add.at(rho_sum, bin_indices[mask], density_3d[mask])
        np.add.at(bin_count, bin_indices[mask], 1.0)

        # Avoid division by zero for empty bins
        rho_avg = np.zeros(n_bins, dtype=np.float64)
        nonzero = bin_count > 0
        rho_avg[nonzero] = rho_sum[nonzero] / bin_count[nonzero]

        # 7. Bin centres
        r_bins = (np.arange(n_bins) + 0.5) * bin_width

        return r_bins, rho_avg

    @staticmethod
    def reconstruct_3d_density(
        interpolator: interp1d,
        atom_frac_pos: np.ndarray,
        cell: np.ndarray,
        grid_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """Reconstruct a 3-D density grid from a radial interpolator.

        Uses the minimum-image convention to compute distances from the atom
        to every grid point, then evaluates the radial interpolator.

        Parameters
        ----------
        interpolator : scipy.interpolate.interp1d
            Radial density interpolator (e.g. from :meth:`get_interpolator`).
        atom_frac_pos : np.ndarray
            Fractional coordinates ``(3,)`` of the atom.
        cell : np.ndarray
            Lattice vectors as a ``(3, 3)`` array.
        grid_shape : tuple of int
            ``(nx, ny, nz)`` grid dimensions.

        Returns
        -------
        np.ndarray
            Reconstructed density with shape ``(nx, ny, nz)``.
        """
        nx, ny, nz = grid_shape

        # Fractional grid
        fx = np.arange(nx) / nx
        fy = np.arange(ny) / ny
        fz = np.arange(nz) / nz
        gfx, gfy, gfz = np.meshgrid(fx, fy, fz, indexing="ij")

        # Fractional displacement, minimum-image convention
        dfx = gfx - atom_frac_pos[0]
        dfy = gfy - atom_frac_pos[1]
        dfz = gfz - atom_frac_pos[2]
        dfx = dfx - np.round(dfx)
        dfy = dfy - np.round(dfy)
        dfz = dfz - np.round(dfz)

        # Cartesian displacement
        dx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
        dy = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
        dz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]

        distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        return interpolator(distances).reshape(nx, ny, nz)


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------


def build_radial_lookup(
    elements: Union[str, List[str]],
    engine,
    cell_size: float = DEFAULT_CELL_SIZE,
    encut: float = DEFAULT_ENCUT,
    bin_width: float = DEFAULT_BIN_WIDTH,
    max_radius: float = DEFAULT_MAX_RADIUS,
    save_path: Optional[str] = None,
    save_interval: int = 5,
) -> RadialDensityLookup:
    """Build a :class:`RadialDensityLookup` by predicting isolated-atom densities.

    For each element, a single atom is placed at the centre of a cubic cell
    and the charge density is predicted using the provided inference engine.
    The 3-D density is then spherically averaged to produce a 1-D radial
    profile.

    Elements that fail during prediction are automatically skipped and
    recorded in :attr:`RadialDensityLookup.failed_elements`.

    Parameters
    ----------
    elements : str or list of str
        Element symbols to compute (e.g. ``["Si", "O", "Hf"]``), or the
        string ``"all"`` to compute all 118 elements of the periodic table.
    engine : deepaw.inference.InferenceEngine
        A pre-initialised inference engine.
    cell_size : float
        Side length of the cubic cell in Angstroms (default 20.0).
    encut : float
        Plane-wave energy cutoff for automatic grid determination (eV).
    bin_width : float
        Radial bin width in Angstroms.
    max_radius : float
        Maximum radius for the radial profile in Angstroms.
    save_path : str, optional
        If provided, the lookup table is saved to this path periodically
        (every *save_interval* elements) and at the end. This provides
        crash resilience during long precomputation runs.
    save_interval : int
        How often to save intermediate results when *save_path* is set.

    Returns
    -------
    RadialDensityLookup
        Lookup table containing radial profiles for all successfully
        computed elements.
    """
    from ase import Atoms

    from deepaw.utils import calculate_grid_size

    # Resolve "all" to the full periodic table
    if isinstance(elements, str) and elements.lower() == "all":
        element_list = list(ALL_ELEMENTS)
    else:
        element_list = list(elements)

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    failed: List[Tuple[str, str]] = []
    total = len(element_list)

    for i, symbol in enumerate(element_list, start=1):
        print(f"[{i}/{total}] Computing radial density for {symbol} ...")
        logger.info("Building radial profile for %s (%d/%d)", symbol, i, total)

        try:
            # 1. Create isolated atom in a cubic cell
            atoms = Atoms(
                symbols=[symbol],
                positions=[[cell_size / 2] * 3],
                cell=[cell_size] * 3,
                pbc=True,
            )

            # 2. Determine grid shape
            grid_shape = calculate_grid_size(atoms, encut=encut)
            logger.debug("Grid shape for %s: %s", symbol, grid_shape)

            # 3. Predict charge density
            result = engine.predict(atoms=atoms, grid_shape=grid_shape)
            density_3d = result["density_3d"]
            cell_matrix = np.array(atoms.cell)

            # 4. Compute effective bin_width: must be at least the grid spacing
            #    to ensure every radial bin contains grid points.
            grid_spacing = cell_size / min(grid_shape)
            effective_bin_width = max(bin_width, grid_spacing)
            if effective_bin_width > bin_width:
                logger.info(
                    "Adjusted bin_width from %.4f to %.4f A (grid spacing = %.4f A)",
                    bin_width, effective_bin_width, grid_spacing,
                )

            # 5. Spherical average (atom is at fractional (0.5, 0.5, 0.5))
            atom_frac_pos = np.array([0.5, 0.5, 0.5])
            r_bins, rho_avg = RadialDensityLookup.spherical_average(
                density_3d,
                cell_matrix,
                atom_frac_pos,
                bin_width=effective_bin_width,
                max_radius=max_radius,
            )

            data[symbol] = (r_bins, rho_avg)
            print(f"    Done. {len(r_bins)} radial bins (bin_width={effective_bin_width:.4f} A), "
                  f"max rho = {rho_avg.max():.6f}")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            failed.append((symbol, error_msg))
            print(f"    FAILED: {error_msg}")
            logger.warning("Failed to compute radial profile for %s: %s", symbol, error_msg)

        # Periodic save for crash resilience
        if save_path and i % save_interval == 0 and data:
            _tmp = RadialDensityLookup(data=data, failed_elements=failed)
            _tmp.save(save_path)
            print(f"    [checkpoint] Saved {len(data)} elements to {save_path}")

    logger.info("Built radial lookup for %d elements (%d failed): %s",
                len(data), len(failed), sorted(data.keys()))

    lookup = RadialDensityLookup(data=data, failed_elements=failed)

    # Final save
    if save_path and data:
        lookup.save(save_path)
        print(f"\nSaved final lookup table to {save_path}")

    # Summary
    if failed:
        print(f"\n{'='*60}")
        print(f"Summary: {len(data)} succeeded, {len(failed)} failed")
        print(f"{'='*60}")
        print("Failed elements:")
        for sym, msg in failed:
            print(f"  {sym}: {msg}")
        print(f"{'='*60}")

    return lookup
