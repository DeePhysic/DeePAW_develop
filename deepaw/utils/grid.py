"""
网格自动生成工具

根据 VASP 的 FFT 网格规则自动计算合适的电荷密度网格尺寸。
"""
import numpy as np

# 物理常量
HBAR2_OVER_2ME = 3.81001987408  # hbar^2 / (2*m_e), 单位: eV*Angstrom^2
RYDBERG_TO_EV = 13.605826       # 1 Ry = 13.605826 eV
BOHR_TO_ANG = 0.529177249       # 1 Bohr = 0.529177249 Angstrom


def get_vasp_optimal_grid(n_target):
    """找到 >= n_target 的最小整数，其质因数仅包含 2, 3, 5, 7（VASP FFT 要求）。

    Args:
        n_target: 目标网格点数

    Returns:
        int: 满足 VASP FFT 要求的最小网格点数
    """
    def has_only_small_prime_factors(n):
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                n //= p
        return n == 1

    current_n = int(n_target)
    while not has_only_small_prime_factors(current_n):
        current_n += 1
    return current_n


def calculate_grid_size(atoms, encut=500.0):
    """根据 ENCUT 和晶格常数自动计算 VASP 兼容的电荷密度网格。

    Args:
        atoms: ASE Atoms 对象
        encut: 平面波截断能 (eV)，默认 500 eV

    Returns:
        tuple: (nx, ny, nz) 精细网格点数

    Examples:
        >>> from ase.io import read
        >>> atoms = read("POSCAR")
        >>> nx, ny, nz = calculate_grid_size(atoms, encut=500.0)
    """
    lattice_params = atoms.cell.cellpar()[:3]  # a, b, c (Angstrom)

    # 基础网格: 根据 G_max 和倒格矢计算最小网格点数
    ngxyz = np.ceil(
        2.0 + 4.0 * np.sqrt(encut / RYDBERG_TO_EV)
        / (2.0 * np.pi / lattice_params / BOHR_TO_ANG)
    )

    # 调整为 VASP FFT 兼容的网格（质因数仅含 2,3,5,7）
    ngxyz = np.array([get_vasp_optimal_grid(n) for n in ngxyz])

    # 精细网格 = 2 倍基础网格
    fine_grid = 2 * ngxyz

    return tuple(fine_grid.astype(int))
