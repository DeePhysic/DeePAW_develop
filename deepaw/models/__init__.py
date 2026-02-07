"""
DeePAW Models

This module contains the core neural network models for charge density prediction:
- F_nonlocal: Non-local charge density prediction using E3-equivariant networks
- F_local: Local correction using Kolmogorov-Arnold Networks (KAN)
"""

from .f_nonlocal import F_nonlocal, E3DensityModel
from .f_local import F_local, ResidualCorrectionModel

__all__ = [
    "F_nonlocal",
    "F_local",
    "E3DensityModel",  # Legacy alias
    "ResidualCorrectionModel",  # Legacy alias
]

