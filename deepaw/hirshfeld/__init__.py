"""
DeePAW Hirshfeld Charge Analysis

Computes Hirshfeld atomic charges from DeePAW-predicted charge densities
using the deformation density method (Hirshfeld Method B).

Reference: F. L. Hirshfeld, Theoretica Chimica Acta, 44, 129-138 (1977)
"""

from .analysis import HirshfeldAnalysis, HirshfeldResult

__all__ = ["HirshfeldAnalysis", "HirshfeldResult"]
