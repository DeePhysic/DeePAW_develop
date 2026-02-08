"""
DeePAW: Deep Learning for PAW Charge Density Prediction

A professional framework for predicting charge density in materials using
E3-equivariant neural networks with local and non-local corrections.
"""

__version__ = "1.0.0"
__author__ = "DeePAW Team"
__description__ = "Deep Learning for PAW Charge Density Prediction"
__url__ = "https://github.com/your-repo/DeePAW"

from .models.f_nonlocal import F_nonlocal
from .models.f_local import F_local
from .inference import InferenceEngine
from .hirshfeld import HirshfeldAnalysis

__all__ = [
    "F_nonlocal",
    "F_local",
    "InferenceEngine",
    "HirshfeldAnalysis",
    "__version__",
]

