"""
Energy Prediction Module

This module provides energy prediction models using atomic embeddings
extracted from the pretrained DeePAW F_nonlocal model.

Main components:
- EnergyHead: E3-equivariant energy prediction head
- ScalarEnergyHead: Simple MLP-based energy head using only scalar features
- ForceHead: Force prediction via energy gradient
"""

from .models import EnergyHead, ScalarEnergyHead, EnergyForceHead

__all__ = ['EnergyHead', 'ScalarEnergyHead', 'EnergyForceHead']
