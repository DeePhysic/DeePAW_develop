#!/usr/bin/env python
"""
Basic Prediction Example

This example demonstrates how to use DeePAW for charge density prediction.
"""

import torch
import sys
import os

# Add parent directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from deepaw import F_nonlocal, F_local
from deepaw.config import get_device, get_model_config, get_checkpoint_path


def example_single_model():
    """Example: Single model prediction (F_nonlocal only)"""
    print("\n" + "="*70)
    print("Example 1: Single Model Prediction (F_nonlocal)")
    print("="*70)

    # Device setup - use config
    device = get_device()
    print(f"Using device: {device}")

    # Create model with default config
    model_config = get_model_config('f_nonlocal')
    model = F_nonlocal(**model_config)
    model = model.to(device)

    print(f"✓ F_nonlocal model created with default config")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights (if available)
    checkpoint_path = os.path.join(PROJECT_ROOT, get_checkpoint_path('f_nonlocal'))
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print(f"  Model initialized with random weights")
    
    model.eval()
    print("\n✓ Model ready for inference")
    

def example_dual_model():
    """Example: Dual model prediction (F_nonlocal + F_local)"""
    print("\n" + "="*70)
    print("Example 2: Dual Model Prediction (F_nonlocal + F_local)")
    print("="*70)
    
    # Device setup - use config
    device = get_device()
    print(f"Using device: {device}")

    # Create models with default configs
    f_nonlocal_config = get_model_config('f_nonlocal')
    f_local_config = get_model_config('f_local')
    f_nonlocal = F_nonlocal(**f_nonlocal_config)
    f_local = F_local(**f_local_config)

    f_nonlocal = f_nonlocal.to(device)
    f_local = f_local.to(device)

    print(f"✓ F_nonlocal model created with default config")
    print(f"  - Parameters: {sum(p.numel() for p in f_nonlocal.parameters()):,}")
    print(f"✓ F_local model created with default config")
    print(f"  - Parameters: {sum(p.numel() for p in f_local.parameters()):,}")

    # Load pretrained weights - use config
    checkpoint_nonlocal = os.path.join(PROJECT_ROOT, get_checkpoint_path('f_nonlocal'))
    checkpoint_local = os.path.join(PROJECT_ROOT, get_checkpoint_path('f_local'))

    if os.path.exists(checkpoint_nonlocal):
        f_nonlocal.load_state_dict(torch.load(checkpoint_nonlocal, map_location=device))
        print(f"✓ Loaded F_nonlocal weights")

    if os.path.exists(checkpoint_local):
        f_local.load_state_dict(torch.load(checkpoint_local, map_location=device))
        print(f"✓ Loaded F_local weights")
    
    f_nonlocal.eval()
    f_local.eval()
    
    print("\n✓ Models ready for inference")
    print("\nPrediction workflow:")
    print("  1. base_pred, node_rep = f_nonlocal(batch)")
    print("  2. correction, _ = f_local(None, node_rep)")
    print("  3. final_pred = base_pred + correction")


def example_model_info():
    """Example: Display model architecture information"""
    print("\n" + "="*70)
    print("Example 3: Model Architecture Information")
    print("="*70)
    
    print("\n📊 F_nonlocal Architecture:")
    print("  - Type: E3-Equivariant Neural Network")
    print("  - Components:")
    print("    1. E3AtomRepresentationModel")
    print("       - Message passing layers: 3")
    print("       - Spherical harmonics: lmax=4")
    print("       - Radial basis: Gaussian (10 functions)")
    print("    2. E3ProbeMessageModel")
    print("       - Probe-based density prediction")
    print("       - Cutoff radius: 4.0 Å")
    
    print("\n📊 F_local Architecture:")
    print("  - Type: Kolmogorov-Arnold Network (KAN)")
    print("  - Components:")
    print("    1. MLP: Linear(992 → 24)")
    print("    2. KAN: [24, 12, 6, 1]")
    print("       - Grid size: 8")
    print("       - Spline order: 4")
    
    print("\n🎯 Model Combination:")
    print("  - F_nonlocal: Captures global structural features")
    print("  - F_local: Learns local corrections")
    print("  - Final: Combines both for highest accuracy")


def main():
    """Main function"""
    print("\n" + "="*70)
    print("DeePAW: Basic Prediction Examples")
    print("="*70)
    
    # Run examples
    example_single_model()
    example_dual_model()
    example_model_info()
    
    print("\n" + "="*70)
    print("For more examples, see:")
    print("  - examples/real_prediction_example.py")
    print("  - docs/external_prediction_tutorial.ipynb")
    print("  - README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

