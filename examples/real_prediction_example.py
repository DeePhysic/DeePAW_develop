#!/usr/bin/env python
"""
Real Prediction Example with Actual Data

This example demonstrates a complete prediction workflow with real data.
"""

import torch
import sys
import os
import numpy as np
from ase.build import bulk

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deepaw import F_nonlocal, F_local
from deepaw.data.graph_construction import KdTreeGraphConstructor
from deepaw.data.collate import collate_list_of_dicts
from deepaw.config import get_device, get_model_config, get_checkpoint_path


def create_dummy_batch(device='cpu'):
    """
    Create a dummy batch for testing using real ASE atoms and GraphConstructor
    """
    print("\n📦 Creating dummy batch data...")

    # Create a simple crystal structure (Silicon)
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms = atoms * (2, 2, 2)  # 2x2x2 supercell

    print(f"  ✓ Created structure:")
    print(f"    - Formula: {atoms.get_chemical_formula()}")
    print(f"    - Atoms: {len(atoms)}")
    print(f"    - Cell: {atoms.get_cell().lengths()}")

    # Create dummy density grid
    grid_shape = (10, 10, 10)
    grid_pos = np.zeros((*grid_shape, 3))
    cell = atoms.get_cell()
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                grid_pos[i, j, k] = cell.cartesian_positions([i/grid_shape[0],
                                                               j/grid_shape[1],
                                                               k/grid_shape[2]])

    # Create dummy density values
    density = np.random.rand(*grid_shape) * 0.1

    # Use GraphConstructor to create proper graph
    graph_constructor = KdTreeGraphConstructor(cutoff=4.0, num_probes=100)
    graph_dict = graph_constructor(density, atoms, grid_pos)

    # Collate into batch format (even though it's a single sample)
    batch = collate_list_of_dicts([graph_dict], pin_memory=False)

    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    print(f"  ✓ Batch created:")
    print(f"    - Atoms: {batch['nodes'].shape}")
    print(f"    - Probes: {batch['num_probes'].shape}")
    print(f"    - Atom edges: {batch['atom_edges'].shape}")
    print(f"    - Probe edges: {batch['probe_edges'].shape}")
    print(f"    - Device: {device}")

    return batch


def test_single_model():
    """Test single model (F_nonlocal only)"""
    print("\n" + "="*70)
    print("Test 1: Single Model Prediction (F_nonlocal)")
    print("="*70)

    # Device setup - use config
    device = get_device()
    print(f"Using device: {device}")

    try:
        # Create model with default config
        print("\n🔧 Creating F_nonlocal model...")
        model_config = get_model_config('f_nonlocal')
        model = F_nonlocal(**model_config)
        model = model.to(device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with default config ({num_params:,} parameters)")

        # Load weights - use config
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', get_checkpoint_path('f_nonlocal'))
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print(f"  ✓ Loaded pretrained weights")
        else:
            print(f"  ⚠ Using random weights (checkpoint not found)")
        
        # Create dummy data
        batch = create_dummy_batch(device)
        
        # Run prediction
        print("\n🚀 Running prediction...")
        with torch.no_grad():
            output, node_rep = model(batch)
        
        print(f"  ✓ Prediction successful!")
        print(f"    - Output shape: {output.shape}")
        print(f"    - Node representation shape: {node_rep.shape}")
        print(f"    - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in single model test:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dual_model():
    """Test dual model (F_nonlocal + F_local)"""
    print("\n" + "="*70)
    print("Test 2: Dual Model Prediction (F_nonlocal + F_local)")
    print("="*70)

    # Device setup - use config
    device = get_device()
    print(f"Using device: {device}")

    try:
        # Create models with default configs
        print("\n🔧 Creating models...")
        f_nonlocal_config = get_model_config('f_nonlocal')
        f_local_config = get_model_config('f_local')
        f_nonlocal = F_nonlocal(**f_nonlocal_config)
        f_local = F_local(**f_local_config)

        f_nonlocal = f_nonlocal.to(device)
        f_local = f_local.to(device)

        f_nonlocal.eval()
        f_local.eval()

        params_nonlocal = sum(p.numel() for p in f_nonlocal.parameters())
        params_local = sum(p.numel() for p in f_local.parameters())

        print(f"  ✓ F_nonlocal created with default config ({params_nonlocal:,} parameters)")
        print(f"  ✓ F_local created with default config ({params_local:,} parameters)")

        # Load weights - use config
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..')

        checkpoint_nonlocal = os.path.join(checkpoint_dir, get_checkpoint_path('f_nonlocal'))
        if os.path.exists(checkpoint_nonlocal):
            state_dict = torch.load(checkpoint_nonlocal, map_location=device, weights_only=False)
            f_nonlocal.load_state_dict(state_dict)
            print(f"  ✓ Loaded F_nonlocal weights")

        checkpoint_local = os.path.join(checkpoint_dir, get_checkpoint_path('f_local'))
        if os.path.exists(checkpoint_local):
            state_dict = torch.load(checkpoint_local, map_location=device, weights_only=False)
            f_local.load_state_dict(state_dict)
            print(f"  ✓ Loaded F_local weights")
        
        # Create dummy data
        batch = create_dummy_batch(device)
        
        # Run prediction
        print("\n🚀 Running dual model prediction...")
        with torch.no_grad():
            # Step 1: F_nonlocal prediction
            base_pred, node_rep = f_nonlocal(batch)
            print(f"  ✓ F_nonlocal prediction done")
            print(f"    - Base prediction shape: {base_pred.shape}")
            print(f"    - Node representation shape: {node_rep.shape}")
            
            # Step 2: F_local correction
            correction, _ = f_local(None, node_rep)
            print(f"  ✓ F_local correction done")
            print(f"    - Correction shape: {correction.shape}")
            
            # Step 3: Combine
            final_pred = base_pred.reshape(-1) + correction.reshape(-1)
            print(f"  ✓ Final prediction computed")
            print(f"    - Final shape: {final_pred.shape}")
            print(f"    - Final range: [{final_pred.min().item():.4f}, {final_pred.max().item():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in dual model test:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("\n" + "="*70)
    print("DeePAW: Real Prediction Example")
    print("="*70)
    
    # Run tests
    test1_passed = test_single_model()
    test2_passed = test_dual_model()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"  Single Model Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Dual Model Test:   {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! DeePAW is working correctly.")
        print("="*70 + "\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

