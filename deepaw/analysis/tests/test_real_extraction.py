#!/usr/bin/env python
"""
Real test script to verify embedding extraction works with pretrained model.
"""

import os
import sys
import numpy as np

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

print("=" * 70)
print("REAL TEST: Extracting Atomic Embeddings from Random Structure")
print("=" * 70)

try:
    # Step 1: Import modules
    print("\n[Step 1/5] Importing modules...")
    from deepaw.extract_embeddings import AtomicEmbeddingExtractor
    from ase.build import bulk
    from ase import Atoms
    import torch
    print("✓ Modules imported successfully")

    # Step 2: Create test structures
    print("\n[Step 2/5] Creating test structures...")

    # Test 1: Simple Si structure
    si_atoms = bulk('Si', 'diamond', a=5.43)
    print(f"  - Si diamond: {len(si_atoms)} atoms")

    # Test 2: Random structure (small molecule-like)
    positions = np.random.rand(5, 3) * 10  # 5 atoms in 10x10x10 box
    random_atoms = Atoms('C5', positions=positions, cell=[10, 10, 10], pbc=True)
    print(f"  - Random C5: {len(random_atoms)} atoms")

    # Test 3: Larger structure
    large_atoms = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
    print(f"  - Large Si: {len(large_atoms)} atoms")

    # Step 3: Initialize extractor
    print("\n[Step 3/5] Initializing embedding extractor...")
    print("  (This will load the pretrained model)")
    extractor = AtomicEmbeddingExtractor()
    print("✓ Extractor initialized")

    # Step 4: Extract embeddings
    print("\n[Step 4/5] Extracting embeddings...")

    print("\n  Test 1: Si diamond structure")
    emb_si = extractor.extract(si_atoms)
    print(f"    Shape: {emb_si.shape}")
    print(f"    Mean: {emb_si.mean():.6f}")
    print(f"    Std: {emb_si.std():.6f}")
    print(f"    Norm (mean): {np.linalg.norm(emb_si, axis=1).mean():.6f}")

    print("\n  Test 2: Random C5 structure")
    emb_random = extractor.extract(random_atoms)
    print(f"    Shape: {emb_random.shape}")
    print(f"    Mean: {emb_random.mean():.6f}")
    print(f"    Std: {emb_random.std():.6f}")
    print(f"    Norm (mean): {np.linalg.norm(emb_random, axis=1).mean():.6f}")

    print("\n  Test 3: Large Si structure")
    emb_large = extractor.extract(large_atoms)
    print(f"    Shape: {emb_large.shape}")
    print(f"    Mean: {emb_large.mean():.6f}")
    print(f"    Std: {emb_large.std():.6f}")
    print(f"    Norm (mean): {np.linalg.norm(emb_large, axis=1).mean():.6f}")

    # Step 5: Verify results
    print("\n[Step 5/5] Verifying results...")

    # Check shapes
    assert emb_si.shape == (len(si_atoms), emb_si.shape[1]), "Shape mismatch for Si"
    assert emb_random.shape == (len(random_atoms), emb_random.shape[1]), "Shape mismatch for random"
    assert emb_large.shape == (len(large_atoms), emb_large.shape[1]), "Shape mismatch for large"
    print("  ✓ Shapes are correct")

    # Check for NaN/Inf
    assert not np.isnan(emb_si).any(), "Si embeddings contain NaN"
    assert not np.isnan(emb_random).any(), "Random embeddings contain NaN"
    assert not np.isnan(emb_large).any(), "Large embeddings contain NaN"
    assert not np.isinf(emb_si).any(), "Si embeddings contain Inf"
    assert not np.isinf(emb_random).any(), "Random embeddings contain Inf"
    assert not np.isinf(emb_large).any(), "Large embeddings contain Inf"
    print("  ✓ No NaN or Inf values")

    # Check embedding dimension
    emb_dim = emb_si.shape[1]
    print(f"  ✓ Embedding dimension: {emb_dim}")

    # Save results
    print("\n[Bonus] Saving results...")
    output_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    extractor.save_embeddings(emb_si, si_atoms,
                             os.path.join(output_dir, 'si_embeddings.npz'))
    extractor.save_embeddings(emb_random, random_atoms,
                             os.path.join(output_dir, 'random_embeddings.npz'))
    extractor.save_embeddings(emb_large, large_atoms,
                             os.path.join(output_dir, 'large_embeddings.npz'))

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - Successfully extracted embeddings from 3 different structures")
    print(f"  - Embedding dimension: {emb_dim}")
    print(f"  - All embeddings are valid (no NaN/Inf)")
    print(f"  - Results saved to: {output_dir}")
    print("\n🎉 The embedding extraction module is working correctly!")

except Exception as e:
    print("\n" + "=" * 70)
    print("❌ TEST FAILED!")
    print("=" * 70)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
