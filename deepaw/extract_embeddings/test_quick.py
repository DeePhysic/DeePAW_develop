#!/usr/bin/env python
"""
Quick test script to verify the embedding extraction module works correctly.
"""

import os
import sys

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

print("=" * 60)
print("Testing Atomic Embedding Extraction Module")
print("=" * 60)

try:
    # Test 1: Import module
    print("\n[1/4] Testing module import...")
    from deepaw.extract_embeddings import AtomicEmbeddingExtractor
    print("✓ Module imported successfully")

    # Test 2: Initialize extractor
    print("\n[2/4] Initializing extractor...")
    extractor = AtomicEmbeddingExtractor()
    print("✓ Extractor initialized")

    # Test 3: Create a simple structure and extract embeddings
    print("\n[3/4] Extracting embeddings from Si structure...")
    from ase.build import bulk
    atoms = bulk('Si', 'diamond', a=5.43)
    print(f"   Structure: {len(atoms)} Si atoms")

    embeddings = extractor.extract(atoms)
    print(f"✓ Embeddings extracted: shape {embeddings.shape}")

    # Test 4: Verify embedding properties
    print("\n[4/4] Verifying embedding properties...")
    import numpy as np

    assert embeddings.shape[0] == len(atoms), "Number of embeddings should match number of atoms"
    assert embeddings.shape[1] > 0, "Embedding dimension should be positive"
    assert not np.isnan(embeddings).any(), "Embeddings should not contain NaN"
    assert not np.isinf(embeddings).any(), "Embeddings should not contain Inf"

    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print("✓ All checks passed")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Module is working correctly.")
    print("=" * 60)

    print("\n📖 Next steps:")
    print("   1. Run examples: python deepaw/extract_embeddings/example_usage.py")
    print("   2. Read documentation: deepaw/extract_embeddings/README.md")
    print("   3. Use in your code:")
    print("      from deepaw.extract_embeddings import AtomicEmbeddingExtractor")
    print("      extractor = AtomicEmbeddingExtractor()")
    print("      embeddings = extractor.extract(atoms)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
