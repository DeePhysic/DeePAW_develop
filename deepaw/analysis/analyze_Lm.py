#!/usr/bin/env python
"""Analyze embedding by L and m components."""

import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase.build import bulk

extractor = AtomicEmbeddingExtractor()

# Create Si diamond
si = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
embeddings = extractor.extract(si)

# Get Type A (even index) and Type B (odd index)
emb_A = embeddings[0]  # First Type A atom
emb_B = embeddings[1]  # First Type B atom

print("=" * 70)
print("Embedding Structure Analysis by L and m")
print("=" * 70)

# Irreps structure from get_irreps(500, 4)
# (mul, (L, parity)) for L=0..4, parity=-1,+1
irreps_info = []
total_mul = 500
lmax = 4
for l in range(lmax + 1):
    for p in [-1, 1]:
        mul = round(total_mul / (lmax + 1) / (2 * l + 1))
        dim = mul * (2 * l + 1)
        parity_str = 'o' if p == -1 else 'e'
        irreps_info.append({
            'L': l,
            'parity': p,
            'parity_str': parity_str,
            'mul': mul,
            'dim': dim,
            'num_m': 2 * l + 1
        })

print(f"\nIrreps structure (total dim = {sum(i['dim'] for i in irreps_info)}):")
print("-" * 50)
for info in irreps_info:
    print(f"  L={info['L']}{info['parity_str']}: mul={info['mul']}, "
          f"m=-{info['L']}..+{info['L']}, dim={info['dim']}")

# Split embeddings by irreps
print("\n" + "=" * 70)
print("Comparison: Type A (atom 0) vs Type B (atom 1)")
print("=" * 70)

idx = 0
for info in irreps_info:
    L = info['L']
    p_str = info['parity_str']
    dim = info['dim']
    mul = info['mul']
    num_m = info['num_m']

    # Extract this irrep's components
    comp_A = emb_A[idx:idx+dim]
    comp_B = emb_B[idx:idx+dim]

    # Reshape to (mul, 2L+1) to separate m components
    comp_A_reshape = comp_A.reshape(mul, num_m)
    comp_B_reshape = comp_B.reshape(mul, num_m)

    # Compare
    diff = comp_A - comp_B
    ratio = comp_B / (comp_A + 1e-10)  # B/A ratio

    print(f"\n--- L={L}{p_str} (dim={dim}) ---")
    print(f"  Type A norm: {np.linalg.norm(comp_A):.6f}")
    print(f"  Type B norm: {np.linalg.norm(comp_B):.6f}")
    print(f"  Difference norm: {np.linalg.norm(diff):.6f}")

    # Check if A ≈ B or A ≈ -B
    cos_sim = np.dot(comp_A, comp_B) / (np.linalg.norm(comp_A) * np.linalg.norm(comp_B) + 1e-10)
    print(f"  Cosine similarity: {cos_sim:.6f}")

    if np.linalg.norm(comp_A) > 0.01:
        if cos_sim > 0.99:
            print(f"  --> A ≈ B (same)")
        elif cos_sim < -0.99:
            print(f"  --> A ≈ -B (opposite sign)")
        else:
            print(f"  --> A and B are different")

    idx += dim
