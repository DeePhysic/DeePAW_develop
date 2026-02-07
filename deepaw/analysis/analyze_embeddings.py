#!/usr/bin/env python
"""
Analyze the extracted embeddings to understand their properties.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

print("=" * 70)
print("EMBEDDING ANALYSIS")
print("=" * 70)

# Load embeddings
output_dir = os.path.join(script_dir, 'outputs')

print("\n[1] Loading saved embeddings...")
si_data = np.load(os.path.join(output_dir, 'si_embeddings.npz'))
random_data = np.load(os.path.join(output_dir, 'random_embeddings.npz'))
large_data = np.load(os.path.join(output_dir, 'large_embeddings.npz'))

si_emb = si_data['embeddings']
random_emb = random_data['embeddings']
large_emb = large_data['embeddings']

print(f"✓ Loaded 3 embedding files")
print(f"  - Si: {si_emb.shape}")
print(f"  - Random: {random_emb.shape}")
print(f"  - Large: {large_emb.shape}")

# Analyze embedding statistics
print("\n[2] Embedding statistics:")
print("\nSi diamond structure:")
print(f"  Mean: {si_emb.mean():.6f}")
print(f"  Std: {si_emb.std():.6f}")
print(f"  Min: {si_emb.min():.6f}")
print(f"  Max: {si_emb.max():.6f}")
print(f"  Norm (per atom): {np.linalg.norm(si_emb, axis=1)}")

print("\nRandom C5 structure:")
print(f"  Mean: {random_emb.mean():.6f}")
print(f"  Std: {random_emb.std():.6f}")
print(f"  Min: {random_emb.min():.6f}")
print(f"  Max: {random_emb.max():.6f}")
print(f"  Norm (per atom): {np.linalg.norm(random_emb, axis=1)}")

print("\nLarge Si structure:")
print(f"  Mean: {large_emb.mean():.6f}")
print(f"  Std: {large_emb.std():.6f}")
print(f"  Min: {large_emb.min():.6f}")
print(f"  Max: {large_emb.max():.6f}")
print(f"  Norm (mean): {np.linalg.norm(large_emb, axis=1).mean():.6f}")
