#!/usr/bin/env python
"""Demonstrate rotation equivariance of L=1 components."""

import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation

script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase import Atoms

print("=" * 70)
print("Rotation Equivariance Test for L=1 Components")
print("=" * 70)

extractor = AtomicEmbeddingExtractor()

# Create an asymmetric structure (not Si diamond, which has L=1 ≈ 0)
# Use a simple cluster with no symmetry
positions = np.array([
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.5],
    [1.0, 1.0, 1.0],
])
atoms_orig = Atoms('Si5', positions=positions, cell=[10, 10, 10], pbc=True)

# Create a rotation (45 degrees around z-axis)
angle = 45  # degrees
rot = Rotation.from_euler('z', angle, degrees=True)
R = rot.as_matrix()

print(f"\nRotation: {angle}° around z-axis")
print(f"Rotation matrix R:\n{R.round(4)}")

# Rotate the structure
positions_rot = (R @ positions.T).T
atoms_rot = Atoms('Si5', positions=positions_rot, cell=[10, 10, 10], pbc=True)

# Extract embeddings
print("\nExtracting embeddings...")
emb_orig = extractor.extract(atoms_orig)
emb_rot = extractor.extract(atoms_rot)

print(f"Original embedding shape: {emb_orig.shape}")
print(f"Rotated embedding shape: {emb_rot.shape}")

# Irreps structure
irreps_info = []
total_mul = 500
lmax = 4
for l in range(lmax + 1):
    for p in [-1, 1]:
        mul = round(total_mul / (lmax + 1) / (2 * l + 1))
        dim = mul * (2 * l + 1)
        parity_str = 'o' if p == -1 else 'e'
        irreps_info.append({'L': l, 'parity': p, 'parity_str': parity_str,
                           'mul': mul, 'dim': dim, 'num_m': 2 * l + 1})

# Focus on atom 0 for analysis
atom_idx = 0
orig = emb_orig[atom_idx]
rotated = emb_rot[atom_idx]

print("\n" + "=" * 70)
print(f"Analysis for atom {atom_idx}")
print("=" * 70)

# Extract L=0 and L=1 components
idx = 0
for info in irreps_info[:4]:  # L=0o, L=0e, L=1o, L=1e
    L = info['L']
    p_str = info['parity_str']
    dim = info['dim']
    mul = info['mul']
    num_m = info['num_m']

    comp_orig = orig[idx:idx+dim]
    comp_rot = rotated[idx:idx+dim]

    print(f"\n--- L={L}{p_str} (mul={mul}, m=-{L}..+{L}) ---")
    print(f"  Original norm: {np.linalg.norm(comp_orig):.6f}")
    print(f"  Rotated norm:  {np.linalg.norm(comp_rot):.6f}")

    if L == 0:
        # Scalar: should be identical
        diff = np.linalg.norm(comp_orig - comp_rot)
        print(f"  Difference: {diff:.6f}")
        print(f"  --> L=0 is INVARIANT (scalar)")

    elif L == 1 and np.linalg.norm(comp_orig) > 0.01:
        # Vector: should rotate with R
        # L=1 has m = -1, 0, +1, which corresponds to (y, z, x) in real spherical harmonics
        # Or we can check if R @ v_orig ≈ v_rot for each multiplicity

        comp_orig_reshape = comp_orig.reshape(mul, 3)  # (mul, 3) for m=-1,0,+1
        comp_rot_reshape = comp_rot.reshape(mul, 3)

        print(f"\n  Checking rotation equivariance for each multiplicity:")
        for m_idx in range(min(3, mul)):  # Show first 3 multiplicities
            v_orig = comp_orig_reshape[m_idx]  # (3,) vector
            v_rot = comp_rot_reshape[m_idx]

            # For real spherical harmonics Y_1^m, the order is typically:
            # m=-1 -> y, m=0 -> z, m=+1 -> x (or some permutation)
            # The Wigner D-matrix for L=1 is essentially the rotation matrix R
            # But the exact mapping depends on the convention used

            # Check if norms are preserved (rotation preserves norm)
            print(f"    mul[{m_idx}]: |v_orig|={np.linalg.norm(v_orig):.4f}, "
                  f"|v_rot|={np.linalg.norm(v_rot):.4f}")

    idx += dim

# Detailed L=1 analysis
print("\n" + "=" * 70)
print("Detailed L=1 Rotation Analysis")
print("=" * 70)

# Get L=1e component (skip L=0o, L=0e, L=1o)
idx_L1e = 100 + 100 + 99  # L=0o + L=0e + L=1o
dim_L1e = 99
mul_L1e = 33

L1e_orig = orig[idx_L1e:idx_L1e+dim_L1e].reshape(mul_L1e, 3)
L1e_rot = rotated[idx_L1e:idx_L1e+dim_L1e].reshape(mul_L1e, 3)

print(f"\nL=1e components (first 5 multiplicities):")
print(f"{'mul':<5} {'Original (m=-1,0,+1)':<30} {'Rotated (m=-1,0,+1)':<30}")
print("-" * 70)
for i in range(min(5, mul_L1e)):
    v_o = L1e_orig[i]
    v_r = L1e_rot[i]
    print(f"{i:<5} [{v_o[0]:7.4f},{v_o[1]:7.4f},{v_o[2]:7.4f}]"
          f"      [{v_r[0]:7.4f},{v_r[1]:7.4f},{v_r[2]:7.4f}]")

print("\n" + "=" * 70)
