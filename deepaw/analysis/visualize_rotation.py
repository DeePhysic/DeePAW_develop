#!/usr/bin/env python
"""Visualize rotation equivariance of embeddings."""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase import Atoms

print("=" * 70)
print("Visualizing Rotation Equivariance")
print("=" * 70)

extractor = AtomicEmbeddingExtractor()

# Create asymmetric structure
positions = np.array([
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.5],
    [1.0, 1.0, 1.0],
])
atoms_orig = Atoms('Si5', positions=positions, cell=[10, 10, 10], pbc=True)

# Multiple rotations
angles = [0, 30, 60, 90, 120, 150, 180]

# Collect embeddings for different rotations
L0_values = []  # L=0e component (should be constant)
L1_vectors = []  # L=1o first multiplicity (should rotate)

print("\nExtracting embeddings for different rotations...")
for angle in angles:
    rot = Rotation.from_euler('z', angle, degrees=True)
    R = rot.as_matrix()
    pos_rot = (R @ positions.T).T
    atoms_rot = Atoms('Si5', positions=pos_rot, cell=[10, 10, 10], pbc=True)

    emb = extractor.extract(atoms_rot)
    atom0_emb = emb[0]

    # L=0e: indices 100:200
    L0e = atom0_emb[100:200]
    L0_values.append(np.linalg.norm(L0e))

    # L=1o: indices 200:299, reshape to (33, 3)
    L1o = atom0_emb[200:299].reshape(33, 3)
    L1_vectors.append(L1o[0])  # First multiplicity

    print(f"  {angle}°: L0e_norm={L0_values[-1]:.4f}, L1o[0]={L1_vectors[-1].round(4)}")

L1_vectors = np.array(L1_vectors)

# Create visualization
fig = plt.figure(figsize=(16, 5))

# Panel 1: L=0 invariance
ax1 = fig.add_subplot(131)
ax1.bar(range(len(angles)), L0_values, color='steelblue', edgecolor='black')
ax1.set_xticks(range(len(angles)))
ax1.set_xticklabels([f'{a}°' for a in angles])
ax1.set_xlabel('Rotation Angle', fontsize=12)
ax1.set_ylabel('L=0e Norm', fontsize=12)
ax1.set_title('L=0 (Scalar): INVARIANT', fontsize=14, fontweight='bold')
ax1.set_ylim([0, max(L0_values) * 1.2])
ax1.axhline(y=L0_values[0], color='red', linestyle='--', label='Original')
ax1.legend()

# Panel 2: L=1 vector rotation (2D projection)
ax2 = fig.add_subplot(132)
colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))
for i, (angle, vec, c) in enumerate(zip(angles, L1_vectors, colors)):
    # Plot m=-1 vs m=0 (x-y plane in spherical harmonics)
    ax2.arrow(0, 0, vec[0]*50, vec[1]*50, head_width=0.3, head_length=0.2,
              fc=c, ec=c, alpha=0.8, linewidth=2)
    ax2.scatter([vec[0]*50], [vec[1]*50], c=[c], s=100, edgecolors='black', zorder=5)
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_xlabel('m=-1 component', fontsize=12)
ax2.set_ylabel('m=0 component', fontsize=12)
ax2.set_title('L=1 Vector: ROTATES', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 180))
cbar = plt.colorbar(sm, ax=ax2)
cbar.set_label('Rotation Angle (°)')

# Panel 3: L=1 norm (should be constant)
ax3 = fig.add_subplot(133)
L1_norms = np.linalg.norm(L1_vectors, axis=1)
ax3.bar(range(len(angles)), L1_norms, color='coral', edgecolor='black')
ax3.set_xticks(range(len(angles)))
ax3.set_xticklabels([f'{a}°' for a in angles])
ax3.set_xlabel('Rotation Angle', fontsize=12)
ax3.set_ylabel('L=1o[0] Norm', fontsize=12)
ax3.set_title('L=1 Norm: INVARIANT', fontsize=14, fontweight='bold')
ax3.set_ylim([0, max(L1_norms) * 1.2])
ax3.axhline(y=L1_norms[0], color='red', linestyle='--', label='Original')
ax3.legend()

plt.suptitle('E3 Equivariance: L=0 Invariant, L=1 Equivariant',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

# Save
output_dir = os.path.join(script_dir, 'defect_visualization')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'rotation_equivariance.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")
plt.close()

print("\n" + "=" * 70)
print("Visualization complete!")
print("=" * 70)
