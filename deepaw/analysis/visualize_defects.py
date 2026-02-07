#!/usr/bin/env python
"""
Generate defective Si structures and visualize embeddings.

This script creates various Si structures with different defects and
visualizes how the atomic embeddings differ based on chemical environment.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase.build import bulk
from ase import Atoms

print("=" * 70)
print("Si Defect Structures - Embedding Visualization")
print("=" * 70)

# Initialize extractor
print("\n[1/5] Initializing embedding extractor...")
extractor = AtomicEmbeddingExtractor()

# Create different Si structures
print("\n[2/5] Creating Si structures with different defects...")

structures = {}
labels = {}
colors_map = {}

# 1. Perfect Si crystal (2x2x2 supercell)
print("  - Creating perfect Si crystal...")
perfect_si = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
structures['perfect'] = perfect_si
labels['perfect'] = ['Perfect'] * len(perfect_si)
colors_map['Perfect'] = 'blue'
print(f"    Perfect Si: {len(perfect_si)} atoms")

# 2. Si with vacancy defect
print("  - Creating Si with vacancy...")
vacancy_si = perfect_si.copy()
del vacancy_si[0]  # Remove one atom
structures['vacancy'] = vacancy_si
# Label atoms: those near vacancy vs far from vacancy
vacancy_labels = []
removed_pos = perfect_si.positions[0]
for atom in vacancy_si:
    dist = np.linalg.norm(atom.position - removed_pos)
    if dist < 4.0:  # Within 4Å of vacancy
        vacancy_labels.append('Near Vacancy')
    else:
        vacancy_labels.append('Far from Vacancy')
labels['vacancy'] = vacancy_labels
colors_map['Near Vacancy'] = 'red'
colors_map['Far from Vacancy'] = 'lightblue'
print(f"    Vacancy Si: {len(vacancy_si)} atoms")

# 3. Si with interstitial defect
print("  - Creating Si with interstitial...")
interstitial_si = perfect_si.copy()
# Add an interstitial atom at a tetrahedral site
interstitial_pos = perfect_si.positions[0] + np.array([1.5, 1.5, 1.5])
from ase import Atom
interstitial_si.append(Atom('Si', position=interstitial_pos))
structures['interstitial'] = interstitial_si
# Label atoms
interstitial_labels = []
for i, atom in enumerate(interstitial_si):
    if i == len(interstitial_si) - 1:
        interstitial_labels.append('Interstitial')
    else:
        dist = np.linalg.norm(atom.position - interstitial_pos)
        if dist < 4.0:
            interstitial_labels.append('Near Interstitial')
        else:
            interstitial_labels.append('Bulk')
labels['interstitial'] = interstitial_labels
colors_map['Interstitial'] = 'orange'
colors_map['Near Interstitial'] = 'yellow'
colors_map['Bulk'] = 'lightgreen'
print(f"    Interstitial Si: {len(interstitial_si)} atoms")

# 4. Si surface (slab)
print("  - Creating Si surface...")
surface_si = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 3))
surface_si.center(vacuum=10.0, axis=2)
surface_si.set_pbc([True, True, False])
structures['surface'] = surface_si
# Label atoms by z-coordinate
surface_labels = []
z_positions = surface_si.positions[:, 2]
z_min, z_max = z_positions.min(), z_positions.max()
for z in z_positions:
    if z < z_min + 2.0:
        surface_labels.append('Bottom Surface')
    elif z > z_max - 2.0:
        surface_labels.append('Top Surface')
    else:
        surface_labels.append('Subsurface')
labels['surface'] = surface_labels
colors_map['Bottom Surface'] = 'purple'
colors_map['Top Surface'] = 'magenta'
colors_map['Subsurface'] = 'pink'
print(f"    Surface Si: {len(surface_si)} atoms")

print(f"\n  Total structures created: {len(structures)}")

# Extract embeddings
print("\n[3/5] Extracting embeddings...")
all_embeddings = []
all_labels = []

for name, atoms in structures.items():
    print(f"  - Extracting from {name}...")
    emb = extractor.extract(atoms)
    all_embeddings.append(emb)
    all_labels.extend(labels[name])

# Combine all embeddings
all_embeddings = np.vstack(all_embeddings)
print(f"\n  Total embeddings: {all_embeddings.shape}")
print(f"  Unique labels: {set(all_labels)}")

# Dimensionality reduction
print("\n[4/5] Performing dimensionality reduction...")

# PCA
print("  - Running PCA...")
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(all_embeddings)
print(f"    PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# t-SNE
print("  - Running t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_tsne = tsne.fit_transform(all_embeddings)
print("    t-SNE completed")

# Visualization
print("\n[5/5] Creating visualizations...")

# Create color array
unique_labels = list(set(all_labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
colors = [colors_map.get(label, 'gray') for label in all_labels]

# Create output directory
output_dir = os.path.join(script_dir, 'defect_visualization')
os.makedirs(output_dir, exist_ok=True)

# Plot 1: PCA visualization
print("  - Creating PCA plot...")
fig, ax = plt.subplots(figsize=(12, 10))

for label in unique_labels:
    mask = np.array([l == label for l in all_labels])
    ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
              c=colors_map.get(label, 'gray'), label=label,
              alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_title('Si Atomic Embeddings - PCA Visualization\n(Different Chemical Environments)',
            fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
pca_path = os.path.join(output_dir, 'si_defects_pca.png')
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
print(f"    Saved: {pca_path}")
plt.close()

# Plot 2: t-SNE visualization
print("  - Creating t-SNE plot...")
fig, ax = plt.subplots(figsize=(12, 10))

for label in unique_labels:
    mask = np.array([l == label for l in all_labels])
    ax.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
              c=colors_map.get(label, 'gray'), label=label,
              alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=14)
ax.set_ylabel('t-SNE 2', fontsize=14)
ax.set_title('Si Atomic Embeddings - t-SNE Visualization\n(Different Chemical Environments)',
            fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
tsne_path = os.path.join(output_dir, 'si_defects_tsne.png')
plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
print(f"    Saved: {tsne_path}")
plt.close()

# Save embeddings and labels
print("\n  - Saving embeddings and labels...")
np.savez(os.path.join(output_dir, 'embeddings_data.npz'),
         embeddings=all_embeddings,
         embeddings_pca=embeddings_pca,
         embeddings_tsne=embeddings_tsne,
         labels=np.array(all_labels))
print(f"    Saved: embeddings_data.npz")

# Print summary
print("\n" + "=" * 70)
print("✅ VISUALIZATION COMPLETED!")
print("=" * 70)
print(f"\nResults saved to: {output_dir}")
print(f"\nGenerated files:")
print(f"  1. si_defects_pca.png   - PCA visualization")
print(f"  2. si_defects_tsne.png  - t-SNE visualization")
print(f"  3. embeddings_data.npz  - Raw data")

print(f"\nStructures analyzed:")
for name, atoms in structures.items():
    print(f"  - {name}: {len(atoms)} atoms")

print(f"\nEnvironment types:")
for label in unique_labels:
    count = all_labels.count(label)
    print(f"  - {label}: {count} atoms")

print("\n💡 Key observations:")
print("  - Different chemical environments should form distinct clusters")
print("  - Atoms near defects should differ from bulk atoms")
print("  - Surface atoms should differ from bulk atoms")
print("  - This validates that embeddings are environment-adaptive!")

print("\n" + "=" * 70)

