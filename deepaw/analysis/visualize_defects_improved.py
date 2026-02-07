#!/usr/bin/env python
"""
Improved visualization of Si defect embeddings.
Three approaches combined in one figure:
- Panel A: Faceted by structure type (4 subplots)
- Panel B: Two-level comparison (simplified vs detailed)
- Panel C: UMAP visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

from deepaw.extract_embeddings import AtomicEmbeddingExtractor
from ase.build import bulk
from ase import Atom

# Try to import UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    print("Warning: umap-learn not installed. Using t-SNE for Panel C.")
    HAS_UMAP = False

print("=" * 70)
print("Improved Si Defect Visualization")
print("=" * 70)

# ============================================================
# Step 1: Create structures and extract embeddings
# ============================================================
print("\n[1/4] Creating structures...")

extractor = AtomicEmbeddingExtractor()

structures = {}
structure_labels = {}  # Which structure each atom belongs to
env_labels = {}        # Detailed environment label
simple_labels = {}     # Simplified label (Bulk/Defect/Surface)

# 1. Perfect Si
print("  - Perfect Si...")
perfect_si = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 2))
structures['Perfect'] = perfect_si
structure_labels['Perfect'] = ['Perfect'] * len(perfect_si)
env_labels['Perfect'] = ['Perfect Crystal'] * len(perfect_si)
simple_labels['Perfect'] = ['Bulk'] * len(perfect_si)

# 2. Vacancy Si
print("  - Vacancy Si...")
vacancy_si = perfect_si.copy()
removed_pos = perfect_si.positions[0].copy()
del vacancy_si[0]
structures['Vacancy'] = vacancy_si

vac_struct = []
vac_env = []
vac_simple = []
for atom in vacancy_si:
    dist = np.linalg.norm(atom.position - removed_pos)
    vac_struct.append('Vacancy')
    if dist < 4.0:
        vac_env.append('Near Vacancy')
        vac_simple.append('Near Defect')
    else:
        vac_env.append('Far from Vacancy')
        vac_simple.append('Bulk')
structure_labels['Vacancy'] = vac_struct
env_labels['Vacancy'] = vac_env
simple_labels['Vacancy'] = vac_simple

# 3. Interstitial Si
print("  - Interstitial Si...")
interstitial_si = perfect_si.copy()
interstitial_pos = perfect_si.positions[0] + np.array([1.5, 1.5, 1.5])
interstitial_si.append(Atom('Si', position=interstitial_pos))
structures['Interstitial'] = interstitial_si

int_struct = []
int_env = []
int_simple = []
for i, atom in enumerate(interstitial_si):
    int_struct.append('Interstitial')
    if i == len(interstitial_si) - 1:
        int_env.append('Interstitial Atom')
        int_simple.append('Defect')
    else:
        dist = np.linalg.norm(atom.position - interstitial_pos)
        if dist < 4.0:
            int_env.append('Near Interstitial')
            int_simple.append('Near Defect')
        else:
            int_env.append('Bulk')
            int_simple.append('Bulk')
structure_labels['Interstitial'] = int_struct
env_labels['Interstitial'] = int_env
simple_labels['Interstitial'] = int_simple

# 4. Surface Si
print("  - Surface Si...")
surface_si = bulk('Si', 'diamond', a=5.43).repeat((2, 2, 3))
surface_si.center(vacuum=10.0, axis=2)
surface_si.set_pbc([True, True, False])
structures['Surface'] = surface_si

surf_struct = []
surf_env = []
surf_simple = []
z_positions = surface_si.positions[:, 2]
z_min, z_max = z_positions.min(), z_positions.max()
for z in z_positions:
    surf_struct.append('Surface')
    if z < z_min + 2.0 or z > z_max - 2.0:
        surf_env.append('Surface Atom')
        surf_simple.append('Surface')
    else:
        surf_env.append('Subsurface')
        surf_simple.append('Bulk')
structure_labels['Surface'] = surf_struct
env_labels['Surface'] = surf_env
simple_labels['Surface'] = surf_simple

# ============================================================
# Step 2: Extract embeddings
# ============================================================
print("\n[2/4] Extracting embeddings...")

all_embeddings = []
all_structure_labels = []
all_env_labels = []
all_simple_labels = []

for name in ['Perfect', 'Vacancy', 'Interstitial', 'Surface']:
    print(f"  - {name}...")
    emb = extractor.extract(structures[name])
    all_embeddings.append(emb)
    all_structure_labels.extend(structure_labels[name])
    all_env_labels.extend(env_labels[name])
    all_simple_labels.extend(simple_labels[name])

all_embeddings = np.vstack(all_embeddings)
print(f"  Total: {all_embeddings.shape[0]} atoms, {all_embeddings.shape[1]} dimensions")

# ============================================================
# Step 3: Dimensionality reduction
# ============================================================
print("\n[3/4] Dimensionality reduction...")

print("  - Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_tsne = tsne.fit_transform(all_embeddings)

if HAS_UMAP:
    print("  - Running UMAP...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    emb_umap = umap.fit_transform(all_embeddings)
else:
    print("  - Using t-SNE for UMAP panel (umap-learn not installed)")
    emb_umap = emb_tsne.copy()

# ============================================================
# Step 4: Create visualization
# ============================================================
print("\n[4/4] Creating visualization...")

# Color schemes
structure_colors = {
    'Perfect': '#3498db',
    'Vacancy': '#e74c3c',
    'Interstitial': '#f39c12',
    'Surface': '#9b59b6'
}

simple_colors = {
    'Bulk': '#3498db',
    'Near Defect': '#e74c3c',
    'Defect': '#c0392b',
    'Surface': '#9b59b6'
}

env_colors = {
    'Perfect Crystal': '#3498db',
    'Near Vacancy': '#e74c3c',
    'Far from Vacancy': '#85c1e9',
    'Interstitial Atom': '#c0392b',
    'Near Interstitial': '#f39c12',
    'Bulk': '#82e0aa',
    'Surface Atom': '#9b59b6',
    'Subsurface': '#d7bde2'
}

# Create figure with GridSpec
fig = plt.figure(figsize=(20, 16))

from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25)

# ============================================================
# Panel A: Faceted by structure type (top row, 4 subplots)
# ============================================================
ax_a1 = fig.add_subplot(gs[0, 0])
ax_a2 = fig.add_subplot(gs[0, 1])
ax_a3 = fig.add_subplot(gs[0, 2])
ax_a4 = fig.add_subplot(gs[0, 3])
axes_a = [ax_a1, ax_a2, ax_a3, ax_a4]
structure_names = ['Perfect', 'Vacancy', 'Interstitial', 'Surface']

for ax, struct_name in zip(axes_a, structure_names):
    # Get mask for this structure
    mask = np.array([s == struct_name for s in all_structure_labels])

    # Plot background (other structures in gray)
    ax.scatter(emb_tsne[~mask, 0], emb_tsne[~mask, 1],
               c='lightgray', alpha=0.3, s=30, label='Other')

    # Plot this structure with environment colors
    struct_env = np.array(all_env_labels)[mask]
    struct_emb = emb_tsne[mask]

    unique_envs = list(set(struct_env))
    for env in unique_envs:
        env_mask = struct_env == env
        ax.scatter(struct_emb[env_mask, 0], struct_emb[env_mask, 1],
                   c=env_colors.get(env, 'gray'), label=env,
                   alpha=0.8, s=60, edgecolors='black', linewidth=0.5)

    ax.set_title(f'{struct_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=7, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(0.5, 0.97, 'A. Faceted by Structure Type (t-SNE)',
         ha='center', fontsize=14, fontweight='bold')

# ============================================================
# Panel B: Two-level comparison (middle row)
# ============================================================
ax_b1 = fig.add_subplot(gs[1, 0:2])
ax_b2 = fig.add_subplot(gs[1, 2:4])

# B1: Simplified labels (3 categories)
for simple_label in ['Bulk', 'Near Defect', 'Defect', 'Surface']:
    mask = np.array([s == simple_label for s in all_simple_labels])
    if mask.sum() > 0:
        ax_b1.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                      c=simple_colors[simple_label], label=simple_label,
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.3)

ax_b1.set_title('Simplified: 3 Categories', fontsize=12, fontweight='bold')
ax_b1.legend(loc='best', fontsize=9, framealpha=0.9)
ax_b1.set_xlabel('t-SNE 1', fontsize=10)
ax_b1.set_ylabel('t-SNE 2', fontsize=10)
ax_b1.spines['top'].set_visible(False)
ax_b1.spines['right'].set_visible(False)

# B2: Detailed labels (all categories)
for env_label in env_colors.keys():
    mask = np.array([e == env_label for e in all_env_labels])
    if mask.sum() > 0:
        ax_b2.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1],
                      c=env_colors[env_label], label=env_label,
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.3)

ax_b2.set_title('Detailed: All Categories', fontsize=12, fontweight='bold')
ax_b2.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
ax_b2.set_xlabel('t-SNE 1', fontsize=10)
ax_b2.set_ylabel('t-SNE 2', fontsize=10)
ax_b2.spines['top'].set_visible(False)
ax_b2.spines['right'].set_visible(False)

fig.text(0.5, 0.64, 'B. Two-Level Comparison (t-SNE)',
         ha='center', fontsize=14, fontweight='bold')

# ============================================================
# Panel C: UMAP visualization (bottom row)
# ============================================================
ax_c1 = fig.add_subplot(gs[2, 0:2])
ax_c2 = fig.add_subplot(gs[2, 2:4])

# C1: UMAP with structure colors
for struct_name in structure_names:
    mask = np.array([s == struct_name for s in all_structure_labels])
    ax_c1.scatter(emb_umap[mask, 0], emb_umap[mask, 1],
                  c=structure_colors[struct_name], label=struct_name,
                  alpha=0.7, s=50, edgecolors='black', linewidth=0.3)

ax_c1.set_title('By Structure Type', fontsize=12, fontweight='bold')
ax_c1.legend(loc='best', fontsize=9, framealpha=0.9)
method_name = 'UMAP' if HAS_UMAP else 't-SNE'
ax_c1.set_xlabel(f'{method_name} 1', fontsize=10)
ax_c1.set_ylabel(f'{method_name} 2', fontsize=10)
ax_c1.spines['top'].set_visible(False)
ax_c1.spines['right'].set_visible(False)

# C2: UMAP with simplified colors
for simple_label in ['Bulk', 'Near Defect', 'Defect', 'Surface']:
    mask = np.array([s == simple_label for s in all_simple_labels])
    if mask.sum() > 0:
        ax_c2.scatter(emb_umap[mask, 0], emb_umap[mask, 1],
                      c=simple_colors[simple_label], label=simple_label,
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.3)

ax_c2.set_title('By Environment Type', fontsize=12, fontweight='bold')
ax_c2.legend(loc='best', fontsize=9, framealpha=0.9)
ax_c2.set_xlabel(f'{method_name} 1', fontsize=10)
ax_c2.set_ylabel(f'{method_name} 2', fontsize=10)
ax_c2.spines['top'].set_visible(False)
ax_c2.spines['right'].set_visible(False)

fig.text(0.5, 0.31, f'C. {method_name} Visualization',
         ha='center', fontsize=14, fontweight='bold')

# ============================================================
# Save figure
# ============================================================
output_dir = os.path.join(script_dir, 'defect_visualization')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'si_defects_improved.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")
plt.close()

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nPanel A: Faceted by structure type")
print("  - Each subplot shows one structure with others grayed out")
print("  - Easier to see environment differences within each structure")

print("\nPanel B: Two-level comparison")
print("  - Left: Simplified (Bulk / Near Defect / Defect / Surface)")
print("  - Right: Detailed (all 8 environment types)")

print(f"\nPanel C: {method_name} visualization")
print("  - Left: Colored by structure type")
print("  - Right: Colored by environment type")
if HAS_UMAP:
    print("  - UMAP often preserves global structure better than t-SNE")

print("\nAtom counts by category:")
for label in set(all_simple_labels):
    count = all_simple_labels.count(label)
    print(f"  - {label}: {count}")

print("\n" + "=" * 70)
