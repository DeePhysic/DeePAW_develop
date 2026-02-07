"""
Test script for energy prediction pipeline.
"""

from pathlib import Path

import torch
from ase.db import connect

# Test database loading
DB_PATH = str(Path(__file__).parent.parent.parent / 'fe_data' / 'cif_db(mp).db')

print("=" * 50)
print("Testing Energy Prediction Pipeline")
print("=" * 50)

# 1. Test database
print("\n1. Testing database...")
db = connect(DB_PATH)
print(f"   Total entries: {len(db)}")
row = db.get(1)
print(f"   Sample energy: {row.formation_energy:.4f}")
print(f"   Sample formula: {row.toatoms().get_chemical_formula()}")

# 2. Test embedding extractor
print("\n2. Testing embedding extractor...")
from deepaw.extract_embeddings import AtomicEmbeddingExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

extractor = AtomicEmbeddingExtractor(device=device)
atoms = row.toatoms()
emb = extractor.extract(atoms, return_numpy=False)
print(f"   Embedding shape: {emb.shape}")

# 3. Test model
print("\n3. Testing energy head model...")
from deepaw.energy_prediction.models import ScalarEnergyHead

model = ScalarEnergyHead().to(device)
emb_gpu = emb.to(device)
with torch.no_grad():
    pred = model(emb_gpu)
print(f"   Predicted energy: {pred.item():.4f}")
print(f"   Target energy: {row.formation_energy:.4f}")

# 4. Test dataset
print("\n4. Testing dataset...")
from deepaw.energy_prediction.dataset import EnergyDataset, collate_fn
from torch.utils.data import DataLoader

dataset = EnergyDataset(DB_PATH, extractor=extractor, precompute_embeddings=False)
print(f"   Dataset size: {len(dataset)}")

# Test single sample
sample = dataset[0]
print(f"   Sample embeddings shape: {sample['embeddings'].shape}")
print(f"   Sample energy: {sample['energy'].item():.4f}")

print("\n" + "=" * 50)
print("All tests passed!")
print("=" * 50)
