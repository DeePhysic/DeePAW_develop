"""
Dataset for Energy Prediction

Handles loading crystal structures and their energies for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

from ase import Atoms
from ase.db import connect


class EnergyDataset(Dataset):
    """
    Dataset for crystal energy prediction.

    Loads structures and energies from ASE database.
    Extracts embeddings on-the-fly or uses pre-computed embeddings.
    """

    def __init__(
        self,
        db_path: str,
        extractor=None,
        energy_key: str = 'formation_energy',
        precompute_embeddings: bool = False,
        device: str = 'cpu'
    ):
        """
        Args:
            db_path: Path to ASE database
            extractor: AtomicEmbeddingExtractor instance
            energy_key: Key for energy in database
            precompute_embeddings: Whether to precompute all embeddings
            device: Device for embeddings
        """
        self.db_path = db_path
        self.extractor = extractor
        self.energy_key = energy_key
        self.device = device

        # Load database info
        self.db = connect(db_path)
        self.num_samples = len(self.db)

        # Cache for embeddings
        self.embeddings_cache = {}
        if precompute_embeddings and extractor is not None:
            self._precompute_all_embeddings()

    def _precompute_all_embeddings(self):
        """Precompute embeddings for all structures."""
        print("Precomputing embeddings...")
        from tqdm import tqdm
        for idx in tqdm(range(self.num_samples)):
            row = self.db.get(idx + 1)
            atoms = row.toatoms()
            emb = self.extractor.extract(atoms, return_numpy=False)
            self.embeddings_cache[idx] = emb.cpu()
        print("Done!")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with 'embeddings', 'energy', 'num_atoms'
        """
        # Get from cache or compute
        if idx in self.embeddings_cache:
            embeddings = self.embeddings_cache[idx]
        else:
            row = self.db.get(idx + 1)
            atoms = row.toatoms()
            embeddings = self.extractor.extract(atoms, return_numpy=False)

        # Get energy
        row = self.db.get(idx + 1)
        energy = getattr(row, self.energy_key, 0.0)

        return {
            'embeddings': embeddings,
            'energy': torch.tensor(energy, dtype=torch.float32),
            'num_atoms': torch.tensor(embeddings.shape[0], dtype=torch.long)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching.

    Concatenates embeddings and creates batch indices.
    """
    embeddings = torch.cat([b['embeddings'] for b in batch], dim=0)
    energies = torch.stack([b['energy'] for b in batch])
    num_atoms = torch.stack([b['num_atoms'] for b in batch])

    # Create batch index
    batch_idx = torch.cat([
        torch.full((n.item(),), i, dtype=torch.long)
        for i, n in enumerate(num_atoms)
    ])

    return {
        'embeddings': embeddings,
        'energy': energies,
        'num_atoms': num_atoms,
        'batch_idx': batch_idx
    }
