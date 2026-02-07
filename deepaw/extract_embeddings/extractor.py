"""
Atomic Embedding Extractor

Extract atomic embeddings from crystal structures using pretrained F_nonlocal model.
The embeddings are 992-dimensional vectors that encode electron cloud shape information
through spherical harmonics (s, p, d, f, g orbitals).
"""

import os
import sys
import torch
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path

# ASE imports for structure handling
from ase import Atoms
from ase.io import read as ase_read

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

# DeePAW imports
from deepaw.models.f_nonlocal import AtomicConfigurationModel
from deepaw.config import get_model_config, get_checkpoint_path, get_device


class AtomicEmbeddingExtractor:
    """
    Extract atomic embeddings from crystal structures.

    This class loads the pretrained AtomicConfigurationModel (the first part of F_nonlocal)
    and extracts 992-dimensional atomic embeddings that encode electron cloud information.

    Features:
    - Input: Crystal structure (ASE Atoms, CIF, POSCAR, etc.)
    - Output: 992-dim embeddings per atom
    - Supports batch processing
    - GPU acceleration

    Example:
        >>> extractor = AtomicEmbeddingExtractor()
        >>> atoms = read('structure.cif')
        >>> embeddings = extractor.extract(atoms)
        >>> print(embeddings.shape)  # (n_atoms, 992)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        **model_kwargs
    ):
        """
        Initialize the embedding extractor.

        Args:
            checkpoint_path: Path to pretrained model checkpoint.
                           If None, uses default from config.
            device: Device to run on ('cuda' or 'cpu').
                   If None, auto-detects.
            **model_kwargs: Additional arguments for model configuration.
        """
        # Set device
        self.device = device if device else get_device()
        print(f"Using device: {self.device}")

        # Get model configuration
        config = get_model_config('f_nonlocal')
        config.update(model_kwargs)

        # Initialize model (only AtomicConfigurationModel part)
        self.model = AtomicConfigurationModel(
            num_interactions=config['num_interactions'],
            num_neighbors=config['num_neighbors'],
            mul=config['mul'],
            lmax=config['lmax'],
            cutoff=config['cutoff'],
            basis=config['basis'],
            num_basis=config['num_basis']
        ).to(self.device)

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path('f_nonlocal')
            checkpoint_path = os.path.join(deepaw_root, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please ensure the pretrained model is available."
            )

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load only the AtomicConfigurationModel weights
        model_state = {}
        for key, value in checkpoint.items():
            if key.startswith('atomic_configuration_model.'):
                new_key = key.replace('atomic_configuration_model.', '')
                model_state[new_key] = value

        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        print("Model loaded successfully!")
        print(f"Configuration: cutoff={config['cutoff']}Å, lmax={config['lmax']}, "
              f"neighbors={config['num_neighbors']}")

        # Store configuration
        self.cutoff = config['cutoff']
        self.num_neighbors = config['num_neighbors']

    def _build_atom_graph(self, atoms: Atoms) -> Dict[str, torch.Tensor]:
        """
        Build graph structure from atoms (only atom-atom edges, no probes).

        Args:
            atoms: ASE Atoms object

        Returns:
            Dictionary containing graph data in batch format
        """
        from ase.neighborlist import neighbor_list

        # Get atomic numbers and positions
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Build neighbor list
        edge_src, edge_dst, edge_shift = neighbor_list(
            'ijS', atoms, self.cutoff, self_interaction=False
        )

        # Convert to torch tensors with batch dimension
        # The model expects batch format even for single structure
        graph_dict = {
            'nodes': torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(0),  # (1, n_atoms)
            # Note: Model expects edges[:, 0] = neighbor, edges[:, 1] = center
            # ASE returns: edge_src = i (center), edge_dst = j (neighbor)
            # So we swap: edges[:, 0] = edge_dst (neighbor), edges[:, 1] = edge_src (center)
            'atom_edges': torch.stack([
                torch.tensor(edge_dst, dtype=torch.long),  # neighbor (j)
                torch.tensor(edge_src, dtype=torch.long)   # center (i)
            ], dim=1).unsqueeze(0),  # (1, n_edges, 2)
            'atom_edges_displacement': torch.tensor(edge_shift, dtype=torch.float32).unsqueeze(0),  # (1, n_edges, 3)
            'atom_xyz': torch.tensor(positions, dtype=torch.float32).unsqueeze(0),  # (1, n_atoms, 3)
            'cell': torch.tensor(np.array(cell), dtype=torch.float32).unsqueeze(0),  # (1, 3, 3)
            'num_nodes': torch.tensor([len(atoms)], dtype=torch.long),  # (1,)
            'num_atom_edges': torch.tensor([len(edge_src)], dtype=torch.long),  # (1,)
        }

        return graph_dict

    def extract(self, atoms: Atoms, return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract atomic embeddings from a crystal structure.

        Args:
            atoms: ASE Atoms object representing the crystal structure
            return_numpy: If True, return numpy array; if False, return torch tensor

        Returns:
            Atomic embeddings of shape (n_atoms, 992)
            Each row is the 992-dimensional embedding for one atom
        """
        # Build graph structure
        graph_dict = self._build_atom_graph(atoms)

        # Move to device
        for key in graph_dict:
            if isinstance(graph_dict[key], torch.Tensor):
                graph_dict[key] = graph_dict[key].to(self.device)

        # Extract embeddings
        with torch.no_grad():
            nodes_list = self.model(graph_dict)

            # nodes_list is a list of node features from each layer
            # Use the last layer's output as the atomic embeddings
            embeddings = nodes_list[-1]  # Shape: (n_atoms, embedding_dim)

        # Return as numpy or torch
        if return_numpy:
            return embeddings.cpu().numpy()
        else:
            return embeddings

    def extract_from_file(
        self,
        filepath: Union[str, Path],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract atomic embeddings from a structure file.

        Args:
            filepath: Path to structure file (CIF, POSCAR, xyz, etc.)
            return_numpy: If True, return numpy array; if False, return torch tensor

        Returns:
            Atomic embeddings of shape (n_atoms, 992)
        """
        atoms = ase_read(str(filepath))
        return self.extract(atoms, return_numpy=return_numpy)

    def extract_batch(
        self,
        atoms_list: List[Atoms],
        return_numpy: bool = True,
        show_progress: bool = True
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Extract atomic embeddings from multiple structures.

        Args:
            atoms_list: List of ASE Atoms objects
            return_numpy: If True, return numpy arrays; if False, return torch tensors
            show_progress: If True, show progress bar

        Returns:
            List of atomic embeddings, one per structure
        """
        embeddings_list = []

        iterator = atoms_list
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(atoms_list, desc="Extracting embeddings")
            except ImportError:
                print("tqdm not installed, progress bar disabled")

        for atoms in iterator:
            embeddings = self.extract(atoms, return_numpy=return_numpy)
            embeddings_list.append(embeddings)

        return embeddings_list

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        atoms: Atoms,
        output_path: Union[str, Path],
        format: str = 'npz'
    ):
        """
        Save atomic embeddings to file.

        Args:
            embeddings: Atomic embeddings array (n_atoms, 992)
            atoms: Original ASE Atoms object
            output_path: Path to save file
            format: Save format ('npz', 'npy', or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'npz':
            # Save with metadata
            np.savez(
                output_path,
                embeddings=embeddings,
                atomic_numbers=atoms.get_atomic_numbers(),
                positions=atoms.get_positions(),
                cell=np.array(atoms.get_cell()),
                pbc=atoms.get_pbc()
            )
        elif format == 'npy':
            # Save only embeddings
            np.save(output_path, embeddings)
        elif format == 'json':
            # Save as JSON (for interoperability)
            import json
            data = {
                'embeddings': embeddings.tolist(),
                'atomic_numbers': atoms.get_atomic_numbers().tolist(),
                'positions': atoms.get_positions().tolist(),
                'n_atoms': len(atoms)
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Embeddings saved to: {output_path}")

    def get_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics of embeddings.

        Args:
            embeddings: Atomic embeddings array (n_atoms, 992)

        Returns:
            Dictionary with statistics
        """
        return {
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'norm_std': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }
