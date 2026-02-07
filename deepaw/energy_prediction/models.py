"""
Energy Prediction Head Models

These models take atomic embeddings from DeePAW's F_nonlocal model
and predict crystal energy (and optionally forces).
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Linear
from typing import Optional, Tuple


class EnergyHead(nn.Module):
    """
    E3-equivariant energy prediction head.

    Uses e3nn Linear to project equivariant atomic embeddings to scalar energy.
    This preserves rotational invariance of the energy prediction.

    Args:
        input_irreps: Input irreps string (default: from F_nonlocal output)
        hidden_dim: Hidden dimension for MLP after projection
    """

    def __init__(
        self,
        input_irreps: str = "62x0e+62x0o+20x1e+20x1o+12x2e+12x2o+8x3e+8x3o+6x4e+6x4o",
        hidden_dim: int = 64
    ):
        super().__init__()

        self.input_irreps = o3.Irreps(input_irreps)

        # E3-equivariant linear projection to scalar
        self.linear = Linear(self.input_irreps, "0e")

        # MLP for additional expressiveness
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        num_atoms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict total energy from atomic embeddings.

        Args:
            atom_embeddings: (N_total_atoms, 992) atomic embeddings
            batch_idx: (N_total_atoms,) batch index for each atom
            num_atoms: (batch_size,) number of atoms per structure

        Returns:
            energy: (batch_size,) predicted energy per structure
        """
        # Project to scalar
        atom_energy = self.linear(atom_embeddings)  # (N, 1)
        atom_energy = self.mlp(atom_energy)  # (N, 1)

        # Sum over atoms in each structure
        if batch_idx is not None:
            # Use scatter_add for batched data
            batch_size = batch_idx.max().item() + 1
            energy = torch.zeros(batch_size, device=atom_energy.device)
            energy.scatter_add_(0, batch_idx, atom_energy.squeeze(-1))
        elif num_atoms is not None:
            # Split and sum
            atom_energy_list = torch.split(atom_energy.squeeze(-1), num_atoms.tolist())
            energy = torch.stack([e.sum() for e in atom_energy_list])
        else:
            # Single structure
            energy = atom_energy.sum()

        return energy


class EnergyForceHead(nn.Module):
    """
    Energy and force prediction head.

    Predicts energy directly, and computes forces as negative gradient
    of energy with respect to atomic positions.

    Args:
        input_irreps: Input irreps string
        hidden_dim: Hidden dimension for energy MLP
    """

    def __init__(
        self,
        input_irreps: str = "62x0e+62x0o+20x1e+20x1o+12x2e+12x2o+8x3e+8x3o+6x4e+6x4o",
        hidden_dim: int = 64
    ):
        super().__init__()
        self.energy_head = EnergyHead(input_irreps, hidden_dim)

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        positions: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        num_atoms: Optional[torch.Tensor] = None,
        compute_force: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict energy and forces.

        Args:
            atom_embeddings: (N, 992) atomic embeddings
            positions: (N, 3) atomic positions (requires grad for force)
            batch_idx: (N,) batch index for each atom
            num_atoms: (batch_size,) number of atoms per structure
            compute_force: Whether to compute forces

        Returns:
            energy: (batch_size,) predicted energy
            forces: (N, 3) predicted forces (or None)
        """
        if compute_force:
            positions.requires_grad_(True)

        energy = self.energy_head(atom_embeddings, batch_idx, num_atoms)

        forces = None
        if compute_force:
            # Force = -dE/dr
            grad = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=self.training,
                retain_graph=True
            )[0]
            forces = -grad

        return energy, forces


class ScalarEnergyHead(nn.Module):
    """
    Simple MLP-based energy head using only scalar (L=0) features.

    This is faster and simpler than the full equivariant head.
    Only uses the L=0 components of the atomic embeddings.

    Args:
        scalar_dim: Dimension of scalar features (L=0 components)
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(
        self,
        scalar_dim: int = 124,  # 62x0e + 62x0o = 124 scalar features
        hidden_dims: list = [128, 64, 32]
    ):
        super().__init__()
        self.scalar_dim = scalar_dim

        # Build MLP
        layers = []
        in_dim = scalar_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.SiLU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        atom_embeddings: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        num_atoms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict total energy from atomic embeddings.

        Args:
            atom_embeddings: (N_total_atoms, 992) atomic embeddings
            batch_idx: (N_total_atoms,) batch index for each atom
            num_atoms: (batch_size,) number of atoms per structure

        Returns:
            energy: (batch_size,) predicted energy per structure
        """
        # Extract only scalar features (first scalar_dim dimensions)
        scalar_feat = atom_embeddings[:, :self.scalar_dim]

        # Predict atomic energy
        atom_energy = self.mlp(scalar_feat)  # (N, 1)

        # Sum over atoms
        if batch_idx is not None:
            batch_size = batch_idx.max().item() + 1
            energy = torch.zeros(batch_size, device=atom_energy.device)
            energy.scatter_add_(0, batch_idx, atom_energy.squeeze(-1))
        elif num_atoms is not None:
            atom_energy_list = torch.split(atom_energy.squeeze(-1), num_atoms.tolist())
            energy = torch.stack([e.sum() for e in atom_energy_list])
        else:
            energy = atom_energy.sum()

        return energy
