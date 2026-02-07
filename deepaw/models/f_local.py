"""
F_local: Local charge density correction model using KAN

This module implements the local correction model that refines the predictions
from the non-local model (F_nonlocal) using Kolmogorov-Arnold Networks (KAN).
"""

from torch import nn
from kan import KAN


class F_local(nn.Module):
    """
    F_local: Local correction model using KAN
    
    This model learns local corrections to the charge density predictions
    from F_nonlocal using a combination of MLP and KAN networks.
    
    The architecture consists of:
    1. MLP layer: Projects node representations to lower dimension
    2. KAN network: Learns non-linear corrections
    
    Args:
        input_dim (int): Input dimension (default: 992)
        hidden_dim (int): Hidden dimension after MLP (default: 24)
        kan_width (list): KAN network width (default: [24, 12, 6, 1])
        kan_grid (int): KAN grid size (default: 8)
        kan_k (int): KAN spline order (default: 4)
        seed (int): Random seed for KAN initialization (default: 42)
    """
    
    def __init__(
        self,
        input_dim=992,
        hidden_dim=32,
        kan_width=None,
        kan_grid=8,
        kan_k=4,
        seed=42
    ):
        super(F_local, self).__init__()

        if kan_width is None:
            kan_width = [hidden_dim, 6, 1]
        
        # MLP projection layer
        self.mlp = nn.Linear(input_dim, hidden_dim)
        
        # KAN network for non-linear correction
        self.kan = KAN(width=kan_width, grid=kan_grid, k=kan_k, seed=seed)
        
        # Initialize MLP weights to zero for stable training
        nn.init.constant_(self.mlp.weight, 0) 
        nn.init.constant_(self.mlp.bias, 0)
    
    def forward(self, input_dict, node_rep):
        """
        Forward pass
        
        Args:
            input_dict: Input dictionary (not used, kept for compatibility)
            node_rep: Node representations from F_nonlocal (shape: [N, input_dim])
        
        Returns:
            correction: Local corrections (shape: [N, 1])
            None: Placeholder for compatibility
        """
        # Project to hidden dimension
        hidden = self.mlp(node_rep)
        
        # Apply KAN for non-linear correction
        correction = self.kan(hidden)
        
        return correction, None


# Legacy alias for backward compatibility
ResidualCorrectionModel = F_local

