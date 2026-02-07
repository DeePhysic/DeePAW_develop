"""
DeePAW Configuration

This module contains default configuration parameters for DeePAW models and scripts.
All default values can be overridden when instantiating models or running scripts.
"""

# =============================================================================
# Model Architecture Configuration
# =============================================================================

# F_nonlocal (E3-Equivariant Model) Default Parameters
F_NONLOCAL_DEFAULTS = {
    "num_interactions": 3,      # Number of message passing layers
    "num_neighbors": 20,         # K-nearest neighbors for graph construction
    "mul": 500,                  # Multiplicity for irreducible representations
    "lmax": 4,                   # Maximum spherical harmonic degree
    "cutoff": 4.0,              # Cutoff radius in Angstroms
    "num_basis": 10,            # Number of radial basis functions
    "basis": "gaussian",         # Type of radial basis ("gaussian" or "bessel")
    "spin": False,              # Whether to use spin-polarized calculations
}

# F_local (KAN Correction Model) Default Parameters
F_LOCAL_DEFAULTS = {
    "input_dim": 992,           # Input dimension (node representation size)
    "hidden_dim": 32,           # Hidden dimension after MLP projection
    "kan_width": None,          # KAN network width (None = [hidden_dim, 6, 1])
    "kan_grid": 8,              # KAN grid size
    "kan_k": 4,                 # KAN spline order
    "seed": 42,                 # Random seed for KAN initialization
}

# =============================================================================
# Data Processing Configuration
# =============================================================================

# Graph Construction Parameters
GRAPH_CONSTRUCTION_DEFAULTS = {
    "cutoff": 4.0,              # Cutoff radius for neighbor search (Angstroms)
    "num_probes": None,         # Number of probe points (None = use all grid points)
    "max_neighbors": 20,        # Maximum number of neighbors per atom
    "use_gpu_graph": None,      # GPU graph construction (None=auto, True=force, False=disable)
}

# Data Loading Parameters
DATA_LOADER_DEFAULTS = {
    "batch_size": 1,            # Batch size for training/inference
    "num_workers": 0,           # Number of data loading workers
    "pin_memory": False,        # Whether to pin memory for GPU transfer
    "shuffle": True,            # Whether to shuffle data
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_DEFAULTS = {
    "learning_rate": 1e-3,      # Initial learning rate
    "weight_decay": 0.0,        # L2 regularization weight
    "max_epochs": 100,          # Maximum number of training epochs
    "patience": 10,             # Early stopping patience
    "random_seed": 42,          # Random seed for reproducibility
}

# =============================================================================
# Checkpoint Configuration
# =============================================================================

# Default checkpoint paths (relative to project root)
CHECKPOINT_PATHS = {
    "f_nonlocal": "checkpoints/f_nonlocal.pth",
    "f_local": "checkpoints/f_local.pth",
}

# =============================================================================
# Output Configuration
# =============================================================================

# Output directory paths (relative to project root)
OUTPUT_PATHS = {
    "predictions": "outputs/predictions/",
    "predictions_dual": "outputs/predictions_dual/",
    "logs": "outputs/logs/",
    "checkpoints": "checkpoints/",
}

# CHGCAR file generation settings
CHGCAR_DEFAULTS = {
    "file_extension": ".vasp",  # File extension for CHGCAR files
    "precision": 5,             # Number of decimal places in output
}

# =============================================================================
# Server Configuration
# =============================================================================

SERVER_DEFAULTS = {
    "socket_path": "~/.deepaw/server.sock",
    "pid_file": "~/.deepaw/server.pid",
    "http_host": "0.0.0.0",
    "http_port": 8265,
    "use_compile": False,
    "data_batch_size": 3000,
    "use_dual_model": True,
}

# =============================================================================
# Device Configuration
# =============================================================================

def get_device():
    """
    Get the default device for computation.
    
    Returns:
        str: 'cuda' if GPU is available, otherwise 'cpu'
    """
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# Utility Functions
# =============================================================================

def get_model_config(model_name):
    """
    Get default configuration for a specific model.
    
    Args:
        model_name (str): Name of the model ('f_nonlocal' or 'f_local')
    
    Returns:
        dict: Default configuration parameters
    
    Raises:
        ValueError: If model_name is not recognized
    """
    configs = {
        'f_nonlocal': F_NONLOCAL_DEFAULTS,
        'f_local': F_LOCAL_DEFAULTS,
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")
    
    return configs[model_name].copy()


def get_checkpoint_path(model_name):
    """
    Get default checkpoint path for a specific model.

    Args:
        model_name (str): Name of the model ('f_nonlocal' or 'f_local')

    Returns:
        str: Path to checkpoint file

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in CHECKPOINT_PATHS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(CHECKPOINT_PATHS.keys())}")

    return CHECKPOINT_PATHS[model_name]

