# DeePAW: Deep Learning for PAW strategy Charge Density Prediction

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A deep learning algorithm for predicting charge density inspired by the projector augmented wave (PAW) method**

[Installation](#installation) •
[Quick Start](#quick-start) •
[Documentation](#documentation) •
[Citation](#citation)

</div>

---

## 🌟 Overview

**DeePAW** is a deep learning framework for predicting charge density in crystalline materials. It combines:

- **F_nonlocal**: Smooth nonlocal electron distribution via **e3nn** and **mlp**
- **F_local**: Sharp localized electron distribution via **node rep** and **kan**

This two-stage approach achieves high accuracy by first capturing global structural features, then refining predictions with learned local corrections.

### Key Features

✨ **PAW like Architecture** - Preserves rotational symmetry for physical consistency  
🚀 **High Performance** - Optimized for multi-GPU training and inference  
📊 **Pretrained Models** - Ready-to-use models trained on Materials Project  
🔧 **Flexible** - Easy to adapt for custom datasets and materials  
📦 **Production Ready** - Clean API and comprehensive documentation  

---

## 🧭 Quick Navigation

### 👋 New Users - Start Here!
1. **[Installation](#installation)** - Set up DeePAW 
2. **[Quick Start](#quick-start)** - Run your first prediction
3. **[Examples](examples/README.md)** - Step-by-step tutorials
4. **[Interactive Tutorial](docs/external_prediction_tutorial.ipynb)** - Jupyter notebook walkthrough

### 📚 Documentation
- **[Quick Start Guide](docs/QUICKSTART.md)** - Comprehensive getting started guide
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Understand the codebase
- **[CHGCAR Generation](docs/CHGCAR_SCRIPTS_GUIDE.md)** - Generate VASP output files
- **[Full Documentation](docs/)** - Complete documentation index

### 🔬 Researchers & Developers
- **[Model Architecture](#model-architecture)** - Technical details
- **[Training](#training)** - Train your own models
- **[API Reference](#usage)** - Detailed API documentation
- **[Tests](tests/)** - Unit and integration tests

### 🚀 Common Tasks
| Task | Resource |
|------|----------|
| Install DeePAW | [Installation](#installation) |
| Run basic prediction | [examples/basic_prediction.py](examples/basic_prediction.py) |
| Generate CHGCAR files | [deepaw/scripts/predict_chgcar.py](deepaw/scripts/predict_chgcar.py) |
| Use dual model | [examples/real_prediction_example.py](examples/real_prediction_example.py) |
| Understand architecture | [Model Architecture](#model-architecture) |
| Train custom model | [Training](#training) |

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Energy Prediction Extension](#energy-prediction-extension)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Pretrained Models](#pretrained-models)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.0+ (for GPU support)
- Conda (recommended)

### Environment Setup

```bash
# Create conda environment
conda create -n DeePAW python=3.10
conda activate DeePAW

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install e3nn ase pymatgen scipy tqdm accelerate pykan

# Clone and install DeePAW
git clone https://github.com/SuthPhy2Ai/DeePAW.git
cd DeePAW
pip install -e .
```

### Verify Installation

```bash
python -c "from deepaw import F_nonlocal, F_local; print('DeePAW installed successfully!')"
```

---

## 🚀 Quick Start

### Basic Prediction

```python
import torch
from deepaw import F_nonlocal, F_local

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Non-local model (base prediction)
model_nonlocal = F_nonlocal(num_basis=10).to(device)
model_nonlocal.load_state_dict(torch.load('checkpoints/f_nonlocal.pth'))

# Local model (correction)
model_local = F_local().to(device)
model_local.load_state_dict(torch.load('checkpoints/f_local.pth'))

# Predict
model_nonlocal.eval()
model_local.eval()

with torch.no_grad():
    # Get base prediction and node representations
    base_pred, node_rep = model_nonlocal(batch)
    
    # Get local correction
    correction, _ = model_local(None, node_rep)
    
    # Final prediction
    final_pred = base_pred + correction
```

### Command Line Interface

```bash
# Dual model prediction (highest accuracy)
export DEEPAW_DB_PATH=/path/to/your/database.db
python deepaw/scripts/predict_dual.py

# Single model prediction (faster)
python deepaw/scripts/predict_single.py

# CHGCAR file prediction
python deepaw/scripts/predict_chgcar.py
```

---

## ⚙️ Configuration

DeePAW uses **relative paths** and **environment variables** for flexible configuration.

### Directory Structure

```
DeePAW/
├── checkpoints/          # Pretrained weights (auto-loaded)
│   ├── f_nonlocal.pth   # Non-local model (8.0 MB)
│   └── f_local.pth      # Local correction (151 KB)
├── data/                 # Your database files
├── outputs/              # Prediction outputs (auto-created)
└── examples/             # Working examples
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DEEPAW_DB_PATH` | Database file path | `./data/chg_mp_large.db` |
| `DEEPAW_OUTPUT_DIR` | Output directory | `./outputs/predictions/` |

### Quick Configuration

```bash
# Set database path
export DEEPAW_DB_PATH=/path/to/your/database.db

# Run prediction
cd DeePAW
python deepaw/scripts/predict_dual.py
```

**For detailed configuration options, see [CONFIG.md](CONFIG.md)**

---

## 🏗️ Model Architecture

DeePAW uses a two-stage architecture combining non-local and local models:

### F_nonlocal (Non-local Model)

The non-local model consists of two sub-models working in sequence:

#### 1. AtomicConfigurationModel
Learns atomic representations through E3-equivariant message passing:

```
Atomic Structure (positions, species, cell)
      ↓
One-hot Encoding (119 atomic species → 119x0e irreps)
      ↓
InteractionBlock (3 message passing layers)
  ├─ Convolution: E3-Equivariant Tensor Products
  │   └─ Radial Basis: Gaussian basis (10 functions, cutoff=4.0Å)
  ├─ Spherical Harmonics: lmax=4 (up to g-orbitals)
  ├─ Gate Activation: SiLU for even, Tanh for odd parity
  └─ Irreps: ~500 multiplicity across L=0,1,2,3,4
      ↓
Atomic Representations (3 layers of node features)
```

**Key Parameters:**
- **Irreps**: `167x0o + 167x0e + 56x1o + 56x1e + 33x2o + 33x2e + ...` (up to L=4)
- **Radial Network**: [num_basis=10, 100] neurons
- **Cutoff**: 4.0 Å
- **Neighbors**: 20 per atom

#### 2. AtomicPotentialModel
Predicts charge density at probe points from atomic representations:

```
Atomic Representations + Probe Points
      ↓
InteractionBlock (3 layers, one-way message passing)
  ├─ ConvolutionOneWay: Atoms → Probes
  ├─ Edge Features: Atom-probe distances & directions
  └─ Same irreps structure as AtomicConfigurationModel
      ↓
Readout Layer (Linear projection to scalar)
      ↓
Charge Density at Probe Points
```

**Total Parameters**: ~1.9M
**Input**: Atomic structure (positions, species, cell) + probe point positions
**Output**: Charge density values at probe points + node representations (992-dim)

### F_local (Local Correction Model)

Refines predictions using Kolmogorov-Arnold Networks:

```
Node Representations (992-dim from F_nonlocal)
      ↓
MLP Layer (992 → 32)
  └─ Initialized to zero for stable training
      ↓
KAN Network [32, 6, 1]
  ├─ Grid size: 8
  ├─ Spline order: 4
  └─ Learns non-linear local corrections
      ↓
Local Corrections (per-probe)
```

**Total Parameters**: ~36K
**Input**: Node representations from F_nonlocal (992-dim)
**Output**: Correction values added to base predictions

### Combined Workflow

```
Structure → F_nonlocal → Base Prediction  ─┐
                ↓                          │
         Node Representations              │
                ↓                          │
            F_local → Corrections ─────────┤
                                           ↓
                                  Final Prediction
```

---

## Energy Prediction Extension

DeePAW includes an **energy prediction model** that leverages the pre-trained atomic embeddings from F_nonlocal to predict formation energies of crystal structures.

### GNN Energy Model

Located in `deepaw/rep_sub_model/GNN/`, this module provides:

- **Transfer Learning**: Uses 992D atomic embeddings from F_nonlocal ✅
- **PCA Reduction**: Reduces embeddings to 64D (98%+ variance retained) ✅
- **Advanced Architecture**: Edge updates, multi-head attention, message passing ✅
- **ASE Integration**: Calculator interface for energy prediction ✅
- **AtomRef Support**: Conversion between formation and total energies ✅

**Current Status:**
- ✅ **Energy Prediction**: Fully functional
- 🚧 **Force Prediction**: Under development (required for structure relaxation)

### Architecture

```
Atomic Embeddings (992D from F_nonlocal)
      ↓
PCA Reduction (64D)
      ↓
Input Projection (64D → 128D)
      ↓
Edge Update Conv Layers (×3-4)
  ├─ Edge Feature Updates
  ├─ Multi-Head Attention
  └─ Message Passing
      ↓
Global Pooling (mean)
      ↓
MLP Predictor
      ↓
Formation Energy
      ↓ (optional)
AtomRef Correction
      ↓
Total Energy
```

### Quick Start

```bash
# Preprocess data with PCA
cd deepaw/rep_sub_model/GNN
python preprocess.py --pca_dim 64 --cutoff 4.0

# Train model
python train.py

# Test energy prediction
python test_calculator.py
```

**Example Usage:**

```python
from ase_calculator import GNNCalculator
from ase.build import bulk

# Initialize calculator
calc = GNNCalculator(
    model_path='checkpoints/best_model.pth',
    embeddings_path='atom_embeddings.json'
)

# Create structure and predict energy
atoms = bulk('Si', 'diamond', a=5.43)
atoms.calc = calc
energy = atoms.get_potential_energy()
print(f"Energy: {energy:.4f} eV")
```

**⚠️ Note:** Structure relaxation is not yet available as force prediction is under development.

**For detailed documentation, see [deepaw/rep_sub_model/GNN/README.md](deepaw/rep_sub_model/GNN/README.md)**

---

## 📖 Usage

### Single Model Prediction

Use only F_nonlocal for fast predictions:

```python
from deepaw import F_nonlocal

model = F_nonlocal(
    num_interactions=3,
    num_neighbors=20,
    mul=500,
    lmax=4,
    cutoff=4.0,
    num_basis=10
)

prediction, node_rep = model(batch)
```

### Dual Model Prediction

Combine F_nonlocal + F_local for highest accuracy:

```python
from deepaw import F_nonlocal, F_local

# Initialize models
f_nonlocal = F_nonlocal(num_basis=10)
f_local = F_local()

# Load pretrained weights
f_nonlocal.load_state_dict(torch.load('checkpoints/f_nonlocal.pth'))
f_local.load_state_dict(torch.load('checkpoints/f_local.pth'))

# Predict
base_pred, node_rep = f_nonlocal(batch)
correction, _ = f_local(None, node_rep)
final_pred = base_pred + correction
```

---

## 📊 Dataset Preparation

Before training, you need to prepare your dataset from VASP charge density calculations. DeePAW provides three tools in the `prep_db/` directory to convert VASP output into ASE databases.

### 🔧 Available Tools

| Tool | Purpose | Use Case |
|------|---------|----------|
| **makedb.py** | Extract CHGCAR → Database | Simple directory structures |
| **nelec.py** | Process with material IDs | Organized materials databases |
| **chunk.py** | Split large structures | Very large systems (>1000 atoms) |

---

### 1️⃣ makedb.py - Basic CHGCAR to Database Conversion

Recursively searches for CHGCAR files and extracts charge density data into an ASE database.

#### Usage

```bash
cd prep_db

# Basic usage with default settings
python makedb.py --root_path /path/to/vasp/calculations --output_db my_dataset.db

# Search for specific CHGCAR pattern
python makedb.py --root_path /path/to/data --output_db output.db --chgcar_pattern "CHGCAR"

# Enable verbose error messages
python makedb.py --root_path /path/to/data --output_db output.db --verbose
```

#### Arguments

- `--root_path`: Root directory containing VASP calculations (default: current directory)
- `--output_db`: Output database filename (default: `charge_density.db`)
- `--chgcar_pattern`: Pattern to match CHGCAR files (default: `"CHGCAR"`)
- `--verbose`: Print detailed error messages

---

### 2️⃣ nelec.py - Advanced Processing with Material IDs

Process charge density from structured VASP calculation directories, supporting both CHGCAR and HDF5 formats.

#### Usage

```bash
cd prep_db

# Process CHGCAR files
python nelec.py --input_dir /path/to/materials --output_db materials.db --format chgcar

# Process HDF5 files (vaspwave.h5)
python nelec.py --input_dir /path/to/materials --output_db materials.db --format hdf5

# Process specific material IDs
python nelec.py --input_dir /path/to/materials --output_db materials.db \
    --material_ids mat001 mat002 mat003 --format chgcar

# Custom file paths within material directories
python nelec.py --input_dir /path/to/materials --output_db materials.db \
    --format chgcar --chgcar_path "calculation/CHGCAR"
```

#### Arguments

- `--input_dir`: Base directory containing material subdirectories (required)
- `--output_db`: Output database filename (default: `charge_density.db`)
- `--format`: Input format - `chgcar` or `hdf5` (default: `chgcar`)
- `--material_ids`: List of specific material IDs to process (optional)
- `--chgcar_path`: Relative path to CHGCAR within material directory (default: `scf/ns/CHGCAR`)
- `--hdf5_path`: Relative path to HDF5 file (default: `scf/ns/vaspwave.h5`)
- `--poscar_path`: Relative path to POSCAR for HDF5 mode (default: `scf/ns/POSCAR`)
- `--verbose`: Print detailed error messages

#### Expected Directory Structure

```
input_dir/
├── material_001/
│   └── scf/ns/
│       ├── CHGCAR
│       ├── POSCAR
│       └── vaspwave.h5
├── material_002/
│   └── scf/ns/
│       └── ...
└── ...
```

---

### 3️⃣ chunk.py - Split Large Structures

Divide very large atomic structures into smaller overlapping chunks for parallel computation.

#### Usage

```bash
cd prep_db

# Basic usage
python chunk.py --input_db large_structure.db --output_db chunks.db \
    --structure_id 1 --n_divisions 6 --overlap 0.2

# Custom grid spacing
python chunk.py --input_db large.db --output_db chunks.db \
    --n_divisions 4 --overlap 0.15 --grid_spacing 0.1

# Enable periodic boundary conditions for chunks
python chunk.py --input_db large.db --output_db chunks.db --pbc
```

#### Arguments

- `--input_db`: Input database containing large structure (required)
- `--output_db`: Output database for chunks (default: `structure_chunks.db`)
- `--structure_id`: ID of structure to chunk (default: 1)
- `--n_divisions`: Number of divisions per axis, creates n³ chunks (default: 6)
- `--overlap`: Overlap ratio between chunks, 0.0-1.0 (default: 0.2)
- `--grid_spacing`: Grid spacing for charge density in Angstrom (default: 0.1)
- `--pbc`: Use periodic boundary conditions for chunks

#### How It Works

1. Loads a large structure from the input database
2. Calculates bounding box and divides it into n×n×n chunks
3. Each chunk overlaps with neighbors by the specified ratio
4. Atoms are assigned to chunks based on their positions
5. Each chunk is saved as a separate entry with metadata

#### Example

For a structure with 10,000 atoms divided into 6×6×6 = 216 chunks with 20% overlap:
- Each chunk contains a subset of atoms
- Chunks overlap to ensure continuity at boundaries
- Enables parallel prediction on different chunks
- Results can be stitched back together

---

### 📊 Workflow Examples

#### Example 1: Simple Dataset Creation

```bash
cd prep_db

# Step 1: Collect CHGCAR files into database
python makedb.py \
    --root_path /data/vasp_calculations \
    --output_db training_data.db \
    --verbose

# Result: training_data.db with all charge density data
```

#### Example 2: Materials Database with IDs

```bash
cd prep_db

# Step 1: Process materials with ID tracking
python nelec.py \
    --input_dir /data/materials_project \
    --output_db mp_dataset.db \
    --format chgcar \
    --verbose

# Result: mp_dataset.db with material_id field for each entry
```

#### Example 3: Large Structure Processing

```bash
cd prep_db

# Step 1: Create database with large structure
# (assume you already have large_system.db)

# Step 2: Split into chunks
python chunk.py \
    --input_db large_system.db \
    --output_db system_chunks.db \
    --structure_id 1 \
    --n_divisions 6 \
    --overlap 0.2

# Step 3: Process chunks in parallel (use your prediction script)
# Each chunk can be processed independently

# Result: Predictions for very large systems that don't fit in memory
```

---

### 📝 Database Format

All scripts create ASE databases with the following format:

```python
from ase.db import connect

db = connect('dataset.db')
row = db.get(1)

# Access data
atoms = row.toatoms()           # ASE Atoms object
chg = row.data['chg']           # Flattened charge density (1D array)
nx, ny, nz = row.data['nx'], row.data['ny'], row.data['nz']  # Grid shape

# Reshape charge density
import numpy as np
chg_3d = chg.reshape(nx, ny, nz)
```

**Database entries contain:**
- `atoms`: ASE Atoms object with structure
- `data['chg']`: Flattened charge density array
- `data['nx']`, `data['ny']`, `data['nz']`: Grid dimensions
- `material_id` (nelec.py only): Material identifier
- `chunk_index`, `i`, `j`, `k` (chunk.py only): Chunk position information

---

### 🔍 Tips and Best Practices

#### Performance

- **makedb.py**: Fast for simple directory structures, processes files sequentially
- **nelec.py**: Better for organized material databases, supports batch processing
- **chunk.py**: Essential for structures with >1000 atoms

#### Memory Considerations

- Large CHGCAR files (>1GB) may require significant RAM
- Use chunking for structures with dense grids (>200×200×200)
- Database files can be large - ensure sufficient disk space

#### Error Handling

- All scripts skip corrupted files and continue processing
- Use `--verbose` flag to see detailed error messages
- Check output summary for failed files

#### Data Quality

- Verify grid dimensions match your VASP calculations
- Check that atomic positions are correctly loaded
- For HDF5 format, ensure vaspwave.h5 contains charge density data

---

### 🐛 Troubleshooting

**Issue**: "Cannot find CHGCAR files"
- **Solution**: Check `--root_path` or `--input_dir` is correct
- **Solution**: Verify CHGCAR files exist in subdirectories

**Issue**: "Error reading HDF5 file"
- **Solution**: Ensure h5py is installed: `pip install h5py`
- **Solution**: Check HDF5 file structure matches expected format

**Issue**: "Out of memory when chunking"
- **Solution**: Reduce `--n_divisions` to create fewer, larger chunks
- **Solution**: Process structure in smaller batches

**Issue**: "Database is too large"
- **Solution**: Charge density grids are large - this is expected
- **Solution**: Consider downsampling grids if appropriate for your use case

---

## 🎓 Training

### Load Prepared Dataset

```python
from deepaw.data import DensityData
from torch.utils.data import DataLoader

# Load dataset (created using prep_db tools)
dataset = DensityData(
    db_path="training_data.db",  # Database from prep_db scripts
    num_probes=100,
    cutoff=4.0
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=MyCollator()
)
```

### Train F_nonlocal

```python
import torch.optim as optim

model = F_nonlocal()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.L1Loss()

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        pred, _ = model(batch)
        loss = criterion(pred, batch['target'])
        loss.backward()
        optimizer.step()
```

### Train F_local

```python
# Freeze F_nonlocal and train F_local
f_nonlocal.eval()
f_local = F_local()
optimizer = optim.Adam(f_local.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        with torch.no_grad():
            base_pred, node_rep = f_nonlocal(batch)

        correction, _ = f_local(None, node_rep)
        final_pred = base_pred + correction

        loss = criterion(final_pred, batch['target'])
        loss.backward()
        optimizer.step()
```

---

## 📦 Pretrained Models

We provide pretrained models in the `checkpoints/` directory:

| Model | Parameters | File Size | Path |
|-------|-----------|-----------|------|
| F_nonlocal | ~1.9M | 8.0 MB | `checkpoints/f_nonlocal.pth` |
| F_local | ~36K | 151 KB | `checkpoints/f_local.pth` |

**Note**: Models are trained on Materials Project dataset. The dual model (F_nonlocal + F_local) provides higher accuracy than using F_nonlocal alone.

---

## 📚 Documentation

### 📖 Complete Documentation

For comprehensive documentation, tutorials, and guides, see the **[docs/](docs/)** directory:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[External Prediction Tutorial](docs/external_prediction_tutorial.ipynb)** - Interactive Jupyter notebook (Recommended!)
- **[CHGCAR Scripts Guide](docs/CHGCAR_SCRIPTS_GUIDE.md)** - Generate VASP CHGCAR files
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Architecture and organization
- **[Class Renaming Summary](docs/CLASS_RENAMING_SUMMARY.md)** - API documentation

**Note**: Dataset preparation tools and detailed usage are documented in the [Dataset Preparation](#dataset-preparation) section above.

### Project Structure

```
DeePAW/
├── deepaw/                    # Main package
│   ├── models/               # Neural network models
│   │   ├── f_nonlocal.py    # Non-local model (E3NN)
│   │   ├── f_local.py       # Local correction (KAN)
│   │   └── irreps_tools.py  # E3NN utilities
│   ├── rep_sub_model/        # 🔋 Energy prediction extension
│   │   └── GNN/             # Graph neural network for energies
│   │       ├── model.py     # Advanced GNN architecture
│   │       ├── train.py     # Training pipeline
│   │       ├── dataset.py   # Dataset loader
│   │       ├── preprocess.py # PCA reduction & preprocessing
│   │       ├── ase_calculator.py # ASE integration
│   │       └── README.md    # Detailed documentation
│   ├── data/                # Data processing
│   │   ├── dataset.py       # Dataset classes
│   │   ├── layer.py         # Data layers
│   │   └── chgcar_writer.py # VASP file I/O
│   ├── utils/               # Utilities
│   │   └── data_loader.py   # Data loading helpers
│   └── scripts/             # Prediction scripts
│       ├── predict_chgcar.py      # Single model CHGCAR
│       └── predict_chgcar_dual.py # Dual model CHGCAR
├── prep_db/                 # 🔧 Dataset preparation tools
│   ├── makedb.py           # CHGCAR → Database converter
│   ├── nelec.py            # Advanced processing with IDs
│   ├── chunk.py            # Split large structures
│   └── README.md           # Detailed usage guide
├── checkpoints/             # Pretrained models
├── examples/                # Example scripts
├── tests/                   # Unit tests
└── docs/                    # 📚 Documentation (START HERE!)

```

### API Reference

#### F_nonlocal

```python
F_nonlocal(
    num_interactions=3,    # Number of message passing layers
    num_neighbors=20,      # Maximum neighbors per atom
    mul=500,              # Irreps multiplicity (total across all L)
    lmax=4,               # Maximum angular momentum (0-4: s,p,d,f,g)
    cutoff=4.0,           # Cutoff radius in Angstroms
    basis="gaussian",     # Radial basis type
    num_basis=10,         # Number of radial basis functions
    spin=False            # Include spin (not implemented)
)
```

**Returns**: `(predictions, node_representations)`
- `predictions`: Charge density at probe points (shape: [batch, num_probes])
- `node_representations`: 992-dim features for F_local (shape: [num_probes, 992])

#### F_local

```python
F_local(
    input_dim=992,         # Input dimension (from F_nonlocal)
    hidden_dim=32,         # Hidden dimension after MLP projection
    kan_width=[32,6,1],    # KAN network architecture (default)
    kan_grid=8,            # KAN grid size for spline basis
    kan_k=4,               # KAN spline order
    seed=42                # Random seed for reproducibility
)
```

**Returns**: `(corrections, None)`
- `corrections`: Local correction values (shape: [num_probes, 1])
- Second return value is `None` (kept for API compatibility)

---

## 🔬 Advanced Usage

### Multi-GPU Training

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

for batch in dataloader:
    optimizer.zero_grad()
    pred, _ = model(batch)
    loss = criterion(pred, batch['target'])
    accelerator.backward(loss)
    optimizer.step()
```

### Custom Dataset

```python
from deepaw.data import DensityData

class CustomDensityData(DensityData):
    def __init__(self, custom_params):
        super().__init__(...)
        # Add custom initialization

    def __getitem__(self, idx):
        # Custom data loading
        return data
```

---

## 📄 Citation

If you use DeePAW in your research, please cite:

```bibtex
@article{deepaw2024,
  title={DeePAW: Deep Learning for PAW Charge Density Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [e3nn](https://e3nn.org/)
- Trained on [Materials Project](https://materialsproject.org/) data
- KAN implementation from [pykan](https://github.com/KindXiaoming/pykan)

---

## 📧 Contact

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/SuthPhy2Ai/DeePAW/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SuthPhy2Ai/DeePAW/discussions)

---

<div align="center">

**Made with ❤️ by the DeePAW Team**

[⬆ Back to Top](#deepaw-deep-learning-for-paw-charge-density-prediction)

</div>



