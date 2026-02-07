# DeePAW Project Structure

## 📁 Directory Organization

```
DeePAW/
│
├── 📦 deepaw/                          # Main package
│   ├── __init__.py                    # Package initialization
│   │
│   ├── 🧠 models/                     # Neural network models
│   │   ├── __init__.py
│   │   ├── f_nonlocal.py             # F_nonlocal (E3-equivariant model)
│   │   ├── f_local.py                # F_local (KAN correction model)
│   │   └── irreps_tools.py           # E3NN utilities
│   │
│   ├── 📊 data/                       # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py                # Dataset classes (DensityData)
│   │   ├── chgcar_writer.py          # VASP CHGCAR I/O
│   │   ├── chgcar_writer_huge.py     # Large-scale CHGCAR I/O
│   │   ├── layer.py                  # Data layers (pad_and_stack)
│   │   ├── collate.py                # Batch collation
│   │   ├── graph_construction.py     # Graph building
│   │   └── split.py                  # Data splitting
│   │
│   ├── 🛠️ utils/                      # Utilities
│   │   ├── __init__.py
│   │   └── data_loader.py            # Data loading helpers
│   │
│   └── 🚀 scripts/                    # Prediction scripts
│       ├── __init__.py
│       ├── predict_dual.py           # Dual model prediction
│       ├── predict_single.py         # Single model prediction
│       ├── predict_chgcar.py         # CHGCAR prediction
│       └── convert_legacy_weights.py # Weight conversion utility
│
├── 💾 checkpoints/                    # Pretrained models
│   ├── f_nonlocal.pth                # F_nonlocal weights (1.9M params)
│   └── f_local.pth                   # F_local weights (36K params)
│
├── 📚 examples/                       # Example scripts
│   ├── __init__.py
│   └── basic_prediction.py           # Basic usage examples
│
├── 🧪 tests/                          # Unit tests
│   ├── __init__.py
│   └── test_models.py                # Model tests
│
├── 📖 docs/                           # Documentation
│
├── 📄 Configuration Files
│   ├── setup.py                      # Package setup
│   ├── requirements.txt              # Python dependencies
│   ├── environment.yml               # Conda environment
│   ├── MANIFEST.in                   # Package manifest
│   └── .gitignore                    # Git ignore rules
│
├── 📝 Documentation
│   ├── README.md                     # Main README (GitHub-ready)
│   ├── PROJECT_STRUCTURE.md          # This file
│   └── LICENSE                       # License file
│
└── 🔧 Utilities
    └── test_installation.py          # Installation test script
```

---

## 🏗️ Architecture Overview

### Model Hierarchy

```
deepaw/
├── F_nonlocal (E3DensityModel)
│   ├── E3AtomRepresentationModel
│   │   └── InteractionBlock (3 layers)
│   │       ├── E3-Equivariant Convolutions
│   │       ├── Spherical Harmonics (lmax=4)
│   │       └── Radial Basis Functions
│   │
│   └── E3ProbeMessageModel
│       ├── Probe Point Generation
│       └── Charge Density Prediction
│
└── F_local (ResidualCorrectionModel)
    ├── MLP (992 → 32)
    └── KAN Network [32, 6, 1]
```

---

## 📦 Package Components

### Core Models (`deepaw/models/`)

| File | Description | Key Classes |
|------|-------------|-------------|
| `f_nonlocal.py` | Non-local charge density model | `F_nonlocal`, `E3AtomRepresentationModel`, `E3ProbeMessageModel` |
| `f_local.py` | Local correction model | `F_local` |
| `irreps_tools.py` | E3NN utilities | `InteractionBlock`, `RadialBasis`, `get_irreps` |

### Data Processing (`deepaw/data/`)

| File | Description | Key Classes |
|------|-------------|-------------|
| `dataset.py` | Dataset classes | `DensityData`, `MyCollator`, `GraphConstructor` |
| `chgcar_writer.py` | VASP file I/O | `DensityData`, `MyCollator` |
| `layer.py` | Data layers | `pad_and_stack` |

### Scripts (`deepaw/scripts/`)

| File | Description | Usage |
|------|-------------|-------|
| `predict_dual.py` | Dual model prediction | Highest accuracy |
| `predict_single.py` | Single model prediction | Faster inference |
| `predict_chgcar.py` | CHGCAR file prediction | VASP integration |

---

## 🔄 Model Name Mapping

### Legacy → New Names

| Legacy Name | New Name | Type |
|-------------|----------|------|
| `E3DensityModel` | `F_nonlocal` | Non-local model |
| `ResidualCorrectionModel` | `F_local` | Local correction |

**Note**: Legacy names are still supported as aliases for backward compatibility.

---

## 📊 Model Specifications

### F_nonlocal

- **Parameters**: 1,903,389
- **Input**: Atomic structure (positions, species, cell)
- **Output**: Charge density at probe points + node representations
- **Architecture**: E3-equivariant message passing
- **Checkpoint**: `checkpoints/f_nonlocal.pth`

### F_local

- **Parameters**: 36,410
- **Input**: Node representations from F_nonlocal (992-dim)
- **Output**: Local corrections
- **Architecture**: MLP + KAN
- **Checkpoint**: `checkpoints/f_local.pth`

---

## 🚀 Usage Patterns

### Import Patterns

```python
# Recommended (new names)
from deepaw import F_nonlocal, F_local

# Also supported (legacy names)
from deepaw.models.f_nonlocal import E3DensityModel
from deepaw.models.f_local import ResidualCorrectionModel
```

### Prediction Workflow

```python
# 1. Single model
model = F_nonlocal()
prediction, node_rep = model(batch)

# 2. Dual model
f_nonlocal = F_nonlocal()
f_local = F_local()

base_pred, node_rep = f_nonlocal(batch)
correction, _ = f_local(None, node_rep)
final_pred = base_pred + correction
```

---

## 🔧 Development

### Adding New Features

1. **New Model**: Add to `deepaw/models/`
2. **New Data Loader**: Add to `deepaw/data/`
3. **New Script**: Add to `deepaw/scripts/`
4. **Tests**: Add to `tests/`
5. **Examples**: Add to `examples/`

### Testing

```bash
# Run all tests
pytest tests/

# Test installation
python test_installation.py

# Test specific module
pytest tests/test_models.py -v
```

---

## 📝 File Naming Conventions

- **Models**: `f_*.py` (e.g., `f_nonlocal.py`, `f_local.py`)
- **Scripts**: `predict_*.py` or descriptive names
- **Tests**: `test_*.py`
- **Examples**: `*_example.py` or descriptive names
- **Utilities**: descriptive names (e.g., `data_loader.py`)

---

## 🎯 Key Design Principles

1. **Modularity**: Each component is self-contained
2. **Backward Compatibility**: Legacy names supported via aliases
3. **Clear Naming**: `F_nonlocal` and `F_local` reflect model purposes
4. **Professional Structure**: Follows Python package best practices
5. **Documentation**: Comprehensive docstrings and README

---

## 📧 Maintenance

For questions about the project structure:
- See `README_NEW.md` for usage documentation
- See `setup.py` for package configuration
- See `tests/` for testing examples


