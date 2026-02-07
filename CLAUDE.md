# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DeePAW** is a deep learning framework for predicting charge density in crystalline materials using a two-stage PAW-inspired architecture. Python 3.10+, PyTorch 2.0+, e3nn, ASE, pykan.

- **F_nonlocal**: E3-equivariant GNN for smooth nonlocal electron distribution (~1.9M params)
- **F_local**: KAN-based local correction model (~36K params)

## Installation & Environment

```bash
conda create -n DeePAW python=3.10
conda activate DeePAW
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install e3nn ase pymatgen scipy tqdm accelerate pykan
pip install -e .
```

Dev tools: `pip install -e ".[dev]"` (adds pytest, black, isort, flake8, mypy).

Verify: `python -c "from deepaw import F_nonlocal, F_local; print('OK')"`

## Common Commands

```bash
# Examples (run from project root)
python examples/basic_prediction.py
python examples/predict_hfo2.py          # Working example with included hfo2_chgd.db

# CHGCAR prediction
python deepaw/scripts/predict_chgcar.py       # Single model (F_nonlocal only)
python deepaw/scripts/predict_chgcar_dual.py  # Dual model (F_nonlocal + F_local)

# Console entry points (after pip install -e .)
deepaw-predict              # Dual model prediction
deepaw-predict-single       # Single model prediction
deepaw-predict-chgcar       # CHGCAR file prediction

# Dataset preparation (from prep_db/ directory)
python prep_db/makedb.py --root_path /path/to/vasp --output_db dataset.db
python prep_db/nelec.py --input_dir /path/to/materials --output_db materials.db
python prep_db/chunk.py --input_db large.db --output_db chunks.db --n_divisions 6

# Energy prediction (GNN extension)
cd deepaw/rep_sub_model/GNN
python preprocess.py --pca_dim 64 --cutoff 4.0
python train.py
```

No formal test suite exists. The `tests/` directory referenced in README does not exist. Ad-hoc tests live in `deepaw/analysis/tests/` and `deepaw/energy_prediction/test_pipeline.py`.

## Architecture

### Two-Stage Pipeline

```
Atomic Structure → F_nonlocal → (base_prediction, node_representations_992d)
                                        ↓
                   node_representations → F_local → correction
                                        ↓
                   final_prediction = base_prediction + correction
```

**F_nonlocal must always run before F_local** — the local model consumes the 992-dim node representations produced by the nonlocal model.

### F_nonlocal internals (`deepaw/models/f_nonlocal.py`)

Two sub-models composed sequentially:

1. **AtomicConfigurationModel** — E3-equivariant message passing over atoms
   - 3 InteractionBlock layers with spherical harmonics up to L=4
   - Irreps: ~500 multiplicity across L=0..4 (s,p,d,f,g orbitals)
   - RadialBasis: 10 Gaussian functions, 4.0A cutoff, 20 neighbors
   - Gate activations: SiLU (even parity), Tanh (odd parity)

2. **AtomicPotentialModel** — One-way atom-to-probe convolutions
   - Same irreps structure, projects to scalar charge density at probe points
   - Returns `(predictions, node_representations)`

### F_local internals (`deepaw/models/f_local.py`)

- MLP projection: 992 -> 32 (zero-initialized for stable training start)
- KAN network: [32, 6, 1] with grid=8, spline_order=4
- Returns `(corrections, None)`

### Key shared utilities (`deepaw/models/irreps_tools.py`)

- `get_irreps(mul, lmax)` — Builds e3nn irreps strings
- `InteractionBlock` — E3-equivariant message passing with tensor products
- `RadialBasis` — Gaussian/Bessel radial basis functions

### Data pipeline (`deepaw/data/`)

- **DensityData, MyCollator, GraphConstructor** — exported from `chgcar_writer.py`
- **KdTreeGraphConstructor** — from `graph_construction.py`
- Database format: ASE `.db` files with flattened charge density in `row.data['chg']` and grid dims in `row.data['nx/ny/nz']`
- `chgcar_writer.py` and `chgcar_writer_huge.py` are large files (~30K lines each) handling VASP CHGCAR I/O

### Configuration (`deepaw/config.py`)

All model defaults are centralized here. Use `get_model_config('f_nonlocal')` or `get_model_config('f_local')` to get default parameter dicts. Checkpoint paths via `get_checkpoint_path()`.

### Energy prediction extension (`deepaw/energy_prediction/`)

- **EnergyHead, ScalarEnergyHead** — energy prediction models
- **EnergyDataset** — dataset class
- Uses embeddings from F_nonlocal; checkpoint at `checkpoints/best_energy_model.pth`

### Legacy aliases

- `E3DensityModel` = `F_nonlocal`
- `ResidualCorrectionModel` = `F_local`

Both old and new names work via re-exports in `deepaw/models/__init__.py`.

## Environment Variables

- `DEEPAW_DB_PATH`: Database file path (default: `./data/chg_mp_large.db`)
- `DEEPAW_OUTPUT_DIR`: Output directory (default: `./outputs/predictions/`)

## Key Constraints

- All scripts assume execution from the **project root** (relative paths throughout)
- Checkpoint files must exist before loading: `checkpoints/f_nonlocal.pth` (8MB), `checkpoints/f_local.pth` (151KB)
- E3NN irreps manipulation is non-trivial — always use helpers in `irreps_tools.py`
- The `kan` import comes from the `pykan` package (not `kan`)
- Training outputs go to `outputs/` (auto-created)
- The included `examples/hfo2_chgd.db` is the only bundled database for testing
