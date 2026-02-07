# DeePAW Examples

This directory contains example scripts demonstrating how to use DeePAW for charge density prediction.

## 📚 Available Examples

### 1. Basic Prediction (`basic_prediction.py`)

**Purpose**: Demonstrates the basic usage of DeePAW models

**What it covers**:
- Creating F_nonlocal model (single model approach)
- Creating F_nonlocal + F_local models (dual model approach)
- Loading pretrained weights
- Model architecture information

**How to run**:
```bash
cd /path/to/DeePAW
conda activate DeePAW
python examples/basic_prediction.py
```

**Expected output**:
- Model creation confirmation
- Parameter counts
- Architecture details
- Checkpoint loading status

**Use case**: Quick introduction to DeePAW models without actual data

---

### 2. Real Prediction Example (`real_prediction_example.py`)

**Purpose**: Complete prediction workflow with actual crystal structure data

**What it covers**:
- Creating crystal structures using ASE
- Building graph representations
- Running single model predictions
- Running dual model predictions
- Comparing prediction outputs

**How to run**:
```bash
cd /path/to/DeePAW
conda activate DeePAW
python examples/real_prediction_example.py
```

**Expected output**:
- Structure information (formula, atoms, cell)
- Batch data shapes
- Prediction results with output ranges
- Test pass/fail status

**Use case**: Understanding the complete prediction pipeline from structure to charge density

**Requirements**:
- Pretrained model checkpoints in `checkpoints/` directory
- ASE library for structure creation

---

## 🚀 Quick Start

If you're new to DeePAW, we recommend following this order:

1. **Start here**: `basic_prediction.py`
   - Understand model architecture
   - See parameter counts
   - Learn about single vs dual model approaches

2. **Next step**: `real_prediction_example.py`
   - See actual predictions
   - Understand data flow
   - Learn graph construction

3. **Advanced usage**: `../docs/external_prediction_tutorial.ipynb`
   - Interactive Jupyter notebook
   - Complete workflow with visualization
   - CHGCAR file generation

---

## 📋 Prerequisites

Before running these examples, ensure you have:

1. **Installed DeePAW**:
   ```bash
   conda env create -f environment.yml
   conda activate DeePAW
   pip install -e .
   ```

2. **Downloaded pretrained models** (if available):
   - `checkpoints/f_nonlocal.pth` (~8 MB)
   - `checkpoints/f_local.pth` (~151 KB)

3. **Verified installation**:
   ```bash
   python test_installation.py
   ```

---

## 🔧 Customization

### Modifying Model Parameters

Both examples use default parameters. You can customize:

```python
# In basic_prediction.py or real_prediction_example.py
model = F_nonlocal(
    num_interactions=3,    # Number of message passing layers
    num_neighbors=20,      # K-nearest neighbors
    mul=500,              # Multiplicity for irreps
    lmax=4,               # Maximum spherical harmonic degree
    cutoff=4.0,           # Cutoff radius (Angstrom)
    num_basis=10,         # Number of radial basis functions
    spin=False            # Spin-polarized calculation
)
```

### Using Your Own Structures

In `real_prediction_example.py`, modify the structure creation:

```python
from ase.io import read

# Load from file instead of creating bulk structure
atoms = read('your_structure.cif')  # or .vasp, .xyz, etc.
```

---

## 📊 Understanding the Output

### Single Model Output
```
Output shape: torch.Size([N])  # N = number of probe points
Output range: [min, max]       # Predicted charge density values
```

### Dual Model Output
```
Base prediction shape: torch.Size([N])
Correction shape: torch.Size([N])
Final prediction: base + correction
```

---

## 🆘 Troubleshooting

### "Checkpoint not found"
- Download pretrained models or train your own
- Check that files exist in `checkpoints/` directory

### "CUDA out of memory"
- Reduce structure size
- Use CPU instead: `device = 'cpu'`
- Reduce `num_probes` in graph construction

### "Import error"
- Verify installation: `python test_installation.py`
- Check conda environment: `conda activate DeePAW`
- Reinstall package: `pip install -e .`

---

## 📚 Next Steps

After running these examples:

1. **Generate CHGCAR files**: See `deepaw/scripts/predict_chgcar.py`
2. **Interactive tutorial**: Try `docs/external_prediction_tutorial.ipynb`
3. **Read documentation**: Check `docs/QUICKSTART.md`
4. **Understand architecture**: See `docs/PROJECT_STRUCTURE.md`

---

## 💡 Tips

- **Start simple**: Run `basic_prediction.py` first to verify setup
- **Check GPU**: Examples automatically use CUDA if available
- **Read output**: Pay attention to shapes and ranges
- **Experiment**: Modify parameters to understand their effects

---

**Happy predicting!** 🎉

For more information, see the main [README.md](../README.md) and [documentation](../docs/).

