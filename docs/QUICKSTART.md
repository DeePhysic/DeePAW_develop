# DeePAW Quick Start Guide

Get started with DeePAW in 5 minutes!

---

## 1️⃣ Setup Environment

```bash
# Activate conda environment
conda activate DeePAW

# Verify installation
cd DeePAW
python -c "from deepaw import F_nonlocal, F_local; print('✅ DeePAW ready!')"
```

---

## 2️⃣ Run Example

```bash
# Test with working example
python examples/real_prediction_example.py
```

**Expected output:**
```
✅ Single Model Test: PASSED
✅ Dual Model Test: PASSED
🎉 All tests passed! DeePAW is working correctly.
```

---

## 3️⃣ Configure Your Data

### Option A: Environment Variable (Recommended)

```bash
export DEEPAW_DB_PATH=/path/to/your/database.db
```

### Option B: Copy to Default Location

```bash
mkdir -p data
cp /path/to/your/database.db data/
```

---

## 4️⃣ Run Predictions

### Dual Model (Highest Accuracy)

```bash
python deepaw/scripts/predict_dual.py
```

### Single Model (Faster)

```bash
python deepaw/scripts/predict_single.py
```

### Generate CHGCAR Files

**Single Model (F_nonlocal only)**:
```bash
python deepaw/scripts/predict_chgcar.py
```

**Dual Model (F_nonlocal + F_local, highest accuracy)**:
```bash
python deepaw/scripts/predict_chgcar_dual.py
```

---

## 5️⃣ Use in Your Code

```python
import torch
from deepaw import F_nonlocal, F_local

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
f_nonlocal = F_nonlocal(num_basis=10).to(device)
f_local = F_local().to(device)

# Load pretrained weights
f_nonlocal.load_state_dict(torch.load('checkpoints/f_nonlocal.pth'))
f_local.load_state_dict(torch.load('checkpoints/f_local.pth'))

# Run prediction
with torch.no_grad():
    base_pred, node_rep = f_nonlocal(batch)
    correction, _ = f_local(None, node_rep)
    final_pred = base_pred + correction
```

---

---

## 🎯 Predict Specific Structure (e.g., id=1)

Edit `deepaw/scripts/predict_dual.py` after line ~513:

```python
dataset = DensityData(mysql_url)

# Add this to only predict id=1:
from torch.utils.data import Subset
test_dataset = Subset(dataset, [0])  # id=1 is at index 0

# Then use test_dataset in your dataloader
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=MyCollator(mysql_url, cutoff=4, num_probes=300)
)
```

---

## 📚 Next Steps

- **Full Documentation**: Read [README.md](README.md) for complete guide
- **Project Structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Examples**: Explore `examples/` directory for more use cases

---

## ⚠️ Troubleshooting

### "Database not found"
```bash
# Set the correct path
export DEEPAW_DB_PATH=/correct/path/to/database.db
```

### "Checkpoint not found"
```bash
# Verify checkpoints exist
ls -lh checkpoints/
# Should show: f_nonlocal.pth (8.0 MB), f_local.pth (151 KB)
```

### Import errors
```bash
# Make sure you're in the DeePAW directory
cd DeePAW
python your_script.py
```

---

## ✅ Checklist

- [ ] Environment activated
- [ ] Example runs successfully
- [ ] Database path configured
- [ ] Checkpoints verified
- [ ] Ready to predict!

---

**Need help?** Check [README.md](README.md) for full documentation

