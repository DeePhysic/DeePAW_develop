# How to Use the External Prediction Tutorial Notebook

## 📓 Overview

The `external_prediction_tutorial.ipynb` notebook provides a complete, interactive tutorial for using DeePAW models for charge density prediction.

## 🚀 Quick Start

### 1. Launch Jupyter

```bash
# Activate your DeePAW environment
conda activate DeePAW

# Navigate to the DeePAW directory
cd /path/to/DeePAW

# Launch Jupyter
jupyter notebook docs/external_prediction_tutorial.ipynb
```

### 2. Run the Notebook

Once Jupyter opens:
1. Click on `external_prediction_tutorial.ipynb`
2. Run cells sequentially using `Shift + Enter`
3. Or run all cells: `Cell` → `Run All`

## 📋 What's Included

The notebook covers:

1. **Setup and Imports** - Import all required libraries
2. **Device Configuration** - Set up GPU/CPU
3. **Load Models** - Load F_nonlocal and F_local with pretrained weights
4. **Load Database** - Connect to the example database
5. **Prediction Functions** - Both dual and single model prediction
6. **Examples** - Run predictions on real structures
7. **Comparison** - Compare single vs dual model results

## 🎯 Expected Output

When you run the notebook successfully, you should see:

```
✓ Imports successful
✓ PyTorch version: 2.5.1+cu124
✓ CUDA available: True

Using device: cuda
GPU: NVIDIA GeForce RTX 4090

✓ Models loaded successfully!
  - F_nonlocal: 1,903,389 parameters
  - F_local: 36,410 parameters
  - Total: 1,939,799 parameters

✓ Dataset loaded
  - Total structures: 119

Predicting structure 1...
✓ Prediction complete
  - Total points: 512000
  - Value range: [-0.000056, 1.314371]
  - Mean: 0.000133
```

## 🔧 Customization

### Use Your Own Structure

To predict on your own structure, modify the database path:

```python
# Instead of the example database
db_path = os.path.join(deepaw_root, 'examples', 'isolated_atomspred.db')

# Use your own database
db_path = '/path/to/your/database.db'
```

### Change Structure ID

```python
# Database IDs start from 1
structure_id = 1  # Change to any valid ID in your database
predictions_dual = predict_charge_density_dual(structure_id, verbose=True)
```

### Adjust Batch Size

In the prediction functions, you can modify the batch size:

```python
# In MyCollator, the batch_size is set to 3000 by default
# This is defined in deepaw/data/chgcar_writer.py
# For larger structures, you may need to reduce this
```

## 📊 Understanding the Output

### Prediction Array

The prediction function returns a numpy array:

```python
predictions = predict_charge_density_dual(structure_id)
# predictions.shape: (num_grid_points,)
# predictions contains charge density values in e/Å³
```

### Grid Points

The number of grid points depends on the structure:
- Determined by `nx`, `ny`, `nz` in the database
- Total points = `nx * ny * nz`
- Example: 80 × 80 × 80 = 512,000 points

## 🐛 Troubleshooting

### Database ID Error

```
err match id 0
```

**Solution**: Database IDs start from 1, not 0. Use `structure_id = 1` or higher.

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: 
1. Use CPU instead: `device = 'cpu'`
2. Or reduce the number of probe points in `MyCollator`

### Import Error

```
ModuleNotFoundError: No module named 'deepaw'
```

**Solution**: Make sure you're in the correct conda environment and DeePAW is installed:

```bash
conda activate DeePAW
pip install -e .
```

## 💡 Tips

1. **Start Simple**: Run the notebook as-is first to verify everything works
2. **Incremental Changes**: Make one change at a time when customizing
3. **Save Often**: Save your notebook frequently (`Ctrl+S`)
4. **Restart Kernel**: If you encounter issues, try `Kernel` → `Restart & Run All`

## 📚 Next Steps

After running the notebook:

1. Try different structures from the database
2. Modify the prediction functions for your use case
3. Export predictions to CHGCAR format (see `CHGCAR_SCRIPTS_GUIDE.md`)
4. Integrate the prediction code into your own scripts

## 🆘 Getting Help

If you encounter issues:

1. Check this guide first
2. Review the main [README.md](../README.md)
3. Look at the [CHGCAR_SCRIPTS_GUIDE.md](CHGCAR_SCRIPTS_GUIDE.md)
4. Check the example scripts in `../examples/`

---

**Happy Predicting! 🚀**

