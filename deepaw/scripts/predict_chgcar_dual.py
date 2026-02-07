#!/usr/bin/env python
"""
CHGCAR Dual Model Prediction Script - Predicts charge density using F_nonlocal + F_local
and writes VASP CHGCAR files
"""

# Add DeePAW package to path
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

# Third-party imports
import torch
import numpy as np
from ase.db import connect
from torch.utils.data import DataLoader
from tqdm import tqdm
from pymatgen.io.vasp import Chgcar
from ase.calculators.vasp import VaspChargeDensity

# DeePAW imports
from deepaw import F_nonlocal, F_local
from deepaw.data.chgcar_writer import DensityData, MyCollator
from deepaw.config import (
    get_device,
    get_model_config,
    get_checkpoint_path,
    TRAINING_DEFAULTS
)

# Set device and random seed
device = get_device()
random_seed = TRAINING_DEFAULTS['random_seed']
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Initialize models with default configs
f_nonlocal_config = get_model_config('f_nonlocal')
f_local_config = get_model_config('f_local')
f_nonlocal = F_nonlocal(**f_nonlocal_config)
f_local = F_local(**f_local_config)

# Database path - configurable via environment variable or use default
default_db = os.path.join(deepaw_root, 'examples', 'isolated_atomspred.db')
mysql_url = os.environ.get('DEEPAW_DB_PATH', default_db)
print(f"Using database: {mysql_url}")
if not os.path.exists(mysql_url):
    print(f"Error: Database not found at {mysql_url}")
    print("Please set DEEPAW_DB_PATH environment variable or place database at:")
    print(f"  {default_db}")
    sys.exit(1)

db_ = connect(mysql_url)
dataset = DensityData(mysql_url)
total_count = len(dataset)
print(f"Total count: {total_count}")

# Checkpoint paths - use config
checkpoint_nonlocal = os.path.join(deepaw_root, get_checkpoint_path('f_nonlocal'))
checkpoint_local = os.path.join(deepaw_root, get_checkpoint_path('f_local'))

if not os.path.exists(checkpoint_nonlocal):
    raise FileNotFoundError(f"F_nonlocal checkpoint not found: {checkpoint_nonlocal}")
if not os.path.exists(checkpoint_local):
    raise FileNotFoundError(f"F_local checkpoint not found: {checkpoint_local}")

# Load checkpoints
f_nonlocal.load_state_dict(torch.load(checkpoint_nonlocal, map_location=device))
f_local.load_state_dict(torch.load(checkpoint_local, map_location=device))

f_nonlocal = f_nonlocal.to(device)
f_local = f_local.to(device)

print("Models loaded successfully!")


def pred_full_chgdensity_dual(mpid):
    """Predict charge density using both F_nonlocal and F_local models"""
    all_pred = []
    
    val_dataloader = DataLoader(
        [dataset[mpid]],
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=MyCollator(mysql_url, cutoff=4, num_probes=None, inference=True)
    )

    with torch.no_grad():
        for step, (big_batch) in enumerate(val_dataloader):
            for batch in tqdm(big_batch, total=len(big_batch)):
                _skip = {'probe_target', 'total_num_probes'}
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in _skip}
                
                # F_nonlocal prediction
                output_nonlocal, node_rep = f_nonlocal(batch)
                output_nonlocal = output_nonlocal.view(-1)
                
                # F_local correction
                correction, _ = f_local(None, node_rep)
                correction = correction.view(-1)
                
                # Combined prediction
                output_final = output_nonlocal + correction
                
                all_pred.append(output_final.detach().cpu())
            break
    
    all_pred = torch.cat(all_pred, dim=0).numpy()
    return all_pred


def write_chgcar_via_mpid(input_array, atoms, nx, ny, nz, obj_name):
    """Write charge density to VASP CHGCAR format"""
    datas = input_array
    density = datas.reshape(nx, ny, nz)
    aug_chgcar_file = None

    # Retrieve augmentation, if requested
    if aug_chgcar_file is not None:
        aug = Chgcar.from_file(aug_chgcar_file).data_aug
    else:
        aug = None

    # Extract spin, if available
    if len(density.shape) == 4:  # implies a spin channel exists
        charge_grid = density[..., 0]
        spin_grid = density[..., 1]
    else:
        charge_grid = density
        spin_grid = np.zeros_like(density)

    # Create Chgcar object
    vcd = VaspChargeDensity(filename=None)
    vcd.atoms.append(atoms)
    vcd.chg.append(charge_grid)

    if aug is not None:
        vcd.aug = "".join(aug["total"])
        vcd.augdiff = "".join(aug["diff"])

    vcd.write(f'{obj_name}.vasp', format="chgcar")


# Main prediction loop
for dataid in tqdm(range(1, total_count), total=total_count):
    db_ = connect(mysql_url)
    rows = []
    for row in db_.select(dataset[dataid]):
        rows.append(row)
    atoms = rows[0].toatoms()

    # Get structure name
    try:
        try:
            fname = rows[0].data["mpid"]
        except:
            fname = rows[0].mpid
    except:
        fname = f"dataid_pbc_{dataid}"

    print(f"Processing {fname} (dual model)")

    # Predict using both models
    pred_array = pred_full_chgdensity_dual(dataid)
    nx, ny, nz = rows[0].data.nx, rows[0].data.ny, rows[0].data.nz

    # Output directory
    output_dir = os.path.join(deepaw_root, 'outputs', 'predictions_dual')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)

    # Write CHGCAR file
    write_chgcar_via_mpid(pred_array, atoms, nx, ny, nz, output_path)
    print(f"Saved dual model prediction to: {output_path}.vasp")

print("\n" + "="*70)
print("All predictions completed!")
print(f"Output directory: {os.path.join(deepaw_root, 'outputs', 'predictions_dual')}")
print("="*70)

