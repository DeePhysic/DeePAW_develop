#!/usr/bin/env python
"""
CHGCAR Prediction Script - Predicts charge density and writes VASP CHGCAR files
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
from deepaw import F_nonlocal
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

# Initialize model with default config
model_config = get_model_config('f_nonlocal')
model = F_nonlocal(**model_config)

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

# Checkpoint path - use config
checkpoint_path = os.path.join(deepaw_root, get_checkpoint_path('f_nonlocal'))
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model = model.to(device)
def pred_full_chgdensity(mpid):
    # all_true = []
    all_pred = []
    # all_true_corr = []
    # all_pred_corr = []
    # flag = []
    # countt = 0
    val_dataloader = DataLoader(
            [dataset[mpid]],
            batch_size=1,
            num_workers=0,
            shuffle=False,  # Important to keep the order of the data
            pin_memory=True,  # Important for GPU usage
            collate_fn=MyCollator(mysql_url, cutoff=4, num_probes=None, inference=True)
            )

    with torch.no_grad():
        for step, (big_batch) in enumerate(val_dataloader):
            for batch in tqdm(big_batch, total=len(big_batch)):
                _skip = {'probe_target', 'total_num_probes'}
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in _skip}
                output_old,_ = model(batch)
                output_old = output_old.view(-1)
                # tru = batch['probe_target'].view(-1)
                # all_true.append(tru.detach().cpu())
                all_pred.append(output_old.detach().cpu())
            break
    # all_true = torch.cat(all_true, dim=0).numpy()
    all_pred = torch.cat(all_pred, dim=0).numpy()
    return all_pred
def write_chgcar_via_mpid(input_array,atoms,nx,ny,nz,obj_name):
    datas   = input_array
    density = datas.reshape(nx, ny, nz)
    aug_chgcar_file = None
    # retrieve augmentation, if requested
    if aug_chgcar_file is not None:
        aug = Chgcar.from_file(aug_chgcar_file).data_aug
    else:
        aug = None
    # extract spin, if available
    if len(density.shape) == 4:  # implies a spin channel exists
        charge_grid = density[..., 0]
        spin_grid = density[..., 1]
    else:
        charge_grid = density
        spin_grid = np.zeros_like(density)
    # create Chgcar object
    vcd = VaspChargeDensity(filename=None)
    vcd.atoms.append(atoms)
    vcd.chg.append(charge_grid)
    #vcd.chgdiff.append(spin_grid)
    if aug is not None:
        vcd.aug = "".join(aug["total"])
        vcd.augdiff = "".join(aug["diff"])
    vcd.write(f'{obj_name}.vasp', format="chgcar")

for dataid in tqdm(range(1,total_count),total=total_count):
    # dataid = 122
    db_ = connect(mysql_url)
    rows = []
    for row in db_.select(dataset[dataid]):
        rows.append(row)
    atoms = rows[0].toatoms()
    try:
        try:
            fname = rows[0].data["mpid"]
        except:
            fname = rows[0].mpid
    except:
        fname = f"dataid_pbc_{dataid}"
    print(f"Processing {fname}")
    pred_array  = pred_full_chgdensity(dataid)
    nx, ny, nz = rows[0].data.nx, rows[0].data.ny, rows[0].data.nz
    # nx, ny, nz = 144,144,448
    # Output directory - use relative path
    output_dir = os.path.join(deepaw_root, 'outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, fname)
    _ = write_chgcar_via_mpid(pred_array, atoms, nx, ny, nz, output_path)
    print(f"Saved prediction to: {output_path}")
