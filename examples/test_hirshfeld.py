#!/usr/bin/env python
"""
Test Hirshfeld charge analysis on HfO2.

Steps:
  1. Precompute radial density lookup table for Hf and O.
  2. Run Hirshfeld analysis in "lookup" mode.
  3. Run Hirshfeld analysis in "onthefly" mode.
  4. Compare results from both modes.
"""

import os
import sys
import logging
import numpy as np

# Ensure project root is on the path
script_dir = os.path.dirname(os.path.abspath(__file__))
deepaw_root = os.path.abspath(os.path.join(script_dir, ".."))
if deepaw_root not in sys.path:
    sys.path.insert(0, deepaw_root)

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from ase.db import connect
from deepaw.inference import InferenceEngine
from deepaw.hirshfeld import HirshfeldAnalysis
from deepaw.hirshfeld.radial_lookup import build_radial_lookup

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = os.path.join(deepaw_root, "examples", "hfo2_chgd.db")
LOOKUP_TABLE_PATH = os.path.join(deepaw_root, "outputs", "hfo2_radial_table.npz")
ENCUT = 500.0

# ── Load structure ─────────────────────────────────────────────────────────
print("=" * 70)
print("HfO2 Hirshfeld Charge Analysis Test")
print("=" * 70)

db = connect(DB_PATH)
row = db.get(1)
atoms = row.toatoms()
grid_shape = (row.data["nx"], row.data["ny"], row.data["nz"])

print(f"Structure: {row.formula}")
print(f"Atoms: {len(atoms)}")
print(f"Symbols: {atoms.get_chemical_symbols()}")
print(f"Grid shape (from DB): {grid_shape}")

# ── Initialise inference engine ────────────────────────────────────────────
print("\nInitialising InferenceEngine ...")
engine = InferenceEngine(use_dual_model=True)

# ── Step 1: Precompute radial lookup table ─────────────────────────────────
print("\n" + "=" * 70)
print("Step 1: Precompute radial density lookup table for Hf and O")
print("=" * 70)

os.makedirs(os.path.dirname(LOOKUP_TABLE_PATH), exist_ok=True)

elements = sorted(set(atoms.get_chemical_symbols()))
print(f"Elements to precompute: {elements}")

lookup = build_radial_lookup(
    elements=elements,
    engine=engine,
    cell_size=10.0,
    encut=ENCUT,
    bin_width=0.02,
    max_radius=5.0,
)
lookup.save(LOOKUP_TABLE_PATH)
print(f"Saved lookup table to: {LOOKUP_TABLE_PATH}")

# Quick sanity check on radial profiles
for elem in elements:
    r, rho = lookup._data[elem]
    print(f"  {elem}: {len(r)} bins, rho_max={rho.max():.6f}, rho at r=0.01={rho[0]:.6f}")

# ── Step 2: Hirshfeld analysis – lookup mode ──────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Hirshfeld analysis (lookup mode)")
print("=" * 70)

ha_lookup = HirshfeldAnalysis(
    engine=engine,
    mode="lookup",
    lookup_path=LOOKUP_TABLE_PATH,
    encut=ENCUT,
)
result_lookup = ha_lookup.analyze(atoms, grid_shape=grid_shape)
print(result_lookup)

# ── Step 3: Hirshfeld analysis – onthefly mode ────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Hirshfeld analysis (onthefly mode)")
print("=" * 70)

ha_onthefly = HirshfeldAnalysis(
    engine=engine,
    mode="onthefly",
    encut=ENCUT,
)
result_onthefly = ha_onthefly.analyze(atoms, grid_shape=grid_shape)
print(result_onthefly)

# ── Step 4: Compare results ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4: Comparison of lookup vs onthefly")
print("=" * 70)

print(f"\n{'Index':>6}  {'Species':>8}  {'Lookup':>12}  {'Onthefly':>12}  {'Diff':>12}")
print("-" * 60)
for i, sym in enumerate(result_lookup.symbols):
    q_l = result_lookup.charges[i]
    q_o = result_onthefly.charges[i]
    diff = q_l - q_o
    print(f"{i:>6d}  {sym:>8s}  {q_l:>12.6f}  {q_o:>12.6f}  {diff:>12.6f}")

print("-" * 60)
print(f"{'Total':>17s}  {result_lookup.total_charge:>12.6f}  "
      f"{result_onthefly.total_charge:>12.6f}  "
      f"{result_lookup.total_charge - result_onthefly.total_charge:>12.6f}")
print(f"{'Deform. int.':>17s}  {result_lookup.deformation_integral:>12.6f}  "
      f"{result_onthefly.deformation_integral:>12.6f}")

max_diff = np.max(np.abs(result_lookup.charges - result_onthefly.charges))
print(f"\nMax absolute charge difference: {max_diff:.6f}")

# ── Validation checks ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Validation")
print("=" * 70)

# Check 1: Charge conservation (total charge ~ 0 for neutral system)
print(f"[Lookup]   Total charge: {result_lookup.total_charge:.6f}  "
      f"(should be ~0 for neutral system)")
print(f"[Onthefly] Total charge: {result_onthefly.total_charge:.6f}")

# Check 2: Hf should be positive (cation), O should be negative (anion)
hf_charges_l = [result_lookup.charges[i] for i, s in enumerate(result_lookup.symbols) if s == "Hf"]
o_charges_l = [result_lookup.charges[i] for i, s in enumerate(result_lookup.symbols) if s == "O"]
print(f"\n[Lookup] Hf charges: {[f'{q:.4f}' for q in hf_charges_l]}")
print(f"[Lookup] O  charges: {[f'{q:.4f}' for q in o_charges_l]}")

hf_positive = all(q > 0 for q in hf_charges_l)
o_negative = all(q < 0 for q in o_charges_l)
print(f"\nHf positive (cation)? {'YES' if hf_positive else 'NO'}")
print(f"O  negative (anion)?  {'YES' if o_negative else 'NO'}")

# Check 3: Equivalent atoms should have similar charges
if len(hf_charges_l) > 1:
    hf_std = np.std(hf_charges_l)
    print(f"\nHf charge std dev: {hf_std:.6f} (should be small for equivalent sites)")
if len(o_charges_l) > 1:
    o_std = np.std(o_charges_l)
    print(f"O  charge std dev: {o_std:.6f} (should be small for equivalent sites)")

# ── Save persistent output ────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(deepaw_root, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

result_lookup.save(os.path.join(OUTPUT_DIR, "hfo2_hirshfeld_lookup.txt"))
result_onthefly.save(os.path.join(OUTPUT_DIR, "hfo2_hirshfeld_onthefly.txt"))
print(f"\nResults saved to:")
print(f"  {os.path.join(OUTPUT_DIR, 'hfo2_hirshfeld_lookup.txt')}")
print(f"  {os.path.join(OUTPUT_DIR, 'hfo2_hirshfeld_onthefly.txt')}")

print("\n" + "=" * 70)
print("Test complete!")
print("=" * 70)
