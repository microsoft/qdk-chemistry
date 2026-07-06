"""
Plot Pareto fronts (physical qubits vs runtime) for all molecules (grayscale).

Reads:
  - mps_sossa_resource_estimation*.json (Fe2S2-20, Fe4S4-36, FeMoCo-76)

Layout: one subplot per molecule.
Legend scheme:
  - Shape distinguishes bond dimension: o=100, s=1000, D=5000, ^=10000
  - MPS only: filled grey
  - MPS+QPE: open (hollow) shape
  - QPE(HF): x marker
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

# ============================================================================
# Load data from both JSON files
# ============================================================================

json_files = [
    base_path / "mps_sossa_resource_estimation_full.json",
]

all_mol_data = {}  # mol_name -> {mps_only: [...], qpe_hf: ..., mps_plus_qpe: [...]}

for jp in json_files:
    if not jp.exists():
        print(f"WARNING: {jp} not found, skipping.")
        continue
    with open(jp) as f:
        data = json.load(f)
    for key in data:
        if key == "parameters":
            continue
        mol = data[key]
        # Only include molecules with actual Pareto data
        has_data = False
        if mol.get("mps_only"):
            for e in mol["mps_only"]:
                if "pareto_front" in e:
                    has_data = True
                    break
        if mol.get("qpe_hf") and "pareto_front" in mol["qpe_hf"]:
            has_data = True
        if not has_data:
            continue
        all_mol_data[key] = mol

print(f"Molecules with Pareto data: {list(all_mol_data.keys())}")

# ============================================================================
# Grayscale marker scheme
# ============================================================================

# Different shapes by bond dimension
CHI_MARKERS = {
    100: "o",       # circle
    1000: "s",      # square
    5000: "D",      # diamond
    10000: "^",     # triangle up
}

FILL_COLOR = "0.5"   # grey for filled (MPS only)
EDGE_COLOR = "black"
QPE_MARKER = "x"     # QPE(HF)
MARKER_SIZE = 80

# ============================================================================
# Plot
# ============================================================================

n_mols = len(all_mol_data)
if n_mols == 0:
    print("No Pareto data found.")
    exit()

nrows, ncols = 1, n_mols
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * n_mols, 6), squeeze=False)

for idx, (mol_name, mol) in enumerate(all_mol_data.items()):
    ax = axes[0, idx]

    # --- MPS only --- filled grey
    for entry in mol.get("mps_only", []):
        if "pareto_front" not in entry:
            continue
        if entry.get("label", "").startswith("real"):
            continue
        chi = entry["bond_dim"]
        marker = CHI_MARKERS.get(chi, "o")
        pareto = entry["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=marker, color=FILL_COLOR, s=MARKER_SIZE,
                   edgecolors=EDGE_COLOR, linewidths=0.8, zorder=3, alpha=0.8,
                   label=f"MPS χ={chi}")

    # --- QPE(HF) --- x marker
    qpe_hf = mol.get("qpe_hf")
    if qpe_hf and "pareto_front" in qpe_hf:
        pareto = qpe_hf["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=QPE_MARKER, color=EDGE_COLOR, s=MARKER_SIZE,
                   linewidths=1.5, zorder=4, alpha=0.8,
                   label="QPE-HF")

    # --- MPS + QPE --- open shape
    for entry in mol.get("mps_plus_qpe", []):
        if "pareto_front" not in entry:
            continue
        if entry.get("label", "").startswith("real"):
            continue
        chi = entry["bond_dim"]
        marker = CHI_MARKERS.get(chi, "o")
        pareto = entry["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=marker, facecolors="none",
                   s=MARKER_SIZE, edgecolors="0.2", linewidths=1.5,
                   zorder=3, alpha=0.7,
                   label=f"QPE-MPS χ={chi}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Qubits", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title(mol_name, fontsize=13, fontweight="bold")
    if idx == 0:
        ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

# Hide unused subplots
for idx in range(n_mols, ncols):
    axes[0, idx].set_visible(False)

fig.suptitle("Pareto Fronts: Physical Qubits vs Runtime", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = base_path / "pareto_all_molecules.png"
fig.savefig(out_path, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
