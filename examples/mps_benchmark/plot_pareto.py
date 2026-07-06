"""
Plot Pareto fronts (physical qubits vs runtime) for all molecules.

Reads:
  - mps_sossa_resource_estimation.json (Fe2S2-20, Fe4S4-36, FeMoCo-76)
  - mps_sossa_resource_estimation_g1.json (P450-G1-43)

Layout: one subplot per molecule.
Legend scheme:
  - Marker type distinguishes method: 'o' = MPS only, '^' = QPE(HF), 's' = MPS+QPE
  - Color distinguishes bond dimension: tab:blue=100, tab:orange=1000, tab:red=5000, tab:purple=10000
  - QPE(HF) always uses tab:green with '^' marker
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
    base_path / "mps_sossa_resource_estimation.json",
    base_path / "mps_sossa_resource_estimation_g1.json",
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
# Color/marker scheme
# ============================================================================

# Colors by bond dimension
CHI_COLORS = {
    100: "tab:blue",
    1000: "tab:orange",
    5000: "tab:red",
    10000: "tab:purple",
}

# Markers by method
MARKER_MPS = "o"        # MPS only
MARKER_QPE_HF = "^"    # QPE(HF)
MARKER_MPS_QPE = "s"   # MPS + QPE

QPE_HF_COLOR = "tab:green"

# ============================================================================
# Plot
# ============================================================================

n_mols = len(all_mol_data)
if n_mols == 0:
    print("No Pareto data found.")
    exit()

nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), squeeze=False)

for idx, (mol_name, mol) in enumerate(all_mol_data.items()):
    row, col = divmod(idx, ncols)
    ax = axes[row, col]

    # --- MPS only ---
    for entry in mol.get("mps_only", []):
        if "pareto_front" not in entry:
            continue
        chi = entry["bond_dim"]
        color = CHI_COLORS.get(chi, "tab:gray")
        pareto = entry["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=MARKER_MPS, color=color, s=80,
                   edgecolors="k", linewidths=0.5, zorder=3, alpha=0.6,
                   label=f"MPS χ={chi}")

    # --- QPE(HF) ---
    qpe_hf = mol.get("qpe_hf")
    if qpe_hf and "pareto_front" in qpe_hf:
        pareto = qpe_hf["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=MARKER_QPE_HF, color=QPE_HF_COLOR, s=100,
                   edgecolors="k", linewidths=0.5, zorder=4, alpha=0.6,
                   label="QPE(HF)")

    # --- MPS + QPE ---
    for entry in mol.get("mps_plus_qpe", []):
        if "pareto_front" not in entry:
            continue
        chi = entry["bond_dim"]
        color = CHI_COLORS.get(chi, "tab:gray")
        pareto = entry["pareto_front"]
        qubits = [p["qubits"] for p in pareto]
        runtime = [p["runtime"] for p in pareto]
        ax.scatter(qubits, runtime, marker=MARKER_MPS_QPE, color=color, s=80,
                   edgecolors="k", linewidths=0.5, zorder=3, alpha=0.6,
                   label=f"MPS+QPE χ={chi}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Qubits", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title(mol_name, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

# Hide unused subplots
for idx in range(n_mols, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row, col].set_visible(False)

fig.suptitle("Pareto Fronts: Physical Qubits vs Runtime", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path = base_path / "pareto_all_molecules.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
