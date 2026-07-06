"""
Plot runtime and physical qubits vs bond dimension for all molecules.

Reads data from mps_sossa_resource_estimation_*.json files:
  - Fe2S2-20, Fe4S4-36, FeMoCo-76

Produces a single figure with 3 rows x N columns (one column per molecule):
  - Top row: Runtime vs bond dimension
  - Middle row: Toffoli count vs bond dimension
  - Bottom row: Physical qubits vs bond dimension
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

base_path = Path(__file__).parent

JSON_FILES = [
    base_path / "mps_sossa_resource_estimation_fe2s2.json",
    base_path / "mps_sossa_resource_estimation_fe4s4.json",
    base_path / "mps_sossa_resource_estimation_femoco.json",
    base_path / "mps_sossa_resource_estimation_g1.json"
]

TARGET_CHIS = [100, 1000, 5000, 10000]


def ns_to_seconds(ns):
    return ns / 1e9


# ============================================================================
# Load data for all molecules
# ============================================================================

mol_data = []

for json_path in JSON_FILES:
    if not json_path.exists():
        print(f"WARNING: {json_path} not found, skipping.")
        continue

    with open(json_path) as f:
        data = json.load(f)

    mol_key = [k for k in data.keys() if k != "parameters"][0]
    mol = data[mol_key]
    mol_label = mol_key

    print(f"\n{'='*60}")
    print(f"  {mol_label}")
    print(f"{'='*60}")

    # Extract MPS-only data
    mps_only_runtime = {}
    mps_only_qubits = {}
    mps_only_toffoli = {}
    for entry in mol.get("mps_only", []):
        label = entry["label"]
        if label.startswith("real"):
            continue
        chi = entry["bond_dim"]
        if chi in TARGET_CHIS:
            mps_only_runtime[chi] = ns_to_seconds(entry["runtime_ns"])
            mps_only_qubits[chi] = entry["physical_qubits"]
            mps_only_toffoli[chi] = entry["logical_toffoli"]

    # Extract MPS + QPE data
    mps_qpe_runtime = {}
    mps_qpe_qubits = {}
    mps_qpe_toffoli = {}
    for entry in mol.get("mps_plus_qpe", []):
        label = entry["label"]
        if label.startswith("real"):
            continue
        chi = entry["bond_dim"]
        if chi in TARGET_CHIS:
            mps_qpe_runtime[chi] = ns_to_seconds(entry["runtime_ns"])
            mps_qpe_qubits[chi] = entry["physical_qubits"]
            mps_qpe_toffoli[chi] = entry["total_toffoli"]

    # QPE (HF) baseline
    qpe_hf = mol.get("qpe_hf")
    qpe_hf_runtime = ns_to_seconds(qpe_hf["runtime_ns"]) if qpe_hf else None
    qpe_hf_qubits = qpe_hf["physical_qubits"] if qpe_hf else None
    qpe_hf_toffoli = qpe_hf["logical_toffoli"] if qpe_hf else None

    # Prepare sorted arrays
    mps_chis = sorted(mps_only_runtime.keys())
    mps_rt = [mps_only_runtime[c] for c in mps_chis]
    mps_qb = [mps_only_qubits[c] for c in mps_chis]
    mps_tf = [mps_only_toffoli[c] for c in mps_chis]

    qpe_chis = sorted(mps_qpe_runtime.keys())
    qpe_rt = [mps_qpe_runtime[c] for c in qpe_chis]
    qpe_qb = [mps_qpe_qubits[c] for c in qpe_chis]
    qpe_tf = [mps_qpe_toffoli[c] for c in qpe_chis]

    if not mps_chis:
        print(f"  No MPS-only data for target bond dims, skipping.")
        continue

    print(f"  MPS only:  χ={mps_chis}")
    print(f"             runtime(s) = {[f'{r:.1f}' for r in mps_rt]}")
    print(f"             qubits     = {mps_qb}")
    if qpe_chis:
        print(f"  MPS+QPE:   χ={qpe_chis}")
        print(f"             runtime(s) = {[f'{r:.1f}' for r in qpe_rt]}")
        print(f"             qubits     = {qpe_qb}")
    if qpe_hf_runtime:
        print(f"  QPE(HF):   runtime = {qpe_hf_runtime:.1f} s, qubits = {qpe_hf_qubits:,}")

    mol_data.append({
        "label": mol_label,
        "mps_chis": mps_chis,
        "mps_rt": mps_rt,
        "mps_qb": mps_qb,
        "mps_tf": mps_tf,
        "qpe_chis": qpe_chis,
        "qpe_rt": qpe_rt,
        "qpe_qb": qpe_qb,
        "qpe_tf": qpe_tf,
        "qpe_hf_runtime": qpe_hf_runtime,
        "qpe_hf_qubits": qpe_hf_qubits,
        "qpe_hf_toffoli": qpe_hf_toffoli,
    })

# ============================================================================
# Create combined figure: 2 rows x N columns
# ============================================================================

n_mols = len(mol_data)
if n_mols == 0:
    print("No data to plot.")
    exit()

fig, axes = plt.subplots(3, n_mols, figsize=(6 * n_mols, 14), squeeze=False)

for col, md in enumerate(mol_data):
    ax_rt = axes[0, col]
    ax_tf = axes[1, col]
    ax_qb = axes[2, col]

    # --- Top row: Runtime ---
    ax_rt.plot(md["mps_chis"], md["mps_rt"], "o-", color="tab:blue",
               markersize=7, linewidth=2, label="MPS State Prep Only")
    if md["qpe_chis"]:
        ax_rt.plot(md["qpe_chis"], md["qpe_rt"], "s-", color="tab:red",
                   markersize=7, linewidth=2, label="MPS + SOSSA QPE")
    if md["qpe_hf_runtime"]:
        ax_rt.axhline(md["qpe_hf_runtime"], color="tab:green", linestyle="--",
                      linewidth=2, label=f"QPE(HF) = {md['qpe_hf_runtime']:.0f} s")

    ax_rt.set_xscale("log")
    ax_rt.set_yscale("log")
    ax_rt.set_xlabel("Bond Dimension (χ)", fontsize=11)
    ax_rt.set_ylabel("Runtime (s)", fontsize=11)
    ax_rt.set_title(f"{md['label']}", fontsize=12, fontweight="bold")
    ax_rt.legend(fontsize=9)
    ax_rt.grid(True, which="both", alpha=0.3)
    ax_rt.set_xticks(md["mps_chis"])
    ax_rt.set_xticklabels([str(c) for c in md["mps_chis"]])

    # --- Middle row: Toffoli Count ---
    ax_tf.plot(md["mps_chis"], md["mps_tf"], "o-", color="tab:blue",
               markersize=7, linewidth=2, label="MPS State Prep Only")
    if md["qpe_chis"]:
        ax_tf.plot(md["qpe_chis"], md["qpe_tf"], "s-", color="tab:red",
                   markersize=7, linewidth=2, label="MPS + SOSSA QPE")
    if md["qpe_hf_toffoli"]:
        ax_tf.axhline(md["qpe_hf_toffoli"], color="tab:green", linestyle="--",
                      linewidth=2, label=f"QPE(HF) = {md['qpe_hf_toffoli']:,}")

    ax_tf.set_xscale("log")
    ax_tf.set_yscale("log")
    ax_tf.set_xlabel("Bond Dimension (χ)", fontsize=11)
    ax_tf.set_ylabel("Toffoli Count", fontsize=11)
    ax_tf.legend(fontsize=9)
    ax_tf.grid(True, which="both", alpha=0.3)
    ax_tf.set_xticks(md["mps_chis"])
    ax_tf.set_xticklabels([str(c) for c in md["mps_chis"]])

    # --- Bottom row: Physical Qubits ---
    ax_qb.plot(md["mps_chis"], md["mps_qb"], "o-", color="tab:blue",
               markersize=7, linewidth=2, label="MPS State Prep Only")
    if md["qpe_chis"]:
        ax_qb.plot(md["qpe_chis"], md["qpe_qb"], "s-", color="tab:red",
                   markersize=7, linewidth=2, label="MPS + SOSSA QPE")
    if md["qpe_hf_qubits"]:
        ax_qb.axhline(md["qpe_hf_qubits"], color="tab:green", linestyle="--",
                      linewidth=2, label=f"QPE(HF) = {md['qpe_hf_qubits']:,}")

    ax_qb.set_xscale("log")
    ax_qb.set_yscale("log")
    ax_qb.set_xlabel("Bond Dimension (χ)", fontsize=11)
    ax_qb.set_ylabel("Physical Qubits", fontsize=11)
    ax_qb.legend(fontsize=9)
    ax_qb.grid(True, which="both", alpha=0.3)
    ax_qb.set_xticks(md["mps_chis"])
    ax_qb.set_xticklabels([str(c) for c in md["mps_chis"]])

fig.suptitle("Resource Estimation vs Bond Dimension",
             fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = base_path / "resource_estimation_combined.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nSaved combined figure: {out_path}")

print("\nDone.")
