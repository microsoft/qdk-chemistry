"""
Plot Pareto front comparison: Sparse vs Dense MPS for Fe2S2 (grayscale).

Reads: mps_sparse_resource_estimation_fe2s2_phase_bits_15_full.json

Grayscale scheme (shape-based):
  - Sparse: circle ('o')
  - Dense: diamond ('D')
  - MPS only: filled marker
  - MPS + QPE: open marker (facecolor='none')
  - QPE(HF): 'x' marker
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_path = Path(__file__).parent
json_path = base_path / "mps_sparse_resource_estimation_fe2s2_phase_bits_15.json"

with open(json_path) as f:
    data = json.load(f)

fe = data["Fe2S2-20"]

# ============================================================================
# Grayscale marker scheme
# ============================================================================

MARKER_SPARSE = "o"   # circle for sparse
MARKER_DENSE = "D"    # diamond for dense
MARKER_QPE_HF = "x"  # x for QPE(HF)

MARKER_SIZE = 90
ALPHA = 0.8
COLOR = "black"
FILL_COLOR = "0.5"  # grey for filled markers

# ============================================================================
# Plot
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# --- QPE(HF) --- x marker
entry = fe.get("qpe_hf")
if entry and "pareto_front" in entry:
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    ax.scatter(qubits, runtime, marker=MARKER_QPE_HF, color=COLOR,
               s=MARKER_SIZE, alpha=ALPHA, linewidths=1.5,
               zorder=4, label="QPE(HF)")

# --- Sparse MPS only --- filled circle
entry = fe.get("sparse_mps_only")
if entry and "pareto_front" in entry:
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    ax.scatter(qubits, runtime, marker=MARKER_SPARSE, color=FILL_COLOR,
               s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.0,
               zorder=3, label="Sparse MPS State Prep")

# --- Sparse MPS + QPE --- open circle
entry = fe.get("sparse_mps_plus_qpe")
if entry and "pareto_front" in entry:
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    ax.scatter(qubits, runtime, marker=MARKER_SPARSE, facecolors="none",
               s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.5,
               zorder=3, label="Sparse MPS + QPE")

# --- Dense MPS only --- filled diamond
entry = fe.get("dense_mps_only")
if entry and "pareto_front" in entry:
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    ax.scatter(qubits, runtime, marker=MARKER_DENSE, color=FILL_COLOR,
               s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.0,
               zorder=3, label="Dense MPS State Prep")

# --- Dense MPS + QPE --- open diamond
entry = fe.get("dense_mps_plus_qpe")
if entry and "pareto_front" in entry:
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    ax.scatter(qubits, runtime, marker=MARKER_DENSE, facecolors="none",
               s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.5,
               zorder=3, label="Dense MPS + QPE")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Physical Qubits", fontsize=13)
ax.set_ylabel("Runtime (s)", fontsize=13)
ax.set_title("Fe2S2-20: Sparse vs Dense MPS Pareto Front\n Bond Dimensions: 1000", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="best")
ax.grid(True, which="both", alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
out_path = base_path / "pareto_fe2s2_sparse_vs_dense.png"
fig.savefig(out_path, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
