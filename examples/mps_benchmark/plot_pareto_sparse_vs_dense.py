"""
Plot Pareto front comparison: Sparse vs Dense MPS (grayscale, 1x2 subplots).

Reads:
  - mps_sparse_resource_estimation_fe2s2.json
  - mps_sossa_resource_estimation_g1.json

Grayscale scheme (shape-based):
  - Sparse: circle ('o')
  - Dense: diamond ('D')
  - MPS only: filled marker (grey)
  - MPS + QPE: open marker (facecolor='none')
  - QPE(HF): 'x' marker
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

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


def plot_pareto_on_ax(ax, entry, marker, label, filled=True):
    """Plot a single Pareto series on the given axes."""
    if entry is None or "pareto_front" not in entry:
        return
    pareto = entry["pareto_front"]
    qubits = [p["qubits"] for p in pareto]
    runtime = [p["runtime"] for p in pareto]
    if marker == MARKER_QPE_HF:
        ax.scatter(qubits, runtime, marker=marker, color=COLOR,
                   s=MARKER_SIZE, alpha=ALPHA, linewidths=1.5,
                   zorder=4, label=label)
    elif filled:
        ax.scatter(qubits, runtime, marker=marker, color=FILL_COLOR,
                   s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.0,
                   zorder=3, label=label)
    else:
        ax.scatter(qubits, runtime, marker=marker, facecolors="none",
                   s=MARKER_SIZE, alpha=ALPHA, edgecolors=COLOR, linewidths=1.5,
                   zorder=3, label=label)


def style_ax(ax, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Qubits", fontsize=12)
    ax.set_ylabel("Runtime (s)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)


# ============================================================================
# Load data
# ============================================================================

# --- Fe2S2 ---
with open(base_path / "mps_sparse_resource_estimation_fe2s2.json") as f:
    fe2s2_data = json.load(f)
fe = fe2s2_data["Fe2S2-20"]

# --- P450 G1 ---
with open(base_path / "mps_sequential_vs_sparse_comparison_g1.json") as f:
    g1_data = json.load(f)
g1 = g1_data["datasets"]["g1"]

# ============================================================================
# Plot (1x2 subplots)
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Fe2S2-20 (sparse vs dense) ---
plot_pareto_on_ax(ax1, fe.get("qpe_hf"), MARKER_QPE_HF, "QPE(HF)")
plot_pareto_on_ax(ax1, fe.get("sparse_mps_only"), MARKER_SPARSE, "Sparse MPS State Prep", filled=True)
plot_pareto_on_ax(ax1, fe.get("sparse_mps_plus_qpe"), MARKER_SPARSE, "Sparse MPS + QPE", filled=False)
plot_pareto_on_ax(ax1, fe.get("dense_mps_only"), MARKER_DENSE, "Dense MPS State Prep", filled=True)
plot_pareto_on_ax(ax1, fe.get("dense_mps_plus_qpe"), MARKER_DENSE, "Dense MPS + QPE", filled=False)
style_ax(ax1, "Fe2S2-20\nBond Dimension: 1000")

# --- Right: P450 G1 (sparse vs dense/sequential) ---
plot_pareto_on_ax(ax2, g1.get("sparse"), MARKER_SPARSE, "Sparse MPS State Prep", filled=True)
plot_pareto_on_ax(ax2, g1.get("sequential"), MARKER_DENSE, "Dense MPS State Prep", filled=True)
style_ax(ax2, f"P450-G1-43\nBond Dimension: {g1['max_bond_dim']}")

plt.tight_layout()
out_path = base_path / "pareto_sparse_vs_dense.png"
fig.savefig(out_path, dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
