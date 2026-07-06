"""
MPS State Preparation: Sequential vs Sparse Comparison

Compares resource estimates for MPS state preparation between:
  1. MPSSequentialStatePreparation (dense, CSD + Givens)
  2. MPSSparseStatePreparation (exploits U(1) block-sparsity)

Datasets:
  - Fe2S2-2_small: 20 sites, bond dim ≤ 1000 (from qdk-chemistry/examples/)
  - P450 G-1: 43 sites, bond dim ≤ 1000 (from ressource_estimates/p450_enzyme/G-1/)

Also runs QRE v3 on each circuit for full physical resource estimation.

Usage:
  python mps_compare_sequential_vs_sparse.py
  python mps_compare_sequential_vs_sparse.py --datasets fe2s2
  python mps_compare_sequential_vs_sparse.py --datasets g1
  python mps_compare_sequential_vs_sparse.py --datasets all --phase-bits 12
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from qdk.qre import estimate, plot_estimates, PSSPC, LatticeSurgery
from qdk.qre.instruction_ids import LATTICE_SURGERY
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk.qre.property_keys import DISTANCE, NUM_TS_PER_ROTATION, PHYSICAL_COMPUTE_QUBITS
from qdk_chemistry.algorithms.state_preparation import MPSSequentialStatePreparation
from qdk_chemistry.algorithms.state_preparation.mps_sparse import MPSSparseStatePreparation
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.utils import Logger

Logger.set_global_level(Logger.LogLevel.off)

# ============================================================================
# Dataset definitions
# ============================================================================

DATASETS = {
    "g1": {
        "label": "P450 G-1 (43 sites)",
        "path": Path("/blob_user/ressource_estimates/p450_enzyme/G-1/tensors_compressed") / "tensors_compressed_",
        "description": "Cytochrome P450 compound G-1, 43 spatial orbitals, chi_max~1000",
    },
}

# QRE architecture
ARCHITECTURE = Majorana(error_rate=1e-5)
ISA_QUERY = ThreeAux.q() * RoundBasedFactory.q(use_cache=True, code_query=ThreeAux.q())
TRACE_QUERY = (
    PSSPC.q()
    * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 10)])
)
MAX_ERROR = 0.01

OUTPUT_JSON = Path("mps_sequential_vs_sparse_comparison_g1.json")


# ============================================================================
# Helpers
# ============================================================================


def load_mps_from_compressed(path_prefix):
    """Load MPS tensors from compressed .npz files."""
    n_tensors = len(glob.glob(str(path_prefix) + "[0-9]*.npz"))
    if n_tensors == 0:
        raise FileNotFoundError(f"No tensors found at {path_prefix}*.npz")
    sparse_tensors = [load_npz(f"{path_prefix}{i}.npz") for i in range(n_tensors)]
    dense_tensors = [t.toarray() for t in sparse_tensors]
    return MPSWavefunction(dense_tensors)


def run_qre(circuit, name):
    """Run QRE v3 and return (extra_dict, qre_result_object)."""
    app = circuit.get_qre_application()
    qre_result = estimate(app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR, name=name)
    r = qre_result[0]

    extra = {
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        "compute_distance": int(r.source[LATTICE_SURGERY].instruction[DISTANCE]),
        "compute_qubits": int(r.properties[PHYSICAL_COMPUTE_QUBITS]),
        "num_ts_per_rotation": int(r.properties[NUM_TS_PER_ROTATION]),
        "factories": [],
    }
    for fid, factory_result in r.factories.items():
        extra["factories"].append({
            "copies": factory_result.copies,
            "runs": factory_result.runs,
            "error_rate": factory_result.error_rate,
            "states": factory_result.states,
        })

    # Add factory summary column
    qre_result.add_factory_summary_column()
    df = qre_result.as_frame()
    if "factories" in df.columns:
        extra["factory_summary"] = str(df["factories"].iloc[0])

    return extra, qre_result


def _serialize_pareto(qre_result):
    """Extract Pareto front data from a QRE result into a JSON-serializable list."""
    qre_result.add_factory_summary_column()
    df = qre_result.as_frame()
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                record[col] = float(val)
            elif isinstance(val, (np.bool_,)):
                record[col] = bool(val)
            elif hasattr(val, "total_seconds"):
                # Convert pandas Timedelta to seconds (float)
                record[col] = val.total_seconds()
            else:
                record[col] = val
        records.append(record)
    return records


def _save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   [saved to {path}]")


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Sequential vs Sparse MPS State Preparation.",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["all"],
        choices=list(DATASETS.keys()) + ["all"],
        help="Which datasets to run. Default: all",
    )
    parser.add_argument(
        "--phase-bits", "-p",
        type=int,
        default=15,
        help="Number of rotation bits for phase gradient. Default: 15",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_JSON,
        help=f"Output JSON path. Default: {OUTPUT_JSON}",
    )
    parser.add_argument(
        "--skip-qre",
        action="store_true",
        help="Skip QRE v3 estimation (only compute logical counts).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip computation; load results from JSON and regenerate plots only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if "all" in args.datasets:
        selected = list(DATASETS.keys())
    else:
        selected = args.datasets

    phase_bits = args.phase_bits
    output_json = args.output

    # ------------------------------------------------------------------
    # Plot-only mode: load JSON and regenerate Pareto plots
    # ------------------------------------------------------------------
    if args.plot_only:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not output_json.exists():
            print(f"ERROR: {output_json} not found. Run without --plot-only first.")
            raise SystemExit(1)

        with open(output_json) as f:
            saved = json.load(f)

        for ds_key, ds_result in saved["datasets"].items():
            if ds_key not in selected and "all" not in args.datasets:
                continue
            seq_pareto = ds_result.get("sequential", {}).get("pareto_front")
            sparse_pareto = ds_result.get("sparse", {}).get("pareto_front")
            if not seq_pareto or not sparse_pareto:
                print(f"   No Pareto data for {ds_key}, skipping.")
                continue

            df_seq = pd.DataFrame(seq_pareto)
            df_sparse = pd.DataFrame(sparse_pareto)

            fig, ax = plt.subplots(figsize=(15, 9.5))
            ax.scatter(df_seq["qubits"], df_seq["runtime"], marker="x", s=120, linewidths=2.5, label=f"{ds_result['label']} Sequential")
            ax.scatter(df_sparse["qubits"], df_sparse["runtime"], marker="x", s=120, linewidths=2.5, label=f"{ds_result['label']} Sparse")
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import ScalarFormatter
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
                axis.get_major_formatter().set_scientific(False)
                axis.get_major_formatter().set_useOffset(False)
            ax.set_xlabel("Physical qubits", fontsize=30)
            ax.set_ylabel("Runtime (s)", fontsize=30)
            ax.tick_params(axis="both", which="major", labelsize=26)
            ax.tick_params(axis="both", which="minor", labelsize=24)
            ax.legend(fontsize=20, loc="best")
            ax.set_title(f"{ds_result['label']}: Dense vs Sparse Pareto", fontsize=28)

            fig_path = output_json.parent / f"pareto_{ds_key}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Pareto plot saved to: {fig_path}")

        print("Done (plot-only mode).")
        raise SystemExit(0)

    print("\nMPS Sequential vs Sparse State Preparation Comparison")
    print("=" * 70)
    print(f"   Datasets: {selected}")
    print(f"   Phase bits: {phase_bits}")
    print(f"   QRE: {'enabled' if not args.skip_qre else 'disabled'}")
    print(f"   Output: {output_json}")
    print()

    all_results = {
        "parameters": {
            "phase_bits": phase_bits,
            "architecture": "Majorana(error_rate=1e-5)",
            "isa": "ThreeAux + RoundBasedFactory",
            "max_error": MAX_ERROR,
        },
        "datasets": {},
    }

    for ds_key in selected:
        ds = DATASETS[ds_key]
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds['label']}")
        print(f"#   {ds['description']}")
        print(f"{'#'*70}")

        # Load tensors
        t0 = time.perf_counter()
        try:
            mps = load_mps_from_compressed(ds["path"])
        except FileNotFoundError as e:
            print(f"   ERROR: {e}")
            print("   Skipping this dataset.")
            continue
        t_load = time.perf_counter() - t0

        print(f"   Loaded: {mps.num_sites} sites, {mps.num_qubits} qubits")
        print(f"   Bond dims: {mps.bond_dims}")
        print(f"   Max bond dim: {max(mps.bond_dims)}")
        print(f"   Load time: {t_load:.2f}s")

        ds_result = {
            "label": ds["label"],
            "num_sites": mps.num_sites,
            "num_qubits": mps.num_qubits,
            "bond_dims": mps.bond_dims,
            "max_bond_dim": max(mps.bond_dims),
            "sequential": None,
            "sparse": None,
        }

        # -----------------------------------------------------------
        # Sequential (Dense) State Preparation
        # -----------------------------------------------------------
        print(f"\n   --- Sequential (Dense) State Preparation ---")
        t0 = time.perf_counter()

        algo_seq = MPSSequentialStatePreparation()
        algo_seq.settings().set("rotation_bits", phase_bits)
        algo_seq.settings().set("fast_grouped_resource_estimation", True)

        circuit_seq = algo_seq.run(mps)
        lc_seq = circuit_seq.estimate().logical_counts
        t_seq = time.perf_counter() - t0

        print(f"   Toffoli (CCZ): {lc_seq['cczCount']:,}")
        print(f"   Logical qubits: {lc_seq['numQubits']:,}")
        print(f"   Time: {t_seq:.2f}s")

        seq_result = {
            "logical_toffoli": lc_seq["cczCount"],
            "logical_qubits": lc_seq["numQubits"],
            "build_time_s": round(t_seq, 2),
        }

        qre_result_seq = None
        if not args.skip_qre:
            print("   Running QRE v3...")
            t0 = time.perf_counter()
            qre_extra, qre_result_seq = run_qre(circuit_seq, f"{ds['label']} Sequential")
            t_qre = time.perf_counter() - t0
            seq_result.update(qre_extra)
            seq_result["qre_time_s"] = round(t_qre, 2)
            seq_result["pareto_front"] = _serialize_pareto(qre_result_seq)
            print(f"   Physical qubits: {qre_extra['physical_qubits']:,}")
            print(f"   Runtime: {qre_extra['runtime_ns']}")
            print(f"   Compute distance: {qre_extra['compute_distance']}")
            print(f"   Factory: {qre_extra.get('factory_summary', 'N/A')}")
            print(f"   QRE time: {t_qre:.2f}s")

        ds_result["sequential"] = seq_result

        # -----------------------------------------------------------
        # Sparse State Preparation
        # -----------------------------------------------------------
        print(f"\n   --- Sparse (Block-Sparsity) State Preparation ---")
        t0 = time.perf_counter()

        algo_sparse = MPSSparseStatePreparation()
        algo_sparse.settings().set("rotation_bits", phase_bits)
        circuit_sparse = algo_sparse.run(mps)
        lc_sparse = circuit_sparse.estimate().logical_counts
        t_sparse = time.perf_counter() - t0

        print(f"   Toffoli (CCZ): {lc_sparse['cczCount']:,}")
        print(f"   Logical qubits: {lc_sparse['numQubits']:,}")
        print(f"   Time: {t_sparse:.2f}s")

        sparse_result = {
            "logical_toffoli": lc_sparse["cczCount"],
            "logical_qubits": lc_sparse["numQubits"],
            "build_time_s": round(t_sparse, 2),
        }

        qre_result_sparse = None
        if not args.skip_qre:
            print("   Running QRE v3...")
            t0 = time.perf_counter()
            qre_extra, qre_result_sparse = run_qre(circuit_sparse, f"{ds['label']} Sparse")
            t_qre = time.perf_counter() - t0
            sparse_result.update(qre_extra)
            sparse_result["qre_time_s"] = round(t_qre, 2)
            sparse_result["pareto_front"] = _serialize_pareto(qre_result_sparse)
            print(f"   Physical qubits: {qre_extra['physical_qubits']:,}")
            print(f"   Runtime: {qre_extra['runtime_ns']}")
            print(f"   Compute distance: {qre_extra['compute_distance']}")
            print(f"   Factory: {qre_extra.get('factory_summary', 'N/A')}")
            print(f"   QRE time: {t_qre:.2f}s")

        ds_result["sparse"] = sparse_result

        # -----------------------------------------------------------
        # Pareto plot: Dense vs Sparse
        # -----------------------------------------------------------
        if not args.skip_qre and qre_result_seq is not None and qre_result_sparse is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plot_estimates(
                [qre_result_seq, qre_result_sparse],
                runtime_unit="s",
                figsize=(15, 9.5),
                scatter_args={"marker": "x", "s": 120, "linewidths": 2.5},
            )
            ax = fig.axes[0]
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xlabel(ax.get_xlabel(), fontsize=30)
            ax.set_ylabel(ax.get_ylabel(), fontsize=30)
            ax.tick_params(axis="both", labelsize=26)
            ax.legend(fontsize=26, loc="upper right")
            ax.set_title(f"{ds['label']}: Dense vs Sparse Pareto", fontsize=28)

            fig_path = output_json.with_suffix("") / f"pareto_{ds_key}.png" if output_json.suffix == "" else output_json.parent / f"pareto_{ds_key}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Pareto plot saved to: {fig_path}")

        # -----------------------------------------------------------
        # Comparison
        # -----------------------------------------------------------
        ratio_tof = lc_seq["cczCount"] / lc_sparse["cczCount"] if lc_sparse["cczCount"] > 0 else float("inf")
        ratio_qubits = lc_seq["numQubits"] / lc_sparse["numQubits"] if lc_sparse["numQubits"] > 0 else float("inf")

        print(f"\n   --- Comparison ---")
        print(f"   Toffoli ratio (seq/sparse): {ratio_tof:.3f}x")
        print(f"   Qubit ratio (seq/sparse):   {ratio_qubits:.3f}x")
        if not args.skip_qre and seq_result.get("physical_qubits") and sparse_result.get("physical_qubits"):
            ratio_phys = seq_result["physical_qubits"] / sparse_result["physical_qubits"]
            print(f"   Physical qubit ratio:       {ratio_phys:.3f}x")

        ds_result["comparison"] = {
            "toffoli_ratio_seq_over_sparse": round(ratio_tof, 4),
            "qubit_ratio_seq_over_sparse": round(ratio_qubits, 4),
        }

        all_results["datasets"][ds_key] = ds_result
        _save_results(all_results, output_json)

    # ===================================================================
    # Summary table
    # ===================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    rows = []
    for ds_key, ds_result in all_results["datasets"].items():
        if ds_result.get("sequential"):
            rows.append({
                "Dataset": ds_result["label"],
                "Method": "Sequential",
                "Toffoli": ds_result["sequential"]["logical_toffoli"],
                "Logical Qubits": ds_result["sequential"]["logical_qubits"],
                "Physical Qubits": ds_result["sequential"].get("physical_qubits", "N/A"),
                "Runtime (ns)": ds_result["sequential"].get("runtime_ns", "N/A"),
            })
        if ds_result.get("sparse"):
            rows.append({
                "Dataset": ds_result["label"],
                "Method": "Sparse",
                "Toffoli": ds_result["sparse"]["logical_toffoli"],
                "Logical Qubits": ds_result["sparse"]["logical_qubits"],
                "Physical Qubits": ds_result["sparse"].get("physical_qubits", "N/A"),
                "Runtime (ns)": ds_result["sparse"].get("runtime_ns", "N/A"),
            })

    if rows:
        df_summary = pd.DataFrame(rows)
        print(df_summary.to_string(index=False))

    print(f"\nResults saved to: {output_json}")
