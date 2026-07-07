"""
Sparse vs Dense MPS Resource Estimation Comparison.

Unified script for comparing sparse (U(1)-block-sparsity) vs dense (sequential)
MPS state preparation, with optional SOSSA QPE wrapping.

Supported molecules:
  - fe2s2: Fe2S2-20 (N=20, R=14, B=15, C=5) — full comparison incl. QPE
  - g1:    P450 G-1 (43 sites) — MPS-only comparison (no QPE, R/B/C unavailable)

Note:
  Recommended to run on HPC due to high memory requirements at large bond
  dimensions.

Usage:
  python run_sparse_vs_dense.py --molecule fe2s2
  python run_sparse_vs_dense.py --molecule g1
  python run_sparse_vs_dense.py --molecule g1 --output my_results.json
"""

import argparse
import math
import time
from pathlib import Path

from qdk.qre import PSSPC, LatticeSurgery
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk_chemistry.algorithms.state_preparation import MPSSequentialStatePreparation
from qdk_chemistry.algorithms.state_preparation.mps_sparse import MPSSparseStatePreparation
from qdk_chemistry.utils import Logger

from utils import load_mps_tensors, make_fake_factorized_hamiltonian, run_qre, save_results

Logger.set_global_level(Logger.LogLevel.info)

# ============================================================================
# QRE Configuration
# ============================================================================

ARCHITECTURE = Majorana(error_rate=1e-5)
TRACE_QUERY = (
    PSSPC.q()
    * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
)
ISA_QUERY = ThreeAux.q() * RoundBasedFactory.q(use_cache=True, code_query=ThreeAux.q())
MAX_ERROR = 0.01
SIGMA_E = 1e-3


# ============================================================================
# Molecule definitions
# ============================================================================

MOLECULES = {
    "fe2s2": {
        "label": "Fe2S2-20",
        "tensor_path": Path("fe2s2-2_small") / "tensors_compressed" / "tensors_compressed_",
        "N": 20,
        "R": 14,
        "B": 15,
        "C": 5,
        "b_rot": 15,
        "b_coeff": 11,
        "lambda_eff": 6.4690,
        "has_qpe": True,
    },
    "g1": {
        "label": "P450-G1-43",
        "tensor_path": Path("G-1") / "tensors_compressed" / "tensors_compressed_",
        "N": 43,
        "R": None,
        "B": None,
        "C": None,
        "b_rot": 15,
        "b_coeff": None,
        "lambda_eff": None,
        "has_qpe": False,
    },
}


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse vs Dense MPS Resource Estimation Comparison.",
    )
    parser.add_argument(
        "--molecule", "-m",
        required=True,
        choices=list(MOLECULES.keys()),
        help="Which molecule to run.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path. Default: auto-generated from molecule name.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mol_cfg = MOLECULES[args.molecule]
    mol_label = mol_cfg["label"]

    phase_bits = mol_cfg.get("b_rot")
    output_json = args.output or Path(f"mps_sparse_vs_dense_{args.molecule}_pb_{phase_bits}.json")

    Logger.info(f"\nSparse vs Dense MPS Comparison — {mol_label}")
    Logger.info("=" * 70)
    Logger.info(f"   Phase bits: {phase_bits}")
    Logger.info(f"   QPE: {'enabled' if mol_cfg['has_qpe'] else 'disabled (R/B/C not available)'}")
    Logger.info(f"   Output: {output_json}")
    Logger.info("")

    # Load tensors
    tensor_path = mol_cfg["tensor_path"]
    # Check that the tensor data directory exists
    tensor_dir = tensor_path.parent
    if not tensor_dir.exists():
        Logger.info(f"ERROR: Tensor directory not found: {tensor_dir}")
        Logger.info("")
        Logger.info("  The MPS tensor data must be extracted before running this script.")
        Logger.info("  If the data is provided as a .tar.gz archive, extract it with:")
        Logger.info(f"    tar -xzf <archive>.tar.gz -C {tensor_dir.parent}")
        Logger.info("")
        raise SystemExit(1)

    Logger.info("  Loading MPS tensors...")
    mps = load_mps_tensors(tensor_path)
    max_chi = max(mps.bond_dims)
    Logger.info(f"   Sites: {mps.num_sites}, qubits: {mps.num_qubits}, chi_max: {max_chi}")
    Logger.info(f"   Bond dims: {mps.bond_dims}")
    Logger.info("")

    all_results = {
        "parameters": {
            "molecule": mol_label,
            "phase_bits": phase_bits,
            "sigma_E": SIGMA_E,
            "architecture": "Majorana(error_rate=1e-5)",
            "isa": "ThreeAux + RoundBasedFactory",
            "max_error": MAX_ERROR,
            "max_bond_dim": max_chi,
        },
        mol_label: {
            "params": mol_cfg,
            "qpe_hf": None,
            "sparse_mps_only": None,
            "sparse_mps_plus_qpe": None,
            "dense_mps_only": None,
            "dense_mps_plus_qpe": None,
        },
    }

    # ===================================================================
    # Part 1: QPE(HF) — only if has_qpe
    # ===================================================================
    if mol_cfg["has_qpe"]:
        from qdk_chemistry.algorithms import create
        from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import (
            QdkStandardQpeCircuitBuilder,
        )
        from qdk_chemistry.data import (
            AlgorithmRef,
            Configuration,
            StateVectorContainer,
            Wavefunction,
        )

        N = mol_cfg["N"]
        lambda_eff = mol_cfg["lambda_eff"]
        num_queries = math.ceil(math.pi * lambda_eff / (2 * SIGMA_E))
        num_phase_qubits = math.ceil(math.log2(num_queries))

        Logger.info(f"{'='*70}")
        Logger.info("  PART 1: SOSSA QPE with HF Initial State")
        Logger.info(f"{'='*70}")
        Logger.info(f"   Heisenberg queries: {num_queries}, phase qubits: {num_phase_qubits}")

        t0 = time.perf_counter()
        fh = make_fake_factorized_hamiltonian(N, mol_cfg["R"], mol_cfg["B"], mol_cfg["C"])
        orbitals = fh.get_orbitals()

        state_prep = create("state_prep", "sparse_isometry_gf2x")
        hf_config = Configuration.canonical_hf_configuration(N, N, 2 * N)
        hf_wavefunction = Wavefunction(
            StateVectorContainer(hf_config, orbitals, "electrons")
        )
        hf_circuit = state_prep.run(hf_wavefunction)

        qpe_builder = QdkStandardQpeCircuitBuilder(
            num_bits=num_phase_qubits,
            controlled_circuit_mapper=AlgorithmRef(
                "controlled_circuit_mapper",
                "sossa",
                outer_prepare=AlgorithmRef("state_prep", "alias_sampling"),
                inner_prepare_algorithm="controlled_alias_sampling",
                select_algorithm="qrom_phase_gradient",
                rotation_bit_precision=mol_cfg["b_rot"],
                coefficient_bit_precision=mol_cfg["b_coeff"],
            ),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        )

        qpe_circuits_hf = qpe_builder.run(
            state_preparation=hf_circuit, qubit_hamiltonian=fh
        )
        qpe_circuit_hf = qpe_circuits_hf[0]
        lc = qpe_circuit_hf.estimate().logical_counts
        Logger.info(f"   Toffoli (CCZ): {lc['cczCount']:,}")
        Logger.info(f"   Logical qubits: {lc['numQubits']:,}")

        extra = run_qre(qpe_circuit_hf, f"{mol_label} QPE(HF)", ARCHITECTURE, ISA_QUERY, TRACE_QUERY, MAX_ERROR)
        all_results[mol_label]["qpe_hf"] = {
            "label": "QPE(HF)",
            "logical_toffoli": lc["cczCount"],
            "logical_qubits": lc["numQubits"],
            **extra,
        }
        save_results(all_results, output_json)
    else:
        Logger.info("  Skipping QPE(HF) — R/B/C not available for this molecule.")
        qpe_builder = None
        fh = None

    # ===================================================================
    # Part 2: Sparse MPS State Preparation
    # ===================================================================
    Logger.info(f"\n{'='*70}")
    Logger.info("  PART 2: Sparse MPS State Preparation")
    Logger.info(f"{'='*70}")

    t0 = time.perf_counter()
    algo_sparse = MPSSparseStatePreparation()
    algo_sparse.settings().set("rotation_bits", phase_bits)
    sparse_circ = algo_sparse.run(mps)
    sparse_lc = sparse_circ.estimate().logical_counts
    t_build = time.perf_counter() - t0
    Logger.info(f"   Build time: {t_build:.1f}s")
    Logger.info(f"   Toffoli (CCZ): {sparse_lc['cczCount']:,}")
    Logger.info(f"   Logical qubits: {sparse_lc['numQubits']:,}")

    extra = run_qre(sparse_circ, f"{mol_label} Sparse MPS", ARCHITECTURE, ISA_QUERY, TRACE_QUERY, MAX_ERROR)
    all_results[mol_label]["sparse_mps_only"] = {
        "label": f"Sparse MPS chi={max_chi}",
        "bond_dim": max_chi,
        "logical_toffoli": sparse_lc["cczCount"],
        "logical_qubits": sparse_lc["numQubits"],
        "build_time_s": round(t_build, 2),
        **extra,
    }
    save_results(all_results, output_json)

    # ===================================================================
    # Part 3: Sparse MPS + QPE (only if has_qpe)
    # ===================================================================
    if mol_cfg["has_qpe"]:
        Logger.info(f"\n{'='*70}")
        Logger.info("  PART 3: Sparse MPS + SOSSA QPE")
        Logger.info(f"{'='*70}")

        t0 = time.perf_counter()
        qpe_circuits_sparse = qpe_builder.run(
            state_preparation=sparse_circ, qubit_hamiltonian=fh
        )
        qpe_circuit_sparse = qpe_circuits_sparse[0]
        lc = qpe_circuit_sparse.estimate().logical_counts
        t_build_qpe = time.perf_counter() - t0

        sossa_only_tof = lc["cczCount"] - sparse_lc["cczCount"]
        Logger.info(f"   MPS Toffoli: {sparse_lc['cczCount']:,}")
        Logger.info(f"   SOSSA Toffoli: {sossa_only_tof:,}")
        Logger.info(f"   Total Toffoli: {lc['cczCount']:,}")

        extra = run_qre(qpe_circuit_sparse, f"{mol_label} SparseMPS+QPE", ARCHITECTURE, ISA_QUERY, TRACE_QUERY, MAX_ERROR)
        all_results[mol_label]["sparse_mps_plus_qpe"] = {
            "label": f"Sparse MPS+QPE chi={max_chi}",
            "bond_dim": max_chi,
            "mps_toffoli": sparse_lc["cczCount"],
            "sossa_toffoli": sossa_only_tof,
            "total_toffoli": lc["cczCount"],
            "logical_qubits": lc["numQubits"],
            **extra,
        }
        save_results(all_results, output_json)

    # ===================================================================
    # Part 4: Dense MPS State Preparation
    # ===================================================================
    Logger.info(f"\n{'='*70}")
    Logger.info("  PART 4: Dense MPS State Preparation")
    Logger.info(f"{'='*70}")

    t0 = time.perf_counter()
    algo_dense = MPSSequentialStatePreparation()
    algo_dense.settings().set("rotation_bits", phase_bits)
    algo_dense.settings().set("fast_grouped_resource_estimation", True)
    dense_circ = algo_dense.run(mps)
    dense_lc = dense_circ.estimate().logical_counts
    t_build = time.perf_counter() - t0
    Logger.info(f"   Build time: {t_build:.1f}s")
    Logger.info(f"   Toffoli (CCZ): {dense_lc['cczCount']:,}")
    Logger.info(f"   Logical qubits: {dense_lc['numQubits']:,}")

    extra = run_qre(dense_circ, f"{mol_label} Dense MPS", ARCHITECTURE, ISA_QUERY, TRACE_QUERY, MAX_ERROR)
    all_results[mol_label]["dense_mps_only"] = {
        "label": f"Dense MPS chi={max_chi}",
        "bond_dim": max_chi,
        "logical_toffoli": dense_lc["cczCount"],
        "logical_qubits": dense_lc["numQubits"],
        "build_time_s": round(t_build, 2),
        **extra,
    }
    save_results(all_results, output_json)

    # ===================================================================
    # Part 5: Dense MPS + QPE (only if has_qpe)
    # ===================================================================
    if mol_cfg["has_qpe"]:
        Logger.info(f"\n{'='*70}")
        Logger.info("  PART 5: Dense MPS + SOSSA QPE")
        Logger.info(f"{'='*70}")

        t0 = time.perf_counter()
        qpe_circuits_dense = qpe_builder.run(
            state_preparation=dense_circ, qubit_hamiltonian=fh
        )
        qpe_circuit_dense = qpe_circuits_dense[0]
        lc = qpe_circuit_dense.estimate().logical_counts
        t_build_qpe = time.perf_counter() - t0

        sossa_only_tof = lc["cczCount"] - dense_lc["cczCount"]
        Logger.info(f"   MPS Toffoli: {dense_lc['cczCount']:,}")
        Logger.info(f"   SOSSA Toffoli: {sossa_only_tof:,}")
        Logger.info(f"   Total Toffoli: {lc['cczCount']:,}")

        extra = run_qre(qpe_circuit_dense, f"{mol_label} DenseMPS+QPE", ARCHITECTURE, ISA_QUERY, TRACE_QUERY, MAX_ERROR)
        all_results[mol_label]["dense_mps_plus_qpe"] = {
            "label": f"Dense MPS+QPE chi={max_chi}",
            "bond_dim": max_chi,
            "mps_toffoli": dense_lc["cczCount"],
            "sossa_toffoli": sossa_only_tof,
            "total_toffoli": lc["cczCount"],
            "logical_qubits": lc["numQubits"],
            **extra,
        }
        save_results(all_results, output_json)
