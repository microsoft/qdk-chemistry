"""
Full Resource Estimation Comparison for Fe2S2.

Runs resource estimation in order:
  1. SOSSA QPE with HF initial state (QPE only)
  2. Sparse MPS State Preparation
  3. Sparse MPS + SOSSA QPE
  4. Dense MPS State Preparation
  5. Dense MPS + SOSSA QPE

Molecule:
  - Fe2S2-20: N=20, R=14, B=15, C=5

Results (including T factory info, Pareto front) are saved to JSON.

Usage:
  cd examples/mps_benchmark && python run_sparse_resource_estimation_fe2s2.py
  python run_sparse_resource_estimation_fe2s2.py --output my_results.json
"""

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdk.qre import PSSPC, LatticeSurgery, estimate
from qdk.qre.instruction_ids import LATTICE_SURGERY
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk.qre.property_keys import DISTANCE, NUM_TS_PER_ROTATION, PHYSICAL_COMPUTE_QUBITS
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import (
    QdkStandardQpeCircuitBuilder,
)
from qdk_chemistry.algorithms.state_preparation import MPSSequentialStatePreparation
from qdk_chemistry.algorithms.state_preparation.mps_sparse import (
    MPSSparseStatePreparation,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    FactorizedHamiltonianContainer,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.utils import Logger
from scipy.sparse import load_npz

Logger.set_global_level(Logger.LogLevel.off)

# ============================================================================
# Configuration
# ============================================================================

MOL_PARAMS = dict(N=20, R=14, B=15, C=5, b_rot=15, b_coeff=11, lambda_eff=6.4690)

COMPRESSED_PATH = Path("fe2s2-2_small") / "tensors_compressed" / "tensors_compressed_"
PHASE_BITS = MOL_PARAMS["b_rot"]  # Number of bits for phase rotations in MPS state prep
SIGMA_E = 1e-3

ARCHITECTURE = Majorana(error_rate=1e-5)
TRACE_QUERY = (
    PSSPC.q()
    * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
)
ISA_QUERY = ThreeAux.q() * RoundBasedFactory.q(use_cache=True, code_query=ThreeAux.q())
MAX_ERROR = 0.01

OUTPUT_JSON = Path(f"mps_sparse_resource_estimation_fe2s2_phase_bits_{PHASE_BITS}_full.json")


# ============================================================================
# Helpers
# ============================================================================


def load_mps_tensors():
    """Load Fe2S2 MPS tensors from disk."""
    n_tensors = len(glob.glob(str(COMPRESSED_PATH) + "[0-9]*.npz"))
    if n_tensors == 0:
        raise FileNotFoundError(
            f"No tensors found at {COMPRESSED_PATH}. "
            "Please extract fe2s2-2_small.tar.gz first:\n"
            "  tar -xzf fe2s2-2_small.tar.gz"
        )
    sparse_tensors = [load_npz(f"{COMPRESSED_PATH}{i}.npz") for i in range(n_tensors)]
    dense_tensors = [t.toarray() for t in sparse_tensors]
    return MPSWavefunction(dense_tensors)


def make_fake_factorized_hamiltonian(N, R, B, C, seed=42):
    """Create a fake FactorizedHamiltonianContainer with given (N, R, B, C)."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((N, N))
    h1 = (h1 + h1.T) / 2

    u_matrices = np.zeros(R * B * N)
    for ri in range(R):
        for bi in range(B):
            v = rng.standard_normal(N)
            v /= np.linalg.norm(v)
            u_matrices[ri * B * N + bi * N : ri * B * N + (bi + 1) * N] = v

    w_matrices = rng.standard_normal(R * B * C) * 0.1
    wb_matrix = rng.standard_normal((R, C)) * 0.1
    orbitals = ModelOrbitals(N)
    inactive_fock = np.zeros((N, N))

    return FactorizedHamiltonianContainer(
        R, B, C, 0.0, u_matrices, w_matrices, h1, wb_matrix, inactive_fock, orbitals,
    )


def _save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   [saved to {path}]")


def _print_full_qre_result(qre_result, label=""):
    """Print QRE table and return extra data (including Pareto front) for JSON."""
    qre_result.add_column(
        "compute_distance",
        lambda entry: entry.source[LATTICE_SURGERY].instruction[DISTANCE],
    )
    qre_result.add_column(
        "compute qubits",
        lambda entry: entry.properties[PHYSICAL_COMPUTE_QUBITS],
    )
    qre_result.add_column(
        "num_ts_per_rotation",
        lambda entry: entry.properties[NUM_TS_PER_ROTATION],
    )
    qre_result.add_factory_summary_column()
    df = qre_result.as_frame()
    print(f"\n   --- Full QRE Results{' (' + label + ')' if label else ''} ---")
    print(df.to_string(index=False))
    print()

    r = qre_result[-1]
    extra = {
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
    if "factories" in df.columns:
        extra["factory_summary"] = str(df["factories"].iloc[0])

    pareto_front = []
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
            elif hasattr(val, 'total_seconds'):
                record[col] = val.total_seconds()
            else:
                record[col] = str(val) if not isinstance(val, (int, float, bool, str, type(None))) else val
        pareto_front.append(record)
    extra["pareto_front"] = pareto_front

    return extra


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Resource Estimation Comparison for Fe2S2 (Sparse + Dense MPS).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=OUTPUT_JSON,
        help=f"Output JSON path. Default: {OUTPUT_JSON}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_json = args.output
    N = MOL_PARAMS["N"]
    lambda_eff = MOL_PARAMS["lambda_eff"]
    num_queries = math.ceil(math.pi * lambda_eff / (2 * SIGMA_E))
    num_phase_qubits = math.ceil(math.log2(num_queries))

    print("\nFull Resource Estimation Comparison — Fe2S2-20")
    print("=" * 70)
    print(f"   N={N}, lambda_eff={lambda_eff}, queries={num_queries}")
    print(f"   Phase bits (MPS): {PHASE_BITS}")
    print(f"   Output: {output_json}")
    print()

    all_results = {
        "parameters": {
            "molecule": "Fe2S2-20",
            "phase_bits_mps": PHASE_BITS,
            "sigma_E": SIGMA_E,
            "architecture": "Majorana(error_rate=1e-5)",
            "isa": "ThreeAux + RoundBasedFactory",
            "max_error": MAX_ERROR,
        },
        "Fe2S2-20": {
            "params": MOL_PARAMS,
            "qpe_hf": None,
            "sparse_mps_only": None,
            "sparse_mps_plus_qpe": None,
            "dense_mps_only": None,
            "dense_mps_plus_qpe": None,
        },
    }

    # ===================================================================
    # Part 1: SOSSA QPE with HF initial state
    # ===================================================================
    print(f"{'='*70}")
    print("  PART 1: SOSSA QPE with HF Initial State")
    print(f"{'='*70}")
    print(f"   Heisenberg queries: {num_queries}, phase qubits: {num_phase_qubits}")

    t0 = time.perf_counter()
    fh = make_fake_factorized_hamiltonian(N, MOL_PARAMS["R"], MOL_PARAMS["B"], MOL_PARAMS["C"])
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
            rotation_bit_precision=MOL_PARAMS["b_rot"],
            coefficient_bit_precision=MOL_PARAMS["b_coeff"],
        ),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    )

    qpe_circuits_hf = qpe_builder.run(
        state_preparation=hf_circuit, qubit_hamiltonian=fh
    )
    qpe_circuit_hf = qpe_circuits_hf[0]
    lc = qpe_circuit_hf.estimate().logical_counts

    print(f"   Toffoli (CCZ): {lc['cczCount']:,}")
    print(f"   Logical qubits: {lc['numQubits']:,}")

    print("   Running QRE v3...")
    app = qpe_circuit_hf.get_qre_application()
    qre_result = estimate(
        app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
        name="Fe2S2-20 QPE(HF)",
    )
    r = qre_result[-1]
    t_total = time.perf_counter() - t0
    print(f"   Physical qubits: {r.qubits:,}")
    print(f"   Runtime: {r.runtime}")
    print(f"   Total time: {t_total:.1f}s")
    extra = _print_full_qre_result(qre_result, "Fe2S2-20 QPE(HF)")

    all_results["Fe2S2-20"]["qpe_hf"] = {
        "label": "QPE(HF)",
        "logical_toffoli": lc["cczCount"],
        "logical_qubits": lc["numQubits"],
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        **extra,
    }
    _save_results(all_results, output_json)

    # ===================================================================
    # Load real Fe2S2 tensors
    # ===================================================================
    print(f"\n{'='*70}")
    print("  Loading Fe2S2 tensors...")
    print(f"{'='*70}")
    mps = load_mps_tensors()
    max_chi = max(mps.bond_dims)
    print(f"   Loaded: {mps.num_sites} sites, chi_max={max_chi}")
    print(f"   Bond dims: {mps.bond_dims}")
    print()

    # ===================================================================
    # Part 2: Sparse MPS State Preparation
    # ===================================================================
    print(f"{'='*70}")
    print("  PART 2: Sparse MPS State Preparation")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    algo_sparse = MPSSparseStatePreparation()
    algo_sparse.settings().set("rotation_bits", PHASE_BITS)
    sparse_circ = algo_sparse.run(mps)
    sparse_lc = sparse_circ.estimate().logical_counts
    t_build = time.perf_counter() - t0
    print(f"   Build time: {t_build:.1f}s")
    print(f"   Toffoli (CCZ): {sparse_lc['cczCount']:,}")
    print(f"   Logical qubits: {sparse_lc['numQubits']:,}")

    print("   Running QRE v3...")
    t0 = time.perf_counter()
    app = sparse_circ.get_qre_application()
    qre_result = estimate(
        app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
        name="Fe2S2-20 Sparse MPS",
    )
    r = qre_result[-1]
    t_qre = time.perf_counter() - t0
    print(f"   Physical qubits: {r.qubits:,}")
    print(f"   Runtime: {r.runtime}")
    print(f"   QRE time: {t_qre:.1f}s")
    extra = _print_full_qre_result(qre_result, "Fe2S2-20 Sparse MPS")

    all_results["Fe2S2-20"]["sparse_mps_only"] = {
        "label": f"Sparse MPS chi={max_chi}",
        "bond_dim": max_chi,
        "logical_toffoli": sparse_lc["cczCount"],
        "logical_qubits": sparse_lc["numQubits"],
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        "build_time_s": round(t_build, 2),
        **extra,
    }
    _save_results(all_results, output_json)

    # ===================================================================
    # Part 3: Sparse MPS + SOSSA QPE
    # ===================================================================
    print(f"\n{'='*70}")
    print("  PART 3: Sparse MPS + SOSSA QPE")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    qpe_circuits_sparse = qpe_builder.run(
        state_preparation=sparse_circ, qubit_hamiltonian=fh
    )
    qpe_circuit_sparse = qpe_circuits_sparse[0]
    lc = qpe_circuit_sparse.estimate().logical_counts
    t_build_qpe = time.perf_counter() - t0

    sossa_only_tof = lc["cczCount"] - sparse_lc["cczCount"]
    print(f"   MPS Toffoli: {sparse_lc['cczCount']:,}")
    print(f"   SOSSA Toffoli: {sossa_only_tof:,}")
    print(f"   Total Toffoli: {lc['cczCount']:,}")
    print(f"   Logical qubits: {lc['numQubits']:,}")

    print("   Running QRE v3...")
    t0 = time.perf_counter()
    app = qpe_circuit_sparse.get_qre_application()
    qre_result = estimate(
        app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
        name="Fe2S2-20 SparseMPS+QPE",
    )
    r = qre_result[-1]
    t_qre = time.perf_counter() - t0
    print(f"   Physical qubits: {r.qubits:,}")
    print(f"   Runtime: {r.runtime}")
    print(f"   QRE time: {t_qre:.1f}s")
    extra = _print_full_qre_result(qre_result, "Fe2S2-20 SparseMPS+QPE")

    all_results["Fe2S2-20"]["sparse_mps_plus_qpe"] = {
        "label": f"Sparse MPS+QPE chi={max_chi}",
        "bond_dim": max_chi,
        "mps_toffoli": sparse_lc["cczCount"],
        "sossa_toffoli": sossa_only_tof,
        "total_toffoli": lc["cczCount"],
        "logical_qubits": lc["numQubits"],
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        **extra,
    }
    _save_results(all_results, output_json)

    # ===================================================================
    # Part 4: Dense MPS State Preparation
    # ===================================================================
    print(f"\n{'='*70}")
    print("  PART 4: Dense MPS State Preparation")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    algo_dense = MPSSequentialStatePreparation()
    algo_dense.settings().set("rotation_bits", PHASE_BITS)
    # algo_dense.settings().set("fast_grouped_resource_estimation", True)
    dense_circ = algo_dense.run(mps)
    dense_lc = dense_circ.estimate().logical_counts
    t_build = time.perf_counter() - t0
    print(f"   Build time: {t_build:.1f}s")
    print(f"   Toffoli (CCZ): {dense_lc['cczCount']:,}")
    print(f"   Logical qubits: {dense_lc['numQubits']:,}")

    print("   Running QRE v3...")
    t0 = time.perf_counter()
    app = dense_circ.get_qre_application()
    qre_result = estimate(
        app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
        name="Fe2S2-20 Dense MPS",
    )
    r = qre_result[-1]
    t_qre = time.perf_counter() - t0
    print(f"   Physical qubits: {r.qubits:,}")
    print(f"   Runtime: {r.runtime}")
    print(f"   QRE time: {t_qre:.1f}s")
    extra = _print_full_qre_result(qre_result, "Fe2S2-20 Dense MPS")

    all_results["Fe2S2-20"]["dense_mps_only"] = {
        "label": f"Dense MPS chi={max_chi}",
        "bond_dim": max_chi,
        "logical_toffoli": dense_lc["cczCount"],
        "logical_qubits": dense_lc["numQubits"],
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        "build_time_s": round(t_build, 2),
        **extra,
    }
    _save_results(all_results, output_json)

    # ===================================================================
    # Part 5: Dense MPS + SOSSA QPE
    # ===================================================================
    print(f"\n{'='*70}")
    print("  PART 5: Dense MPS + SOSSA QPE")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    qpe_circuits_dense = qpe_builder.run(
        state_preparation=dense_circ, qubit_hamiltonian=fh
    )
    qpe_circuit_dense = qpe_circuits_dense[0]
    lc = qpe_circuit_dense.estimate().logical_counts
    t_build_qpe = time.perf_counter() - t0

    sossa_only_tof = lc["cczCount"] - dense_lc["cczCount"]
    print(f"   MPS Toffoli: {dense_lc['cczCount']:,}")
    print(f"   SOSSA Toffoli: {sossa_only_tof:,}")
    print(f"   Total Toffoli: {lc['cczCount']:,}")
    print(f"   Logical qubits: {lc['numQubits']:,}")

    print("   Running QRE v3...")
    t0 = time.perf_counter()
    app = qpe_circuit_dense.get_qre_application()
    qre_result = estimate(
        app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
        name="Fe2S2-20 DenseMPS+QPE",
    )
    r = qre_result[-1]
    t_qre = time.perf_counter() - t0
    print(f"   Physical qubits: {r.qubits:,}")
    print(f"   Runtime: {r.runtime}")
    print(f"   QRE time: {t_qre:.1f}s")
    extra = _print_full_qre_result(qre_result, "Fe2S2-20 DenseMPS+QPE")

    all_results["Fe2S2-20"]["dense_mps_plus_qpe"] = {
        "label": f"Dense MPS+QPE chi={max_chi}",
        "bond_dim": max_chi,
        "mps_toffoli": dense_lc["cczCount"],
        "sossa_toffoli": sossa_only_tof,
        "total_toffoli": lc["cczCount"],
        "logical_qubits": lc["numQubits"],
        "physical_qubits": r.qubits,
        "runtime_ns": r.runtime,
        **extra,
    }
    _save_results(all_results, output_json)

    # ===================================================================
    # Summary
    # ===================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    res = all_results["Fe2S2-20"]
    print(f"   QPE(HF):           Toffoli={res['qpe_hf']['logical_toffoli']:,}")
    print(f"   Sparse MPS only:   Toffoli={res['sparse_mps_only']['logical_toffoli']:,}")
    print(f"   Sparse MPS+QPE:    Toffoli={res['sparse_mps_plus_qpe']['total_toffoli']:,}")
    print(f"   Dense MPS only:    Toffoli={res['dense_mps_only']['logical_toffoli']:,}")
    print(f"   Dense MPS+QPE:     Toffoli={res['dense_mps_plus_qpe']['total_toffoli']:,}")
    print(f"\nDone. Results: {output_json}")
