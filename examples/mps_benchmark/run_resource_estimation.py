"""
Multi-Molecule Resource Estimation: Fe2S2, Fe4S4, FeMoCo.

Runs resource estimation scenarios for multiple iron-sulfur cluster molecules:
  1. MPS Dense State Preparation at various bond dimensions
  2. SOSSA QPE with Hartree-Fock initial state
  3. MPS + SOSSA QPE (MPS as initial state for QPE)

Molecules:
  - Fe2S2-20:  N=20, R=14, B=15, C=5
  - Fe4S4-36:  N=36, R=9,  B=18, C=18
  - FeMoCo-54: N=54, R=10, B=27, C=27

Results (including T factory info, compute distance, compute qubits,
num_ts_per_rotation) are saved incrementally to JSON after each estimation.

Prerequisites:
  - pip install 'qdk-chemistry[qre]'
  - For real tensor data: extract fe2s2-2_small.tar.gz in examples/

Usage:
  cd examples/mps_benchmark && python run_resource_estimation.py
  python run_resource_estimation.py --molecules Fe2S2-20 Fe4S4-36
  python run_resource_estimation.py --molecules FeMoCo-76 --bond-dims 100 1000
  python run_resource_estimation.py --molecules all --bond-dims 100 1000 5000 10000
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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

# Suppress verbose logging
Logger.set_global_level(Logger.LogLevel.off)

# ============================================================================
# Configuration
# ============================================================================

# Molecule parameters (from Low2025 / SOSSA notebook)
MOLECULES = [
    ("Fe2S2-20", dict(N=20, R=14, B=15, C=5, b_rot=15, b_coeff=11, lambda_eff=6.4690)),
    ("Fe4S4-36", dict(N=36, R=9, B=18, C=18, b_rot=17, b_coeff=13, lambda_eff=14.9842)),
    ("FeMoCo-76", dict(N=76, R=15, B=57, C=19, b_rot=15, b_coeff=9,  lambda_eff=43.6538)),
#    ("P450-G1-43", dict(N=43, R=None, B=None, C=None, b_rot=10, b_coeff=None, lambda_eff=None)),
]

# MPS tensor path for Fe2S2 real tensors (extract fe2s2-2_small.tar.gz first)
COMPRESSED_PATH = Path("fe2s2-2_small") / "tensors_compressed" / "tensors_compressed_"
# PHASE_BITS = 10  # rotation precision for MPS state prep

# Target energy precision
SIGMA_E = 1e-3  # 1 mHa

# QRE v3 architecture
ARCHITECTURE = Majorana(error_rate=1e-5)
TRACE_QUERY = (
    PSSPC.q()
    * LatticeSurgery.q(slow_down_factor=[1.0 * j for j in range(1, 20)])
)
ISA_QUERY = ThreeAux.q() * RoundBasedFactory.q(use_cache=True, code_query=ThreeAux.q())
MAX_ERROR = 0.01

# MPS settings
SITE_DIM = 4
DEFAULT_BOND_DIMS = [100, 1000, 5000, 10000]
OUTPUT_JSON = Path("mps_sossa_resource_estimation_full.json")

# Valid molecule names for CLI
MOLECULE_NAMES = [m[0] for m in MOLECULES]
MOLECULE_MAP = {m[0]: m[1] for m in MOLECULES}


# ============================================================================
# Helpers
# ============================================================================


def make_fake_factorized_hamiltonian(N, R, B, C, seed=42):
    """Create a fake FactorizedHamiltonianContainer with given (N, R, B, C).

    The actual tensor values are random but the dimensions match real data,
    which is sufficient for resource estimation (gate counts depend on dimensions).
    """
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
        R,
        B,
        C,
        0.0,
        u_matrices,
        w_matrices,
        h1,
        wb_matrix,
        inactive_fock,
        orbitals,
    )


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


def _save_results(results, path=OUTPUT_JSON):
    """Save results dict to JSON, called after each estimation completes."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   [saved to {path}]")


def _print_full_qre_result(qre_result, label=""):
    """Add extra columns to QRE result table, print it, and return extra data for JSON."""
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

    # Extract extra data for JSON storage
    r = qre_result[-1]
    extra = {
        "compute_distance": int(r.source[LATTICE_SURGERY].instruction[DISTANCE]),
        "compute_qubits": int(r.properties[PHYSICAL_COMPUTE_QUBITS]),
        "num_ts_per_rotation": int(r.properties[NUM_TS_PER_ROTATION]),
        "factories": [],
    }
    for fid, factory_result in r.factories.items():
        extra["factories"].append(
            {
                "copies": factory_result.copies,
                "runs": factory_result.runs,
                "error_rate": factory_result.error_rate,
                "states": factory_result.states,
            }
        )
    # Also store the human-readable summary
    if "factories" in df.columns:
        extra["factory_summary"] = str(df["factories"].iloc[0])

    # Store full Pareto front (all QRE results, not just the first)
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
# Main — Multi-molecule comparison: MPS, QPE(HF), MPS+QPE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MPS + SOSSA QPE Resource Estimation for iron-sulfur clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python run_resource_estimation.py --molecules Fe2S2-20 Fe4S4-36\n"
               "  python run_resource_estimation.py --molecules all --bond-dims 100 1000\n"
               "  python run_resource_estimation.py --molecules FeMoCo-76 --bond-dims 100 5000 10000\n",
    )
    parser.add_argument(
        "--molecules", "-m",
        nargs="+",
        default=["all"],
        choices=MOLECULE_NAMES + ["all"],
        help=f"Which molecules to run. Choices: {MOLECULE_NAMES} or 'all'. Default: all",
    )
    parser.add_argument(
        "--bond-dims", "-b",
        nargs="+",
        type=int,
        default=DEFAULT_BOND_DIMS,
        help=f"Bond dimensions to evaluate. Default: {DEFAULT_BOND_DIMS}",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_JSON,
        help=f"Output JSON path. Default: {OUTPUT_JSON}",
    )
    parser.add_argument(
        "--skip-qpe",
        action="store_true",
        help="Skip SOSSA QPE parts (2 & 3), only run MPS state prep sweep.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve which molecules to run
    if "all" in args.molecules:
        selected_molecules = MOLECULES
    else:
        selected_molecules = [(name, MOLECULE_MAP[name]) for name in args.molecules]

    bond_dims = sorted(args.bond_dims)
    output_json = args.output

    print("\nMPS + SOSSA QPE Resource Estimation")
    print("=" * 70)
    print(f"   Molecules: {[m[0] for m in selected_molecules]}")
    print(f"   Bond dimensions: {bond_dims}")
    print(f"   Site dim: {SITE_DIM}, Phase bits: {[m[1]['b_rot'] for m in selected_molecules]}")
    print(f"   sigma_E = {SIGMA_E} Ha, max_error = {MAX_ERROR}")
    print(f"   Output: {output_json}")
    print()

    all_results = {
        "parameters": {
            "molecules": [m[0] for m in selected_molecules],
            "site_dim": SITE_DIM,
            "phase_bits_mps": [m[1]['b_rot'] for m in selected_molecules],
            "bond_dims": bond_dims,
            "sigma_E": SIGMA_E,
            "architecture": "Majorana(error_rate=1e-5)",
            "isa": "ThreeAux + RoundBasedFactory",
            "max_error": MAX_ERROR,
        },
    }

    # Initialize result containers per molecule
    for mol_name, params in selected_molecules:
        all_results[mol_name] = {
            "params": params,
            "mps_only": [],
            "qpe_hf": None,
            "mps_plus_qpe": [],
        }

    # ===================================================================
    # Load real tensors
    # ===================================================================
    mps_real = None
    try:
        mps_real = load_mps_tensors()
        print(f"   Loaded real Fe2S2 tensors: chi_max = {max(mps_real.bond_dims)}")
    except FileNotFoundError as e:
        print(f"   WARNING: {e}")
        print("   Skipping real tensor runs for Fe2S2.")
    print()

    # ===================================================================
    # Loop over each molecule
    # ===================================================================
    for mol_name, params in selected_molecules:
        N = params["N"]
        num_sites = N  # each spatial orbital is one site with dim=4

        if not args.skip_qpe:
            lambda_eff = params["lambda_eff"]
            num_queries = math.ceil(math.pi * lambda_eff / (2 * SIGMA_E))
            num_phase_qubits = math.ceil(math.log2(num_queries))

        print("\n" + "#" * 70)
        print(f"# MOLECULE: {mol_name}")
        print(f"#   N={N}, R={params['R']}, B={params['B']}, C={params['C']}")
        if not args.skip_qpe:
            print(f"#   lambda_eff={lambda_eff}, queries={num_queries}, phase_qubits={num_phase_qubits}")
        print("#" * 70)

        # ---------------------------------------------------------------
        # Part 1: MPS State Preparation (dense) at various bond dims
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"  PART 1: MPS Dense State Preparation -- {mol_name}")
        print(f"{'='*70}")

        # Real tensors (only for Fe2S2)
        real_mps_for_mol = None
        if mol_name == "Fe2S2-20" and mps_real is not None:
            real_mps_for_mol = mps_real

        if real_mps_for_mol is not None:
            print(f"\n--- Real {mol_name} (DMRG, chi_max ~ {max(real_mps_for_mol.bond_dims)}) ---")
            t0 = time.perf_counter()
            max_chi = max(real_mps_for_mol.bond_dims)
            print(f"   Max bond dim: {max_chi}, Bond dims: {real_mps_for_mol.bond_dims}")

            algo = MPSSequentialStatePreparation()
            algo.settings().set("rotation_bits", params["b_rot"])
            algo.settings().set("fast_grouped_resource_estimation", True)
            circuit = algo.run(real_mps_for_mol)
            lc = circuit.estimate().logical_counts

            app = circuit.get_qre_application()
            qre_result = estimate(
                app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
                name=f"{mol_name} MPS real",
            )
            r = qre_result[-1]
            t_total = time.perf_counter() - t0
            print(
                f"   Toffoli={lc['cczCount']:,}, qubits={lc['numQubits']:,}, "
                f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_total:.1f}s"
            )
            extra = _print_full_qre_result(qre_result, f"{mol_name} MPS real")

            all_results[mol_name]["mps_only"].append(
                {
                    "label": "real(DMRG)",
                    "bond_dim": max_chi,
                    "logical_toffoli": lc["cczCount"],
                    "logical_qubits": lc["numQubits"],
                    "physical_qubits": r.qubits,
                    "runtime_ns": r.runtime,
                    **extra,
                }
            )
            _save_results(all_results, output_json)

        if args.skip_qpe:
            # MPS only sweep — no QPE
            # Fake MPS at various bond dims
            for chi in bond_dims:
                print(f"\n--- {mol_name} Fake MPS chi = {chi} ---")
                t0 = time.perf_counter()
                mps = MPSWavefunction.random(
                    num_sites=num_sites, bond_dim=chi, site_dim=SITE_DIM
                )
                max_chi_fake = max(mps.bond_dims)
                print(f"   Max bond dim: {max_chi_fake}, Sites: {num_sites}")

                algo = MPSSequentialStatePreparation()
                algo.settings().set("rotation_bits", params["b_rot"])
                algo.settings().set("fast_grouped_resource_estimation", True)
                circuit = algo.run(mps)
                lc = circuit.estimate().logical_counts

                app = circuit.get_qre_application()
                qre_result = estimate(
                    app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
                    name=f"{mol_name} MPS chi={chi}",
                )
                r = qre_result[-1]
                t_total = time.perf_counter() - t0
                print(
                    f"   Toffoli={lc['cczCount']:,}, qubits={lc['numQubits']:,}, "
                    f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_total:.1f}s"
                )
                extra = _print_full_qre_result(qre_result, f"{mol_name} MPS chi={chi}")

                all_results[mol_name]["mps_only"].append(
                    {
                        "label": str(chi),
                        "bond_dim": max_chi_fake,
                        "logical_toffoli": lc["cczCount"],
                        "logical_qubits": lc["numQubits"],
                        "physical_qubits": r.qubits,
                        "runtime_ns": r.runtime,
                        **extra,
                    }
                )
                _save_results(all_results, output_json)
            continue

        # ---------------------------------------------------------------
        # Full workflow: QPE(HF), then MPS-only + MPS+QPE (reusing circuits)
        # ---------------------------------------------------------------

        # Part 2: SOSSA QPE with HF initial state
        print(f"\n{'='*70}")
        print(f"  PART 2: SOSSA QPE with HF Initial State -- {mol_name}")
        print(f"{'='*70}")
        print(f"   N={N}, lambda_eff={lambda_eff}, sigma_E={SIGMA_E}")
        print(f"   Heisenberg queries: {num_queries}, phase qubits: {num_phase_qubits}")

        t0 = time.perf_counter()
        fh = make_fake_factorized_hamiltonian(N, params["R"], params["B"], params["C"])
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
                rotation_bit_precision=params["b_rot"],
                coefficient_bit_precision=params["b_coeff"],
            ),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        )
        qpe_circuits_hf = qpe_builder.run(
            state_preparation=hf_circuit, qubit_hamiltonian=fh
        )
        qpe_circuit_hf = qpe_circuits_hf[0]
        lc = qpe_circuit_hf.estimate().logical_counts

        app = qpe_circuit_hf.get_qre_application()
        qre_result = estimate(
            app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
            name=f"{mol_name} SOSSA QPE (HF)",
        )
        r = qre_result[-1]
        t_total = time.perf_counter() - t0
        print(
            f"   Toffoli={lc['cczCount']:,}, qubits={lc['numQubits']:,}, "
            f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_total:.1f}s"
        )
        extra = _print_full_qre_result(qre_result, f"{mol_name} SOSSA QPE (HF)")

        all_results[mol_name]["qpe_hf"] = {
            "label": "QPE(HF)",
            "logical_toffoli": lc["cczCount"],
            "logical_qubits": lc["numQubits"],
            "physical_qubits": r.qubits,
            "runtime_ns": r.runtime,
            **extra,
        }
        _save_results(all_results, output_json)

        # ---------------------------------------------------------------
        # Parts 1 & 3: MPS-only + MPS+QPE (build MPS circuit once, reuse)
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"  PARTS 1 & 3: MPS State Prep + MPS+QPE -- {mol_name}")
        print(f"{'='*70}")

        # Fake MPS at various bond dims — build once, run MPS-only QRE and MPS+QPE QRE
        for chi in bond_dims:
            print(f"\n--- {mol_name} MPS chi={chi} (MPS-only + MPS+QPE) ---")
            t0 = time.perf_counter()

            mps_wfn = MPSWavefunction.random(
                num_sites=num_sites, bond_dim=chi, site_dim=SITE_DIM
            )
            max_chi_fake = max(mps_wfn.bond_dims)
            print(f"   Max bond dim: {max_chi_fake}, Sites: {num_sites}")

            algo_mps = MPSSequentialStatePreparation()
            algo_mps.settings().set("rotation_bits", params["b_rot"])
            algo_mps.settings().set("fast_grouped_resource_estimation", True)
            mps_circ = algo_mps.run(mps_wfn)
            mps_lc = mps_circ.estimate().logical_counts

            # --- MPS-only QRE ---
            app = mps_circ.get_qre_application()
            qre_result = estimate(
                app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
                name=f"{mol_name} MPS chi={chi}",
            )
            r = qre_result[-1]
            t_mps = time.perf_counter() - t0
            print(
                f"   [MPS-only] Toffoli={mps_lc['cczCount']:,}, qubits={mps_lc['numQubits']:,}, "
                f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_mps:.1f}s"
            )
            extra = _print_full_qre_result(qre_result, f"{mol_name} MPS chi={chi}")

            all_results[mol_name]["mps_only"].append(
                {
                    "label": str(chi),
                    "bond_dim": max_chi_fake,
                    "logical_toffoli": mps_lc["cczCount"],
                    "logical_qubits": mps_lc["numQubits"],
                    "physical_qubits": r.qubits,
                    "runtime_ns": r.runtime,
                    **extra,
                }
            )
            _save_results(all_results, output_json)

            # --- MPS+QPE (reuse mps_circ) ---
            t0_qpe = time.perf_counter()
            qpe_circuits_mps = qpe_builder.run(
                state_preparation=mps_circ, qubit_hamiltonian=fh
            )
            qpe_circuit_mps = qpe_circuits_mps[0]
            lc = qpe_circuit_mps.estimate().logical_counts

            app = qpe_circuit_mps.get_qre_application()
            qre_result = estimate(
                app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
                name=f"{mol_name} MPS({chi})+QPE",
            )
            r = qre_result[-1]
            t_qpe = time.perf_counter() - t0_qpe

            sossa_only_tof = lc["cczCount"] - mps_lc["cczCount"]
            print(
                f"   [MPS+QPE] MPS Toffoli={mps_lc['cczCount']:,}, SOSSA Toffoli={sossa_only_tof:,}, "
                f"Total Toffoli={lc['cczCount']:,}"
            )
            print(
                f"   qubits={lc['numQubits']:,}, phys_qubits={r.qubits:,}, "
                f"runtime={r.runtime}, time={t_qpe:.1f}s"
            )
            extra = _print_full_qre_result(qre_result, f"{mol_name} MPS({chi})+QPE")

            all_results[mol_name]["mps_plus_qpe"].append(
                {
                    "label": str(chi),
                    "bond_dim": chi,
                    "mps_toffoli": mps_lc["cczCount"],
                    "sossa_toffoli": sossa_only_tof,
                    "total_toffoli": lc["cczCount"],
                    "logical_qubits": lc["numQubits"],
                    "physical_qubits": r.qubits,
                    "runtime_ns": r.runtime,
                    **extra,
                }
            )
            _save_results(all_results, output_json)

        # Real MPS + QPE (only for Fe2S2)
        if mol_name == "Fe2S2-20" and mps_real is not None:
            print(f"\n--- {mol_name} MPS(real) + MPS(real)+QPE ---")
            t0 = time.perf_counter()

            algo_mps = MPSSequentialStatePreparation()
            algo_mps.settings().set("rotation_bits", params["b_rot"])
            algo_mps.settings().set("fast_grouped_resource_estimation", True)
            mps_circ = algo_mps.run(mps_real)
            mps_lc = mps_circ.estimate().logical_counts

            # MPS-only QRE for real tensors (already done in Part 1 above,
            # but the circuit is needed for QPE anyway)
            qpe_circuits_real = qpe_builder.run(
                state_preparation=mps_circ, qubit_hamiltonian=fh
            )
            qpe_circuit_real = qpe_circuits_real[0]
            lc = qpe_circuit_real.estimate().logical_counts

            app = qpe_circuit_real.get_qre_application()
            qre_result = estimate(
                app, ARCHITECTURE, ISA_QUERY, TRACE_QUERY, max_error=MAX_ERROR,
                name=f"{mol_name} MPS(real)+QPE",
            )
            r = qre_result[-1]
            t_total = time.perf_counter() - t0

            sossa_only_tof = lc["cczCount"] - mps_lc["cczCount"]
            print(
                f"   MPS Toffoli={mps_lc['cczCount']:,}, SOSSA Toffoli={sossa_only_tof:,}, "
                f"Total Toffoli={lc['cczCount']:,}"
            )
            print(
                f"   qubits={lc['numQubits']:,}, phys_qubits={r.qubits:,}, "
                f"runtime={r.runtime}, time={t_total:.1f}s"
            )
            extra = _print_full_qre_result(qre_result, f"{mol_name} MPS(real)+QPE")

            all_results[mol_name]["mps_plus_qpe"].append(
                {
                    "label": "real(DMRG)",
                    "bond_dim": max(mps_real.bond_dims),
                    "mps_toffoli": mps_lc["cczCount"],
                    "sossa_toffoli": sossa_only_tof,
                    "total_toffoli": lc["cczCount"],
                    "logical_qubits": lc["numQubits"],
                    "physical_qubits": r.qubits,
                    "runtime_ns": r.runtime,
                    **extra,
                }
            )
            _save_results(all_results, output_json)

    # ===================================================================
    # Print summary table
    # ===================================================================
    print("\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)

    for mol_name, params in selected_molecules:
        print(f"\n{'='*70}")
        print(f"  {mol_name}  (N={params['N']}, R={params['R']}, B={params['B']}, C={params['C']})")
        print(f"{'='*70}")

        print("\n  MPS State Preparation Only:")
        if all_results[mol_name]["mps_only"]:
            df_mps = pd.DataFrame(all_results[mol_name]["mps_only"])
            print(df_mps.to_string(index=False))

        print("\n  SOSSA QPE with HF:")
        qpe_hf = all_results[mol_name]["qpe_hf"]
        if qpe_hf:
            print(f"    Toffoli: {qpe_hf['logical_toffoli']:,}")
            print(f"    Logical qubits: {qpe_hf['logical_qubits']:,}")
            print(f"    Physical qubits: {qpe_hf['physical_qubits']:,}")
            print(f"    Runtime: {qpe_hf['runtime_ns']}")
            print(f"    Compute distance: {qpe_hf['compute_distance']}")
            print(f"    Compute qubits: {qpe_hf['compute_qubits']}")
            print(f"    Num Ts/rotation: {qpe_hf['num_ts_per_rotation']}")
            if qpe_hf.get("factory_summary"):
                print(f"    Factories: {qpe_hf['factory_summary']}")

        print("\n  MPS + SOSSA QPE:")
        if all_results[mol_name]["mps_plus_qpe"]:
            df_combined = pd.DataFrame(all_results[mol_name]["mps_plus_qpe"])
            print(df_combined.to_string(index=False))

    print()
