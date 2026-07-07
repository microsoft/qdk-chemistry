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

Note:
  Recommended to run on HPC due to high memory requirements at large bond
  dimensions.

Prerequisites:
  - pip install 'qdk-chemistry[qre]'

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
import math
import time
from pathlib import Path

import pandas as pd
from qdk.qre import PSSPC, LatticeSurgery, estimate
from qdk.qre.models import Majorana, RoundBasedFactory, ThreeAux
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import (
    QdkStandardQpeCircuitBuilder,
)
from qdk_chemistry.algorithms.state_preparation import MPSSequentialStatePreparation
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.utils import Logger

from utils import (
    extract_qre_extras,
    make_fake_factorized_hamiltonian,
    print_qre_table,
    save_results,
)

# Suppress verbose logging
Logger.set_global_level(Logger.LogLevel.info)

# ============================================================================
# Configuration
# ============================================================================

# Molecule parameters (from Low2025)
MOLECULES = [
    ("Fe2S2-20", dict(N=20, R=14, B=15, C=5, b_rot=15, b_coeff=11, lambda_eff=6.4690)),
    ("Fe4S4-36", dict(N=36, R=9, B=18, C=18, b_rot=17, b_coeff=13, lambda_eff=14.9842)),
    ("FeMoCo-76", dict(N=76, R=15, B=57, C=19, b_rot=15, b_coeff=9,  lambda_eff=43.6538)),
]

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
        choices=[m[0] for m in MOLECULES] + ["all"],
        help=f"Which molecules to run. Choices: {[m[0] for m in MOLECULES]} or 'all'. Default: all",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve which molecules to run
    if "all" in args.molecules:
        selected_molecules = MOLECULES
    else:
        selected_molecules = [(name, {m[0]: m[1] for m in MOLECULES}[name]) for name in args.molecules]

    bond_dims = sorted(args.bond_dims)
    output_json = args.output

    Logger.info("\nMPS + SOSSA QPE Resource Estimation")
    Logger.info("=" * 70)
    Logger.info(f"   Molecules: {[m[0] for m in selected_molecules]}")
    Logger.info(f"   Bond dimensions: {bond_dims}")
    Logger.info(f"   Site dim: {SITE_DIM}, Phase bits: {[m[1]['b_rot'] for m in selected_molecules]}")
    Logger.info(f"   sigma_E = {SIGMA_E} Ha, max_error = {MAX_ERROR}")
    Logger.info(f"   Output: {output_json}")
    Logger.info("")

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
    # Loop over each molecule
    # ===================================================================
    for mol_name, params in selected_molecules:
        N = params["N"]
        num_sites = N  # each spatial orbital is one site with dim=4

        lambda_eff = params["lambda_eff"]
        num_queries = math.ceil(math.pi * lambda_eff / (2 * SIGMA_E))
        num_phase_qubits = math.ceil(math.log2(num_queries))

        Logger.info("\n" + "#" * 70)
        Logger.info(f"# MOLECULE: {mol_name}")
        Logger.info(f"#   N={N}, R={params['R']}, B={params['B']}, C={params['C']}")
        Logger.info(f"#   lambda_eff={lambda_eff}, queries={num_queries}, phase_qubits={num_phase_qubits}")
        Logger.info("#" * 70)

        # ---------------------------------------------------------------
        # SOSSA QPE with HF initial state
        # ---------------------------------------------------------------
        Logger.info(f"\n{'='*70}")
        Logger.info(f"  SOSSA QPE with HF Initial State -- {mol_name}")
        Logger.info(f"{'='*70}")
        Logger.info(f"   N={N}, lambda_eff={lambda_eff}, sigma_E={SIGMA_E}")
        Logger.info(f"   Heisenberg queries: {num_queries}, phase qubits: {num_phase_qubits}")

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
        Logger.info(
            f"   Toffoli={lc['cczCount']:,}, qubits={lc['numQubits']:,}, "
            f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_total:.1f}s"
        )
        print_qre_table(qre_result, f"{mol_name} SOSSA QPE (HF)")
        extra = extract_qre_extras(qre_result)
        all_results[mol_name]["qpe_hf"] = {
            "label": "QPE(HF)",
            "logical_toffoli": lc["cczCount"],
            "logical_qubits": lc["numQubits"],
            "physical_qubits": r.qubits,
            "runtime_ns": r.runtime,
            **extra,
        }
        save_results(all_results, output_json)

        # ---------------------------------------------------------------
        # MPS-only + MPS+QPE
        # ---------------------------------------------------------------
        Logger.info(f"\n{'='*70}")
        Logger.info(f"  PARTS 1 & 3: MPS State Prep + MPS+QPE -- {mol_name}")
        Logger.info(f"{'='*70}")

        # Fake MPS at various bond dims
        for chi in bond_dims:
            Logger.info(f"\n--- {mol_name} MPS chi={chi} (MPS-only + MPS+QPE) ---")
            t0 = time.perf_counter()

            mps_wfn = MPSWavefunction.random(
                num_sites=num_sites, bond_dim=chi, site_dim=SITE_DIM
            )
            max_chi_fake = max(mps_wfn.bond_dims)
            Logger.info(f"   Max bond dim: {max_chi_fake}, Sites: {num_sites}")

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
            Logger.info(
                f"   [MPS-only] Toffoli={mps_lc['cczCount']:,}, qubits={mps_lc['numQubits']:,}, "
                f"phys_qubits={r.qubits:,}, runtime={r.runtime}, time={t_mps:.1f}s"
            )
            print_qre_table(qre_result, f"{mol_name} MPS chi={chi}")
            extra = extract_qre_extras(qre_result)
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
            save_results(all_results, output_json)

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
            Logger.info(
                f"   [MPS+QPE] MPS Toffoli={mps_lc['cczCount']:,}, SOSSA Toffoli={sossa_only_tof:,}, "
                f"Total Toffoli={lc['cczCount']:,}"
            )
            Logger.info(
                f"   qubits={lc['numQubits']:,}, phys_qubits={r.qubits:,}, "
                f"runtime={r.runtime}, time={t_qpe:.1f}s"
            )
            print_qre_table(qre_result, f"{mol_name} MPS({chi})+QPE")
            extra = extract_qre_extras(qre_result)
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
            save_results(all_results, output_json)
