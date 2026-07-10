"""Minimal iQPE for H2/STO-3G (4 qubits, 1 iteration) → QIR → AC1000 emulator.

Runs SCF on H2 in minimal basis, maps to a 4-qubit Jordan-Wigner Hamiltonian,
generates one iQPE iteration circuit, and submits the QIR to the AC1000
emulator via the qdk_chemistry Azure Quantum plugin.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, MajoranaMapping
from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumEmulator
from qdk_chemistry.utils import Logger

# Silence library logs
Logger.set_global_level(Logger.LogLevel.off)

# =============================================================================
# Azure Quantum configuration (from environment variables)
# =============================================================================
WORKSPACE_SUBSCRIPTION_ID = os.environ.get("AZURE_QUANTUM_SUBSCRIPTION_ID", "")
WORKSPACE_RESOURCE_GROUP = os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP", "")
WORKSPACE_NAME = os.environ.get("AZURE_QUANTUM_WORKSPACE_NAME", "")
WORKSPACE_LOCATION = os.environ.get("AZURE_QUANTUM_LOCATION", "")
TARGET_NAME = os.environ.get("AZURE_QUANTUM_TARGET_NAME", "")

# =============================================================================
# iQPE parameters
# =============================================================================
M_PRECISION = 1  # 1 bit → 1 iteration
SHOTS_PER_BIT = 1
EMULATOR_SHOTS = 100


# =============================================================================
# 1. Build H2/STO-3G system (4 qubits)
# =============================================================================
def build_h2_system():
    """SCF on H2 at equilibrium, Jordan-Wigner map → 4-qubit Hamiltonian.

    Returns qubit_hamiltonian, state_prep_circuit, and reference energy.
    """
    import qdk_chemistry.plugins.qiskit  # noqa: F401
    from qdk_chemistry.data import Structure

    # H2 at ~0.74 Å (equilibrium)
    xyz_content = "2\nH2 equilibrium\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7414\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_path = Path(f.name)
    structure = Structure.from_xyz_file(xyz_path)
    print("Molecule: H2 at 0.7414 Å")

    # SCF in minimal basis
    scf_solver = create("scf_solver")
    e_hf, wfn_hf = scf_solver.run(
        structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
    )
    print(f"HF energy: {e_hf:.6f} Hartree")

    # Build the molecular Hamiltonian (no active-space reduction → full 4 spin-orbitals)
    ham_constructor = create("hamiltonian_constructor")
    hamiltonian = ham_constructor.run(wfn_hf.get_orbitals())

    # Exact diagonalization for reference
    alpha_e, beta_e = wfn_hf.get_active_num_electrons()
    mc = create("multi_configuration_calculator", "macis_cas")
    e_fci, wfn_fci = mc.run(hamiltonian, alpha_e, beta_e)
    print(f"FCI energy: {e_fci:.6f} Hartree")

    # Jordan-Wigner qubit Hamiltonian
    n_spin_orbitals = 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
    qubit_mapper = create("qubit_mapper", "qiskit")
    qubit_hamiltonian = qubit_mapper.run(
        hamiltonian, MajoranaMapping.jordan_wigner(n_spin_orbitals)
    )
    print(f"Qubit Hamiltonian: {qubit_hamiltonian.get_summary()}")

    # State prep: FCI wavefunction via sparse isometry
    state_prep = create("state_prep", "sparse_isometry_gf2x")
    state_prep_circuit = state_prep.run(wfn_fci)
    print("State prep circuit built (sparse isometry GF2+X)")

    return qubit_hamiltonian, state_prep_circuit, hamiltonian, e_fci


# =============================================================================
# 2. Generate one iQPE iteration circuit → QIR
# =============================================================================
def generate_single_iqpe_qir(state_prep_circuit, qubit_hamiltonian):
    """Generate a single iQPE iteration circuit and return its QIR string."""
    evolution_time = np.pi / qubit_hamiltonian.schatten_norm
    print(f"Evolution time: {evolution_time:.6f}")

    unitary_builder = AlgorithmRef(
        "hamiltonian_unitary_builder", "trotter", time=evolution_time
    )
    circuit_mapper = AlgorithmRef("controlled_circuit_mapper", "pauli_sequence")

    builder = create(
        "qpe_circuit_builder",
        "qdk_iterative",
        num_bits=M_PRECISION,
        num_iteration=0,
        unitary_builder=unitary_builder,
        controlled_circuit_mapper=circuit_mapper,
    )
    circuits = builder.run(state_prep_circuit, qubit_hamiltonian)
    circuit = circuits[0]

    qir_string = str(circuit.get_qir())
    print(f"QIR generated ({len(qir_string)} chars)")

    return qir_string, evolution_time


# =============================================================================
# 3. Submit to AC1000 emulator via plugin
# =============================================================================
def get_executor():
    """Create an AzureQuantumEmulator executor (no Clifford rounding, no seed)."""
    return AzureQuantumEmulator(
        subscription_id=WORKSPACE_SUBSCRIPTION_ID,
        resource_group=WORKSPACE_RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        location=WORKSPACE_LOCATION,
        target_name=TARGET_NAME,
        simulation_type="cliffordrounding",
        seed=-1,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    """H2/STO-3G iQPE → QIR → AC1000."""
    t0 = time.perf_counter()

    print("=" * 60)
    print("Step 1: Build H2/STO-3G system (4 qubits)")
    print("=" * 60)
    qubit_hamiltonian, state_prep_circuit, hamiltonian, e_fci = build_h2_system()

    print("\n" + "=" * 60)
    print("Step 2: Generate single iQPE iteration → QIR")
    print("=" * 60)
    qir_string, evolution_time = generate_single_iqpe_qir(
        state_prep_circuit, qubit_hamiltonian
    )

    # Save QIR locally
    qir_path = Path("iqpe_h2_iter0.qir")
    qir_path.write_text(qir_string)
    print(f"QIR saved to {qir_path}")

    print("\n" + "=" * 60)
    print("Step 3: Submit to AC1000 emulator (clifford mode, no seed)")
    print("=" * 60)
    from qdk_chemistry.data import Circuit

    circuit = Circuit(qir=qir_string)
    executor = get_executor()
    print(f"Target: {TARGET_NAME}")
    result = executor.run(circuit, shots=EMULATOR_SHOTS)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"FCI reference energy: {e_fci:.6f} Hartree")
    print(f"Bitstring counts: {result.bitstring_counts}")
    print(f"Total shots: {result.total_shots}")
    print(f"Wall time:   {time.perf_counter() - t0:.1f}s")


def generate_only():
    """Generate QIR without submitting."""
    qubit_hamiltonian, state_prep_circuit, hamiltonian, e_fci = build_h2_system()
    qir_string, evolution_time = generate_single_iqpe_qir(
        state_prep_circuit, qubit_hamiltonian
    )
    qir_path = Path("iqpe_h2_iter0.qir")
    qir_path.write_text(qir_string)
    print(f"\nQIR written to {qir_path}")
    print(f"FCI reference energy: {e_fci:.6f} Hartree")
    print(f"Evolution time: {evolution_time:.6f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-only":
        generate_only()
    else:
        main()
