# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

r"""Iterative QPE demo that highlights Trotterized time evolution.

This script walks through the QDK/Chemistry's SCF → CASCI → IQPE pipeline and builds the
controlled time-evolution unitaries with Trotterized decomposition.
The measured phase therefore reflects the usual trade-off between circuit depth and
Trotter error.
"""

import numpy as np

try:
    from qiskit import qasm3, transpile
    from qiskit_aer import AerSimulator
except ImportError as ex:
    raise ImportError(
        "Qiskit and Qiskit-Aer must be installed to run this example. "
        "Please install via 'pip install qiskit qiskit-aer'.",
    ) from ex

from qdk_chemistry.algorithms import (
    IterativePhaseEstimation,
    create,
)
from qdk_chemistry.data import QpeResult, Structure
from qdk_chemistry.utils.wavefunction import get_top_determinants

ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 2
M_PRECISION = 10  # number of phase qubits ~ bits of precision
T_TIME = 0.1  # evolution time; lower if you see 2π wrap
SHOTS_PER_ITERATION = 10
SIMULATOR_SEED = 42


########################################################################################
# 1. QDK/Chemistry calculation for H₂ (0.76 Å, STO-3G)
########################################################################################
structure = Structure(
    np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float), ["H", "H"]
)  # Geometry in bohr

scf_solver = create("scf_solver", basis_set="sto-3g")
scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)


########################################################################################
# 2. Find active-space Hamiltonian and CASCI energy
########################################################################################
selector = create(
    "active_space_selector",
    "qdk_valence",
    num_active_electrons=ACTIVE_ELECTRONS,
    num_active_orbitals=ACTIVE_ORBITALS,
)
active_orbitals = selector.run(scf_wavefunction).get_orbitals()

constructor = create("hamiltonian_constructor")
active_hamiltonian = constructor.run(active_orbitals)

n_alpha = n_beta = ACTIVE_ELECTRONS // 2
mc_calculator = create("multi_configuration_calculator")
casci_energy, casci_wavefunction = mc_calculator.run(
    active_hamiltonian, n_alpha, n_beta
)
core_energy = active_hamiltonian.get_core_energy()

print("=== Generating QDK/Chemistry artifacts for H2 (0.76 Å, STO-3G) ===")
print(f"  SCF total energy:   {scf_energy: .8f} Hartree")
print(f"  CASCI total energy: {casci_energy: .8f} Hartree")


########################################################################################
# 3. Preparing the qubit Hamiltonian and sparse-isometry trial state
########################################################################################
qubit_mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
qubit_hamiltonian = qubit_mapper.run(active_hamiltonian)

qubit_pauli_op = qubit_hamiltonian.pauli_ops
num_spin_orbitals = qubit_hamiltonian.num_qubits

top_configurations = get_top_determinants(casci_wavefunction, max_determinants=2)
pmc = create("projected_multi_configuration_calculator")
E_sparse, sparse_wavefunction = pmc.run(
    active_hamiltonian, list(top_configurations.keys())
)

sparse_state_prep = create("state_prep", algorithm_name="sparse_isometry_gf2x")
state_prep = qasm3.loads(sparse_state_prep.run(sparse_wavefunction))
state_prep = transpile(
    state_prep,
    basis_gates=["cx", "rz", "h", "x", "s", "sdg"],
    optimization_level=1,
)
state_prep.name = "casci_sparse_isometry"

print("\nSparse-isometry state preparation circuit:")
print(state_prep.draw(output="text"))


########################################################################################
# 4. Build and run the Trotterized iterative QPE circuit
########################################################################################
simulator = AerSimulator(method="statevector", seed_simulator=SIMULATOR_SEED)
iqpe = IterativePhaseEstimation(qubit_hamiltonian, T_TIME)

print("\n=== Running iterative phase estimation (Trotterized) ===")
print(f"  Hamiltonian terms: {len(qubit_pauli_op.paulis)}")
print(f"  System qubits (spin orbitals): {num_spin_orbitals}")
print(f"  Electron sector (alpha, beta): ({n_alpha}, {n_beta})")

phase_feedback = 0.0
bits: list[int] = []

for iteration in range(M_PRECISION):
    iter_info = iqpe.create_iteration(
        state_prep,
        iteration=iteration,
        total_iterations=M_PRECISION,
        phase_correction=phase_feedback,
    )
    compiled = transpile(iter_info.circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=SHOTS_PER_ITERATION).result()
    counts = result.get_counts()
    measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

    bits.append(measured_bit)
    phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)


########################################################################################
# 5. Process and display results
########################################################################################
result = QpeResult.from_phase_fraction(
    method=IterativePhaseEstimation.algorithm,
    phase_fraction=phase_fraction,
    evolution_time=T_TIME,
    bits_msb_first=bits,
    reference_energy=casci_energy,
)

phase_angle_measured = result.phase_angle
phase_angle_canonical = result.canonical_phase_angle
raw_energy = result.raw_energy
candidate_energies = result.branching
estimated_electronic_energy = (
    result.resolved_energy if result.resolved_energy is not None else raw_energy
)
estimated_total_energy = estimated_electronic_energy + core_energy

print(f"Measured bits (MSB → LSB): {list(result.bits_msb_first or [])}")
print(
    f"Phase fraction φ (measured): {result.phase_fraction:.6f} (angle = {phase_angle_measured:.6f} rad)"
)
if not np.isclose(result.phase_fraction, result.canonical_phase_fraction):
    print(
        f"Canonical phase fraction φ: {result.canonical_phase_fraction:.6f} (angle = {phase_angle_canonical:.6f} rad)",
    )
print(f"Raw energy_from_phase output: {raw_energy:+.8f} Hartree")
print("Candidate energies (alias checks):")
for energy in candidate_energies:
    print(f"  E = {energy:+.8f} Hartree")
print(f"Estimated electronic energy: {estimated_electronic_energy:.8f} Hartree")
print(f"Estimated total energy: {estimated_total_energy:.8f} Hartree")
print(f"Reference total energy (CASCI): {casci_energy:.8f} Hartree")
print(f"Reference sparse energy (CASCI): {E_sparse:.8f} Hartree")
iterative_energy_error = estimated_total_energy - casci_energy
print(f"Energy difference (QPE - CASCI): {iterative_energy_error:+.8e} Hartree")
print(
    "Energy error is large due to Trotterization and finite numerical resolution in this demo."
)
