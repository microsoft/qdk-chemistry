# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

r"""Iterative Quantum Phase Estimation demos for non-commuting two-qubit Hamiltonians.

This example demonstrates the use of the native IterativePhaseEstimation algorithm
from QDK\Chemistry to estimate ground state energies of non-commuting Hamiltonians.

Two examples are shown:
1. H = 0.519 * (X ⊗ I) + (Z ⊗ Z) - A simple non-commuting Hamiltonian
2. H = -0.0289(X₁ + X₂) + 0.0541(Z₁ + Z₂) + 0.0150 X₁X₂ + 0.0590 Z₁Z₂ -
   A more complex molecular-inspired Hamiltonian

The IterativePhaseEstimation class handles the QPE protocol, iteratively measuring
phase bits and updating feedback corrections to extract accurate eigenvalue estimates.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qdk.chemistry.algorithms import IterativePhaseEstimation
from qdk.chemistry.data import QpeResult, QubitHamiltonian

TIME_STEP = np.pi / 4
NUM_BITS = 6
SHOTS_PER_ITERATION = 3
SIMULATOR_SEED = 42
REFERENCE_ENERGY_1 = 1.1266592208826944

# Hamiltonian H = 0.519 * (X ⊗ I) + (Z ⊗ Z)
hamiltonian_op = SparsePauliOp.from_list(
    [
        ("XI", 0.519),
        ("ZZ", 1.0),
    ]
)
hamiltonian = QubitHamiltonian(pauli_strings=hamiltonian_op.paulis.to_labels(), coefficients=hamiltonian_op.coeffs)

# Trial state: 0.9714|00> + 0.2370|10>
trial_state = np.array([0.97, 0.0, np.sqrt(1 - 0.97**2), 0.0], dtype=complex)
state_prep = QuantumCircuit(2, name="trial")
state_prep.initialize(trial_state, [0, 1])

iqpe = IterativePhaseEstimation(hamiltonian, TIME_STEP)
simulator = AerSimulator(seed_simulator=SIMULATOR_SEED)

phase_feedback = 0.0
bits: list[int] = []

for iteration in range(NUM_BITS):
    iteration_info = iqpe.create_iteration(
        state_prep,
        iteration=iteration,
        total_iterations=NUM_BITS,
        phase_correction=phase_feedback,
    )
    compiled = transpile(iteration_info.circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=SHOTS_PER_ITERATION).result()
    counts = result.get_counts()
    measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

    bits.append(measured_bit)
    phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)
result_1 = QpeResult.from_phase_fraction(
    method=IterativePhaseEstimation.algorithm,
    phase_fraction=phase_fraction,
    evolution_time=TIME_STEP,
    bits_msb_first=bits,
    reference_energy=REFERENCE_ENERGY_1,
)
phase_angle = result_1.phase_angle
phase_angle_canonical = result_1.canonical_phase_angle
raw_energy = result_1.raw_energy
candidate_energies = result_1.branching
estimated_energy = result_1.resolved_energy if result_1.resolved_energy is not None else raw_energy

print("=== Iterative QPE: Non-commuting Hamiltonian Example ===")
print("Hamiltonian: H = 0.519 * XI + ZZ")
print(f"Time step t = pi / 4 ({TIME_STEP:.6f}) and {NUM_BITS} phase bits\n")
print(f"Measured bits (MSB → LSB): {list(result_1.bits_msb_first or [])}")
print(f"Phase fraction φ (measured): {result_1.phase_fraction:.6f}")
print(f"Phase angle     (measured) : {phase_angle:.6f} rad")
if not np.isclose(result_1.phase_fraction, result_1.canonical_phase_fraction):
    print(
        f"Canonical phase fraction φ: {result_1.canonical_phase_fraction:.6f} "
        f"(angle = {phase_angle_canonical:.6f} rad)",
    )
print(f"Raw energy_from_phase output: {raw_energy:+.8f} Hartree")
print("Candidate energies (alias checks):")
for energy in candidate_energies:
    print(f"  E = {energy:+.8f} Hartree")
print(f"\nReference energy: {REFERENCE_ENERGY_1:+.8f} Hartree")
print(f"Estimated energy: {estimated_energy:+.8f} Hartree")


# --- Iterative QPE on H = -0.0289(X1 + X2) + 0.0541(Z1 + Z2) + 0.0150 X1X2 + 0.0590 Z1Z2 ---

TIME_STEP_2 = np.pi / 4
NUM_BITS_2 = 11
SHOTS_PER_ITERATION_2 = 3
SIMULATOR_SEED_2 = 42
REFERENCE_ENERGY = -0.0887787

hamiltonian_op_2 = SparsePauliOp.from_list(
    [
        ("XI", -0.0289),
        ("IX", -0.0289),
        ("ZI", 0.0541),
        ("IZ", 0.0541),
        ("XX", 0.0150),
        ("ZZ", 0.0590),
    ]
)
hamiltonian_2 = QubitHamiltonian(hamiltonian_op_2.paulis.to_labels(), hamiltonian_op_2.coeffs)

trial_state_2 = np.array([0.0, 0.47, 0.47, 0.75], dtype=complex)
trial_state_2 /= np.linalg.norm(trial_state_2)
state_prep_2 = QuantumCircuit(2, name="trial_2")
state_prep_2.initialize(trial_state_2, [0, 1])

iqpe_2 = IterativePhaseEstimation(hamiltonian_2, TIME_STEP_2)
simulator_2 = AerSimulator(seed_simulator=SIMULATOR_SEED_2)

phase_feedback_2 = 0.0
bits_2: list[int] = []

for iteration in range(NUM_BITS_2):
    iteration_info = iqpe_2.create_iteration(
        state_prep_2,
        iteration=iteration,
        total_iterations=NUM_BITS_2,
        phase_correction=phase_feedback_2,
    )
    compiled = transpile(iteration_info.circuit, simulator_2, optimization_level=0)
    result = simulator_2.run(compiled, shots=SHOTS_PER_ITERATION_2).result()
    counts = result.get_counts()
    measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

    bits_2.append(measured_bit)
    phase_feedback_2 = iqpe_2.update_phase_feedback(phase_feedback_2, measured_bit)

phase_fraction_2 = iqpe_2.phase_fraction_from_feedback(phase_feedback_2)
result_2 = QpeResult.from_phase_fraction(
    method=IterativePhaseEstimation.algorithm,
    phase_fraction=phase_fraction_2,
    evolution_time=TIME_STEP_2,
    bits_msb_first=bits_2,
    reference_energy=REFERENCE_ENERGY,
)
phase_angle_2 = result_2.phase_angle
phase_angle_canonical_2 = result_2.canonical_phase_angle
raw_energy_2 = result_2.raw_energy
candidate_energies_2 = result_2.branching
estimated_energy_2 = result_2.resolved_energy if result_2.resolved_energy is not None else raw_energy_2

print("\n=== Iterative QPE: Second Non-commuting Hamiltonian Example ===")
print("Hamiltonian: H = -0.0289(X1 + X2) + 0.0541(Z1 + Z2) + 0.0150 X1X2 + 0.0590 Z1Z2")
print(f"Time step t = pi / 4 ({TIME_STEP_2:.6f}) and {NUM_BITS_2} phase bits\n")
print(f"Measured bits (MSB → LSB): {list(result_2.bits_msb_first or [])}")
print(f"Phase fraction φ (measured): {result_2.phase_fraction:.6f}")
print(f"Phase angle     (measured) : {phase_angle_2:.6f} rad")
if not np.isclose(result_2.phase_fraction, result_2.canonical_phase_fraction):
    print(
        f"Canonical phase fraction φ: {result_2.canonical_phase_fraction:.6f} "
        f"(angle = {phase_angle_canonical_2:.6f} rad)",
    )
print(f"Raw energy_from_phase output: {raw_energy_2:+.8f} Hartree")
print("Candidate energies (alias checks):")
for energy in candidate_energies_2:
    print(f"  E = {energy:+.8f} Hartree")
print(f"\nReference energy: {REFERENCE_ENERGY:+.8f} Hartree")
print(f"Estimated energy: {estimated_energy_2:+.8f} Hartree")
