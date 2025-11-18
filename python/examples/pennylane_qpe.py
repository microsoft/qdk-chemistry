"""Iterative QPE demo in PennyLane using exact matrix exponentials.

The controlled evolutions are implemented with ``qml.evolve`` so every unitary
is synthesized via a direct matrix exponential on PennyLane's backend—there is
no Trotter approximation in this example. Use it to cross-check the Trotterized
``sample_e2e_qpe_direct.py`` workflow or to explore PennyLane-native tooling.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pennylane as qml

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import QpeResult, Structure

ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 2
M_PRECISION = 12  # number of phase qubits ~ bits of precision
T_TIME = 0.1  # evolution time; lower if you see 2π wrap

# ------------------------------------------------------------------
# QDK/Chemistry calculation for H₂ (1.44 Bohr bond length in STO-3G)
# ------------------------------------------------------------------
structure = Structure(np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float), ["H", "H"])  # Geometry in bohr
nuclear_repulsion = structure.calculate_nuclear_repulsion_energy()

scf_solver = create("scf_solver")
scf_settings = scf_solver.settings()  # Access settings object
scf_settings.set("basis_set", "sto-3g")  # Set basis set
scf_energy, scf_wavefunction = scf_solver.solve(structure, charge=0, spin_multiplicity=1)

# Create ActiveSpaceSelector instance using alternative settings access:
#  Use convenience method to set settings during creation using keyword arguments
selector = create(
    "active_space_selector",
    "valence",
    num_active_electrons=ACTIVE_ELECTRONS,
    num_active_orbitals=ACTIVE_ORBITALS,
)
active_orbitals = selector.run(scf_wavefunction)

constructor = create("hamiltonian_constructor")
active_hamiltonian = constructor.run(active_orbitals)

n_alpha = n_beta = ACTIVE_ELECTRONS // 2
multi_configuration_calculator = create("multi_configuration_calculator")
casci_energy, _ = multi_configuration_calculator.calculate(active_hamiltonian, n_alpha, n_beta)

core_energy = active_hamiltonian.get_core_energy()

one_body = np.array(active_hamiltonian.get_one_body_integrals(), dtype=float)
norb = one_body.shape[0]
two_body_flat = np.array(active_hamiltonian.get_two_body_integrals(), dtype=float)
two_body = two_body_flat.reshape((norb,) * 4)  # Make a rank-4 tensor in chemists' notation (pq|rs)
two_body_phys = np.transpose(two_body, (0, 2, 1, 3))  # Transpose as Pennylane expects physicists' notation <pq|rs>

print("=== Generating QDK/Chemistry artifacts for H2 (0.76 Å, STO-3G) ===")
print(f"  SCF total energy:   {scf_energy: .8f} Hartree")
print(f"  CASCI total energy: {casci_energy: .8f} Hartree")

# ------------------------------------------------------------------
# Preparing the Hamiltonian and QPE circuit in PennyLane
# ------------------------------------------------------------------
constant = np.array([core_energy], dtype=float)
fermionic_sentence = qml.qchem.fermionic_observable(constant, one_body, two_body_phys)
H_qubit_raw = qml.jordan_wigner(fermionic_sentence)
num_spin_orbitals = len(H_qubit_raw.wires)
num_spatial_orbitals = num_spin_orbitals // 2

phase_wires = list(range(M_PRECISION))
sys_wires = [w + M_PRECISION for w in range(num_spin_orbitals)]
wire_map = dict(zip(H_qubit_raw.wires, sys_wires, strict=False))
H_qubit = H_qubit_raw.map_wires(wire_map)
powers_of_two = [2**i for i in range(len(phase_wires))]

print(f"  Hamiltonian terms: {len(H_qubit)}")
print(f"  System qubits (spin orbitals): {num_spin_orbitals}")
print(f"  Electron sector (alpha, beta): ({n_alpha}, {n_beta})")

dev = qml.device("default.qubit", wires=phase_wires + sys_wires, shots=None)


@qml.qnode(dev)
def qpe():
    """Run the PennyLane QPE circuit and return phase-register probabilities."""
    for w in phase_wires:
        qml.Hadamard(wires=w)

    alpha_wires = sys_wires[:num_spatial_orbitals]
    beta_wires = sys_wires[num_spatial_orbitals:]
    for wire in alpha_wires[:n_alpha]:
        qml.PauliX(wires=wire)
    for wire in beta_wires[:n_beta]:
        qml.PauliX(wires=wire)

    for exponent, ctrl_wire in zip(powers_of_two[::-1], phase_wires, strict=False):
        qml.ctrl(qml.evolve, control=ctrl_wire)(H_qubit, exponent * T_TIME)

    qml.adjoint(qml.QFT)(wires=phase_wires)
    return qml.probs(wires=phase_wires)


# ------------------------------------------------------------------
# Simulate the QPE circuit and analyze the results
# ------------------------------------------------------------------
probs = qpe()
bit_labels = [format(i, f"0{M_PRECISION}b") for i in range(len(probs))]

dominant_index = int(np.argmax(probs))
dominant_bits = bit_labels[dominant_index]
phase_fraction = dominant_index / (2**M_PRECISION)
reference_total = casci_energy
result = QpeResult.from_phase_fraction(
    method="pennylane_qpe",
    phase_fraction=phase_fraction,
    evolution_time=T_TIME,
    bitstring_msb_first=dominant_bits,
    reference_energy=reference_total,
)
raw_energy = result.raw_energy
candidate_energies = result.branching
estimated_total_energy = result.resolved_energy if result.resolved_energy is not None else raw_energy

print(f"\nMost likely phase bitstring: {dominant_bits}")
print(f"Phase fraction φ (measured): {result.phase_fraction:.6f}")
if not np.isclose(result.phase_fraction, result.canonical_phase_fraction):
    print(
        f"Canonical phase fraction φ: {result.canonical_phase_fraction:.6f} "
        f"(angle = {result.canonical_phase_angle:.6f} rad)",
    )
print(f"Estimated total energy:        {estimated_total_energy:.8f} Hartree")
print("Candidate energies (alias checks):")
for energy in candidate_energies:
    print(f"  E = {energy:+.8f} Hartree")
print(f"Reference total energy (CASCI): {reference_total:.8f} Hartree")
print(f"Total energy difference (QPE - CASCI): {estimated_total_energy - reference_total:+.8e} Hartree")
print(
    "\nDiagnostic: PennyLane's controlled evolve applies exp(-i H t) exactly, so this residual "
    "difference is dominated by finite phase-register resolution rather than Trotterization."
)
