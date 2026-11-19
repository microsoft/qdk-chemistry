"""Known-state and molecular examples of real stabilizer 2-Renyi entropy."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# start-cell-setup
from math import log2

import numpy as np
from numpy.typing import NDArray
from qdk_chemistry import algorithms
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    StateVectorContainer,
    Structure,
    Wavefunction,
)


def create_qubit_wavefunction(coefficients: NDArray[np.float64]) -> Wavefunction:
    """Represent a small mathematical qubit state using public configuration APIs."""
    num_qubits = coefficients.size.bit_length() - 1
    configurations = [
        Configuration.from_bitstring(format(x, f"0{num_qubits}b")[::-1])
        for x in range(coefficients.size)
    ]
    return Wavefunction(
        StateVectorContainer(coefficients, configurations, ModelOrbitals(num_qubits))
    )


# end-cell-setup

# start-cell-known-states
sre_calculator = algorithms.create("stabilizer_renyi_entropy", "replica")
h_state = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])
two_h_states = np.kron(h_state, h_state)
entangled_magic = np.empty_like(two_h_states)
# CNOT: qubit zero controls a flip of qubit one.
for x, amplitude in enumerate(two_h_states):
    entangled_magic[x ^ ((x & 1) << 1)] = amplitude

known_states = [
    ("Plus", np.array([1.0, 1.0]) / np.sqrt(2), 0.0),
    ("H", h_state, log2(4 / 3)),
    ("Bell Phi+", np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2), 0.0),
    ("Bell Phi-", np.array([1.0, 0.0, 0.0, -1.0]) / np.sqrt(2), 0.0),
    ("Bell Psi+", np.array([0.0, 1.0, 1.0, 0.0]) / np.sqrt(2), 0.0),
    ("Bell Psi-", np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2), 0.0),
    ("H tensor H", two_h_states, 2 * log2(4 / 3)),
    ("CNOT(H tensor H)", entangled_magic, 2 * log2(4 / 3)),
]
for name, coefficients, expected in known_states:
    result = sre_calculator.run(create_qubit_wavefunction(coefficients))
    np.testing.assert_allclose(result.entropy, expected, atol=1e-12, rtol=0)
    print(f"{name}: M2 = {result.entropy:.12f} bits; expected {expected:.12f}")
# end-cell-known-states

# start-cell-h2
structure = Structure.from_xyz("""\
2
H2 at 0.740848 angstrom
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740848
""")
_, mean_field = algorithms.create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
hamiltonian = algorithms.create("hamiltonian_constructor").run(
    mean_field.get_orbitals()
)
energy, ground_state = algorithms.create(
    "multi_configuration_calculator", "macis_cas"
).run(hamiltonian, 1, 1)
h2_result = sre_calculator.run(ground_state)
print(f"H2 CAS(2,2)/STO-3G energy: {energy:.10f} Eh")
print(h2_result.get_summary())
# end-cell-h2
