"""Executable MPS, controlled-SOSSA, and LCU examples for the 2.4 release notes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-mps
import numpy as np

from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    MPSContainer,
    MPSSite,
    Wavefunction,
)
from qdk_chemistry.data import symmetry as sym

product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
physical_basis = [
    Configuration.from_spin_half_string(state) for state in ["0", "u", "d", "2"]
]
sites = []
for occupied_state in [1, 2]:
    values = np.zeros((4, 1))  # Flattened (left bond, physical) by right bond.
    values[occupied_state, 0] = 1.0
    tensor = sym.SymmetryBlockedTensorRank3(
        [product] * 3,
        [{label: 1}, {label: 4}, {label: 1}],
        [((label,) * 3, values)],
    )
    sites.append(MPSSite(tensor, [label], [label], [label], physical_basis))

mps = Wavefunction(MPSContainer(sites, ModelOrbitals(2), orthogonality_center=0))
restored = Wavefunction.from_json(mps.to_json())
assert restored.content_hash() == mps.content_hash()
print(f"MPS sites: {restored.get_container().num_sites()}")
# end-cell-mps
################################################################################

################################################################################
# start-cell-sossa-input
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, MajoranaMapping, Structure

structure = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])  # Bohr.
_, reference = create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
cholesky = create(
    "hamiltonian_constructor", "qdk_cholesky", cholesky_tolerance=1e-8
).run(reference.get_orbitals())
factorized = create("hamiltonian_factorization", "double_factorization").run(cholesky)
num_orbitals = factorized.get_container().get_num_orbitals()
operator = create("qubit_mapper", "sum_of_squares", threshold=0.0).run(
    factorized, MajoranaMapping.jordan_wigner(2 * num_orbitals)
)
trial_state = create("state_prep", "sparse_isometry").run(reference)
# end-cell-sossa-input
################################################################################

################################################################################
# start-cell-standard-sossa
qpe_builder = create(
    "qpe_circuit_builder",
    "qdk_standard",
    num_bits=3,
    unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    controlled_circuit_mapper=AlgorithmRef(
        "controlled_circuit_mapper",
        "sossa",
        outer_prepare_algorithm=AlgorithmRef("state_prep", "dense_pure_state"),
        inner_prepare_algorithm="direct",
        select_algorithm="qrom_phase_gradient",
        rotation_bit_precision=8,
    ),
)
(circuit,) = qpe_builder.run(trial_state, operator)
print(f"Standard SOSSA QPE qubits: {circuit.num_qubits}")
# end-cell-standard-sossa
################################################################################

################################################################################
# start-cell-lcu-select
from qdk_chemistry.data import QubitOperator

pauli_operator = QubitOperator.from_sparse_terms(
    2, [{0: "Z"}, {0: "X", 1: "X"}], np.array([0.5, 0.25])
)
lcu = create("hamiltonian_unitary_builder", "lcu").run(pauli_operator)
unary_select = create("circuit_mapper", "prepare_select_prepare").run(lcu)
legacy_select = create(
    "circuit_mapper", "prepare_select_prepare", select_method="multi_controlled"
).run(lcu)
print(f"LCU callable widths: {unary_select.num_qubits}, {legacy_select.num_qubits}")
# end-cell-lcu-select
################################################################################
