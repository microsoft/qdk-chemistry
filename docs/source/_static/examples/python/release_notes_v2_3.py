"""Executable double-factorization and unary-SOSSA examples for the 2.3 release notes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import ceil, pi

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, MajoranaMapping, Structure


################################################################################
# start-cell-factorization
structure = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1, 1])
_, scf_wavefunction = create("scf_solver").run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
cholesky = create(
    "hamiltonian_constructor", "qdk_cholesky", cholesky_tolerance=1e-8
).run(scf_wavefunction.get_orbitals())
factorized = create("hamiltonian_factorization", "double_factorization").run(cholesky)
# end-cell-factorization
################################################################################

################################################################################
# start-cell-sossa
num_alpha, num_beta = scf_wavefunction.get_active_num_electrons()
reference_energy, reference_wavefunction = create(
    "multi_configuration_calculator", "macis_cas"
).run(cholesky, num_alpha, num_beta)

num_orbitals = factorized.get_container().get_num_orbitals()
operator = create("qubit_mapper", "sum_of_squares", threshold=0.0).run(
    factorized, MajoranaMapping.jordan_wigner(2 * num_orbitals)
)
unitary = create(
    "hamiltonian_unitary_builder",
    "sossa",
    reference_ground_state_energy=reference_energy,
).run(operator)
container = unitary.get_container()
# end-cell-sossa
################################################################################

################################################################################
# start-cell-unary-sossa
trial_state = create("state_prep", "sparse_isometry").run(reference_wavefunction)
unitary_builder = AlgorithmRef(
    "hamiltonian_unitary_builder",
    "sossa",
    reference_ground_state_energy=reference_energy,
)
epsilon_pea = 1e-3  # Phase-estimation energy target in Hartree (1 mEh).
num_queries = ceil(pi * container.lambda_eff / (2 * epsilon_pea))

qpe_builder = create(
    "qpe_circuit_builder",
    "qdk_unary",
    num_queries=num_queries,
    compute_capacity=-1,
    unitary_builder=unitary_builder,
    circuit_mapper=AlgorithmRef(
        "circuit_mapper",
        "sossa",
        outer_prepare_algorithm=AlgorithmRef("state_prep", "dense_pure_state"),
        inner_prepare_algorithm="direct",
        select_algorithm="direct",
        coefficient_bit_precision=8,
        rotation_bit_precision=8,
    ),
)
circuit = qpe_builder.run(trial_state, operator)[0]
print(f"Phase-estimation target: {epsilon_pea} Hartree; queries: {num_queries}")
logical_counts = circuit.estimate().logical_counts
print(f"Logical counts: {logical_counts}")
# end-cell-unary-sossa
################################################################################
