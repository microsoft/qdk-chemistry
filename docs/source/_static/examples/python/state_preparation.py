"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create a StatePreparation instance
sparse_prep = create("state_prep", "sparse_isometry")
dense_prep = create("state_prep", "dense_pure_state")
regular_prep = create("state_prep", "qiskit_regular_isometry")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Transpilation is configured on the algorithm that emits the Qiskit circuit.
regular_prep.settings().set("transpile", True)
regular_prep.settings().set("basis_gates", ["rz", "cz", "sdg", "h"])
regular_prep.settings().set("transpile_optimization_level", 3)

# Select other dense preparation method for sparse isometry
sparse_isometry_with_qiskit_dense_prep = create(
    "state_prep",
    "sparse_isometry",
    dense_state_prep=AlgorithmRef(
        "state_prep",
        "qiskit_regular_isometry",
        transpile=True,
        basis_gates=["rz", "cz", "sdg", "h"],
        transpile_optimization_level=3,
    ),
)
# end-cell-configure
################################################################################

################################################################################
# start-cell-configure-binary-encoding
# Compress the reduced subspace with binary encoding
binary_encoded_prep = create("state_prep", "sparse_isometry", binary_encoding=True)

# Select the algorithm that prepares the compressed dense register
binary_encoded_prep.settings().set(
    "dense_state_prep", AlgorithmRef("state_prep", "dense_pure_state")
)
# end-cell-configure-binary-encoding
################################################################################

################################################################################
# start-cell-configure-matrix-product-state
# MPS state preparation with block sparse method
mps_prep = create(
    "state_prep",
    "matrix_product_state",
    unitary_synthesis=AlgorithmRef("circuit_mapper", "unitary_synthesis_block_sparse"),
)
# end-cell-configure-matrix-product-state
################################################################################

################################################################################
# start-cell-spin-adapted-matrix-product-state
import numpy as np
from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    MPSContainer,
    MPSSite,
    Wavefunction,
)
from qdk_chemistry.data import symmetry as sym

# Sector labels (N, 2S) on a particle-number and a total-spin axis
symmetries = sym.SymmetryProduct(
    [
        sym.axes.particle_number(2),
        sym.SymmetryAxis(
            sym.AxisName.TotalSpin,
            [sym.TotalSpinValue(0), sym.TotalSpinValue(1)],
            False,
        ),
    ]
)
empty, doublet, pair = (
    sym.SymmetryLabel([sym.axes.particle_number_value(n), sym.TotalSpinValue(two_s)])
    for n, two_s in ((0, 0), (1, 1), (2, 0))
)
physical = {empty: 1, doublet: 1, pair: 1}
reduced_basis = [
    Configuration.from_spin_half_string(state) for state in ("0", "u", "2")
]

# Two-orbital singlet: the first site is left-isometric, the last holds the normalized amplitudes
first = sym.SymmetryBlockedTensorRank3(
    [symmetries] * 3,
    [{empty: 1}, physical, dict(physical)],
    [((empty, label, label), np.ones((1, 1))) for label in (empty, doublet, pair)],
)
amplitudes = {empty: 0.1, doublet: 0.2, pair: -0.97}
norm = np.linalg.norm(list(amplitudes.values()))
last = sym.SymmetryBlockedTensorRank3(
    [symmetries] * 3,
    [dict(physical), physical, {pair: 1}],
    [
        ((empty, pair, pair), np.full((1, 1), amplitudes[empty] / norm)),
        ((doublet, doublet, pair), np.full((1, 1), amplitudes[doublet] / norm)),
        ((pair, empty, pair), np.full((1, 1), amplitudes[pair] / norm)),
    ],
)
sites = [
    MPSSite(
        first, [empty], [empty, doublet, pair], [empty, doublet, pair], reduced_basis
    ),
    MPSSite(
        last, [empty, doublet, pair], [empty, doublet, pair], [pair], reduced_basis
    ),
]
# Left-canonical, with the orthogonality center on the last site
mps = MPSContainer(sites, ModelOrbitals(2), None, 1)

# Spin-adapted MPS are loaded with Householder site synthesis
spin_adapted_prep = create(
    "state_prep",
    "matrix_product_state",
    unitary_synthesis=AlgorithmRef(
        "circuit_mapper", "unitary_synthesis_householder", max_swap_address_bits=0
    ),
)
spin_adapted_circuit = spin_adapted_prep.run(Wavefunction(mps))
# end-cell-spin-adapted-matrix-product-state
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.data import Structure

# Specify a structure
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run scf
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Compute the Hamiltonian
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# Compute CAS wavefunction
cas_solver = create("multi_configuration_calculator", "macis_cas")
E_cas, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# Construct the circuit
regular_circuit = regular_prep.run(wfn_cas)
sparse_circuit = sparse_prep.run(wfn_cas)
dense_circuit = dense_prep.run(wfn_cas)
print(f"Regular isometry circuit:\n{regular_circuit.get_qiskit_circuit()}")
print(f"Sparse isometry circuit:\n{sparse_circuit.get_qsharp_circuit()}")
print(f"Dense pure state circuit:\n{dense_circuit.get_qsharp_circuit()}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("state_prep"))
# ['dense_pure_state', 'sparse_isometry', 'matrix_product_state', 'alias_sampling', 'qrom', 'qiskit_regular_isometry']
# end-cell-list-implementations
################################################################################
