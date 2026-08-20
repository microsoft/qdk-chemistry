"""Code snippets for v2.0.0 release notes.

Each cell is a runnable snippet included in the Sphinx release notes via
``literalinclude`` with ``start-after`` / ``end-before`` markers.  The file
is executed end-to-end by the ``test_docs_examples.py`` test harness, which
gates it to an installed ``2.0.x`` library (see the release-notes version pin
in that test).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import MajoranaMapping, Structure

# ---------------------------------------------------------------------------
# Shared pipeline: H2 → SCF → active space → active-space Hamiltonian.
# (used by the mapping / builder / grouper cells below)
# ---------------------------------------------------------------------------
_structure = Structure(
    np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=float), ["H", "H"]
)
_scf = create("scf_solver")
_E_scf, _wfn_scf = _scf.run(
    _structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
_selector = create(
    "active_space_selector",
    "qdk_valence",
    num_active_electrons=2,
    num_active_orbitals=2,
)
_active_orbitals = _selector.run(_wfn_scf).get_orbitals()
_hamiltonian = create("hamiltonian_constructor").run(_active_orbitals)
_n_spin_orbitals = 2 * _hamiltonian.get_orbitals().get_num_molecular_orbitals()

# Aliases for display in the cells below
structure = _structure
hamiltonian = _hamiltonian
n_spin_orbitals = _n_spin_orbitals

# ===========================================================================
# Fermion-to-qubit mapping as data (MajoranaMapping)
# ===========================================================================

################################################################################
# start-cell-majorana-mapping
mapping = MajoranaMapping.jordan_wigner(num_modes=n_spin_orbitals)
mapper = create("qubit_mapper", "qdk")
qubit_operator = mapper.run(hamiltonian, mapping)
# end-cell-majorana-mapping
################################################################################

print(f"Qubit operator: {qubit_operator.num_qubits} qubits")

# ===========================================================================
# Term grouping
# ===========================================================================

################################################################################
# start-cell-term-grouper
grouper = create("term_grouper", "qubit_wise_commuting")
grouped = grouper.run(qubit_operator)
# end-cell-term-grouper
################################################################################

# ===========================================================================
# Block encoding / LCU qubitization
# ===========================================================================

################################################################################
# start-cell-lcu
builder = create("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
walk = builder.run(qubit_operator)
# end-cell-lcu
################################################################################

# ===========================================================================
# Zassenhaus product formula
# ===========================================================================

################################################################################
# start-cell-zassenhaus
builder = create(
    "hamiltonian_unitary_builder", "zassenhaus", order=2, num_divisions=4, time=1.0
)
unitary = builder.run(qubit_operator)
# end-cell-zassenhaus
################################################################################

# ===========================================================================
# Nuclear gradients and Hessians
# ===========================================================================

################################################################################
# start-cell-nuclear-derivatives
# Analytic gradients
grad_calc = create("nuclear_derivative_calculator", "qdk")
energy, gradients, _hessian, wavefunction = grad_calc.run(
    structure, charge=0, spin_multiplicity=1, seed_or_basis="sto-3g"
)

# Numeric Hessian (central finite differences)
hess_calc = create(
    "nuclear_derivative_calculator", "qdk_finite_difference", compute_hessian=True
)
energy, gradients, hessian, wavefunction = hess_calc.run(
    structure, charge=0, spin_multiplicity=1, seed_or_basis="sto-3g"
)
# end-cell-nuclear-derivatives
################################################################################

print(f"Nuclear derivative energy: {energy:.10f} Hartree")

# ===========================================================================
# Stabilized SCF
# ===========================================================================

################################################################################
# start-cell-stabilized-scf
scf_solver = create("scf_solver", "qdk_stabilized")
energy, wavefunction = scf_solver.run(structure, 0, 1, "sto-3g")
# end-cell-stabilized-scf
################################################################################

print(f"Stabilized SCF energy: {energy:.10f} Hartree")
