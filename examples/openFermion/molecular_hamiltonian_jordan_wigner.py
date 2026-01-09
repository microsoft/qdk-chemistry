# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""qdk-chemistry molecular Hamiltonian + OpenFermion Jordan-Wigner transformation example.

This example demonstrates the use of QDK/Chemistry tools in preparing the electronic
structure Hamiltonian, which is then passed to OpenFermion to perform the Jordan-Wigner
transformation. The key points are (1) QDK/Chemistry provides integrals in the spatial molecular
orbital basis, which need to be converted to integrals in the spin-orbital basis that OpenFermion
expects, and (2) the two-electron molecular integrals from QDK/Chemistry are in chemist's notation
and need to be packed in the physicist's notation that OpenFermion expects.

This example is adapted from the introduction to OpenFermion tutorial:
https://quantumai.google/openfermion/tutorials/intro_to_openfermion
Due to minor differences in the Hamiltonian (stemming from different molecular orbitals), the
calculated energy values differ from those in the OpenFermion tutorial by O(1e-6) Hartree.
"""

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.utils import Logger

# OpenFermion must be installed to run this example.
try:
    import openfermion
except ImportError as e:
    raise ImportError(
        "OpenFermion is not installed. Please install OpenFermion to run this example: pip install openfermion"
    ) from e

Logger.set_global_level("info")

# Set Hamiltonian parameters.

ACTIVE_ELECTRONS = 2
ACTIVE_ORBITALS = 2

########################################################################################
# 1. QDK/Chemistry calculation for LiH (1.45 Å, STO-3G)
########################################################################################
structure = Structure(
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.45 * ANGSTROM_TO_BOHR]], dtype=float),
    ["Li", "H"],
)  # Geometry in bohr

scf_solver = create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)


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

Logger.info("=== Generating QDK/Chemistry artifacts for LiH (1.45 Å, STO-3G) ===")
Logger.info(f"  SCF total energy:   {scf_energy: .8f} Hartree")

########################################################################################
# 3. Preparing the qubit Hamiltonian for OpenFermion Jordan-Wigner transformation
########################################################################################

# For restricted Hartree-Fock, the alpha and beta blocks are equal.
one_body_aa, one_body_bb = active_hamiltonian.get_one_body_integrals()
one_body = np.array(
    one_body_aa, dtype=float
)  # One-electron integrals (use spin up block only)

norb = one_body.shape[0]  # Number of spatial orbitals

# Obtain a rank-4 tensor in chemist's notation (pq|rs) from QDK
two_body_aaaa, two_body_aabb, two_body_bbbb = (
    active_hamiltonian.get_two_body_integrals()
)
two_body = np.array(two_body_aaaa, dtype=float).reshape((norb,) * 4)

# Convert to OpenFermion physicist's notation <pr|sq>. Note that the last two indices may be switched
# from what you expect from other physicist's notation. OpenFermion takes the integral notation below to be consistent
# with the order of operators.
# ĝ = ½ Σ (pq|rs) p† r† s q = ½ Σ ⟨pr|sq⟩ p† r† s q
two_body_phys = np.transpose(two_body, (0, 2, 3, 1))

# Note: the spinorb_from_spatial function from OpenFermion works for restricted Hamiltonians only
# If unrestricted Hamiltonians are needed, write a custom function and pay special attention to the ordering of the
# two-electron integrals, especially in the mix-spin scenarios.
one_body_coefficients, two_body_coefficients = (
    openfermion.chem.molecular_data.spinorb_from_spatial(one_body, two_body_phys)
)

core_energy = active_hamiltonian.get_core_energy()  # Core energy constant

# Get the Hamiltonian in an active space.
open_fermion_molecular_hamiltonian = (
    openfermion.ops.representations.InteractionOperator(
        core_energy, one_body_coefficients, 0.5 * two_body_coefficients
    )
)

# Map operator to fermions and qubits.
fermion_hamiltonian = openfermion.transforms.get_fermion_operator(
    open_fermion_molecular_hamiltonian
)
qubit_hamiltonian = openfermion.transforms.jordan_wigner(fermion_hamiltonian)
qubit_hamiltonian.compress()
Logger.info(
    "=== The Jordan-Wigner Hamiltonian in canonical basis (interleaved ordering): ==="
)
message = str(qubit_hamiltonian)
Logger.info(message)

# Get sparse operator and ground state energy.
sparse_hamiltonian = openfermion.linalg.get_sparse_operator(qubit_hamiltonian)
energy, state = openfermion.linalg.get_ground_state(sparse_hamiltonian)
Logger.info(f"Ground state energy is {energy: .15f} Hartree.")
