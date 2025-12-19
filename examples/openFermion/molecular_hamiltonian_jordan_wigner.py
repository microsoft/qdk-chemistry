# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""qdk-chemistry + PennyLane quantum phase estimation example.

This example demonstrates the use of PennyLane to implement traditional QFT-based Quantum Phase Estimation (QPE)
using QDK/Chemistry tools for preparing the electronic structure problem.
This example does not use Trotterization; instead, it leverages PennyLane's ability to implement the time-evolution operator `exp(−i*coeff*H)` exactly for a given qubit Hamiltonian.
"""

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

import numpy
import scipy
import scipy.linalg

from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.utils import Logger

try:
    import openfermion
    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from openfermion.linalg import get_ground_state, get_sparse_operator

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
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.45 * ANGSTROM_TO_BOHR]], dtype=float), ["Li", "H"]
)  # Geometry in bohr

scf_solver = create("scf_solver", basis_set="sto-3g")
scf_energy, scf_wavefunction = scf_solver.run(
        structure, charge=0, spin_multiplicity=1
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

n_alpha = n_beta = ACTIVE_ELECTRONS // 2
mc_calculator = create("multi_configuration_calculator")
casci_energy, casci_wavefunction = mc_calculator.run(
    active_hamiltonian, n_alpha, n_beta
)

Logger.info("=== Generating QDK/Chemistry artifacts for LiH (1.45 Å, STO-3G) ===")
Logger.info(f"  SCF total energy:   {scf_energy: .8f} Hartree")
Logger.info(f"  CASCI total energy: {casci_energy: .8f} Hartree")


########################################################################################
# 3. Preparing the qubit Hamiltonian and sparse-isometry trial state
########################################################################################

one_body_aa, one_body_bb = np.array(
    active_hamiltonian.get_one_body_integrals(), dtype=float
)  # One-electron integrals

norb = one_body_aa.shape[0]  # Number of spatial orbitals

# Obtain a rank-4 tensor in chemists' notation (pq|rs) from QDK
(two_body_integrals, _, _) = active_hamiltonian.get_two_body_integrals()
two_body_flat = np.array(two_body_integrals, dtype=float)  # Two-electron integrals
two_body = two_body_flat.reshape((norb,) * 4)  

# Convert to open Fermion physicists' notation <pr|sq>. Note that the last two indices may be switched
# from what you expect in other physicists' notation. OpenFermion takes the integral notaion below to be consistent
# with the order of operators.
# ĝ = ½ Σ (pq|rs) p† r† s q = ½ Σ ⟨pr|sq⟩ p† r† s q
two_body_phys = np.transpose(two_body, (0, 2, 3, 1)) 


# make spacial integrals into spin orbitals, ordered as alpha_1, beta_1, alpha_2, beta_2, ...
n_spin_orbitals = 2 * norb
one_body_coefficients = np.zeros((n_spin_orbitals, n_spin_orbitals))
two_body_coefficients = np.zeros((n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals))

# For those less familiar with vectorized notation:
# one_body_coefficients[2 * p, 2 * q] = one_body_aa[p, q]
# one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_bb[p, q]
# two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = two_body_phys[ p, q, r, s ]
# two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = two_body_phys[p, q, r, s]
# two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = two_body_phys[ p, q, r, s ]
# two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = two_body_phys[  p, q, r, s ]
one_body_coefficients[0::2, 0::2] = one_body_aa
one_body_coefficients[1::2, 1::2] = one_body_bb
two_body_coefficients[0::2, 0::2, 0::2, 0::2] = two_body_phys
two_body_coefficients[1::2, 1::2, 1::2, 1::2] = two_body_phys
two_body_coefficients[0::2, 1::2, 1::2, 0::2] = two_body_phys # note order of last two indices again
two_body_coefficients[1::2, 0::2, 0::2, 1::2] = two_body_phys

core_energy = active_hamiltonian.get_core_energy()  # Core energy constant

# Get the Hamiltonian in an active space.
openFermion_molecular_hamiltonian = openfermion.ops.representations.InteractionOperator(
    core_energy, one_body_coefficients, 1/2 * two_body_coefficients
)

# Map operator to fermions and qubits.
fermion_hamiltonian = get_fermion_operator(openFermion_molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
qubit_hamiltonian.compress()
Logger.info("=== The Jordan-Wigner Hamiltonian in canonical basis : ===")
message = str(qubit_hamiltonian)
Logger.info(message)

# Get sparse operator and ground state energy.
sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
energy, state = get_ground_state(sparse_hamiltonian)
Logger.info(f"Ground state energy before rotation is {energy: .15f} Hartree.")

# Randomly rotate.
n_orbitals = openFermion_molecular_hamiltonian.n_qubits // 2
n_variables = int(n_orbitals * (n_orbitals - 1) / 2)
numpy.random.seed(1)
random_angles = numpy.pi * (1. - 2. * numpy.random.rand(n_variables))
kappa = numpy.zeros((n_orbitals, n_orbitals))
index = 0
for p in range(n_orbitals):
    for q in range(p + 1, n_orbitals):
        kappa[p, q] = random_angles[index]
        kappa[q, p] = -numpy.conjugate(random_angles[index])
        index += 1

    # Build the unitary rotation matrix.
    difference_matrix = kappa + kappa.transpose()
    rotation_matrix = scipy.linalg.expm(kappa)

    # Apply the unitary.
    openFermion_molecular_hamiltonian.rotate_basis(rotation_matrix)

# Get qubit Hamiltonian in rotated basis.
qubit_hamiltonian = jordan_wigner(openFermion_molecular_hamiltonian)
qubit_hamiltonian.compress()

Logger.info("=== The Jordan-Wigner Hamiltonian in rotated basis : ===")
message = str(qubit_hamiltonian)
Logger.info(message)

# Get sparse Hamiltonian and energy in rotated basis.
sparse_hamiltonian = get_sparse_operator(qubit_hamiltonian)
energy, state = get_ground_state(sparse_hamiltonian)
Logger.info(f"Ground state energy after rotation is {energy: .15f} Hartree.")
