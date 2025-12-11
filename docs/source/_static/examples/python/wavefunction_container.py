"""Wavefunction container examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import (
    Orbitals,
    SlaterDeterminantContainer,
    CasWavefunctionContainer,
    SciWavefunctionContainer,
    CoupledClusterContainer,
    MP2Container,
    Configuration,
    Wavefunction,
    Hamiltonian,
    BasisSet,
    Shell,
    OrbitalType,
)


def make_minimal_orbitals():
    """Helper function to create orbitals.

    These are used in the different constructors for wfn containers."""

    # Create STO-1G basis set for H2
    shells = [
        Shell(0, OrbitalType.S, exponents=[1.0], coefficients=[1.0]),
        Shell(1, OrbitalType.S, exponents=[1.0], coefficients=[1.0]),
    ]
    basis_set = BasisSet("STO-1G_H2", shells=shells)

    # Create bonding and antibonding MOs from AOs
    coefficients = np.array(([0.7071, 0.7071], [0.7071, -0.7071]))
    energies = [-1.0, 0.5]

    # Orbital constructor requires coefficients, energies, optionally AO overlap
    # matrix, and basis set
    orbitals = Orbitals(coefficients, energies, None, basis_set)
    return orbitals


def make_minimal_hamiltonian(orbitals):
    """Helper function to create a minimal Hamiltonian.

    This is needed for MP2 container examples."""

    # Create minimal one- and two-electron integrals for H2
    h_core = np.array([[-1.5, -0.8], [-0.8, 0.5]])  # Core Hamiltonian

    # Two-electron integrals in MO basis, stored as flattened vector
    # These are stored like i*norb*norb*norb + j*norb*norb + k*norb + l
    # In other words, if we want to access an integral element in the vector,
    # (ij|kl), we can access using this indexing.

    # For H2: norb=2, so we need 2^4=16 elements
    eri = np.zeros(16)

    # Set some representative values for H2 two-electron integrals
    # Format: (ij|kl) in physicist notation
    eri[0] = 1.0  # (00|00) - index 0*8 + 0*4 + 0*2 + 0 = 0
    eri[5] = 0.6  # (01|01) - index 0*8 + 1*4 + 0*2 + 1 = 5
    eri[10] = 0.6  # (10|10) - index 1*8 + 0*4 + 1*2 + 0 = 10
    eri[15] = 0.8  # (11|11) - index 1*8 + 1*4 + 1*2 + 1 = 15
    eri[3] = 0.4  # (00|11) - index 0*8 + 0*4 + 1*2 + 1 = 3
    eri[12] = 0.4  # (11|00) - index 1*8 + 1*4 + 0*2 + 0 = 12

    # Core energy (nuclear repulsion + core electron contributions)
    core_energy = 0.0

    # Inactive Fock matrix (empty for minimal example)
    inactive_fock = np.zeros((2, 2))

    # Create Hamiltonian
    hamiltonian = Hamiltonian(h_core, eri, orbitals, core_energy, inactive_fock)
    return hamiltonian


################################################################################
# start-cell-create-slater
# Use helper function to get orbitals
orbitals = make_minimal_orbitals()

# Create a simple Slater determinant wavefunction for H2 ground state
# 2 electrons in bonding sigma orbital
det = Configuration("20")

# Constructor takes single determinant and orbitals as input
sd_container = SlaterDeterminantContainer(det, orbitals)
sd_wavefunction = Wavefunction(sd_container)
# end-cell-create-slater
################################################################################

################################################################################
# start-cell-create-cas
# Create a CAS wavefunction for H2
# CAS(2,2) = 2 electrons in 2 MOs (bonding and antibonding)
# All possible configurations:
cas_dets = [
    Configuration("20"),  # both electrons in bonding MO (ground state)
    Configuration("ud"),  # alpha in bonding, beta in antibonding
    Configuration("du"),  # beta in bonding, alpha in antibonding
    Configuration("02"),  # both electrons in antibonding
]

# Coefficients (normalized later by container)
cas_coeffs = np.array([0.95, 0.15, 0.15, 0.05])

# Create a CAS wavefunction: requires all coefficients and determinants,
# as well as orbitals, in constructor
cas_container = CasWavefunctionContainer(cas_coeffs, cas_dets, orbitals)
cas_wavefunction = Wavefunction(cas_container)
# end-cell-create-cas
################################################################################

################################################################################
# start-cell-create-sci
# Create an SCI wavefunction for H2
# SCI selects only the most important configurations/determinants from the full space
sci_dets = [
    Configuration("20"),  # both electrons in bonding MO (ground state)
    Configuration("du"),  # alpha in bonding, beta in antibonding
    Configuration("ud"),  # beta in bonding, alpha in antibonding
]

# Coefficients for selected determinants
sci_coeffs = np.array([0.96, 0.15, 0.15])

# Create a SCI wavefunction: requires selected coefficients and determinants, as well
# as orbitals, in constructor
sci_container = SciWavefunctionContainer(sci_coeffs, sci_dets, orbitals)
sci_wavefunction = Wavefunction(sci_container)
# end-cell-create-sci
################################################################################

################################################################################
# start-cell-create-mp2
# Create an MP2 wavefunction for H2
# MP2 uses a reference wavefunction and Hamiltonian to compute amplitudes on demand

# Use the Slater determinant as reference
orbitals = make_minimal_orbitals()
hamiltonian = make_minimal_hamiltonian(orbitals)
ref_det = Configuration("20")
sd_container = SlaterDeterminantContainer(ref_det, orbitals)
ref_wavefunction = Wavefunction(sd_container)

# Create MP2 container: requires Hamiltonian and reference wavefunction
# Amplitudes are computed lazily when first requested
mp2_container = MP2Container(hamiltonian, ref_wavefunction, "mp")
mp2_wavefunction = Wavefunction(mp2_container)
# end-cell-create-mp2
################################################################################

################################################################################
# start-cell-create-cc
# Create a coupled cluster wavefunction for H2
# CC uses a reference wavefunction and pre-computed amplitudes

# Use the Slater determinant as reference
orbitals = make_minimal_orbitals()
ref_det = Configuration("20")
sd_container = SlaterDeterminantContainer(ref_det, orbitals)
ref_wavefunction = Wavefunction(sd_container)

# Create example T1 and T2 amplitudes
# T1: occupied-virtual excitations (1 occ × 1 virt = 1 element for H2)
t1_amplitudes = np.array([0.05])

# T2: occupied-occupied to virtual-virtual excitations
# (1 occ pair × 1 virt pair = 1 element for H2)
t2_amplitudes = np.array([0.15])

# Create CC container: requires reference wavefunction, orbitals, and amplitudes
cc_container = CoupledClusterContainer(
    orbitals, ref_wavefunction, t1_amplitudes, t2_amplitudes
)
cc_wavefunction = Wavefunction(cc_container)
# end-cell-create-cc
################################################################################

################################################################################
# start-cell-access-data
# Access coefficient(s) and determinant(s) - SD has only one
coeffs = sd_wavefunction.get_coefficients()
dets = sd_wavefunction.get_active_determinants()

# Get orbital information
orbitals_ref = sd_wavefunction.get_orbitals()

# Get electron counts
n_alpha, n_beta = sd_wavefunction.get_total_num_electrons()

# Get RDMs
rdm1_aa, rdm1_bb = sd_wavefunction.get_active_one_rdm_spin_dependent()
rdm1_total = sd_wavefunction.get_active_one_rdm_spin_traced()
rdm2_aaaa, rdm2_aabb, rdm2_bbbb = sd_wavefunction.get_active_two_rdm_spin_dependent()
rdm2_total = sd_wavefunction.get_active_two_rdm_spin_traced()

# Get single orbital entropies
entropies = sd_wavefunction.get_single_orbital_entropies()
# end-cell-access-data
################################################################################

################################################################################
# start-cell-access-amplitudes
# Access T1 and T2 amplitudes from MP2 and CC containers

# For MP2 - amplitudes computed on demand
t2_abab_mp2, t2_aaaa_mp2, t2_bbbb_mp2 = (
    mp2_wavefunction.get_container().get_t2_amplitudes()
)

# For CC - amplitudes stored during construction
t1_aa, t1_bb = cc_wavefunction.get_container().get_t1_amplitudes()
t2_abab_cc, t2_aaaa_cc, t2_bbbb_cc = cc_wavefunction.get_container().get_t2_amplitudes()
# end-cell-access-amplitudes
################################################################################
