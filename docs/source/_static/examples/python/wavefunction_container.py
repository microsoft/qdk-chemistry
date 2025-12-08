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
    Configuration,
    Wavefunction,
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
