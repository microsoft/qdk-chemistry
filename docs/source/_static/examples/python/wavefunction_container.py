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
   OrbitalType
)

################################################################################
# start-cell-create-slater
# Create a simple Slater determinant wavefunction

# Create basis set
shells = []
atom_index = 0
functions_created = 0

# Create shells to reach 3 AOs
while functions_created < 3:
    remaining = 3 - functions_created

    if remaining >= 3:
        # Add a P shell (3 functions: Px, Py, Pz)
        exps = np.array([1.0, 0.5])
        coefs = np.array([0.6, 0.4])
        shell = Shell(atom_index, OrbitalType.P, exps, coefs)
        shells.append(shell)
        functions_created += 3
    elif remaining >= 1:
        # Add S shells for remaining functions (1 function each)
        for _ in range(remaining):
            exps = np.array([1.0])
            coefs = np.array([1.0])
            shell = Shell(atom_index, OrbitalType.S, exps, coefs)
            shells.append(shell)
            functions_created += 1
basis_set = BasisSet('dummy', shells)

# Create orbitals
coefficients = np.eye(3)  # Identity matrix for 3 orbitals
energies = np.array([-1.0, -0.5, 0.2])  # Example orbital energies
orbitals = Orbitals(coefficients, energies, None, basis_set)

# Single determinant with 2 electrons in 3 orbitals (active space size 2)
det = Configuration("20")  # First orbital doubly occupied, others empty
sd_container = SlaterDeterminantContainer(det, orbitals)
sd_wavefunction = Wavefunction(sd_container)
# end-cell-create-slater
################################################################################

################################################################################
# start-cell-create-cas
# Create a CAS wavefunction with multiple determinants
# Same 3 orbitals, active space of 2 orbitals with 2 electrons
cas_dets = [Configuration("20"),    # |2,0⟩ 
            Configuration("ud")]    # |1,1⟩
cas_coeffs = np.array([0.9, 0.436])
cas_container = CasWavefunctionContainer(cas_coeffs, cas_dets, orbitals)
cas_wavefunction = Wavefunction(cas_container)
# end-cell-create-cas
################################################################################

################################################################################
# start-cell-create-sci
# Create an SCI wavefunction with selected determinants
# Same 3 orbitals, selected configurations for 2 electrons in 2 active orbitals
sci_dets = [Configuration("20"),    # |2,0⟩
            Configuration("ud"),    # |1,1⟩ 
            Configuration("02")]    # |0,2⟩
sci_coeffs = np.array([0.85, 0.4, 0.3])
sci_container = SciWavefunctionContainer(sci_coeffs, sci_dets, orbitals)
sci_wavefunction = Wavefunction(sci_container)
# end-cell-create-sci
################################################################################

################################################################################
# start-cell-access-data
# Access basic wavefunction data
coeffs = sd_wavefunction.get_coefficients()
dets = sd_wavefunction.get_active_determinants()

# Get orbital information
orbitals_ref = sd_wavefunction.get_orbitals()

# Get electron counts
n_alpha, n_beta = sd_wavefunction.get_total_num_electrons()

# Check availability of reduced density matrices
has_1rdm_spin_dep = sd_wavefunction.has_one_rdm_spin_dependent()
has_1rdm_spin_traced = sd_wavefunction.has_one_rdm_spin_traced()
has_2rdm_spin_dep = sd_wavefunction.has_two_rdm_spin_dependent()
has_2rdm_spin_traced = sd_wavefunction.has_two_rdm_spin_traced()

# Access reduced density matrices if available
if has_1rdm_spin_dep:
    rdm1_aa, rdm1_bb = sd_wavefunction.get_active_one_rdm_spin_dependent()

if has_1rdm_spin_traced:
    rdm1_total = sd_wavefunction.get_active_one_rdm_spin_traced()

if has_2rdm_spin_dep:
    rdm2_aaaa, rdm2_aabb, rdm2_bbbb = sd_wavefunction.get_active_two_rdm_spin_dependent()

if has_2rdm_spin_traced:
    rdm2_total = sd_wavefunction.get_active_two_rdm_spin_traced()

# Get single orbital entropies
entropies = sd_wavefunction.get_single_orbital_entropies()
# end-cell-access-data
################################################################################
