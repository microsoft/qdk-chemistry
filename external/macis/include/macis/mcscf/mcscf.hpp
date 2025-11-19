/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>

namespace macis {

/**
 * @brief Configuration settings for MCSCF calculations
 *
 * This structure contains all the parameters needed to control the behavior
 * of Multi-Configuration Self-Consistent Field (MCSCF) calculations.
 */
struct MCSCFSettings {
  /** @brief Maximum number of macro iterations in the MCSCF procedure */
  size_t max_macro_iter = 100;

  /** @brief Maximum allowed orbital step size in the optimization */
  double max_orbital_step = 0.5;

  /** @brief Convergence tolerance for the orbital gradient in MCSCF */
  double orb_grad_tol_mcscf = 5e-6;

  /** @brief Enable DIIS acceleration for orbital optimization */
  bool enable_diis = true;

  /** @brief Iteration number at which to start DIIS extrapolation */
  size_t diis_start_iter = 3;

  /** @brief Number of vector pairs to keep in DIIS subspace */
  size_t diis_nkeep = 10;

  // size_t max_bfgs_iter      = 100;
  // double orb_grad_tol_bfgs  = 5e-7;

  /** @brief Convergence tolerance for the CI residual */
  double ci_res_tol = 1e-8;

  /** @brief Maximum size of the CI subspace */
  size_t ci_max_subspace = 200;

  /** @brief Tolerance for CI matrix elements */
  double ci_matel_tol = std::numeric_limits<double>::epsilon();
};

/**
 * @brief Perform Complete Active Space Self-Consistent Field (CASSCF)
 * calculation with DIIS acceleration
 *
 * This function implements the CASSCF method using DIIS (Direct Inversion in
 * the Iterative Subspace) acceleration for orbital optimization. It optimizes
 * both the molecular orbitals and the CI coefficients within the active space.
 *
 * @param settings MCSCF calculation settings and convergence parameters
 * @param nalpha Number of alpha electrons in the active space
 * @param nbeta Number of beta electrons in the active space
 * @param norb Total number of molecular orbitals
 * @param ninact Number of inactive (doubly occupied) orbitals
 * @param nact Number of active orbitals
 * @param nvirt Number of virtual (unoccupied) orbitals
 * @param E_core Nuclear repulsion and frozen core energy
 * @param T One-electron integrals matrix (size norb x norb)
 * @param LDT Leading dimension of the T matrix
 * @param V Two-electron integrals tensor (size norb^4)
 * @param LDV Leading dimension of the V tensor
 * @param A1RDM Active space one-particle reduced density matrix (size nact x
 * nact)
 * @param LDD1 Leading dimension of the A1RDM matrix
 * @param A2RDM Active space two-particle reduced density matrix (size nact^4)
 * @param LDD2 Leading dimension of the A2RDM tensor
 * @param comm MPI communicator for parallel calculations (if MPI is enabled)
 *
 * @return Converged CASSCF energy
 */
double casscf_diis(MCSCFSettings settings, NumElectron nalpha,
                   NumElectron nbeta, NumOrbital norb, NumInactive ninact,
                   NumActive nact, NumVirtual num_virtual_orbitals,
                   double E_core, double* T, size_t LDT, double* V, size_t LDV,
                   double* A1RDM, size_t LDD1, double* A2RDM,
                   size_t LDD2 MACIS_MPI_CODE(, MPI_Comm comm));

}  // namespace macis
