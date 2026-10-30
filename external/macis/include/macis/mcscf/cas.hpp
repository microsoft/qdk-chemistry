/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/mcscf/mcscf.hpp>
#include <macis/types.hpp>
#include <macis/util/mpi.hpp>
#include <utility>
#include <vector>

namespace macis {

/**
 *  @brief Compute the CAS-CI ground state and its spin-traced 1- and 2-RDMs
 *
 *  The active-space full CI problem is solved directly from the integrals
 *  (FCIHamiltonian / fci_ground_state, default search space).
 *  `settings.ci_max_subspace` bounds the number of Davidson iterations and
 *  `settings.ci_res_tol` is the residual tolerance.
 *
 *  @param[in] settings Settings for the CI calculation
 *  @param[in] norb     Number of orbitals
 *  @param[in] nalpha   Number of alpha electrons
 *  @param[in] nbeta    Number of beta electrons
 *  @param[in] T        The one-body Hamiltonian
 *  @param[in] V        The two-body Hamiltonian
 *  @param[out] ORDM    The CAS-CI 1-RDM (written only if ORDM and TRDM are
 *                      both non-null)
 *  @param[out] TRDM    The CAS-CI 2-RDM
 *  @param[in,out] C    The CAS-CI CI vector, alpha-major in the determinant
 *                      order of generate_hilbert_space; used as the initial
 *                      guess when it is sized and non-zero
 *  @param[in]  comm    MPI Communicator; every rank solves the problem and the
 *                      results of rank 0 are broadcast
 */
double compute_casci_rdms(MCSCFSettings settings, NumOrbital norb,
                          size_t nalpha, size_t nbeta, double* T, double* V,
                          double* ORDM, double* TRDM,
                          std::vector<double>& C MACIS_MPI_CODE(,
                                                                MPI_Comm comm));

/**
 * @brief Functor wrapper around compute_casci_rdms for use as the RDM operator
 * of MCSCF implementations
 */
struct CASRDMFunctor {
  /**
   * @brief Static function that forwards arguments to compute_casci_rdms
   *
   * @param args Variadic arguments forwarded to compute_casci_rdms
   * @return Result of compute_casci_rdms function call
   */
  template <typename... Args>
  static auto rdms(Args&&... args) {
    return compute_casci_rdms(std::forward<Args>(args)...);
  }
};

}  // namespace macis
