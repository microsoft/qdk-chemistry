// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/eri.h>

#include <cstring>
#include <memory>
#include <vector>

namespace qdk::chemistry::scf {

// Forward declaration of implementation classes
namespace libint2::direct {
class ERI;
}

/**
 * @brief Direct SCF electron repulsion integral calculator using Libint2
 *
 * Implements the direct SCF algorithm where 4-center electron repulsion
 * integrals (μν|λσ) are computed on-the-fly during Fock matrix construction
 * rather than being pre-computed and stored in memory. This approach is
 * essential for large basis sets where the O(N⁴) memory requirement for storing
 * all integrals would be prohibitive.
 *
 * Range-separated functionals:
 * - Supports standard Coulomb operator: 1/r₁₂
 * - Supports attenuated operators: erf(ωr₁₂)/r₁₂ for hybrid functionals
 * - Both evaluated on-the-fly without additional storage
 */
class LIBINT2_DIRECT : public ERI {
 public:
  /**
   * @brief Construct direct SCF ERI calculator using Libint2
   *
   * @param unrestricted Whether this is an unrestricted (UHF/UKS) spin
   * calculation
   * @param basis_set Atomic orbital basis set for molecular orbitals
   * @param mpi MPI parallelization configuration
   *
   * @throws std::runtime_error if Libint2 initialization fails
   * @throws std::runtime_error if screening matrix computation fails
   */
  LIBINT2_DIRECT(bool unrestricted, BasisSet& basis_set, ParallelConfig mpi);

  /**
   * @brief Destructor
   */
  ~LIBINT2_DIRECT() noexcept;

  /**
   * @brief Public interface for building J and K matrices
   * @see ERI::build_JK for API details
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) override;

 private:
  /**
   * @brief Build Coulomb (J) and exchange (K) matrices via direct integral
   * evaluation
   * @see ERI::build_JK for API details
   */
  void build_JK_impl_(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) override;

  /**
   * @brief Public wrapper for get_gradients with direct-specific optimizations
   * @see ERI::get_gradients for API details
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) override;

  /**
   * @brief Perform first quarter transformation to MO basis
   * @see ERI::quarter_trans for API details
   */
  void quarter_trans_impl(size_t nt, const double* C, double* out) override;

  /// PIMPL pointer to implementation
  std::unique_ptr<libint2::direct::ERI> eri_impl_;
};
}  // namespace qdk::chemistry::scf
