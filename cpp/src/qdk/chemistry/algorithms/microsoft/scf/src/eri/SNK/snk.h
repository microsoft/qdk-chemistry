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

// Forward declaration of implementation class
namespace snk {
class ERI;
}

/**
 * @brief Semi-numerical exchange (sn-K) electron repulsion integral calculator
 *
 * Implements an efficient approximation to the exact exchange matrix for
 * Hartree-Fock and hybrid density functional theory calculations, as
 * implemented in GauXC.
 *
 * See:
 *   J. Chem. Theory Comput. 2020, 16, 3, 1456–1468 for algorithmic details
 *   J. Chem. Phys. 158, 234104 (2023) for implementation details in GauXC
 *
 * @note J matrix must be provided by another method (conventional or DF)
 */
class SNK : public ERI {
 public:
  /**
   * @brief Construct semi-numerical exchange calculator using GauXC
   * @param unrestricted Whether this is an unrestricted (UHF/UKS) spin
   * calculation
   * @param basis_set Atomic orbital basis set for molecular orbitals
   * @param gauxc_input GauXC configuration specifying grid quality,
   * partitioning, etc.
   * @param xc_name Exchange-correlation functional name (used for consistency
   * checking)
   * @param mpi MPI parallelization configuration (distribution scheme)
   *
   * @throws std::runtime_error if GauXC initialization fails
   */
  SNK(bool unrestricted, BasisSet& basis_set, GAUXCInput gauxc_input,
      std::string xc_name, ParallelConfig mpi);

  /**
   * @brief Destructor
   */
  ~SNK() noexcept;

  /**
   * @brief Build the J and K matrices
   * @see ERI::build_JK for API details
   * @throws std::runtime_error if J is requested (non-null pointer)
   * @throws std::runtime_error if omega ≠ 0 (range-separated not implemented)
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) override;

 private:
  /**
   * @brief Build semi-numerical exchange (K) matrix via numerical quadrature
   * @see ERI::build_JK for API details
   */
  void build_JK_impl_(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) override;

  /**
   * @brief Public wrapper for get_gradients (NOT YET IMPLEMENTED)
   * @throws std::runtime_error if invoked (not implemented)
   * @see ERI::get_gradients for API details
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) override;

  /**
   * @brief Quarter transformation to MO basis (NOT SUPPORTED)
   * @throws std::runtime_error if invoked (not implemented)
   * @see ERI::quarter_trans for API details
   */
  void quarter_trans_impl(size_t nt, const double* C, double* out) override;

  /// PIMPL pointer to implementation
  std::unique_ptr<snk::ERI> eri_impl_;
};

}  // namespace qdk::chemistry::scf
