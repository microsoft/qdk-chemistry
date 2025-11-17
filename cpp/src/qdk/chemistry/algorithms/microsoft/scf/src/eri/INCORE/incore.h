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

// Forward declarations of implementation classes
namespace incore {
class ERI;
class ERI_DF;
}  // namespace incore

/**
 * @brief In-core conventional 4-center electron repulsion integral (ERI)
 * calculator
 *
 * Implements conventional 4-index ERI computation with full storage of all
 * (μν|λσ) integrals in memory for subsequent fast Fock matrix construction.
 *
 * Memory scaling: O(N⁴) where N = number of basis functions
 * Computational scaling: O(N⁴) for integral computation
 *
 * @note Only practical for small to medium-sized systems (< 500 basis
 * functions)
 */
class ERIINCORE : public ERI {
 public:
  /**
   * @brief Construct in-core conventional ERI calculator
   *
   * Initializes the ERI calculator and computes/stores all (μν|λσ) integrals
   * in memory. Integral computation uses Schwarz screening to skip negligible
   * shell quartets based on the tolerance threshold.
   *
   * For range-separated functionals (ω ≠ 0), additionally computes (and stores)
   * integrals of the form:
   *   (μν|erf(ωr₁₂)/r₁₂|λσ) or (μν|erfc(ωr₁₂)/r₁₂|λσ)
   *
   * @param unrestricted Whether this is an unrestricted (UHF/UKS) calculation
   * @param basis_set Atomic orbital basis set for integral evaluation
   * @param mpi MPI parallelization configuration
   * @param omega Range-separation parameter (0.0 for standard Coulomb operator)
   *
   * @throws std::bad_alloc if insufficient memory to store all integrals
   * @throws std::runtime_error if integral computation fails
   */
  ERIINCORE(bool unrestricted, BasisSet& basis_set, ParallelConfig mpi,
            double omega);

  /**
   * @brief Destructor - releases all stored integral data
   *
   * Frees memory used for storing the 4-center integral tensor. Marked noexcept
   * to ensure safe cleanup during exception unwinding.
   */
  ~ERIINCORE() noexcept;

 private:
  /**
   * @brief Build Coulomb (J) and exchange (K) matrices from stored integrals
   * @see ERI::build_JK for API details
   */
  void build_JK_impl_(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) override;

  /**
   * @brief Compute nuclear gradients of density-fitted J and K matrices
   * @see ERI::get_gradients for API details
   * @note This implementation *does not* support exchange (K) matrix
   * construction. Function parameters are kept for interface compatibility.
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) override;

  /**
   * @brief Perform first quarter transformation to MO basis
   * @see ERI::quarter_trans for API details
   */
  void quarter_trans_impl(size_t nt, const double* C, double* out) override;

  /// PIMPL pointer to implementation
  std::unique_ptr<incore::ERI> eri_impl_;
};

/**
 * @brief In-core density-fitted 3-center ERI calculator with auxiliary basis
 *
 * Implements density fitting (DF) or resolution of identity (RI) approximation
 * for electron repulsion integrals. Stores 3-index integrals (Q|μν) and the
 * inverse metric (Q|P)⁻¹ in memory for efficient approximate Fock matrix
 * builds.
 *
 * The DF approximation factorizes 4-center integrals as:
 *   (μν|λσ) ≈ Σ_QP (μν|Q)(Q|P)⁻¹(P|λσ)
 *
 * Features:
 * - In-memory storage of 3-center integrals (Q|μν)
 * - Pre-computed and stored Coulomb metric (Q|P)⁻¹
 * - Fast approximate Coulomb matrix construction (DF-J)
 * - Analytical nuclear gradients
 * - Quarter transformation for MO-basis integrals
 *
 * Memory scaling: O(N²M) where N = orbital basis size, M = auxiliary basis size
 * Computational scaling: O(N²M) for Fock matrix construction
 */
class ERIINCORE_DF : public ERI {
 public:
  /**
   * @brief Construct in-core density-fitted ERI calculator
   *
   * Initializes the DF-ERI calculator by:
   * 1. Computing all 3-center integrals (Q|μν) with orbital and auxiliary bases
   * 2. Computing the 2-center Coulomb metric (Q|P) from auxiliary basis
   * 3. Inverting the metric to obtain (Q|P)⁻¹ via Cholesky decomposition
   * 4. Storing both (Q|μν) and (Q|P)⁻¹ in memory for subsequent Fock builds
   *
   * @param unrestricted Whether this is an unrestricted (UHF/UKS) calculation
   * @param orbital_basis_set Primary basis set for molecular orbitals
   * @param aux_basis_set Auxiliary basis set for density fitting
   * @param mpi MPI parallelization configuration
   *
   * @throws std::bad_alloc if insufficient memory for 3-center integral storage
   * @throws std::runtime_error if metric matrix is singular or ill-conditioned
   */
  ERIINCORE_DF(bool unrestricted, BasisSet& orbital_basis_set,
               BasisSet& aux_basis_set, ParallelConfig mpi);

  /**
   * @brief Destructor - releases stored integral data and metric
   */
  ~ERIINCORE_DF() noexcept;

 private:
  /**
   * @brief Build approximate Coulomb (J) matrix using Density Fitting
   *
   * Constructs the Coulomb matrix efficiently using the density-fitted
   * approximation: J[μν] ≈ Σ_QP (μν|Q) (Q|P)⁻¹ (P|λσ) P[λσ]
   * @see ERI::build_JK for API details
   * @note This implementation *does not* support exchange (K) matrix
   * construction. Function parameters are kept for interface compatibility.
   */
  void build_JK_impl_(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) override;

  /**
   * @brief Compute nuclear gradients of density-fitted J and K matrices
   * @see ERI::get_gradients for API details
   * @note This implementation *does not* support exchange (K) matrix
   * construction. Function parameters are kept for interface compatibility.
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) override;

  /**
   * @brief Perform first quarter transformation to MO basis
   * @see ERI::quarter_trans for API details
   * @note This instance of ERI does not implement this function. An exception
   * will be thrown.
   */
  void quarter_trans_impl(size_t nt, const double* C, double* out) override;

  /// PIMPL pointer to DF implementation
  std::unique_ptr<incore::ERI_DF> eri_impl_;
};

}  // namespace qdk::chemistry::scf
