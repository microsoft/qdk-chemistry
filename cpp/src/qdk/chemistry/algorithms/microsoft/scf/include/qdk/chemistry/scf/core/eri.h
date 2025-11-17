// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/enums.h>
#include <qdk/chemistry/scf/core/scf.h>

namespace qdk::chemistry::scf {
class ERIMultiplexer;  // Forward declaration

/**
 * @brief Base class for Electron Repulsion Integral (ERI) engines
 *
 * Provides interface for computing J (Coulomb) and K (exchange) matrices,
 * their energtic gradients, and molecular orbital transformations.
 */
class ERI {
  friend ERIMultiplexer;

 public:
  /**
   * @brief Construct ERI engine
   *
   * @param unrestricted Whether this is an unrestricted calculation (UHF/UKS)
   * @param tol Integral screening tolerance
   * @param basis_set The atomic orbital basis set
   * @param mpi Parallelism configuration
   */
  ERI(bool unrestricted, double tol, BasisSet& basis_set, ParallelConfig mpi)
      : unrestricted_(unrestricted),
        tolerance_(tol),
        basis_set_(basis_set),
        mpi_(mpi) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~ERI() = default;

  /**
   * @brief Build Coulomb (J) and exchange (K) matrices
   *
   * Computes J[μν] = Σ_λσ P[λσ](μν|λσ) and K[μν] = Σ_λσ P[λσ](μλ|νσ)
   *
   * Accepts null outputs, in which case the calculation of that term is
   * skipped.
   *
   * @param P Density matrix (input, size: num_density_matrices ×
   * num_atomic_orbitals × num_atomic_orbitals)
   * @param J Output Coulomb matrix (size: num_density_matrices ×
   * num_atomic_orbitals × num_atomic_orbitals)
   * @param K Output exchange matrix (size: num_density_matrices ×
   * num_atomic_orbitals × num_atomic_orbitals)
   * @param alpha Scaling coefficient for K (un-attenuated)
   * @param beta Scaling coefficient for K (long range)
   * @param omega Range-separation parameter ω for long-range corrected
   * functionals (0 = no range separation)
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations. ``NAO`` is the number of atomic orbitals.
   */
  virtual void build_JK(const double* P, double* J, double* K, double alpha,
                        double beta, double omega);

  /**
   * @brief Compute gradients of J and K energies with respect to nuclear
   * coordinates (pure virtual)
   *
   * Calculates Tr[∂J/∂R * P] and Tr[∂K/∂R * P] for analytical gradient
   * calculations.
   *
   * @param P Density matrix (input, size: num_density_matrices ×
   * num_atomic_orbitals × num_atomic_orbitals)
   * @param dJ Output Coulomb gradient (size: 3 × natoms)
   * @param dK Output Exchange gradient (size: 3 × natoms)
   * @param alpha Scaling coefficient for dK (un-attenuated)
   * @param beta Scaling coefficient for dK (long range)
   * @param omega Range-separation parameter ω for long-range corrected
   * functionals (0 = no range separation)
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *     unrestricted calculations. ``NAO`` is the number of atomic orbitals.
   */
  virtual void get_gradients(const double* P, double* dJ, double* dK,
                             double alpha, double beta, double omega) = 0;

  /**
   * @brief Perform quarter transformation of ERI tensor
   *
   * Transforms one index of the 4-center ERI tensor from AO to MO basis:
   * (μν|λk) → (pν|λk) where p are molecular orbitals.
   *
   * @param nt Number of transformed MO indices
   * @param C MO coefficient matrix (NAO × NMO)
   * @param out Output transformed tensor (NMO * NAO^3)
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations. ``NAO`` is the number of atomic orbitals.
   *       ``NMO`` is the number of molecular orbitals.
   */
  virtual void quarter_trans(size_t nt, const double* C, double* out);

  /**
   * @brief Create traditional (non-DF) ERI engine
   *
   * Factory method that selects the appropriate ERI implementation
   * based on data in the SCF Configuration.
   *
   * @param basis_set Atomic orbital basis set
   * @param cfg SCF configuration
   * @param omega Range-separation parameter ω
   * @return Shared pointer to ERI engine
   */
  static std::shared_ptr<ERI> create(BasisSet& basis_set, const SCFConfig& cfg,
                                     double omega);

  /**
   * @brief Create density-fitted ERI engine
   *
   * Factory method for density fitting (RI approximation) ERI engines.
   * Uses auxiliary basis set for 3-center integral approximation.
   *
   * @param basis_set Primary atomic orbital basis set
   * @param aux_basis_set Auxiliary (fitting) basis set
   * @param cfg SCF configuration
   * @param omega Range-separation parameter ω
   * @return Shared pointer to DF-ERI engine
   */
  static std::shared_ptr<ERI> create(BasisSet& basis_set,
                                     BasisSet& aux_basis_set,
                                     const SCFConfig& cfg, double omega);

 protected:
  /**
   * @brief Implementation of build_JK
   *
   * Derived classes must implement this to provide actual J/K computation.
   *
   * @see ERI::build_JK for API documentation
   */
  virtual void build_JK_impl_(const double* P, double* J, double* K,
                              double alpha, double beta, double omega) = 0;

  /**
   * @brief Implementation the first quarter transformation for MO2AO
   *
   * Derived classes must implement this to provide MO transformation.
   *
   * @see ERI::quarter_trans for API documentation
   */
  virtual void quarter_trans_impl(size_t nt, const double* C, double* out) = 0;

  bool unrestricted_;    ///< Whether this is an unrestricted calculation
  double tolerance_;     ///< Integral screening threshold
  BasisSet& basis_set_;  ///< Reference to the atomic orbital basis set
  ParallelConfig mpi_;   ///< MPI parallelization configuration
};
}  // namespace qdk::chemistry::scf
