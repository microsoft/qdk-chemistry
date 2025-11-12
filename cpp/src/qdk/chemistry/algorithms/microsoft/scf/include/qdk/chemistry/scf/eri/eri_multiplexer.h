// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/eri.h>

#include <memory>

namespace qdk::chemistry::scf {
/**
 * @brief ERI multiplexer for routing different integral types to specialized
 * implementations
 *
 * The ERIMultiplexer class acts as a dispatcher that routes different types of
 * electron repulsion integral (ERI) operations to specialized backend
 * implementations. This design allows optimal performance by using different
 * methods for:
 * - J matrix (Coulomb integrals)
 * - K matrix (exchange integrals)
 * - Gradients
 * - Quarter transformations (for post-HF methods)
 *
 * For example, density fitting (DF-J) can be used for Coulomb integrals while
 * conventional methods handle exchange
 */
class ERIMultiplexer : public ERI {
  std::shared_ptr<ERI>
      j_impl_;  ///< Implementation for J (Coulomb) matrix construction
  std::shared_ptr<ERI>
      k_impl_;  ///< Implementation for K (exchange) matrix construction
  std::shared_ptr<ERI>
      grad_impl_;  ///< Implementation for gradient calculations
  std::shared_ptr<ERI>
      qt_impl_;  ///< Implementation for quarter transformations

  /**
   * @brief Construct multiplexer without auxiliary basis
   *
   * Creates an ERIMultiplexer using the primary basis set for all integral
   * evaluations. Different methods can still be specified via cfg for J, K,
   * and gradient operations.
   *
   * @param basis_set Primary basis set for integral evaluation
   * @param cfg SCF configuration specifying ERI methods for each operation type
   * @param omega Range-separation parameter for range-separated hybrid
   * functionals
   */
  ERIMultiplexer(BasisSet& basis_set, const SCFConfig& cfg, double omega);

  /**
   * @brief Construct multiplexer with auxiliary basis for density fitting
   *
   * Creates an ERIMultiplexer that can use density fitting (DF-J) with the
   * auxiliary basis for Coulomb integrals
   *
   * @param basis_set Primary basis set for integral evaluation
   * @param aux_basis_set Auxiliary basis set for density fitting
   * @param cfg SCF configuration specifying ERI methods and DF settings
   * @param omega Range-separation parameter for range-separated hybrid
   * functionals (bohr⁻¹)
   */
  ERIMultiplexer(BasisSet& basis_set, BasisSet& aux_basis_set,
                 const SCFConfig& cfg, double omega);

  /**
   * @brief Default constructor (private, used by factory methods)
   */
  ERIMultiplexer() noexcept = default;

 public:
  /**
   * @brief Build Coulomb (J) and exchange (K) matrices
   *
   * Routes J and K matrix construction to their respective implementations.
   * If the same method is configured for both, they are computed together
   * for efficiency. Otherwise, J and K are computed separately.
   *
   * @see ERI::build_JK for API documentation
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) override;

  /**
   * @brief Compute gradients of J and K energies
   *
   * Routes gradient computation to the configured gradient implementation.
   * Gradients are needed for geometry optimization and molecular dynamics.
   *
   * @see ERI::get_gradients for API documentation
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) override;

  /**
   * @brief Perform quarter transformation of two-electron integrals
   *
   * Routes quarter transformation to the configured implementation.
   * Quarter transformation converts (μν|λk) integrals to (pν|λk) where
   * p is an MO index
   *
   * @see ERI::quarter_trans for API documentation
   */
  void quarter_trans(size_t nt, const double* C, double* out) override;

  /**
   * @brief Factory method to create shared_ptr to ERIMultiplexer
   *
   * @tparam Args Constructor argument types (deduced)
   * @param args Constructor arguments (basis set, config, omega, etc.)
   * @return std::shared_ptr<ERI> Shared pointer to newly created multiplexer
   */
  template <typename... Args>
  static std::shared_ptr<ERI> create(Args&&... args) {
    return std::shared_ptr<ERIMultiplexer>(
        new ERIMultiplexer(std::forward<Args>(args)...));
  }

 protected:
  /**
   * @brief Stub implementation for build_JK_impl
   *
   * Empty override since routing is handled in the public build_JK method.
   * The actual work is delegated to j_impl_ and k_impl_.
   */
  void build_JK_impl_(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) override {};

  /**
   * @brief Stub implementation for quarter_trans_impl
   *
   * Empty override since routing is handled in the public quarter_trans method.
   * The actual work is delegated to qt_impl_.
   */
  void quarter_trans_impl(size_t nt, const double* C, double* out) override {};
};
}  // namespace qdk::chemistry::scf
