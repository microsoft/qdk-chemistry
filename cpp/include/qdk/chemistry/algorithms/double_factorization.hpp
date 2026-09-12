// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @class DoubleFactorizerSettings
 * @brief Settings container for DoubleFactorizer.
 *
 * @see DoubleFactorizer
 */
class DoubleFactorizerSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor that initializes the default settings.
   */
  DoubleFactorizerSettings() {
    set_default<double>(
        "truncation_threshold", 1e-12,
        "Cutoff for the pivoted Cholesky decomposition of the two-electron "
        "supermatrix: pivoting stops once the largest remaining residual "
        "diagonal drops to it. Must be non-negative; 0.0 keeps every "
        "numerically resolvable fragment. Ignored when the input Hamiltonian "
        "is backed by a CholeskyHamiltonianContainer.",
        qdk::chemistry::data::BoundConstraint<double>{
            0.0, std::numeric_limits<double>::max()});
  }
  ~DoubleFactorizerSettings() override = default;
};

/**
 * @class DoubleFactorizer
 * @brief Exact double factorization of a Hamiltonian's two-electron integrals
 *        :cite:`vonBurg2021`.
 *
 * Maps a Hamiltonian carrying dense four-index two-electron integrals to an
 * equivalent Hamiltonian backed by a
 * qdk::chemistry::data::FactorizedHamiltonianContainer, whose two-electron
 * tensor is stored as a sum of low-rank squares
 *   g_pqrs = sum_t (sum_b eps^t_b U^t_bp U^t_bq)
 *                  (sum_b' eps^t_b' U^t_b'r U^t_b's).
 *
 * The factorization has two steps. The first produces Cholesky vectors L with
 * g = L L^T; the second diagonalizes each vector into its (U, eps) fragment.
 * Only the first step depends on how the input stores its integrals:
 *
 * - Dense four-index integrals are reshaped into the (pq),(rs) supermatrix,
 *   given chemist permutation symmetry by averaging, and reduced by a pivoted
 *   Cholesky decomposition costing O(naux * norb^4), which stops at the
 *   numerical rank.
 * - A qdk::chemistry::data::CholeskyHamiltonianContainer already stores such
 *   vectors, so they are consumed directly. `"truncation_threshold"` is
 *   ignored.
 *
 * A Cholesky decomposition exists only for a positive semi-definite
 * supermatrix, and the caller guarantees that property. Exact two-electron
 * integrals have it by construction. Indefiniteness is rejected once it shows
 * up in the residual diagonal; a negative direction whose diagonal stays below
 * the truncation threshold is truncated away undetected.
 *
 * The one-electron integrals, core energy, orbitals, inactive Fock matrix and
 * Hamiltonian type are carried over unchanged.
 */
class DoubleFactorizer
    : public Algorithm<DoubleFactorizer, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>> {
 public:
  /**
   * @brief Default constructor. Uses default DoubleFactorizerSettings.
   */
  DoubleFactorizer() {
    _settings = std::make_unique<DoubleFactorizerSettings>();
  }

  /**
   * @brief Virtual destructor.
   */
  ~DoubleFactorizer() override = default;

  /**
   * @brief Double-factorize a Hamiltonian.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian to factorize. Must be restricted and
   *        carry two-electron integrals.
   * \endcond
   * @return A new Hamiltonian backed by a FactorizedHamiltonianContainer.
   *
   * @note Settings are automatically locked when this method is called.
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name.
   *
   * @return "double_factorization".
   */
  std::string name() const override { return "double_factorization"; }

  /**
   * @brief Access the algorithm's type name.
   *
   * @return "hamiltonian_factorization".
   */
  std::string type_name() const final { return "hamiltonian_factorization"; };

 protected:
  /**
   * @brief Factorize the two-electron tensor.
   *
   * @throws std::invalid_argument if `hamiltonian` is null, unrestricted, or
   *         carries no two-electron integrals, or if `truncation_threshold`
   *         discards every fragment.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian) const override;
};

/**
 * @brief Factory class for creating double factorizer instances.
 *
 * @see DoubleFactorizer
 */
struct HamiltonianFactorizationFactory
    : public AlgorithmFactory<DoubleFactorizer, HamiltonianFactorizationFactory> {
  static std::string algorithm_type_name() {
    return "hamiltonian_factorization";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "double_factorization"; }
};

}  // namespace qdk::chemistry::algorithms
