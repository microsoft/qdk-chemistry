// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

/**
 * @file
 * @brief Double factorization of a Hamiltonian's two-electron integrals.
 *
 * @note Equation numbers here refer to :cite:`Low2025`.
 */

/**
 * @class DoubleFactorizerSettings
 * @brief Settings container for DoubleFactorizer.
 *
 * @see DoubleFactorizer
 */
class DoubleFactorizerSettings : public qdk::chemistry::data::Settings {
 public:
  /// Default eigenvalue threshold for retaining two-body fragments.
  static constexpr double DEFAULT_TRUNCATION_THRESHOLD = 1e-12;

  /**
   * @brief Constructor that initializes the default settings.
   */
  DoubleFactorizerSettings() {
    set_default<double>(
        "truncation_threshold", DEFAULT_TRUNCATION_THRESHOLD,
        "Drop fragments whose two-electron supermatrix eigenvalue magnitude "
        "is below this threshold. Must be non-negative; 0.0 retains every "
        "fragment, including the numerically null ones.",
        qdk::chemistry::data::BoundConstraint<double>{
            0.0, std::numeric_limits<double>::max()});
  }
  ~DoubleFactorizerSettings() override = default;
};

/// A single low-rank ("perfect square") two-electron fragment:
///
/// g^(f)_pqrs = sign * (sum_b eps_b U_pb U_qb) (sum_b' eps_b' U_rb' U_sb')
struct TwoBodyFragment {
  Eigen::MatrixXd U;    ///< norb x norb orbital rotation. Column b is
                        ///< new-orbital vector b in the original basis.
  Eigen::VectorXd eps;  ///< norb coefficients, scaled by
                        ///< sqrt(|supermatrix eigenvalue|).
  double sign = 1.0;    ///< +1.0 or -1.0.
};

/// Eigen-decompose the spin-free two-electron tensor g_pqrs, flattened as
/// p*norb^3 + q*norb^2 + r*norb + s, into low-rank fragments.
///
/// @param two_body_integrals Flattened two-electron tensor, size norb^4.
///        Chemist permutation symmetry is imposed by averaging, not verified.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Fragments whose supermatrix eigenvalue
///        magnitude falls below this threshold are dropped. 0.0 retains every
///        fragment.
/// @return The retained fragments, sorted by decreasing eigenvalue magnitude.
/// @throws std::invalid_argument if `norb` is zero, if `truncation_threshold`
///         is negative or NaN, or if `two_body_integrals` is not norb^4 long
///         or contains a non-finite value.
/// @throws std::runtime_error if a LAPACK diagonalization fails.
std::vector<TwoBodyFragment> eigen_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold =
        DoubleFactorizerSettings::DEFAULT_TRUNCATION_THRESHOLD);

/**
 * @class DoubleFactorizer
 * @brief Exact double factorization by nested eigen-decomposition
 *        :cite:`vonBurg2021`.
 *
 * Maps a Hamiltonian carrying dense four-index two-electron integrals to an
 * equivalent Hamiltonian backed by a
 * qdk::chemistry::data::FactorizedHamiltonianContainer, whose two-electron
 * tensor is stored as a signed sum of low-rank fragments
 *   g_pqrs = sum_t s_t (sum_b eps^t_b U^t_bp U^t_bq)
 *                      (sum_b' eps^t_b' U^t_b'r U^t_b's).
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
   * @return "eigen_decomposition".
   */
  std::string name() const final { return "eigen_decomposition"; }

  /**
   * @brief Access the algorithm's type name.
   *
   * @return "double_factorizer".
   */
  std::string type_name() const final { return "double_factorizer"; };

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
struct DoubleFactorizerFactory
    : public AlgorithmFactory<DoubleFactorizer, DoubleFactorizerFactory> {
  static std::string algorithm_type_name() { return "double_factorizer"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "eigen_decomposition"; }
};

}  // namespace qdk::chemistry::algorithms
