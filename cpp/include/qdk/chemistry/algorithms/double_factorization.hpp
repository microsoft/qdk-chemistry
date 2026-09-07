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

/// First-step factorization of the two-electron supermatrix.
enum class DoubleFactorizationMethod {
  Cholesky,  ///< Pivoted Cholesky, O(naux * norb^4).
  Eigen,     ///< Dense eigen-decomposition, O(norb^6).
};

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
    set_default("method", std::string("eigen_decomposition"),
                "First-step factorization of the two-electron supermatrix. "
                "\"eigen_decomposition\" diagonalizes it in O(norb^6); "
                "\"cholesky\" runs a pivoted Cholesky in O(naux * norb^4), "
                "stops at the numerical rank, reuses stored three-center "
                "integrals when the Hamiltonian carries them, and falls back "
                "to \"eigen_decomposition\" when the supermatrix is not "
                "positive semi-definite.",
                qdk::chemistry::data::ListConstraint<std::string>{
                    {"eigen_decomposition", "cholesky"}});

    set_default<double>(
        "truncation_threshold", 1e-12,
        "Drop fragments whose squared coefficient norm ||eps||^2 is below "
        "this threshold. For eigen_decomposition this equals the two-electron "
        "supermatrix eigenvalue magnitude. Must be non-negative; 0.0 keeps "
        "every fragment the method produces, including the numerically null "
        "ones, though cholesky still stops at the numerical rank.",
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
  Eigen::VectorXd eps;  ///< norb coefficients. The fragment's weight is
                        ///< ||eps||^2, which for eigen_decompose_two_body
                        ///< equals the supermatrix eigenvalue magnitude.
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
    double truncation_threshold = 1e-12);

/// Build low-rank fragments from Cholesky vectors of the two-electron
/// supermatrix, i.e. from an L satisfying g_pqrs = sum_Q L_(pq),Q L_(rs),Q.
///
/// Column Q is reshaped into the norb x norb matrix M^Q, whose eigenpairs give
/// the fragment directly. Because L already carries the fragment magnitude,
/// `eps` is used unscaled, and every fragment has sign +1: an L exists only
/// when the supermatrix is positive semi-definite.
///
/// @param cholesky_vectors norb^2 x naux matrix. Row index is the row-major
///        pair p*norb + q, column index is the Cholesky (auxiliary) index.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Fragments whose squared coefficient norm
///        ||eps||^2 falls below this threshold are dropped.
/// @return The retained fragments, sorted by decreasing ||eps||^2.
/// @throws std::invalid_argument if `norb` is zero, if `truncation_threshold`
///         is negative or NaN, or if `cholesky_vectors` does not have norb^2
///         rows or contains a non-finite value.
/// @throws std::runtime_error if a LAPACK diagonalization fails.
std::vector<TwoBodyFragment> fragments_from_cholesky_vectors(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    double truncation_threshold = 1e-12);

/// Decompose the spin-free two-electron tensor g_pqrs, flattened as
/// p*norb^3 + q*norb^2 + r*norb + s, by pivoted Cholesky decomposition of the
/// two-electron supermatrix :cite:`Beebe1977` :cite:`Koch2003`.
///
/// This costs O(naux * norb^4) rather than the O(norb^6) of a dense
/// diagonalization, and stops at the numerical rank instead of materializing
/// all norb^2 eigenpairs.
///
/// A Cholesky decomposition exists only for a positive semi-definite
/// supermatrix. Exact two-electron integrals are positive semi-definite, but
/// approximate or synthetic ones need not be, so a detected breakdown falls
/// back to eigen_decompose_two_body() rather than failing. The fallback is
/// observable in the result: it is the only way a fragment can carry sign -1.
///
/// @param two_body_integrals Flattened two-electron tensor, size norb^4.
///        Chemist permutation symmetry is imposed by averaging, not verified.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Fragments whose squared coefficient norm
///        ||eps||^2 falls below this threshold are dropped. This is the same
///        quantity the eigen path thresholds, so a given value selects the
///        same fragments from either method.
/// @return The retained fragments, sorted by decreasing ||eps||^2.
/// @throws std::invalid_argument if `norb` is zero, if `truncation_threshold`
///         is negative or NaN, or if `two_body_integrals` is not norb^4 long
///         or contains a non-finite value.
/// @throws std::runtime_error if a LAPACK diagonalization fails.
std::vector<TwoBodyFragment> cholesky_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold = 1e-12);

/**
 * @class DoubleFactorizer
 * @brief Exact double factorization of a Hamiltonian's two-electron integrals
 *        :cite:`vonBurg2021`.
 *
 * Maps a Hamiltonian carrying dense four-index two-electron integrals to an
 * equivalent Hamiltonian backed by a
 * qdk::chemistry::data::FactorizedHamiltonianContainer, whose two-electron
 * tensor is stored as a signed sum of low-rank fragments
 *   g_pqrs = sum_t s_t (sum_b eps^t_b U^t_bp U^t_bq)
 *                      (sum_b' eps^t_b' U^t_b'r U^t_b's).
 *
 * The `"method"` setting selects the first factorization step: an
 * eigen-decomposition of the two-electron supermatrix, or a pivoted Cholesky
 * decomposition of it :cite:`Beebe1977` :cite:`Koch2003`. Both produce the
 * same container and threshold the same quantity, so a given
 * `"truncation_threshold"` selects the same fragments from either.
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
   * @return "qdk".
   */
  std::string name() const override { return "qdk"; }

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

 private:
  /**
   * @brief Produce the low-rank fragments for a Hamiltonian.
   *
   * Only this step differs between methods; the container assembly, the
   * carried-over one-body data and the validation are shared.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The validated, restricted input Hamiltonian.
   * @param norb Number of active spatial orbitals.
   * @param truncation_threshold Fragments whose squared coefficient norm
   *        ||eps||^2 falls below this threshold are dropped.
   * @param method First-step factorization to apply.
   * \endcond
   * @return The retained fragments.
   */
  std::vector<TwoBodyFragment> _compute_fragments(
      const data::Hamiltonian& hamiltonian, std::size_t norb,
      double truncation_threshold, DoubleFactorizationMethod method) const;
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
  static std::string default_algorithm_name() { return "qdk"; }
};

}  // namespace qdk::chemistry::algorithms
