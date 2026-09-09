// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <lapack.hh>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::unique_ptr<DoubleFactorizer> make_double_factorizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<DoubleFactorizer>();
}

/// First factorization: pivoted Cholesky decomposition of the two-electron
/// supermatrix.
///
/// Pivoting runs in the symmetric-pair basis of dimension norb(norb+1)/2
/// rather than the full norb^2, since the supermatrix is symmetric under
/// p<->q. Accepted vectors are expanded back to the norb^2 pair layout on the
/// way out, so that computed and stored three-center integrals feed the same
/// second factorization.
///
/// Chemist permutation symmetry is imposed by averaging rather than verified,
/// which also keeps the norb^4 supermatrix from ever being materialized.
///
/// @param two_body_integrals Flattened two-electron tensor, size norb^4,
///        indexed p*norb^3 + q*norb^2 + r*norb + s.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Pivoting stops once the largest remaining
///        residual diagonal falls to or below this value.
/// @return norb^2 x naux matrix L with g_pqrs = sum_Q L_(pq),Q L_(rs),Q.
/// @throws std::invalid_argument if `two_body_integrals` is not norb^4 long,
///         contains a non-finite value, or is not positive semi-definite,
///         since no Cholesky decomposition exists in that last case.
Eigen::MatrixXd cholesky_vectors_from_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::size_t pair_dim = norb * norb;
  const std::size_t expected = pair_dim * pair_dim;
  if (static_cast<std::size_t>(two_body_integrals.size()) != expected) {
    throw std::invalid_argument(
        "double_factorizer: expected norb^4 = " + std::to_string(expected) +
        " elements for norb = " + std::to_string(norb) + ", got " +
        std::to_string(two_body_integrals.size()) + ".");
  }

  if (!two_body_integrals.allFinite()) {
    throw std::invalid_argument(
        "double_factorizer: two_body_integrals contains a non-finite "
        "value (NaN or infinity).");
  }

  const Eigen::Map<const RowMajorMatrix> raw(
      two_body_integrals.data(), static_cast<Eigen::Index>(pair_dim),
      static_cast<Eigen::Index>(pair_dim));

  // keeps only pairs satisfying (p\le q).
  const std::size_t reduced_dim = norb * (norb + 1) / 2;
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(reduced_dim);
  for (std::size_t p = 0; p < norb; ++p) {
    for (std::size_t q = p; q < norb; ++q) {
      pairs.emplace_back(p, q);
    }
  }

  // Symmetrize the input
  Eigen::MatrixXd reduced(reduced_dim, reduced_dim);
  for (std::size_t p = 0; p < reduced_dim; ++p) {
    const auto [i, j] = pairs[p];
    const Eigen::Index ij = static_cast<Eigen::Index>(i * norb + j);
    const Eigen::Index ji = static_cast<Eigen::Index>(j * norb + i);
    for (std::size_t q = 0; q < reduced_dim; ++q) {
      const auto [k, l] = pairs[q];
      const Eigen::Index kl = static_cast<Eigen::Index>(k * norb + l);
      const Eigen::Index lk = static_cast<Eigen::Index>(l * norb + k);
      reduced(p, q) =
          0.125 * (raw(ij, kl) + raw(ji, kl) + raw(ij, lk) + raw(ji, lk) +
                   raw(kl, ij) + raw(kl, ji) + raw(lk, ij) + raw(lk, ji));
    }
  }

  Eigen::VectorXd residual_diagonal = reduced.diagonal();
  // Numerical floor relative to the largest diagonal
  const double diagonal_scale = std::max(residual_diagonal.maxCoeff(), 0.0);
  const double noise_floor = std::numeric_limits<double>::epsilon() *
                             static_cast<double>(reduced_dim) * diagonal_scale;
  const double stop_threshold = std::max(truncation_threshold, noise_floor);

  std::vector<Eigen::VectorXd> vectors;
  vectors.reserve(reduced_dim);
  // At every iteration, selects the largest remaining diagonal
  for (std::size_t step = 0; step < reduced_dim; ++step) {
    Eigen::Index pivot = 0;
    const double pivot_value = residual_diagonal.maxCoeff(&pivot);
    const double most_negative = residual_diagonal.minCoeff();

    // A residual diagonal that goes convincingly negative means the
    // supermatrix has a negative eigenvalue, so no L with g = L L^T exists.
    if (most_negative < -noise_floor) {
      throw std::invalid_argument(
          "double_factorizer: the two-electron supermatrix is not positive "
          "semi-definite, so it has no Cholesky decomposition. Its most "
          "negative residual diagonal is " +
          std::to_string(most_negative) + ".");
    }

    if (pivot_value <= stop_threshold) {
      break;
    }

    Eigen::VectorXd column = reduced.col(pivot);
    for (const auto& vector : vectors) {
      column -= vector * vector[pivot];
    }
    column /= std::sqrt(pivot_value);

    residual_diagonal -= column.cwiseAbs2();
    vectors.push_back(std::move(column));
  }

  // The pivot loop only ever inspects the residual *diagonal*, so a negative
  // eigenvalue whose eigenvector is spread across off-diagonal entries can
  // reach this point undetected.
  // Certify: for a positive semi-definite residual R the largest entry lies
  // on the diagonal (|R_ij| <= sqrt(R_ii R_jj)), so max|g - L L^T| cannot
  // exceed the value pivoting stopped at. The safety factor keeps rounding and
  // truncation error.
  constexpr double residual_safety_factor = 8.0;
  const double deviation_tolerance =
      residual_safety_factor * (stop_threshold + noise_floor);

  Eigen::MatrixXd basis(static_cast<Eigen::Index>(reduced_dim),
                        static_cast<Eigen::Index>(vectors.size()));
  for (std::size_t q = 0; q < vectors.size(); ++q) {
    basis.col(static_cast<Eigen::Index>(q)) = vectors[q];
  }

  constexpr Eigen::Index block_columns = 256;
  const auto columns = static_cast<Eigen::Index>(reduced_dim);
  double max_deviation = 0.0;
  for (Eigen::Index start = 0; start < columns; start += block_columns) {
    const Eigen::Index width = std::min(block_columns, columns - start);
    const Eigen::MatrixXd reconstructed =
        basis * basis.middleRows(start, width).transpose();
    max_deviation = std::max(max_deviation,
                             (reduced.middleCols(start, width) - reconstructed)
                                 .cwiseAbs()
                                 .maxCoeff());
  }

  if (max_deviation > deviation_tolerance) {
    throw std::invalid_argument(
        "double_factorizer: the two-electron supermatrix is not positive "
        "semi-definite, so it has no Cholesky decomposition. The recovered "
        "vectors reproduce it only to " +
        std::to_string(max_deviation) + ", against a tolerance of " +
        std::to_string(deviation_tolerance) + ".");
  }

  // Expand each reduced vector over both orders of its orbital pair.
  Eigen::MatrixXd cholesky_vectors =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(pair_dim),
                            static_cast<Eigen::Index>(vectors.size()));
  for (std::size_t q = 0; q < vectors.size(); ++q) {
    const Eigen::Index target = static_cast<Eigen::Index>(q);
    for (std::size_t p = 0; p < reduced_dim; ++p) {
      const auto [i, j] = pairs[p];
      cholesky_vectors(static_cast<Eigen::Index>(i * norb + j), target) =
          vectors[q][p];
      cholesky_vectors(static_cast<Eigen::Index>(j * norb + i), target) =
          vectors[q][p];
    }
  }
  return cholesky_vectors;
}

/// Second factorization, shared by both inputs: diagonalize every Cholesky
/// vector of an L satisfying g_pqrs = sum_Q L_(pq),Q L_(rs),Q, writing the
/// result straight into the flat arrays FactorizedHamiltonianContainer takes.
///
/// Column Q reshapes into the norb x norb matrix M^Q, whose eigenpairs are
/// rank Q: the eigenvectors are the orbital rotation and the eigenvalues the
/// coefficients. Because L already carries the fragment magnitude, the
/// eigenvalues are used unscaled.
///
/// @param cholesky_vectors norb^2 x naux matrix. Row index is the row-major
///        pair p*norb + q, column index is the Cholesky (auxiliary) index.
/// @param norb Number of (spatial) orbitals.
/// @param u_matrices Resized to naux * norb * norb and filled rank by rank.
/// @param w_matrices Resized to naux * norb and filled rank by rank.
/// @return The number of ranks written, one per Cholesky vector.
/// @throws std::invalid_argument if `cholesky_vectors` does not have norb^2
///         rows or contains a non-finite value.
/// @throws std::runtime_error if a LAPACK diagonalization fails.
std::size_t fragments_from_cholesky_vectors(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    Eigen::VectorXd& u_matrices, Eigen::VectorXd& w_matrices) {
  QDK_LOG_TRACE_ENTERING();

  const std::size_t pair_dim = norb * norb;
  if (static_cast<std::size_t>(cholesky_vectors.rows()) != pair_dim) {
    throw std::invalid_argument(
        "double_factorizer: expected norb^2 = " + std::to_string(pair_dim) +
        " rows for norb = " + std::to_string(norb) + ", got " +
        std::to_string(cholesky_vectors.rows()) + ".");
  }

  if (!cholesky_vectors.allFinite()) {
    throw std::invalid_argument(
        "double_factorizer: cholesky_vectors contains a non-finite value.");
  }

  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);
  const Eigen::Index num_ranks = cholesky_vectors.cols();

  // syev leaves the eigenvectors as columns in column-major order, which is
  // bytewise the row-major [basis, orbital] layout the container stores, so
  // each rank is diagonalized straight into its slot.
  u_matrices.resize(num_ranks * num_orbitals * num_orbitals);
  w_matrices.resize(num_ranks * num_orbitals);

  bool warned_asymmetric = false;
  for (Eigen::Index q = 0; q < num_ranks; ++q) {
    const Eigen::Map<const RowMajorMatrix> pair_matrix(
        cholesky_vectors.col(q).data(), num_orbitals, num_orbitals);
    Eigen::Map<Eigen::MatrixXd> rotation(
        u_matrices.data() + q * num_orbitals * num_orbitals, num_orbitals,
        num_orbitals);

    // `rotation` maps the output buffer and `pair_matrix` is a const map, so
    // the caller's vectors are never written through. A Cholesky vector is
    // pair already.
    if (!warned_asymmetric) {
      const double asymmetry =
          (pair_matrix - pair_matrix.transpose()).cwiseAbs().maxCoeff();
      const double scale = pair_matrix.cwiseAbs().maxCoeff();
      if (asymmetry > 1e-8 * std::max(scale, 1.0)) {
        QDK_LOGGER().warn(
            "double_factorizer: Cholesky vector {} is not symmetric in its "
            "orbital pair (largest asymmetry {:.3e}); only its symmetric part "
            "is factorized.",
            static_cast<long long>(q), asymmetry);
        warned_asymmetric = true;
      }
    }
    rotation = 0.5 * (pair_matrix + pair_matrix.transpose());

    const int64_t info = lapack::syev(
        lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
        rotation.data(), static_cast<int64_t>(norb),
        w_matrices.data() + q * num_orbitals);
    if (info != 0) {
      throw std::runtime_error(
          "double_factorizer: LAPACK syev failed to diagonalize (info=" +
          std::to_string(info) + ").");
    }
  }

  return static_cast<std::size_t>(num_ranks);
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DoubleFactorizer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  QDK_LOG_TRACE_ENTERING();

  using qdk::chemistry::data::CholeskyHamiltonianContainer;
  using qdk::chemistry::data::FactorizedHamiltonianContainer;

  if (!hamiltonian) {
    throw std::invalid_argument(type_name() + ": hamiltonian is null");
  }
  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        type_name() + " currently only supports restricted Hamiltonians.");
  }
  if (!hamiltonian->has_two_body_integrals()) {
    throw std::invalid_argument(
        type_name() +
        ": the Hamiltonian carries no two-body integrals to factorize.");
  }

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");

  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  if (norb == 0) {
    throw std::invalid_argument(type_name() +
                                ": norb must be greater than zero.");
  }
  // The settings constraint already rejects a negative threshold when it is
  // set, but NaN compares false against any bound and so survives it. It has
  // to be caught here because it would make every pivot comparison false and
  // silently yield zero fragments.
  if (std::isnan(truncation_threshold)) {
    throw std::invalid_argument(type_name() +
                                ": truncation_threshold must be a number, "
                                "got NaN.");
  }

  // First factorization. Stored three-center integrals already are one, so
  // they are used as-is and the dense norb^4 tensor is never formed; that is
  // also why truncation_threshold does not apply to them.
  const Eigen::MatrixXd* cholesky_vectors = nullptr;
  if (hamiltonian->has_container_type<CholeskyHamiltonianContainer>()) {
    const Eigen::MatrixXd& three_center =
        hamiltonian->get_container<CholeskyHamiltonianContainer>()
            .get_three_center_integrals()
            .first;

    if (static_cast<std::size_t>(three_center.rows()) == norb * norb) {
      QDK_LOGGER().debug(
          "double_factorizer: using the stored three-center integrals as the "
          "first factorization; truncation_threshold={:.3e} is ignored.",
          truncation_threshold);
      cholesky_vectors = &three_center;
    } else {
      QDK_LOGGER().debug(
          "double_factorizer: stored three-center integrals have {} rows but "
          "num_orbitals={} implies {}, decomposing the dense tensor instead.",
          three_center.rows(), norb, norb * norb);
    }
  }

  Eigen::MatrixXd computed_vectors;
  if (cholesky_vectors == nullptr) {
    const Eigen::VectorXd& two_body =
        std::get<0>(hamiltonian->get_two_body_integrals());
    computed_vectors =
        cholesky_vectors_from_two_body(two_body, norb, truncation_threshold);
    cholesky_vectors = &computed_vectors;
  }

  // Second factorization, identical for either source. The container takes
  // R = num_ranks fragments over B = norb bases, with C = 1 copy each.
  Eigen::VectorXd u_matrices;
  Eigen::VectorXd w_matrices;
  const std::size_t num_ranks = fragments_from_cholesky_vectors(
      *cholesky_vectors, norb, u_matrices, w_matrices);

  QDK_LOGGER().debug(
      "double_factorizer: num_orbitals={}, truncation_threshold={}, retained "
      "{} fragments.",
      norb, truncation_threshold, num_ranks);

  if (num_ranks == 0) {
    throw std::invalid_argument(
        type_name() +
        ": truncation_threshold=" + std::to_string(truncation_threshold) +
        " leaves the factorized Hamiltonian with no two-body term at all.");
  }

  const Eigen::MatrixXd wb_matrix =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(num_ranks), 1);
  constexpr double energy_gap = 0.0;

  const Eigen::MatrixXd inactive_fock =
      hamiltonian->has_inactive_fock_matrix()
          ? hamiltonian->get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      h_alpha, u_matrices, w_matrices, wb_matrix, hamiltonian->get_orbitals(),
      hamiltonian->get_core_energy(), inactive_fock, energy_gap,
      hamiltonian->get_type());

  return std::make_shared<data::Hamiltonian>(std::move(container));
}

void DoubleFactorizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  DoubleFactorizerFactory::register_instance(&make_double_factorizer);
}

}  // namespace qdk::chemistry::algorithms
