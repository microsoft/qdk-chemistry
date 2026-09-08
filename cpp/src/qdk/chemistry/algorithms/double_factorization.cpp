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

/// A single low-rank ("perfect square") two-electron fragment:
///
/// g^(f)_pqrs = (sum_b eps_b U_pb U_qb) (sum_b' eps_b' U_rb' U_sb')
///
/// This is an implementation detail of the two-step factorization: the first
/// step produces Cholesky vectors, the second turns each one into a fragment.
/// The public surface is the resulting container, not this.
struct TwoBodyFragment {
  Eigen::MatrixXd U;    ///< norb x norb orbital rotation. Column b is
                        ///< new-orbital vector b in the original basis.
  Eigen::VectorXd eps;  ///< norb coefficients.
};

std::unique_ptr<DoubleFactorizer> make_double_factorizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<DoubleFactorizer>();
}

void validate_norb_and_threshold(std::size_t norb, double truncation_threshold,
                                 const std::string& context) {
  if (norb == 0) {
    throw std::invalid_argument(context + ": norb must be greater than zero.");
  }

  if (truncation_threshold < 0.0 || std::isnan(truncation_threshold)) {
    throw std::invalid_argument(context +
                                ": truncation_threshold must be "
                                "non-negative, got " +
                                std::to_string(truncation_threshold) + ".");
  }
}

/// Reshape a length-norb^2 pair vector, indexed row-major as p*norb + q, into
/// an norb x norb matrix and diagonalize it.
void diagonalize_pair_vector(const double* pair_vector, std::size_t norb,
                             const std::string& context,
                             Eigen::MatrixXd& eigenvectors,
                             Eigen::VectorXd& eigenvalues) {
  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);
  const Eigen::Map<const RowMajorMatrix> raw(pair_vector, num_orbitals,
                                             num_orbitals);
  eigenvectors = 0.5 * (raw + raw.transpose());
  eigenvalues.resize(num_orbitals);

  const int64_t info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
      eigenvectors.data(), static_cast<int64_t>(norb), eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(context +
                             ": LAPACK syev failed to diagonalize (info=" +
                             std::to_string(info) + ").");
  }
}

/// Reshape the flattened tensor into the (pq),(rs) supermatrix and impose the
/// (pq)<->(rs) generator by averaging.
Eigen::MatrixXd build_supermatrix(const Eigen::VectorXd& two_body_integrals,
                                  std::size_t pair_dim) {
  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);
  const Eigen::Map<const RowMajorMatrix> raw(two_body_integrals.data(),
                                             pair_size, pair_size);
  return 0.5 * (raw + raw.transpose());
}

void validate_two_body_integrals(const Eigen::VectorXd& two_body_integrals,
                                 std::size_t norb, std::size_t pair_dim,
                                 const std::string& context) {
  const std::size_t expected = pair_dim * pair_dim;
  if (static_cast<std::size_t>(two_body_integrals.size()) != expected) {
    throw std::invalid_argument(
        context + ": expected norb^4 = " + std::to_string(expected) +
        " elements for norb = " + std::to_string(norb) + ", got " +
        std::to_string(two_body_integrals.size()) + ".");
  }

  if (!two_body_integrals.allFinite()) {
    throw std::invalid_argument(context +
                                ": two_body_integrals contains a non-finite "
                                "value (NaN or infinity).");
  }
}

double fragment_coefficient_one_norm(const TwoBodyFragment& fragment) {
  return fragment.eps.cwiseAbs().sum();
}

void validate_three_center(const Eigen::MatrixXd& three_center,
                           std::size_t norb, const std::string& context) {
  const std::size_t pair_dim = norb * norb;
  if (static_cast<std::size_t>(three_center.rows()) != pair_dim) {
    throw std::invalid_argument(
        context + ": expected norb^2 = " + std::to_string(pair_dim) +
        " rows for norb = " + std::to_string(norb) + ", got " +
        std::to_string(three_center.rows()) + ".");
  }

  if (!three_center.allFinite()) {
    throw std::invalid_argument(
        context +
        ": cholesky_vectors contains a non-finite value (NaN or infinity).");
  }
}

/// Second factorization, shared by both inputs: diagonalize each Cholesky
/// vector of an L satisfying g_pqrs = sum_Q L_(pq),Q L_(rs),Q.
///
/// Column Q is reshaped into the norb x norb matrix M^Q, whose eigenpairs are
/// the fragment. Because L already carries the fragment magnitude, `eps` is
/// used unscaled.
///
/// @param cholesky_vectors norb^2 x naux matrix. Row index is the row-major
///        pair p*norb + q, column index is the Cholesky (auxiliary) index.
/// @param norb Number of (spatial) orbitals.
/// @param context Caller name, used in exception messages.
/// @return One fragment per column, sorted by decreasing sum_b |eps_b|.
std::vector<TwoBodyFragment> fragments_from_cholesky_vectors(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    const std::string& context) {
  QDK_LOG_TRACE_ENTERING();

  validate_three_center(cholesky_vectors, norb, context);

  std::vector<TwoBodyFragment> fragments;
  fragments.reserve(static_cast<std::size_t>(cholesky_vectors.cols()));
  for (Eigen::Index q = 0; q < cholesky_vectors.cols(); ++q) {
    TwoBodyFragment fragment;
    diagonalize_pair_vector(cholesky_vectors.col(q).data(), norb, context,
                            fragment.U, fragment.eps);
    fragments.push_back(std::move(fragment));
  }

  std::sort(fragments.begin(), fragments.end(),
            [](const TwoBodyFragment& a, const TwoBodyFragment& b) {
              return fragment_coefficient_one_norm(a) >
                     fragment_coefficient_one_norm(b);
            });

  return fragments;
}

/// First factorization: pivoted Cholesky decomposition of the two-electron
/// supermatrix :cite:`Beebe1977` :cite:`Koch2003`.
///
/// Pivoting runs in the symmetric-pair basis of dimension norb(norb+1)/2
/// rather than the full norb^2, since the supermatrix is symmetric under
/// p<->q. Each accepted vector is then expanded back to the norb^2 pair basis,
/// so the result has the same layout a CholeskyHamiltonianContainer stores and
/// can feed the same second factorization.
///
/// @return norb^2 x naux matrix L with g = L L^T.
/// @throws std::invalid_argument if the supermatrix is not positive
///         semi-definite, since no Cholesky decomposition exists then.
Eigen::MatrixXd cholesky_vectors_from_supermatrix(
    const Eigen::MatrixXd& supermatrix, std::size_t norb,
    double truncation_threshold, const std::string& context) {
  QDK_LOG_TRACE_ENTERING();

  const std::size_t pair_dim = norb * norb;
  const std::size_t reduced_dim = norb * (norb + 1) / 2;
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(reduced_dim);
  for (std::size_t p = 0; p < norb; ++p) {
    for (std::size_t q = p; q < norb; ++q) {
      pairs.emplace_back(p, q);
    }
  }

  Eigen::MatrixXd reduced(reduced_dim, reduced_dim);
  for (std::size_t p = 0; p < reduced_dim; ++p) {
    const auto [i, j] = pairs[p];
    for (std::size_t q = 0; q < reduced_dim; ++q) {
      const auto [k, l] = pairs[q];
      reduced(p, q) = 0.25 * (supermatrix(i * norb + j, k * norb + l) +
                              supermatrix(j * norb + i, k * norb + l) +
                              supermatrix(i * norb + j, l * norb + k) +
                              supermatrix(j * norb + i, l * norb + k));
    }
  }

  Eigen::VectorXd residual_diagonal = reduced.diagonal();
  const double diagonal_scale = std::max(residual_diagonal.maxCoeff(), 0.0);
  const double noise_floor = std::numeric_limits<double>::epsilon() *
                             static_cast<double>(reduced_dim) * diagonal_scale;
  const double stop_threshold = std::max(truncation_threshold, noise_floor);

  std::vector<Eigen::VectorXd> vectors;
  vectors.reserve(reduced_dim);
  for (std::size_t step = 0; step < reduced_dim; ++step) {
    Eigen::Index pivot = 0;
    const double pivot_value = residual_diagonal.maxCoeff(&pivot);
    // A residual diagonal that goes convincingly negative means the
    // supermatrix has a negative eigenvalue, so no L with g = L L^T exists.
    // Continuing would return a factorization of a different tensor, so this
    // is reported rather than truncated away.
    if (residual_diagonal.minCoeff() < -noise_floor) {
      throw std::invalid_argument(
          context +
          ": the two-electron supermatrix is not positive semi-definite, so "
          "it has no Cholesky decomposition. Its most negative residual "
          "diagonal is " +
          std::to_string(residual_diagonal.minCoeff()) + ".");
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

  Eigen::MatrixXd cholesky_vectors(static_cast<Eigen::Index>(pair_dim),
                                   static_cast<Eigen::Index>(vectors.size()));
  for (std::size_t q = 0; q < vectors.size(); ++q) {
    Eigen::VectorXd expanded = Eigen::VectorXd::Zero(pair_dim);
    for (std::size_t p = 0; p < reduced_dim; ++p) {
      const auto [i, j] = pairs[p];
      expanded[static_cast<Eigen::Index>(i * norb + j)] = vectors[q][p];
      expanded[static_cast<Eigen::Index>(j * norb + i)] = vectors[q][p];
    }
    cholesky_vectors.col(static_cast<Eigen::Index>(q)) = expanded;
  }
  return cholesky_vectors;
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DoubleFactorizer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  QDK_LOG_TRACE_ENTERING();

  using qdk::chemistry::data::CholeskyHamiltonianContainer;
  using qdk::chemistry::data::FactorizedHamiltonianContainer;

  if (!hamiltonian) {
    throw std::invalid_argument(name() + ": hamiltonian is null");
  }
  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        name() + " currently only supports restricted Hamiltonians.");
  }
  if (!hamiltonian->has_two_body_integrals()) {
    throw std::invalid_argument(
        name() +
        ": the Hamiltonian carries no two-body integrals to factorize.");
  }

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");

  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());
  const std::string context = name();
  validate_norb_and_threshold(norb, truncation_threshold, context);

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
      cholesky_vectors = &three_center;
    } else {
      QDK_LOGGER().debug(
          "{}: stored three-center integrals have {} rows but "
          "num_orbitals={} implies {}, decomposing the dense tensor instead.",
          context, three_center.rows(), norb, norb * norb);
    }
  }

  Eigen::MatrixXd computed_vectors;
  if (cholesky_vectors == nullptr) {
    const Eigen::VectorXd& two_body =
        std::get<0>(hamiltonian->get_two_body_integrals());
    const std::size_t pair_dim = norb * norb;
    validate_two_body_integrals(two_body, norb, pair_dim, context);
    computed_vectors =
        cholesky_vectors_from_supermatrix(build_supermatrix(two_body, pair_dim),
                                          norb, truncation_threshold, context);
    cholesky_vectors = &computed_vectors;
  }

  // Second factorization, identical for either source.
  auto fragments =
      fragments_from_cholesky_vectors(*cholesky_vectors, norb, context);

  QDK_LOGGER().debug(
      "{}: num_orbitals={}, truncation_threshold={}, retained {} fragments.",
      context, norb, truncation_threshold, fragments.size());

  if (fragments.empty()) {
    throw std::invalid_argument(
        name() +
        ": truncation_threshold=" + std::to_string(truncation_threshold) +
        " leaves the factorized Hamiltonian with no two-body term at all.");
  }

  // R = number of fragments, B = norb bases, C = 1.
  const std::size_t num_ranks = fragments.size();
  const std::size_t num_bases = norb;
  const std::size_t num_copies = 1;

  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  Eigen::VectorXd u_matrices(num_ranks * num_bases * norb);
  Eigen::VectorXd w_matrices(num_ranks * num_bases * num_copies);

  for (std::size_t r = 0; r < num_ranks; ++r) {
    const TwoBodyFragment& fragment = fragments[r];
    const Eigen::Index rank = static_cast<Eigen::Index>(r);
    const Eigen::Index bases = static_cast<Eigen::Index>(num_bases);
    const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);

    w_matrices.segment(rank * bases, bases) = fragment.eps;
    Eigen::Map<RowMajorMatrix>(u_matrices.data() + rank * bases * num_orbitals,
                               bases, num_orbitals) = fragment.U.transpose();
  }

  const Eigen::MatrixXd wb_matrix =
      Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(num_ranks),
                            static_cast<Eigen::Index>(num_copies));
  constexpr double energy_gap = 0.0;

  const Eigen::MatrixXd inactive_fock =
      hamiltonian->has_inactive_fock_matrix()
          ? hamiltonian->get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      hamiltonian->get_core_energy(), u_matrices, w_matrices, wb_matrix,
      h_alpha, inactive_fock, hamiltonian->get_orbitals(), energy_gap,
      hamiltonian->get_type());

  return std::make_shared<data::Hamiltonian>(std::move(container));
}

void DoubleFactorizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  DoubleFactorizerFactory::register_instance(&make_double_factorizer);
}

}  // namespace qdk::chemistry::algorithms
