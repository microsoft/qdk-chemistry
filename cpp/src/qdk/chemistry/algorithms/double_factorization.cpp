// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

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

DoubleFactorizationMethod method_from_string(const std::string& method) {
  if (method == "cholesky") {
    return DoubleFactorizationMethod::Cholesky;
  }
  if (method == "eigen_decomposition") {
    return DoubleFactorizationMethod::Eigen;
  }
  throw std::invalid_argument("double_factorizer: unknown method " + method);
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
                           std::size_t norb, double truncation_threshold,
                           const std::string& context) {
  validate_norb_and_threshold(norb, truncation_threshold, context);

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

std::vector<TwoBodyFragment> factorize_by_eigendecomposition(
    const Eigen::MatrixXd& supermatrix, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "factorize_by_eigendecomposition";
  const std::size_t pair_dim = norb * norb;
  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);

  Eigen::MatrixXd supermatrix_eigenvectors = supermatrix;
  Eigen::VectorXd supermatrix_eigenvalues(pair_dim);

  // Dense diagonalization costs O(norb^6) and materializes all norb^2
  // eigenpairs. The "cholesky" method avoids both for a positive
  // semi-definite supermatrix.
  const int64_t supermatrix_info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(pair_dim),
      supermatrix_eigenvectors.data(), static_cast<int64_t>(pair_dim),
      supermatrix_eigenvalues.data());
  if (supermatrix_info != 0) {
    throw std::runtime_error(
        "factorize_by_eigendecomposition: LAPACK syev failed to "
        "diagonalize the two-body supermatrix (info=" +
        std::to_string(supermatrix_info) + ").");
  }

  // Sort by decreasing |eigenvalue| so the largest contributions come first.
  // The reshaped eigenvector has unit Frobenius norm, so |eigenvalue| is
  // exactly the fragment's squared coefficient norm ||eps||^2.
  // Within a degenerate block the eigenvector basis LAPACK returns is
  // arbitrary, so eps is not fixed by the tensor alone.
  std::vector<std::size_t> order(pair_dim);
  for (std::size_t n = 0; n < pair_dim; ++n) {
    order[n] = n;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return std::abs(supermatrix_eigenvalues[a]) >
           std::abs(supermatrix_eigenvalues[b]);
  });

  std::vector<TwoBodyFragment> fragments;
  fragments.reserve(pair_dim);
  for (std::size_t n : order) {
    const double eigenvalue = supermatrix_eigenvalues[n];
    if (std::abs(eigenvalue) < truncation_threshold) {
      continue;
    }

    TwoBodyFragment fragment;
    Eigen::VectorXd fragment_eigenvalues;
    diagonalize_pair_vector(supermatrix_eigenvectors.data() +
                                static_cast<Eigen::Index>(n) * pair_size,
                            norb, context, fragment.U, fragment_eigenvalues);

    fragment.sign = (eigenvalue >= 0.0) ? 1.0 : -1.0;
    fragment.eps = std::sqrt(std::abs(eigenvalue)) * fragment_eigenvalues;

    fragments.push_back(std::move(fragment));
  }

  return fragments;
}

std::vector<TwoBodyFragment> fragments_from_cholesky_vectors(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "fragments_from_cholesky_vectors";
  validate_three_center(cholesky_vectors, norb, truncation_threshold, context);

  std::vector<TwoBodyFragment> fragments;
  fragments.reserve(static_cast<std::size_t>(cholesky_vectors.cols()));
  for (Eigen::Index q = 0; q < cholesky_vectors.cols(); ++q) {
    TwoBodyFragment fragment;
    diagonalize_pair_vector(cholesky_vectors.col(q).data(), norb, context,
                            fragment.U, fragment.eps);

    // The Cholesky vector already carries the fragment magnitude, so eps is
    // not rescaled the way the eigen path rescales a unit-norm eigenvector.
    if (fragment.eps.squaredNorm() < truncation_threshold) {
      continue;
    }

    fragment.sign = 1.0;
    fragments.push_back(std::move(fragment));
  }

  std::sort(fragments.begin(), fragments.end(),
            [](const TwoBodyFragment& a, const TwoBodyFragment& b) {
              return fragment_coefficient_one_norm(a) >
                     fragment_coefficient_one_norm(b);
            });

  return fragments;
}

/// Eigen-decompose g = L L^T from its factor L, without forming the dense
/// norb^4 tensor. L L^T shares its non-zero eigenvalues with the naux x naux
/// Gram matrix L^T L, whose eigenvector w lifts to an eigenvector of the
/// supermatrix as L w / sqrt(s). That costs O(naux^2 * norb^2 + naux^3)
/// instead of the O(norb^6) dense diagonalization, and yields the same
/// fragments.
///
/// naux may exceed the rank of g, which is normal for an active space. The
/// redundant directions are exact zero modes of the Gram matrix, so they are
/// dropped rather than returned as numerical noise.
std::vector<TwoBodyFragment> fragments_from_cholesky_vectors_by_gram(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "fragments_from_cholesky_vectors_by_gram";
  validate_three_center(cholesky_vectors, norb, truncation_threshold, context);

  const Eigen::Index naux = cholesky_vectors.cols();
  std::vector<TwoBodyFragment> fragments;
  if (naux == 0) {
    return fragments;
  }

  Eigen::MatrixXd gram = cholesky_vectors.transpose() * cholesky_vectors;
  Eigen::VectorXd gram_eigenvalues(naux);
  const int64_t info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(naux),
      gram.data(), static_cast<int64_t>(naux), gram_eigenvalues.data());
  if (info != 0) {
    throw std::runtime_error(
        context +
        ": LAPACK syev failed to diagonalize the naux x naux Gram "
        "matrix (info=" +
        std::to_string(info) + ").");
  }

  // g is a Gram matrix, so its exact eigenvalues are non-negative and the
  // zero modes from redundant columns land at +-eps*scale. Scaling the floor
  // by naux keeps those below it, the same way factorize_by_cholesky scales
  // its own floor by the pair dimension.
  const double noise_floor = std::numeric_limits<double>::epsilon() *
                             static_cast<double>(naux) *
                             std::max(gram_eigenvalues.maxCoeff(), 0.0);
  const double drop_threshold = std::max(truncation_threshold, noise_floor);

  fragments.reserve(static_cast<std::size_t>(naux));
  // syev returns ascending eigenvalues; walk backwards for decreasing weight.
  for (Eigen::Index n = naux; n-- > 0;) {
    const double eigenvalue = gram_eigenvalues[n];
    if (eigenvalue <= drop_threshold) {
      continue;
    }

    const Eigen::VectorXd lifted =
        cholesky_vectors * gram.col(n) / std::sqrt(eigenvalue);

    TwoBodyFragment fragment;
    Eigen::VectorXd fragment_eigenvalues;
    diagonalize_pair_vector(lifted.data(), norb, context, fragment.U,
                            fragment_eigenvalues);

    // The lifted eigenvector is unit-norm, so it is rescaled exactly as the
    // dense eigen path rescales its own, giving ||eps||^2 == |eigenvalue|.
    fragment.sign = 1.0;
    fragment.eps = std::sqrt(eigenvalue) * fragment_eigenvalues;

    fragments.push_back(std::move(fragment));
  }

  return fragments;
}

/// Build fragments from stored three-center integrals for either method.
std::vector<TwoBodyFragment> fragments_from_three_center(
    const Eigen::MatrixXd& cholesky_vectors, std::size_t norb,
    double truncation_threshold, DoubleFactorizationMethod method) {
  return method == DoubleFactorizationMethod::Cholesky
             ? fragments_from_cholesky_vectors(cholesky_vectors, norb,
                                               truncation_threshold)
             : fragments_from_cholesky_vectors_by_gram(cholesky_vectors, norb,
                                                       truncation_threshold);
}

std::vector<TwoBodyFragment> factorize_by_cholesky(
    const Eigen::MatrixXd& supermatrix, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "factorize_by_cholesky";
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

  std::vector<Eigen::VectorXd> cholesky_vectors;
  cholesky_vectors.reserve(reduced_dim);
  for (std::size_t step = 0; step < reduced_dim; ++step) {
    Eigen::Index pivot = 0;
    const double pivot_value = residual_diagonal.maxCoeff(&pivot);
    if (residual_diagonal.minCoeff() < -noise_floor) {
      QDK_LOGGER().debug(
          "factorize_by_cholesky: supermatrix is not positive semi-definite, "
          "falling back to factorize_by_eigendecomposition.");
      return factorize_by_eigendecomposition(supermatrix, norb,
                                             truncation_threshold);
    }
    if (pivot_value <= stop_threshold) {
      break;
    }

    Eigen::VectorXd column = reduced.col(pivot);
    for (const auto& vector : cholesky_vectors) {
      column -= vector * vector[pivot];
    }
    column /= std::sqrt(pivot_value);

    residual_diagonal -= column.cwiseAbs2();
    cholesky_vectors.push_back(std::move(column));
  }

  std::vector<TwoBodyFragment> fragments;
  fragments.reserve(cholesky_vectors.size());
  for (const auto& vector : cholesky_vectors) {
    Eigen::VectorXd expanded = Eigen::VectorXd::Zero(pair_dim);
    for (std::size_t p = 0; p < reduced_dim; ++p) {
      const auto [i, j] = pairs[p];
      expanded[static_cast<Eigen::Index>(i * norb + j)] = vector[p];
      expanded[static_cast<Eigen::Index>(j * norb + i)] = vector[p];
    }

    TwoBodyFragment fragment;
    diagonalize_pair_vector(expanded.data(), norb, context, fragment.U,
                            fragment.eps);
    fragment.sign = 1.0;
    fragments.push_back(std::move(fragment));
  }

  std::sort(fragments.begin(), fragments.end(),
            [](const TwoBodyFragment& a, const TwoBodyFragment& b) {
              return fragment_coefficient_one_norm(a) >
                     fragment_coefficient_one_norm(b);
            });
  return fragments;
}

}  // namespace

std::vector<TwoBodyFragment> double_factorize(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold, DoubleFactorizationMethod method) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "double_factorize";
  validate_norb_and_threshold(norb, truncation_threshold, context);
  const std::size_t pair_dim = norb * norb;
  validate_two_body_integrals(two_body_integrals, norb, pair_dim, context);
  const Eigen::MatrixXd supermatrix =
      build_supermatrix(two_body_integrals, pair_dim);

  return method == DoubleFactorizationMethod::Cholesky
             ? factorize_by_cholesky(supermatrix, norb, truncation_threshold)
             : factorize_by_eigendecomposition(supermatrix, norb,
                                               truncation_threshold);
}

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

  const std::string method_name = _settings->get<std::string>("method");
  const DoubleFactorizationMethod method = method_from_string(method_name);

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");

  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  // Stored three-center integrals already are the first factorization.
  const Eigen::MatrixXd* stored_vectors = nullptr;
  if (hamiltonian->has_container_type<CholeskyHamiltonianContainer>()) {
    const Eigen::MatrixXd& three_center =
        hamiltonian->get_container<CholeskyHamiltonianContainer>()
            .get_three_center_integrals()
            .first;

    if (static_cast<std::size_t>(three_center.rows()) == norb * norb) {
      stored_vectors = &three_center;
    } else {
      QDK_LOGGER().debug(
          "double_factorizer: stored three-center integrals have {} rows but "
          "num_orbitals={} implies {}, decomposing the dense tensor instead.",
          three_center.rows(), norb, norb * norb);
    }
  }

  auto fragments =
      stored_vectors != nullptr
          ? fragments_from_three_center(*stored_vectors, norb,
                                        truncation_threshold, method)
          : double_factorize(std::get<0>(hamiltonian->get_two_body_integrals()),
                             norb, truncation_threshold, method);

  QDK_LOGGER().debug(
      "{}: method={}, num_orbitals={}, truncation_threshold={}, "
      "retained {} of {} candidate fragments.",
      name(), method_name, norb, truncation_threshold, fragments.size(),
      norb * norb);

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
  Eigen::VectorXd signs(num_ranks);

  for (std::size_t r = 0; r < num_ranks; ++r) {
    const TwoBodyFragment& fragment = fragments[r];
    const Eigen::Index rank = static_cast<Eigen::Index>(r);
    const Eigen::Index bases = static_cast<Eigen::Index>(num_bases);
    const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);

    signs(rank) = fragment.sign;
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
      h_alpha, inactive_fock, hamiltonian->get_orbitals(), signs, energy_gap,
      hamiltonian->get_type());

  return std::make_shared<data::Hamiltonian>(std::move(container));
}

void DoubleFactorizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  DoubleFactorizerFactory::register_instance(&make_double_factorizer);
}

}  // namespace qdk::chemistry::algorithms
