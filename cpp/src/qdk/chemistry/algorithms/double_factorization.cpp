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
#include <vector>

namespace qdk::chemistry::algorithms {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::unique_ptr<DoubleFactorizer> make_double_factorizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<DoubleFactorizer>();
}

/// Map the `"method"` setting onto the enum. The setting's ListConstraint
/// already rejects anything else, so an unknown value here means the
/// constraint and this mapping have drifted apart.
DoubleFactorizationMethod method_from_string(const std::string& method) {
  if (method == "cholesky") {
    return DoubleFactorizationMethod::Cholesky;
  }
  if (method == "eigen_decomposition") {
    return DoubleFactorizationMethod::Eigen;
  }
  throw std::invalid_argument("double_factorizer: unknown method \"" + method +
                              "\".");
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
/// an norb x norb matrix and diagonalize it. Averaging enforces the p<->q
/// generator, which is load-bearing rather than cosmetic: within a degenerate
/// block the basis LAPACK returns is arbitrary and need not be symmetric.
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
                             ": LAPACK syev failed to diagonalize a "
                             "fragment matrix (info=" +
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

}  // namespace

std::vector<TwoBodyFragment> eigen_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "eigen_decompose_two_body";
  validate_norb_and_threshold(norb, truncation_threshold, context);

  const std::size_t pair_dim = norb * norb;
  validate_two_body_integrals(two_body_integrals, norb, pair_dim, context);

  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);

  // Assumes chemist permutation symmetry: averaging imposes the (pq)<->(rs)
  // and p<->q generators rather than checking them, and the rest follow.
  Eigen::MatrixXd supermatrix_eigenvectors =
      build_supermatrix(two_body_integrals, pair_dim);
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
        "eigen_decompose_two_body: LAPACK syev failed to diagonalize the "
        "two-body supermatrix (info=" +
        std::to_string(supermatrix_info) + ").");
  }

  // Sort by decreasing |eigenvalue| so the largest contributions come first.
  // The reshaped eigenvector has unit Frobenius norm, so |eigenvalue| is
  // exactly the fragment weight ||eps||^2 that the Cholesky path sorts on.
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
  validate_norb_and_threshold(norb, truncation_threshold, context);

  const std::size_t pair_dim = norb * norb;
  if (static_cast<std::size_t>(cholesky_vectors.rows()) != pair_dim) {
    throw std::invalid_argument(
        context + ": expected norb^2 = " + std::to_string(pair_dim) +
        " rows for norb = " + std::to_string(norb) + ", got " +
        std::to_string(cholesky_vectors.rows()) + ".");
  }

  if (!cholesky_vectors.allFinite()) {
    throw std::invalid_argument(
        context +
        ": cholesky_vectors contains a non-finite value (NaN or infinity).");
  }

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

  // The pivot order is the order in which residual error is removed, which is
  // not the order of fragment weight, so sorting is a separate step.
  std::sort(fragments.begin(), fragments.end(),
            [](const TwoBodyFragment& a, const TwoBodyFragment& b) {
              return a.eps.squaredNorm() > b.eps.squaredNorm();
            });

  return fragments;
}

std::vector<TwoBodyFragment> cholesky_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  const std::string context = "cholesky_decompose_two_body";
  validate_norb_and_threshold(norb, truncation_threshold, context);

  const std::size_t pair_dim = norb * norb;
  validate_two_body_integrals(two_body_integrals, norb, pair_dim, context);

  const Eigen::MatrixXd supermatrix =
      build_supermatrix(two_body_integrals, pair_dim);
  Eigen::VectorXd residual_diagonal = supermatrix.diagonal();

  // Roundoff in the residual diagonal accumulates across the rank-one updates,
  // so both tolerances have to scale with the magnitude of the supermatrix and
  // with its dimension. A bare machine epsilon reports a positive
  // semi-definite supermatrix as indefinite once norb grows past a handful of
  // orbitals, which would silently send every realistic input down the
  // O(norb^6) fallback.
  const double scale = std::max(residual_diagonal.maxCoeff(), 1.0);
  const double noise_floor = std::numeric_limits<double>::epsilon() * scale *
                             static_cast<double>(pair_dim);

  // A pivot's fragment weight obeys ||eps||^2 <= pair_dim * pivot, so stopping
  // here can only skip fragments that truncation would have dropped anyway.
  const double stop_threshold = std::max(
      noise_floor, truncation_threshold / static_cast<double>(pair_dim));

  // A supermatrix built from a tensor with p<->q symmetry annihilates every
  // antisymmetric pair vector, so its range, and hence the Cholesky rank, is
  // bounded by the symmetric pair dimension.
  const Eigen::Index max_rank =
      static_cast<Eigen::Index>(norb * (norb + 1) / 2);
  Eigen::MatrixXd cholesky_vectors(static_cast<Eigen::Index>(pair_dim),
                                   max_rank);
  Eigen::Index num_vectors = 0;
  bool is_positive_semi_definite = true;

  while (num_vectors < max_rank) {
    // Indefiniteness shows up on the minimum residual diagonal, not on the
    // pivot: the pivot is the maximum and merely decays towards zero, which
    // hides a negative direction rather than exposing it.
    if (residual_diagonal.minCoeff() < -noise_floor) {
      is_positive_semi_definite = false;
      break;
    }

    Eigen::Index pivot = 0;
    const double pivot_value = residual_diagonal.maxCoeff(&pivot);
    if (pivot_value <= stop_threshold) {
      break;
    }

    Eigen::VectorXd column = supermatrix.col(pivot);
    for (Eigen::Index s = 0; s < num_vectors; ++s) {
      column -= cholesky_vectors.col(s) * cholesky_vectors(pivot, s);
    }
    column /= std::sqrt(pivot_value);

    residual_diagonal -= column.cwiseAbs2();
    cholesky_vectors.col(num_vectors) = column;
    ++num_vectors;
  }

  // Exhausting the rank bound with error left over means the input violated
  // the p<->q symmetry the bound assumes. The eigen path imposes that symmetry
  // per fragment instead of relying on it, so it still returns a usable
  // factorization.
  if (is_positive_semi_definite && num_vectors == max_rank &&
      residual_diagonal.maxCoeff() > stop_threshold) {
    is_positive_semi_definite = false;
  }

  if (!is_positive_semi_definite) {
    QDK_LOGGER().debug(
        "cholesky_decompose_two_body: supermatrix is not positive "
        "semi-definite for num_orbitals={}, falling back to "
        "eigen_decompose_two_body.",
        norb);
    return eigen_decompose_two_body(two_body_integrals, norb,
                                    truncation_threshold);
  }

  return fragments_from_cholesky_vectors(cholesky_vectors.leftCols(num_vectors),
                                         norb, truncation_threshold);
}

std::vector<TwoBodyFragment> DoubleFactorizer::_compute_fragments(
    const data::Hamiltonian& hamiltonian, std::size_t norb,
    double truncation_threshold, DoubleFactorizationMethod method) const {
  QDK_LOG_TRACE_ENTERING();

  using qdk::chemistry::data::CholeskyHamiltonianContainer;

  // Stored three-center integrals already are the first factorization, so
  // reusing them skips both expanding them into a dense norb^4 tensor and
  // re-deriving a decomposition that is already on hand.
  if (method == DoubleFactorizationMethod::Cholesky &&
      hamiltonian.has_container_type<CholeskyHamiltonianContainer>()) {
    const Eigen::MatrixXd& three_center =
        hamiltonian.get_container<CholeskyHamiltonianContainer>()
            .get_three_center_integrals()
            .first;

    if (static_cast<std::size_t>(three_center.rows()) == norb * norb) {
      return fragments_from_cholesky_vectors(three_center, norb,
                                             truncation_threshold);
    }

    QDK_LOGGER().debug(
        "double_factorizer: stored three-center integrals have {} rows "
        "but num_orbitals={} implies {}, decomposing the dense tensor instead.",
        three_center.rows(), norb, norb * norb);
  }

  const Eigen::VectorXd& g_aaaa =
      std::get<0>(hamiltonian.get_two_body_integrals());

  return method == DoubleFactorizationMethod::Cholesky
             ? cholesky_decompose_two_body(g_aaaa, norb, truncation_threshold)
             : eigen_decompose_two_body(g_aaaa, norb, truncation_threshold);
}

std::shared_ptr<data::Hamiltonian> DoubleFactorizer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  QDK_LOG_TRACE_ENTERING();

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
        ": the Hamiltonian carries no two-body integrals to "
        "factorize.");
  }

  const std::string method_name = _settings->get<std::string>("method");
  const DoubleFactorizationMethod method = method_from_string(method_name);

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");

  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());

  // Deriving norb from the one-body block rather than the two-body tensor
  // keeps implementations that never materialize a dense tensor viable.
  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  auto fragments =
      _compute_fragments(*hamiltonian, norb, truncation_threshold, method);

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
