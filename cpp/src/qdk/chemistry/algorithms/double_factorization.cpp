// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <lapack.hh>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

namespace {

using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::unique_ptr<DoubleFactorizer> make_double_factorizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<DoubleFactorizer>();
}

/// Reject a two-electron tensor that lacks the chemist permutation symmetry
/// the decomposition assumes.
///
/// eigen_decompose_two_body projects onto the (pq)<->(rs) and p<->q symmetric
/// parts before diagonalizing. Those two permutations generate the full
/// 8-fold real-orbital symmetry, so checking them covers all of it. Without
/// this guard an asymmetric tensor is silently replaced by its projection,
/// which can differ from the input by an order-one relative error while every
/// other check passes.
///
/// The tolerance is relative to the largest element so the test does not
/// depend on the tensor's units. A 4-index integral transform loses only a
/// few digits, so the default leaves several orders of margin over round-off.
void validate_chemist_symmetry(
    const Eigen::Map<const RowMajorMatrix>& supermatrix, std::size_t norb,
    double symmetry_tolerance) {
  QDK_LOG_TRACE_ENTERING();

  const double max_abs_element = supermatrix.cwiseAbs().maxCoeff();
  const double tolerance = symmetry_tolerance * max_abs_element;
  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);

  const auto asymmetry_error = [&](const char* permutation, std::size_t p,
                                   std::size_t q, std::size_t r, std::size_t s,
                                   double deviation) {
    std::ostringstream message;
    message << std::scientific;
    message << "eigen_decompose_two_body: two_body_integrals is not symmetric"
            << " under " << permutation << ". g[" << p << "," << q << "," << r
            << "," << s << "] differs from its image by " << deviation
            << ", above the tolerance " << tolerance << " (symmetry_tolerance "
            << symmetry_tolerance << " times the largest element "
            << max_abs_element
            << "). The decomposition would silently return fragments for the"
               " symmetrized tensor instead.";
    return std::invalid_argument(message.str());
  };

  // (pq) <-> (rs): the supermatrix itself must be symmetric.
  Eigen::Index row = 0;
  Eigen::Index col = 0;
  const double pair_deviation =
      (supermatrix - supermatrix.transpose()).cwiseAbs().maxCoeff(&row, &col);
  if (pair_deviation > tolerance) {
    throw asymmetry_error("(pq)<->(rs)", static_cast<std::size_t>(row) / norb,
                          static_cast<std::size_t>(row) % norb,
                          static_cast<std::size_t>(col) / norb,
                          static_cast<std::size_t>(col) % norb, pair_deviation);
  }

  // p <-> q: rows (p,q) and (q,p) of the supermatrix must agree. Combined with
  // the check above this implies r <-> s, so the two together cover the full
  // 8-fold symmetry. Costs O(norb^4) against an O(norb^6) diagonalization.
  for (Eigen::Index p = 0; p < num_orbitals; ++p) {
    for (Eigen::Index q = p + 1; q < num_orbitals; ++q) {
      const double deviation = (supermatrix.row(p * num_orbitals + q) -
                                supermatrix.row(q * num_orbitals + p))
                                   .cwiseAbs()
                                   .maxCoeff(&row, &col);
      if (deviation > tolerance) {
        throw asymmetry_error("p<->q", static_cast<std::size_t>(p),
                              static_cast<std::size_t>(q),
                              static_cast<std::size_t>(col) / norb,
                              static_cast<std::size_t>(col) % norb, deviation);
      }
    }
  }
}

}  // namespace

std::vector<TwoBodyFragment> eigen_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold, double symmetry_tolerance) {
  QDK_LOG_TRACE_ENTERING();

  if (norb == 0) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: norb must be greater than zero.");
  }
  // NaN is rejected explicitly: it compares false against every eigenvalue
  // magnitude, so it would silently retain the whole decomposition instead of
  // truncating.
  if (truncation_threshold < 0.0 || std::isnan(truncation_threshold)) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: truncation_threshold must be "
        "non-negative, got " +
        std::to_string(truncation_threshold) + ".");
  }
  // NaN would likewise compare false against every deviation and disable the
  // symmetry check entirely.
  if (symmetry_tolerance < 0.0 || std::isnan(symmetry_tolerance)) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: symmetry_tolerance must be non-negative, "
        "got " +
        std::to_string(symmetry_tolerance) + ".");
  }

  const std::size_t pair_dim = norb * norb;
  const std::size_t expected = pair_dim * pair_dim;
  if (static_cast<std::size_t>(two_body_integrals.size()) != expected) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: expected norb^4 = " +
        std::to_string(expected) +
        " elements for norb = " + std::to_string(norb) + ", got " +
        std::to_string(two_body_integrals.size()) + ".");
  }
  // A non-finite entry would otherwise survive every check below. NaN
  // compares false against the symmetry tolerance, and it makes the sort
  // comparator |a| > |b| false in both directions, which is not a strict weak
  // ordering and is undefined behavior in std::sort. Infinities reach the
  // same place, since the symmetrization turns Inf - Inf into NaN.
  if (!two_body_integrals.allFinite()) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: two_body_integrals contains a non-finite "
        "value (NaN or infinity).");
  }

  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);
  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);

  // Reshape g_ijkl into the (ij),(kl) supermatrix. syev overwrites its input
  // with the eigenvectors, so the symmetrized copy is built directly in the
  // output buffer; the symmetrization is defensive against numerical noise in
  // the input tensor.
  const Eigen::Map<const RowMajorMatrix> raw_supermatrix(
      two_body_integrals.data(), pair_size, pair_size);

  validate_chemist_symmetry(raw_supermatrix, norb, symmetry_tolerance);

  Eigen::MatrixXd supermatrix_eigenvectors =
      0.5 * (raw_supermatrix + raw_supermatrix.transpose());

  Eigen::VectorXd supermatrix_eigenvalues(pair_dim);
  // Dense diagonalization costs O(norb^6) and materializes all norb^2
  // eigenpairs. An iterative solver that recovers only a limited number of
  // leading ranks will be added later, for decomposing large-size
  // Hamiltonians where the full spectrum is neither affordable nor needed.
  // syev reads only the lower triangle.
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
  // Within a degenerate block the eigenvector basis LAPACK returns is
  // arbitrary, so eps and lambda_df are not fixed by the tensor alone. A
  // stable sort would not change that: the freedom is in the basis, not the
  // ordering. The reconstructed tensor is invariant either way.
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

    // Reshape the eigenvector into an norb x norb matrix. For a
    // non-degenerate eigenvalue this matrix is automatically symmetric
    // because g_ijkl = g_jikl; symmetrize defensively to guard against
    // degenerate subspaces / numerical noise.
    const Eigen::Map<const RowMajorMatrix> raw_fragment(
        supermatrix_eigenvectors.data() +
            static_cast<Eigen::Index>(n) * pair_size,
        num_orbitals, num_orbitals);
    Eigen::MatrixXd fragment_matrix =
        0.5 * (raw_fragment + raw_fragment.transpose());

    Eigen::VectorXd fragment_eigenvalues(norb);
    const int64_t fragment_info =
        lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower,
                     static_cast<int64_t>(norb), fragment_matrix.data(),
                     static_cast<int64_t>(norb), fragment_eigenvalues.data());
    if (fragment_info != 0) {
      throw std::runtime_error(
          "eigen_decompose_two_body: LAPACK syev failed to diagonalize a "
          "fragment matrix (info=" +
          std::to_string(fragment_info) + ").");
    }

    TwoBodyFragment fragment;
    fragment.sign = (eigenvalue >= 0.0) ? 1.0 : -1.0;
    fragment.eps = std::sqrt(std::abs(eigenvalue)) * fragment_eigenvalues;
    fragment.U = std::move(fragment_matrix);

    // Contribution to the block-encoding 1-norm, 1/4 (sum_b |eps_b|)^2,
    // matching FactorizedHamiltonianContainer::get_lambda() (Eq. 34) and
    // von Burg 2021 Eq. 16. Patel 2025 Eq. 17 writes the same quantity as
    // 1/2 (sum_b |eps_b|)^2, but in a convention whose two-body operator
    // carries no 1/2, so its coefficients are ours scaled by 1/sqrt(2).
    const double eps_abs_sum = fragment.eps.array().abs().sum();
    fragment.lambda_df = 0.25 * eps_abs_sum * eps_abs_sum;

    fragments.push_back(std::move(fragment));
  }

  return fragments;
}

std::shared_ptr<data::Hamiltonian> DoubleFactorizer::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  QDK_LOG_TRACE_ENTERING();

  using qdk::chemistry::data::FactorizedHamiltonianContainer;

  if (!hamiltonian) {
    throw std::invalid_argument("DoubleFactorizer: hamiltonian is null");
  }
  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        "DoubleFactorizer currently only supports restricted Hamiltonians.");
  }
  if (!hamiltonian->has_two_body_integrals()) {
    throw std::invalid_argument(
        "The Hamiltonian carries no two-body integrals to factorize.");
  }

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");
  const double symmetry_tolerance =
      _settings->get<double>("symmetry_tolerance");

  auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  auto fragments = eigen_decompose_two_body(g_aaaa, norb, truncation_threshold,
                                            symmetry_tolerance);

  QDK_LOGGER().debug(
      "DoubleFactorizer: num_orbitals={}, truncation_threshold={}, "
      "retained {} of {} candidate fragments.",
      norb, truncation_threshold, fragments.size(), norb * norb);

  if (fragments.empty()) {
    throw std::invalid_argument(
        "DoubleFactorizer: truncation_threshold=" +
        std::to_string(truncation_threshold) +
        " leaves the factorized Hamiltonian with no two-body term at all.");
  }

  // R = number of fragments, B = norb bases, C = 1
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

  // WB and the energy gap are not produced by the plain factorization.
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
