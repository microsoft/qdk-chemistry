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

}  // namespace

std::vector<TwoBodyFragment> eigen_decompose_two_body(
    const Eigen::VectorXd& two_body_integrals, std::size_t norb,
    double truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  if (norb == 0) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: norb must be greater than zero.");
  }

  if (truncation_threshold < 0.0 || std::isnan(truncation_threshold)) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: truncation_threshold must be "
        "non-negative, got " +
        std::to_string(truncation_threshold) + ".");
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

  if (!two_body_integrals.allFinite()) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: two_body_integrals contains a non-finite "
        "value (NaN or infinity).");
  }

  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);
  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);

  // Reshape g_ijkl into the (ij),(kl) supermatrix.
  const Eigen::Map<const RowMajorMatrix> raw_supermatrix(
      two_body_integrals.data(), pair_size, pair_size);

  // Assumes chemist permutation symmetry: averaging imposes the (pq)<->(rs)
  // and p<->q generators rather than checking them, and the rest follow.
  Eigen::MatrixXd supermatrix_eigenvectors =
      0.5 * (raw_supermatrix + raw_supermatrix.transpose());
  Eigen::VectorXd supermatrix_eigenvalues(pair_dim);

  // Dense diagonalization costs O(norb^6) and materializes all norb^2
  // eigenpairs. An iterative solver that recovers only a limited number of
  // leading ranks will be added later.
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

    // Reshape the eigenvector into an norb x norb matrix. Averaging enforces
    // the p<->q generator.
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

  // Both accessors return tuples of references into the container, so binding
  // the alpha element alone leaves it valid after the tuple expires.
  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());
  const Eigen::VectorXd& g_aaaa =
      std::get<0>(hamiltonian->get_two_body_integrals());

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  auto fragments = eigen_decompose_two_body(g_aaaa, norb, truncation_threshold);

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

  // R = number of fragments, B = norb bases, C = 1.
  //
  // Every fragment keeps all norb bases even when some fragment eigenvalues in
  // `eps` are negligible. The container stores `u_matrices` as a rectangular
  // R x B x norb tensor, so B has to be uniform across ranks; dropping bases
  // per fragment would need either padding, which saves nothing, or ragged
  // storage, which the container does not support. Zero-weight bases
  // contribute nothing to the reconstructed tensor or to Lambda, so the only
  // cost is storage.
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
