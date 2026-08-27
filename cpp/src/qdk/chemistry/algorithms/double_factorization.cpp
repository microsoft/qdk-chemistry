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

  const std::size_t pair_dim = norb * norb;
  const std::size_t expected = pair_dim * pair_dim;
  if (static_cast<std::size_t>(two_body_integrals.size()) != expected) {
    throw std::invalid_argument(
        "eigen_decompose_two_body: expected a tensor of norb^4 = " +
        std::to_string(expected) +
        " elements for norb = " + std::to_string(norb) + ", got " +
        std::to_string(two_body_integrals.size()) + ".");
  }

  const Eigen::Index pair_size = static_cast<Eigen::Index>(pair_dim);
  const Eigen::Index orbitals = static_cast<Eigen::Index>(norb);

  // g_pqrs is flattened as (pq)*norb^2 + (rs), which is already the row-major
  // layout of the (pq),(rs) supermatrix. Symmetrizing out of the mapped input
  // (rather than in place) both guards against numerical noise in the caller's
  // tensor and avoids aliasing the destination.
  const Eigen::Map<const RowMajorMatrix> raw_supermatrix(
      two_body_integrals.data(), pair_size, pair_size);
  Eigen::MatrixXd supermatrix =
      0.5 * (raw_supermatrix + raw_supermatrix.transpose());

  Eigen::MatrixXd supermatrix_eigenvectors = supermatrix;
  Eigen::VectorXd supermatrix_eigenvalues(pair_dim);
  // syev overwrites its input with the eigenvectors and reads only the lower
  // triangle.
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

    // Column n of the eigenvector matrix is contiguous and indexed by p*norb+q,
    // so it maps directly onto an norb x norb row-major matrix. g_pqrs =
    // g_qprs makes it symmetric; symmetrize defensively against degenerate
    // subspaces and numerical noise. syev then overwrites it with U.
    const Eigen::Map<const RowMajorMatrix> raw_fragment(
        supermatrix_eigenvectors.data() +
            static_cast<Eigen::Index>(n) * pair_size,
        orbitals, orbitals);
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

    const double eps_abs_sum = fragment.eps.array().abs().sum();
    fragment.lambda_df = 0.5 * eps_abs_sum * eps_abs_sum;

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
        "DoubleFactorizer currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }
  if (!hamiltonian->has_two_body_integrals()) {
    throw std::invalid_argument(
        "DoubleFactorizer: the Hamiltonian carries no two-body "
        "integrals to factorize.");
  }

  const double truncation_threshold =
      _settings->get<double>("truncation_threshold");

  auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

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
        " discarded every fragment, which would leave the factorized "
        "Hamiltonian with no two-body term at all. Lower the threshold.");
  }

  // Container layout: one rank per fragment, B = norb bases, a single copy.
  //   U[r,b,p] = U^r_pb,  W[r,b,0] = eps^r_b
  // Each rank owns a contiguous slice of both flat buffers, so rank r's U
  // slice viewed row-major as (B x N) is exactly fragment.U transposed.
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
    const Eigen::Index orbitals = static_cast<Eigen::Index>(norb);

    signs(rank) = fragment.sign;
    w_matrices.segment(rank * bases, bases) = fragment.eps;
    Eigen::Map<RowMajorMatrix>(u_matrices.data() + rank * bases * orbitals,
                               bases, orbitals) = fragment.U.transpose();
  }

  // WB and the energy gap are block-encoding parameters that a plain
  // factorization neither reads nor produces.
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
