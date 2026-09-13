// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <lapack.hh>
#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace qdk::chemistry::algorithms {

std::shared_ptr<data::Hamiltonian> DoubleFactorization::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  QDK_LOG_TRACE_ENTERING();

  using qdk::chemistry::data::CholeskyHamiltonianContainer;
  using qdk::chemistry::data::FactorizedHamiltonianContainer;
  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  if (!hamiltonian) {
    throw std::invalid_argument(type_name() + ": hamiltonian is null");
  }
  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        type_name() + " currently only supports restricted Hamiltonians.");
  }
  if (!hamiltonian->has_container_type<CholeskyHamiltonianContainer>()) {
    throw std::invalid_argument(
        "double_factorization requires a CholeskyHamiltonianContainer with "
        "MO three-center factors in norb^2-by-naux pair layout.");
  }
  if (!hamiltonian->has_two_body_integrals()) {
    throw std::invalid_argument(
        type_name() +
        ": the Hamiltonian carries no two-body integrals to factorize.");
  }

  const Eigen::MatrixXd& h_alpha =
      std::get<0>(hamiltonian->get_one_body_integrals());
  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());
  if (norb == 0) {
    throw std::invalid_argument(type_name() +
                                ": norb must be greater than zero.");
  }

  const Eigen::MatrixXd& cholesky_vectors =
      hamiltonian->get_container<CholeskyHamiltonianContainer>()
          .get_three_center_integrals()
          .first;

  const std::size_t pair_dim = norb * norb;
  if (static_cast<std::size_t>(cholesky_vectors.rows()) != pair_dim) {
    throw std::invalid_argument(
        "double_factorization: expected norb^2 = " + std::to_string(pair_dim) +
        " rows for norb = " + std::to_string(norb) + ", got " +
        std::to_string(cholesky_vectors.rows()) +
        "; MO three-center factors must use pair order p*norb+q with one "
        "auxiliary vector per column.");
  }

  if (!cholesky_vectors.allFinite()) {
    throw std::invalid_argument(
        "double_factorization: cholesky_vectors contains a non-finite value.");
  }

  const Eigen::Index num_orbitals = static_cast<Eigen::Index>(norb);
  const Eigen::Index num_ranks = cholesky_vectors.cols();
  Eigen::VectorXd u_matrices(num_ranks * num_orbitals * num_orbitals);
  Eigen::VectorXd w_matrices(num_ranks * num_orbitals);

  // syev leaves the eigenvectors as columns in column-major order, which is
  // bytewise the row-major [basis, orbital] layout the container stores, so
  // each rank is diagonalized straight into its slot.
  for (Eigen::Index q = 0; q < num_ranks; ++q) {
    const Eigen::Map<const RowMajorMatrix> pair_matrix(
        cholesky_vectors.col(q).data(), num_orbitals, num_orbitals);
    Eigen::Map<Eigen::MatrixXd> rotation(
        u_matrices.data() + q * num_orbitals * num_orbitals, num_orbitals,
        num_orbitals);

    const double asymmetry =
        (pair_matrix - pair_matrix.transpose()).cwiseAbs().maxCoeff();
    const double scale = pair_matrix.cwiseAbs().maxCoeff();
    if (asymmetry > 1e-8 * std::max(scale, 1.0)) {
      throw std::invalid_argument("double_factorization: Cholesky vector " +
                                  std::to_string(q) +
                                  " is not symmetric in its orbital pair.");
    }
    rotation = 0.5 * (pair_matrix + pair_matrix.transpose());

    const int64_t info = lapack::syev(
        lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
        rotation.data(), static_cast<int64_t>(norb),
        w_matrices.data() + q * num_orbitals);
    if (info != 0) {
      throw std::runtime_error(
          "double_factorization: LAPACK syev failed to diagonalize (info=" +
          std::to_string(info) + ").");
    }
  }

  QDK_LOGGER().debug(
      "double_factorization: num_orbitals={}, factorized {} supplied "
      "fragments.",
      norb, num_ranks);

  if (num_ranks == 0) {
    throw std::invalid_argument(
        "double_factorization: the Hamiltonian contains no MO Cholesky "
        "vectors.");
  }

  const Eigen::MatrixXd wb_matrix = Eigen::MatrixXd::Zero(num_ranks, 1);

  const Eigen::MatrixXd inactive_fock =
      hamiltonian->has_inactive_fock_matrix()
          ? hamiltonian->get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<FactorizedHamiltonianContainer>(
      h_alpha, u_matrices, w_matrices, wb_matrix, hamiltonian->get_orbitals(),
      hamiltonian->get_core_energy(), inactive_fock, hamiltonian->get_type());

  return std::make_shared<data::Hamiltonian>(std::move(container));
}

}  // namespace qdk::chemistry::algorithms
