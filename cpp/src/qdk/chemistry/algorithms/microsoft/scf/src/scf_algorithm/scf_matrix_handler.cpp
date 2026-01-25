// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf_matrix_handler.h"

#include <cmath>
#include <lapack.hh>
#include <qdk/chemistry/utils/logger.hpp>

#include "util/timer.h"

namespace qdk::chemistry::scf {

namespace impl {
class ROHFMatrixHandler {
 public:
  explicit ROHFMatrixHandler() {
    QDK_LOG_TRACE_ENTERING();
    effective_Fock_matrix_ = RowMajorMatrix::Zero(0, 0);
    total_density_matrix_ = RowMajorMatrix::Zero(0, 0);
  }

  void receive_F_P_matrices(const RowMajorMatrix& F, RowMajorMatrix& P) {
    QDK_LOG_TRACE_ENTERING();
    int num_atomic_orbitals = static_cast<int>(F.cols());
    effective_Fock_matrix_ = RowMajorMatrix::Zero(
        num_atomic_orbitals, num_atomic_orbitals);
    total_density_matrix_ = RowMajorMatrix::Zero(
        num_atomic_orbitals, num_atomic_orbitals);
  }

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  const RowMajorMatrix& get_fock_matrix() {
    QDK_LOG_TRACE_ENTERING();
    return effective_Fock_matrix_;
  }

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  RowMajorMatrix& get_density_matrix() {
    QDK_LOG_TRACE_ENTERING();
    return total_density_matrix_;
  }

 private:
  RowMajorMatrix effective_Fock_matrix_;
  RowMajorMatrix total_density_matrix_;
};

}  // namespace impl

ROHFMatrixHandler::ROHFMatrixHandler() {
  QDK_LOG_TRACE_ENTERING();
  handler_impl_ = std::make_unique<impl::ROHFMatrixHandler>();
}

void ROHFMatrixHandler::receive_F_P_matrices(const RowMajorMatrix& F,
                                             RowMajorMatrix& P) {
  handler_impl_->receive_F_P_matrices(F, P);
}

const RowMajorMatrix& ROHFMatrixHandler::get_fock_matrix() {
  return handler_impl_->get_fock_matrix();
}

RowMajorMatrix& ROHFMatrixHandler::get_density_matrix() {
  return handler_impl_->get_density_matrix();
}

}  // namespace qdk::chemistry::scf
