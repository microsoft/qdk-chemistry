// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "rohf_matrix_handler.h"

#include <cmath>
#include <iostream>
#include <lapack.hh>
#include <qdk/chemistry/utils/logger.hpp>

#include "util/timer.h"

namespace qdk::chemistry::scf {

namespace impl {
class ROHFMatrixHandler {
 public:
  explicit ROHFMatrixHandler() {
    QDK_LOG_TRACE_ENTERING();
    effective_F_ = RowMajorMatrix::Zero(0, 0);
    total_P_ = RowMajorMatrix::Zero(0, 0);
  }

  /**
   * @brief Build ROHF effective Fock matrix and total Density matrix
   *
   * @param[in] F Spin-blocked Fock matrix
   * @param[in] C Orbital coefficients matrix
   * @param[in] P Spin-blocked density matrix
   * @param[in] nelec_alpha Number of alpha electrons
   * @param[in] nelec_beta Number of beta electrons, less than nelec_alpha
   */
  void build_ROHF_F_P_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta) {
    QDK_LOG_TRACE_ENTERING();
    int num_molecular_orbitals = static_cast<int>(F.cols());

    if (C.isZero()) {
      effective_F_ = F.block(0, 0, num_molecular_orbitals, num_molecular_orbitals);
    } else {
        RowMajorMatrix F_up_mo =
          RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
      RowMajorMatrix F_dn_mo = F_up_mo;
      RowMajorMatrix effective_F_mo = F_up_mo;

      F_up_mo.noalias() =
          C.transpose() *
          F.block(0, 0, num_molecular_orbitals, num_molecular_orbitals) * C;
      F_dn_mo.noalias() =
          C.transpose() *
          F.block(num_molecular_orbitals, 0, num_molecular_orbitals,
                  num_molecular_orbitals) *
          C;

      auto average_block = [&](int row, int col, int rows, int cols) {
        if (rows <= 0 || cols <= 0) return;
        effective_F_mo.block(row, col, rows, cols).noalias() =
            0.5 * (F_up_mo.block(row, col, rows, cols) +
                   F_dn_mo.block(row, col, rows, cols));
      };
      auto copy_block = [&](const RowMajorMatrix& src, int row, int col, int rows,
                            int cols) {
        if (rows <= 0 || cols <= 0) return;
        effective_F_mo.block(row, col, rows, cols) =
            src.block(row, col, rows, cols);
      };

      int nd = nelec_beta;
      int ns = nelec_alpha - nelec_beta;
      int nv = num_molecular_orbitals - nelec_alpha;
      average_block(0, 0, nd, nd);               // F_c^{dd}
      average_block(0, nd + ns, nd, nv);         // F_c^{dv}
      average_block(nd + ns, 0, nv, nd);         // F_c^{vd}
      average_block(nd + ns, nd + ns, nv, nv);   // F_c^{vv}
      average_block(nd, nd, ns, ns);             // F_c^{ss}
      copy_block(F_dn_mo, 0, nd, nd, ns);        // F_dn^{ds}
      copy_block(F_dn_mo, nd, 0, ns, nd);        // F_dn^{sd}
      copy_block(F_up_mo, nd, nd + ns, ns, nv);  // F_up^{sv}
      copy_block(F_up_mo, nd + ns, nd, nv, ns);  // F_up^{vs}

      effective_F_ =
          RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
      effective_F_.noalias() = C * effective_F_mo * C.transpose();
      effective_F_ = 0.5 * (effective_F_ + effective_F_.transpose().eval());
    }
    

    total_P_ = P.block(0, 0, num_molecular_orbitals, num_molecular_orbitals) +
               P.block(num_molecular_orbitals, 0, num_molecular_orbitals,
                       num_molecular_orbitals);
  }

  /**
   * @brief Get reference to Fock matrix
   *
   * @return Reference to Fock matrix
   */
  const RowMajorMatrix& get_fock_matrix() {
    QDK_LOG_TRACE_ENTERING();
    return effective_F_;
  }

  const RowMajorMatrix& get_density_matrix() {
    QDK_LOG_TRACE_ENTERING();
    return total_P_;
  }

  /**
   * @brief Get reference to Density matrix
   *
   * @return Reference to Density matrix
   */
  RowMajorMatrix& density_matrix() {
    QDK_LOG_TRACE_ENTERING();
    return total_P_;
  }

  /**
   * @brief Update density matrix from total density matrix
   *
   * @param[out] P Density matrix to update
   * @param[in] C Orbital coefficients matrix
   * @param[in] nelec_alpha Number of alpha electrons
   * @param[in] nelec_beta Number of beta electrons, less than nelec_alpha
   */
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             int nelec_alpha, int nelec_beta) {
    QDK_LOG_TRACE_ENTERING();
    int num_atomic_orbitals = C.rows();

    auto build_density = [&](auto& target, int n_occ) {
      if (n_occ <= 0) {
        target.setZero();
        return;
      }
      target.noalias() = C.block(0, 0, num_atomic_orbitals, n_occ) *
                         C.block(0, 0, num_atomic_orbitals, n_occ).transpose();
    };

    auto P_alpha = P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    auto P_beta = P.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                          num_atomic_orbitals);
    build_density(P_alpha, nelec_alpha);
    build_density(P_beta, nelec_beta);
  }

 private:
  RowMajorMatrix effective_F_;
  RowMajorMatrix total_P_;
};

}  // namespace impl

ROHFMatrixHandler::ROHFMatrixHandler() {
  QDK_LOG_TRACE_ENTERING();
  handler_impl_ = std::make_unique<impl::ROHFMatrixHandler>();
}

ROHFMatrixHandler::~ROHFMatrixHandler() noexcept = default;

void ROHFMatrixHandler::build_ROHF_F_P_matrix(const RowMajorMatrix& F,
                                              const RowMajorMatrix& C,
                                              const RowMajorMatrix& P,
                                              int nelec_alpha, int nelec_beta) {
  handler_impl_->build_ROHF_F_P_matrix(F, C, P, nelec_alpha, nelec_beta);
}

const RowMajorMatrix& ROHFMatrixHandler::get_fock_matrix() {
  return handler_impl_->get_fock_matrix();
}

const RowMajorMatrix& ROHFMatrixHandler::get_density_matrix() {
  return handler_impl_->get_density_matrix();
}

RowMajorMatrix& ROHFMatrixHandler::density_matrix() {
  return handler_impl_->density_matrix();
}

// Implementation for updating density matrix from
// total density matrix and orbital coefficients
void ROHFMatrixHandler::update_density_matrix(RowMajorMatrix& P,
                                              const RowMajorMatrix& C,
                                              int nelec_alpha, int nelec_beta) {
  handler_impl_->update_density_matrix(P, C, nelec_alpha, nelec_beta);
}

}  // namespace qdk::chemistry::scf
