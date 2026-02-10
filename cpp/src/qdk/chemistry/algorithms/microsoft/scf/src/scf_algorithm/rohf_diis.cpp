// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "rohf_diis.h"

#include <qdk/chemistry/utils/logger.hpp>

#include "util/macros.h"
#include "../scf/scf_impl.h"

namespace qdk::chemistry::scf {

ROHFDIIS::ROHFDIIS(const SCFContext& ctx, size_t subspace_size)
    : DIISBase(ctx, true, subspace_size) {
  QDK_LOG_TRACE_ENTERING();
  const int num_atomic_orbitals =
      static_cast<int>(ctx.basis_set->num_atomic_orbitals);
  effective_F_ = RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
  total_P_ = RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
}

const RowMajorMatrix& ROHFDIIS::get_active_fock(
    const SCFImpl& /*scf_impl*/) const {
  QDK_LOG_TRACE_ENTERING();
  return effective_F_;
}

RowMajorMatrix& ROHFDIIS::active_density(SCFImpl& /*scf_impl*/) {
  QDK_LOG_TRACE_ENTERING();
  return total_P_;
}

const RowMajorMatrix& ROHFDIIS::get_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return effective_F_;
}

const RowMajorMatrix& ROHFDIIS::get_density_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return total_P_;
}

RowMajorMatrix& ROHFDIIS::density_matrix() {
  QDK_LOG_TRACE_ENTERING();
  return total_P_;
}

void ROHFDIIS::update_density_matrix(RowMajorMatrix& P,
                                     const RowMajorMatrix& C,
                                     bool /*unrestricted*/, int nelec_alpha,
                                     int nelec_beta) {
  QDK_LOG_TRACE_ENTERING();
  const int num_atomic_orbitals = static_cast<int>(C.rows());

  auto build_density = [&](auto&& target, int n_occ) {
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

void ROHFDIIS::build_rohf_f_p_matrix(const RowMajorMatrix& F,
                                     const RowMajorMatrix& C,
                                     const RowMajorMatrix& P,
                                     int nelec_alpha, int nelec_beta) {
  QDK_LOG_TRACE_ENTERING();
  const int num_atomic_orbitals = static_cast<int>(F.cols());

  RowMajorMatrix new_total =
      P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) +
      P.block(num_atomic_orbitals, 0, num_atomic_orbitals, num_atomic_orbitals);
  bool density_changed = true;
  if (total_P_.rows() == num_atomic_orbitals &&
      total_P_.cols() == num_atomic_orbitals) {
    density_changed = !total_P_.isApprox(new_total);
  }
  if (!density_changed) {
    return;
  }

  total_P_ = new_total;

  if (effective_F_.rows() != num_atomic_orbitals ||
      effective_F_.cols() != num_atomic_orbitals) {
    effective_F_ = RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
  }

  if (C.isZero()) {
    effective_F_.noalias() =
        F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    return;
  } else {
    const int num_molecular_orbitals = static_cast<int>(C.cols());
    RowMajorMatrix F_up_mo =
        RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
    RowMajorMatrix F_dn_mo = F_up_mo;
    RowMajorMatrix effective_F_mo = F_up_mo;

    F_up_mo.noalias() = C.transpose() *
                        F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) *
                        C;
    F_dn_mo.noalias() = C.transpose() *
                        F.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                                num_atomic_orbitals) *
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

    const int nd = nelec_beta;
    const int ns = nelec_alpha - nelec_beta;
    const int nv = num_molecular_orbitals - nelec_alpha;

    average_block(0, 0, nd, nd);
    average_block(0, nd + ns, nd, nv);
    average_block(nd + ns, 0, nv, nd);
    average_block(nd + ns, nd + ns, nv, nv);
    average_block(nd, nd, ns, ns);
    copy_block(F_dn_mo, 0, nd, nd, ns);
    copy_block(F_dn_mo, nd, 0, ns, nd);
    copy_block(F_up_mo, nd, nd + ns, ns, nv);
    copy_block(F_up_mo, nd + ns, nd, nv, ns);

    RowMajorMatrix C_inv = C.inverse();
    effective_F_.noalias() = C_inv.transpose() * effective_F_mo * C_inv;
    effective_F_ = 0.5 * (effective_F_ + effective_F_.transpose().eval());
  }
}

}  // namespace qdk::chemistry::scf
