// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "rohf_diis.h"

#include <lapack.hh>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <vector>

#include "../scf/scf_impl.h"
#include "util/macros.h"

namespace qdk::chemistry::scf {

namespace impl {

/**
 * @brief Implementation class for ROHFDIIS
 *
 * This class hides the implementation details of ROHF-specific density and
 * Fock matrix construction, following the PImpl idiom.
 */
class ROHFDIIS {
 public:
  /**
   * @brief Construct the ROHF DIIS implementation
   * @param num_atomic_orbitals Number of atomic orbitals for matrix sizing
   */
  explicit ROHFDIIS(int num_atomic_orbitals)
      : effective_F_(
            RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals)),
        total_P_(
            RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals)) {}

  /**
   * @brief Update the spin-blocked density matrix after solving the
   * eigenproblem
   * @param P Spin-blocked density matrix to overwrite
   * @param C Molecular orbital coefficients used to rebuild densities
   * @param nelec_alpha Number of alpha electrons
   * @param nelec_beta Number of beta electrons
   */
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             int nelec_alpha, int nelec_beta) {
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

  /**
   * @brief Build cached ROHF Fock/density matrices from spin-blocked inputs
   * @param F Spin-blocked Fock matrix (alpha stacked on beta)
   * @param C Molecular orbital coefficient matrix
   * @param P Spin-blocked density matrix
   * @param nelec_alpha Number of alpha electrons
   * @param nelec_beta Number of beta electrons
   */
  void build_rohf_f_p_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta) {
    QDK_LOG_TRACE_ENTERING();
    const int num_atomic_orbitals = static_cast<int>(F.cols());

    RowMajorMatrix new_total =
        P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) +
        P.block(num_atomic_orbitals, 0, num_atomic_orbitals,
                num_atomic_orbitals);

    total_P_ = new_total;

    if (effective_F_.rows() != num_atomic_orbitals ||
        effective_F_.cols() != num_atomic_orbitals) {
      effective_F_ =
          RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
    }

    if (C.isZero()) {
      effective_F_.noalias() =
          F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
      return;
    } else {
      // Build the ROHF effective Fock matrix using the standard MO-based
      // construction The form of effective Fock matrix can be found in Guest
      // and Saunders (1974) and Plakhutin and Davidson (2014)
      const int num_molecular_orbitals = static_cast<int>(C.cols());
      RowMajorMatrix F_up_mo =
          RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
      RowMajorMatrix F_dn_mo = F_up_mo;
      RowMajorMatrix effective_F_mo = F_up_mo;

      F_up_mo.noalias() =
          C.transpose() *
          F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) * C;
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
      auto copy_block = [&](const RowMajorMatrix& src, int row, int col,
                            int rows, int cols) {
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

      // Get the effective Fock matrix in AO basis by C^{-T} F_mo C^{-1}
      // Solve C^T X = effective_F_mo and then C^T X = (C^{-T} F_mo)^T
      // to avoid forming C^{-1} explicitly.
      // Unlike blasxx, lapackxx only handles column-major matrices
      const int matrix_dim = num_molecular_orbitals;
      using ColMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic,
                                           Eigen::Dynamic, Eigen::ColMajor>;
      // Row-major storage implies the raw buffer already matches (C^T) in
      // column-major ordering, so this copy materializes C^T without an
      // explicit transpose.
      ColMajorMatrix Ct =
          Eigen::Map<const ColMajorMatrix>(C.data(), matrix_dim, C.rows());
      ColMajorMatrix temp_rhs = effective_F_mo;
      std::vector<int64_t> ipiv(matrix_dim);

      auto info = lapack::getrf(matrix_dim, matrix_dim, Ct.data(), matrix_dim,
                                ipiv.data());
      if (info != 0) {
        throw std::runtime_error("getrf failed while factorizing C^T");
      }

      info =
          lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                        matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
      if (info != 0) {
        throw std::runtime_error("getrs failed while solving C^T X = F_mo");
      }

      temp_rhs.transposeInPlace();  // get F_mo^T C^{-1}
      info =
          lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                        matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
      if (info != 0) {
        throw std::runtime_error("getrs failed while solving C^T X = M^T");
      }

      effective_F_ = temp_rhs.transpose();
      if (!effective_F_.isApprox(effective_F_.transpose())) {
        effective_F_ = 0.5 * (effective_F_ + effective_F_.transpose().eval());
      }
    }
  }

  /**
   * @brief Get the effective Fock matrix (const)
   */
  const RowMajorMatrix& get_effective_F() const { return effective_F_; }

  /**
   * @brief Get the total density matrix (const)
   */
  const RowMajorMatrix& get_total_P() const { return total_P_; }

  /**
   * @brief Get the total density matrix (mutable)
   */
  RowMajorMatrix& get_total_P() { return total_P_; }

 private:
  /// Cached effective Fock matrix expressed in the AO basis
  RowMajorMatrix effective_F_;
  /// Cached total density matrix (alpha + beta) exposed to DIIS
  RowMajorMatrix total_P_;
};

}  // namespace impl

ROHFDIIS::ROHFDIIS(const SCFContext& ctx, size_t subspace_size)
    : DIISBase(ctx, subspace_size),
      impl_(std::make_unique<impl::ROHFDIIS>(
          static_cast<int>(ctx.basis_set->num_atomic_orbitals))) {
  QDK_LOG_TRACE_ENTERING();
}

ROHFDIIS::~ROHFDIIS() noexcept = default;

const RowMajorMatrix& ROHFDIIS::get_active_fock(
    const SCFImpl& /*scf_impl*/) const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_effective_F();
}

RowMajorMatrix& ROHFDIIS::active_density(SCFImpl& /*scf_impl*/) {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_total_P();
}

const RowMajorMatrix& ROHFDIIS::get_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_effective_F();
}

const RowMajorMatrix& ROHFDIIS::get_density_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_total_P();
}

RowMajorMatrix& ROHFDIIS::density_matrix() {
  QDK_LOG_TRACE_ENTERING();
  return impl_->get_total_P();
}

void ROHFDIIS::update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                                     bool /*unrestricted*/, int nelec_alpha,
                                     int nelec_beta) {
  impl_->update_density_matrix(P, C, nelec_alpha, nelec_beta);
}

void ROHFDIIS::build_rohf_f_p_matrix(const RowMajorMatrix& F,
                                     const RowMajorMatrix& C,
                                     const RowMajorMatrix& P, int nelec_alpha,
                                     int nelec_beta) {
  impl_->build_rohf_f_p_matrix(F, C, P, nelec_alpha, nelec_beta);
}

}  // namespace qdk::chemistry::scf
