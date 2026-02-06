// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "diis_base.h"

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/utils/logger.hpp>

#include <deque>
#include <limits>
#include <stdexcept>

#include "../scf/scf_impl.h"
#include "util/macros.h"

namespace qdk::chemistry::scf {

namespace impl {

class DIIS {
 public:
  explicit DIIS(const SCFContext& ctx, bool rohf_enabled,
                const size_t subspace_size);
  void iterate(const RowMajorMatrix& P, const RowMajorMatrix& F,
               const RowMajorMatrix& S);

  const RowMajorMatrix& get_extrapolated_fock() const { return F_extrapolated_; }
  double get_diis_error() const { return diis_error_; }

 private:
  void extrapolate_(const RowMajorMatrix& x, const RowMajorMatrix& error,
                    RowMajorMatrix& x_diis);
  void delete_oldest_();
  void apply_level_shift_(const RowMajorMatrix& F, const RowMajorMatrix& P,
                          const RowMajorMatrix& S, RowMajorMatrix& F_ls,
                          const double mu) const;

  bool rohf_enabled_;
  const SCFContext& ctx_;
  size_t subspace_size_;
  std::deque<RowMajorMatrix> hist_;
  std::deque<RowMajorMatrix> errors_;
  RowMajorMatrix B_;
  RowMajorMatrix F_extrapolated_;
  double diis_error_ = std::numeric_limits<double>::infinity();
};

DIIS::DIIS(const SCFContext& ctx, bool rohf_enabled, const size_t subspace_size)
    : rohf_enabled_(rohf_enabled), ctx_(ctx), subspace_size_(subspace_size) {
  QDK_LOG_TRACE_ENTERING();
  if (subspace_size <= 0) {
    throw std::invalid_argument("subspace_size must be greater than 0");
  }
}

void DIIS::iterate(const RowMajorMatrix& P, const RowMajorMatrix& F,
                   const RowMajorMatrix& S) {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;

  int num_atomic_orbitals = ctx_.basis_set->num_atomic_orbitals;
  int num_density_matrices = (ctx_.cfg->unrestricted || rohf_enabled_) ? 2 : 1;
  int num_orbital_sets = ctx_.cfg->unrestricted ? 2 : 1;

  RowMajorMatrix error = RowMajorMatrix::Zero(
      num_orbital_sets * num_atomic_orbitals, num_atomic_orbitals);
  diis_error_ = SCFAlgorithm::calculate_og_error_(F, P, S, error,
                                                  num_orbital_sets);

  RowMajorMatrix F_extrapolated = RowMajorMatrix::Zero(
      num_orbital_sets * num_atomic_orbitals, num_atomic_orbitals);
  if (cfg->scf_algorithm.level_shift > 0.0) {
    double mu = cfg->scf_algorithm.level_shift;
    apply_level_shift_(F, P, S, F_extrapolated, mu);
  } else {
    extrapolate_(F, error, F_extrapolated);
  }

  F_extrapolated_ = F_extrapolated;
}

void DIIS::extrapolate_(const RowMajorMatrix& x, const RowMajorMatrix& error,
                        RowMajorMatrix& x_diis) {
  QDK_LOG_TRACE_ENTERING();
  x_diis = x;
  if (hist_.size() == subspace_size_) delete_oldest_();
  hist_.push_back(x);
  errors_.push_back(error);

  size_t n = hist_.size();
  RowMajorMatrix B_old = B_;
  B_ = RowMajorMatrix::Zero(n, n);
  if (n > 1) {
    B_.block(0, 0, n - 1, n - 1) = B_old;
  }

  for (size_t i = 0; i < n; i++) {
    B_(i, n - 1) = B_(n - 1, i) = errors_[i].cwiseProduct(errors_[n - 1]).sum();
  }

  for (;;) {
    size_t rank = hist_.size() + 1;
    RowMajorMatrix A(rank, rank);
    A.col(0).setConstant(-1.0);
    A.row(0).setConstant(-1.0);
    A(0, 0) = 0.0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rank);
    rhs[0] = -1.0;

    double b_max = B_.maxCoeff();
    if (b_max == 0.0) {
      x_diis = x;
      return;
    }

    A.block(1, 1, rank - 1, rank - 1) =
        B_.block(0, 0, rank - 1, rank - 1) / b_max;

    Eigen::ColPivHouseholderQR<RowMajorMatrix> qr = A.colPivHouseholderQr();
    Eigen::VectorXd c = qr.solve(rhs);
    double absdet = qr.absDeterminant();

    constexpr double diis_linear_dependence_threshold = 1e-12;
    if (absdet < diis_linear_dependence_threshold) {
      delete_oldest_();
      rank = hist_.size() + 1;
      if (rank <= 1) {
        x_diis = x;
        return;
      }
      continue;
    } else {
      x_diis.setZero();
      for (size_t i = 0; i < hist_.size(); i++) {
        x_diis += c[i + 1] * hist_[i];
      }
      break;
    }
  }
}

void DIIS::delete_oldest_() {
  QDK_LOG_TRACE_ENTERING();
  if (hist_.empty()) return;
  hist_.pop_front();
  errors_.pop_front();
  size_t sz = B_.rows();
  if (sz > 1) {
    Eigen::MatrixXd tmp = B_.block(1, 1, sz - 1, sz - 1);
    B_ = tmp;
  } else {
    B_.resize(0, 0);
  }
}

void DIIS::apply_level_shift_(const RowMajorMatrix& F, const RowMajorMatrix& P,
                              const RowMajorMatrix& S, RowMajorMatrix& F_ls,
                              const double mu) const {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;
  int num_atomic_orbitals = static_cast<int>(S.cols());
  int num_density_matrices = cfg->unrestricted ? 2 : 1;

  RowMajorMatrix SPS = RowMajorMatrix::Zero(
      num_density_matrices * num_atomic_orbitals, num_atomic_orbitals);
  RowMajorMatrix SP =
      RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);

  for (int spin = 0; spin < num_density_matrices; spin++) {
    SP.noalias() =
        S * Eigen::Map<const RowMajorMatrix>(
                P.data() + spin * num_atomic_orbitals * num_atomic_orbitals,
                num_atomic_orbitals, num_atomic_orbitals);
    SPS.block(spin * num_atomic_orbitals, 0, num_atomic_orbitals,
              num_atomic_orbitals)
        .noalias() = SP * S;
  }

  const double* Fdata = F.data();
  double* F_lsdata = F_ls.data();
  const double* SPSdata = SPS.data();
  const double* Sdata = S.data();

  for (int i = 0;
       i < num_density_matrices * num_atomic_orbitals * num_atomic_orbitals;
       i++) {
    F_lsdata[i] = Fdata[i] +
                  (Sdata[i % (num_atomic_orbitals * num_atomic_orbitals)] -
                   SPSdata[i]) *
                      mu;
  }
}

}  // namespace impl

DIISBase::DIISBase(const SCFContext& ctx, bool rohf_enabled,
                   const size_t subspace_size)
    : SCFAlgorithm(ctx, rohf_enabled),
      diis_impl_(
          std::make_unique<impl::DIIS>(ctx, rohf_enabled, subspace_size)) {
  QDK_LOG_TRACE_ENTERING();
}

DIISBase::~DIISBase() noexcept = default;

void DIISBase::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  before_diis_iteration(scf_impl);

  const RowMajorMatrix& F = get_active_fock(scf_impl);
  RowMajorMatrix& working_density = active_density(scf_impl);

  auto& C = scf_impl.orbitals_matrix();
  const auto& S = scf_impl.overlap();
  const auto& X = scf_impl.get_orthogonalization_matrix();
  auto& eigenvalues = scf_impl.eigenvalues();

  diis_impl_->iterate(working_density, F, S);
  const RowMajorMatrix& F_extrapolated = diis_impl_->get_extrapolated_fock();

  const auto* cfg = ctx_.cfg;
  const auto nelec_vec = scf_impl.get_num_electrons();
  const int nelec[2] = {nelec_vec[0], nelec_vec[1]};
  const int num_atomic_orbitals = scf_impl.get_num_atomic_orbitals();
  const int num_molecular_orbitals = scf_impl.get_num_molecular_orbitals();
  const int num_orbital_sets = scf_impl.get_num_orbital_sets();

  for (int i = 0; i < num_orbital_sets; ++i) {
    solve_fock_eigenproblem(F_extrapolated, S, X, C, eigenvalues,
                            working_density, nelec, num_atomic_orbitals,
                            num_molecular_orbitals, i, cfg->unrestricted);
  }

  auto& density_matrix = scf_impl.density_matrix();
  update_density_matrix(density_matrix, C, ctx_.cfg->unrestricted, nelec[0],
                        nelec[1]);
  after_diis_iteration(scf_impl);
}

void DIISBase::before_diis_iteration(SCFImpl& /*scf_impl*/) {
  QDK_LOG_TRACE_ENTERING();
}

void DIISBase::after_diis_iteration(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;
  double diis_error = current_diis_error();
  bool should_apply_damping = cfg->scf_algorithm.enable_damping &&
                              diis_error > cfg->scf_algorithm.damping_threshold;
  if (should_apply_damping) {
    double factor = cfg->scf_algorithm.damping_factor;
    auto& P_scf_impl = scf_impl.density_matrix();
    P_scf_impl = P_last_ * factor + P_scf_impl * (1.0 - factor);
  }
}

bool DIISBase::uses_total_density_view() const {
  QDK_LOG_TRACE_ENTERING();
  return false;
}

double DIISBase::current_diis_error() const {
  QDK_LOG_TRACE_ENTERING();
  return diis_impl_->get_diis_error();
}

}  // namespace qdk::chemistry::scf
