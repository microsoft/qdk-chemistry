// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/scf/core/scf_algorithm.h"

#include <qdk/chemistry/scf/config.h>

#include <cmath>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>

#include "../scf/scf_impl.h"
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>

#include "util/gpu/matrix_op.h"
#endif

#include <lapack.hh>

#include "asahf.h"
#include "diis.h"
#include "diis_gdm.h"
#include "gdm.h"

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

namespace qdk::chemistry::scf {

SCFAlgorithm::SCFAlgorithm(const SCFContext& ctx)
    : ctx_(ctx),
      step_count_(0),
      last_energy_(0.0),
      density_rms_(std::numeric_limits<double>::infinity()),
      delta_energy_(std::numeric_limits<double>::infinity()) {
  QDK_LOG_TRACE_ENTERING();
  auto num_atomic_orbitals = ctx.basis_set->num_atomic_orbitals;
  auto num_density_matrices = ctx.cfg->unrestricted ? 2 : 1;
  P_last_ = RowMajorMatrix::Zero(num_density_matrices * num_atomic_orbitals,
                                 num_atomic_orbitals);
}

std::shared_ptr<SCFAlgorithm> SCFAlgorithm::create(const SCFContext& ctx) {
  QDK_LOG_TRACE_ENTERING();
  const auto& cfg = *ctx.cfg;

  switch (cfg.scf_algorithm.method) {
    case SCFAlgorithmName::ASAHF:
      return std::make_shared<AtomicSphericallyAveragedHartreeFock>(
          ctx, cfg.scf_algorithm.diis_subspace_size);

    case SCFAlgorithmName::DIIS:
      return std::make_shared<DIIS>(ctx, cfg.scf_algorithm.diis_subspace_size);

    case SCFAlgorithmName::GDM:
      return std::make_shared<GDM>(ctx, cfg.scf_algorithm.gdm_config);

    case SCFAlgorithmName::DIIS_GDM:
      return std::make_shared<DIIS_GDM>(ctx,
                                        cfg.scf_algorithm.diis_subspace_size,
                                        cfg.scf_algorithm.gdm_config);

    default:
      throw std::invalid_argument(
          fmt::format("Unknown SCF algorithm method: {}",
                      static_cast<int>(cfg.scf_algorithm.method)));
  }
}

void SCFAlgorithm::solve_fock_eigenproblem(
    const RowMajorMatrix& F, const RowMajorMatrix& S, const RowMajorMatrix& X,
    RowMajorMatrix& C, RowMajorMatrix& eigenvalues, RowMajorMatrix& P,
    const int num_occupied_orbitals[2], int num_atomic_orbitals,
    int num_molecular_orbitals, int idx_spin, bool unrestricted) {
  QDK_LOG_TRACE_ENTERING();
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{0, 0, 255}, "solve_eigen"};
#endif
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  // solve F_ C_ = e S_ C_ by (conditioned) transformation to F_' C_' = e C_',
  // where F_' = X_.transpose() . F_ . X_; the original C_ is obtained as C_ =
  // X_ . C_'
  auto X_d = cuda::alloc<double>(num_atomic_orbitals * num_molecular_orbitals);
  CUDA_CHECK(cudaMemcpy(X_d->data(), X.data(), sizeof(double) * X.size(),
                        cudaMemcpyHostToDevice));
  auto F_d = cuda::alloc<double>(num_atomic_orbitals * num_atomic_orbitals);
  CUDA_CHECK(cudaMemcpy(
      F_d->data(),
      F.data() + idx_spin * num_atomic_orbitals * num_atomic_orbitals,
      sizeof(double) * num_atomic_orbitals * num_atomic_orbitals,
      cudaMemcpyHostToDevice));

  auto tmp = cuda::alloc<double>(num_molecular_orbitals * num_atomic_orbitals);
  matrix_op::bmm(
      X_d->data(), {1, num_atomic_orbitals, num_molecular_orbitals, true},
      F_d->data(), {num_atomic_orbitals, num_atomic_orbitals}, tmp->data());

  auto V = cuda::alloc<double>(num_molecular_orbitals * num_molecular_orbitals);
  matrix_op::bmm(tmp->data(), {num_molecular_orbitals, num_atomic_orbitals},
                 X_d->data(), {num_atomic_orbitals, num_molecular_orbitals},
                 V->data());

  auto W = cuda::alloc<double>(num_molecular_orbitals);
  cusolver::ManagedcuSolverHandle handle;
  cusolver::syevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                  num_molecular_orbitals, V->data(), num_molecular_orbitals,
                  W->data());

  auto C_d = cuda::alloc<double>(num_atomic_orbitals * num_molecular_orbitals);
  matrix_op::bmm(
      X_d->data(), {num_atomic_orbitals, num_molecular_orbitals}, V->data(),
      {1, num_molecular_orbitals, num_molecular_orbitals, true}, C_d->data());

  auto C_t = tmp;
  matrix_op::transpose(
      C_d->data(), {num_atomic_orbitals, num_molecular_orbitals}, C_t->data());

  auto P_d = F_d;
  auto alpha = unrestricted ? 1.0 : 2.0;
  matrix_op::bmm_ex(
      alpha, C_t->data(),
      {1, num_occupied_orbitals[idx_spin], num_atomic_orbitals, true},
      C_t->data(), {num_occupied_orbitals[idx_spin], num_atomic_orbitals}, 0.0,
      P_d->data());
  CUDA_CHECK(cudaMemcpy(
      P.data() + idx_spin * num_atomic_orbitals * num_atomic_orbitals,
      P_d->data(), sizeof(double) * num_atomic_orbitals * num_atomic_orbitals,
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      C.data() + idx_spin * num_atomic_orbitals * num_molecular_orbitals,
      C_d->data(),
      sizeof(double) * num_atomic_orbitals * num_molecular_orbitals,
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(eigenvalues.data() + idx_spin * num_molecular_orbitals,
                        W->data(), sizeof(double) * num_molecular_orbitals,
                        cudaMemcpyDeviceToHost));
#else
  Eigen::Map<const RowMajorMatrix> F_dm(
      F.data() + idx_spin * num_atomic_orbitals * num_atomic_orbitals,
      num_atomic_orbitals, num_atomic_orbitals);
  Eigen::Map<RowMajorMatrix> P_dm(
      P.data() + idx_spin * num_atomic_orbitals * num_atomic_orbitals,
      num_atomic_orbitals, num_atomic_orbitals);
  Eigen::Map<RowMajorMatrix> C_dm(
      C.data() + idx_spin * num_atomic_orbitals * num_molecular_orbitals,
      num_atomic_orbitals, num_molecular_orbitals);
  RowMajorMatrix tmp1 = X.transpose() * F_dm;
  RowMajorMatrix tmp2 = tmp1 * X;
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, num_molecular_orbitals,
               tmp2.data(), num_molecular_orbitals,
               eigenvalues.data() + idx_spin * num_molecular_orbitals);
  tmp2.transposeInPlace();  // Row major
  C_dm.noalias() = X * tmp2;
  auto alpha = unrestricted ? 1.0 : 2.0;
  P_dm.noalias() =
      alpha *
      C_dm.block(0, 0, num_atomic_orbitals, num_occupied_orbitals[idx_spin]) *
      C_dm.block(0, 0, num_atomic_orbitals, num_occupied_orbitals[idx_spin])
          .transpose();
#endif
}

double SCFAlgorithm::calculate_og_error_(const RowMajorMatrix& F,
                                         const RowMajorMatrix& P,
                                         const RowMajorMatrix& S,
                                         RowMajorMatrix& error_matrix,
                                         bool unrestricted) {
  QDK_LOG_TRACE_ENTERING();
  int num_atomic_orbitals = static_cast<int>(S.cols());
  int num_density_matrices = unrestricted ? 2 : 1;

  RowMajorMatrix FP(num_atomic_orbitals, num_atomic_orbitals);

  error_matrix = RowMajorMatrix::Zero(
      num_density_matrices * num_atomic_orbitals, num_atomic_orbitals);
  for (auto i = 0; i < num_density_matrices; ++i) {
    Eigen::Map<RowMajorMatrix> error_dm(
        error_matrix.data() + i * num_atomic_orbitals * num_atomic_orbitals,
        num_atomic_orbitals, num_atomic_orbitals);
    FP.noalias() = Eigen::Map<const RowMajorMatrix>(
                       F.data() + i * num_atomic_orbitals * num_atomic_orbitals,
                       num_atomic_orbitals, num_atomic_orbitals) *
                   Eigen::Map<const RowMajorMatrix>(
                       P.data() + i * num_atomic_orbitals * num_atomic_orbitals,
                       num_atomic_orbitals, num_atomic_orbitals);
    error_dm.noalias() = FP * S;
    for (size_t ibf = 0; ibf < num_atomic_orbitals; ibf++) {
      error_dm(ibf, ibf) = 0.0;
      for (size_t jbf = 0; jbf < ibf; ++jbf) {
        auto e_ij = error_dm(ibf, jbf);
        auto e_ji = error_dm(jbf, ibf);
        error_dm(ibf, jbf) = e_ij - e_ji;
        error_dm(jbf, ibf) = e_ji - e_ij;
      }
    }
  }
  return error_matrix.lpNorm<Eigen::Infinity>();
}

bool SCFAlgorithm::check_convergence(const SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  const auto* cfg = ctx_.cfg;
  auto& res = ctx_.result;

  int num_atomic_orbitals = scf_impl.get_num_atomic_orbitals();

  // Calculate energy using SCFImpl method
  double energy = res.scf_total_energy;
  delta_energy_ = energy - last_energy_;
  density_rms_ =
      (P_last_ - scf_impl.get_density_matrix()).norm() / num_atomic_orbitals;

  // Calculate orbital gradient error
  RowMajorMatrix error_matrix;
  double og_error =
      calculate_og_error_(scf_impl.get_fock_matrix(),
                          scf_impl.get_density_matrix(), scf_impl.overlap(),
                          error_matrix, cfg->unrestricted) /
      num_atomic_orbitals;

  bool converged = density_rms_ < cfg->scf_algorithm.density_threshold &&
                   og_error < cfg->scf_algorithm.og_threshold;

  QDK_LOGGER().info(
      "Step {:03}: E={:.15e}, DE={:+.15e}, |DP|={:.15e}, |DG|={:.15e}, ",
      step_count_, energy, delta_energy_, density_rms_, og_error);

  // Increment step counter
  step_count_++;

  // Store current values before iteration
  P_last_ = scf_impl.get_density_matrix();
  last_energy_ = res.scf_total_energy;
  return converged;
}

}  // namespace qdk::chemistry::scf
