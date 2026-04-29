// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/scf/core/scf_algorithm.h"

#include <qdk/chemistry/scf/config.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <vector>

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

#include <blas.hh>

namespace qdk::chemistry::scf {

void compute_atba_gemm(const double* A, const double* B, double* C, int m,
                       int n, std::vector<double>& workspace,
                       blas::Layout layout) {
  if (A == nullptr || B == nullptr || C == nullptr) {
    throw std::invalid_argument("compute_atba_gemm: null matrix pointer.");
  }
  if (m < 0 || n < 0) {
    throw std::invalid_argument("compute_atba_gemm: negative dimensions.");
  }
  if (m == 0 || n == 0) {
    return;
  }

  const size_t required_workspace_size = static_cast<size_t>(m) * n;
  if (workspace.size() < required_workspace_size) {
    throw std::invalid_argument(
        "compute_atba_gemm: workspace is smaller than m * n.");
  }

  const int lda = (layout == blas::Layout::RowMajor) ? n : m;
  const int ldb = m;
  const int ldc = n;

  const int ld_workspace = (layout == blas::Layout::RowMajor) ? n : m;

  // workspace = B * A
  blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, m, n, m, 1.0, B, ldb,
             A, lda, 0.0, workspace.data(), ld_workspace);

  // C = A^T * workspace
  blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, n, n, m, 1.0, A, lda,
             workspace.data(), ld_workspace, 0.0, C, ldc);
}

SCFAlgorithm::SCFAlgorithm(const SCFContext& ctx)
    : ctx_(ctx),
      step_count_(0),
      last_energy_(0.0),
      density_rms_(std::numeric_limits<double>::infinity()),
      delta_energy_(std::numeric_limits<double>::infinity()) {
  QDK_LOG_TRACE_ENTERING();
  auto num_atomic_orbitals = ctx.basis_set->num_atomic_orbitals;
  auto num_density_matrices =
      (ctx.cfg->scf_orbital_type == SCFOrbitalType::Unrestricted ||
       ctx.cfg->scf_orbital_type == SCFOrbitalType::RestrictedOpenShell)
          ? 2
          : 1;
  P_last_ = RowMajorMatrix::Zero(num_density_matrices * num_atomic_orbitals,
                                 num_atomic_orbitals);

  if (ctx.cfg->scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    rohf_effective_fock_ =
        RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
    rohf_total_density_ =
        RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
  }
}

SCFAlgorithm::~SCFAlgorithm() noexcept = default;

std::shared_ptr<SCFAlgorithm> SCFAlgorithm::create(const SCFContext& ctx) {
  QDK_LOG_TRACE_ENTERING();
  const auto& cfg = *ctx.cfg;
  const bool rohf_enabled =
      (cfg.scf_orbital_type == SCFOrbitalType::RestrictedOpenShell);

  switch (cfg.scf_algorithm.method) {
    case SCFAlgorithmName::ASAHF:
      if (rohf_enabled) {
        throw std::runtime_error("ROHF-enabled ASAHF is not supported!");
      }
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
    int num_molecular_orbitals, int idx_spin) {
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
#endif
}

void SCFAlgorithm::update_density_matrix(RowMajorMatrix& P,
                                         const RowMajorMatrix& C,
                                         bool unrestricted, int nelec_alpha,
                                         int nelec_beta) {
  QDK_LOG_TRACE_ENTERING();
  const int num_orbital_sets = unrestricted ? 2 : 1;
  const int num_atomic_orbitals =
      static_cast<int>(ctx_.basis_set->num_atomic_orbitals);

  if (C.rows() != num_atomic_orbitals * num_orbital_sets) {
    throw std::invalid_argument(
        "Coefficient matrix rows do not match orbital set count");
  }

  // For ASAHF and ROHF, the density matrix construction is different and
  // will be handled in the overridden methods
  const double occupancy_factor = unrestricted ? 1.0 : 2.0;
  for (int i = 0; i < num_orbital_sets; ++i) {
    const int n_occ = (i == 0) ? nelec_alpha : nelec_beta;
    auto block = P.block(i * num_atomic_orbitals, 0, num_atomic_orbitals,
                         num_atomic_orbitals);
    if (n_occ <= 0) {
      block.setZero();
      continue;
    }

    const auto coeff_block =
        C.block(i * num_atomic_orbitals, 0, num_atomic_orbitals, n_occ);
    block.noalias() = occupancy_factor * coeff_block * coeff_block.transpose();
  }
}

bool SCFAlgorithm::try_get_rohf_convergence_matrices(
    const SCFImpl& scf_impl, const RowMajorMatrix*& fock_matrix,
    const RowMajorMatrix*& density_matrix) {
  QDK_LOG_TRACE_ENTERING();
  if (ctx_.cfg->scf_orbital_type != SCFOrbitalType::RestrictedOpenShell) {
    return false;
  }

  const auto nelec_vec = scf_impl.get_num_electrons();
  build_rohf_f_p_matrix(
      scf_impl.get_fock_matrix(), scf_impl.get_orbitals_matrix(),
      scf_impl.get_density_matrix(), nelec_vec[0], nelec_vec[1],
      rohf_effective_fock_, rohf_total_density_);

  fock_matrix = &get_rohf_convergence_fock_matrix();
  density_matrix = &get_rohf_convergence_density_matrix();
  return true;
}

void SCFAlgorithm::build_rohf_f_p_matrix(const RowMajorMatrix& F,
                                         const RowMajorMatrix& C,
                                         const RowMajorMatrix& P,
                                         int nelec_alpha, int nelec_beta,
                                         RowMajorMatrix& effective_fock,
                                         RowMajorMatrix& total_density) {
  QDK_LOG_TRACE_ENTERING();
  const int num_atomic_orbitals = static_cast<int>(C.rows());
  const int num_molecular_orbitals = static_cast<int>(C.cols());
  if (num_atomic_orbitals != num_molecular_orbitals) {
    throw std::runtime_error(
        "ROHF build requires number of atomic orbitals to equal number of "
        "molecular orbitals!");
  }

  total_density =
      P.block(0, 0, num_atomic_orbitals, num_atomic_orbitals) +
      P.block(num_atomic_orbitals, 0, num_atomic_orbitals, num_atomic_orbitals);

  if (effective_fock.rows() != num_atomic_orbitals ||
      effective_fock.cols() != num_atomic_orbitals) {
    effective_fock =
        RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);
  }

  if (C.isZero()) {
    effective_fock.noalias() =
        F.block(0, 0, num_atomic_orbitals, num_atomic_orbitals);
    return;
  }

  RowMajorMatrix F_up_mo =
      RowMajorMatrix::Zero(num_molecular_orbitals, num_molecular_orbitals);
  RowMajorMatrix F_dn_mo = F_up_mo;
  RowMajorMatrix effective_F_mo = F_up_mo;

  const double* C_block_ptr = C.data();
  const double* F_up_block_ptr = F.data();
  const double* F_dn_block_ptr =
      F.data() + num_atomic_orbitals * num_atomic_orbitals;
  std::vector<double> atba_workspace(static_cast<size_t>(num_atomic_orbitals) *
                                     num_molecular_orbitals);
  compute_atba_gemm(C_block_ptr, F_up_block_ptr, F_up_mo.data(),
                    num_atomic_orbitals, num_molecular_orbitals, atba_workspace,
                    blas::Layout::RowMajor);
  compute_atba_gemm(C_block_ptr, F_dn_block_ptr, F_dn_mo.data(),
                    num_atomic_orbitals, num_molecular_orbitals, atba_workspace,
                    blas::Layout::RowMajor);

  auto average_block = [&effective_F_mo, &F_up_mo, &F_dn_mo](
                           int row, int col, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    effective_F_mo.block(row, col, rows, cols).noalias() =
        0.5 * (F_up_mo.block(row, col, rows, cols) +
               F_dn_mo.block(row, col, rows, cols));
  };
  auto copy_block = [&effective_F_mo](const RowMajorMatrix& src, int row,
                                      int col, int rows, int cols) {
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

  // Transform the effective Fock matrix back to AO basis by solving
  // C^{-T} * F_MO * C^{-1} = F_AO
  // We use LAPACK's getrf/getrs to solve the linear systems involving C^T and
  // C without explicitly inverting C
  const int matrix_dim = num_molecular_orbitals;
  using ColMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  // LAPACK expects column-major layout, so we copy the row-major data into a
  // column-major matrix without transposing the logical layout
  ColMajorMatrix Ct =
      Eigen::Map<const ColMajorMatrix>(C.data(), matrix_dim, C.rows());
  // F_MO is symmetric, so we can use it directly as the right-hand side
  // without transposing
  ColMajorMatrix temp_rhs = effective_F_mo;
  std::vector<int64_t> ipiv(matrix_dim);

  auto info =
      lapack::getrf(matrix_dim, matrix_dim, Ct.data(), matrix_dim, ipiv.data());
  if (info != 0) {
    throw std::runtime_error("getrf failed while factorizing C^T");
  }

  info = lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                       matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
  if (info != 0) {
    throw std::runtime_error("getrs failed while solving C^T X = F_mo");
  }

  temp_rhs.transposeInPlace();
  info = lapack::getrs(lapack::Op::NoTrans, matrix_dim, matrix_dim, Ct.data(),
                       matrix_dim, ipiv.data(), temp_rhs.data(), matrix_dim);
  if (info != 0) {
    throw std::runtime_error("getrs failed while solving C^T X = M^T");
  }

  effective_fock = temp_rhs.transpose();
  if (!effective_fock.isApprox(effective_fock.transpose())) {
    effective_fock = 0.5 * (effective_fock + effective_fock.transpose().eval());
  }
}

const RowMajorMatrix& SCFAlgorithm::get_rohf_convergence_fock_matrix() const {
  QDK_LOG_TRACE_ENTERING();
  if (rohf_effective_fock_.size() == 0) {
    throw std::logic_error("ROHF convergence cache not initialized");
  }
  return rohf_effective_fock_;
}

const RowMajorMatrix& SCFAlgorithm::get_rohf_convergence_density_matrix()
    const {
  QDK_LOG_TRACE_ENTERING();
  if (rohf_total_density_.size() == 0) {
    throw std::logic_error("ROHF convergence cache not initialized");
  }
  return rohf_total_density_;
}

RowMajorMatrix& SCFAlgorithm::rohf_convergence_density_matrix() {
  QDK_LOG_TRACE_ENTERING();
  if (rohf_total_density_.size() == 0) {
    throw std::logic_error("ROHF convergence cache not initialized");
  }
  return rohf_total_density_;
}

double SCFAlgorithm::calculate_og_error_(const RowMajorMatrix& F,
                                         const RowMajorMatrix& P,
                                         const RowMajorMatrix& S,
                                         RowMajorMatrix& error_matrix,
                                         int num_orbital_sets) {
  QDK_LOG_TRACE_ENTERING();
  int num_atomic_orbitals = static_cast<int>(S.cols());
  if (num_orbital_sets != 1 && num_orbital_sets != 2) {
    throw std::invalid_argument("num_orbital_sets must be 1 or 2");
  }

  RowMajorMatrix FP(num_atomic_orbitals, num_atomic_orbitals);

  error_matrix = RowMajorMatrix::Zero(num_orbital_sets * num_atomic_orbitals,
                                      num_atomic_orbitals);
  for (auto i = 0; i < num_orbital_sets; ++i) {
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
  int num_orbital_sets = scf_impl.get_num_orbital_spin_blocks();

  const RowMajorMatrix* F_ptr;
  const RowMajorMatrix* P_ptr;

  if (ctx_.cfg->scf_orbital_type == SCFOrbitalType::RestrictedOpenShell) {
    if (!try_get_rohf_convergence_matrices(scf_impl, F_ptr, P_ptr)) {
      throw std::logic_error(
          "ROHF convergence matrices are not provided by this SCF algorithm");
    }
  } else {
    F_ptr = &scf_impl.get_fock_matrix();
    P_ptr = &scf_impl.get_density_matrix();
  }

  // Fock matrix for RHF; effective Fock matrix for ROHF;
  // spin-blocked Fock matrices for UHF
  const auto& F = *F_ptr;

  // Total density matrix for RHF and ROHF; spin-blocked density matrices for
  // UHF
  const auto& P = *P_ptr;

  double og_error = calculate_og_error_(F, P, scf_impl.overlap(), error_matrix,
                                        num_orbital_sets) /
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
