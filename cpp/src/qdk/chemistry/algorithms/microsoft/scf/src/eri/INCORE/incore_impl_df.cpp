// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <spdlog/spdlog.h>

#include <blas.hh>
#include <lapack.hh>
#include <libint2.hpp>
#include <stdexcept>

#include "incore_impl.h"
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#include <qdk/chemistry/scf/util/gpu/cublas_utils.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>
#endif
#include <qdk/chemistry/scf/util/libint2_util.h>

namespace qdk::chemistry::scf::incore {

ERI_DF::ERI_DF(bool unr, const BasisSet& obs, const BasisSet& abs,
               ParallelConfig mpi)
    : DensityFittingBase(unr, obs, abs, mpi,
#ifdef QDK_CHEMISTRY_ENABLE_GPU
                         true /*gpu*/
#else
                         false /*gpu*/
#endif
      ) {

  // Distribute on AUX index
  const size_t naux = abs_.nbf();
  const size_t naux_dist = naux / mpi_.world_size;
  loc_i_st_ = mpi_.world_rank * naux_dist;
  loc_i_en_ = loc_i_st_ + naux_dist;
  if (mpi_.world_rank == mpi_.world_size - 1) {
    loc_i_en_ = naux;
  }

  // Generate DF-ERIs
  generate_eri_();
}

void ERI_DF::generate_eri_() {
  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t naux = abs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t eri_sz = num_atomic_orbitals2 * (loc_i_en_ - loc_i_st_);

  if (!mpi_.world_rank) spdlog::trace("Generating DF-ERIs via Libint2");
  h_eri_ = libint2_util::eri_df(basis_mode_, obs_, abs_, loc_i_st_, loc_i_en_);

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (!gpu()) {
    if (!mpi_.world_rank) spdlog::trace("Saving DF-ERIs on Host Memory");
  } else {
    // Allocate ERIs on the device and ship data
    if (!mpi_.world_rank) {
      spdlog::trace("Saving DF-ERIs in Device Memory");
      spdlog::trace("Using cuTensor for GPU DF-ERI Contraction");
    }
    CUDA_CHECK(cudaMallocAsync(&d_eri_, eri_sz * sizeof(double), 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpy(d_eri_, h_eri_.get(), sizeof(double) * eri_sz,
                          cudaMemcpyHostToDevice));
    h_eri_ = nullptr;  // Clear host memory
  }
#endif
}

void ERI_DF::build_JK(const double* P, double* J, double* K, double alpha,
                      double beta, double omega) {
  if (std::abs(omega) > 1e-12)
    throw std::runtime_error(
        "ERIINCORE_DF Does Not Support Range-Separated Hybrids");
  if (std::abs(alpha) + std::abs(beta) > 1e-12 and K)
    throw std::runtime_error("ERIINCORE_DF + Exchange is Not Yet Implemented");
  if (not J) throw std::runtime_error("ERIINCORE_DF is only valid for DF-J");

  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t naux = abs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t naux_loc = loc_i_en_ - loc_i_st_;
  const double one = 1.0, zero = 0.0;

  // Form P[tota] if unrestricted
  std::vector<double> P_total;
  const double* P_use = P;
  if (unrestricted_) {
    P_total.resize(num_atomic_orbitals2, 0.0);
    for (auto i = 0; i < num_atomic_orbitals2; ++i)
      P_total[i] = P[i] + P[i + num_atomic_orbitals2];
    P_use = P_total.data();
  }

  std::vector<double> X(naux, 0.0);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  double *dP, *dJ, *dX;
  if (gpu()) {
    CUDA_CHECK(cudaMallocAsync(&dP, num_atomic_orbitals2 * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&dJ, num_atomic_orbitals2 * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&dX, naux * sizeof(double), 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    // This is not needed if serial as the GEMV sets X
    if (mpi_.world_size > 1) {
      CUDA_CHECK(cudaMemset(dX, 0, naux * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(dP, P_use, num_atomic_orbitals2 * sizeof(double),
                          cudaMemcpyHostToDevice));
    std::vector<double>().swap(P_total);  // Deallocate host copy of P_total
  }
  auto dX_loc = dX + loc_i_st_;
#endif
  auto X_loc = X.data() + loc_i_st_;

  // Form X(I) = (pq|I) P(p,q)
  if (gpu()) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    CUBLAS_CHECK(cublasDgemv(*cublasHandle_, CUBLAS_OP_T, num_atomic_orbitals2,
                             naux_loc, &one, d_eri_, num_atomic_orbitals2, dP,
                             1, &zero, dX_loc, 1));
#endif
  } else {
    blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, num_atomic_orbitals2,
               naux_loc, 1.0, h_eri_.get(), num_atomic_orbitals2, P_use, 1, 0.0,
               X_loc, 1);
  }

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    if (gpu())
      CUDA_CHECK(cudaMemcpy(X.data(), dX, naux * sizeof(double),
                            cudaMemcpyDeviceToHost));
#endif
    MPI_Allreduce(MPI_IN_PLACE, X.data(), naux, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    if (gpu())
      CUDA_CHECK(cudaMemcpy(dX, X.data(), naux * sizeof(double),
                            cudaMemcpyHostToDevice));
#endif
  }
#endif

  // Form Y(J) = V**1(J,I) * X(I)
  if (gpu()) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    solve_metric_system_device(dX, naux);
#endif
  } else {
    solve_metric_system_host(X.data(), naux);
  }

  // Form J(p,q) = (pq|I) * Y(I)
  // Reduction happens in ERI::build_JK
  if (gpu()) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    CUBLAS_CHECK(cublasDgemv(*cublasHandle_, CUBLAS_OP_N, num_atomic_orbitals2,
                             naux_loc, &one, d_eri_, num_atomic_orbitals2,
                             dX_loc, 1, &zero, dJ, 1));
#endif
  } else {
    blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, num_atomic_orbitals2,
               naux_loc, 1.0, h_eri_.get(), num_atomic_orbitals2, X_loc, 1, 0.0,
               J, 1);
  }

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (gpu()) {
    CUDA_CHECK(cudaMemcpy(J, dJ, num_atomic_orbitals2 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dJ));
    CUDA_CHECK(cudaFree(dX));
  }
#endif

  if (K) std::fill_n(K, (unrestricted_ ? 2 : 1) * num_atomic_orbitals2, 0.0);
  if (unrestricted_) {
    // J[alpha] contains J_total and J[beta] is zero - this is hacky but it
    // works
    for (size_t i = 0; i < num_atomic_orbitals2; ++i) {
      J[i + num_atomic_orbitals2] = 0.0;
    }
  }
}

void ERI_DF::get_gradients(const double* P, double* dJ, double* dK,
                           double alpha, double beta, double omega) {
  if (std::abs(omega) > 1e-12)
    throw std::runtime_error(
        "ERIINCORE_DF gradients Does Not Support Range-Separated Hybrids");
  if (std::abs(alpha) + std::abs(beta) > 1e-12 and dK)
    throw std::runtime_error(
        "ERIINCORE_DF + Exchange gradients is Not Yet Implemented");
  if (not dJ)
    throw std::runtime_error("ERIINCORE_DF is only valid for DF-J gradients");

  const size_t num_atomic_orbitals = obs_.nbf();
  const size_t naux = abs_.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t n_atoms = n_atoms_;
  const size_t naux_loc = loc_i_en_ - loc_i_st_;
  const double one = 1.0, zero = 0.0;

  // Zero out dJ on entry
  std::fill_n(dJ, 3 * n_atoms, 0.0);

  // Form P[tota] if unrestricted
  std::vector<double> P_total;
  const double* P_use = P;
  if (unrestricted_) {
    P_total.resize(num_atomic_orbitals2, 0.0);
    for (auto i = 0; i < num_atomic_orbitals2; ++i)
      P_total[i] = P[i] + P[i + num_atomic_orbitals2];
    P_use = P_total.data();
  }

  std::vector<double> X(naux, 0.0);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  double *dP, *dX;
  if (gpu()) {
    CUDA_CHECK(cudaMallocAsync(&dP, num_atomic_orbitals2 * sizeof(double), 0));
    CUDA_CHECK(cudaMallocAsync(&dX, naux * sizeof(double), 0));
    CUDA_CHECK(cudaStreamSynchronize(0));

    // This is not needed if serial as the GEMV sets X
    if (mpi_.world_size > 1) {
      CUDA_CHECK(cudaMemset(dX, 0, naux * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(dP, P_use, num_atomic_orbitals2 * sizeof(double),
                          cudaMemcpyHostToDevice));
  }
  auto dX_loc = dX + loc_i_st_;
#endif
  auto X_loc = X.data() + loc_i_st_;

  // Form X(I) = \Sum_{pq} (pq|I) P(p,q)
  if (gpu()) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    CUBLAS_CHECK(cublasDgemv(*cublasHandle_, CUBLAS_OP_T, num_atomic_orbitals2,
                             naux_loc, &one, d_eri_, num_atomic_orbitals2, dP,
                             1, &zero, dX_loc, 1));
#endif
  } else {
    blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, num_atomic_orbitals2,
               naux_loc, 1.0, h_eri_.get(), num_atomic_orbitals2, P_use, 1, 0.0,
               X_loc, 1);
  }

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    if (gpu())
      CUDA_CHECK(cudaMemcpy(X.data(), dX, naux * sizeof(double),
                            cudaMemcpyDeviceToHost));
#endif
    MPI_Allreduce(MPI_IN_PLACE, X.data(), naux, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    if (gpu())
      CUDA_CHECK(cudaMemcpy(dX, X.data(), naux * sizeof(double),
                            cudaMemcpyHostToDevice));
#endif
  }
#endif

  // Form Y(J) = V**1(J,I) * X(I)
  if (gpu()) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
    solve_metric_system_device(dX, naux);
#endif
  } else {
    solve_metric_system_host(X.data(), naux);
  }

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  // The following codes will not use gpu
  if (gpu()) {
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaMemcpy(X.data(), dX, naux * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
  }
#endif

  // Form part 1 of dJ^x: \Sum_{pqI} P(p,q) * (pq|I)^x * Y(I)
  libint2_util::eri_df_grad(dJ, P_use, X.data(), basis_mode_, obs_, abs_,
                            obs_sh2atom_, abs_sh2atom_, n_atoms, mpi_);

  // Form part 2 of dJ^x: -1/2 \Sum_{IJ}(I|J)^x * Y(I) * Y(J)
  libint2_util::metric_df_grad(dJ, X.data(), basis_mode_, abs_, abs_sh2atom_,
                               n_atoms, mpi_);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  if (mpi_.world_size > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : dJ, dJ, 3 * n_atoms,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  if (dK) std::fill_n(dK, 3 * n_atoms, 0.0);
}

void ERI_DF::quarter_trans(size_t nt, const double* C, double* out) {
  throw std::runtime_error("INCORE_DF QUARTER_TRANS NYI");
}

ERI_DF::~ERI_DF() noexcept {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (d_eri_) CUDA_CHECK(cudaFree(d_eri_));
  if (d_metric_) CUDA_CHECK(cudaFree(d_metric_));
#endif
};

std::unique_ptr<ERI_DF> ERI_DF::make_incore_eri(bool unr, const BasisSet& obs,
                                                const BasisSet& abs,
                                                ParallelConfig mpi) {
  return std::make_unique<ERI_DF>(unr, obs, abs, mpi);
}

}  // namespace qdk::chemistry::scf::incore
