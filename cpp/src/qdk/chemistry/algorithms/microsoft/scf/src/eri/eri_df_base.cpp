// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "eri_df_base.h"

#include <qdk/chemistry/scf/util/libint2_util.h>
#include <spdlog/spdlog.h>

#include "util/blas.h"
#include "util/lapack.h"

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cublas_utils.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>
#endif

#define QDK_CHEMISTRY_DF_CHOLESKY 0x0
#define QDK_CHEMISTRY_DF_LU 0x1
#define QDK_CHEMISTRY_DF_INV_METHOD QDK_CHEMISTRY_DF_CHOLESKY
namespace qdk::chemistry::scf {

DensityFittingBase::DensityFittingBase(bool unr, const BasisSet& obs,
                                       const BasisSet& abs, ParallelConfig mpi,
                                       bool gpu) {
  if (obs.mode != abs.mode)
    throw std::runtime_error(
        "ERI_DF Cannot Mix Basis Modes Between OBS and ABS");

  unrestricted_ = unr;
  n_atoms_ = obs.mol->n_atoms;
  obs_ = libint2_util::convert_to_libint_basisset(obs);
  abs_ = libint2_util::convert_to_libint_basisset(abs);
  basis_mode_ = obs.mode;
  mpi_ = mpi;
  gpu_ = gpu;
  for (size_t i = 0; i < obs.shells.size(); i++) {
    obs_sh2atom_.push_back(obs.shells[i].atom_index);
  }
  for (size_t i = 0; i < abs.shells.size(); i++) {
    abs_sh2atom_.push_back(abs.shells[i].atom_index);
  }

#ifndef QDK_CHEMISTRY_ENABLE_GPU
  if (gpu) throw std::runtime_error("GPUs are not enabled");
#endif

  // Generate DF Metric
  generate_metric();
}

void DensityFittingBase::generate_metric() {
  const size_t naux = abs_.nbf();
  const size_t metric_sz = naux * naux;

  if (!mpi_.world_rank) spdlog::trace("Generating DF Metric via Libint2");
  h_metric_ = libint2_util::metric_df(basis_mode_, abs_);
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  cublasHandle_ = std::make_unique<cublas::ManagedcuBlasHandle>();
#endif

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (gpu_) {
    // Allocate ERIs on the device and ship data
    if (!mpi_.world_rank) spdlog::trace("Saving DF Metric in Device Memory");
    CUDA_CHECK(cudaMallocAsync(&d_metric_, metric_sz * sizeof(double), 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaMemcpy(d_metric_, h_metric_.get(),
                          sizeof(double) * metric_sz, cudaMemcpyHostToDevice));
    h_metric_ = nullptr;  // Clear host memory

    // Create handle
    cusolverHandle_ = std::make_unique<cusolver::ManagedcuSolverHandle>();
#if QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_CHOLESKY
    cusolver::potrf(*cusolverHandle_, CUBLAS_FILL_MODE_LOWER, naux, d_metric_,
                    naux);
#else
    throw std::runtime_error("CUDF Requires QDK_CHEMISTRY_DF_CHOLESKY");
#endif
  } else {
#endif
    if (!mpi_.world_rank) spdlog::trace("Saving DF Metric in Host Memory");
#if QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_CHOLESKY
    // Cholesky factorization
    lapack::potrf("L", naux, h_metric_.get(), naux);
#elif QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_LU
  h_metric_ipiv_ = std::make_unique<int[]>(naux);
  lapack::getrf(naux, naux, h_metric_.get(), naux, h_metric_ipiv_.get());
#endif
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  }
#endif
}

#ifdef QDK_CHEMISTRY_ENABLE_GPU
void DensityFittingBase::solve_metric_system_device(double* X, size_t LDX) {
  const size_t naux = abs_.num_basis_funcs();
#if QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_CHOLESKY
  cusolver::potrs(*cusolverHandle_, CUBLAS_FILL_MODE_LOWER, naux, 1, d_metric_,
                  naux, X, LDX);
#elif QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_LU
  throw std::runtime_error("CUDF + LU NYI");
#endif
}
#endif

void DensityFittingBase::solve_metric_system_host(double* X, size_t LDX) {
  const size_t naux = abs_.nbf();
#if QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_CHOLESKY
  lapack::potrs("L", naux, 1, h_metric_.get(), naux, X, LDX);
#elif QDK_CHEMISTRY_DF_INV_METHOD == QDK_CHEMISTRY_DF_LU
  lapack::getrs("N", naux, 1, h_metric_.get(), naux, h_metric_ipiv_.get(), X,
                LDX);
#endif
}
}  // namespace qdk::chemistry::scf
