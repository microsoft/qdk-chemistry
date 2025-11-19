// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

/**
 * @brief Check cuSOLVER API call for errors
 *
 * Throws std::runtime_error if the cuSOLVER call fails.
 */
#define CUSOLVER_CHECK(err)                                                  \
  do {                                                                       \
    cusolverStatus_t err_ = (err);                                           \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                                   \
      spdlog::critical("cuSOLVER error {} at {}:{}", static_cast<int>(err_), \
                       __FILE__, __LINE__);                                  \
      throw std::runtime_error("cuSOLVER error");                            \
    }                                                                        \
  } while (0)

namespace qdk::chemistry::scf::cusolver {

/**
 * @brief RAII wrapper for cuSOLVER handle management
 *
 * Automatically creates and destroys cuSOLVER dense handle to prevent resource
 * leaks.
 */
class ManagedcuSolverHandle {
  cusolverDnHandle_t handle_;

 public:
  /**
   * @brief Create a cuSOLVER dense handle
   */
  inline ManagedcuSolverHandle() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }

  /**
   * @brief Destroy the cuSOLVER handle
   */
  inline ~ManagedcuSolverHandle() noexcept { cusolverDnDestroy(handle_); }

  /**
   * @brief Implicit conversion to cusolverDnHandle_t for API calls
   */
  inline operator cusolverDnHandle_t() { return handle_; }
};

/**
 * @brief Perform Cholesky factorization on device
 *
 * Performs in-place Cholesky factorization of a symmetric positive definite
 * matrix stored in device memory. This wrapper also allocates workspace
 * if necessary and checks for successful completion.
 *
 * @see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-potrf for
 * API details
 */
void potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int N, double* A,
           int LDA);

/**
 * @brief Solve linear system using Cholesky factorization
 *
 * Performs in-place Cholesky solve given Cholesky factors stored
 * in device memory. This wrapper also allocates workspace
 * if necessary and checks for successful completion.
 *
 * @see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-potrs for
 * API details
 */
void potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int N, int NRHS,
           const double* A, int LDA, double* B, int LDB);

/**
 * @brief Solve a linear symmetric eigenvalue problem
 *
 * Performs an in-place eigenvalue decomposition of a symmetric matrix. If
 * requested eigenvectors are returned in place of the input matrix, otherwise
 * the input matrix is destroyed. This wrapper also allocates workspace if
 * necessary and checks for successful completion.
 *
 * @see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-syevd for
 * API details
 */
void syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
           cublasFillMode_t uplo, int N, double* A, int LDA, double* W);
}  // namespace qdk::chemistry::scf::cusolver
#endif  // QDK_CHEMISTRY_ENABLE_GPU
