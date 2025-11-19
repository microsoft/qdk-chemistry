// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

/**
 * @brief Check cuBLAS API call for errors and throw exception on failure
 *
 * Macro that wraps cuBLAS API calls to automatically check return status.
 * If the call fails (returns any status other than CUBLAS_STATUS_SUCCESS),
 * logs a critical error message and throws std::runtime_error.
 *
 * Usage:
 * @code
 *   CUBLAS_CHECK(cublasCreate(&handle));
 *   CUBLAS_CHECK(cublasDgemm(...));
 * @endcode
 *
 * @param err cuBLAS API call expression that returns cublasStatus_t
 * @throws std::runtime_error if cuBLAS call fails
 */
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      spdlog::critical("cuBLAS error {} at {}:{}", static_cast<int>(err_), \
                       __FILE__, __LINE__);                                \
      throw std::runtime_error("cuBLAS error");                            \
    }                                                                      \
  } while (0)

namespace qdk::chemistry::scf::cublas {

/**
 * @brief RAII wrapper for cuBLAS handle management
 *
 * Provides automatic lifetime management for cuBLAS handles
 *
 * The class provides implicit conversion to cublasHandle_t, allowing it to
 * be used directly in cuBLAS API calls.
 *
 * Example usage:
 * @code
 *   ManagedcuBlasHandle handle;
 *   cublasDgemm(handle, ...);  // Implicit conversion
 * @endcode
 */
class ManagedcuBlasHandle {
  cublasHandle_t handle_;  ///< The managed cuBLAS handle for GPU operations

 public:
  /**
   * @brief Create and initialize a cuBLAS handle
   *
   * Calls cublasCreate() to initialize a new cuBLAS handle for the current
   * CUDA device context. The handle can be used for subsequent cuBLAS calls.
   *
   * @throws std::runtime_error if cuBLAS handle creation fails
   */
  inline ManagedcuBlasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }

  /**
   * @brief Destroy the cuBLAS handle and release GPU resources
   *
   * Calls cublasDestroy() to free resources associated with the handle.
   * Marked noexcept as destructors should not throw exceptions.
   */
  inline ~ManagedcuBlasHandle() noexcept { cublasDestroy(handle_); }

  /**
   * @brief Implicit conversion to cublasHandle_t for API calls
   *
   * Allows ManagedcuBlasHandle to be used directly in cuBLAS function calls
   * without explicitly accessing the underlying handle.
   *
   * @return cublasHandle_t The underlying cuBLAS handle
   */
  inline operator cublasHandle_t() { return handle_; }
};

}  // namespace qdk::chemistry::scf::cublas
#endif  // QDK_CHEMISTRY_ENABLE_GPU
