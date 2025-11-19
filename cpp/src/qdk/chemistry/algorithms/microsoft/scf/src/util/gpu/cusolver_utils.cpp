// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>

namespace qdk::chemistry::scf::cusolver {

void potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int N, double* A,
           int LDA) {
  int* d_info;
  int lwork;
  double* work = nullptr;

  // Allocate space for device-side INFO;
  CUDA_CHECK(cudaMallocAsync(&d_info, sizeof(int), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Get workspace
  CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo, N, A, LDA, &lwork));
  if (lwork) CUDA_CHECK(cudaMallocAsync(&work, lwork * sizeof(double), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Perform POTRF
  CUSOLVER_CHECK(
      cusolverDnDpotrf(handle, uplo, N, A, LDA, work, lwork, d_info));

  // Check operation status
  int h_info;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info)
    throw std::runtime_error("CUSOLVER DPOTRF FAILED WITH INFO = " +
                             std::to_string(h_info));

  // Cleanup workspace and INFO;
  if (lwork) CUDA_CHECK(cudaFree(work));
  CUDA_CHECK(cudaFree(d_info));
}

void potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int N, int NRHS,
           const double* A, int LDA, double* B, int LDB) {
  // Allocate space for device-side INFO;
  int* d_info;
  CUDA_CHECK(cudaMallocAsync(&d_info, sizeof(int), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));

  CUSOLVER_CHECK(
      cusolverDnDpotrs(handle, uplo, N, NRHS, A, LDA, B, LDB, d_info));

  // Check operation status
  int h_info;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info)
    throw std::runtime_error("CUSOLVER DPOTRF FAILED WITH INFO = " +
                             std::to_string(h_info));

  // Cleanup INFO;
  CUDA_CHECK(cudaFree(d_info));
}

void syevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
           cublasFillMode_t uplo, int N, double* A, int LDA, double* W) {
  int* d_info;
  int lwork;
  double* work = nullptr;

  // Allocate space for device-side INFO;
  CUDA_CHECK(cudaMallocAsync(&d_info, sizeof(int), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Get workspace
  CUSOLVER_CHECK(
      cusolverDnDsyevd_bufferSize(handle, jobz, uplo, N, A, LDA, W, &lwork));
  if (lwork) CUDA_CHECK(cudaMallocAsync(&work, lwork * sizeof(double), 0));
  CUDA_CHECK(cudaStreamSynchronize(0));

  // Perform SYEVD
  CUSOLVER_CHECK(
      cusolverDnDsyevd(handle, jobz, uplo, N, A, LDA, W, work, lwork, d_info));

  // Check operation status
  int h_info;
  CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_info)
    throw std::runtime_error("CUSOLVER DSYEVD FAILED WITH INFO = " +
                             std::to_string(h_info));

  // Cleanup workspace and INFO;
  if (lwork) CUDA_CHECK(cudaFree(work));
  CUDA_CHECK(cudaFree(d_info));
}

}  // namespace qdk::chemistry::scf::cusolver
#endif  // QDK_CHEMISTRY_ENABLE_GPU
