// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/device.h>

#include <limits>
#include <qdk/chemistry/utils/logger.hpp>

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <cuda_runtime.h>
#endif

namespace qdk::chemistry::scf::device {

namespace {
// Thread-local so that a caller running several solves in parallel cannot have
// one change the device under another.
thread_local bool g_use_gpu = false;
}  // namespace

bool gpu_available() {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  static const bool available = [] {
    int count = 0;
    // cudaGetDeviceCount is the cheapest probe that also reports a driver
    // mismatch, and it must not throw: absence of a GPU is a normal condition,
    // not an error.
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
      cudaGetLastError();
      QDK_LOGGER().debug("No usable CUDA device ({}); SCF will run on the CPU.",
                         cudaGetErrorString(status));
      return false;
    }
    return count > 0;
  }();
  return available;
#else
  return false;
#endif
}

bool use_gpu() { return g_use_gpu; }

bool use_gpu_for_dense(size_t n) {
  // Transfers outweigh dense GPU work below this dimension.
  constexpr size_t kMinDenseDimension = 1000;
  return g_use_gpu && n >= kMinDenseDimension;
}

bool device_can_hold(size_t bytes) {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  if (!gpu_available()) return false;
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return bytes < free_bytes;
#else
  (void)bytes;
  return false;
#endif
}

bool set_use_gpu(bool enable) {
  g_use_gpu = enable && gpu_available();
  return g_use_gpu;
}

bool incore_eri_can_fit(size_t basis_functions) {
  size_t bytes = sizeof(double);
  for (int index = 0; index < 4; ++index) {
    if (basis_functions &&
        bytes > std::numeric_limits<size_t>::max() / basis_functions)
      return false;
    bytes *= basis_functions;
  }
  if (bytes > std::numeric_limits<size_t>::max() - bytes / 8) return false;
  return device_can_hold(bytes + bytes / 8);
}

}  // namespace qdk::chemistry::scf::device
