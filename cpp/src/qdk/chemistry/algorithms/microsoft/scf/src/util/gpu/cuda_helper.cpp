// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cuda_helper.h>
#include <spdlog/spdlog.h>

namespace qdk::chemistry::scf::cuda {

/**
 * @brief Maximum number of CUDA devices that can be managed simultaneously
 *
 * This constant defines the upper limit on the number of CUDA devices that
 * the memory pool initialization system can track. It's used to size static
 * arrays for per-device initialization state tracking.
 */
constexpr int MAX_NUM_DEVICE = 32;

int get_current_device() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void init_memory_pool(int device) {
  static bool initialized[MAX_NUM_DEVICE] = {false};
  if (!initialized[device]) {
    spdlog::trace("Configure memory pool for device {}", device);
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, device);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold,
                            &threshold);
    initialized[device] = true;
  }
}

void trim_memory_pool(size_t bytes, int device) {
  spdlog::trace("Trim memory pool to {} bytes for device {}", bytes, device);
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, device);
  CUDA_CHECK(cudaMemPoolTrimTo(mempool, bytes));
}

void show_memory_pool_info(int device) {
  cudaMemPool_t mempool;
  cudaDeviceGetDefaultMemPool(&mempool, device);
  auto print_attr = [&](cudaMemPoolAttr attr, const char* attr_name) {
    size_t val;
    CUDA_CHECK(cudaMemPoolGetAttribute(mempool, attr, &val));
    fmt::print("{:<50}: {}\n", attr_name, val);
  };

  fmt::print("{:=^80}\n", fmt::format("GPU#{}", device));
  print_attr(cudaMemPoolAttrReleaseThreshold,
             "cudaMemPoolAttrReleaseThreshold");
  print_attr(cudaMemPoolAttrReservedMemCurrent,
             "cudaMemPoolAttrReservedMemCurrent");
  print_attr(cudaMemPoolAttrReservedMemHigh, "cudaMemPoolAttrReservedMemHigh");
  print_attr(cudaMemPoolAttrUsedMemCurrent, "cudaMemPoolAttrUsedMemCurrent");
  print_attr(cudaMemPoolAttrUsedMemHigh, "cudaMemPoolAttrUsedMemHigh");
  fmt::print("{:=^80}\n", "");
}

}  // namespace qdk::chemistry::scf::cuda
#endif  // QDK_CHEMISTRY_ENABLE_GPU
