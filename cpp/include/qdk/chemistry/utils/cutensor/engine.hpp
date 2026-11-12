// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace qdk::chemistry::utils::cutensor {

using Extents = std::vector<int64_t>;
using Modes = std::vector<int32_t>;

/// Number of visible CUDA devices; 0 when no driver or device is available.
int device_count() noexcept;

class Engine;

/**
 * @brief Dense row-major double tensor in device memory.
 *
 * Copies share storage. The storage is allocated and released in stream order
 * on the engine that created it, so a tensor must only be used by that engine.
 */
class DeviceTensor {
 public:
  DeviceTensor() = default;

  /// Uninitialized tensor of the given extents (rank 0 holds one element).
  DeviceTensor(Extents extents, Engine& engine);

  const Extents& extents() const noexcept { return extents_; }
  std::size_t rank() const noexcept { return extents_.size(); }
  std::size_t volume() const noexcept;
  double* data() noexcept { return data_.get(); }
  const double* data() const noexcept { return data_.get(); }

 private:
  Extents extents_;
  std::shared_ptr<double> data_;
};

/**
 * @brief Execution context of one worker thread: a device, a stream, and the
 *        cuTENSOR/cuBLAS handles, plan cache, and workspace bound to them.
 *
 * All operations are asynchronous on the engine's stream unless documented
 * otherwise. cuTENSOR plans are cached by operation, extents, and normalized
 * modes, so repeated contraction shapes reuse their plan.
 */
class Engine {
 public:
  explicit Engine(int device);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  int device() const noexcept { return device_; }
  cudaStream_t stream() const noexcept { return stream_; }

  /// Engine made current on the calling thread by an EngineScope.
  static Engine& current();

  /// Engine of worker @p worker (device `worker % device_count()`), created on
  /// first use and kept for the lifetime of the process.
  static Engine& for_worker(std::size_t worker);

  double* allocate(std::size_t count);
  /// Any thread may release; @p count is the size passed to allocate().
  void release(double* pointer, std::size_t count) noexcept;

  void upload(const double* host, DeviceTensor& destination);
  /// Copies to host memory and waits for the copy.
  void download(const DeviceTensor& source, double* host);
  void copy(const DeviceTensor& source, DeviceTensor& destination);

  /// D[modes_d] = alpha * A[modes_a] * B[modes_b]; modes shared by A, B, and D
  /// are batch modes, modes shared only by A and B are contracted.
  void contract(double alpha, const DeviceTensor& a, const Modes& modes_a,
                const DeviceTensor& b, const Modes& modes_b, DeviceTensor& d,
                const Modes& modes_d);

  /// B[modes_b] = alpha * A[modes_a].
  void permute(double alpha, const DeviceTensor& a, const Modes& modes_a,
               DeviceTensor& b, const Modes& modes_b);

  /// D[modes_d] = alpha * A[modes_a] + gamma * D[modes_d].
  void add(double alpha, const DeviceTensor& a, const Modes& modes_a,
           double gamma, DeviceTensor& d, const Modes& modes_d);

  /// y += alpha * x for tensors of identical layout.
  void axpy(double alpha, const DeviceTensor& x, DeviceTensor& y);

  /// Bilinear dot product of identically laid out tensors; waits for it.
  double dot(const DeviceTensor& a, const DeviceTensor& b);

  /// Frobenius norm; waits for it.
  double norm2(const DeviceTensor& a);

 private:
  struct Plan {
    std::vector<cutensorTensorDescriptor_t> descriptors;
    cutensorOperationDescriptor_t operation = nullptr;
    cutensorPlan_t plan = nullptr;
    uint64_t workspace = 0;
  };

  cutensorTensorDescriptor_t descriptor(
      const Extents& extents,
      std::vector<cutensorTensorDescriptor_t>& owned) const;
  template <typename CreateOperation>
  Plan& plan(const std::string& key, CreateOperation&& create);
  void* workspace(uint64_t bytes);

  int device_;
  cudaStream_t stream_ = nullptr;
  cutensorHandle_t cutensor_ = nullptr;
  cublasHandle_t cublas_ = nullptr;
  cutensorPlanPreference_t preference_ = nullptr;
  std::unordered_map<std::string, Plan> plans_;
  void* workspace_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  // Released blocks by power-of-two size class; reusing one on this stream is
  // ordered after all earlier work on it.
  std::mutex free_mutex_;
  std::vector<std::vector<void*>> free_;
};

/// Makes an engine current on the calling thread (and selects its device)
/// for the lifetime of the scope.
class EngineScope {
 public:
  explicit EngineScope(Engine& engine);
  ~EngineScope();
  EngineScope(const EngineScope&) = delete;
  EngineScope& operator=(const EngineScope&) = delete;

 private:
  Engine* previous_;
};

}  // namespace qdk::chemistry::utils::cutensor
