// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <qdk/chemistry/utils/cutensor/engine.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::utils::cutensor {

namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("cuTENSOR backend: ") + what + ": " +
                             cudaGetErrorString(status));
}

void check(cutensorStatus_t status, const char* what) {
  if (status != CUTENSOR_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuTENSOR backend: ") + what + ": " +
                             cutensorGetErrorString(status));
}

void check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("cuTENSOR backend: ") + what +
                             ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

thread_local Engine* current_engine = nullptr;

Extents row_major_strides(const Extents& extents) {
  Extents strides(extents.size(), 1);
  for (std::size_t i = extents.size(); i-- > 1;)
    strides[i - 1] = strides[i] * extents[i];
  return strides;
}

template <typename T>
void append(std::string& key, const std::vector<T>& values) {
  const auto size = static_cast<uint32_t>(values.size());
  key.append(reinterpret_cast<const char*>(&size), sizeof(size));
  key.append(reinterpret_cast<const char*>(values.data()),
             values.size() * sizeof(T));
}

constexpr int kSmallestClass = 8, kLargestClass = 26;  // 256 B to 64 MiB

// Index of the pooled power-of-two size class holding `bytes`, or -1.
int size_class(std::size_t bytes) {
  int bits = kSmallestClass;
  while ((std::size_t{1} << bits) < bytes) ++bits;
  return bits <= kLargestClass ? bits - kSmallestClass : -1;
}

}  // namespace

int device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

DeviceTensor::DeviceTensor(Extents extents, Engine& engine)
    : extents_(std::move(extents)),
      data_(engine.allocate(volume()),
            [engine = &engine, count = volume()](double* pointer) {
              engine->release(pointer, count);
            }) {}

std::size_t DeviceTensor::volume() const noexcept {
  return std::accumulate(extents_.begin(), extents_.end(), std::size_t{1},
                         std::multiplies<>{});
}

Engine::Engine(int device)
    : device_(device), free_(kLargestClass - kSmallestClass + 1) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  // Keep freed stream-ordered allocations pooled instead of returning them to
  // the driver at every synchronization.
  cudaMemPool_t pool;
  check(cudaDeviceGetDefaultMemPool(&pool, device_),
        "cudaDeviceGetDefaultMemPool");
  uint64_t threshold = std::numeric_limits<uint64_t>::max();
  check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                &threshold),
        "cudaMemPoolSetAttribute");
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "cudaStreamCreate");
  check(cutensorCreate(&cutensor_), "cutensorCreate");
  check(cublasCreate(&cublas_), "cublasCreate");
  check(cublasSetStream(cublas_, stream_), "cublasSetStream");
  check(cutensorCreatePlanPreference(cutensor_, &preference_,
                                     CUTENSOR_ALGO_DEFAULT,
                                     CUTENSOR_JIT_MODE_NONE),
        "cutensorCreatePlanPreference");
}

Engine::~Engine() {
  (void)cudaSetDevice(device_);
  for (auto& blocks : free_)
    for (void* block : blocks) (void)cudaFreeAsync(block, stream_);
  (void)cudaStreamSynchronize(stream_);
  for (auto& [key, entry] : plans_) {
    (void)cutensorDestroyPlan(entry.plan);
    (void)cutensorDestroyOperationDescriptor(entry.operation);
    for (auto descriptor : entry.descriptors)
      (void)cutensorDestroyTensorDescriptor(descriptor);
  }
  if (workspace_) (void)cudaFree(workspace_);
  (void)cutensorDestroyPlanPreference(preference_);
  (void)cublasDestroy(cublas_);
  (void)cutensorDestroy(cutensor_);
  (void)cudaStreamDestroy(stream_);
}

Engine& Engine::current() {
  if (!current_engine)
    throw std::runtime_error(
        "cuTENSOR backend: no engine is active on this thread");
  return *current_engine;
}

Engine& Engine::for_worker(std::size_t worker) {
  struct Slot {
    std::once_flag once;
    Engine* engine = nullptr;
  };
  static std::mutex mutex;
  // Engines own streams and handles that device tensors released at any time
  // may still use, so they live until the process exits.
  static auto* slots = new std::vector<std::unique_ptr<Slot>>;
  const int count = device_count();
  if (count == 0)
    throw std::runtime_error("cuTENSOR backend: no CUDA device is visible");
  Slot* slot;
  {
    std::lock_guard lock(mutex);
    while (slots->size() <= worker) slots->push_back(std::make_unique<Slot>());
    slot = (*slots)[worker].get();
  }
  // Workers set up their devices concurrently.
  std::call_once(slot->once, [&] {
    slot->engine = new Engine(static_cast<int>(worker % count));
  });
  return *slot->engine;
}

double* Engine::allocate(std::size_t count) {
  const std::size_t bytes = std::max<std::size_t>(count, 1) * sizeof(double);
  const int pooled = size_class(bytes);
  if (pooled >= 0) {
    std::lock_guard lock(free_mutex_);
    if (!free_[pooled].empty()) {
      void* pointer = free_[pooled].back();
      free_[pooled].pop_back();
      return static_cast<double*>(pointer);
    }
  }
  void* pointer = nullptr;
  check(cudaMallocAsync(
            &pointer,
            pooled >= 0 ? std::size_t{1} << (pooled + kSmallestClass) : bytes,
            stream_),
        "cudaMallocAsync");
  return static_cast<double*>(pointer);
}

void Engine::release(double* pointer, std::size_t count) noexcept {
  if (!pointer) return;
  const int pooled =
      size_class(std::max<std::size_t>(count, 1) * sizeof(double));
  if (pooled >= 0) {
    std::lock_guard lock(free_mutex_);
    free_[pooled].push_back(pointer);
    return;
  }
  // Tensors may be released on any thread, e.g. when a worker's cache is freed.
  int previous = device_;
  (void)cudaGetDevice(&previous);
  if (previous != device_) (void)cudaSetDevice(device_);
  (void)cudaFreeAsync(pointer, stream_);
  if (previous != device_) (void)cudaSetDevice(previous);
}

void Engine::upload(const double* host, DeviceTensor& destination) {
  check(cudaMemcpyAsync(destination.data(), host,
                        destination.volume() * sizeof(double),
                        cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync (upload)");
}

void Engine::download(const DeviceTensor& source, double* host) {
  check(cudaMemcpyAsync(host, source.data(), source.volume() * sizeof(double),
                        cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync (download)");
  check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

void Engine::copy(const DeviceTensor& source, DeviceTensor& destination) {
  check(cudaMemcpyAsync(destination.data(), source.data(),
                        source.volume() * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream_),
        "cudaMemcpyAsync (copy)");
}

cutensorTensorDescriptor_t Engine::descriptor(
    const Extents& extents,
    std::vector<cutensorTensorDescriptor_t>& owned) const {
  const auto strides = row_major_strides(extents);
  cutensorTensorDescriptor_t result = nullptr;
  check(cutensorCreateTensorDescriptor(
            cutensor_, &result, static_cast<uint32_t>(extents.size()),
            extents.data(), strides.data(), CUTENSOR_R_64F, 128),
        "cutensorCreateTensorDescriptor");
  owned.push_back(result);
  return result;
}

template <typename CreateOperation>
Engine::Plan& Engine::plan(const std::string& key, CreateOperation&& create) {
  if (auto found = plans_.find(key); found != plans_.end())
    return found->second;
  Plan entry;
  create(entry);
  uint64_t estimate = 0;
  check(cutensorEstimateWorkspaceSize(cutensor_, entry.operation, preference_,
                                      CUTENSOR_WORKSPACE_DEFAULT, &estimate),
        "cutensorEstimateWorkspaceSize");
  check(cutensorCreatePlan(cutensor_, &entry.plan, entry.operation, preference_,
                           estimate),
        "cutensorCreatePlan");
  check(cutensorPlanGetAttribute(cutensor_, entry.plan,
                                 CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                 &entry.workspace, sizeof(entry.workspace)),
        "cutensorPlanGetAttribute");
  return plans_.emplace(key, std::move(entry)).first->second;
}

void* Engine::workspace(uint64_t bytes) {
  if (bytes > workspace_bytes_) {
    if (workspace_) (void)cudaFreeAsync(workspace_, stream_);
    check(cudaMallocAsync(&workspace_, bytes, stream_),
          "cudaMallocAsync (workspace)");
    workspace_bytes_ = bytes;
  }
  return workspace_;
}

void Engine::contract(double alpha, const DeviceTensor& a, const Modes& modes_a,
                      const DeviceTensor& b, const Modes& modes_b,
                      DeviceTensor& d, const Modes& modes_d) {
  std::string key = "contract";
  for (const auto* extents : {&a.extents(), &b.extents(), &d.extents()})
    append(key, *extents);
  for (const auto* modes : {&modes_a, &modes_b, &modes_d}) append(key, *modes);
  auto& entry = plan(key, [&](Plan& created) {
    const auto da = descriptor(a.extents(), created.descriptors);
    const auto db = descriptor(b.extents(), created.descriptors);
    const auto dd = descriptor(d.extents(), created.descriptors);
    check(cutensorCreateContraction(cutensor_, &created.operation, da,
                                    modes_a.data(), CUTENSOR_OP_IDENTITY, db,
                                    modes_b.data(), CUTENSOR_OP_IDENTITY, dd,
                                    modes_d.data(), CUTENSOR_OP_IDENTITY, dd,
                                    modes_d.data(), CUTENSOR_COMPUTE_DESC_64F),
          "cutensorCreateContraction");
  });
  const double beta = 0.0;
  check(cutensorContract(cutensor_, entry.plan, &alpha, a.data(), b.data(),
                         &beta, d.data(), d.data(), workspace(entry.workspace),
                         entry.workspace, stream_),
        "cutensorContract");
}

void Engine::permute(double alpha, const DeviceTensor& a, const Modes& modes_a,
                     DeviceTensor& b, const Modes& modes_b) {
  if (modes_a == modes_b) {
    copy(a, b);
    if (alpha != 1.0)
      check(cublasDscal_64(cublas_, static_cast<int64_t>(b.volume()), &alpha,
                           b.data(), 1),
            "cublasDscal");
    return;
  }
  std::string key = "permute";
  append(key, a.extents());
  append(key, modes_a);
  append(key, modes_b);
  auto& entry = plan(key, [&](Plan& created) {
    const auto da = descriptor(a.extents(), created.descriptors);
    const auto db = descriptor(b.extents(), created.descriptors);
    check(cutensorCreatePermutation(cutensor_, &created.operation, da,
                                    modes_a.data(), CUTENSOR_OP_IDENTITY, db,
                                    modes_b.data(), CUTENSOR_COMPUTE_DESC_64F),
          "cutensorCreatePermutation");
  });
  check(cutensorPermute(cutensor_, entry.plan, &alpha, a.data(), b.data(),
                        stream_),
        "cutensorPermute");
}

void Engine::add(double alpha, const DeviceTensor& a, const Modes& modes_a,
                 double gamma, DeviceTensor& d, const Modes& modes_d) {
  std::string key = "add";
  append(key, a.extents());
  append(key, d.extents());
  append(key, modes_a);
  append(key, modes_d);
  auto& entry = plan(key, [&](Plan& created) {
    const auto da = descriptor(a.extents(), created.descriptors);
    const auto dd = descriptor(d.extents(), created.descriptors);
    check(cutensorCreateElementwiseBinary(
              cutensor_, &created.operation, da, modes_a.data(),
              CUTENSOR_OP_IDENTITY, dd, modes_d.data(), CUTENSOR_OP_IDENTITY,
              dd, modes_d.data(), CUTENSOR_OP_ADD, CUTENSOR_COMPUTE_DESC_64F),
          "cutensorCreateElementwiseBinary");
  });
  check(
      cutensorElementwiseBinaryExecute(cutensor_, entry.plan, &alpha, a.data(),
                                       &gamma, d.data(), d.data(), stream_),
      "cutensorElementwiseBinaryExecute");
}

void Engine::axpy(double alpha, const DeviceTensor& x, DeviceTensor& y) {
  check(cublasDaxpy_64(cublas_, static_cast<int64_t>(x.volume()), &alpha,
                       x.data(), 1, y.data(), 1),
        "cublasDaxpy");
}

double Engine::dot(const DeviceTensor& a, const DeviceTensor& b) {
  double result = 0.0;
  check(cublasDdot_64(cublas_, static_cast<int64_t>(a.volume()), a.data(), 1,
                      b.data(), 1, &result),
        "cublasDdot");
  return result;
}

double Engine::norm2(const DeviceTensor& a) {
  double result = 0.0;
  check(cublasDnrm2_64(cublas_, static_cast<int64_t>(a.volume()), a.data(), 1,
                       &result),
        "cublasDnrm2");
  return result;
}

EngineScope::EngineScope(Engine& engine) : previous_(current_engine) {
  check(cudaSetDevice(engine.device()), "cudaSetDevice");
  current_engine = &engine;
}

EngineScope::~EngineScope() {
  current_engine = previous_;
  if (previous_) (void)cudaSetDevice(previous_->device());
}

}  // namespace qdk::chemistry::utils::cutensor
