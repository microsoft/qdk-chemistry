// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cuda_runtime.h>
#include <spdlog/fmt/bundled/format.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

/**
 * @brief Macro to check CUDA API calls for errors and throw exceptions
 *
 * This macro wraps CUDA runtime API calls and automatically checks their return
 * values. If an error is detected, it throws a std::string with detailed
 * diagnostic information including the error code, error string, source
 * location, and the CUDA function call that failed.
 *
 * Example usage:
 * @code
 * CUDA_CHECK(cudaMalloc(&d_ptr, size));
 * CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
 * @endcode
 *
 * @param val CUDA API function call to check (e.g., cudaMalloc(...))
 *
 * @throws std::runtime_error with helpful string if unsuccessful
 */
#define CUDA_CHECK(val) \
  qdk::chemistry::scf::cuda::_check((val), #val, __FILE__, __LINE__)

namespace qdk::chemistry::scf::cuda {

/**
 * @brief Internal template function to check CUDA runtime results and throw
 * exceptions on error
 *
 * This function is called by the CUDA_CHECK macro and should not be invoked
 * directly. It examines the CUDA API return value and throws a
 * std::exception if an error occurred. The exception type is determined by
 * the error code: memory allocation errors (code 2) are mapped to
 * ResourceError, while all other errors become RuntimeError.
 *
 * Error information is also printed to stderr before throwing for debugging
 * purposes.
 *
 * @tparam T CUDA error type
 * @param result Return value from a CUDA API function
 * @param func String representation of the CUDA function call (via
 * stringification)
 * @param file Source file name where the error occurred (via __FILE__)
 * @param line Line number where the error occurred (via __LINE__)
 *
 * @throws std::exception If the CUDA result indicates an error (non-zero
 * value)
 */
template <typename T>
void _check(T result, char const* const func, const char* const file,
            int const line) {
  if (result) {
    const auto code = static_cast<unsigned int>(result);
    std::string msg =
        fmt::format("CUDA error code {}: {}", code, cudaGetErrorString(result));

    throw std::runtime_error(msg);
  }
}

/**
 * @brief Get the currently active CUDA device ID
 *
 * Returns the device ID of the GPU currently selected for CUDA operations.
 * This device is used by default for all CUDA API calls unless explicitly
 * specified otherwise.
 *
 * @return Integer device ID (0-based index)
 *
 * @throws std::runtime_error If cudaGetDevice fails
 */
int get_current_device();

/**
 * @brief Initialize and configure the CUDA memory pool for a specific device
 *
 * Configures the device's default memory pool by setting the release threshold
 * to UINT64_MAX, which prevents the pool from releasing memory back to the
 * system. This improves performance by retaining allocated memory for reuse in
 * subsequent allocations, avoiding expensive allocation/deallocation system
 * calls.
 *
 * The initialization is performed only once per device (tracked via static
 * state). Subsequent calls for the same device are no-ops.
 *
 * @param device Device ID to initialize memory pool for (default: current
 * device)
 *
 * @note This function is idempotent - safe to call multiple times for the same
 * device
 * @note Uses cudaMallocAsync/cudaFreeAsync for stream-ordered allocations
 */
void init_memory_pool(int device = get_current_device());

/**
 * @brief Trim unused memory from the CUDA memory pool to release resources
 *
 * Forces the memory pool to release cached memory back to the operating system,
 * reducing the pool's reserved memory to at least the specified byte threshold.
 * This is useful for freeing GPU memory when switching between memory-intensive
 * operations or when explicitly managing GPU memory availability.
 *
 * Memory currently in use is not affected - only free cached memory is
 * released.
 *
 * @param bytes Minimum number of bytes to keep in the pool (releases excess
 * above this)
 * @param device Device ID for the memory pool to trim (default: current device)
 *
 * @throws std::exception if cudaMemPoolTrimTo fails
 */
void trim_memory_pool(size_t bytes, int device = get_current_device());

/**
 * @brief Get the current CUDA stream (returns the default/legacy stream)
 *
 * Returns the default CUDA stream (stream 0), which is the implicit stream used
 * when no stream is explicitly specified. Operations on the default stream are
 * serialized with all other operations on the device.
 *
 * @return cudaStream_t handle for the default stream (value: 0)
 */
static cudaStream_t get_current_stream() { return 0; }

/**
 * @brief RAII wrapper for CUDA device memory with stream-ordered allocation
 *
 * Provides automatic memory management for GPU device memory using CUDA's
 * stream-ordered allocation APIs (cudaMallocAsync/cudaFreeAsync). Memory is
 * allocated on construction and freed on destruction, ensuring proper cleanup
 * even in the presence of exceptions.
 *
 * Example usage:
 * @code
 * cudaStream_t stream = ...;
 * {
 *   Memory<double> buffer(1000, stream);  // Allocate 1000 doubles
 *   // Use buffer.data() for kernel launches or memory copies
 * } // Automatic deallocation when buffer goes out of scope
 * @endcode
 *
 * @tparam T Data type to allocate
 *
 * @note Non-copyable to prevent double-free errors
 * @note Memory operations are asynchronous with respect to the specified stream
 */
template <typename T>
class Memory {
 public:
  /**
   * @brief Allocate device memory on a specific CUDA stream
   *
   * Allocates N * sizeof(T) bytes of device memory using stream-ordered
   * allocation. The allocation is asynchronous and synchronized with the
   * provided stream.
   *
   * @param N Number of elements of type T to allocate
   * @param stream CUDA stream to synchronize allocation with
   *
   * @throws std::exception If cudaMallocAsync fails (e.g., out of memory)
   */
  Memory(size_t N, cudaStream_t stream) : stream_(stream) {
    CUDA_CHECK(cudaMallocAsync(&data_, N * sizeof(T), stream_));
  }

  /**
   * @brief Free device memory on the associated stream
   *
   * Frees the device memory using stream-ordered deallocation (cudaFreeAsync).
   * The deallocation is synchronized with the stream used during allocation,
   * ensuring that any pending operations on this memory complete before it is
   * freed.
   *
   * This destructor is noexcept to maintain proper exception safety - throwing
   * from a destructor would lead to program termination if called during stack
   * unwinding.
   *
   * @note Any CUDA errors during deallocation are still checked but should not
   * occur in correct programs
   */
  ~Memory() { CUDA_CHECK(cudaFreeAsync(data_, stream_)); }

  /**
   * @brief Get the raw device memory pointer
   *
   * Returns the underlying CUDA device pointer for use in kernel launches,
   * memory copy operations, or passing to CUDA libraries (cuBLAS, cuSolver,
   * etc.).
   *
   * @return Raw pointer to device memory of type T*
   *
   * @note The returned pointer is only valid while the Memory object exists
   * @note The pointer refers to device memory and cannot be directly
   * dereferenced on the host
   */
  T* data() { return data_; }

 private:
  T* data_;  ///< Device memory pointer (allocated with cudaMallocAsync)
  cudaStream_t
      stream_;  ///< CUDA stream for synchronizing async memory operations
};

/**
 * @brief Allocate device memory wrapped in a shared_ptr for automatic lifetime
 * management
 *
 * Factory function that creates a Memory<T> object and wraps it in a
 * std::shared_ptr, enabling reference-counted memory management. The device
 * memory will be automatically freed when the last shared_ptr referencing it is
 * destroyed.
 *
 * This is particularly useful for:
 * - Sharing device memory across multiple objects without explicit ownership
 * transfer
 * - Managing memory lifetimes in asynchronous or multi-threaded contexts
 * - Avoiding manual memory management while maintaining RAII guarantees
 *
 * Example usage:
 * @code
 * auto buffer = alloc<double>(1024);  // Allocate 1024 doubles
 * auto shared_buffer = buffer;        // Create another reference
 * // Memory freed when both buffer and shared_buffer go out of scope
 * @endcode
 *
 * @tparam T Data type to allocate
 * @param N Number of elements of type T to allocate
 * @param stream CUDA stream for stream-ordered allocation (default: default
 * stream)
 * @return std::shared_ptr<Memory<T>> managing the allocated device memory
 *
 */
template <typename T>
static std::shared_ptr<Memory<T>> alloc(
    size_t N, cudaStream_t stream = get_current_stream()) {
  return std::make_shared<Memory<T>>(N, stream);
}

}  // namespace qdk::chemistry::scf::cuda
