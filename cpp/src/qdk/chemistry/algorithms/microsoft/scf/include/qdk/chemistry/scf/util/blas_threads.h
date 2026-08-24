// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf::util {

/// @brief BLAS backends whose threading this build knows how to control.
/// Which one (if any) is bound is decided at configure time; see the BLAS
/// section of scf/src/CMakeLists.txt.
enum class BlasVendor { Unknown, OpenBLAS, IntelMKL, BLIS };

/// @brief Human readable name of a BLAS vendor.
const char* to_string(BlasVendor vendor);

/// @brief BLAS backend whose thread count this build controls, bound at link
/// time by CMake. Unknown means ScopedBlasThreads is a no-op.
BlasVendor detected_blas_vendor();

/// @brief Current BLAS thread count, or 0 if the backend cannot report it.
/// For diagnostics and tests; production code should use ScopedBlasThreads.
int get_blas_num_threads();

/**
 * @brief RAII guard that pins BLAS to a single thread while active and
 * restores the previous count once the outermost guard exits.
 *
 * GauXC's OpenMP-parallel grid loop calls BLAS from many threads at once; if
 * BLAS is also multi-threaded those threads collide inside its shared worker
 * pool, oversubscribing the machine and, for some backends, corrupting results.
 *
 * The count is process-global, so nesting is tracked by a shared,
 * mutex-protected depth: the first guard pins, the last restores. That is also
 * why the count is not configurable -- a nested guard could not be honored
 * without overriding the count an enclosing one relies on.
 *
 * A no-op if the backend exposes no thread-control API (warned once) or cannot
 * report its current count.
 */
class ScopedBlasThreads {
 public:
  ScopedBlasThreads();
  ~ScopedBlasThreads();

  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads(ScopedBlasThreads&&) = delete;
  ScopedBlasThreads& operator=(ScopedBlasThreads&&) = delete;

  /// @brief Whether this guard actually changed/holds the BLAS thread count.
  bool active() const { return active_; }

 private:
  bool active_ = false;
};

}  // namespace qdk::chemistry::scf::util
