// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf::util {

/// @brief Current BLAS thread count, or 0 if this build binds no thread-control
/// API. For diagnostics and tests; prefer ScopedBlasThreads.
int blas_get_num_threads();

/// @brief Request `n` BLAS threads process-wide; ignored if `n < 1`. For
/// diagnostics and tests; prefer ScopedBlasThreads.
void blas_set_num_threads(int n);

/**
 * @brief RAII guard that pins BLAS to a single thread while active and
 * restores the previous count once the outermost guard exits.
 *
 * GauXC's OpenMP-parallel grid loop calls BLAS from many threads at once; if
 * BLAS is also multi-threaded those threads collide inside its shared worker
 * pool, oversubscribing the machine and, for some backends, corrupting results.
 *
 * The count is process-global, so nesting is tracked by a mutex-protected
 * depth: the first guard pins, the last restores. That is also why the count
 * is not configurable -- a nested guard could not be honored without
 * overriding the count an enclosing one relies on.
 */
class ScopedBlasThreads {
 public:
  ScopedBlasThreads();
  ~ScopedBlasThreads();

  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads(ScopedBlasThreads&&) = delete;
  ScopedBlasThreads& operator=(ScopedBlasThreads&&) = delete;

  /// @brief Whether this guard holds the BLAS thread count. False when no
  /// thread-control API is bound, or the backend cannot report a count.
  bool active() const { return active_; }

 private:
  bool active_ = false;
};

}  // namespace qdk::chemistry::scf::util
