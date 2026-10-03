// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/blas_threads.h>

#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>

#include "blas_threads_backend.h"

namespace qdk::chemistry::scf::util {

namespace {

/// @brief Whether a thread-control API is bound, warning once if not.
bool blas_thread_control_available() {
  static const bool available = [] {
    const char* const name = detail::blas_backend_name();
    if (name != nullptr) {
      QDK_LOGGER().debug("Using {} thread control (bound at link time)", name);
      return true;
    }
    QDK_LOGGER().warn(
        "No BLAS thread-control API is bound into this build, so nested BLAS "
        "threading cannot be disabled automatically. Restrict your BLAS to one "
        "thread via its environment variable (OPENBLAS_NUM_THREADS, "
        "MKL_NUM_THREADS, BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS), or "
        "rebuild against a BLAS that exports one (the configure step reports "
        "which backend, if any, was bound).");
    return false;
  }();
  return available;
}

/// @brief Shared state backing ScopedBlasThreads: the BLAS thread count is
/// process-global, so this is too. `saved` is meaningful only while
/// `depth > 0`; both are touched only under `mutex`.
struct BlasThreadState {
  std::mutex mutex;
  int depth = 0;
  int saved = 0;
};

BlasThreadState& blas_thread_state() {
  static BlasThreadState state;
  return state;
}

}  // namespace

int blas_get_num_threads() { return detail::blas_backend_get_num_threads(); }

void blas_set_num_threads(int n) {
  if (n < 1) return;
  detail::blas_backend_set_num_threads(n);
}

ScopedBlasThreads::ScopedBlasThreads() {
  if (!blas_thread_control_available()) return;

  BlasThreadState& state = blas_thread_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.depth == 0) {
    // Decline rather than guess: nothing to restore on exit.
    const int current = blas_get_num_threads();
    if (current < 1) return;
    state.saved = current;
    blas_set_num_threads(1);
  }
  // After the fallible part, so a declined guard leaves no depth behind.
  ++state.depth;
  active_ = true;
}

ScopedBlasThreads::~ScopedBlasThreads() {
  if (!active_) return;

  BlasThreadState& state = blas_thread_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (--state.depth == 0) {
    blas_set_num_threads(state.saved);
  }
}

}  // namespace qdk::chemistry::scf::util
