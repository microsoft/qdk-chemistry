// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/config.h>

namespace qdk::chemistry::scf::util {

/**
 * @brief BLAS backends whose threading can be controlled at runtime.
 *
 * Enumerators are generated from QDK_CHEMISTRY_BLAS_BACKEND_TABLE in
 * scf/config.h, which CMake emits from the table it probes with.
 * BlasVendor::Unknown means no recognized thread-control API was found.
 */
enum class BlasVendor {
  Unknown,
#define QDK_CHEMISTRY_BLAS_VENDOR_ENUMERATOR(token, vendor, label, set_fn, \
                                             get_fn, type)                 \
  vendor,
  QDK_CHEMISTRY_BLAS_BACKEND_TABLE(QDK_CHEMISTRY_BLAS_VENDOR_ENUMERATOR)
#undef QDK_CHEMISTRY_BLAS_VENDOR_ENUMERATOR
};

/// @brief Human readable name of a BLAS vendor.
const char* to_string(BlasVendor vendor);

/**
 * @brief BLAS backend whose thread count this build controls (resolved once,
 * on first use).
 *
 * The backend is bound at link time where CMake could probe one, and looked
 * up in the running process otherwise.
 *
 * @return The bound vendor, or BlasVendor::Unknown when no supported
 *         thread-control API is available, in which case ScopedBlasThreads is
 *         a no-op.
 */
BlasVendor detected_blas_vendor();

/**
 * @brief Current BLAS thread count.
 *
 * Exposed for diagnostics and to let tests verify that ScopedBlasThreads
 * pins and restores the count; production code should use ScopedBlasThreads
 * rather than managing the count itself.
 *
 * @return Number of threads, or 0 if the BLAS backend cannot report it.
 */
int get_blas_num_threads();

/**
 * @brief RAII guard that pins BLAS to a fixed thread count while active and
 * restores the previous count once the outermost guard exits.
 *
 * Motivation: GauXC's OpenMP-parallel grid loop calls BLAS from many threads
 * at once. If BLAS is also multi-threaded, those threads collide inside the
 * BLAS backend's own shared worker pool, oversubscribing the machine and, for
 * some backends, corrupting results. Pinning BLAS to a single thread for the
 * duration of such a region avoids that.
 *
 * The BLAS thread count is process-global state, so this guard uses a shared,
 * mutex-protected nesting depth instead of per-instance state: only the first
 * guard to start changes it, and only the last one to finish restores it. This
 * keeps concurrent/recursive use safe.
 *
 * If the BLAS backend in use exposes no thread-control API, the guard is a
 * no-op (a one-time warning is logged).
 */
class ScopedBlasThreads {
 public:
  /// @param num_threads Thread count to pin BLAS to while the guard is alive.
  explicit ScopedBlasThreads(int num_threads = 1);
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
