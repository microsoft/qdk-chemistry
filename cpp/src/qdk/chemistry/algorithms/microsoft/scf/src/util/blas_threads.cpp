// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>

// The thread-control API this build links against, named by CMake (see the BLAS
// section of ../CMakeLists.txt). Binding here is the only way to reach the BLAS
// our own calls go to: a symbol found in the running process may belong to a
// different BLAS, and a statically linked one exports nothing to find.
#if defined(QDK_CHEMISTRY_BLAS_LINKED_SET_FN)
extern "C" {
void QDK_CHEMISTRY_BLAS_LINKED_SET_FN(QDK_CHEMISTRY_BLAS_LINKED_TYPE);
QDK_CHEMISTRY_BLAS_LINKED_TYPE QDK_CHEMISTRY_BLAS_LINKED_GET_FN(void);
}
#endif

namespace qdk::chemistry::scf::util {

namespace {

#if !defined(QDK_CHEMISTRY_BLAS_VENDOR)
#define QDK_CHEMISTRY_BLAS_VENDOR ""
#endif

// Raw `BLAS_VENDOR` string from the CMake BLAS search (e.g. "OpenBLAS",
// "IntelMKL", "ReferenceBLAS"), used only to make the "no thread-control API"
// warning below actionable. sizeof > 1 means the string is not empty.
constexpr const char* kConfiguredBlasVendor =
    sizeof(QDK_CHEMISTRY_BLAS_VENDOR) > 1 ? QDK_CHEMISTRY_BLAS_VENDOR
                                          : "unknown";

/// @brief Resolved thread-control API of the BLAS backend bound at link time.
struct BlasThreadApi {
  BlasVendor vendor = BlasVendor::Unknown;
  std::function<int()> get_num_threads;
  std::function<void(int)> set_num_threads;

  bool valid() const {
    return static_cast<bool>(get_num_threads) &&
           static_cast<bool>(set_num_threads);
  }
};

/**
 * @brief Narrow a backend's thread count to int.
 *
 * BLIS types it as dim_t, 32- or 64-bit depending on its build; the table
 * declares the 64-bit form, and a 32-bit callee returns its value in the low
 * half. Thread counts are small, so keeping the low 32 bits is lossless.
 */
template <typename T>
int narrow_thread_count(T value) {
  return static_cast<int>(static_cast<std::int32_t>(value));
}

/// @brief Thread-control API bound at link time, if CMake found one.
BlasThreadApi linked_blas_thread_api() {
#if defined(QDK_CHEMISTRY_BLAS_LINKED_SET_FN)
  return {
      BlasVendor::QDK_CHEMISTRY_BLAS_LINKED_VENDOR,
      [] { return narrow_thread_count(QDK_CHEMISTRY_BLAS_LINKED_GET_FN()); },
      [](int n) {
        QDK_CHEMISTRY_BLAS_LINKED_SET_FN(
            static_cast<QDK_CHEMISTRY_BLAS_LINKED_TYPE>(n));
      }};
#else
  return {};
#endif
}

const BlasThreadApi& blas_thread_api() {
  static const BlasThreadApi api = [] {
    if (BlasThreadApi linked = linked_blas_thread_api(); linked.valid()) {
      QDK_LOGGER().debug("Using {} BLAS thread control (bound at link time)",
                         to_string(linked.vendor));
      return linked;
    }

    QDK_LOGGER().warn(
        "No supported BLAS thread-control API found (configured BLAS vendor: "
        "'{}'). Nested BLAS threading cannot be disabled automatically; if you "
        "see oversubscription or wrong results, restrict your BLAS to a single "
        "thread via its environment variable (e.g. OPENBLAS_NUM_THREADS, "
        "MKL_NUM_THREADS, BLIS_NUM_THREADS or VECLIB_MAXIMUM_THREADS), or "
        "configure with -DQDK_CHEMISTRY_BLAS_THREAD_API=<backend>.",
        kConfiguredBlasVendor);
    return BlasThreadApi{};
  }();
  return api;
}

/**
 * @brief Shared state backing ScopedBlasThreads.
 *
 * The BLAS thread count is process-global, so this is too. `saved` is
 * meaningful only while `depth > 0`; both are touched only under `mutex`.
 */
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

const char* to_string(BlasVendor vendor) {
  switch (vendor) {
#define QDK_CHEMISTRY_BLAS_VENDOR_LABEL(token, name, label, set_fn, get_fn, \
                                        type)                               \
  case BlasVendor::name:                                                    \
    return label;
    QDK_CHEMISTRY_BLAS_BACKEND_TABLE(QDK_CHEMISTRY_BLAS_VENDOR_LABEL)
#undef QDK_CHEMISTRY_BLAS_VENDOR_LABEL
    case BlasVendor::Unknown:
      break;
  }
  return "unknown";
}

BlasVendor detected_blas_vendor() { return blas_thread_api().vendor; }

int get_blas_num_threads() {
  const BlasThreadApi& api = blas_thread_api();
  return api.valid() ? api.get_num_threads() : 0;
}

ScopedBlasThreads::ScopedBlasThreads() {
  const BlasThreadApi& api = blas_thread_api();
  if (!api.valid()) return;

  BlasThreadState& state = blas_thread_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.depth == 0) {
    // Decline rather than guess: a backend that cannot report leaves nothing
    // to restore on exit.
    const int current = api.get_num_threads();
    if (current < 1) return;
    state.saved = current;
    api.set_num_threads(1);
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
    blas_thread_api().set_num_threads(state.saved);
  }
}

}  // namespace qdk::chemistry::scf::util
