// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <functional>
#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>

// Thread-control API this build links against, selected by the BLAS vendor the
// configure step found (see the BLAS section of ../CMakeLists.txt). Link-time
// binding is the only way to reach the BLAS our own calls go to: a symbol found
// in the running process may belong to a different BLAS, and a statically
// linked one exports nothing to find. Vendors without such an API -- Accelerate
// on macOS, reference BLAS, or whatever a Windows build resolves to -- define
// none of these and fall through to the no-op path below.
#if defined(QDK_CHEMISTRY_BLAS_VENDOR_OPENBLAS)
extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_VENDOR_INTELMKL)
extern "C" {
void MKL_Set_Num_Threads(int);
int MKL_Get_Max_Threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_VENDOR_BLIS)
// The header rather than a hand-written prototype: BLIS types the count as
// dim_t, which is 32- or 64-bit depending on how BLIS was built, and declaring
// the wrong width is an ABI mismatch (GCC LTO: -Wlto-type-mismatch).
#if __has_include(<blis/blis.h>)
#include <blis/blis.h>
#else
#include <blis.h>
#endif
#endif

namespace qdk::chemistry::scf::util {

namespace {

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

/// @brief Thread-control API bound at link time, if this build has one.
BlasThreadApi linked_blas_thread_api() {
#if defined(QDK_CHEMISTRY_BLAS_VENDOR_OPENBLAS)
  return {BlasVendor::OpenBLAS, [] { return openblas_get_num_threads(); },
          [](int n) { openblas_set_num_threads(n); }};
#elif defined(QDK_CHEMISTRY_BLAS_VENDOR_INTELMKL)
  // MKL reports the maximum rather than a current count; they agree except
  // while an MKL_Set_Num_Threads_Local override is in effect, which we do not
  // use. Narrowing is not a concern: thread counts are small.
  return {BlasVendor::IntelMKL, [] { return MKL_Get_Max_Threads(); },
          [](int n) { MKL_Set_Num_Threads(n); }};
#elif defined(QDK_CHEMISTRY_BLAS_VENDOR_BLIS)
  return {BlasVendor::BLIS,
          [] { return static_cast<int>(bli_thread_get_num_threads()); },
          [](int n) { bli_thread_set_num_threads(static_cast<dim_t>(n)); }};
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
        "No BLAS thread-control API is bound into this build. Nested BLAS "
        "threading cannot be disabled automatically; if you see "
        "oversubscription or wrong results, restrict your BLAS to a single "
        "thread via its environment variable (e.g. OPENBLAS_NUM_THREADS, "
        "MKL_NUM_THREADS, BLIS_NUM_THREADS or VECLIB_MAXIMUM_THREADS), or "
        "reconfigure with -DBLAS_VENDOR=<OpenBLAS|IntelMKL|BLIS>: supplying "
        "BLAS_LIBRARIES skips the search that would otherwise detect the "
        "vendor, so name the vendor of the BLAS this build already links.");
    return BlasThreadApi{};
  }();
  return api;
}

/// @brief Shared state backing ScopedBlasThreads. The BLAS thread count is
/// process-global, so this is too; `saved` is meaningful only while
/// `depth > 0`, and both are touched only under `mutex`.
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
    case BlasVendor::OpenBLAS:
      return "OpenBLAS";
    case BlasVendor::IntelMKL:
      return "Intel MKL";
    case BlasVendor::BLIS:
      return "BLIS";
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
    // Decline rather than guess: nothing to restore on exit.
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
