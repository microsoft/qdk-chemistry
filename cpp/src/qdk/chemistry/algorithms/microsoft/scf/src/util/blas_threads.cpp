// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/blas_threads.h>

#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>

// Thread-control API this build links against. The BLAS section of
// ../CMakeLists.txt passes the vendor down as
// QDK_CHEMISTRY_BLAS_VENDOR=QDK_CHEMISTRY_BLAS_VENDOR_<vendor>, which the IDs
// below turn into an integer the preprocessor can compare. Binding at link
// time is what guarantees we reach the BLAS our own calls go to -- a symbol
// found in the running process may belong to a different one -- and it has to
// be a compile-time choice: the untaken branches would otherwise reference
// symbols this build never links. Vendors with no such API (Accelerate on
// macOS, reference BLAS) leave the macro undefined and take the no-op path.
#define QDK_CHEMISTRY_BLAS_VENDOR_None 0
#define QDK_CHEMISTRY_BLAS_VENDOR_OpenBLAS 1
#define QDK_CHEMISTRY_BLAS_VENDOR_IntelMKL 2
#define QDK_CHEMISTRY_BLAS_VENDOR_BLIS 3

#if !defined(QDK_CHEMISTRY_BLAS_VENDOR)
#define QDK_CHEMISTRY_BLAS_VENDOR QDK_CHEMISTRY_BLAS_VENDOR_None
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_None
// An unknown identifier evaluates to 0 in #if, so a vendor with no ID above is
// indistinguishable from _None. Naming a vendor is a request for its API, so
// fail rather than silently take the no-op path.
#error "QDK_CHEMISTRY_BLAS_VENDOR names a BLAS vendor with no ID in blas_threads.cpp"
#endif

#if QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_OpenBLAS
extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
}
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_IntelMKL
extern "C" {
void MKL_Set_Num_Threads(int);
int MKL_Get_Max_Threads(void);
}
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_BLIS
// The header rather than a prototype: BLIS types the count as dim_t, whose
// width is a BLIS build option, and declaring it wrong is an ABI mismatch.
#if __has_include(<blis/blis.h>)
#include <blis/blis.h>
#else
#include <blis.h>
#endif
#endif

namespace qdk::chemistry::scf::util {

namespace {

/// @brief Request `n` BLAS threads; a no-op without a thread-control API --
/// hence [[maybe_unused]], for that empty body under -Wall -Wextra.
void blas_set_num_threads([[maybe_unused]] int n) {
#if QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_OpenBLAS
  openblas_set_num_threads(n);
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_IntelMKL
  MKL_Set_Num_Threads(n);
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_BLIS
  bli_thread_set_num_threads(static_cast<dim_t>(n));
#endif
}

/// @brief Whether a thread-control API is bound, warning once if not.
bool blas_thread_control_available() {
  static const bool available = [] {
#if QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_OpenBLAS
    QDK_LOGGER().debug("Using OpenBLAS thread control (bound at link time)");
    return true;
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_IntelMKL
    QDK_LOGGER().debug("Using Intel MKL thread control (bound at link time)");
    return true;
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_BLIS
    QDK_LOGGER().debug("Using BLIS thread control (bound at link time)");
    return true;
#else
    QDK_LOGGER().warn(
        "No BLAS thread-control API is bound into this build, so nested BLAS "
        "threading cannot be disabled automatically. Restrict your BLAS to one "
        "thread via its environment variable (OPENBLAS_NUM_THREADS, "
        "MKL_NUM_THREADS, BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS), or "
        "reconfigure with -DBLAS_VENDOR=<OpenBLAS|IntelMKL|BLIS> to name the "
        "vendor of the BLAS this build already links.");
    return false;
#endif
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

int blas_get_num_threads() {
#if QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_OpenBLAS
  return openblas_get_num_threads();
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_IntelMKL
  // MKL reports the maximum rather than a current count; they agree unless an
  // MKL_Set_Num_Threads_Local override is in effect, which we do not use.
  return MKL_Get_Max_Threads();
#elif QDK_CHEMISTRY_BLAS_VENDOR == QDK_CHEMISTRY_BLAS_VENDOR_BLIS
  return static_cast<int>(bli_thread_get_num_threads());
#else
  return 0;
#endif
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
