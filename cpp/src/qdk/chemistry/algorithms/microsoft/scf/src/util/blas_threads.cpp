// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <qdk/chemistry/utils/logger.hpp>

#if defined(_WIN32)
// Keep windows.h from pulling in unrelated headers and from defining the
// min/max macros, which break standard headers used elsewhere.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// The thread-control API of the BLAS this build links against, as probed by
// CMake (see the BLAS section of ../CMakeLists.txt). At most one of the
// QDK_CHEMISTRY_BLAS_LINKS_* macros is defined. Binding directly is both the
// most accurate answer (it is by construction the BLAS our calls end up in)
// and the only one that works for a statically linked BLAS, whose symbols no
// module exports and which therefore cannot be discovered at runtime.
#if defined(QDK_CHEMISTRY_BLAS_LINKS_OPENBLAS)
extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_INTELMKL)
extern "C" {
void MKL_Set_Num_Threads(int);
int MKL_Get_Max_Threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_BLIS)
// BLIS types the thread count as dim_t, which is 32- or 64-bit depending on
// how BLIS was configured. std::int64_t is compatible with both: a 32-bit
// callee reads the low half of the argument register, and thread counts are
// small enough that truncating the returned value is lossless.
extern "C" {
void bli_thread_set_num_threads(std::int64_t);
std::int64_t bli_thread_get_num_threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_FLEXIBLAS)
extern "C" {
void flexiblas_set_num_threads(int);
int flexiblas_get_num_threads(void);
}
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_NVPL)
extern "C" {
void nvpl_blas_set_num_threads(int);
int nvpl_blas_get_max_threads(void);
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

/**
 * @brief Look up an optional symbol in the current process.
 *
 * The BLAS backend is a build/deployment choice of the user, so no vendor
 * specific symbol may be linked at all. Resolving the thread-control entry
 * points at runtime keeps this file free of link-time dependencies on any
 * particular BLAS implementation.
 */
void* find_symbol(const char* name) {
#if defined(_WIN32)
  // Modules that may export a BLAS thread-control API. Only already loaded
  // modules are inspected; nothing new is brought into the process.
  static const char* const kModules[] = {
      "openblas.dll",  "libopenblas.dll",    "mkl_rt.dll",
      "mkl_rt.2.dll",  "blis.dll",           "AOCL-LibBlis-Win-MT-dll.dll",
      "flexiblas.dll", "nvpl_blas_core.dll",
  };
  for (const char* module_name : kModules) {
    if (HMODULE handle = GetModuleHandleA(module_name)) {
      if (FARPROC symbol = GetProcAddress(handle, name)) {
        return reinterpret_cast<void*>(symbol);
      }
    }
  }
  // Statically linked BLAS: the symbol may live in this module or the
  // executable itself.
  HMODULE self = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&find_symbol), &self);
  for (HMODULE handle : {self, GetModuleHandleA(nullptr)}) {
    if (handle == nullptr) continue;
    if (FARPROC symbol = GetProcAddress(handle, name)) {
      return reinterpret_cast<void*>(symbol);
    }
  }
  return nullptr;
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

template <typename Fn>
Fn find_function(const char* name) {
  return reinterpret_cast<Fn>(find_symbol(name));
}

/// @brief Resolved thread-control API of whichever BLAS backend is loaded.
struct BlasThreadApi {
  BlasVendor vendor = BlasVendor::Unknown;
  std::function<int()> get_num_threads;
  std::function<void(int)> set_num_threads;

  bool valid() const {
    return static_cast<bool>(get_num_threads) &&
           static_cast<bool>(set_num_threads);
  }
};

/// @brief Thread-control API bound at link time, if CMake found one.
BlasThreadApi linked_blas_thread_api() {
#if defined(QDK_CHEMISTRY_BLAS_LINKS_OPENBLAS)
  return {BlasVendor::OpenBLAS, [] { return openblas_get_num_threads(); },
          [](int n) { openblas_set_num_threads(n); }};
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_INTELMKL)
  return {BlasVendor::IntelMKL, [] { return MKL_Get_Max_Threads(); },
          [](int n) { MKL_Set_Num_Threads(n); }};
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_BLIS)
  return {
      BlasVendor::BLIS,
      [] {
        return static_cast<int>(
            static_cast<std::int32_t>(bli_thread_get_num_threads()));
      },
      [](int n) { bli_thread_set_num_threads(static_cast<std::int64_t>(n)); }};
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_FLEXIBLAS)
  return {BlasVendor::FlexiBLAS, [] { return flexiblas_get_num_threads(); },
          [](int n) { flexiblas_set_num_threads(n); }};
#elif defined(QDK_CHEMISTRY_BLAS_LINKS_NVPL)
  return {BlasVendor::NVPL, [] { return nvpl_blas_get_max_threads(); },
          [](int n) { nvpl_blas_set_num_threads(n); }};
#else
  return {};
#endif
}

BlasThreadApi try_openblas() {
  // Also matches OpenBLAS-compatible drop-ins such as ARM Performance
  // Libraries' OpenBLAS interface.
  using SetFn = void (*)(int);
  using GetFn = int (*)(void);
  auto set_fn = find_function<SetFn>("openblas_set_num_threads");
  auto get_fn = find_function<GetFn>("openblas_get_num_threads");
  if (!set_fn || !get_fn) return {};
  return {BlasVendor::OpenBLAS, [get_fn] { return get_fn(); },
          [set_fn](int n) { set_fn(n); }};
}

BlasThreadApi try_mkl() {
  // MKL_Set_Num_Threads / MKL_Get_Max_Threads are the C entry points (the
  // lowercase spellings are the Fortran ones and take pointers).
  using SetFn = void (*)(int);
  using GetFn = int (*)(void);
  auto set_fn = find_function<SetFn>("MKL_Set_Num_Threads");
  auto get_fn = find_function<GetFn>("MKL_Get_Max_Threads");
  if (!set_fn || !get_fn) return {};
  return {BlasVendor::IntelMKL, [get_fn] { return get_fn(); },
          [set_fn](int n) { set_fn(n); }};
}

BlasThreadApi try_blis() {
  // BLIS (and AMD's AOCL-BLAS fork) types the thread count as dim_t, which is
  // either 32- or 64-bit depending on how BLIS was configured. Calling through
  // 64-bit prototypes is safe for both: a 32-bit callee simply reads the low
  // half of the argument register, and the returned thread count is small
  // enough that truncating the result is lossless.
  using SetFn = void (*)(std::int64_t);
  using GetFn = std::int64_t (*)(void);
  auto set_fn = find_function<SetFn>("bli_thread_set_num_threads");
  auto get_fn = find_function<GetFn>("bli_thread_get_num_threads");
  if (!set_fn || !get_fn) return {};
  return {BlasVendor::BLIS,
          [get_fn] {
            return static_cast<int>(static_cast<std::int32_t>(get_fn()));
          },
          [set_fn](int n) { set_fn(static_cast<std::int64_t>(n)); }};
}

BlasThreadApi try_flexiblas() {
  using SetFn = void (*)(int);
  using GetFn = int (*)(void);
  auto set_fn = find_function<SetFn>("flexiblas_set_num_threads");
  auto get_fn = find_function<GetFn>("flexiblas_get_num_threads");
  if (!set_fn || !get_fn) return {};
  return {BlasVendor::FlexiBLAS, [get_fn] { return get_fn(); },
          [set_fn](int n) { set_fn(n); }};
}

BlasThreadApi try_nvpl() {
  using SetFn = void (*)(int);
  using GetFn = int (*)(void);
  auto set_fn = find_function<SetFn>("nvpl_blas_set_num_threads");
  auto get_fn = find_function<GetFn>("nvpl_blas_get_max_threads");
  if (!set_fn || !get_fn) return {};
  return {BlasVendor::NVPL, [get_fn] { return get_fn(); },
          [set_fn](int n) { set_fn(n); }};
}

const BlasThreadApi& blas_thread_api() {
  static const BlasThreadApi api = [] {
    // Prefer the API bound at link time: that BLAS is by construction the one
    // our calls go to, and it is the only option when BLAS is linked
    // statically (no module exports the symbols to look up at runtime).
    if (BlasThreadApi linked = linked_blas_thread_api(); linked.valid()) {
      QDK_LOGGER().debug("Using {} BLAS thread control (bound at link time)",
                         to_string(linked.vendor));
      return linked;
    }

    // Same fixed order as the CMake probe: a dispatch layer can also expose
    // its backend's API, so ask the dispatcher first; the rest are mutually
    // exclusive in practice.
    using Probe = BlasThreadApi (*)();
    for (const Probe probe :
         {try_flexiblas, try_openblas, try_mkl, try_blis, try_nvpl}) {
      BlasThreadApi api = probe();
      if (api.valid()) {
        QDK_LOGGER().debug("Using {} BLAS thread control (found at runtime)",
                           to_string(api.vendor));
        return api;
      }
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

std::mutex& thread_count_mutex() {
  static std::mutex mutex;
  return mutex;
}

int& scope_depth() {
  static int depth = 0;
  return depth;
}

int& saved_num_threads() {
  static int num_threads = 1;
  return num_threads;
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
    case BlasVendor::FlexiBLAS:
      return "FlexiBLAS";
    case BlasVendor::NVPL:
      return "NVPL BLAS";
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

ScopedBlasThreads::ScopedBlasThreads(int num_threads) {
  const BlasThreadApi& api = blas_thread_api();
  if (!api.valid() || num_threads < 1) return;

  std::lock_guard<std::mutex> lock(thread_count_mutex());
  if (scope_depth()++ == 0) {
    const int current = api.get_num_threads();
    saved_num_threads() = current > 0 ? current : 1;
    api.set_num_threads(num_threads);
  }
  active_ = true;
}

ScopedBlasThreads::~ScopedBlasThreads() {
  if (!active_) return;

  std::lock_guard<std::mutex> lock(thread_count_mutex());
  if (--scope_depth() == 0) {
    blas_thread_api().set_num_threads(saved_num_threads());
  }
}

}  // namespace qdk::chemistry::scf::util
