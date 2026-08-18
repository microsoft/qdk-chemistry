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

// The thread-control API this build links against, named by CMake (see the BLAS
// section of ../CMakeLists.txt). Binding directly is the most accurate answer
// and the only one that works for a statically linked BLAS, whose symbols no
// module exports and which therefore cannot be discovered at runtime.
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
  return {BlasVendor::QDK_CHEMISTRY_BLAS_LINKED_VENDOR,
          [] { return narrow_thread_count(QDK_CHEMISTRY_BLAS_LINKED_GET_FN()); },
          [](int n) {
            QDK_CHEMISTRY_BLAS_LINKED_SET_FN(
                static_cast<QDK_CHEMISTRY_BLAS_LINKED_TYPE>(n));
          }};
#else
  return {};
#endif
}

/**
 * @brief Resolve one backend's thread-control API in the running process.
 *
 * @tparam T The type the backend uses for a thread count.
 * @return The API, or an invalid BlasThreadApi if either half is missing.
 */
template <typename T>
BlasThreadApi try_backend(BlasVendor vendor, const char* set_name,
                          const char* get_name) {  using SetFn = void (*)(T);
  using GetFn = T (*)(void);
  auto set_fn = find_function<SetFn>(set_name);
  auto get_fn = find_function<GetFn>(get_name);
  if (!set_fn || !get_fn) return {};
  return {vendor, [get_fn] { return narrow_thread_count(get_fn()); },
          [set_fn](int n) { set_fn(static_cast<T>(n)); }};
}

// Runtime probes, generated from the backend table so they keep the order CMake
// probes at configure time.
#define QDK_CHEMISTRY_BLAS_RUNTIME_PROBE(token, vendor, label, set_fn, get_fn, \
                                         type)                                \
  +[]() -> BlasThreadApi {                                                     \
    return try_backend<type>(BlasVendor::vendor, #set_fn, #get_fn);            \
  },
constexpr BlasThreadApi (*kRuntimeProbes[])() = {
    QDK_CHEMISTRY_BLAS_BACKEND_TABLE(QDK_CHEMISTRY_BLAS_RUNTIME_PROBE)};
#undef QDK_CHEMISTRY_BLAS_RUNTIME_PROBE

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

    // Fall back to whichever backend is loaded in this process.
    for (const auto probe : kRuntimeProbes) {
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
#define QDK_CHEMISTRY_BLAS_VENDOR_LABEL(token, name, label, set_fn, get_fn, \
                                        type)                              \
  case BlasVendor::name:                                                   \
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
