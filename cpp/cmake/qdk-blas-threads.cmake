# Selects the BLAS thread-control backend (see scf/src/util/blas_threads_backend.h).
#
# One small blas_threads_<vendor>.cpp implements the thread-count API per
# vendor; this picks the one that actually compiles and links against the BLAS
# resolved by this build. A link probe rather than a BLAS_VENDOR comparison,
# because BLAS_LIBRARIES can be set with no vendor label at all, and a
# correctly labelled facade (libblas.so.3 over OpenBLAS) exports none of the
# openblas_ entry points.

set(QDK_CHEMISTRY_BLAS_VENDORS OpenBLAS IntelMKL BLIS)
set(QDK_CHEMISTRY_BLAS_SRC_OpenBLAS blas_threads_openblas.cpp)
set(QDK_CHEMISTRY_BLAS_SRC_IntelMKL blas_threads_mkl.cpp)
set(QDK_CHEMISTRY_BLAS_SRC_BLIS     blas_threads_blis.cpp)
set(QDK_CHEMISTRY_BLAS_SYMS_OpenBLAS openblas_set_num_threads openblas_get_num_threads)
set(QDK_CHEMISTRY_BLAS_SYMS_IntelMKL MKL_Set_Num_Threads MKL_Get_Max_Threads)
set(QDK_CHEMISTRY_BLAS_SYMS_BLIS bli_thread_set_num_threads bli_thread_get_num_threads)

set(QDK_CHEMISTRY_BLAS_THREAD_API "AUTO" CACHE STRING
    "BLAS thread-control backend: AUTO, NONE or one of ${QDK_CHEMISTRY_BLAS_VENDORS}")
set_property(CACHE QDK_CHEMISTRY_BLAS_THREAD_API PROPERTY STRINGS
    AUTO ${QDK_CHEMISTRY_BLAS_VENDORS} NONE)

# Sets ${out_src} to the backend source to compile, and the cache variables
# QDK_CHEMISTRY_BLAS_LINK / QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS.
function(qdk_select_blas_thread_backend src_dir out_src)
  if(TARGET BLAS::BLAS)
    set(link BLAS::BLAS)
  else()
    set(link ${BLAS_LIBRARIES})
  endif()

  if(QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "AUTO")
    set(candidates ${QDK_CHEMISTRY_BLAS_VENDORS})
  elseif(QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "NONE")
    set(candidates "")
  elseif(QDK_CHEMISTRY_BLAS_THREAD_API IN_LIST QDK_CHEMISTRY_BLAS_VENDORS)
    set(candidates ${QDK_CHEMISTRY_BLAS_THREAD_API})
  else()
    message(FATAL_ERROR
      "Invalid QDK_CHEMISTRY_BLAS_THREAD_API='${QDK_CHEMISTRY_BLAS_THREAD_API}'. "
      "Expected AUTO, NONE or one of: ${QDK_CHEMISTRY_BLAS_VENDORS}.")
  endif()

  # try_compile caches its result variable, so probes run once -- but they must
  # not outlive the BLAS they were run against.
  string(SHA256 key "${link}|${BLAS_INCLUDE_DIRS}")
  if(NOT key STREQUAL "${QDK_CHEMISTRY_BLAS_PROBE_KEY}")
    foreach(vendor IN LISTS QDK_CHEMISTRY_BLAS_VENDORS)
      unset(QDK_CHEMISTRY_BLAS_PROBE_${vendor} CACHE)
    endforeach()
    set(QDK_CHEMISTRY_BLAS_PROBE_KEY "${key}" CACHE INTERNAL "")
  endif()

  set(backend "")
  foreach(vendor IN LISTS candidates)
    if(link AND NOT backend)
      try_compile(QDK_CHEMISTRY_BLAS_PROBE_${vendor}
        "${CMAKE_CURRENT_BINARY_DIR}/blas_probe_${vendor}"
        SOURCES
          "${src_dir}/${QDK_CHEMISTRY_BLAS_SRC_${vendor}}"
          "${src_dir}/blas_threads_probe.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${BLAS_INCLUDE_DIRS}"
        LINK_LIBRARIES ${link}
        CXX_STANDARD ${CMAKE_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)
      if(QDK_CHEMISTRY_BLAS_PROBE_${vendor})
        set(backend ${vendor})
      endif()
    endif()
  endforeach()

  if(backend)
    message(STATUS "QDK Chemistry BLAS thread control: ${backend}")
    set(${out_src} "${src_dir}/${QDK_CHEMISTRY_BLAS_SRC_${backend}}" PARENT_SCOPE)
    set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS "${QDK_CHEMISTRY_BLAS_SYMS_${backend}}"
        CACHE INTERNAL "Thread-control symbols bound into the library")
    set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "${backend}" CACHE INTERNAL
        "BLAS vendor whose thread-control symbols are bound into the library")
  else()
    if(NOT QDK_CHEMISTRY_BLAS_THREAD_API MATCHES "^(AUTO|NONE)$")
      message(FATAL_ERROR
        "QDK_CHEMISTRY_BLAS_THREAD_API='${QDK_CHEMISTRY_BLAS_THREAD_API}' does not "
        "link against the BLAS resolved here (${BLAS_LIBRARIES}).")
    endif()
    if(NOT QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "NONE")
      message(WARNING
        "No BLAS thread-control API links against the BLAS resolved here "
        "(${BLAS_LIBRARIES}), so ScopedBlasThreads is a no-op and BLAS threads "
        "cannot be pinned around GauXC. Restrict BLAS to one thread via its "
        "environment variable (OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, "
        "BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS) to avoid oversubscription.")
    endif()
    set(${out_src} "${src_dir}/blas_threads_none.cpp" PARENT_SCOPE)
    set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS "" CACHE INTERNAL
        "Thread-control symbols bound into the library")
    set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "" CACHE INTERNAL
        "BLAS vendor whose thread-control symbols are bound into the library")
  endif()

  set(QDK_CHEMISTRY_BLAS_LINK "${link}" CACHE INTERNAL "BLAS libraries to link")
endfunction()
