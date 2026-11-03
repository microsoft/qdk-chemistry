# Selects the BLAS thread-control backend (see scf/src/util/blas_threads_backend.h).
#
# One small blas_threads_<vendor>.cpp implements the thread-count API per
# vendor, and the vendor find_package(BLAS) reports picks which one. A single
# link probe then confirms that choice, because a vendor label is not a promise
# that the symbols exist: an OpenBLAS built without threading, or a BLIS
# predating bli_thread_set_num_threads, is labelled the same as one that works.
#
# Everything else compiles blas_threads_none.cpp, leaving ScopedBlasThreads a
# no-op. In particular a build that passes BLAS_LIBRARIES directly gets no
# vendor label at all, and is declined rather than guessed at; put the BLAS
# install prefix on CMAKE_PREFIX_PATH instead and let find_package(BLAS)
# identify it.

# Removed after v2.2.1. CMake only reports an unused -D once, and never for a
# value inherited from an existing cache, so say so on every configure.
if(DEFINED QDK_CHEMISTRY_BLAS_THREAD_API)
  message(WARNING
    "QDK_CHEMISTRY_BLAS_THREAD_API is no longer used and is ignored. The "
    "thread-control backend now follows the vendor find_package(BLAS) "
    "reports; put the BLAS install prefix on CMAKE_PREFIX_PATH to choose "
    "one. Remove this variable from the command line and the CMake cache.")
endif()

set(QDK_CHEMISTRY_BLAS_VENDORS OpenBLAS IntelMKL BLIS)
set(QDK_CHEMISTRY_BLAS_SRC_OpenBLAS blas_threads_openblas.cpp)
set(QDK_CHEMISTRY_BLAS_SRC_IntelMKL blas_threads_mkl.cpp)
set(QDK_CHEMISTRY_BLAS_SRC_BLIS     blas_threads_blis.cpp)
set(QDK_CHEMISTRY_BLAS_SYMS_OpenBLAS openblas_set_num_threads openblas_get_num_threads)
set(QDK_CHEMISTRY_BLAS_SYMS_IntelMKL MKL_Set_Num_Threads MKL_Get_Max_Threads)
set(QDK_CHEMISTRY_BLAS_SYMS_BLIS bli_thread_set_num_threads bli_thread_get_num_threads)

# Sets ${out_src} to the backend source to compile, and the cache variables
# QDK_CHEMISTRY_BLAS_LINK / QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS.
function(qdk_select_blas_thread_backend src_dir out_src)
  if(TARGET BLAS::BLAS)
    set(link BLAS::BLAS)
  else()
    set(link ${BLAS_LIBRARIES})
  endif()

  set(backend "")
  set(reason "")

  if(NOT BLAS_VENDOR)
    string(CONCAT reason
        "find_package(BLAS) reported no vendor for the BLAS resolved here "
        "(${BLAS_LIBRARIES}). Passing BLAS_LIBRARIES directly skips vendor "
        "detection; put the BLAS install prefix on CMAKE_PREFIX_PATH instead, "
        "and let find_package(BLAS) identify it.")
  elseif(NOT BLAS_VENDOR IN_LIST QDK_CHEMISTRY_BLAS_VENDORS)
    set(reason "BLAS vendor ${BLAS_VENDOR} exports no thread-count API.")
  elseif(NOT link)
    set(reason "no BLAS libraries have been resolved to link against.")
  else()
    # try_compile re-runs on every configure (it has no built-in skip), so the
    # probe always reflects the BLAS resolved right now. Do not guard it with
    # if(NOT DEFINED ...) -- that would cache the result across a BLAS change.
    try_compile(QDK_CHEMISTRY_BLAS_PROBE_${BLAS_VENDOR}
      "${CMAKE_CURRENT_BINARY_DIR}/blas_probe_${BLAS_VENDOR}"
      SOURCES
        "${src_dir}/${QDK_CHEMISTRY_BLAS_SRC_${BLAS_VENDOR}}"
        "${src_dir}/blas_threads_probe.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${BLAS_INCLUDE_DIRS}"
      LINK_LIBRARIES ${link}
      CXX_STANDARD ${CMAKE_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)
    if(QDK_CHEMISTRY_BLAS_PROBE_${BLAS_VENDOR})
      set(backend ${BLAS_VENDOR})
    else()
      string(CONCAT reason
          "the BLAS resolved here (${BLAS_LIBRARIES}) is labelled "
          "${BLAS_VENDOR} but does not provide its thread-count API.")
    endif()
  endif()

  if(backend)
    message(STATUS "QDK Chemistry BLAS thread control: ${backend}")
    set(${out_src} "${src_dir}/${QDK_CHEMISTRY_BLAS_SRC_${backend}}" PARENT_SCOPE)
    set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS "${QDK_CHEMISTRY_BLAS_SYMS_${backend}}"
        CACHE INTERNAL "Thread-control symbols bound into the library")
    set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "${backend}" CACHE INTERNAL
        "BLAS vendor whose thread-control symbols are bound into the library")
  else()
    message(WARNING
      "BLAS thread control is disabled: ${reason} ScopedBlasThreads will "
      "be a no-op, so BLAS threads cannot be pinned around GauXC. Restrict "
      "BLAS to one thread through its environment variable "
      "(OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, BLIS_NUM_THREADS, "
      "VECLIB_MAXIMUM_THREADS) to avoid oversubscription.")
    message(STATUS "QDK Chemistry BLAS thread control: NONE")
    set(${out_src} "${src_dir}/blas_threads_none.cpp" PARENT_SCOPE)
    set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOLS "" CACHE INTERNAL
        "Thread-control symbols bound into the library")
    set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "" CACHE INTERNAL
        "BLAS vendor whose thread-control symbols are bound into the library")
  endif()

  set(QDK_CHEMISTRY_BLAS_LINK "${link}" CACHE INTERNAL "BLAS libraries to link")
endfunction()
