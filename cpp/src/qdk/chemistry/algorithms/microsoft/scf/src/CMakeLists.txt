target_link_libraries(chemistry
  PUBLIC
    $<$<BOOL:${QDK_CHEMISTRY_ENABLE_MPI}>:MPI::MPI_CXX>
    Libint2::cxx
    ECPINT::ecpint
    $<$<BOOL:${QDK_CHEMISTRY_ENABLE_GPU}>:CUDA::cudart>
    $<$<BOOL:${QDK_CHEMISTRY_ENABLE_GPU}>:CUDA::cublasLt>
    $<$<BOOL:${QDK_CHEMISTRY_ENABLE_GPU}>:CUDA::cusolver>
    $<$<BOOL:${QDK_CHEMISTRY_ENABLE_GPU}>:cutensor>
)
target_include_directories(chemistry PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
  $<INSTALL_INTERFACE:include>
)

# Must set the flags to enable openmp compile for .cu files.
if(QDK_CHEMISTRY_ENABLE_GPU)
  set_target_properties(chemistry PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  target_compile_options(chemistry PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${OpenMP_CXX_FLAGS}>
  )
  target_compile_features(chemistry PRIVATE cuda_std_17)
endif()

set_target_properties(chemistry PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

###############################################################################
# Configure core SCF library
###############################################################################
target_sources(chemistry PRIVATE
  core/basis_set.cpp
  core/scf.cpp
)

###############################################################################
# Configure SCF util library
###############################################################################

target_sources(chemistry PRIVATE
  util/blas_threads.cpp
  util/gauxc_util.cpp
  util/int1e.cpp
  util/libint2_util.cpp
  util/matrix_exp.cpp
  util/similarity_transform.cpp
)
if(QDK_CHEMISTRY_ENABLE_MPI)
  target_sources(chemistry PRIVATE util/mpi_util.cpp)
endif()

if(QDK_CHEMISTRY_ENABLE_GPU)
  target_sources(chemistry PRIVATE
    util/gpu/cuda_helper.cpp
    util/gpu/cusolver_utils.cpp
    util/gpu/cutensor_utils.cpp
  )
endif()

###############################################################################
# Configure eri and exc libraries
###############################################################################

add_subdirectory(eri)
add_subdirectory(exc)

###############################################################################
# Configure scf algorithm library
###############################################################################

add_subdirectory(scf_algorithm)

###############################################################################
# Configure scf library
###############################################################################

target_sources(chemistry PRIVATE
  scf/soad.cpp
  scf/cpscf.cpp
  scf/guess.cpp
  scf/scf_impl.cpp
  scf/ks_impl.cpp
  scf/scf_solver.cpp
)

# Install Resources
if(NOT DEFINED QDK_RESOURCE_PREFIX)
  set(QDK_RESOURCE_PREFIX "share/qdk/chemistry/scf")
endif()

if(QDK_EMBED_RESOURCE_LOCATION)
  set(RESOURCES_DIR "${CMAKE_INSTALL_PREFIX}/${QDK_RESOURCE_PREFIX}/resources")
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../resources
  DESTINATION ${QDK_RESOURCE_PREFIX}
  COMPONENT qdk_chemistry_resources
)

###############################################################################
# Detect the BLAS thread-control API (see util/blas_threads.cpp)
###############################################################################
#
# GauXC calls BLAS from inside an OpenMP-parallel region, so BLAS itself must
# be pinned to a single thread there. Every BLAS backend spells that API
# differently, so instead of assuming OpenBLAS we probe the BLAS the user
# actually configured (found by linalg-cmake-modules via GauXC/MACIS) and bind
# directly to whichever thread-control API links.
#
# This matters for statically linked BLAS (e.g. the x64-windows-static-md
# vcpkg triplet), where the symbols are not exported from any module and so
# cannot be discovered at runtime. When no API links, blas_threads.cpp still
# falls back to a runtime lookup (dlsym / GetProcAddress), which covers builds
# where BLAS is loaded dynamically by someone else.

include(CheckFunctionExists)

set(QDK_CHEMISTRY_BLAS_VENDOR "${BLAS_VENDOR}")

# Escape hatch for BLAS installations the probe below cannot classify (or
# misclassifies): pick the backend explicitly, or NONE to disable the
# link-time binding and rely on the runtime lookup only.
set(QDK_CHEMISTRY_BLAS_THREAD_API "AUTO" CACHE STRING
    "BLAS thread-control API to bind: AUTO, OPENBLAS, INTELMKL, BLIS, FLEXIBLAS, NVPL or NONE")
set_property(CACHE QDK_CHEMISTRY_BLAS_THREAD_API PROPERTY STRINGS
    AUTO OPENBLAS INTELMKL BLIS FLEXIBLAS NVPL NONE)

if(TARGET BLAS::BLAS)
  set(_qdk_blas_link_libs BLAS::BLAS)
else()
  set(_qdk_blas_link_libs ${BLAS_LIBRARIES})
endif()

# Check whether both halves of a vendor's thread-control API link against the
# configured BLAS, and set ${out_var} accordingly.
function(qdk_check_blas_thread_api set_fn get_fn out_var)
  set(CMAKE_REQUIRED_LIBRARIES ${_qdk_blas_link_libs})
  set(CMAKE_REQUIRED_INCLUDES ${BLAS_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_function_exists(${set_fn} QDK_CHEMISTRY_HAS_${set_fn})
  check_function_exists(${get_fn} QDK_CHEMISTRY_HAS_${get_fn})
  if(QDK_CHEMISTRY_HAS_${set_fn} AND QDK_CHEMISTRY_HAS_${get_fn})
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

if(NOT QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "AUTO")
  # Explicit user choice: trust it instead of probing.
  foreach(_qdk_blas_backend OPENBLAS INTELMKL BLIS FLEXIBLAS NVPL)
    if(QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "${_qdk_blas_backend}")
      set(QDK_CHEMISTRY_BLAS_LINKS_${_qdk_blas_backend} TRUE)
    endif()
  endforeach()
  if(NOT QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "NONE"
     AND NOT QDK_CHEMISTRY_BLAS_LINKS_${QDK_CHEMISTRY_BLAS_THREAD_API})
    message(FATAL_ERROR
      "Invalid QDK_CHEMISTRY_BLAS_THREAD_API='${QDK_CHEMISTRY_BLAS_THREAD_API}'. "
      "Expected AUTO, OPENBLAS, INTELMKL, BLIS, FLEXIBLAS, NVPL or NONE.")
  endif()
elseif(_qdk_blas_link_libs)
  qdk_check_blas_thread_api(openblas_set_num_threads openblas_get_num_threads
    QDK_CHEMISTRY_BLAS_LINKS_OPENBLAS)
  qdk_check_blas_thread_api(MKL_Set_Num_Threads MKL_Get_Max_Threads
    QDK_CHEMISTRY_BLAS_LINKS_INTELMKL)
  qdk_check_blas_thread_api(bli_thread_set_num_threads bli_thread_get_num_threads
    QDK_CHEMISTRY_BLAS_LINKS_BLIS)
  qdk_check_blas_thread_api(flexiblas_set_num_threads flexiblas_get_num_threads
    QDK_CHEMISTRY_BLAS_LINKS_FLEXIBLAS)
  qdk_check_blas_thread_api(nvpl_blas_set_num_threads nvpl_blas_get_max_threads
    QDK_CHEMISTRY_BLAS_LINKS_NVPL)
endif()

# config.h.in must end up with at most one backend, but a dispatch layer can
# also expose its backend's API. Ask the dispatcher first; the remaining
# backends are mutually exclusive in practice.
set(_qdk_blas_thread_api "none (runtime lookup only)")
foreach(_qdk_blas_backend FLEXIBLAS OPENBLAS INTELMKL BLIS NVPL)
  if(_qdk_blas_thread_api STREQUAL "none (runtime lookup only)"
     AND QDK_CHEMISTRY_BLAS_LINKS_${_qdk_blas_backend})
    set(_qdk_blas_thread_api "${_qdk_blas_backend}")
    # Bind directly to the BLAS providing the API. It is already linked
    # transitively through GauXC; naming it here keeps the reference explicit
    # and works for statically linked BLAS too.
    target_link_libraries(chemistry PRIVATE ${_qdk_blas_link_libs})
  else()
    unset(QDK_CHEMISTRY_BLAS_LINKS_${_qdk_blas_backend})
  endif()
endforeach()

message(STATUS "QDK Chemistry BLAS vendor: '${QDK_CHEMISTRY_BLAS_VENDOR}', "
               "thread-control API: ${_qdk_blas_thread_api}")

# The runtime fallback uses dlsym(), which lives in libdl on glibc (folded into
# libc on newer glibc, but linking it explicitly keeps older toolchains
# working). CMAKE_DL_LIBS is empty on Windows and macOS, where the equivalent
# APIs come from kernel32 / libSystem.
target_link_libraries(chemistry PRIVATE ${CMAKE_DL_LIBS})

# Generate Config Header
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
install(FILES ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/qdk/chemistry/scf)

# Runtime configuration for QDK-Chemistry SCF resources
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
