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
# Each BLAS backend spells that API differently, so ask the configured BLAS
# which one links instead of assuming OpenBLAS. Binding at link time is what
# makes a statically linked BLAS work (its symbols are exported by no module,
# so blas_threads.cpp cannot find them at runtime).

include(CheckFunctionExists)

set(QDK_CHEMISTRY_BLAS_VENDOR "${BLAS_VENDOR}")

set(QDK_CHEMISTRY_BLAS_THREAD_API "AUTO" CACHE STRING
    "BLAS thread-control API to bind: AUTO, OPENBLAS, INTELMKL, BLIS, FLEXIBLAS, NVPL or NONE")
set_property(CACHE QDK_CHEMISTRY_BLAS_THREAD_API PROPERTY STRINGS
    AUTO OPENBLAS INTELMKL BLIS FLEXIBLAS NVPL NONE)

if(TARGET BLAS::BLAS)
  set(_qdk_blas_link_libs BLAS::BLAS)
else()
  set(_qdk_blas_link_libs ${BLAS_LIBRARIES})
endif()

# Candidate APIs, in probe order: a dispatch layer can also expose its
# backend's API, so ask the dispatcher first.
set(_qdk_blas_backends FLEXIBLAS OPENBLAS INTELMKL BLIS NVPL)
set(_qdk_blas_api_FLEXIBLAS flexiblas_set_num_threads flexiblas_get_num_threads)
set(_qdk_blas_api_OPENBLAS  openblas_set_num_threads  openblas_get_num_threads)
set(_qdk_blas_api_INTELMKL  MKL_Set_Num_Threads       MKL_Get_Max_Threads)
set(_qdk_blas_api_BLIS      bli_thread_set_num_threads bli_thread_get_num_threads)
set(_qdk_blas_api_NVPL      nvpl_blas_set_num_threads nvpl_blas_get_max_threads)

# Sets ${out_var} to ${backend} if both halves of its API link against the
# configured BLAS. A function, not a macro, so CMAKE_REQUIRED_* stays local.
function(qdk_check_blas_thread_api backend out_var)
  list(GET _qdk_blas_api_${backend} 0 set_fn)
  list(GET _qdk_blas_api_${backend} 1 get_fn)
  set(CMAKE_REQUIRED_LIBRARIES ${_qdk_blas_link_libs})
  set(CMAKE_REQUIRED_INCLUDES ${BLAS_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_function_exists(${set_fn} QDK_CHEMISTRY_HAS_${set_fn})
  check_function_exists(${get_fn} QDK_CHEMISTRY_HAS_${get_fn})
  if(QDK_CHEMISTRY_HAS_${set_fn} AND QDK_CHEMISTRY_HAS_${get_fn})
    set(${out_var} "${backend}" PARENT_SCOPE)
  endif()
endfunction()

# Pick at most one backend; empty means blas_threads.cpp uses its runtime lookup.
set(_qdk_blas_thread_api "")
if(QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "AUTO")
  foreach(_qdk_blas_backend IN LISTS _qdk_blas_backends)
    if(_qdk_blas_link_libs AND NOT _qdk_blas_thread_api)
      qdk_check_blas_thread_api(${_qdk_blas_backend} _qdk_blas_thread_api)
    endif()
  endforeach()
elseif(QDK_CHEMISTRY_BLAS_THREAD_API IN_LIST _qdk_blas_backends)
  set(_qdk_blas_thread_api "${QDK_CHEMISTRY_BLAS_THREAD_API}")
elseif(NOT QDK_CHEMISTRY_BLAS_THREAD_API STREQUAL "NONE")
  message(FATAL_ERROR
    "Invalid QDK_CHEMISTRY_BLAS_THREAD_API='${QDK_CHEMISTRY_BLAS_THREAD_API}'. "
    "Expected AUTO, NONE or one of: ${_qdk_blas_backends}.")
endif()

if(_qdk_blas_thread_api)
  # Consumed by config.h.in as #cmakedefine QDK_CHEMISTRY_BLAS_LINKS_<backend>.
  set(QDK_CHEMISTRY_BLAS_LINKS_${_qdk_blas_thread_api} TRUE)
  # Already linked transitively through GauXC; name it for the symbols we call.
  target_link_libraries(chemistry PRIVATE ${_qdk_blas_link_libs})
else()
  set(_qdk_blas_thread_api "none (runtime lookup only)")
endif()

message(STATUS "QDK Chemistry BLAS vendor: '${QDK_CHEMISTRY_BLAS_VENDOR}', "
               "thread-control API: ${_qdk_blas_thread_api}")

# dlsym() for the runtime fallback; empty on Windows and macOS.
target_link_libraries(chemistry PRIVATE ${CMAKE_DL_LIBS})

# Generate Config Header
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
install(FILES ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/qdk/chemistry/scf)

# Runtime configuration for QDK-Chemistry SCF resources
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
