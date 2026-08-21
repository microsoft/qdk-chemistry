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
# which one links instead of assuming OpenBLAS. Binding here is the only way to
# reach the BLAS our own calls go to: a symbol found in the running process may
# belong to a different BLAS, and a statically linked one exports nothing.

include(CheckFunctionExists)

set(QDK_CHEMISTRY_BLAS_VENDOR "${BLAS_VENDOR}")

# Candidate APIs, in probe order: a dispatch layer can also expose its
# backend's API, so ask the dispatcher first. Single source of truth for BLAS
# thread control: drives the configure-time probe below and, through
# QDK_CHEMISTRY_BLAS_BACKEND_TABLE in config.h.in, the BlasVendor enumeration
# in util/blas_threads.h. Columns are setter, getter, thread-count type,
# BlasVendor enumerator, display name.
#
# OPENBLAS also covers drop-ins such as ARM Performance Libraries. BLIS types
# the count as dim_t, 32- or 64-bit depending on its build; the 64-bit form is
# safe for both, as a 32-bit callee reads the low half of the argument register.
set(_qdk_blas_backends FLEXIBLAS OPENBLAS INTELMKL BLIS NVPL)
set(_qdk_blas_api_FLEXIBLAS
    flexiblas_set_num_threads flexiblas_get_num_threads int FlexiBLAS "FlexiBLAS")
set(_qdk_blas_api_OPENBLAS
    openblas_set_num_threads openblas_get_num_threads int OpenBLAS "OpenBLAS")
set(_qdk_blas_api_INTELMKL
    MKL_Set_Num_Threads MKL_Get_Max_Threads int IntelMKL "Intel MKL")
set(_qdk_blas_api_BLIS
    bli_thread_set_num_threads bli_thread_get_num_threads std::int64_t BLIS "BLIS")
set(_qdk_blas_api_NVPL
    nvpl_blas_set_num_threads nvpl_blas_get_max_threads int NVPL "NVPL BLAS")

# Expand the table into the X-macro consumed by config.h.in.
set(_qdk_blas_table_lines "")
foreach(_qdk_blas_backend IN LISTS _qdk_blas_backends)
  list(GET _qdk_blas_api_${_qdk_blas_backend} 0 _qdk_blas_set_fn)
  list(GET _qdk_blas_api_${_qdk_blas_backend} 1 _qdk_blas_get_fn)
  list(GET _qdk_blas_api_${_qdk_blas_backend} 2 _qdk_blas_type)
  list(GET _qdk_blas_api_${_qdk_blas_backend} 3 _qdk_blas_enum)
  list(GET _qdk_blas_api_${_qdk_blas_backend} 4 _qdk_blas_label)
  list(APPEND _qdk_blas_table_lines
       "  X(${_qdk_blas_backend}, ${_qdk_blas_enum}, \"${_qdk_blas_label}\", ${_qdk_blas_set_fn}, ${_qdk_blas_get_fn}, ${_qdk_blas_type})")
endforeach()
string(JOIN " \\\n" QDK_CHEMISTRY_BLAS_BACKEND_TABLE ${_qdk_blas_table_lines})

set(QDK_CHEMISTRY_BLAS_THREAD_API "AUTO" CACHE STRING
    "BLAS thread-control API to bind: AUTO, NONE or one of ${_qdk_blas_backends}")
set_property(CACHE QDK_CHEMISTRY_BLAS_THREAD_API PROPERTY STRINGS
    AUTO ${_qdk_blas_backends} NONE)

if(TARGET BLAS::BLAS)
  set(_qdk_blas_link_libs BLAS::BLAS)
else()
  set(_qdk_blas_link_libs ${BLAS_LIBRARIES})
endif()

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

# Pick at most one backend; empty means ScopedBlasThreads is a no-op, since a
# backend can only be bound here.
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
  # Selected row, consumed by config.h.in as QDK_CHEMISTRY_BLAS_LINKED_*.
  list(GET _qdk_blas_api_${_qdk_blas_thread_api} 0 QDK_CHEMISTRY_BLAS_LINKED_SET_FN)
  list(GET _qdk_blas_api_${_qdk_blas_thread_api} 1 QDK_CHEMISTRY_BLAS_LINKED_GET_FN)
  list(GET _qdk_blas_api_${_qdk_blas_thread_api} 2 QDK_CHEMISTRY_BLAS_LINKED_TYPE)
  list(GET _qdk_blas_api_${_qdk_blas_thread_api} 3 QDK_CHEMISTRY_BLAS_LINKED_VENDOR)
  # Already linked transitively through GauXC; name it for the symbols we call.
  target_link_libraries(chemistry PRIVATE ${_qdk_blas_link_libs})
  # Link-time binding leaves vendor-specific symbols unresolved in the installed
  # archive while the exported dependency is the vendor-agnostic BLAS::BLAS, so
  # cmake/qdkConfig.cmake.in steers and verifies the consumer's BLAS search.
  # Cache variables because this is a subdirectory scope.
  set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "${QDK_CHEMISTRY_BLAS_VENDOR}"
      CACHE INTERNAL
      "BLAS vendor whose thread-control symbols are bound into the library")
  set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOL "${QDK_CHEMISTRY_BLAS_LINKED_SET_FN}"
      CACHE INTERNAL
      "BLAS thread-control symbol bound into the library")
else()
  set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "" CACHE INTERNAL
      "BLAS vendor whose thread-control symbols are bound into the library")
  set(QDK_CHEMISTRY_BLAS_THREAD_SYMBOL "" CACHE INTERNAL
      "BLAS thread-control symbol bound into the library")
  set(_qdk_blas_thread_api "none (thread control disabled)")
endif()

message(STATUS "QDK Chemistry BLAS vendor: '${QDK_CHEMISTRY_BLAS_VENDOR}', "
               "thread-control API: ${_qdk_blas_thread_api}")

# Generate Config Header
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
install(FILES ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/qdk/chemistry/scf)

# Runtime configuration for QDK-Chemistry SCF resources
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
