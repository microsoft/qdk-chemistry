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

# BLAS thread control (see util/blas_threads.cpp). OpenBLAS, Intel MKL and BLIS
# each spell the thread-count API differently, so only the vendor is passed
# down, as QDK_CHEMISTRY_BLAS_VENDOR_<vendor>; blas_threads.cpp names the
# functions. Any other vendor -- Accelerate on macOS, reference BLAS, or none
# reported at all -- exposes no such API, and ScopedBlasThreads compiles to a
# no-op. The match both selects the vendor and keeps the macro name an
# identifier.
#
# BLAS_VENDOR is the knob for the one case the search cannot answer -- passing
# BLAS_LIBRARIES directly skips the vendor search (FindBLAS.cmake), leaving
# BLAS_VENDOR empty, and -DBLAS_VENDOR=<vendor> supplies the missing label.
if(BLAS_VENDOR MATCHES "^(OpenBLAS|IntelMKL|BLIS)$")
  target_compile_definitions(chemistry PRIVATE QDK_CHEMISTRY_BLAS_VENDOR_${BLAS_VENDOR})
  # BLAS is already linked transitively through GauXC; name it here for the
  # vendor symbols we call ourselves. The include directories come with it --
  # the BLIS path needs blis.h for dim_t, whose width is a BLIS build option.
  if(TARGET BLAS::BLAS)
    target_link_libraries(chemistry PRIVATE BLAS::BLAS)
  else()
    target_link_libraries(chemistry PRIVATE ${BLAS_LIBRARIES})
    target_include_directories(chemistry PRIVATE ${BLAS_INCLUDE_DIRS})
  endif()
  # The installed archive references vendor-specific symbols while the exported
  # dependency is the vendor-agnostic BLAS::BLAS, so cmake/qdkConfig.cmake.in
  # steers the consumer's BLAS search to the same provider. A cache variable
  # because this is a subdirectory scope.
  set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "${BLAS_VENDOR}" CACHE INTERNAL
      "BLAS vendor whose thread-control symbols are bound into the library")
  message(STATUS "QDK Chemistry BLAS vendor: '${BLAS_VENDOR}', thread control: enabled")
else()
  set(QDK_CHEMISTRY_REQUIRED_BLAS_VENDOR "" CACHE INTERNAL
      "BLAS vendor whose thread-control symbols are bound into the library")
  message(STATUS "QDK Chemistry BLAS vendor: '${BLAS_VENDOR}', thread control: disabled")
endif()

# Generate Config Header
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
install(FILES ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/qdk/chemistry/scf)

# Runtime configuration for QDK-Chemistry SCF resources
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
