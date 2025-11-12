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
  util/blas.cpp
  util/gauxc_util.cpp
  util/int1e.cpp
  util/lapack.cpp
  util/libint2_util.cpp
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
# Configure scf library
###############################################################################

target_sources(chemistry PRIVATE
  scf/soad.cpp
  scf/cpscf.cpp
  scf/diis.cpp
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

# Generate Config Header
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.h.in ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h)
configure_file(${CMAKE_CURRENT_LIST_DIR}/config.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
install(FILES ${PROJECT_BINARY_DIR}/include/qdk/chemistry/scf/config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/qdk/chemistry/scf)

# Runtime configuration for QDK-Chemistry SCF resources
target_sources(chemistry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
