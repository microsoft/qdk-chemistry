cmake_minimum_required( VERSION 3.13 FATAL_ERROR )
project( lobpcgxx VERSION 0.0.1 LANGUAGES C CXX )

option( LOBPCGXX_ENABLE_TESTS "Enable Testing" ON )

find_package( LAPACK REQUIRED )
find_package( BLAS REQUIRED )

# BLAS++/LAPACK++
set(use_openmp OFF)
if(MACIS_ENABLE_OPENMP)
  set(use_openmp ON)
endif()

# blaspp auto-detects cuda,hip modules and enables device backends
set( gpu_backend "none" CACHE STRING "GPU backend to use" FORCE)
message( WARNING "Turning off GPU Backend for BLAS++/LAPACK++")
set(build_tests OFF CACHE BOOL "Build BLAS++/LAPACK++ tests" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

find_package(lapackpp CONFIG QUIET)
if(NOT lapackpp_FOUND )
include( FetchContent )
  FetchContent_Declare( blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG 13622021629f5fd27591bb7da60bae5b19561f01
  )
  FetchContent_Declare( lapackpp
    GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp.git
    GIT_TAG 5bc9c85201ace48213df5ac7d1ef026c9668dfbd
  )

  FetchContent_MakeAvailable( blaspp )
  FetchContent_MakeAvailable( lapackpp )
  target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )
endif()

add_library( lobpcgxx INTERFACE )
target_link_libraries( lobpcgxx INTERFACE blaspp lapackpp )
target_compile_features( lobpcgxx INTERFACE cxx_std_20 )

target_include_directories( lobpcgxx INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

include(CTest)
if( CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND LOBPCGXX_ENABLE_TESTS AND BUILD_TESTING )
  add_executable( lobpcg_tester examples/tester.cpp )
  target_link_libraries( lobpcg_tester PUBLIC lobpcgxx )
endif()

# INSTALL rules
add_library( lobpcgxx::lobpcgxx ALIAS lobpcgxx )

include( GNUInstallDirs )

# TARGETS
install( TARGETS lobpcgxx
  EXPORT lobpcgxx-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties( lobpcgxx PROPERTIES EXPORT_NAME lobpcgxx )

# Install Headers
install(
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX}
  FILES_MATCHING PATTERN "*.hpp"
)


# Export target to scripe
set( INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/lobpcgxx )
install( EXPORT lobpcgxx-targets
  FILE         lobpcgxx-targets.cmake
  NAMESPACE    lobpcgxx::
  DESTINATION  ${INSTALL_CONFIGDIR}
)


# Create config-version.cmake file
include( CMakePackageConfigHelpers )
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)



# Setup lobpcgxx-config.cmake
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/lobpcgxx-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/lobpcgxx-config-version.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)
