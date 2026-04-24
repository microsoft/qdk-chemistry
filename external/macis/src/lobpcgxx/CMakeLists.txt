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
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

find_package(lapackpp CONFIG QUIET)
if(NOT lapackpp_FOUND )
include( FetchContent )
  # MSVC native cl does not support __PRETTY_FUNCTION__ used in blaspp.
  # Apply a patch to replace it with __FUNCSIG__.
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(_blaspp_patch PATCH_COMMAND "${CMAKE_COMMAND}" -P
          "${QDK_CMAKE_PATCHES_DIR}/blaspp-msvc-pretty-function.cmake")
  else()
      set(_blaspp_patch "")
  endif()
  FetchContent_Declare( blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG v2025.05.28
    ${_blaspp_patch}
  )
  FetchContent_Declare( lapackpp
    GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp.git
    GIT_TAG v2025.05.28
  )

  FetchContent_MakeAvailable( blaspp )
  FetchContent_MakeAvailable( lapackpp )
  target_compile_definitions( lapackpp PUBLIC LAPACK_COMPLEX_CPP )

  # On Windows, blaspp's try_run() mangling detection fails because the
  # OpenBLAS DLL is not on PATH during CMake configure.  OpenBLAS uses
  # lowercase-with-underscore Fortran mangling (e.g. sgetrf_), so force
  # BLAS_FORTRAN_ADD_ on both blaspp and lapackpp targets.
  if(WIN32)
      target_compile_definitions( blaspp   PUBLIC BLAS_FORTRAN_ADD_ )
      target_compile_definitions( lapackpp PUBLIC BLAS_FORTRAN_ADD_ )
  endif()
endif()

add_library( lobpcgxx INTERFACE )
target_link_libraries( lobpcgxx INTERFACE blaspp lapackpp )
target_compile_features( lobpcgxx INTERFACE cxx_std_20 )
target_compile_definitions( lobpcgxx INTERFACE LAPACK_COMPLEX_CPP )

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
