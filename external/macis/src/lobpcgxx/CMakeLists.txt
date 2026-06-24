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
  # MSVC (native cl and clang-cl): apply combined compatibility patch
  # (replaces __PRETTY_FUNCTION__ with __FUNCSIG__ and fixes BLASFinder cache
  # bug). The cache fix is always required on MSVC; the __FUNCSIG__ swap is
  # benign under clang-cl since it also supports __FUNCSIG__.
  if(MSVC)
      set(_blaspp_patch PATCH_COMMAND "${CMAKE_COMMAND}"
          -DPATCH_FILE=${QDK_CMAKE_PATCHES_DIR}/blaspp-msvc.patch
          -P ${QDK_CMAKE_PATCHES_DIR}/apply-patch.cmake)
  else()
      set(_blaspp_patch "")
  endif()
  FetchContent_Declare( blaspp
    GIT_REPOSITORY https://github.com/icl-utk-edu/blaspp.git
    GIT_TAG v2025.05.28
    ${_blaspp_patch}
    UPDATE_DISCONNECTED TRUE
  )
  if(MSVC)
      set(_lapackpp_patch PATCH_COMMAND "${CMAKE_COMMAND}"
          -DPATCH_FILE=${QDK_CMAKE_PATCHES_DIR}/lapackpp-msvc-allocator.patch
          -P ${QDK_CMAKE_PATCHES_DIR}/apply-patch.cmake)
  else()
      set(_lapackpp_patch "")
  endif()
  FetchContent_Declare( lapackpp
    GIT_REPOSITORY https://github.com/icl-utk-edu/lapackpp.git
    GIT_TAG v2025.05.28
    UPDATE_DISCONNECTED TRUE
    ${_lapackpp_patch}
  )

  FetchContent_MakeAvailable( blaspp )
  FetchContent_MakeAvailable( lapackpp )

  if(MSVC)
    # lapackpp sources use std::complex directly but on Windows _MSC_VER causes
    # config.h to skip #include <complex>. Force-include it.
    target_compile_options( lapackpp PRIVATE /FIcomplex )

    # Suppress warnings in upstream blaspp/lapackpp sources
    # /wd4100 unreferenced formal parameter
    # /wd4101 unreferenced local variable
    # /wd4127 conditional expression is constant
    # /wd4242 conversion with possible loss of data
    # /wd4244 conversion with possible loss of data
    # /wd4267 conversion with possible loss of data
    set(COMMON_MSVC_WARNING_SUPPRESSIONS /wd4100 /wd4101 /wd4127 /wd4242 /wd4244 /wd4267)
    target_compile_options(blaspp   PUBLIC ${COMMON_MSVC_WARNING_SUPPRESSIONS})
    target_compile_options(lapackpp PUBLIC ${COMMON_MSVC_WARNING_SUPPRESSIONS})
  endif()
endif()

add_library( lobpcgxx INTERFACE )
target_link_libraries( lobpcgxx INTERFACE blaspp lapackpp )
target_compile_features( lobpcgxx INTERFACE cxx_std_20 )
if(NOT MSVC)
  # On non-MSVC (Linux/macOS), instruct lapack/config.h to use <complex> (C++)
  # rather than <complex.h> (C99).  The C99 header defines the `I` macro as
  # `__extension__ _Complex_I`, which conflicts with Boost template parameter
  # names used in downstream headers.  On MSVC the equivalent is achieved via
  # /FIcomplex on the lapackpp target directly (see above).
  target_compile_definitions( lobpcgxx INTERFACE LAPACK_COMPLEX_CPP )
endif()

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
