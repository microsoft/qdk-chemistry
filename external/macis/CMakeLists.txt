# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

cmake_minimum_required(VERSION 3.15)
project(MACIS VERSION 0.1 LANGUAGES C CXX)

include(CMakeDependentOption)

option( MACIS_ENABLE_OPENMP "Enable OpenMP Bindings" ON )
option( MACIS_ENABLE_BOOST  "Enable Boost" ON )
option( MACIS_ENABLE_PYTHON "Enable Python Bindings" OFF )
option( MACIS_ENABLE_EXAMPLES "Build Examples" ON )

cmake_dependent_option(
  MACIS_ENABLE_MPI
  "Enable MPI Bindings"
  ON
  "NOT MACIS_ENABLE_PYTHON"
  OFF # Force disable if PYTHON is enabled
)

# SPDLOG for logging
set(SPDLOG_INSTALL ON CACHE BOOL "Enable SPDLOG Install" FORCE)
handle_dependency(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.3
  BUILD_TARGET spdlog::spdlog
  INSTALL_TARGET spdlog::spdlog
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# CMake Modules
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )
include( macis-cmake-modules )
if( MACIS_ENABLE_OPENMP )
  find_package(OpenMP)
endif()

include(macis-uarch)

add_subdirectory(src)

if( MACIS_ENABLE_EXAMPLES )
  add_subdirectory(examples)
endif()

# Python bindings
if(MACIS_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

# Tests
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
