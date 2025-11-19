# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.18 FATAL_ERROR )
project( macis_cmake_subproject LANGUAGES C CXX )


if( NOT GIT_REVISION )
  # Get current Git Revision
  find_package( Git REQUIRED )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    OUTPUT_VARIABLE GIT_REVISION
    ERROR_QUIET
  )
  string( STRIP "${GIT_REVISION}" GIT_REVISION )
endif()


if( NOT GITHUB_URL )
  set( GITHUB_URL "https://github.com" )
endif()

if( NOT GITHUB_REPOSITORY )
  set( GITHUB_REPOSITORY "wavefunction91/MACIS-CI.git" )
endif()

message( STATUS "Pulling GIT_URL        = ${GITHUB_URL}" )
message( STATUS "Pulling GIT_REPOSITORY = ${GITHUB_REPOSITORY}" )
message( STATUS "Pulling GIT_REVISION   = ${GIT_REVISION}" )



include( FetchContent )
FetchContent_Declare( macis
  GIT_REPOSITORY "${GITHUB_URL}/${GITHUB_REPOSITORY}"
  GIT_TAG        ${GIT_REVISION}
)
FetchContent_MakeAvailable( macis )

add_executable( macis_link_tester macis_link_tester.cxx )
target_link_libraries( macis_link_tester PUBLIC macis::macis )
