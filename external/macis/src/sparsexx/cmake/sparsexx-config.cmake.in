# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

cmake_minimum_required( VERSION 3.14 FATAL_ERROR )

get_filename_component( sparsexx_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH )

#list( APPEND CMAKE_MODULE_PATH ${sparsexx_CMAKE_DIR} )
include( CMakeFindDependencyMacro )
find_dependency( MPI )
find_dependency( OpenMP )


#list(REMOVE_AT CMAKE_MODULE_PATH -1)

if(NOT TARGET sparsexx::sparsexx)
    include("${sparsexx_CMAKE_DIR}/sparsexx-targets.cmake")
endif()

set(sparsexx_LIBRARIES sparsexx::sparsexx)
