# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

# Packaged Dependencies
set( LOBPCGXX_ENABLE_TESTS OFF CACHE BOOL "" FORCE )
set( SPARSEXX_ENABLE_TESTS OFF CACHE BOOL "" FORCE )
add_subdirectory(lobpcgxx)
add_subdirectory(sparsexx)

add_subdirectory(macis)
