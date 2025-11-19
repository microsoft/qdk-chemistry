# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

include( FetchContent )

FetchContent_Declare( linalg-cmake-modules
  GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_GetProperties( linalg-cmake-modules )
if( NOT linalg-cmake-modules_POPULATED )
  FetchContent_Populate( linalg-cmake-modules )
  list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
endif()
